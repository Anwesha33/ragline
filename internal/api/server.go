// Package api is ragline's HTTP surface: upload documents, ask questions over
// a server-sent-event stream, and read operational statistics.
package api

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/Anwesha33/ragline/internal/chat"
	"github.com/Anwesha33/ragline/internal/config"
	"github.com/Anwesha33/ragline/internal/queue"
	"github.com/Anwesha33/ragline/internal/store"
)

type Server struct {
	cfg      *config.Config
	store    *store.Store
	producer *queue.Producer
	chat     *chat.Service
	log      *slog.Logger
	started  time.Time
}

func NewServer(cfg *config.Config, st *store.Store, p *queue.Producer, c *chat.Service, log *slog.Logger) *Server {
	return &Server{cfg: cfg, store: st, producer: p, chat: c, log: log, started: time.Now()}
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/documents", s.uploadDocument)
	mux.HandleFunc("GET /v1/documents", s.listDocuments)
	mux.HandleFunc("GET /v1/documents/{id}", s.getDocument)
	mux.HandleFunc("POST /v1/chat", s.chatStream)
	mux.HandleFunc("POST /v1/search", s.searchOnly)
	mux.HandleFunc("GET /v1/conversations/{id}", s.getConversation)
	mux.HandleFunc("GET /v1/stats", s.stats)
	mux.HandleFunc("GET /healthz", s.healthz)
	mux.HandleFunc("GET /readyz", s.readyz)
	return s.withLogging(mux)
}

// ------------------------------------------------------------- documents

type uploadRequest struct {
	SourceURI string `json:"source_uri"`
	Title     string `json:"title"`
	Content   string `json:"content"`
}

type uploadResponse struct {
	DocumentID uuid.UUID `json:"document_id"`
	Status     string    `json:"status"`
	Duplicate  bool      `json:"duplicate"`
	StatusURL  string    `json:"status_url"`
}

// uploadDocument accepts either a JSON body or a raw text/markdown body, stores
// it, and queues it for ingestion.
func (s *Server) uploadDocument(w http.ResponseWriter, r *http.Request) {
	const maxUpload = 8 << 20
	body, err := io.ReadAll(io.LimitReader(r.Body, maxUpload))
	if err != nil {
		writeError(w, http.StatusBadRequest, "unreadable body")
		return
	}

	var req uploadRequest
	if strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
		if err := json.Unmarshal(body, &req); err != nil {
			writeError(w, http.StatusBadRequest, "malformed JSON: "+err.Error())
			return
		}
	} else {
		// Raw upload: `curl --data-binary @notes.md -H 'X-Title: Notes'`
		req.Content = string(body)
		req.Title = r.Header.Get("X-Title")
		req.SourceURI = r.Header.Get("X-Source-URI")
	}

	req.Content = strings.TrimSpace(req.Content)
	if req.Content == "" {
		writeError(w, http.StatusBadRequest, "content is empty")
		return
	}
	if req.Title == "" {
		if req.SourceURI != "" {
			req.Title = path.Base(req.SourceURI)
		} else {
			req.Title = firstHeading(req.Content)
		}
	}

	sum := sha256.Sum256([]byte(req.Content))
	doc, created, err := s.store.CreateDocument(r.Context(), &store.Document{
		SourceURI:   req.SourceURI,
		Title:       req.Title,
		ContentHash: hex.EncodeToString(sum[:]),
		ContentType: "text/markdown",
		ByteSize:    len(req.Content),
	}, req.Content)
	if err != nil {
		s.log.Error("create document", "err", err)
		writeError(w, http.StatusInternalServerError, "could not store document")
		return
	}

	if created {
		if err := s.producer.Publish(r.Context(), s.cfg.IngestTopic, queue.IngestJob{
			DocumentID: doc.ID, SourceURI: doc.SourceURI, Title: doc.Title,
			Attempt: 1, EnqueuedAt: time.Now(),
		}); err != nil {
			s.log.Error("publish ingest job", "err", err, "document_id", doc.ID)
			writeError(w, http.StatusServiceUnavailable, "document stored but could not be queued for ingestion")
			return
		}
	}

	status := http.StatusAccepted
	if !created {
		status = http.StatusOK
	}
	writeJSON(w, status, uploadResponse{
		DocumentID: doc.ID, Status: doc.Status, Duplicate: !created,
		StatusURL: "/v1/documents/" + doc.ID.String(),
	})
}

func (s *Server) getDocument(w http.ResponseWriter, r *http.Request) {
	id, err := uuid.Parse(r.PathValue("id"))
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid document id")
		return
	}
	doc, err := s.store.Document(r.Context(), id)
	if errors.Is(err, store.ErrNotFound) {
		writeError(w, http.StatusNotFound, "no such document")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, doc)
}

func (s *Server) listDocuments(w http.ResponseWriter, r *http.Request) {
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	docs, err := s.store.ListDocuments(r.Context(), limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"documents": docs, "count": len(docs)})
}

// ------------------------------------------------------------------ chat

type chatRequest struct {
	Question       string `json:"question"`
	ConversationID string `json:"conversation_id,omitempty"`
	NoCache        bool   `json:"no_cache,omitempty"`
	TopN           int    `json:"top_n,omitempty"`
	// IncludeSourceText returns full passage text on each source rather than a
	// preview. The evaluation harness needs it to judge groundedness against
	// what the model actually read.
	IncludeSourceText bool `json:"include_source_text,omitempty"`
	// Stream defaults to true. A client that wants one JSON object — the
	// evaluation harness, or any non-browser caller — sets it false.
	Stream *bool `json:"stream,omitempty"`
}

func (s *Server) chatStream(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "malformed JSON: "+err.Error())
		return
	}
	if strings.TrimSpace(req.Question) == "" {
		writeError(w, http.StatusBadRequest, "question is required")
		return
	}

	convID := uuid.Nil
	if req.ConversationID != "" {
		parsed, err := uuid.Parse(req.ConversationID)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid conversation_id")
			return
		}
		convID = parsed
	}
	clientID := s.clientID(r)
	if convID != uuid.Nil {
		if _, err := s.store.EnsureConversation(r.Context(), convID, clientID); err != nil {
			s.log.Error("ensure conversation", "err", err)
		}
	}

	chatReq := chat.Request{
		Question: req.Question, ConversationID: convID,
		ClientID: clientID, NoCache: req.NoCache, TopN: req.TopN,
		IncludeSourceText: req.IncludeSourceText,
	}

	streaming := req.Stream == nil || *req.Stream
	if !streaming {
		res, err := s.chat.Answer(r.Context(), chatReq, nil)
		if err != nil {
			s.writeChatError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, res)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported by this server")
		return
	}

	// Headers must be written before the first event. Once they are out, an
	// error can no longer be reported as an HTTP status, which is why the
	// stream carries an explicit "error" event type.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	// Without this, nginx and most CDNs buffer the whole response and the
	// stream arrives as one lump at the end.
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	enc := json.NewEncoder(w)
	emit := func(ev chat.Event) error {
		// A client that has gone away should stop generation rather than have
		// tokens written into a dead socket.
		if err := r.Context().Err(); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "event: %s\ndata: ", ev.Type); err != nil {
			return err
		}
		if err := enc.Encode(ev); err != nil {
			return err
		}
		if _, err := io.WriteString(w, "\n"); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}

	if _, err := s.chat.Answer(r.Context(), chatReq, emit); err != nil {
		if errors.Is(err, context.Canceled) {
			s.log.Info("client disconnected mid-answer")
			return
		}
		s.log.Error("answer failed", "err", err)
		_ = emit(chat.Event{Type: "error", Error: err.Error()})
	}
}

// searchOnly exposes retrieval without generation. It is what makes retrieval
// quality debuggable on its own, and what the evaluation harness measures
// recall with.
func (s *Server) searchOnly(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Query    string `json:"query"`
		TopN     int    `json:"top_n"`
		NoRerank bool   `json:"no_rerank"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "malformed JSON")
		return
	}
	if strings.TrimSpace(req.Query) == "" {
		writeError(w, http.StatusBadRequest, "query is required")
		return
	}
	topN := req.TopN
	if topN <= 0 {
		topN = s.cfg.ContextTopN
	}
	chunks, stats, err := s.chat.Search(r.Context(), req.Query, topN, req.NoRerank)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"chunks": chunks, "timings": stats})
}

func (s *Server) getConversation(w http.ResponseWriter, r *http.Request) {
	id, err := uuid.Parse(r.PathValue("id"))
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid conversation id")
		return
	}
	msgs, err := s.store.History(r.Context(), id, 200)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"conversation_id": id, "messages": msgs})
}

func (s *Server) stats(w http.ResponseWriter, r *http.Request) {
	window := 24 * time.Hour
	if v := r.URL.Query().Get("window"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			window = d
		}
	}
	st, err := s.store.Stats(r.Context(), window)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	corpus, _ := s.store.CorpusStats(r.Context())
	writeJSON(w, http.StatusOK, map[string]any{"window": window.String(), "queries": st, "corpus": corpus})
}

func (s *Server) healthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "uptime_seconds": int(time.Since(s.started).Seconds())})
}

func (s *Server) readyz(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	if err := s.store.Ping(ctx); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"status": "degraded", "postgres": err.Error()})
		return
	}
	corpus, err := s.store.CorpusStats(ctx)
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"status": "degraded", "corpus": err.Error()})
		return
	}
	// An empty corpus is not an error, but it is the first thing anyone should
	// check when every answer is a refusal.
	writeJSON(w, http.StatusOK, map[string]any{"status": "ready", "corpus": corpus})
}

func (s *Server) writeChatError(w http.ResponseWriter, err error) {
	var limited *chat.ErrRateLimited
	if errors.As(err, &limited) {
		w.Header().Set("Retry-After", strconv.Itoa(int(limited.RetryAfter.Seconds())+1))
		writeError(w, http.StatusTooManyRequests, err.Error())
		return
	}
	s.log.Error("answer failed", "err", err)
	writeError(w, http.StatusInternalServerError, err.Error())
}

// clientID is the rate-limit bucket key. A real deployment would use an
// authenticated principal; behind a proxy the first X-Forwarded-For hop is the
// closest available stand-in.
func (s *Server) clientID(r *http.Request) string {
	if k := r.Header.Get("X-Client-ID"); k != "" {
		return k
	}
	if ff := r.Header.Get("X-Forwarded-For"); ff != "" {
		return strings.TrimSpace(strings.Split(ff, ",")[0])
	}
	host, _, err := splitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

func firstHeading(content string) string {
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "#") {
			return strings.TrimSpace(strings.TrimLeft(line, "# "))
		}
		if line != "" {
			if len(line) > 80 {
				return line[:80]
			}
			return line
		}
	}
	return "untitled"
}

func (s *Server) withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)
		s.log.Info("http", "method", r.Method, "path", r.URL.Path,
			"status", rec.status, "duration_ms", time.Since(start).Milliseconds())
	})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

// Flush forwards to the wrapped writer; without it the logging middleware would
// hide the Flusher interface and break streaming entirely.
func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
