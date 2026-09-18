// Package chat is the query path: embed the question, retrieve with hybrid
// search, rerank, generate a grounded answer, and validate its citations.
//
// The ordering of concerns matters as much as the steps:
//
//	rate limit ─▶ cache ─▶ embed ─▶ hybrid search ─▶ rerank ─▶ generate ─▶ verify citations
//	                                      │
//	                    no chunks retrieved ──▶ refuse, do not call the model
//
// Refusing before generation is the cheapest hallucination control there is. A
// model handed an empty context will still produce a confident paragraph; a
// model that is never called cannot.
package chat

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/Anwesha33/ragline/internal/cache"
	"github.com/Anwesha33/ragline/internal/config"
	"github.com/Anwesha33/ragline/internal/embed"
	"github.com/Anwesha33/ragline/internal/llm"
	"github.com/Anwesha33/ragline/internal/rerank"
	"github.com/Anwesha33/ragline/internal/store"
)

const systemPrompt = `You answer questions using only the numbered sources provided in the message. You are a careful technical assistant.

Rules:
1. Every factual claim must be supported by a source, and must carry a citation like [2] naming the source it came from. Cite the specific source, not all of them.
2. If the sources do not contain the answer, say exactly what is missing. Do not fill the gap from general knowledge. "The provided documentation does not cover X" is a correct and useful answer.
3. Do not contradict a source. If two sources disagree, say so and cite both.
4. Answer in prose, not in a list of quotes. Be direct: lead with the answer, then the detail.
5. Keep it proportionate — a one-line question gets a short answer.`

// Request is one question.
type Request struct {
	Question       string
	ConversationID uuid.UUID
	ClientID       string
	// NoCache forces a fresh retrieval and generation, used by the evaluation
	// harness so a measurement never reads a cached answer.
	NoCache bool
	// TopN overrides how many chunks reach the generator.
	TopN int
	// IncludeSourceText returns the full text of each context chunk instead of
	// a preview. Used by the evaluation harness; not something a UI wants.
	IncludeSourceText bool
}

// Citation is a source the answer actually used.
type Citation struct {
	Marker     int       `json:"marker"`
	ChunkID    int64     `json:"chunk_id"`
	DocumentID uuid.UUID `json:"document_id"`
	Title      string    `json:"title"`
	SourceURI  string    `json:"source_uri"`
	Heading    string    `json:"heading,omitempty"`
	// Snippet is a short preview for rendering a citation in a UI.
	Snippet string `json:"snippet"`
	// Content is the full passage, returned only when the caller sets
	// include_source_text. The groundedness judge needs the text the model
	// actually saw: judging an answer against a 240-character preview reports
	// "unsupported" for claims the real source supports perfectly well, which
	// measures the harness rather than the system.
	Content string `json:"content,omitempty"`
}

// Event is what the HTTP layer streams to the client.
type Event struct {
	Type      string     `json:"type"` // "sources" | "token" | "done" | "error"
	Text      string     `json:"text,omitempty"`
	Citations []Citation `json:"citations,omitempty"`
	Usage     *Usage     `json:"usage,omitempty"`
	Error     string     `json:"error,omitempty"`
}

// Usage is the per-answer accounting the client also gets to see.
type Usage struct {
	CacheHit     bool    `json:"cache_hit"`
	Model        string  `json:"model"`
	Reranker     string  `json:"reranker"`
	Retrieved    int     `json:"retrieved_chunks"`
	Used         int     `json:"context_chunks"`
	EmbedMS      int     `json:"embed_ms"`
	SearchMS     int     `json:"search_ms"`
	RerankMS     int     `json:"rerank_ms"`
	GenerateMS   int     `json:"generate_ms"`
	TTFTMS       int     `json:"ttft_ms"`
	TotalMS      int     `json:"total_ms"`
	PromptTokens int     `json:"prompt_tokens"`
	OutputTokens int     `json:"output_tokens"`
	CostUSD      float64 `json:"cost_usd"`
}

// Result is the assembled answer, returned to non-streaming callers and to the
// evaluation harness.
type Result struct {
	Answer string `json:"answer"`
	// Citations are the sources the answer actually cited.
	Citations []Citation `json:"citations"`
	// Sources is every chunk that reached the generator, cited or not.
	// Exposing both is what lets the evaluation harness separate a retrieval
	// failure (the right passage never arrived) from a generation failure (it
	// arrived and the model ignored it) — two problems with entirely different
	// fixes.
	Sources   []Citation    `json:"sources"`
	Retrieved []store.Chunk `json:"-"`
	Usage     Usage         `json:"usage"`
	Refused   bool          `json:"refused"`
	Reason    string        `json:"reason,omitempty"`
}

// ErrRateLimited is returned when the caller's bucket is empty.
type ErrRateLimited struct{ RetryAfter time.Duration }

func (e *ErrRateLimited) Error() string {
	return fmt.Sprintf("rate limited, retry in %s", e.RetryAfter.Round(time.Millisecond))
}

type Service struct {
	Cfg      *config.Config
	Store    *store.Store
	Cache    *cache.Cache
	Embedder *embed.Client
	LLM      *llm.Client
	Reranker rerank.Reranker
	Log      Logger
}

type Logger interface {
	Info(msg string, kv ...any)
	Error(msg string, kv ...any)
}

// Answer runs the full query path, emitting events as they happen. emit may be
// nil for callers that only want the final Result.
func (s *Service) Answer(ctx context.Context, req Request, emit func(Event) error) (*Result, error) {
	start := time.Now()
	if emit == nil {
		emit = func(Event) error { return nil }
	}

	question := strings.TrimSpace(req.Question)
	if question == "" {
		return nil, errors.New("question is empty")
	}

	if err := s.checkRateLimit(ctx, req.ClientID); err != nil {
		return nil, err
	}

	usage := Usage{Model: s.LLM.Model(), Reranker: s.Reranker.Name()}
	topN := req.TopN
	if topN <= 0 {
		topN = s.Cfg.ContextTopN
	}

	// History is loaded first because its presence decides whether this answer
	// is cacheable at all.
	var history []store.Message
	if req.ConversationID != uuid.Nil {
		h, err := s.Store.History(ctx, req.ConversationID, s.Cfg.HistoryTurns*2)
		if err != nil {
			s.Log.Error("load history", "err", err)
		} else {
			history = h
		}
	}

	corpusVersion := s.corpusVersion(ctx)
	cacheKey := cache.AnswerKey(question, corpusVersion)
	// An answer that depended on prior turns is not a function of the question
	// alone, so it must never be served to a different conversation.
	cacheable := len(history) == 0 && !req.NoCache && s.Cache != nil

	if cacheable {
		if cached, err := s.Cache.GetAnswer(ctx, cacheKey); err == nil {
			usage.CacheHit = true
			usage.TotalMS = int(time.Since(start).Milliseconds())
			cits := s.hydrateCitations(ctx, cached.Citations)
			_ = emit(Event{Type: "sources", Citations: cits})
			_ = emit(Event{Type: "token", Text: cached.Answer})
			_ = emit(Event{Type: "done", Usage: &usage})
			s.logQuery(ctx, req, question, cached.Answer, cached.ChunkIDs, cached.Citations, usage, true, "")
			return &Result{Answer: cached.Answer, Citations: cits, Usage: usage}, nil
		}
	}

	// ---- retrieve -------------------------------------------------------
	embedStart := time.Now()
	vector, embedErr := s.embedQuestion(ctx, question)
	usage.EmbedMS = int(time.Since(embedStart).Milliseconds())

	searchStart := time.Now()
	var candidates []store.Chunk
	var err error
	if embedErr != nil {
		// Degrade rather than fail: keyword search alone still answers many
		// questions, and a worse answer beats a 500.
		s.Log.Error("embedding failed; falling back to keyword-only retrieval", "err", embedErr)
		candidates, err = s.Store.KeywordSearch(ctx, question, s.Cfg.RerankTopN)
	} else {
		candidates, err = s.Store.HybridSearch(ctx, vector, question,
			s.Cfg.VectorTopK, s.Cfg.KeywordTopK, s.Cfg.RRFK, s.Cfg.RerankTopN)
	}
	usage.SearchMS = int(time.Since(searchStart).Milliseconds())
	if err != nil {
		return nil, fmt.Errorf("retrieval: %w", err)
	}
	usage.Retrieved = len(candidates)

	if len(candidates) == 0 {
		const reason = "no documents in the corpus matched this question"
		usage.TotalMS = int(time.Since(start).Milliseconds())
		answer := "I could not find anything in the indexed documents that addresses this question."
		_ = emit(Event{Type: "token", Text: answer})
		_ = emit(Event{Type: "done", Usage: &usage})
		s.logQuery(ctx, req, question, answer, nil, nil, usage, false, reason)
		return &Result{Answer: answer, Refused: true, Reason: reason, Usage: usage}, nil
	}

	// ---- rerank ---------------------------------------------------------
	ranked, rerankStats, rerankErr := s.Reranker.Rerank(ctx, question, candidates, topN)
	usage.RerankMS = int(rerankStats.Duration.Milliseconds())
	usage.PromptTokens += rerankStats.PromptTokens
	usage.OutputTokens += rerankStats.OutputTokens
	if rerankErr != nil {
		s.Log.Error("rerank degraded", "err", rerankErr)
	}
	usage.Used = len(ranked)

	// Tell the client what the answer will be grounded in before the first
	// token arrives; the sources render while the text streams.
	sources := make([]Citation, 0, len(ranked))
	for i, c := range ranked {
		cite := Citation{
			Marker: i + 1, ChunkID: c.ID, DocumentID: c.DocumentID,
			Title: c.DocTitle, SourceURI: c.SourceURI, Heading: c.Heading,
			Snippet: snippet(c.Content, 240),
		}
		if req.IncludeSourceText {
			cite.Content = c.Content
		}
		sources = append(sources, cite)
	}
	if err := emit(Event{Type: "sources", Citations: sources}); err != nil {
		return nil, err
	}

	// ---- generate -------------------------------------------------------
	genStart := time.Now()
	var ttft time.Duration
	var firstToken bool

	contents := buildContents(history, question, ranked)
	resp, genErr := s.LLM.Stream(ctx, llm.Request{
		System:   systemPrompt,
		Contents: contents,
		Config:   llm.GenerationConfig{Temperature: 0.2},
	}, func(ev llm.StreamEvent) error {
		if ev.Done {
			return nil
		}
		if !firstToken {
			firstToken = true
			ttft = time.Since(genStart)
		}
		return emit(Event{Type: "token", Text: ev.Text})
	})
	usage.GenerateMS = int(time.Since(genStart).Milliseconds())
	usage.TTFTMS = int(ttft.Milliseconds())
	if genErr != nil {
		return nil, fmt.Errorf("generation: %w", genErr)
	}
	usage.PromptTokens += resp.Usage.PromptTokens
	usage.OutputTokens += resp.Usage.OutputTokens

	// ---- verify citations ----------------------------------------------
	answer := resp.Text
	used, unknown := parseCitations(answer, len(ranked))
	if len(unknown) > 0 {
		// A marker pointing at a source that was never supplied is the clearest
		// possible hallucination signal. Strip it rather than rendering a
		// citation the reader cannot follow.
		answer = stripUnknownMarkers(answer, unknown)
		s.Log.Error("answer cited sources that were not provided", "markers", unknown)
	}
	citations := make([]Citation, 0, len(used))
	for _, m := range used {
		citations = append(citations, sources[m-1])
	}

	usage.TotalMS = int(time.Since(start).Milliseconds())
	usage.CostUSD = s.cost(usage)

	retrievedIDs := make([]int64, 0, len(ranked))
	for _, c := range ranked {
		retrievedIDs = append(retrievedIDs, c.ID)
	}
	citedIDs := make([]int64, 0, len(citations))
	for _, c := range citations {
		citedIDs = append(citedIDs, c.ChunkID)
	}

	if cacheable {
		if err := s.Cache.PutAnswer(ctx, cacheKey, cache.CachedAnswer{
			Answer: answer, ChunkIDs: retrievedIDs, Citations: citedIDs,
			Model: s.LLM.Model(), CachedAt: time.Now(),
		}, s.Cfg.AnswerCacheTTL); err != nil {
			s.Log.Error("cache answer", "err", err)
		}
	}

	if req.ConversationID != uuid.Nil {
		_ = s.Store.AppendMessage(ctx, req.ConversationID, store.Message{Role: "user", Content: question})
		_ = s.Store.AppendMessage(ctx, req.ConversationID, store.Message{
			Role: "assistant", Content: answer, Citations: citedIDs,
		})
	}

	s.logQuery(ctx, req, question, answer, retrievedIDs, citedIDs, usage, true, "")
	_ = emit(Event{Type: "done", Usage: &usage})

	return &Result{Answer: answer, Citations: citations, Sources: sources, Retrieved: ranked, Usage: usage}, nil
}

func (s *Service) checkRateLimit(ctx context.Context, clientID string) error {
	if s.Cache == nil || s.Cfg.RateLimitPerMinute <= 0 {
		return nil
	}
	if clientID == "" {
		clientID = "anonymous"
	}
	ok, retryAfter, err := s.Cache.Allow(ctx, clientID, s.Cfg.RateLimitPerMinute, s.Cfg.RateLimitBurst)
	if err != nil {
		s.Log.Error("rate limiter unavailable; allowing request", "err", err)
		return nil
	}
	if !ok {
		return &ErrRateLimited{RetryAfter: retryAfter}
	}
	return nil
}

func (s *Service) embedQuestion(ctx context.Context, question string) ([]float32, error) {
	key := cache.EmbeddingKey(question, s.Embedder.Model(), s.Embedder.Dim())
	if s.Cache != nil {
		if v, err := s.Cache.GetEmbedding(ctx, key); err == nil && len(v) == s.Embedder.Dim() {
			return v, nil
		}
	}
	// TaskQuery, not TaskDocument: the query and the passages are embedded with
	// different task types on purpose.
	v, err := s.Embedder.Embed(ctx, question, embed.TaskQuery)
	if err != nil {
		return nil, err
	}
	if s.Cache != nil {
		_ = s.Cache.PutEmbedding(ctx, key, v, time.Hour)
	}
	return v, nil
}

// corpusVersion is a cheap fingerprint of corpus state used in the cache key,
// so ingesting a document invalidates every cached answer without an explicit
// purge.
func (s *Service) corpusVersion(ctx context.Context) string {
	st, err := s.Store.CorpusStats(ctx)
	if err != nil {
		// Unknown corpus state means "do not reuse an answer": returning a
		// unique-ish value makes the lookup miss rather than risk a stale hit.
		return "unknown-" + strconv.FormatInt(time.Now().Unix(), 36)
	}
	return fmt.Sprintf("d%d-c%d-%d", st.ReadyDocs, st.Chunks, st.LastIngest.Unix())
}

func (s *Service) hydrateCitations(ctx context.Context, chunkIDs []int64) []Citation {
	out := make([]Citation, 0, len(chunkIDs))
	for i, id := range chunkIDs {
		out = append(out, Citation{Marker: i + 1, ChunkID: id})
	}
	return out
}

func (s *Service) cost(u Usage) float64 {
	return float64(u.PromptTokens)/1e6*s.Cfg.PriceInPerM +
		float64(u.OutputTokens)/1e6*s.Cfg.PriceOutPerM
}

func (s *Service) logQuery(ctx context.Context, req Request, question, _ string,
	retrieved, cited []int64, u Usage, answered bool, refusal string) {

	var convID *uuid.UUID
	if req.ConversationID != uuid.Nil {
		id := req.ConversationID
		convID = &id
	}
	// Telemetry must not ride on the request's context: a client that
	// disconnects mid-answer is exactly the request worth recording.
	logCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 3*time.Second)
	defer cancel()

	if err := s.Store.LogQuery(logCtx, store.QueryLog{
		ConversationID: convID, Question: question, Answered: answered,
		RefusalReason: refusal, CacheHit: u.CacheHit,
		RetrievedChunkIDs: retrieved, CitedChunkIDs: cited,
		EmbedMS: u.EmbedMS, SearchMS: u.SearchMS, RerankMS: u.RerankMS,
		GenerateMS: u.GenerateMS, TTFTMS: u.TTFTMS, TotalMS: u.TotalMS,
		PromptTokens: u.PromptTokens, OutputTokens: u.OutputTokens,
		CostUSD: u.CostUSD, Model: u.Model,
	}); err != nil {
		s.Log.Error("write query log", "err", err)
	}
}

// buildContents assembles the prompt: prior turns, then the sources, then the
// question. The question goes last because instructions closest to the end of
// the prompt are followed most reliably.
func buildContents(history []store.Message, question string, chunks []store.Chunk) []llm.Content {
	contents := make([]llm.Content, 0, len(history)+1)
	for _, m := range history {
		role := llm.RoleUser
		if m.Role == "assistant" {
			role = llm.RoleModel
		}
		contents = append(contents, llm.Content{Role: role, Parts: []llm.Part{{Text: m.Content}}})
	}

	var b strings.Builder
	b.WriteString("Sources:\n")
	for i, c := range chunks {
		title := c.DocTitle
		if c.Heading != "" {
			title += " › " + c.Heading
		}
		fmt.Fprintf(&b, "\n[%d] %s\n%s\n", i+1, title, c.Content)
	}
	fmt.Fprintf(&b, "\n---\nQuestion: %s\n\nAnswer using only the sources above, citing them as [n].", question)

	contents = append(contents, llm.Content{Role: llm.RoleUser, Parts: []llm.Part{{Text: b.String()}}})
	return contents
}

// citationRE matches a citation marker, including the grouped form.
//
// Models cite in at least three shapes — [1], [1][2] and [1, 2] — and the
// grouped form is common. An earlier version of this regex only matched a lone
// bracketed number, so a perfectly cited answer was recorded as having zero
// citations: the answer looked wrong in evaluation while actually being
// correct. The two-digit cap keeps "[1234]" (a reference number in prose) from
// being read as a citation.
var citationRE = regexp.MustCompile(`\[\s*\d{1,2}(?:\s*,\s*\d{1,2})*\s*\]`)

var citationNumberRE = regexp.MustCompile(`\d{1,2}`)

// parseCitations returns the markers the answer used that are valid, and the
// ones that refer to a source that was never supplied.
func parseCitations(answer string, sourceCount int) (used, unknown []int) {
	seenUsed := map[int]bool{}
	seenUnknown := map[int]bool{}
	for _, group := range citationRE.FindAllString(answer, -1) {
		for _, raw := range citationNumberRE.FindAllString(group, -1) {
			n, err := strconv.Atoi(raw)
			if err != nil {
				continue
			}
			switch {
			case n >= 1 && n <= sourceCount:
				if !seenUsed[n] {
					seenUsed[n] = true
					used = append(used, n)
				}
			default:
				if !seenUnknown[n] {
					seenUnknown[n] = true
					unknown = append(unknown, n)
				}
			}
		}
	}
	sort.Ints(used)
	sort.Ints(unknown)
	return used, unknown
}

// stripUnknownMarkers removes references to sources that were never supplied,
// rewriting grouped markers rather than deleting them wholesale so that the
// valid citations inside a group survive.
func stripUnknownMarkers(answer string, unknown []int) string {
	bad := make(map[int]bool, len(unknown))
	for _, n := range unknown {
		bad[n] = true
	}
	return citationRE.ReplaceAllStringFunc(answer, func(group string) string {
		var keep []string
		for _, raw := range citationNumberRE.FindAllString(group, -1) {
			n, err := strconv.Atoi(raw)
			if err != nil || bad[n] {
				continue
			}
			keep = append(keep, raw)
		}
		if len(keep) == 0 {
			return ""
		}
		return fmt.Sprintf("[%s]", strings.Join(keep, ", "))
	})
}

func snippet(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// SearchTimings breaks down a retrieval-only call.
type SearchTimings struct {
	EmbedMS    int    `json:"embed_ms"`
	SearchMS   int    `json:"search_ms"`
	RerankMS   int    `json:"rerank_ms"`
	Reranker   string `json:"reranker"`
	Candidates int    `json:"candidates"`
	Reordered  int    `json:"reordered_by_rerank"`
}

// Search runs retrieval without generation.
//
// This exists because retrieval quality and answer quality fail for different
// reasons and must be debuggable separately: if the right passage is not in
// this response, no amount of prompt work will fix the answer.
func (s *Service) Search(ctx context.Context, query string, topN int, noRerank bool) ([]store.Chunk, SearchTimings, error) {
	t := SearchTimings{Reranker: "none"}

	embedStart := time.Now()
	vector, embedErr := s.embedQuestion(ctx, query)
	t.EmbedMS = int(time.Since(embedStart).Milliseconds())

	searchStart := time.Now()
	var candidates []store.Chunk
	var err error
	if embedErr != nil {
		s.Log.Error("embedding failed; keyword-only search", "err", embedErr)
		candidates, err = s.Store.KeywordSearch(ctx, query, s.Cfg.RerankTopN)
	} else {
		candidates, err = s.Store.HybridSearch(ctx, vector, query,
			s.Cfg.VectorTopK, s.Cfg.KeywordTopK, s.Cfg.RRFK, s.Cfg.RerankTopN)
	}
	t.SearchMS = int(time.Since(searchStart).Milliseconds())
	if err != nil {
		return nil, t, err
	}
	t.Candidates = len(candidates)

	if noRerank {
		out, _, _ := rerank.Noop{}.Rerank(ctx, query, candidates, topN)
		return out, t, nil
	}

	ranked, stats, rerankErr := s.Reranker.Rerank(ctx, query, candidates, topN)
	t.RerankMS = int(stats.Duration.Milliseconds())
	t.Reordered = stats.Reordered
	t.Reranker = s.Reranker.Name()
	if rerankErr != nil {
		s.Log.Error("rerank degraded", "err", rerankErr)
	}
	return ranked, t, nil
}
