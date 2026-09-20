// Command gateway serves ragline's HTTP API: document upload, streaming
// question answering, retrieval-only search, and statistics.
//
// Ingestion itself runs in the Python workers; this process only publishes the
// job. That split is deliberate — the serving path is latency-sensitive and
// concurrency-heavy, which is Go's strength, while chunking and embedding are
// batch work that belongs where the text-processing ecosystem lives.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/Anwesha33/ragline/internal/api"
	"github.com/Anwesha33/ragline/internal/cache"
	"github.com/Anwesha33/ragline/internal/chat"
	"github.com/Anwesha33/ragline/internal/config"
	"github.com/Anwesha33/ragline/internal/embed"
	"github.com/Anwesha33/ragline/internal/llm"
	"github.com/Anwesha33/ragline/internal/queue"
	"github.com/Anwesha33/ragline/internal/rerank"
	"github.com/Anwesha33/ragline/internal/store"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(log)

	cfg, err := config.Load()
	if err != nil {
		log.Error("configuration", "err", err)
		os.Exit(1)
	}
	if cfg.GeminiAPIKey == "" {
		log.Error("GEMINI_API_KEY is required")
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	st, err := openStoreWithRetry(ctx, cfg.PostgresDSN, log)
	if err != nil {
		log.Error("postgres", "err", err)
		os.Exit(1)
	}
	defer st.Close()

	// Set once at startup. BM25 is the default; "ts_rank" selects the old
	// coverage-density scorer, which the evaluation uses as a control arm.
	st.KeywordRanking = store.KeywordRanking(cfg.KeywordRanking)
	log.Info("keyword ranking", "mode", st.KeywordRanking)

	// The gateway owns the schema so the corpus tables exist before the first
	// upload, whether or not an ingest worker has ever started.
	if err := st.Migrate(ctx); err != nil {
		log.Error("migrate", "err", err)
		os.Exit(1)
	}

	rdb := cache.New(cfg.RedisAddr, cfg.RedisPassword, cfg.RedisDB)
	if err := rdb.Ping(ctx); err != nil {
		// Redis carries caching and rate limiting, neither of which is
		// correctness. Degrade rather than refuse to serve.
		log.Warn("redis unavailable: running without cache or rate limiting", "err", err)
		rdb = nil
	} else {
		defer rdb.Close()
	}

	if err := queue.EnsureTopics(ctx, cfg.KafkaBrokers,
		[]string{cfg.IngestTopic, cfg.IngestDLQ}, 3); err != nil {
		log.Warn("could not pre-create topics; relying on auto-creation", "err", err)
	}
	producer := queue.NewProducer(cfg.KafkaBrokers)
	defer producer.Close()

	generator := llm.New(cfg.GeminiAPIKey, cfg.GenerationModel, cfg.LLMTimeout)
	// The reranker runs on a smaller, cheaper model than the generator: scoring
	// relevance is a much easier task than writing a grounded answer, and it
	// sits directly in the latency path.
	reranker := rerank.New(cfg.RerankerMode, llm.New(cfg.GeminiAPIKey, cfg.RerankModel, cfg.LLMTimeout))

	chatSvc := &chat.Service{
		Cfg:      cfg,
		Store:    st,
		Cache:    rdb,
		Embedder: embed.New(cfg.GeminiAPIKey, cfg.EmbeddingModel, cfg.EmbeddingDim, cfg.LLMTimeout),
		LLM:      generator,
		Reranker: reranker,
		Log:      slogAdapter{log},
	}

	srv := api.NewServer(cfg, st, producer, chatSvc, log)
	httpSrv := &http.Server{
		Addr:              cfg.HTTPAddr,
		Handler:           srv.Routes(),
		ReadHeaderTimeout: 10 * time.Second,
		// No WriteTimeout: it would cut off a streaming answer mid-sentence.
		// The per-request context and the LLM client's own timeout bound the
		// work instead.
		IdleTimeout: 120 * time.Second,
	}

	go func() {
		log.Info("gateway listening",
			"addr", cfg.HTTPAddr,
			"generation_model", cfg.GenerationModel,
			"embedding_model", cfg.EmbeddingModel,
			"embedding_dim", cfg.EmbeddingDim,
			"reranker", reranker.Name(),
			"rate_limit_per_minute", cfg.RateLimitPerMinute)
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("http server", "err", err)
			stop()
		}
	}()

	<-ctx.Done()
	log.Info("shutting down; draining in-flight answers")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		log.Error("graceful shutdown failed", "err", err)
	}
}

type slogAdapter struct{ l *slog.Logger }

func (s slogAdapter) Info(msg string, kv ...any)  { s.l.Info(msg, kv...) }
func (s slogAdapter) Error(msg string, kv ...any) { s.l.Error(msg, kv...) }

func openStoreWithRetry(ctx context.Context, dsn string, log *slog.Logger) (*store.Store, error) {
	var lastErr error
	for attempt := 0; attempt < 30; attempt++ {
		st, err := store.Open(ctx, dsn)
		if err == nil {
			return st, nil
		}
		lastErr = err
		log.Info("waiting for postgres", "attempt", attempt+1)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(2 * time.Second):
		}
	}
	return nil, lastErr
}
