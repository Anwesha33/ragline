// Package config loads ragline's configuration from the environment. Defaults
// target the docker-compose stack so the service runs with an empty
// environment apart from the API key.
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	HTTPAddr    string
	PostgresDSN string

	RedisAddr     string
	RedisPassword string
	RedisDB       int

	KafkaBrokers []string
	IngestTopic  string
	IngestDLQ    string

	GeminiAPIKey    string
	GenerationModel string
	EmbeddingModel  string
	RerankModel     string

	// EmbeddingDim is the Matryoshka truncation length. gemini-embedding-001
	// natively emits 3072 dimensions, but pgvector's HNSW index tops out at
	// 2000, so the pipeline asks for a shorter vector and renormalises it.
	EmbeddingDim int

	// Retrieval shape.
	VectorTopK  int // candidates from the ANN search
	KeywordTopK int // candidates from full-text search
	RRFK        int // rank-fusion smoothing constant
	RerankTopN  int // candidates handed to the reranker
	ContextTopN int // chunks that actually reach the generator

	RerankerMode string // "llm" | "off"

	// Rate limiting: a token bucket per client.
	RateLimitPerMinute int
	RateLimitBurst     int

	AnswerCacheTTL time.Duration
	HistoryTurns   int

	LLMTimeout time.Duration

	// Pricing, USD per million tokens, used for the cost column in query_logs.
	PriceInPerM    float64
	PriceOutPerM   float64
	PriceEmbedPerM float64
}

func Load() (*Config, error) {
	c := &Config{
		HTTPAddr:           env("HTTP_ADDR", ":8081"),
		PostgresDSN:        env("POSTGRES_DSN", "postgres://ragline:ragline@localhost:5433/ragline?sslmode=disable"),
		RedisAddr:          env("REDIS_ADDR", "localhost:6380"),
		RedisPassword:      env("REDIS_PASSWORD", ""),
		RedisDB:            envInt("REDIS_DB", 0),
		KafkaBrokers:       strings.Split(env("KAFKA_BROKERS", "localhost:9094"), ","),
		IngestTopic:        env("KAFKA_INGEST_TOPIC", "rag.documents.ingest"),
		IngestDLQ:          env("KAFKA_INGEST_DLQ", "rag.documents.dlq"),
		GeminiAPIKey:       env("GEMINI_API_KEY", ""),
		GenerationModel:    env("GENERATION_MODEL", "gemini-flash-latest"),
		EmbeddingModel:     env("EMBEDDING_MODEL", "gemini-embedding-001"),
		RerankModel:        env("RERANK_MODEL", "gemini-flash-lite-latest"),
		EmbeddingDim:       envInt("EMBEDDING_DIM", 1536),
		VectorTopK:         envInt("VECTOR_TOP_K", 30),
		KeywordTopK:        envInt("KEYWORD_TOP_K", 30),
		RRFK:               envInt("RRF_K", 60),
		RerankTopN:         envInt("RERANK_TOP_N", 12),
		ContextTopN:        envInt("CONTEXT_TOP_N", 5),
		RerankerMode:       env("RERANKER_MODE", "llm"),
		RateLimitPerMinute: envInt("RATE_LIMIT_PER_MINUTE", 30),
		RateLimitBurst:     envInt("RATE_LIMIT_BURST", 10),
		AnswerCacheTTL:     envDur("ANSWER_CACHE_TTL", 15*time.Minute),
		HistoryTurns:       envInt("HISTORY_TURNS", 6),
		LLMTimeout:         envDur("LLM_TIMEOUT", 120*time.Second),
		PriceInPerM:        envFloat("PRICE_IN_PER_M", 0.10),
		PriceOutPerM:       envFloat("PRICE_OUT_PER_M", 0.40),
		PriceEmbedPerM:     envFloat("PRICE_EMBED_PER_M", 0.15),
	}
	if c.EmbeddingDim > 2000 {
		return nil, fmt.Errorf("EMBEDDING_DIM %d exceeds the 2000-dimension limit of pgvector's HNSW index", c.EmbeddingDim)
	}
	if c.ContextTopN > c.RerankTopN {
		return nil, fmt.Errorf("CONTEXT_TOP_N (%d) cannot exceed RERANK_TOP_N (%d)", c.ContextTopN, c.RerankTopN)
	}
	return c, nil
}

func env(k, def string) string {
	if v := strings.TrimSpace(os.Getenv(k)); v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
			return n
		}
	}
	return def
}

func envFloat(k string, def float64) float64 {
	if v := os.Getenv(k); v != "" {
		if f, err := strconv.ParseFloat(strings.TrimSpace(v), 64); err == nil {
			return f
		}
	}
	return def
}

func envDur(k string, def time.Duration) time.Duration {
	if v := os.Getenv(k); v != "" {
		if d, err := time.ParseDuration(strings.TrimSpace(v)); err == nil {
			return d
		}
	}
	return def
}
