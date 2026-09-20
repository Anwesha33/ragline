// Package store is ragline's Postgres layer: the document corpus, hybrid
// retrieval over pgvector plus full-text search, conversation history, and the
// per-query telemetry table.
package store

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

//go:embed schema.sql
var schemaSQL string

const (
	StatusPending  = "pending"
	StatusChunking = "chunking"
	StatusReady    = "ready"
	StatusFailed   = "failed"
)

var ErrNotFound = errors.New("not found")

type Store struct {
	pool *pgxpool.Pool
	// KeywordRanking selects the lexical scorer. Set once at startup from
	// configuration; empty means BM25.
	KeywordRanking KeywordRanking
}

func Open(ctx context.Context, dsn string) (*Store, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, err
	}
	cfg.MaxConns = 12
	cfg.MaxConnLifetime = time.Hour
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, err
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	return &Store{pool: pool, KeywordRanking: KeywordBM25}, nil
}

func (s *Store) Close()                         { s.pool.Close() }
func (s *Store) Ping(ctx context.Context) error { return s.pool.Ping(ctx) }
func (s *Store) Migrate(ctx context.Context) error {
	_, err := s.pool.Exec(ctx, schemaSQL)
	return err
}

// ---------------------------------------------------------------- documents

type Document struct {
	ID          uuid.UUID  `json:"id"`
	SourceURI   string     `json:"source_uri"`
	Title       string     `json:"title"`
	ContentHash string     `json:"content_hash"`
	ContentType string     `json:"content_type"`
	ByteSize    int        `json:"byte_size"`
	Status      string     `json:"status"`
	ChunkCount  int        `json:"chunk_count"`
	Attempts    int        `json:"attempts"`
	LastError   string     `json:"last_error,omitempty"`
	CreatedAt   time.Time  `json:"created_at"`
	IngestedAt  *time.Time `json:"ingested_at,omitempty"`
}

const docCols = `id, source_uri, title, content_hash, content_type, byte_size, status,
	chunk_count, attempts, last_error, created_at, ingested_at`

func scanDoc(row pgx.Row) (*Document, error) {
	var d Document
	err := row.Scan(&d.ID, &d.SourceURI, &d.Title, &d.ContentHash, &d.ContentType,
		&d.ByteSize, &d.Status, &d.ChunkCount, &d.Attempts, &d.LastError, &d.CreatedAt, &d.IngestedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNotFound
	}
	return &d, err
}

// CreateDocument stores the raw text and returns created=false when the exact
// bytes are already in the corpus. Content-addressing the corpus is what makes
// a re-upload free instead of doubling every retrieval result.
func (s *Store) CreateDocument(ctx context.Context, d *Document, raw string) (*Document, bool, error) {
	if d.ID == uuid.Nil {
		d.ID = uuid.New()
	}
	const q = `INSERT INTO documents (id, source_uri, title, content_hash, content_type, byte_size, status, raw_content)
	           VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
	           ON CONFLICT (content_hash) DO NOTHING
	           RETURNING ` + docCols
	created, err := scanDoc(s.pool.QueryRow(ctx, q, d.ID, d.SourceURI, d.Title, d.ContentHash,
		d.ContentType, d.ByteSize, StatusPending, raw))
	if err == nil {
		return created, true, nil
	}
	if !errors.Is(err, ErrNotFound) {
		return nil, false, err
	}
	existing, err := scanDoc(s.pool.QueryRow(ctx, `SELECT `+docCols+` FROM documents WHERE content_hash = $1`, d.ContentHash))
	return existing, false, err
}

func (s *Store) Document(ctx context.Context, id uuid.UUID) (*Document, error) {
	return scanDoc(s.pool.QueryRow(ctx, `SELECT `+docCols+` FROM documents WHERE id = $1`, id))
}

func (s *Store) ListDocuments(ctx context.Context, limit int) ([]*Document, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `SELECT `+docCols+` FROM documents ORDER BY created_at DESC LIMIT $1`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []*Document{}
	for rows.Next() {
		d, err := scanDoc(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

// CorpusStats powers /healthz and the answer-cache key: when the corpus
// changes, cached answers computed against the old corpus must stop being
// served.
type CorpusStats struct {
	Documents  int       `json:"documents"`
	ReadyDocs  int       `json:"ready_documents"`
	Chunks     int       `json:"chunks"`
	LastIngest time.Time `json:"last_ingest_at"`
}

func (s *Store) CorpusStats(ctx context.Context) (CorpusStats, error) {
	var st CorpusStats
	var last *time.Time
	err := s.pool.QueryRow(ctx, `
		SELECT (SELECT count(*) FROM documents),
		       (SELECT count(*) FROM documents WHERE status = $1),
		       (SELECT count(*) FROM chunks),
		       (SELECT max(ingested_at) FROM documents)`, StatusReady).
		Scan(&st.Documents, &st.ReadyDocs, &st.Chunks, &last)
	if last != nil {
		st.LastIngest = *last
	}
	return st, err
}

// ---------------------------------------------------------------- retrieval

// Chunk is one retrieved passage with its provenance and the ranks that earned
// it a place in the candidate set.
type Chunk struct {
	ID          int64     `json:"chunk_id"`
	DocumentID  uuid.UUID `json:"document_id"`
	DocTitle    string    `json:"document_title"`
	SourceURI   string    `json:"source_uri"`
	Ordinal     int       `json:"ordinal"`
	Heading     string    `json:"heading,omitempty"`
	Content     string    `json:"content"`
	RRFScore    float64   `json:"rrf_score"`
	VectorRank  int       `json:"vector_rank,omitempty"`  // 0 = not in the vector result set
	KeywordRank int       `json:"keyword_rank,omitempty"` // 0 = not in the keyword result set
	RerankScore float64   `json:"rerank_score,omitempty"`
}

// hybridSearchSQL fuses two independent rankings with Reciprocal Rank Fusion.
//
// RRF (score = sum over rankers of 1/(k + rank)) is used rather than a weighted
// sum of the raw scores because cosine distance and ts_rank_cd are not on
// comparable scales and their distributions shift with corpus size — any fixed
// weighting is tuned to one corpus and wrong on the next. RRF only reads the
// ordering, so it needs no normalisation and no per-corpus tuning. k=60 damps
// the top-rank advantage; smaller k makes the fusion behave more like "trust
// whichever ranker put it first".
const hybridSearchSQL = `
WITH vector_hits AS (
    SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.embedding <=> $1::vector) AS rank
    FROM chunks c
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> $1::vector
    LIMIT $2
),
keyword_hits AS (
    SELECT c.id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, q) DESC, c.id) AS rank
    FROM chunks c, websearch_to_tsquery('english', $3) q
    WHERE c.tsv @@ q
    ORDER BY ts_rank_cd(c.tsv, q) DESC, c.id
    LIMIT $4
),
fused AS (
    SELECT id, SUM(score) AS rrf FROM (
        SELECT id, 1.0 / ($5 + rank) AS score FROM vector_hits
        UNION ALL
        SELECT id, 1.0 / ($5 + rank) AS score FROM keyword_hits
    ) s GROUP BY id
)
SELECT ch.id, ch.document_id, d.title, d.source_uri, ch.ordinal, ch.heading, ch.content,
       f.rrf,
       COALESCE((SELECT v.rank FROM vector_hits v WHERE v.id = ch.id), 0),
       COALESCE((SELECT k.rank FROM keyword_hits k WHERE k.id = ch.id), 0)
FROM fused f
JOIN chunks ch  ON ch.id = f.id
JOIN documents d ON d.id = ch.document_id
ORDER BY f.rrf DESC, ch.id
LIMIT $6`

// KeywordRanking selects the lexical scorer for the keyword arm.
//
// BM25 is the default because it is the standard lexical baseline and what
// "hybrid search" normally means. ts_rank_cd is kept as a control arm: the
// evaluation harness can run the golden set under each and report the
// difference, which is the only way to claim BM25 helped rather than assume it.
type KeywordRanking string

const (
	KeywordBM25   KeywordRanking = "bm25"
	KeywordTsRank KeywordRanking = "ts_rank"
)

// BM25 constants. k1 controls how quickly term frequency saturates — a term
// appearing ten times is not ten times as relevant as appearing once — and b
// controls how much a long passage is penalised for its length. 1.2 and 0.75
// are the values nearly every implementation uses, and tuning them on 13
// golden questions would be fitting noise rather than tuning.
const (
	bm25K1 = 1.2
	bm25B  = 0.75
)

// hybridSearchBM25SQL is hybridSearchSQL with the keyword arm scored by BM25.
//
// Postgres cannot do this natively: ts_rank_cd is coverage density, which
// rewards query terms appearing close together and knows nothing about how rare
// a term is across the corpus or how long the passage is. BM25 is
//
//	score(D,Q) = Σ IDF(t) · tf(t,D)·(k1+1) / ( tf(t,D) + k1·(1 - b + b·|D|/avgdl) )
//
// with IDF(t) = ln(1 + (N - df(t) + 0.5)/(df(t) + 0.5)) — the Lucene form,
// which stays positive even for a term present in almost every chunk.
//
// The statistics it needs (tf, df, |D|, N, avgdl) are maintained by trigger in
// schema.sql, so they are written in the same transaction as the chunk.
const hybridSearchBM25SQL = `
WITH vector_hits AS (
    SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.embedding <=> $1::vector) AS rank
    FROM chunks c
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> $1::vector
    LIMIT $2
),
query_terms AS (
    SELECT DISTINCT u.lexeme AS term FROM unnest(to_tsvector('english', $3)) u
),
bm25_corpus AS (
    SELECT GREATEST(chunk_count, 1)::float8 AS n,
           GREATEST(total_len, 1)::float8 / GREATEST(chunk_count, 1)::float8 AS avgdl
    FROM corpus_stats WHERE only_row
),
keyword_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY score DESC, id) AS rank
    FROM (
        SELECT ct.chunk_id AS id,
               SUM(
                   ln(1 + (bc.n - cterm.df + 0.5) / (cterm.df + 0.5))
                   * (ct.tf * ($7::float8 + 1))
                   / (ct.tf + $7::float8 * (1 - $8::float8 + $8::float8 * cs.len / bc.avgdl))
               ) AS score
        FROM query_terms qt
        JOIN corpus_terms cterm ON cterm.term = qt.term
        JOIN chunk_terms  ct    ON ct.term = qt.term
        JOIN chunk_stats  cs    ON cs.chunk_id = ct.chunk_id
        CROSS JOIN bm25_corpus bc
        GROUP BY ct.chunk_id
    ) scored
    ORDER BY score DESC, id
    LIMIT $4
),
fused AS (
    SELECT id, SUM(score) AS rrf FROM (
        SELECT id, 1.0 / ($5 + rank) AS score FROM vector_hits
        UNION ALL
        SELECT id, 1.0 / ($5 + rank) AS score FROM keyword_hits
    ) s GROUP BY id
)
SELECT ch.id, ch.document_id, d.title, d.source_uri, ch.ordinal, ch.heading, ch.content,
       f.rrf,
       COALESCE((SELECT v.rank FROM vector_hits v WHERE v.id = ch.id), 0),
       COALESCE((SELECT k.rank FROM keyword_hits k WHERE k.id = ch.id), 0)
FROM fused f
JOIN chunks ch  ON ch.id = f.id
JOIN documents d ON d.id = ch.document_id
ORDER BY f.rrf DESC, ch.id
LIMIT $6`

// HybridSearch runs both retrievers and fuses them.
//
// The embedding is required: pgvector rejects a zero-dimension vector, so an
// empty one is an error rather than a graceful degradation. The keyword-only
// fallback taken when the embedding API is down is KeywordSearch, which the
// chat path calls explicitly — a worse answer beats a 500.
func (s *Store) HybridSearch(ctx context.Context, embedding []float32, query string,
	vectorK, keywordK, rrfK, limit int) ([]Chunk, error) {

	var rows pgx.Rows
	var err error
	if s.KeywordRanking == KeywordTsRank {
		rows, err = s.pool.Query(ctx, hybridSearchSQL,
			vectorLiteral(embedding), vectorK, query, keywordK, rrfK, limit)
	} else {
		rows, err = s.pool.Query(ctx, hybridSearchBM25SQL,
			vectorLiteral(embedding), vectorK, query, keywordK, rrfK, limit, bm25K1, bm25B)
	}
	if err != nil {
		return nil, fmt.Errorf("hybrid search: %w", err)
	}
	defer rows.Close()

	out := []Chunk{}
	for rows.Next() {
		var c Chunk
		if err := rows.Scan(&c.ID, &c.DocumentID, &c.DocTitle, &c.SourceURI, &c.Ordinal,
			&c.Heading, &c.Content, &c.RRFScore, &c.VectorRank, &c.KeywordRank); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// KeywordSearch is the fallback path used when no embedding is available.
// keywordSearchTsRankSQL is the original coverage-density scorer, kept as the
// control arm.
const keywordSearchTsRankSQL = `
	SELECT ch.id, ch.document_id, d.title, d.source_uri, ch.ordinal, ch.heading, ch.content,
	       ts_rank_cd(ch.tsv, qy) AS score,
	       0, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(ch.tsv, qy) DESC, ch.id)
	FROM chunks ch
	JOIN documents d ON d.id = ch.document_id, websearch_to_tsquery('english', $1) qy
	WHERE ch.tsv @@ qy
	ORDER BY score DESC, ch.id
	LIMIT $2`

// keywordSearchBM25SQL scores the same candidates with BM25.
//
// This path matters more than it looks: it is the fallback taken when the
// embedding API is down. If it ranked differently from the keyword arm of
// hybrid search, the degraded mode would be degraded in a second, undocumented
// way — so both use the same scorer.
const keywordSearchBM25SQL = `
	WITH query_terms AS (
	    SELECT DISTINCT u.lexeme AS term FROM unnest(to_tsvector('english', $1)) u
	),
	bm25_corpus AS (
	    SELECT GREATEST(chunk_count, 1)::float8 AS n,
	           GREATEST(total_len, 1)::float8 / GREATEST(chunk_count, 1)::float8 AS avgdl
	    FROM corpus_stats WHERE only_row
	),
	scored AS (
	    SELECT ct.chunk_id AS id,
	           SUM(
	               ln(1 + (bc.n - cterm.df + 0.5) / (cterm.df + 0.5))
	               * (ct.tf * ($3::float8 + 1))
	               / (ct.tf + $3::float8 * (1 - $4::float8 + $4::float8 * cs.len / bc.avgdl))
	           ) AS score
	    FROM query_terms qt
	    JOIN corpus_terms cterm ON cterm.term = qt.term
	    JOIN chunk_terms  ct    ON ct.term = qt.term
	    JOIN chunk_stats  cs    ON cs.chunk_id = ct.chunk_id
	    CROSS JOIN bm25_corpus bc
	    GROUP BY ct.chunk_id
	)
	SELECT ch.id, ch.document_id, d.title, d.source_uri, ch.ordinal, ch.heading, ch.content,
	       sc.score,
	       0, ROW_NUMBER() OVER (ORDER BY sc.score DESC, ch.id)
	FROM scored sc
	JOIN chunks ch   ON ch.id = sc.id
	JOIN documents d ON d.id = ch.document_id
	ORDER BY sc.score DESC, ch.id
	LIMIT $2`

func (s *Store) KeywordSearch(ctx context.Context, query string, limit int) ([]Chunk, error) {
	var rows pgx.Rows
	var err error
	if s.KeywordRanking == KeywordTsRank {
		rows, err = s.pool.Query(ctx, keywordSearchTsRankSQL, query, limit)
	} else {
		rows, err = s.pool.Query(ctx, keywordSearchBM25SQL, query, limit, bm25K1, bm25B)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Chunk{}
	for rows.Next() {
		var c Chunk
		if err := rows.Scan(&c.ID, &c.DocumentID, &c.DocTitle, &c.SourceURI, &c.Ordinal,
			&c.Heading, &c.Content, &c.RRFScore, &c.VectorRank, &c.KeywordRank); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// Neighbours fetches the chunks immediately around a hit. A chunk boundary
// often cuts an answer in half, and pulling the neighbours back in is far
// cheaper than enlarging every chunk.
func (s *Store) Neighbours(ctx context.Context, docID uuid.UUID, ordinal, radius int) ([]Chunk, error) {
	const q = `
	SELECT ch.id, ch.document_id, d.title, d.source_uri, ch.ordinal, ch.heading, ch.content
	FROM chunks ch JOIN documents d ON d.id = ch.document_id
	WHERE ch.document_id = $1 AND ch.ordinal BETWEEN $2 AND $3
	ORDER BY ch.ordinal`
	rows, err := s.pool.Query(ctx, q, docID, ordinal-radius, ordinal+radius)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Chunk{}
	for rows.Next() {
		var c Chunk
		if err := rows.Scan(&c.ID, &c.DocumentID, &c.DocTitle, &c.SourceURI, &c.Ordinal,
			&c.Heading, &c.Content); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// vectorLiteral renders a slice as pgvector's text input format. Passing the
// literal and casting with ::vector avoids a dependency on a pgvector driver
// package for what is ultimately "[1,2,3]".
func vectorLiteral(v []float32) string {
	if len(v) == 0 {
		return "[]"
	}
	var b strings.Builder
	b.Grow(len(v) * 12)
	b.WriteByte('[')
	for i, f := range v {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(float64(f), 'g', -1, 32))
	}
	b.WriteByte(']')
	return b.String()
}

// ------------------------------------------------------------ conversations

type Message struct {
	Role      string    `json:"role"` // "user" | "assistant"
	Content   string    `json:"content"`
	Citations []int64   `json:"citations,omitempty"`
	CreatedAt time.Time `json:"created_at"`
}

func (s *Store) EnsureConversation(ctx context.Context, id uuid.UUID, clientID string) (uuid.UUID, error) {
	if id == uuid.Nil {
		id = uuid.New()
	}
	_, err := s.pool.Exec(ctx,
		`INSERT INTO conversations (id, client_id) VALUES ($1,$2) ON CONFLICT (id) DO UPDATE SET updated_at = now()`,
		id, clientID)
	return id, err
}

func (s *Store) AppendMessage(ctx context.Context, convID uuid.UUID, m Message) error {
	citations := make([]int64, 0, len(m.Citations))
	citations = append(citations, m.Citations...)
	_, err := s.pool.Exec(ctx,
		`INSERT INTO messages (conversation_id, role, content, citations) VALUES ($1,$2,$3,$4)`,
		convID, m.Role, m.Content, citations)
	return err
}

// History returns the last n messages in chronological order. The inner query
// takes the most recent rows; the outer one restores reading order.
func (s *Store) History(ctx context.Context, convID uuid.UUID, n int) ([]Message, error) {
	const q = `
	SELECT role, content, created_at FROM (
		SELECT role, content, created_at, id FROM messages
		WHERE conversation_id = $1 ORDER BY id DESC LIMIT $2
	) t ORDER BY id ASC`
	rows, err := s.pool.Query(ctx, q, convID, n)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Message{}
	for rows.Next() {
		var m Message
		if err := rows.Scan(&m.Role, &m.Content, &m.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// ---------------------------------------------------------------- telemetry

// QueryLog is one answered question, with latency attributed per stage.
type QueryLog struct {
	ConversationID    *uuid.UUID
	Question          string
	Answered          bool
	RefusalReason     string
	CacheHit          bool
	RetrievedChunkIDs []int64
	CitedChunkIDs     []int64
	EmbedMS           int
	SearchMS          int
	RerankMS          int
	GenerateMS        int
	TTFTMS            int
	TotalMS           int
	PromptTokens      int
	OutputTokens      int
	EmbedTokens       int
	CostUSD           float64
	Model             string
}

func (s *Store) LogQuery(ctx context.Context, l QueryLog) error {
	const q = `INSERT INTO query_logs (conversation_id, question, answered, refusal_reason, cache_hit,
		retrieved_chunk_ids, cited_chunk_ids, embed_ms, search_ms, rerank_ms, generate_ms, ttft_ms,
		total_ms, prompt_tokens, output_tokens, embed_tokens, cost_usd, model)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)`
	_, err := s.pool.Exec(ctx, q, l.ConversationID, l.Question, l.Answered, l.RefusalReason, l.CacheHit,
		l.RetrievedChunkIDs, l.CitedChunkIDs, l.EmbedMS, l.SearchMS, l.RerankMS, l.GenerateMS,
		l.TTFTMS, l.TotalMS, l.PromptTokens, l.OutputTokens, l.EmbedTokens, l.CostUSD, l.Model)
	return err
}

// Stats is the aggregate served by /v1/stats: what the system costs and how
// fast it is, read straight from the log table rather than from a metrics
// backend that would have to be stood up separately.
type Stats struct {
	Queries       int     `json:"queries"`
	CacheHitRate  float64 `json:"cache_hit_rate"`
	P50TotalMS    int     `json:"p50_total_ms"`
	P95TotalMS    int     `json:"p95_total_ms"`
	P50TTFTMS     int     `json:"p50_ttft_ms"`
	MeanPromptTok float64 `json:"mean_prompt_tokens"`
	MeanOutputTok float64 `json:"mean_output_tokens"`
	TotalCostUSD  float64 `json:"total_cost_usd"`
	MeanCostUSD   float64 `json:"mean_cost_per_query_usd"`
	RefusalRate   float64 `json:"refusal_rate"`
}

func (s *Store) Stats(ctx context.Context, since time.Duration) (Stats, error) {
	var st Stats
	const q = `
	SELECT count(*),
	       COALESCE(avg(CASE WHEN cache_hit THEN 1.0 ELSE 0.0 END), 0),
	       COALESCE(percentile_disc(0.5) WITHIN GROUP (ORDER BY total_ms), 0),
	       COALESCE(percentile_disc(0.95) WITHIN GROUP (ORDER BY total_ms), 0),
	       COALESCE(percentile_disc(0.5) WITHIN GROUP (ORDER BY ttft_ms), 0),
	       COALESCE(avg(prompt_tokens), 0), COALESCE(avg(output_tokens), 0),
	       COALESCE(sum(cost_usd), 0), COALESCE(avg(cost_usd), 0),
	       COALESCE(avg(CASE WHEN answered THEN 0.0 ELSE 1.0 END), 0)
	FROM query_logs WHERE created_at > now() - $1::interval`
	err := s.pool.QueryRow(ctx, q, fmt.Sprintf("%d seconds", int(since.Seconds()))).
		Scan(&st.Queries, &st.CacheHitRate, &st.P50TotalMS, &st.P95TotalMS, &st.P50TTFTMS,
			&st.MeanPromptTok, &st.MeanOutputTok, &st.TotalCostUSD, &st.MeanCostUSD, &st.RefusalRate)
	return st, err
}
