-- ragline schema.
--
-- One Postgres instance serves three roles that are often split across three
-- systems: the document store, the vector index (pgvector) and the keyword
-- index (native full-text search). Keeping them together is what makes hybrid
-- retrieval a single query with a single consistency story — a chunk is either
-- visible to both searches or to neither.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    id            UUID PRIMARY KEY,
    source_uri    TEXT        NOT NULL,
    title         TEXT        NOT NULL DEFAULT '',
    -- Re-uploading identical bytes must not duplicate the corpus, and must not
    -- pay the embedding bill twice.
    content_hash  TEXT        NOT NULL UNIQUE,
    content_type  TEXT        NOT NULL DEFAULT 'text/markdown',
    byte_size     INTEGER     NOT NULL DEFAULT 0,
    status        TEXT        NOT NULL DEFAULT 'pending',
    chunk_count   INTEGER     NOT NULL DEFAULT 0,
    attempts      INTEGER     NOT NULL DEFAULT 0,
    last_error    TEXT        NOT NULL DEFAULT '',
    raw_content   TEXT        NOT NULL DEFAULT '',
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    ingested_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS documents_status_idx ON documents (status, created_at DESC);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      INTEGER     NOT NULL,
    heading      TEXT        NOT NULL DEFAULT '',
    content      TEXT        NOT NULL,
    token_estimate INTEGER   NOT NULL DEFAULT 0,
    -- 1536 dimensions: gemini-embedding-001 emits 3072, but pgvector's HNSW
    -- index refuses anything over 2000, so the ingest worker asks for a
    -- Matryoshka-truncated vector and renormalises it (see workers/ingest).
    embedding    vector(1536),
    -- Generated, not maintained by the application: a chunk cannot be written
    -- with a stale keyword index because there is no code path that could.
    tsv          tsvector GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', coalesce(heading, '')), 'A') ||
                    setweight(to_tsvector('english', content), 'B')
                 ) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, ordinal)
);

-- Cosine distance, matching the normalised vectors the embedder produces.
-- HNSW rather than IVFFlat: it needs no training pass, so a corpus can grow
-- from zero without an index rebuild step.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id, ordinal);

CREATE TABLE IF NOT EXISTS conversations (
    id         UUID PRIMARY KEY,
    client_id  TEXT        NOT NULL DEFAULT '',
    title      TEXT        NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT        NOT NULL,
    content         TEXT        NOT NULL,
    citations       JSONB       NOT NULL DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages (conversation_id, id);

-- Every answered question, with the latency broken down by stage. This table is
-- the difference between "it feels slow" and "reranking is 60% of p95".
CREATE TABLE IF NOT EXISTS query_logs (
    id                BIGSERIAL PRIMARY KEY,
    conversation_id   UUID,
    question          TEXT        NOT NULL,
    answered          BOOLEAN     NOT NULL DEFAULT TRUE,
    refusal_reason    TEXT        NOT NULL DEFAULT '',
    cache_hit         BOOLEAN     NOT NULL DEFAULT FALSE,
    retrieved_chunk_ids BIGINT[]  NOT NULL DEFAULT '{}',
    cited_chunk_ids     BIGINT[]  NOT NULL DEFAULT '{}',
    embed_ms          INTEGER     NOT NULL DEFAULT 0,
    search_ms         INTEGER     NOT NULL DEFAULT 0,
    rerank_ms         INTEGER     NOT NULL DEFAULT 0,
    generate_ms       INTEGER     NOT NULL DEFAULT 0,
    ttft_ms           INTEGER     NOT NULL DEFAULT 0,
    total_ms          INTEGER     NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER     NOT NULL DEFAULT 0,
    output_tokens     INTEGER     NOT NULL DEFAULT 0,
    embed_tokens      INTEGER     NOT NULL DEFAULT 0,
    cost_usd          NUMERIC(12,8) NOT NULL DEFAULT 0,
    model             TEXT        NOT NULL DEFAULT '',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS query_logs_created_idx ON query_logs (created_at DESC);
