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

-- ---------------------------------------------------------------- BM25
--
-- Postgres full-text search ranks with ts_rank_cd — coverage density — not
-- BM25. They are different functions with different behaviour: ts_rank_cd
-- rewards passages where the query terms appear close together, and has no
-- notion of how rare a term is across the corpus or of how long the passage is.
-- BM25 has both, which is why it is the standard lexical baseline and why the
-- keyword arm of a hybrid retriever is normally described as BM25.
--
-- Postgres core cannot do it, so the term statistics BM25 needs are maintained
-- here: a per-chunk term frequency table, a corpus-wide document frequency
-- table, and the two corpus scalars (chunk count, total length) that give the
-- average chunk length.
--
-- These are maintained by trigger rather than by a materialised view that gets
-- refreshed after ingestion. A refresh would open a window in which a chunk is
-- searchable but its term statistics are not yet counted, which is exactly the
-- half-written state the generated tsvector column was chosen to make
-- impossible. A trigger keeps the statistics in the same transaction as the
-- chunk, so they cannot disagree.

CREATE TABLE IF NOT EXISTS chunk_terms (
    chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    term     TEXT   NOT NULL,
    -- Weighted term frequency. Postgres weights the heading 'A' and the body
    -- 'B'; BM25 has no field weights, so an 'A' occurrence is counted twice.
    -- That is a deliberate approximation of BM25F which preserves the property
    -- the weighted tsvector already had: a term in the heading counts for more.
    tf       INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, term)
);
CREATE INDEX IF NOT EXISTS chunk_terms_term_idx ON chunk_terms (term);

CREATE TABLE IF NOT EXISTS chunk_stats (
    chunk_id BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    -- Length in weighted tokens, the |D| of the BM25 length normalisation.
    len      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS corpus_terms (
    term TEXT PRIMARY KEY,
    -- Document frequency: how many chunks contain this term at all.
    df   BIGINT NOT NULL
);

-- A single row. The CHECK makes a second row impossible rather than merely
-- unlikely.
CREATE TABLE IF NOT EXISTS corpus_stats (
    only_row    BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (only_row),
    chunk_count BIGINT NOT NULL DEFAULT 0,
    total_len   BIGINT NOT NULL DEFAULT 0
);
INSERT INTO corpus_stats (only_row) VALUES (TRUE) ON CONFLICT DO NOTHING;

-- chunk_term_rows explodes a chunk's tsvector into (term, weighted tf).
CREATE OR REPLACE FUNCTION chunk_term_rows(v tsvector)
RETURNS TABLE (term TEXT, tf INTEGER)
LANGUAGE sql IMMUTABLE AS $$
    SELECT u.lexeme,
           -- An 'A'-weighted position counts twice; see chunk_terms.tf.
           SUM(CASE WHEN w = 'A' THEN 2 ELSE 1 END)::INTEGER
    FROM unnest(v) AS u
    CROSS JOIN LATERAL unnest(
        CASE WHEN array_length(u.weights, 1) IS NULL THEN ARRAY['D'::"char"] ELSE u.weights END
    ) AS w
    GROUP BY u.lexeme
$$;

CREATE OR REPLACE FUNCTION chunks_bm25_insert() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    total INTEGER;
BEGIN
    INSERT INTO chunk_terms (chunk_id, term, tf)
    SELECT NEW.id, t.term, t.tf FROM chunk_term_rows(NEW.tsv) t
    ON CONFLICT (chunk_id, term) DO UPDATE SET tf = EXCLUDED.tf;

    SELECT COALESCE(SUM(tf), 0) INTO total FROM chunk_terms WHERE chunk_id = NEW.id;

    INSERT INTO chunk_stats (chunk_id, len) VALUES (NEW.id, total)
    ON CONFLICT (chunk_id) DO UPDATE SET len = EXCLUDED.len;

    INSERT INTO corpus_terms (term, df)
    SELECT t.term, 1 FROM chunk_term_rows(NEW.tsv) t
    ON CONFLICT (term) DO UPDATE SET df = corpus_terms.df + 1;

    UPDATE corpus_stats
       SET chunk_count = chunk_count + 1,
           total_len   = total_len + total
     WHERE only_row;
    RETURN NEW;
END $$;

CREATE OR REPLACE FUNCTION chunks_bm25_delete() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    total INTEGER;
BEGIN
    SELECT COALESCE(len, 0) INTO total FROM chunk_stats WHERE chunk_id = OLD.id;

    UPDATE corpus_terms c
       SET df = c.df - 1
      FROM chunk_term_rows(OLD.tsv) t
     WHERE c.term = t.term;
    DELETE FROM corpus_terms WHERE df <= 0;

    UPDATE corpus_stats
       SET chunk_count = GREATEST(chunk_count - 1, 0),
           total_len   = GREATEST(total_len - COALESCE(total, 0), 0)
     WHERE only_row;
    -- chunk_terms and chunk_stats go with the chunk via ON DELETE CASCADE.
    RETURN OLD;
END $$;

DROP TRIGGER IF EXISTS chunks_bm25_ins ON chunks;
CREATE TRIGGER chunks_bm25_ins AFTER INSERT ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_bm25_insert();

DROP TRIGGER IF EXISTS chunks_bm25_del ON chunks;
CREATE TRIGGER chunks_bm25_del BEFORE DELETE ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_bm25_delete();

-- Backfill, for a corpus that was ingested before BM25 existed.
--
-- The triggers only fire on new rows, so an existing corpus would have chunks
-- that are searchable by tsvector and invisible to BM25 — a silent retrieval
-- regression on upgrade. This runs once: it is a no-op when the statistics are
-- already populated, and a no-op on an empty corpus.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM chunks) AND NOT EXISTS (SELECT 1 FROM chunk_terms) THEN
        INSERT INTO chunk_terms (chunk_id, term, tf)
        SELECT c.id, t.term, t.tf FROM chunks c, chunk_term_rows(c.tsv) t
        ON CONFLICT DO NOTHING;

        INSERT INTO chunk_stats (chunk_id, len)
        SELECT chunk_id, COALESCE(SUM(tf), 0) FROM chunk_terms GROUP BY chunk_id
        ON CONFLICT (chunk_id) DO UPDATE SET len = EXCLUDED.len;

        INSERT INTO corpus_terms (term, df)
        SELECT term, COUNT(DISTINCT chunk_id) FROM chunk_terms GROUP BY term
        ON CONFLICT (term) DO UPDATE SET df = EXCLUDED.df;

        UPDATE corpus_stats SET
            chunk_count = (SELECT COUNT(*) FROM chunks),
            total_len   = (SELECT COALESCE(SUM(len), 0) FROM chunk_stats)
        WHERE only_row;

        RAISE NOTICE 'BM25 statistics backfilled for % chunks', (SELECT COUNT(*) FROM chunks);
    END IF;
END $$;
