# ragline

A retrieval-augmented question-answering platform. Documents are ingested
asynchronously through Kafka, chunked and embedded by Python workers, and
searched with hybrid retrieval — vector similarity fused with full-text search —
then reranked before an LLM answers with citations over a streaming response.

```
  POST /v1/documents                                    POST /v1/chat
        │                                                     │
        ▼                                                     ▼
  ┌───────────┐                                      ┌─────────────────┐
  │  gateway  │── raw text ──▶ Postgres              │     gateway     │
  │    (Go)   │── job ──▶ Kafka                      └────────┬────────┘
  └───────────┘              │                                │
                             ▼                       rate limit (Redis token bucket)
                   ┌───────────────────┐                      │
                   │  ingest workers   │             answer cache (Redis)
                   │     (Python)      │                      │
                   └─────────┬─────────┘             embed query (Gemini, cached)
                             │                                │
              structure-aware chunking                        ▼
                             │                     ┌──────────────────────┐
                    batch embeddings               │  hybrid retrieval    │
                             │                     │  pgvector  +  BM25   │
                             ▼                     │      ↓ RRF fusion    │
                    Postgres + pgvector ──────────▶└──────────┬───────────┘
                    (vector index + tsvector)                 │
                                                       rerank (LLM listwise)
                                                              │
                                                       generate (streaming)
                                                              │
                                                    verify citations ──▶ SSE
```

## What it demonstrates

- **A real ingestion pipeline**, not a loop over a folder: Kafka, retries, a
  dead-letter topic, per-document transactions so a re-ingest replaces a
  document's chunks instead of appending to them, and status you can query.
- **Hybrid retrieval that is actually hybrid.** Vector search and BM25
  full-text search run as independent rankers and are combined with Reciprocal
  Rank Fusion, which needs no score normalisation and no per-corpus tuning.
- **Reranking, measured.** The reranker's contribution to recall and MRR is a
  number in `docs/RESULTS.md`, produced by running the same golden set with it
  on and off.
- **Hallucination control that is structural, not hopeful.** Retrieval failure
  refuses before the model is ever called; every citation the model emits is
  validated against the sources it was actually given; an independent judge
  model scores groundedness in evaluation.
- **Operational honesty.** Every query records latency broken down by stage
  (embed / search / rerank / generate), token counts and cost, in Postgres.

## Quick start

```bash
cp .env.example .env      # add GEMINI_API_KEY
make venv                 # Python environment for the workers
make up                   # postgres+pgvector, redis, kafka, gateway, 2 ingest workers
make ingest               # upload corpus/ and wait for it to be indexed
make ask Q="how long are idempotency keys kept?"
```

`make ask` streams the answer as server-sent events: a `sources` event first, so
the UI can render citations while the text arrives, then `token` events, then a
`done` event carrying latency, token and cost accounting.

```bash
make search Q="webhook retry schedule"   # retrieval only, no generation
make stats                               # p50/p95 latency, cost, cache hit rate
make eval                                # score against the golden set
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/documents` | Upload a document (JSON or raw markdown). Returns `202` and a document id; `200` if identical content is already indexed. |
| `GET` | `/v1/documents/{id}` | Ingestion status, chunk count, and the error if it failed. |
| `POST` | `/v1/chat` | Ask a question. Streams SSE by default; `{"stream": false}` returns one JSON object. |
| `POST` | `/v1/search` | Retrieval only, with per-stage timings. `{"no_rerank": true}` returns fusion order. |
| `GET` | `/v1/conversations/{id}` | Message history. |
| `GET` | `/v1/stats` | Latency percentiles, token and cost aggregates, corpus size. |
| `GET` | `/healthz`, `/readyz` | Liveness, and readiness including corpus state. |

## The retrieval pipeline

1. **Embed the question** with task type `RETRIEVAL_QUERY`. Passages are
   embedded with `RETRIEVAL_DOCUMENT`. Using one task type for both degrades
   retrieval silently — search still works, it just returns worse results.
2. **Two independent searches.** pgvector cosine distance over an HNSW index,
   and **BM25** over a generated `tsvector` that weights headings above body
   text. Postgres cannot do BM25 natively — its `ts_rank_cd` is coverage
   density, with no notion of term rarity or passage length — so the term
   statistics BM25 needs are maintained by trigger, in the same transaction as
   the chunk. `KEYWORD_RANKING=ts_rank` switches back to `ts_rank_cd` as a
   control arm, so the difference can be measured rather than assumed.
3. **Reciprocal Rank Fusion.** `score = Σ 1/(k + rank)` with `k = 60`. RRF reads
   only the ordering, so it needs no normalisation between cosine distance and a
   BM25 score — two quantities on incomparable scales whose distributions shift
   as the corpus grows.
4. **Rerank** the top 12 candidates with a listwise LLM call, keeping the top 5.
   Retrieval optimises for recall; the generator can only read a handful of
   passages, so something has to turn a high-recall candidate set into a
   high-precision context.
5. **Generate**, streaming, with the sources numbered and a system prompt that
   requires a citation per claim and permits refusal.
6. **Verify citations.** Markers pointing at sources that were never supplied
   are stripped and logged — the clearest hallucination signal available.

## Repository layout

```
cmd/gateway            HTTP API, streaming, rate limiting
internal/store         Postgres: corpus, hybrid search SQL, history, telemetry
internal/chat          The query path and citation verification
internal/rerank        Listwise LLM reranker, with a no-op control arm
internal/embed         Gemini embeddings: task types, MRL truncation, renormalisation
internal/llm           Generation, including the SSE streaming client
internal/cache         Redis: token-bucket rate limiting, answer and embedding caches
internal/queue         Kafka producer for ingestion jobs
workers/ingest         Python: Kafka consumer, chunking, batch embedding, upsert
corpus/                Sample documentation corpus
eval/                  Golden question set, scoring harness, chunker tests
docs/                  Architecture, results, interview guide
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — why each piece is shaped this way
- [docs/RESULTS.md](docs/RESULTS.md) — measured retrieval, groundedness, latency and cost
