# Architecture

## Why the language split

The gateway is Go and the ingestion workers are Python, and the boundary is not
arbitrary.

Serving is latency-sensitive and concurrency-heavy: a streaming answer holds a
connection open for seconds while fanning out to an embedding call, two database
queries, a reranking call and a streaming generation call. That is what Go is
good at, and the SSE path in particular benefits from explicit control over
flushing and cancellation.

Ingestion is batch text processing: chunking, tokenising, batching embeddings.
It is throughput-shaped, not latency-shaped, and it is where the text-processing
ecosystem lives. Making it a separate process also means a document that takes
four minutes to embed cannot occupy a serving goroutine.

The contract between them is deliberately narrow: Kafka carries a job that is
nothing but a document id, and Postgres carries the content. The message is a
pointer rather than a payload because Kafka's default message cap is 1MB and
documents routinely exceed it.

## Why one Postgres does three jobs

The corpus, the vector index and the keyword index all live in one database.

The alternative — a dedicated vector database alongside Postgres — means two
systems with two consistency stories. A chunk written to Postgres and not yet to
the vector store is invisible to search; a chunk deleted from Postgres and left
in the vector store is a citation pointing at nothing. Every one of those states
is reachable, and each needs its own reconciliation.

With pgvector, a chunk and its vector are written in the same transaction. A
re-ingest deletes and re-inserts a document's chunks atomically, so the corpus is
never half-old and half-new. The keyword index is a generated `tsvector` column,
which means it cannot be stale: there is no code path that could write a chunk
with the wrong keyword index, because no code writes it at all.

The honest limit is scale. HNSW in Postgres is fine into the millions of
vectors; past that, a purpose-built vector store with sharding and quantisation
starts to win, and the consistency complexity becomes the price of admission.

## Embeddings: two details that fail silently

**Task types.** `gemini-embedding-001` embeds a query and a passage differently
depending on the declared `taskType`. Using `RETRIEVAL_DOCUMENT` for both sides
does not error — search returns results, just worse ones. The Go query path and
the Python ingest path must therefore agree, which is why both files carry the
same comment.

**Matryoshka truncation.** The model natively emits 3072 dimensions. pgvector's
HNSW index refuses more than 2000, so the pipeline requests 1536. The returned
vector is a prefix of the full one and is **not** renormalised — a 1536-dimension
response has an L2 norm around 0.69. Cosine distance tolerates that; inner
product does not, and neither does anything that assumes unit vectors. Both
implementations renormalise before storing or querying.

Changing `EMBEDDING_DIM` invalidates the entire corpus. There is no migration
path other than re-ingesting, because the old and new vectors are not in the
same space.

## Hybrid retrieval and why fusion is RRF

Vector search finds passages that mean the same thing as the question. Keyword
search finds passages containing the exact rare token the question named. Each
fails where the other works: ask "what does NW-4221 mean" and the vector side
struggles, because an error code carries almost no semantic signal; ask "how long
before we can't put money back on the card" and the keyword side finds nothing,
because the document says "refund" and "180 days" and shares no vocabulary with
the question.

Combining them requires a decision about how to merge two rankings. A weighted
sum of raw scores is the obvious approach and the wrong one: cosine distance and
`ts_rank_cd` are on incomparable scales, and their distributions shift with
corpus size and query length, so any fixed weighting is tuned to one corpus and
wrong on the next.

Reciprocal Rank Fusion sidesteps the problem by reading only the ordering:

```
score(chunk) = Σ over retrievers of  1 / (k + rank_in_that_retriever)
```

with `k = 60`. No normalisation, no tuning, and a useful property: two mid-ranked
agreements outrank one top-ranked result from a single retriever. Agreement
between independent retrievers is evidence, and RRF spends it correctly.

The whole thing is one SQL statement with two CTEs, so there is no application
code holding two result sets and no second round trip.

## Reranking

Retrieval optimises for recall — cast a wide net and fuse. But the generator can
only be given a handful of passages, and models attend unevenly across a long
context, so a correct passage at position 9 of 12 is nearly as useless as one
that was never retrieved. Reranking is what converts a high-recall candidate set
into a high-precision top 5.

This is a **listwise LLM reranker**: one call scores all candidates together. The
alternative is a cross-encoder such as `bge-reranker`, which is faster and
cheaper per query but means hosting a model and a GPU. The trade is one extra
API call against a second serving stack.

Two properties make it safe to sit in the request path:

- **It degrades, never fails.** A reranking error, a timeout or an unparseable
  response falls back to fusion order. A worse ordering is a worse answer; a
  failed request is no answer.
- **Its output shape is constrained.** The scoring call declares a
  `responseSchema`. Asking for JSON without one produced
  `{"scores":[[{...}]]}` — valid JSON, wrong shape — often enough to matter. The
  parser stays tolerant anyway, because throwing away a whole scoring pass over
  a formatting wobble is a visible quality drop.

## Hallucination control

Four layers, in the order they fire:

1. **Refuse before generating.** If retrieval returns nothing, the model is never
   called. A model handed an empty context still produces a confident paragraph;
   a model that is not called cannot.
2. **Numbered sources and a prompt that permits refusal.** The system prompt
   states that "the provided documentation does not cover X" is a correct and
   useful answer. Without that, a model treats refusal as failure.
3. **Citation verification.** Every `[n]` marker is checked against the sources
   actually supplied. A marker pointing at a source that does not exist is
   stripped from the answer and logged, because it is both unusable to the reader
   and the clearest hallucination signal available.
4. **An independent judge in evaluation.** A different model reads the answer
   alongside the exact sources it was given and counts unsupported claims. It is
   a different model on purpose: asking a model to grade its own output measures
   self-consistency, not accuracy.

None of these guarantee a correct answer. They make an unsupported answer
detectable, which is the achievable goal.

## Caching, and what must never be cached

Three caches, with different invalidation stories:

- **Answers**, keyed by the normalised question *and a corpus fingerprint*.
  Including the corpus version means ingesting a document invalidates every
  cached answer without an explicit purge pass.
- **Query embeddings**, keyed by text, model and dimension — because switching
  either model or dimension changes the vector space, and serving a cached
  vector across that boundary corrupts retrieval silently.
- **Rate-limit buckets**, which are not really a cache.

An answer that depended on conversation history is **never** cached. "What about
the second one?" means something different in every conversation, and a cache
key built from the question alone would serve one conversation's answer to
another. The check is simply: history present, no caching.

Normalisation is deliberately shallow — case, whitespace, trailing punctuation.
This is exact matching after tidying, not semantic caching. Semantic caching
would need an embedding lookup per query and can serve the answer to a subtly
different question, which is a correctness bug dressed as an optimisation.

## Rate limiting

A token bucket in Redis, evaluated by a Lua script so the read-modify-write is
atomic across gateway instances without a lock.

A fixed window would let a client spend its entire allowance in the first moment
and then hammer the boundary, producing a thundering herd once per window. The
bucket smooths the sustained rate while still allowing a genuine burst.

It **fails open**: if Redis is unreachable, requests proceed. Rate limiting
protects against overload, and turning a cache outage into a total outage is a
worse failure than briefly serving unthrottled traffic. Note that the idempotency
layer in a payments system makes the opposite choice for the opposite reason —
which is the point: fail-open versus fail-closed is a per-dependency decision
about what the failure costs.

## Streaming

The answer streams as server-sent events, and the ordering of events is part of
the design: a `sources` event is emitted **before** generation starts, so the
client renders citations while the text arrives. Then `token` events, then a
`done` event carrying the latency breakdown and cost.

Two details that are easy to get wrong:

- The logging middleware wraps the `ResponseWriter`. If the wrapper does not
  forward `Flush`, streaming silently stops working and the response arrives in
  one lump at the end. There is a test for exactly this.
- `X-Accel-Buffering: no` is set because nginx and most CDNs otherwise buffer the
  entire response, producing the same symptom from outside the process.

The emit callback is invoked synchronously, so a slow consumer applies
backpressure rather than buffering the whole answer in memory, and a disconnected
client aborts generation instead of paying for tokens nobody will read.
