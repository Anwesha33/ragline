# Evaluation results

Reproduce with:

```bash
make eval                                            # full run
./.venv/bin/python eval/run_eval.py --retrieval-only # retrieval + reranking only, no generation
```

## Which configuration produced these numbers

**Every figure in this document was measured with the keyword arm scored by
`ts_rank_cd`**, which was the only lexical scorer at the time. The default is now
BM25 (`KEYWORD_RANKING=bm25`), so a fresh `make eval` measures a different
configuration and the retrieval numbers may not match.

That comparison — BM25 against `ts_rank_cd` on the golden set — has not been run.
It is exactly what the control arm exists for:

```bash
KEYWORD_RANKING=ts_rank make eval    # reproduces the numbers below
KEYWORD_RANKING=bm25    make eval    # the current default
```

Worth knowing before running it: fusion is RRF, which reads only rank *order*, so
the lexical scorer changes the keyword arm's ordering and nothing else. That
bounds how much the end-to-end numbers can move.

## What is measured, and why it is measured this way

Three things, kept separate because they fail for different reasons and have
different fixes:

- **Retrieval** — did the passage containing the answer reach the generator?
- **Answers** — does the answer contain the fact that was asked for, does it
  cite a source, and does it refuse when the corpus cannot support an answer?
- **Groundedness** — an independent model reads the answer alongside the exact
  sources it was given and counts unsupported claims.

The judge runs on a different model from the generator on purpose. Asking a
model to grade its own output measures self-consistency, not accuracy.

The golden set is 13 questions over an 8-document, 45-chunk corpus, labelled by
the retrieval behaviour each one stresses: 4 lexical (a rare exact token that
favours BM25), 5 semantic (a paraphrase sharing almost no vocabulary with the
source, which favours the vector side), 2 multihop (needing two documents), and
2 unanswerable (no support in the corpus, so the only correct behaviour is
refusal).

### The metric that had to be replaced

The first version measured document-level recall — did the right *document*
appear in the top 5. Both retrieval arms scored 1.000 on every question. That
number was true and useless: with 8 documents and 5 slots, almost any retriever
returns the right document, so the metric was measuring the corpus, not the
system.

It was replaced with **fact-level** relevance: a retrieved chunk counts as
relevant only if it contains the fact string the answer is checked against. No
hand-labelled chunk ids are involved, so the metric survives a change to
chunking — which a labelled set would not. Both are reported below; only the
fact-level numbers discriminate.

## Retrieval and reranking

13 questions, no generation. Reranker `gemini-flash-lite-latest`, `CONTEXT_TOP_N=5`,
12 candidates reranked out of 30+30 fused.

| Metric | Fusion only | After reranking |
| --- | ---: | ---: |
| Document recall@5 | 1.000 | 1.000 |
| Document MRR | 1.000 | 1.000 |
| **Fact-level hit@5** | **1.000** | **1.000** |
| **Fact-level MRR** | **0.927** | **1.000** |

Hybrid fusion alone already retrieves the answer-bearing passage somewhere in
the top 5 for every question. What reranking changes is *where*: fact-level MRR
rises from 0.927 to 1.000, meaning that after reranking the answer-bearing
passage is at rank 1 for all 11 answerable questions.

The whole difference comes from one question, and it is worth naming rather than
averaging away. q09 — *"A customer has raised a chargeback. Can we still refund
them through the API?"* — put the passage containing `NW-4093` at rank 5 under
fusion (MRR 0.20) and at rank 1 after reranking. The question says "chargeback";
the document says "dispute". Fusion retrieved it on semantic similarity but
ranked four topically-related refund passages above it; the reranker, which
reads the question and the passage together, saw which one actually answered it.

Reranking reordered a median of 7 of 12 candidates per query, so it is doing
substantial work even where the top-1 was already correct.

## Answers and groundedness

6 questions (`q01, q03, q06, q09, q12, q13`), covering all four question types
including both unanswerable ones. Generator `gemini-3.5-flash-lite`, judge
`gemini-3.1-flash-lite`.

| Metric | Value |
| --- | ---: |
| Answer accuracy on answerable questions | 1.000 |
| Correct refusals on unanswerable questions | 1.000 (2/2) |
| False refusals | 0 |
| **Groundedness (judged)** | **1.000** (6/6) |
| Answers carrying a citation | 0.750 → **1.000** after the citation-parsing fix below |

## Latency and cost

| Stage | Median |
| --- | ---: |
| Embed the question | ~500ms (cached on repeat) |
| Hybrid search (both retrievers, fused, in one query) | **1–2ms** |
| Rerank | 1.6–2.6s |
| Total | 2.2–4.5s |
| **Time to first token** | **0.65–1.1s** |

Cost is ~$0.00042 per query at roughly 3,100 prompt and 300 output tokens.

Two things stand out. **Retrieval is not the bottleneck** — hybrid search over
pgvector plus full-text, fused with RRF in a single SQL statement, returns in
1–2ms. Every second in that table belongs to a model API call. And **reranking
is the largest controllable cost in the latency budget**, at roughly half of
total time. On the free tier this is dominated by provider round-trip rather
than by reasoning tokens (a trivial prompt to the same model takes ~10s under
load), so the honest reading is that these latencies are an upper bound; a
hosted cross-encoder would make the rerank stage tens of milliseconds.

Streaming is why the user-visible number is 0.65–1.1s rather than 2–4s: the
`sources` event is emitted before generation begins, so citations render while
the text arrives.

## Two bugs the evaluation found

Both are worth stating because they are the kind of thing an evaluation exists
to catch.

**The groundedness metric was measuring the harness.** The first run reported
groundedness of 0.286 — five of seven answers "ungrounded". The answers were
correct. The harness was passing the judge a 240-character *preview* of each
source instead of the passage the generator actually read, so the judge
correctly reported that the truncated text did not support the claims. Fixed by
adding `include_source_text` to the chat API and judging against the full
passage; groundedness went to 1.000 with no change to the system under test.

**Citation parsing missed a common format.** Citation rate was 0.750 because one
answer cited as `[1, 2]` — a grouped marker — and the regex only matched a lone
bracketed number. A correctly cited answer was recorded as uncited. Fixed to
accept `[1]`, `[1][2]` and `[1, 2]`, with the marker-stripping path rewritten so
that removing one invented source from a group keeps the valid ones.

## What these numbers do not show

- **Coverage is uneven.** Retrieval is measured on all 13 questions; answers and
  groundedness on 6 of 13. The Gemini free tier allows 20 requests per day *per
  model*, and a full run needs 13 generation + 13 reranking + 13 judge calls.
  On a billed key, `make eval` runs the whole set.
- **Perfect scores on 13 questions mean "no failures yet", not "cannot fail".**
  Fact-level hit@5 of 1.000 over an 8-document corpus says the pipeline handles
  this corpus; it says little about a corpus of 100,000 documents where the
  reranker's candidate window becomes the binding constraint.
- **The corpus was written by the same person who wrote the questions.** That is
  the standard weakness of a hand-built golden set: the questions use vocabulary
  the author knows is in the documents. The semantic questions were deliberately
  paraphrased away from the source wording to blunt this, but it is not
  eliminated.
- **One run each, no variance.** Temperature is 0 for reranking and 0.2 for
  generation, so runs are fairly stable, but "fairly stable" is not measured.
- **Groundedness is judged by a model.** A 1.000 groundedness rate means an
  independent model found no unsupported claim in 6 answers. It is evidence, not
  proof.
