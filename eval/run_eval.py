#!/usr/bin/env python3
"""Score ragline's retrieval and answer quality against the golden set.

Three things are measured, deliberately kept separate because they fail for
different reasons and have different fixes:

  retrieval    Did the passage that contains the answer reach the generator?
               Measured twice — fusion order alone, and after reranking — so
               the reranker's contribution is a number rather than a belief.

  answer       Does the answer contain the fact the question asked for, does it
               cite a source, and does it refuse when the corpus cannot support
               an answer?

  groundedness An independent model reads the answer and the exact sources it
               was given, and reports whether every claim is supported. This is
               the hallucination measurement: an answer can contain the right
               fact and still assert three wrong ones alongside it.

The judge runs on a different model from the generator on purpose. Asking a
model to grade its own output measures its self-consistency, not its accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_API = os.getenv("RAGLINE_API", "http://localhost:8081")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.1-flash-lite")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"

JUDGE_SYSTEM = """You audit whether an answer is supported by the sources it was given.

You are not judging whether the answer is well written, complete, or matches what you personally know. You are judging one thing: is every factual claim in the answer traceable to the numbered sources?

Return JSON:
{
  "supported_claims": <integer>,
  "unsupported_claims": <integer>,
  "grounded": <true if there are no unsupported claims>,
  "unsupported": ["the specific claim that no source supports", ...],
  "notes": "one sentence"
}

A statement that the sources do not cover the question is itself grounded — refusing is not a hallucination. Generic connective phrasing is not a claim. Count only substantive factual assertions."""


def post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def judge(api_key: str, question: str, answer: str, sources: list[dict]) -> dict:
    """Ask an independent model whether the answer is grounded in its sources."""
    # Judge against the full passage, never the preview. Handing the judge a
    # 240-character snippet makes it report "unsupported" for claims the real
    # source states plainly — which measures the harness, not the system.
    rendered = "\n\n".join(
        f"[{s['marker']}] {s.get('title','')} › {s.get('heading','')}\n"
        f"{s.get('content') or s.get('snippet','')}"
        for s in sources
    ) or "(no sources were retrieved)"

    body = {
        "systemInstruction": {"parts": [{"text": JUDGE_SYSTEM}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"Question: {question}\n\nSources:\n{rendered}\n\nAnswer:\n{answer}"}],
            }
        ],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    url = f"{GEMINI_ENDPOINT}/models/{JUDGE_MODEL}:generateContent?key={api_key}"
    for attempt in range(4):
        if attempt:
            time.sleep(2**attempt)
        try:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read())
            text = "".join(
                p.get("text", "")
                for p in payload["candidates"][0]["content"]["parts"]
                if not p.get("thought")
            )
            return json.loads(text)
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503):
                continue
            # Never surface the raw error: the URL carries the API key.
            return {"error": f"judge HTTP {exc.code}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"judge failed: {type(exc).__name__}"}
    return {"error": "judge exhausted retries"}


def sources_of(items: list[dict]) -> list[str]:
    return [s.get("source_uri", "") for s in items]


def fact_relevance(items: list[dict], question: dict) -> list[bool]:
    """Mark which retrieved passages actually contain the answer.

    Document-level recall is the obvious metric and, on a corpus this size, a
    useless one: with eight documents and five slots, almost any retriever
    returns the right *document*, and both arms score 1.000. That measures the
    corpus, not the system.

    Fact-level relevance asks the harder question — did the passage containing
    the answer reach the generator — by testing each retrieved chunk for the
    same fact strings the answer is checked against. It needs no hand-labelled
    chunk ids, so it survives re-chunking, which a labelled set would not.
    """
    needles = [n.lower() for n in question.get("must_include", [])]
    out = []
    for item in items:
        text = (item.get("content") or item.get("snippet") or "").lower()
        if not needles:
            out.append(False)
        elif question.get("match_any"):
            out.append(any(n in text for n in needles))
        else:
            out.append(all(n in text for n in needles))
    return out


def recall_and_mrr_from_flags(flags: list[bool]) -> tuple[float, float]:
    """Hit rate and reciprocal rank over a boolean relevance list."""
    if not flags:
        return float("nan"), float("nan")
    hit = 1.0 if any(flags) else 0.0
    rr = 0.0
    for rank, ok in enumerate(flags, start=1):
        if ok:
            rr = 1.0 / rank
            break
    return hit, rr


def recall_and_mrr(retrieved: list[str], expected: list[str]) -> tuple[float, float]:
    """Recall over expected documents, and reciprocal rank of the first hit."""
    if not expected:
        return float("nan"), float("nan")
    found = {e for e in expected if e in retrieved}
    recall = len(found) / len(expected)
    rr = 0.0
    for rank, src in enumerate(retrieved, start=1):
        if src in expected:
            rr = 1.0 / rank
            break
    return recall, rr


def contains_expected(answer: str, question: dict) -> bool:
    needles = question.get("must_include", [])
    if not needles:
        return True
    low = answer.lower()
    if question.get("match_any"):
        return any(n.lower() in low for n in needles)
    return all(n.lower() in low for n in needles)


def looks_like_refusal(answer: str) -> bool:
    """Detect a refusal without an LLM call.

    Deliberately conservative: it matches the shapes the system prompt asks for
    ("the sources do not cover", "could not find"). A model that refuses in some
    other phrasing is counted as not refusing, which understates the refusal
    rate rather than flattering it.
    """
    low = answer.lower()
    markers = [
        "do not cover", "does not cover", "don't cover", "doesn't cover",
        "not contain", "no information", "could not find", "couldn't find",
        "not addressed", "not mentioned", "not specify", "does not specify",
        "not provided", "no mention", "unable to answer", "cannot answer",
        "not available in", "outside the scope",
    ]
    return any(m in low for m in markers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--golden", default=os.path.join(os.path.dirname(__file__), "golden.json"))
    ap.add_argument("--out", default="eval/report.json")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--skip-judge", action="store_true", help="skip groundedness (saves LLM quota)")
    ap.add_argument(
        "--retrieval-only",
        action="store_true",
        help="measure retrieval and reranking only, skipping generation and the judge. "
             "One reranking call per question instead of three LLM calls, which makes "
             "this cheap enough to run on every change to chunking or fusion.",
    )
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N questions")
    ap.add_argument(
        "--ids",
        default="",
        help="comma-separated question ids to evaluate, e.g. q01,q09,q12. Useful when "
             "the generation arm has to be run on a subset because of provider quota.",
    )
    args = ap.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key and not args.skip_judge:
        print("GEMINI_API_KEY is not set; run with --skip-judge to measure everything else",
              file=sys.stderr)
        return 1

    with open(args.golden) as fh:
        golden = json.load(fh)
    questions = golden["questions"]
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        questions = [q for q in questions if q["id"] in wanted]
    if args.limit:
        questions = questions[: args.limit]

    results = []
    print(f"ragline evaluation — {len(questions)} questions against {args.api}\n")

    for q in questions:
        row: dict = {"id": q["id"], "type": q["type"], "question": q["question"]}
        expected = q.get("expected_sources", [])

        # Arm A: retrieval with fusion order only. No LLM call, so this is free
        # and can be run on every change.
        t0 = time.time()
        try:
            plain = post_json(
                f"{args.api}/v1/search",
                {"query": q["question"], "top_n": args.top_n, "no_rerank": True},
            )
            plain_sources = sources_of(plain["chunks"])
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"search failed: {exc}"
            results.append(row)
            print(f"  {q['id']}  ERROR {row['error']}")
            continue
        row["search_ms"] = int((time.time() - t0) * 1000)
        row["recall_fusion"], row["mrr_fusion"] = recall_and_mrr(plain_sources, expected)
        if q["answerable"]:
            flags = fact_relevance(plain["chunks"], q)
            row["fact_hit_fusion"], row["fact_mrr_fusion"] = recall_and_mrr_from_flags(flags)
            row["fusion_chunks"] = [
                {"source": c.get("source_uri"), "heading": c.get("heading"), "relevant": f}
                for c, f in zip(plain["chunks"], flags)
            ]

        if args.retrieval_only:
            # Reranked retrieval measured through /v1/search rather than through
            # a full answer: one reranking call, no generation, no judge.
            try:
                ranked = post_json(
                    f"{args.api}/v1/search",
                    {"query": q["question"], "top_n": args.top_n},
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"rerank search failed: {exc}"
                results.append(row)
                print(f"  {q['id']}  ERROR {row['error']}")
                continue
            row["recall_reranked"], row["mrr_reranked"] = recall_and_mrr(
                sources_of(ranked["chunks"]), expected
            )
            if q["answerable"]:
                rflags = fact_relevance(ranked["chunks"], q)
                row["fact_hit_reranked"], row["fact_mrr_reranked"] = recall_and_mrr_from_flags(rflags)
                row["reranked_chunks"] = [
                    {"source": c.get("source_uri"), "heading": c.get("heading"),
                     "rerank_score": c.get("rerank_score"), "relevant": f}
                    for c, f in zip(ranked["chunks"], rflags)
                ]
            row["rerank_timings"] = ranked.get("timings", {})
            row["usage"] = {"rerank_ms": ranked.get("timings", {}).get("rerank_ms", 0),
                            "search_ms": ranked.get("timings", {}).get("search_ms", 0),
                            "embed_ms": ranked.get("timings", {}).get("embed_ms", 0)}
            # Retrieval-only mode makes no claim about the answer.
            row["correct"] = None
            row["refused"] = None
            print(
                f"  {q['id']} {q['type']:<12} "
                f"doc {_fmt(row['recall_fusion'])}→{_fmt(row['recall_reranked'])}  "
                f"fact hit {_fmt(row.get('fact_hit_fusion', float('nan')))}→"
                f"{_fmt(row.get('fact_hit_reranked', float('nan')))}  "
                f"fact MRR {_fmt(row.get('fact_mrr_fusion', float('nan')))}→"
                f"{_fmt(row.get('fact_mrr_reranked', float('nan')))}  "
                f"reordered {ranked.get('timings', {}).get('reordered_by_rerank', 0)}"
            )
            results.append(row)
            continue

        # Arm B: the real query path. Its `sources` are the reranked chunks that
        # actually reached the generator, so no extra rerank call is needed.
        try:
            answer = post_json(
                f"{args.api}/v1/chat",
                {
                    "question": q["question"],
                    "stream": False,
                    "no_cache": True,
                    "top_n": args.top_n,
                    "include_source_text": True,
                },
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"chat failed: {exc}"
            results.append(row)
            print(f"  {q['id']}  ERROR {row['error']}")
            continue

        ranked_sources = sources_of(answer.get("sources") or [])
        row["recall_reranked"], row["mrr_reranked"] = recall_and_mrr(ranked_sources, expected)
        if q["answerable"]:
            rflags = fact_relevance(answer.get("sources") or [], q)
            row["fact_hit_reranked"], row["fact_mrr_reranked"] = recall_and_mrr_from_flags(rflags)

        text = answer.get("answer", "")
        usage = answer.get("usage", {})
        row["answer"] = text
        row["refused"] = bool(answer.get("refused")) or looks_like_refusal(text)
        row["citation_count"] = len(answer.get("citations") or [])
        row["usage"] = usage

        if q["answerable"]:
            row["has_expected_fact"] = contains_expected(text, q)
            # An answerable question answered with a citation is the baseline
            # expectation; an uncited answer is not verifiable by the reader.
            row["correct"] = bool(row["has_expected_fact"]) and not row["refused"]
            row["cited"] = row["citation_count"] > 0
        else:
            # For an unanswerable question the only correct behaviour is a
            # refusal. Producing a confident answer here is the failure this
            # whole architecture exists to prevent.
            row["correct"] = row["refused"]
            row["has_expected_fact"] = None
            row["cited"] = None

        if not args.skip_judge:
            verdict = judge(api_key, q["question"], text, answer.get("sources") or [])
            row["judge"] = verdict
            row["grounded"] = verdict.get("grounded") if "error" not in verdict else None

        flag = "ok " if row["correct"] else "FAIL"
        grounded = row.get("grounded")
        gmark = {True: "grounded", False: "UNGROUNDED", None: "judge n/a"}[grounded]
        print(
            f"  {q['id']} {q['type']:<12} {flag}  "
            f"recall {row['recall_fusion'] if row['recall_fusion'] == row['recall_fusion'] else 1.0:.2f}→"
            f"{row['recall_reranked'] if row['recall_reranked'] == row['recall_reranked'] else 1.0:.2f}  "
            f"{usage.get('total_ms', 0)/1000:5.1f}s  ttft {usage.get('ttft_ms', 0)}ms  {gmark}"
        )
        results.append(row)

    report = summarize(results, questions)
    print_summary(report)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"summary": report, "results": results}, fh, indent=2)
    print(f"\nreport written to {args.out}")
    return 0


def _fmt(v: float) -> str:
    return "n/a " if v != v else f"{v:.2f}"


def _mean(values: list[float]) -> float:
    clean = [v for v in values if v == v]  # drop NaN
    return statistics.fmean(clean) if clean else 0.0


def summarize(results: list[dict], questions: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    answerable = [r for r in ok if r["type"] != "unanswerable"]
    unanswerable = [r for r in ok if r["type"] == "unanswerable"]
    judged = [r for r in ok if r.get("grounded") is not None]
    # Rows from a retrieval-only run carry no answer verdict and must not be
    # averaged into answer accuracy as zeros.
    scored = [r for r in ok if r.get("correct") is not None]
    scored_answerable = [r for r in answerable if r.get("correct") is not None]
    scored_unanswerable = [r for r in unanswerable if r.get("correct") is not None]

    return {
        "questions": len(results),
        "evaluated": len(ok),
        "errors": len(results) - len(ok),
        "retrieval": {
            "note": "document-level metrics saturate on a small corpus; the fact-level "
                    "numbers are the ones that discriminate between retrieval settings",
            "doc_recall_fusion_only": round(_mean([r.get("recall_fusion", float("nan")) for r in answerable]), 3),
            "doc_recall_after_rerank": round(_mean([r.get("recall_reranked", float("nan")) for r in answerable]), 3),
            "doc_mrr_fusion_only": round(_mean([r.get("mrr_fusion", float("nan")) for r in answerable]), 3),
            "doc_mrr_after_rerank": round(_mean([r.get("mrr_reranked", float("nan")) for r in answerable]), 3),
            "fact_hit_fusion_only": round(_mean([r.get("fact_hit_fusion", float("nan")) for r in answerable]), 3),
            "fact_hit_after_rerank": round(_mean([r.get("fact_hit_reranked", float("nan")) for r in answerable]), 3),
            "fact_mrr_fusion_only": round(_mean([r.get("fact_mrr_fusion", float("nan")) for r in answerable]), 3),
            "fact_mrr_after_rerank": round(_mean([r.get("fact_mrr_reranked", float("nan")) for r in answerable]), 3),
        },
        "answers": {
            "scored": len(scored),
            "accuracy_answerable": round(
                _mean([1.0 if r["correct"] else 0.0 for r in scored_answerable]), 3
            ),
            "cited_rate": round(_mean([1.0 if r.get("cited") else 0.0 for r in scored_answerable]), 3),
            # None rather than 0.0 when the run contained no unanswerable
            # questions: a mean over an empty set printed as 0.000 reads as
            # "it never refused correctly", which is the opposite of the truth.
            "refusal_correct_on_unanswerable": (
                round(_mean([1.0 if r["correct"] else 0.0 for r in scored_unanswerable]), 3)
                if scored_unanswerable
                else None
            ),
            "unanswerable_scored": len(scored_unanswerable),
            "false_refusals": sum(1 for r in scored_answerable if r["refused"]),
        },
        "groundedness": {
            "judged": len(judged),
            "grounded_rate": round(_mean([1.0 if r["grounded"] else 0.0 for r in judged]), 3),
            "ungrounded": [r["id"] for r in judged if not r["grounded"]],
        },
        "latency": {
            "p50_total_ms": _percentile([r.get("usage", {}).get("total_ms", 0) for r in ok], 50),
            "p95_total_ms": _percentile([r.get("usage", {}).get("total_ms", 0) for r in ok], 95),
            "p50_ttft_ms": _percentile([r.get("usage", {}).get("ttft_ms", 0) for r in ok], 50),
            "mean_rerank_ms": round(_mean([r.get("usage", {}).get("rerank_ms", 0) for r in ok])),
            "mean_search_ms": round(_mean([r.get("usage", {}).get("search_ms", 0) for r in ok])),
            "mean_embed_ms": round(_mean([r.get("usage", {}).get("embed_ms", 0) for r in ok])),
        },
        "cost": {
            "mean_usd_per_query": round(_mean([r.get("usage", {}).get("cost_usd", 0) for r in ok]), 6),
            "mean_prompt_tokens": round(_mean([r.get("usage", {}).get("prompt_tokens", 0) for r in ok])),
            "mean_output_tokens": round(_mean([r.get("usage", {}).get("output_tokens", 0) for r in ok])),
        },
        "by_type": {
            t: round(
                _mean([1.0 if r["correct"] else 0.0 for r in scored if r["type"] == t]), 3
            )
            for t in sorted({q["type"] for q in questions})
            if any(r["type"] == t for r in scored)
        },
    }


def _percentile(values: list[int], pct: float) -> int:
    clean = sorted(v for v in values if v)
    if not clean:
        return 0
    # Nearest-rank, which for a sample this small is more honest than
    # interpolating between two measurements that do not exist.
    idx = min(len(clean) - 1, int(round(pct / 100 * len(clean) + 0.5)) - 1)
    return clean[idx]


def print_summary(r: dict) -> None:
    ret, ans, gr, lat, cost = r["retrieval"], r["answers"], r["groundedness"], r["latency"], r["cost"]
    print(f"\n  questions                    {r['evaluated']}/{r['questions']} evaluated")
    print(f"  doc  recall  fusion→rerank  {ret['doc_recall_fusion_only']:.3f} → {ret['doc_recall_after_rerank']:.3f}")
    print(f"  doc  MRR     fusion→rerank  {ret['doc_mrr_fusion_only']:.3f} → {ret['doc_mrr_after_rerank']:.3f}")
    print(f"  fact hit@k   fusion→rerank  {ret['fact_hit_fusion_only']:.3f} → {ret['fact_hit_after_rerank']:.3f}")
    print(f"  fact MRR     fusion→rerank  {ret['fact_mrr_fusion_only']:.3f} → {ret['fact_mrr_after_rerank']:.3f}")
    if ans["scored"] == 0:
        print("  answers                      not measured (retrieval-only run)")
    else:
        print(f"  answer accuracy (answerable) {ans['accuracy_answerable']:.3f}")
    if ans["scored"]:
        print(f"  answers carrying a citation  {ans['cited_rate']:.3f}")
        if ans["refusal_correct_on_unanswerable"] is None:
            print("  correct refusals             n/a (no unanswerable questions in this run)")
        else:
            print(f"  correct refusals             {ans['refusal_correct_on_unanswerable']:.3f}"
                  f" over {ans['unanswerable_scored']}")
        print(f"  false refusals               {ans['false_refusals']}")
    if gr["judged"]:
        print(f"  groundedness                 {gr['grounded_rate']:.3f} over {gr['judged']} judged")
    if gr["ungrounded"]:
        print(f"    ungrounded: {', '.join(gr['ungrounded'])}")
    print(f"  latency   p50 {lat['p50_total_ms']}ms  p95 {lat['p95_total_ms']}ms  ttft(p50) {lat['p50_ttft_ms']}ms")
    print(f"    embed {lat['mean_embed_ms']}ms | search {lat['mean_search_ms']}ms | rerank {lat['mean_rerank_ms']}ms")
    print(f"  cost      ${cost['mean_usd_per_query']:.6f}/query "
          f"({cost['mean_prompt_tokens']} prompt + {cost['mean_output_tokens']} output tokens)")
    print(f"  accuracy by type             {r['by_type']}")


if __name__ == "__main__":
    raise SystemExit(main())
