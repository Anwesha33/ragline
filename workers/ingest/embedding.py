"""Gemini embedding client for the ingestion worker.

Mirrors internal/embed in the Go service and must stay behaviourally identical:
a corpus embedded by this worker is searched by vectors the Go gateway
produces, so the task types, the output dimensionality and the normalisation
have to match exactly. A mismatch does not raise — it just returns worse
results, forever, which is the worst kind of bug to ship.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    def __init__(self, api_key: str, model: str, dim: int, timeout: int = 90,
                 max_retries: int = 5, batch_size: int = 32):
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.timeout = timeout
        self.max_retries = max_retries
        # Batch size is capped well below the API's limit because a failed
        # batch costs the whole batch: a 32-chunk retry is cheap, a 200-chunk
        # retry is not.
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._batch(texts[i : i + self.batch_size], TASK_DOCUMENT))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._batch([text], TASK_QUERY)[0]

    def _batch(self, texts: list[str], task: str) -> list[list[float]]:
        body = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task,
                    # Matryoshka truncation: the model natively emits 3072
                    # dimensions but pgvector's HNSW index refuses more than
                    # 2000.
                    "outputDimensionality": self.dim,
                }
                for t in texts
            ]
        }
        payload = self._post(f"{ENDPOINT}/models/{self.model}:batchEmbedContents", body)
        embeddings = payload.get("embeddings", [])
        if len(embeddings) != len(texts):
            # Accepting a short response would misalign every vector with its
            # chunk. Retrieval would keep working and return confidently wrong
            # passages, which is far worse than failing here.
            raise EmbeddingError(
                f"asked for {len(texts)} embeddings, received {len(embeddings)}"
            )
        return [normalize(e["values"]) for e in embeddings]

    def _post(self, url: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            if attempt:
                time.sleep(min(2**attempt, 20))
            req = urllib.request.Request(
                f"{url}?key={self.api_key}",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode(errors="replace")
                if exc.code == 429 or exc.code >= 500:
                    last_error = EmbeddingError(f"embedding api {exc.code}: {raw[:200]}")
                    delay = _retry_delay(raw)
                    if delay:
                        time.sleep(min(delay, 60))
                    continue
                # 400/403 will not improve on retry.
                raise EmbeddingError(f"embedding api {exc.code}: {raw[:400]}") from None
            except Exception as exc:  # noqa: BLE001 - network errors of any kind
                # The URL carries the API key, and urllib puts the URL in the
                # error, so never let the raw exception text escape.
                last_error = EmbeddingError(f"embedding request failed: {type(exc).__name__}")

        raise EmbeddingError(f"embedding failed after {self.max_retries} attempts: {last_error}")


def _retry_delay(raw: str) -> float | None:
    """Read google.rpc.RetryInfo out of a 429 body, if present."""
    try:
        details = json.loads(raw).get("error", {}).get("details", [])
    except (ValueError, AttributeError):
        return None
    for d in details:
        if str(d.get("@type", "")).endswith("RetryInfo") and d.get("retryDelay"):
            try:
                return float(str(d["retryDelay"]).rstrip("s"))
            except ValueError:
                return None
    return None


def normalize(values: list[float]) -> list[float]:
    """Scale to unit length.

    Truncated Gemini embeddings come back un-normalised — a 1536-dimension
    response has an L2 norm around 0.69. Cosine distance tolerates that, inner
    product does not, and any code assuming unit vectors silently misbehaves.
    """
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return values
    return [v / norm for v in values]
