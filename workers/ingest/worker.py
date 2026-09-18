"""Kafka-driven ingestion worker.

    rag.documents.ingest ──▶ load raw text ──▶ chunk ──▶ embed ──▶ write chunks
                                                                       │
                                        failure ──▶ retry with backoff │
                                        exhausted ──▶ rag.documents.dlq│
                                                                       ▼
                                                          documents.status = ready

Two properties matter more than throughput here.

Idempotency: re-ingesting a document must replace its chunks, never append to
them. The delete and the insert share one transaction, so a crash halfway
through leaves the previous generation intact rather than half a corpus.

Visibility: a document that fails to ingest must say so. Silent failure in a
RAG pipeline shows up much later as "the assistant does not know about X", and
by then nobody remembers which upload broke.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass

import psycopg
from kafka import KafkaConsumer, KafkaProducer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunking import chunk_document, estimate_tokens  # noqa: E402
from embedding import Embedder, EmbeddingError  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("ingest")


@dataclass
class Settings:
    dsn: str
    brokers: list[str]
    topic: str
    dlq_topic: str
    group_id: str
    api_key: str
    model: str
    dim: int
    max_attempts: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            dsn=os.getenv(
                "POSTGRES_DSN",
                "postgres://ragline:ragline@localhost:5433/ragline?sslmode=disable",
            ),
            brokers=os.getenv("KAFKA_BROKERS", "localhost:9094").split(","),
            topic=os.getenv("KAFKA_INGEST_TOPIC", "rag.documents.ingest"),
            dlq_topic=os.getenv("KAFKA_INGEST_DLQ", "rag.documents.dlq"),
            group_id=os.getenv("KAFKA_CONSUMER_GROUP", "ragline-ingest"),
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
            dim=int(os.getenv("EMBEDDING_DIM", "1536")),
            max_attempts=int(os.getenv("MAX_INGEST_ATTEMPTS", "3")),
        )


class Worker:
    def __init__(self, settings: Settings):
        self.s = settings
        self.embedder = Embedder(settings.api_key, settings.model, settings.dim)
        self.running = True
        self.conn = psycopg.connect(settings.dsn, autocommit=True)
        self.producer = KafkaProducer(
            bootstrap_servers=settings.brokers,
            value_serializer=lambda v: json.dumps(v).encode(),
            acks="all",
        )
        self.consumer = KafkaConsumer(
            settings.topic,
            bootstrap_servers=settings.brokers,
            group_id=settings.group_id,
            value_deserializer=lambda v: json.loads(v.decode()),
            # Offsets are committed only after a document is fully written, so
            # a crash re-delivers the job rather than losing it.
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            max_poll_records=1,
            # Embedding a large document can take minutes; without a generous
            # poll interval the broker evicts the consumer mid-document and the
            # work is repeated forever.
            max_poll_interval_ms=15 * 60 * 1000,
        )

    def stop(self, *_args) -> None:
        log.info("shutdown signal received; finishing current document")
        self.running = False

    def run(self) -> None:
        log.info(
            "ingest worker started topic=%s model=%s dim=%d",
            self.s.topic, self.s.model, self.s.dim,
        )
        while self.running:
            batch = self.consumer.poll(timeout_ms=1000)
            for _partition, messages in batch.items():
                for message in messages:
                    if not self.running:
                        return
                    self.handle(message.value)
                    self.consumer.commit()
        log.info("ingest worker stopped")

    def handle(self, job: dict) -> None:
        doc_id = job.get("document_id")
        attempt = int(job.get("attempt", 1))
        if not doc_id:
            log.error("job has no document_id; dropping: %s", job)
            return

        started = time.time()
        try:
            chunk_count = self.ingest(doc_id)
        except Exception as exc:  # noqa: BLE001 - the worker must not die on one document
            self.fail(doc_id, job, attempt, exc)
            return

        log.info(
            "ingested document=%s chunks=%d seconds=%.1f",
            doc_id, chunk_count, time.time() - started,
        )

    def ingest(self, doc_id: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT raw_content, title FROM documents WHERE id = %s", (doc_id,)
            )
            row = cur.fetchone()
        if row is None:
            raise LookupError(f"document {doc_id} is not in the database")

        raw, title = row
        if not (raw or "").strip():
            raise ValueError("document has no content")

        self.set_status(doc_id, "chunking")

        chunks = chunk_document(raw)
        if not chunks:
            raise ValueError("document produced no chunks")

        vectors = self.embedder.embed_documents([c.embed_text for c in chunks])
        if len(vectors) != len(chunks):
            raise EmbeddingError(
                f"{len(chunks)} chunks but {len(vectors)} vectors"
            )

        # One transaction: a re-ingest replaces the previous generation
        # atomically, so the corpus is never half-old and half-new.
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
                cur.executemany(
                    """
                    INSERT INTO chunks
                        (document_id, ordinal, heading, content, token_estimate, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            doc_id,
                            c.ordinal,
                            c.heading,
                            c.content,
                            estimate_tokens(c.content),
                            _vector_literal(v),
                        )
                        for c, v in zip(chunks, vectors)
                    ],
                )
                cur.execute(
                    """
                    UPDATE documents
                       SET status='ready', chunk_count=%s, ingested_at=now(),
                           updated_at=now(), last_error=''
                     WHERE id=%s
                    """,
                    (len(chunks), doc_id),
                )
        log.info("document=%s title=%r chunks=%d", doc_id, title, len(chunks))
        return len(chunks)

    def fail(self, doc_id: str, job: dict, attempt: int, exc: Exception) -> None:
        reason = f"{type(exc).__name__}: {exc}"
        log.error("ingest failed document=%s attempt=%d: %s", doc_id, attempt, reason)

        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE documents
                   SET status = CASE WHEN %s < %s THEN 'pending' ELSE 'failed' END,
                       attempts = attempts + 1, last_error = %s, updated_at = now()
                 WHERE id = %s
                """,
                (attempt, self.s.max_attempts, reason[:2000], doc_id),
            )

        if attempt < self.s.max_attempts:
            # Republish with a linear backoff. Ingestion is not latency
            # sensitive, so sleeping in the worker is acceptable here in a way
            # it would not be on a serving path — and it keeps the topic
            # topology to two topics instead of a retry ladder.
            time.sleep(min(30 * attempt, 120))
            self.producer.send(
                self.s.topic, {**job, "attempt": attempt + 1, "last_error": reason}
            )
        else:
            self.producer.send(
                self.s.dlq_topic,
                {**job, "attempt": attempt, "last_error": reason},
            )
            log.error("document=%s dead-lettered after %d attempts", doc_id, attempt)
        self.producer.flush(timeout=10)

    def set_status(self, doc_id: str, status: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET status=%s, updated_at=now() WHERE id=%s",
                (status, doc_id),
            )


def _vector_literal(values: list[float]) -> str:
    """pgvector's text input format."""
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"


def main() -> None:
    settings = Settings.from_env()
    if not settings.api_key:
        log.error("GEMINI_API_KEY is required")
        raise SystemExit(1)

    worker = Worker(settings)
    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    worker.run()


if __name__ == "__main__":
    main()
