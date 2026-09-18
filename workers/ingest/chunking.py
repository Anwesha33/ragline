"""Structure-aware chunking for markdown and plain text.

Why not a fixed character window with overlap, which is what every tutorial
does: a fixed window cuts through the middle of a sentence, a code block or a
table, and the retrieved passage then reads as nonsense to both the reranker
and the generator. It also throws away the single most useful retrieval signal
a document has — its headings.

This splits on markdown structure first, then packs sections into chunks that
respect a size budget, and only falls back to sentence-level splitting when a
single section is too large. Every chunk carries the heading path it came from,
which is prepended to the embedded text so that a chunk reading "Set it to 30"
still embeds near "webhook retry backoff".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Token budgets are expressed in characters because tokenising here would mean
# shipping a tokenizer that matches the embedding model's, which is not
# published. The ratio is roughly 4 characters per token for English prose, so
# 2400 characters lands near 600 tokens — comfortably inside any embedding
# model's window and small enough that a chunk is about one idea.
TARGET_CHARS = 2400
MAX_CHARS = 3600
MIN_CHARS = 280
OVERLAP_CHARS = 200

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_RE = re.compile(r"^\s*```")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Chunk:
    ordinal: int
    heading: str
    content: str

    @property
    def embed_text(self) -> str:
        """The text actually sent to the embedding model.

        The heading path is prepended so a chunk whose body is full of pronouns
        and bare numbers still carries its subject into the vector.
        """
        if self.heading:
            return f"{self.heading}\n\n{self.content}"
        return self.content


@dataclass
class _Section:
    heading_path: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    @property
    def heading(self) -> str:
        return " › ".join(self.heading_path)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def split_sections(text: str) -> list[_Section]:
    """Split on markdown headings, tracking the full heading path.

    Fenced code blocks are tracked so that a `# comment` inside a shell snippet
    is not mistaken for a heading — a mistake that silently shreds every
    document containing a bash example.
    """
    sections: list[_Section] = []
    stack: list[str] = []
    current = _Section(heading_path=[])
    in_fence = False

    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            current.lines.append(line)
            continue

        match = None if in_fence else HEADING_RE.match(line)
        if match:
            if current.text:
                sections.append(current)
            level = len(match.group(1))
            title = match.group(2).strip()
            # Pop to the parent level, then push this heading.
            stack = stack[: level - 1]
            stack.append(title)
            current = _Section(heading_path=list(stack))
        else:
            current.lines.append(line)

    if current.text:
        sections.append(current)
    return sections


def _split_oversized(body: str) -> list[str]:
    """Break a too-large section on sentence boundaries, with overlap.

    Overlap exists only here, at forced splits. Overlapping every chunk
    uniformly inflates the corpus by the overlap ratio and makes near-duplicate
    passages compete with each other in the result list; applying it only where
    a split was involuntary keeps the cost proportional to the risk.
    """
    sentences = SENTENCE_END_RE.split(body)
    parts: list[str] = []
    buf = ""

    for sentence in sentences:
        if len(buf) + len(sentence) + 1 <= MAX_CHARS:
            buf = f"{buf} {sentence}".strip()
            continue
        if buf:
            parts.append(buf)
            tail = buf[-OVERLAP_CHARS:]
            # Resume from a word boundary so the overlap does not begin
            # mid-word, which reads as a typo to the model.
            space = tail.find(" ")
            buf = (tail[space + 1 :] if space >= 0 else "") + " " + sentence
            buf = buf.strip()
        else:
            # One sentence longer than the whole budget: a minified blob or a
            # long table row. Hard-split it rather than emitting it whole.
            for i in range(0, len(sentence), MAX_CHARS):
                parts.append(sentence[i : i + MAX_CHARS])
            buf = ""

    if buf:
        parts.append(buf)
    return parts


def chunk_document(text: str) -> list[Chunk]:
    """Turn a document into chunks, in reading order."""
    chunks: list[Chunk] = []
    pending_body = ""
    pending_heading = ""

    def flush() -> None:
        nonlocal pending_body, pending_heading
        body = pending_body.strip()
        if body:
            chunks.append(Chunk(ordinal=len(chunks), heading=pending_heading, content=body))
        pending_body = ""

    for section in split_sections(text):
        body = section.text
        if not body:
            continue

        if len(body) > MAX_CHARS:
            flush()
            for part in _split_oversized(body):
                chunks.append(Chunk(ordinal=len(chunks), heading=section.heading, content=part))
            pending_heading = section.heading
            continue

        # Pack small consecutive sections together, but never across a change
        # of heading — merging two unrelated sections produces a chunk that is
        # relevant to neither query.
        if pending_body and (
            section.heading != pending_heading
            or len(pending_body) + len(body) > TARGET_CHARS
        ):
            flush()

        pending_heading = section.heading
        pending_body = f"{pending_body}\n\n{body}".strip() if pending_body else body

    flush()

    # A trailing scrap ("See also.") retrieves badly and dilutes the index, so
    # fold it into its predecessor — but only when the two share a heading.
    # Folding across a heading boundary produces a chunk that belongs to two
    # different subjects and is relevant to neither query, which is a worse
    # outcome than one small chunk.
    if (
        len(chunks) > 1
        and len(chunks[-1].content) < MIN_CHARS
        and chunks[-1].heading == chunks[-2].heading
    ):
        tail = chunks.pop()
        chunks[-1].content = f"{chunks[-1].content}\n\n{tail.content}"

    for i, c in enumerate(chunks):
        c.ordinal = i
    return chunks


def estimate_tokens(text: str) -> int:
    """Rough token count. Used only for reporting, never for budgeting."""
    return max(1, len(text) // 4)
