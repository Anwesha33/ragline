"""Tests for the ingestion chunker.

Chunking decides what can ever be retrieved. A bug here cannot be fixed by a
better prompt, a better reranker or a better model — the passage simply is not
in the index in a usable form — so it is worth testing more carefully than its
size suggests.
"""

import os
import sys
import unittest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workers", "ingest")
)

from chunking import MAX_CHARS, chunk_document, split_sections  # noqa: E402


class TestSectionSplitting(unittest.TestCase):
    def test_tracks_the_full_heading_path(self):
        doc = "# Top\n\nintro\n\n## Middle\n\nbody\n\n### Deep\n\nleaf\n"
        headings = [s.heading for s in split_sections(doc) if s.text]
        self.assertEqual(headings, ["Top", "Top › Middle", "Top › Middle › Deep"])

    def test_pops_back_to_a_shallower_level(self):
        doc = "# A\n\na\n\n## B\n\nb\n\n# C\n\nc\n"
        headings = [s.heading for s in split_sections(doc) if s.text]
        self.assertEqual(headings, ["A", "A › B", "C"])

    def test_hash_inside_a_code_fence_is_not_a_heading(self):
        # A shell comment inside a fence looks exactly like a markdown heading.
        # Treating it as one shreds every document containing a bash example.
        doc = "# Real\n\n```bash\n# not a heading\necho hi\n```\n\ntail\n"
        headings = [s.heading for s in split_sections(doc) if s.text]
        self.assertEqual(headings, ["Real"])


class TestChunking(unittest.TestCase):
    def test_every_chunk_stays_within_the_budget(self):
        doc = "# Doc\n\n" + ("This is a sentence about payments. " * 400)
        for chunk in chunk_document(doc):
            self.assertLessEqual(len(chunk.content), MAX_CHARS)

    def test_ordinals_are_dense_and_ordered(self):
        doc = "# A\n\n" + ("x " * 900) + "\n\n## B\n\n" + ("y " * 900)
        ordinals = [c.ordinal for c in chunk_document(doc)]
        self.assertEqual(ordinals, list(range(len(ordinals))))

    def test_heading_is_prepended_to_the_embedded_text(self):
        # A chunk reading "Set it to 30" only embeds near "webhook retry
        # backoff" if its heading travels with it into the vector.
        doc = "# Webhooks\n\n## Retry schedule\n\nSet it to 30 seconds.\n"
        chunk = chunk_document(doc)[0]
        self.assertIn("Webhooks › Retry schedule", chunk.embed_text)
        self.assertIn("Set it to 30 seconds.", chunk.embed_text)
        # The stored content stays clean; only the embedded form is decorated.
        self.assertNotIn("›", chunk.content)

    def test_sections_under_different_headings_are_not_merged(self):
        doc = "# D\n\n## Alpha\n\nshort alpha body.\n\n## Beta\n\nshort beta body.\n"
        chunks = chunk_document(doc)
        for chunk in chunks:
            has_alpha = "alpha body" in chunk.content
            has_beta = "beta body" in chunk.content
            self.assertFalse(
                has_alpha and has_beta,
                "unrelated sections were merged into one chunk, which is relevant to neither query",
            )

    def test_oversized_section_is_split_with_overlap(self):
        body = "Sentence number {} explains a distinct detail. " * 1
        doc = "# D\n\n" + "".join(
            f"Sentence number {i} explains a distinct detail. " for i in range(400)
        )
        chunks = chunk_document(doc)
        self.assertGreater(len(chunks), 1)
        # Consecutive chunks from a forced split should share a little text, so
        # a fact sitting on the boundary survives in at least one of them.
        tail = chunks[0].content[-80:]
        self.assertTrue(
            any(word in chunks[1].content for word in tail.split()[-4:]),
            "forced splits should carry overlap",
        )
        _ = body

    def test_empty_and_whitespace_documents_produce_nothing(self):
        self.assertEqual(chunk_document(""), [])
        self.assertEqual(chunk_document("   \n\n  \t "), [])

    def test_trailing_scrap_folds_only_within_one_heading(self):
        # Same heading: the scrap is absorbed, because a two-word chunk
        # retrieves badly and dilutes the index.
        same = "# D\n\n" + ("A meaningful paragraph about refunds. " * 40) + "\n\nSee also.\n"
        chunks = chunk_document(same)
        self.assertIn("See also.", chunks[-1].content)
        self.assertGreater(len(chunks[-1].content), 100)

        # Different heading: the scrap stays on its own. Merging it would
        # produce a chunk covering two subjects, which is relevant to neither
        # query — worse than one small chunk.
        across = "# D\n\n" + ("A meaningful paragraph about refunds. " * 40) + "\n\n## Note\n\nSee also.\n"
        chunks = chunk_document(across)
        self.assertEqual(chunks[-1].content.strip(), "See also.")
        self.assertEqual(chunks[-1].heading, "D › Note")


if __name__ == "__main__":
    unittest.main()
