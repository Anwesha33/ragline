package cache

import "testing"

// The cache key must collapse the differences that should not cause a miss,
// and must not collapse anything that changes the answer.
func TestNormalizeQuestion(t *testing.T) {
	same := []string{
		"How long are idempotency keys kept?",
		"how long are idempotency keys kept",
		"  How   long are idempotency   keys kept?!  ",
		"HOW LONG ARE IDEMPOTENCY KEYS KEPT.",
	}
	first := normalizeQuestion(same[0])
	for _, q := range same[1:] {
		if normalizeQuestion(q) != first {
			t.Fatalf("%q normalised to %q, want %q", q, normalizeQuestion(q), first)
		}
	}
	if normalizeQuestion("how long are refunds kept") == first {
		t.Fatal("a different question must not normalise to the same key")
	}
}

// An answer is only valid for the corpus it was grounded in. Ingesting a
// document must therefore change the key, which is how cached answers are
// invalidated without an explicit purge.
func TestAnswerKeyIncludesCorpusVersion(t *testing.T) {
	a := AnswerKey("what is the refund window", "d8-c45-1726704000")
	b := AnswerKey("what is the refund window", "d9-c52-1726790000")
	if a == b {
		t.Fatal("a corpus change must change the answer cache key")
	}
	if a != AnswerKey("What is the refund window?", "d8-c45-1726704000") {
		t.Fatal("cosmetic differences in the question must not change the key")
	}
}

// Switching model or dimension changes the vector space entirely; serving a
// cached vector across that boundary would silently corrupt retrieval.
func TestEmbeddingKeyIsScopedToModelAndDimension(t *testing.T) {
	base := EmbeddingKey("refund window", "gemini-embedding-001", 1536)
	if base == EmbeddingKey("refund window", "gemini-embedding-001", 768) {
		t.Fatal("dimension must be part of the embedding cache key")
	}
	if base == EmbeddingKey("refund window", "other-model", 1536) {
		t.Fatal("model must be part of the embedding cache key")
	}
}

func TestKeysAreNamespacedAndBounded(t *testing.T) {
	k := AnswerKey("q", "v")
	if len(k) == 0 || k[:4] != "ans:" {
		t.Fatalf("answer key %q is not namespaced", k)
	}
	if len(k) > 40 {
		t.Fatalf("key %q is longer than intended; raw text may be leaking into it", k)
	}
}
