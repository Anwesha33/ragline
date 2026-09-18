package store

import (
	"strings"
	"testing"
)

func TestVectorLiteral(t *testing.T) {
	got := vectorLiteral([]float32{0.5, -0.25, 0})
	if got != "[0.5,-0.25,0]" {
		t.Fatalf("vectorLiteral = %q", got)
	}
	if vectorLiteral(nil) != "[]" {
		t.Fatalf("empty vector = %q", vectorLiteral(nil))
	}
}

// A vector rendered in scientific notation is rejected by pgvector's parser,
// and float32 values near zero are exactly where a naive formatter reaches for
// it.
func TestVectorLiteralAvoidsPrecisionLoss(t *testing.T) {
	got := vectorLiteral([]float32{0.123456789, 1e-8})
	if strings.Count(got, ",") != 1 {
		t.Fatalf("unexpected shape: %q", got)
	}
	if !strings.HasPrefix(got, "[0.12345679") {
		t.Fatalf("float32 precision was truncated: %q", got)
	}
}

// Reciprocal Rank Fusion is the reason the two retrievers can be combined
// without normalising their scores. This documents the arithmetic the SQL
// performs, so a change to RRF_K has a test that explains what it does.
func TestRRFArithmetic(t *testing.T) {
	const k = 60
	rrf := func(ranks ...int) float64 {
		var sum float64
		for _, r := range ranks {
			sum += 1.0 / float64(k+r)
		}
		return sum
	}
	// A chunk found first by both retrievers must outrank one found first by
	// only one of them.
	both := rrf(1, 1)
	onlyVector := rrf(1)
	if both <= onlyVector {
		t.Fatalf("agreement between retrievers must win: %f vs %f", both, onlyVector)
	}
	// Two mid-ranked agreements should beat a single top rank, which is the
	// property that makes fusion useful rather than a tie-breaker.
	if rrf(5, 5) <= onlyVector {
		t.Fatalf("two agreeing mid ranks (%f) should beat one top rank (%f)", rrf(5, 5), onlyVector)
	}
}
