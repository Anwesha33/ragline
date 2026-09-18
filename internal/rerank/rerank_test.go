package rerank

import (
	"context"
	"testing"

	"github.com/Anwesha33/ragline/internal/store"
)

func TestParseScoresAcceptsTheIntendedShape(t *testing.T) {
	got, err := parseScores(`{"scores":[{"index":0,"score":7.5},{"index":2,"score":10}]}`, 3)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	want := []float64{7.5, 0, 10}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("scores = %v, want %v", got, want)
		}
	}
}

// The model really does return this. It is valid JSON and completely unusable
// unless the parser flattens it.
func TestParseScoresAcceptsNestedArray(t *testing.T) {
	got, err := parseScores(`{"scores":[[{"index":0,"score":3},{"index":1,"score":9}]]}`, 2)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got[0] != 3 || got[1] != 9 {
		t.Fatalf("scores = %v", got)
	}
}

func TestParseScoresAcceptsBareNumbersPositionally(t *testing.T) {
	got, err := parseScores(`{"scores":[1,2,3]}`, 3)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got[0] != 1 || got[2] != 3 {
		t.Fatalf("scores = %v", got)
	}
}

func TestParseScoresAcceptsFencedAndEnvelopeless(t *testing.T) {
	if _, err := parseScores("```json\n{\"scores\":[{\"index\":0,\"score\":5}]}\n```", 1); err != nil {
		t.Fatalf("fenced: %v", err)
	}
	if _, err := parseScores(`[{"index":0,"score":5}]`, 1); err != nil {
		t.Fatalf("envelopeless: %v", err)
	}
}

func TestParseScoresIgnoresOutOfRangeIndexes(t *testing.T) {
	// One valid score among nonsense is still worth keeping.
	got, err := parseScores(`{"scores":[{"index":99,"score":10},{"index":1,"score":4}]}`, 2)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got[1] != 4 {
		t.Fatalf("scores = %v", got)
	}
}

func TestParseScoresRejectsUnusableOutput(t *testing.T) {
	for _, bad := range []string{
		`I think passage 2 is best.`,
		`{"scores":[]}`,
		`{"scores":[{"index":50,"score":1}]}`,
	} {
		if _, err := parseScores(bad, 3); err == nil {
			t.Fatalf("expected %q to be rejected", bad)
		}
	}
}

func TestNoopKeepsFusionOrderAndTruncates(t *testing.T) {
	in := []store.Chunk{{ID: 1}, {ID: 2}, {ID: 3}}
	out, _, err := Noop{}.Rerank(context.Background(), "q", in, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || out[0].ID != 1 || out[1].ID != 2 {
		t.Fatalf("out = %+v", out)
	}
	// Asking for more than exists must not panic or pad.
	out, _, _ = Noop{}.Rerank(context.Background(), "q", in, 99)
	if len(out) != 3 {
		t.Fatalf("out = %+v", out)
	}
}

func TestNewSelectsRerankerByMode(t *testing.T) {
	if got := New("off", nil).Name(); got != "none" {
		t.Fatalf("mode off -> %q", got)
	}
	if got := New("", nil).Name(); got != "none" {
		t.Fatalf("empty mode -> %q", got)
	}
}
