package api

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestFirstHeading(t *testing.T) {
	cases := map[string]string{
		"# Webhooks\n\nbody":       "Webhooks",
		"\n\n## Retry schedule\nx": "Retry schedule",
		"plain first line\nsecond": "plain first line",
		"":                         "untitled",
		"   \n\n  ":                "untitled",
	}
	for in, want := range cases {
		if got := firstHeading(in); got != want {
			t.Fatalf("firstHeading(%q) = %q, want %q", in, got, want)
		}
	}
	// A very long first line becomes a title, not a paragraph.
	long := firstHeading(strings.Repeat("a", 200))
	if len(long) != 80 {
		t.Fatalf("long title was not truncated: %d chars", len(long))
	}
}

func TestClientIDPrefersExplicitHeader(t *testing.T) {
	s := &Server{}
	r := httptest.NewRequest("POST", "/v1/chat", nil)
	r.RemoteAddr = "10.0.0.5:4444"
	if got := s.clientID(r); got != "10.0.0.5" {
		t.Fatalf("bare request -> %q, want the remote host", got)
	}

	// Behind a proxy the first hop is the closest stand-in for a principal.
	r.Header.Set("X-Forwarded-For", "203.0.113.9, 10.0.0.1")
	if got := s.clientID(r); got != "203.0.113.9" {
		t.Fatalf("forwarded -> %q", got)
	}

	// An explicit client id wins, because it is the only one that survives a
	// NAT sitting in front of many merchants.
	r.Header.Set("X-Client-ID", "merchant_42")
	if got := s.clientID(r); got != "merchant_42" {
		t.Fatalf("explicit -> %q", got)
	}
}

// The logging middleware wraps the ResponseWriter. If the wrapper does not
// forward Flush, streaming silently stops working — the response arrives in one
// lump at the end, which is the exact failure this project exists to avoid.
func TestStatusRecorderForwardsFlush(t *testing.T) {
	rec := httptest.NewRecorder()
	wrapped := &statusRecorder{ResponseWriter: rec, status: 200}

	var flusher interface{ Flush() } = wrapped
	flusher.Flush()

	if !rec.Flushed {
		t.Fatal("Flush was not forwarded to the underlying writer")
	}
}

func TestStatusRecorderCapturesStatus(t *testing.T) {
	rec := httptest.NewRecorder()
	wrapped := &statusRecorder{ResponseWriter: rec, status: 200}
	wrapped.WriteHeader(503)
	if wrapped.status != 503 || rec.Code != 503 {
		t.Fatalf("status = %d / %d", wrapped.status, rec.Code)
	}
}
