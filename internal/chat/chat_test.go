package chat

import (
	"strings"
	"testing"

	"github.com/Anwesha33/ragline/internal/llm"
	"github.com/Anwesha33/ragline/internal/store"
)

func TestParseCitationsSeparatesValidFromInvented(t *testing.T) {
	used, unknown := parseCitations(
		"Keys last 24 hours [2]. Refunds take 180 days [1][2]. Also see [9].", 3)
	if len(used) != 2 || used[0] != 1 || used[1] != 2 {
		t.Fatalf("used = %v, want [1 2] deduplicated and sorted", used)
	}
	if len(unknown) != 1 || unknown[0] != 9 {
		t.Fatalf("unknown = %v, want [9]", unknown)
	}
}

// Models cite as [1], [1][2] and [1, 2]. Missing the grouped form made a
// correctly cited answer look uncited in evaluation.
func TestParseCitationsAcceptsGroupedMarkers(t *testing.T) {
	used, unknown := parseCitations("It exceeds the captured amount [1, 2]. Also [3 , 4].", 4)
	if len(used) != 4 {
		t.Fatalf("used = %v, want all four markers", used)
	}
	if len(unknown) != 0 {
		t.Fatalf("unknown = %v", unknown)
	}
}

// A group containing one invented source must keep its valid members.
func TestStripUnknownMarkersRewritesGroups(t *testing.T) {
	got := stripUnknownMarkers("Supported [1, 9] and [8].", []int{8, 9})
	if !strings.Contains(got, "[1]") {
		t.Fatalf("valid member of the group was lost: %q", got)
	}
	if strings.Contains(got, "9") || strings.Contains(got, "8") {
		t.Fatalf("invented sources survived: %q", got)
	}
}

func TestParseCitationsIgnoresNonCitationBrackets(t *testing.T) {
	// A three-digit number is not a source marker; neither is prose in
	// brackets. Treating them as citations would either invent sources or
	// strip legitimate text out of the answer.
	used, unknown := parseCitations("See RFC [1234] and the note [see below] and [1].", 2)
	if len(used) != 1 || used[0] != 1 {
		t.Fatalf("used = %v, want [1]", used)
	}
	if len(unknown) != 0 {
		t.Fatalf("unknown = %v, want none", unknown)
	}
}

// An answer citing a source that was never supplied is the clearest
// hallucination signal available, and rendering it would show the reader a
// reference they cannot follow.
func TestStripUnknownMarkers(t *testing.T) {
	got := stripUnknownMarkers("Supported [1] but invented [7].", []int{7})
	if strings.Contains(got, "[7]") {
		t.Fatalf("unknown marker survived: %q", got)
	}
	if !strings.Contains(got, "[1]") {
		t.Fatalf("valid marker was removed: %q", got)
	}
}

func TestSnippetCollapsesWhitespaceAndTruncates(t *testing.T) {
	got := snippet("a\n\n  b\tc   d", 100)
	if got != "a b c d" {
		t.Fatalf("snippet = %q", got)
	}
	long := snippet(strings.Repeat("x", 500), 10)
	if len([]rune(long)) != 11 { // 10 characters plus the ellipsis
		t.Fatalf("snippet length = %d for %q", len([]rune(long)), long)
	}
}

func TestBuildContentsPutsQuestionLast(t *testing.T) {
	contents := buildContents(nil, "how long are keys kept?", nil)
	if len(contents) != 1 {
		t.Fatalf("want one content block, got %d", len(contents))
	}
	text := contents[0].Parts[0].Text
	idxSources := strings.Index(text, "Sources:")
	idxQuestion := strings.Index(text, "Question:")
	if idxSources < 0 || idxQuestion < 0 || idxQuestion < idxSources {
		t.Fatalf("the question must come after the sources:\n%s", text)
	}
}

func TestBuildContentsMapsHistoryRoles(t *testing.T) {
	history := []store.Message{
		{Role: "user", Content: "first question"},
		{Role: "assistant", Content: "first answer"},
	}
	contents := buildContents(history, "follow up", nil)
	if len(contents) != 3 {
		t.Fatalf("want 2 history turns plus the new question, got %d", len(contents))
	}
	if contents[0].Role != llm.RoleUser || contents[1].Role != llm.RoleModel {
		t.Fatalf("roles = %q, %q; assistant history must map to the model role",
			contents[0].Role, contents[1].Role)
	}
}
