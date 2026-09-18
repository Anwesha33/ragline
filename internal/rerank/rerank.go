// Package rerank reorders retrieval candidates by actual relevance to the
// question.
//
// Why a reranker exists at all: retrieval optimises for recall. Vector search
// and BM25 both answer "which passages look related", and fusing them widens
// the net further. But the generator can only be given a handful of passages,
// and putting the right one at position 9 of 12 is nearly as bad as not
// retrieving it — models attend unevenly across a long context. The reranker's
// job is to take a high-recall candidate set and produce a high-precision top-5.
//
// This is a listwise LLM reranker: one call scores the whole candidate list
// together. The alternative, a cross-encoder like bge-reranker, is faster and
// cheaper per query but means hosting a model and a GPU. The trade is one extra
// LLM call (~300ms, measured in docs/RESULTS.md) against a second serving
// stack, and for this system the call wins.
package rerank

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/Anwesha33/ragline/internal/llm"
	"github.com/Anwesha33/ragline/internal/store"
)

const systemPrompt = `You score how well each passage answers a question. You are not answering the question.

Score each passage from 0 to 10:
  10  directly and completely answers the question
   7  contains most of the answer, or the specific fact asked for
   4  same topic and useful supporting context, but does not answer it
   1  same general subject area, does not help
   0  irrelevant

Judge only what the passage actually says. A passage that merely repeats the question's wording without answering it scores low. A passage that answers the question in different words scores high.`

// Reranker reorders candidates. The interface exists so the query path can run
// with reranking disabled, which is how docs/RESULTS.md measures what it buys.
type Reranker interface {
	Rerank(ctx context.Context, question string, candidates []store.Chunk, topN int) ([]store.Chunk, Stats, error)
	Name() string
}

// Stats reports what the rerank pass cost and how much it changed.
type Stats struct {
	Duration     time.Duration
	PromptTokens int
	OutputTokens int
	// Reordered counts candidates whose position changed, which is the honest
	// answer to "is the reranker doing anything".
	Reordered int
	Fallback  bool
}

// Noop keeps fusion order. Used as the control arm in evaluation and as the
// automatic fallback when the reranker fails.
type Noop struct{}

func (Noop) Name() string { return "none" }

func (Noop) Rerank(_ context.Context, _ string, candidates []store.Chunk, topN int) ([]store.Chunk, Stats, error) {
	if topN > len(candidates) {
		topN = len(candidates)
	}
	return candidates[:topN], Stats{}, nil
}

// LLMReranker scores candidates with a small, fast model.
type LLMReranker struct {
	Client *llm.Client
	// MaxCandidates bounds the prompt. Beyond roughly 20 passages a listwise
	// reranker's accuracy degrades anyway, and the prompt cost stops being
	// worth it.
	MaxCandidates int
	// MaxPassageChars truncates each passage in the *scoring* prompt only. The
	// full text still reaches the generator; the reranker needs enough to judge
	// relevance, not the whole passage.
	MaxPassageChars int
}

func NewLLM(c *llm.Client) *LLMReranker {
	return &LLMReranker{Client: c, MaxCandidates: 20, MaxPassageChars: 1200}
}

func (r *LLMReranker) Name() string { return "llm:" + r.Client.Model() }

type scoredItem struct {
	Index int     `json:"index"`
	Score float64 `json:"score"`
}

// scoreSchema pins the response shape. Without it the model periodically
// returns a nested array — {"scores":[[{...},{...}]]} — which is valid JSON and
// completely unusable.
func scoreSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"scores": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"index": map[string]any{"type": "integer"},
						"score": map[string]any{"type": "number"},
					},
					"required": []string{"index", "score"},
				},
			},
		},
		"required": []string{"scores"},
	}
}

func (r *LLMReranker) Rerank(ctx context.Context, question string, candidates []store.Chunk, topN int) ([]store.Chunk, Stats, error) {
	start := time.Now()
	stats := Stats{}

	if len(candidates) == 0 {
		return candidates, stats, nil
	}
	// Nothing to reorder: scoring would be pure cost.
	if len(candidates) <= 1 || topN >= len(candidates) {
		return Noop{}.Rerank(ctx, question, candidates, topN)
	}

	pool := candidates
	if r.MaxCandidates > 0 && len(pool) > r.MaxCandidates {
		pool = pool[:r.MaxCandidates]
	}

	prompt := r.buildPrompt(question, pool)
	resp, err := r.Client.Generate(ctx, llm.Request{
		System:   systemPrompt,
		Contents: []llm.Content{{Role: llm.RoleUser, Parts: []llm.Part{{Text: prompt}}}},
		JSON:     true,
		Config:   llm.GenerationConfig{Temperature: 0, ResponseSchema: scoreSchema()},
	})
	if err != nil {
		// Reranking is an optimisation. If it fails, serving the fusion order
		// is a worse answer, not a failed request.
		out, _, _ := Noop{}.Rerank(ctx, question, candidates, topN)
		stats.Duration = time.Since(start)
		stats.Fallback = true
		return out, stats, fmt.Errorf("rerank failed, falling back to fusion order: %w", err)
	}
	stats.PromptTokens = resp.Usage.PromptTokens
	stats.OutputTokens = resp.Usage.OutputTokens

	scores, err := parseScores(resp.Text, len(pool))
	if err != nil {
		out, _, _ := Noop{}.Rerank(ctx, question, candidates, topN)
		stats.Duration = time.Since(start)
		stats.Fallback = true
		return out, stats, fmt.Errorf("rerank returned unusable scores, falling back: %w", err)
	}

	ranked := make([]store.Chunk, len(pool))
	copy(ranked, pool)
	for i := range ranked {
		ranked[i].RerankScore = scores[i]
	}
	// Stable sort so that candidates the model scored identically keep their
	// fusion order rather than being shuffled arbitrarily.
	sort.SliceStable(ranked, func(i, j int) bool {
		return ranked[i].RerankScore > ranked[j].RerankScore
	})

	originalPos := map[int64]int{}
	for i, c := range pool {
		originalPos[c.ID] = i
	}
	for i, c := range ranked {
		if originalPos[c.ID] != i {
			stats.Reordered++
		}
	}

	if topN > len(ranked) {
		topN = len(ranked)
	}
	stats.Duration = time.Since(start)
	return ranked[:topN], stats, nil
}

func (r *LLMReranker) buildPrompt(question string, candidates []store.Chunk) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Question: %s\n\nPassages:\n", question)
	for i, c := range candidates {
		text := c.Content
		if r.MaxPassageChars > 0 && len(text) > r.MaxPassageChars {
			text = text[:r.MaxPassageChars] + "…"
		}
		title := c.DocTitle
		if c.Heading != "" {
			title += " › " + c.Heading
		}
		fmt.Fprintf(&b, "\n[%d] (%s)\n%s\n", i, title, text)
	}
	fmt.Fprintf(&b, "\nReturn JSON: {\"scores\":[{\"index\":0,\"score\":7.5}, ...]} with one entry for every passage index 0 to %d.", len(candidates)-1)
	return b.String()
}

// parseScores maps the model's output onto the candidate list.
//
// It is lenient by design: an omitted index scores 0 and an out-of-range index
// is ignored, because the alternative — failing the whole rerank because the
// model skipped one passage — throws away 19 good scores over 1 missing one.
// parseScores maps the model's output onto the candidate list.
//
// The response schema makes the expected shape overwhelmingly likely, but this
// stays defensive because a reranker that throws away a whole scoring pass over
// a formatting wobble costs a visible drop in answer quality. Three shapes are
// accepted: the intended array of objects, a nested array (which is what a
// schema-less call produced often enough to be worth handling), and a bare
// array of numbers positionally matched to the candidates.
//
// It is also lenient about content: an omitted index scores 0 and an
// out-of-range index is ignored, because failing the pass over one missing
// entry throws away every other good score.
func parseScores(text string, n int) ([]float64, error) {
	raw := text
	if j, ok := llm.ExtractJSON(text); ok {
		raw = j
	}

	var envelope struct {
		Scores json.RawMessage `json:"scores"`
	}
	if err := json.Unmarshal([]byte(raw), &envelope); err != nil || len(envelope.Scores) == 0 {
		// Some responses drop the envelope and return the array directly.
		envelope.Scores = json.RawMessage(raw)
	}

	items, err := flattenScoreItems(envelope.Scores)
	if err != nil {
		return nil, fmt.Errorf("%w (got %.160q)", err, text)
	}
	if len(items) == 0 {
		return nil, fmt.Errorf("no scores in response")
	}

	out := make([]float64, n)
	seen := 0
	for _, s := range items {
		if s.Index < 0 || s.Index >= n {
			continue
		}
		out[s.Index] = s.Score
		seen++
	}
	if seen == 0 {
		return nil, fmt.Errorf("no score referred to a valid passage index")
	}
	return out, nil
}

// flattenScoreItems accepts [{index,score}], [[{index,score}]] and [7,10,...].
func flattenScoreItems(raw json.RawMessage) ([]scoredItem, error) {
	var direct []scoredItem
	if err := json.Unmarshal(raw, &direct); err == nil {
		return direct, nil
	}

	var nested [][]scoredItem
	if err := json.Unmarshal(raw, &nested); err == nil {
		var out []scoredItem
		for _, group := range nested {
			out = append(out, group...)
		}
		return out, nil
	}

	// A bare list of numbers is positional: the nth score belongs to the nth
	// candidate.
	var bare []float64
	if err := json.Unmarshal(raw, &bare); err == nil {
		out := make([]scoredItem, 0, len(bare))
		for i, v := range bare {
			out = append(out, scoredItem{Index: i, Score: v})
		}
		return out, nil
	}

	return nil, errors.New("scores were not an array of {index, score}, a nested array, or an array of numbers")
}

// New builds the reranker named by mode.
func New(mode string, client *llm.Client) Reranker {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", "off", "none":
		return Noop{}
	default:
		return NewLLM(client)
	}
}
