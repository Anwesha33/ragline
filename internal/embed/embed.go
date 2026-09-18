// Package embed wraps Gemini's embedding endpoint.
//
// Two details here are easy to get wrong and expensive to debug:
//
//  1. Task types. gemini-embedding-001 embeds a query and a document into
//     slightly different spaces depending on the declared taskType. Using
//     RETRIEVAL_DOCUMENT for both sides measurably degrades retrieval, and the
//     failure is silent — search still returns results, just worse ones.
//
//  2. Matryoshka truncation. Asking for fewer than the native 3072 dimensions
//     returns a prefix of the full vector that is NOT re-normalised: a
//     1536-dimension response comes back with an L2 norm around 0.69. Cosine
//     distance in pgvector tolerates that, but inner-product distance does not,
//     and neither does any code that assumes unit vectors. This package
//     renormalises, so every vector leaving it has norm 1.
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"
)

const endpoint = "https://generativelanguage.googleapis.com/v1beta"

// TaskType tells the model whether it is embedding a question or a passage.
type TaskType string

const (
	TaskQuery    TaskType = "RETRIEVAL_QUERY"
	TaskDocument TaskType = "RETRIEVAL_DOCUMENT"
)

type Client struct {
	apiKey     string
	model      string
	dim        int
	http       *http.Client
	maxRetries int
}

func New(apiKey, model string, dim int, timeout time.Duration) *Client {
	if timeout == 0 {
		timeout = 60 * time.Second
	}
	return &Client{apiKey: apiKey, model: model, dim: dim,
		http: &http.Client{Timeout: timeout}, maxRetries: 4}
}

func (c *Client) Model() string { return c.model }
func (c *Client) Dim() int      { return c.dim }

type content struct {
	Parts []part `json:"parts"`
}
type part struct {
	Text string `json:"text"`
}

type embedRequest struct {
	Model                string   `json:"model"`
	Content              content  `json:"content"`
	TaskType             TaskType `json:"taskType,omitempty"`
	OutputDimensionality int      `json:"outputDimensionality,omitempty"`
}

type embedResponse struct {
	Embedding struct {
		Values []float32 `json:"values"`
	} `json:"embedding"`
}

type batchRequest struct {
	Requests []embedRequest `json:"requests"`
}

type batchResponse struct {
	Embeddings []struct {
		Values []float32 `json:"values"`
	} `json:"embeddings"`
}

// Embed returns a unit-length vector for one text.
func (c *Client) Embed(ctx context.Context, text string, task TaskType) ([]float32, error) {
	body := embedRequest{
		Model:                "models/" + c.model,
		Content:              content{Parts: []part{{Text: text}}},
		TaskType:             task,
		OutputDimensionality: c.dim,
	}
	raw, err := c.post(ctx, fmt.Sprintf("%s/models/%s:embedContent?key=%s", endpoint, c.model, c.apiKey), body)
	if err != nil {
		return nil, err
	}
	var out embedResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("decode embedding: %w", err)
	}
	if len(out.Embedding.Values) == 0 {
		return nil, fmt.Errorf("embedding response contained no vector")
	}
	return Normalize(out.Embedding.Values), nil
}

// EmbedBatch embeds several texts in one call. The ingestion path lives or dies
// on this: one request per chunk turns a 200-chunk document into 200 round
// trips and exhausts any per-minute request quota long before the token quota.
func (c *Client) EmbedBatch(ctx context.Context, texts []string, task TaskType) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	reqs := make([]embedRequest, 0, len(texts))
	for _, t := range texts {
		reqs = append(reqs, embedRequest{
			Model:                "models/" + c.model,
			Content:              content{Parts: []part{{Text: t}}},
			TaskType:             task,
			OutputDimensionality: c.dim,
		})
	}
	raw, err := c.post(ctx, fmt.Sprintf("%s/models/%s:batchEmbedContents?key=%s", endpoint, c.model, c.apiKey),
		batchRequest{Requests: reqs})
	if err != nil {
		return nil, err
	}
	var out batchResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("decode batch embeddings: %w", err)
	}
	if len(out.Embeddings) != len(texts) {
		// Silently accepting a short response would misalign every vector with
		// its chunk, which is the worst possible corruption here: retrieval
		// keeps working and returns confidently wrong passages.
		return nil, fmt.Errorf("asked for %d embeddings, got %d", len(texts), len(out.Embeddings))
	}
	vectors := make([][]float32, len(out.Embeddings))
	for i, e := range out.Embeddings {
		vectors[i] = Normalize(e.Values)
	}
	return vectors, nil
}

func (c *Client) post(ctx context.Context, url string, body any) ([]byte, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			wait := time.Duration(1<<attempt) * time.Second
			if wait > 15*time.Second {
				wait = 15 * time.Second
			}
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(wait):
			}
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := c.http.Do(req)
		if err != nil {
			lastErr = c.redact(err)
			continue
		}
		raw, readErr := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
		resp.Body.Close()
		if readErr != nil {
			lastErr = readErr
			continue
		}
		if resp.StatusCode == http.StatusOK {
			return raw, nil
		}
		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("embedding api %d: %s", resp.StatusCode, truncate(string(raw), 200))
			continue
		}
		return nil, fmt.Errorf("embedding api %d: %s", resp.StatusCode, truncate(string(raw), 400))
	}
	return nil, fmt.Errorf("embedding request failed after %d attempts: %w", c.maxRetries+1, lastErr)
}

// Normalize scales a vector to unit length. A zero vector is returned
// unchanged rather than producing NaNs.
func Normalize(v []float32) []float32 {
	var sum float64
	for _, f := range v {
		sum += float64(f) * float64(f)
	}
	norm := math.Sqrt(sum)
	if norm == 0 {
		return v
	}
	out := make([]float32, len(v))
	for i, f := range v {
		out[i] = float32(float64(f) / norm)
	}
	return out
}

// Cosine is used by the evaluation harness, not by the query path — Postgres
// does the real similarity search.
func Cosine(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// redact strips the API key out of an error. net/http embeds the full request
// URL in transport errors, and this API passes the key as a query parameter, so
// an unredacted error logged anywhere leaks the credential.
func (c *Client) redact(err error) error {
	if err == nil || c.apiKey == "" {
		return err
	}
	msg := err.Error()
	if !strings.Contains(msg, c.apiKey) {
		return err
	}
	return errors.New(strings.ReplaceAll(msg, c.apiKey, "REDACTED"))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
