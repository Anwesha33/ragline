// Package llm wraps Gemini's generateContent and its server-sent-events
// streaming variant.
//
// Streaming is the point of this package. In a RAG system the user waits for
// retrieval, reranking and then generation; if the answer only appears when
// generation finishes, the perceived latency is the sum of all three. Streaming
// turns that into time-to-first-token, and TTFT is the number the user
// actually feels.
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const endpoint = "https://generativelanguage.googleapis.com/v1beta"

type Client struct {
	apiKey     string
	model      string
	http       *http.Client
	maxRetries int
	// retryBudget caps total time spent retrying one call, so a quota outage
	// fails the request instead of holding a connection open for minutes.
	retryBudget time.Duration
}

func New(apiKey, model string, timeout time.Duration) *Client {
	if timeout == 0 {
		timeout = 120 * time.Second
	}
	return &Client{
		apiKey: apiKey, model: model,
		http:        &http.Client{Timeout: timeout},
		maxRetries:  4,
		retryBudget: 60 * time.Second,
	}
}

func (c *Client) Model() string { return c.model }

type Part struct {
	Text             string `json:"text,omitempty"`
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
	Thought          bool   `json:"thought,omitempty"`
}

type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts"`
}

const (
	RoleUser  = "user"
	RoleModel = "model"
)

type GenerationConfig struct {
	Temperature      float64 `json:"temperature,omitempty"`
	MaxOutputTokens  int     `json:"maxOutputTokens,omitempty"`
	ResponseMIMEType string  `json:"responseMimeType,omitempty"`
	// ResponseSchema constrains the shape of a JSON response. Asking for JSON
	// without a schema gets valid JSON of an unpredictable shape — the
	// reranker was handed {"scores":[[{...}]]} instead of {"scores":[{...}]}
	// often enough to matter. A schema makes the shape the API's problem
	// rather than the parser's.
	ResponseSchema map[string]any `json:"responseSchema,omitempty"`
}

func (c GenerationConfig) isZero() bool {
	return c.Temperature == 0 && c.MaxOutputTokens == 0 &&
		c.ResponseMIMEType == "" && len(c.ResponseSchema) == 0
}

type Usage struct {
	PromptTokens int `json:"promptTokenCount"`
	OutputTokens int `json:"candidatesTokenCount"`
	TotalTokens  int `json:"totalTokenCount"`
}

type Request struct {
	System   string
	Contents []Content
	Config   GenerationConfig
	JSON     bool
}

type Response struct {
	Text         string
	FinishReason string
	Usage        Usage
	Latency      time.Duration
}

type apiRequest struct {
	SystemInstruction *Content          `json:"systemInstruction,omitempty"`
	Contents          []Content         `json:"contents"`
	GenerationConfig  *GenerationConfig `json:"generationConfig,omitempty"`
}

type apiResponse struct {
	Candidates []struct {
		Content      Content `json:"content"`
		FinishReason string  `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata  Usage `json:"usageMetadata"`
	PromptFeedback *struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback,omitempty"`
}

func (r Request) build() apiRequest {
	body := apiRequest{Contents: r.Contents}
	if r.System != "" {
		body.SystemInstruction = &Content{Parts: []Part{{Text: r.System}}}
	}
	cfg := r.Config
	if r.JSON {
		cfg.ResponseMIMEType = "application/json"
	}
	if !cfg.isZero() {
		body.GenerationConfig = &cfg
	}
	return body
}

// Generate is the non-streaming path, used for reranking and evaluation where
// only the final text matters.
func (c *Client) Generate(ctx context.Context, req Request) (*Response, error) {
	if c.apiKey == "" {
		return nil, errors.New("GEMINI_API_KEY is not set")
	}
	payload, err := json.Marshal(req.build())
	if err != nil {
		return nil, err
	}
	url := fmt.Sprintf("%s/models/%s:generateContent?key=%s", endpoint, c.model, c.apiKey)

	start := time.Now()
	raw, err := c.doWithRetry(ctx, url, payload)
	if err != nil {
		return nil, err
	}
	var ar apiResponse
	if err := json.Unmarshal(raw, &ar); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if len(ar.Candidates) == 0 {
		reason := ""
		if ar.PromptFeedback != nil {
			reason = ar.PromptFeedback.BlockReason
		}
		return nil, fmt.Errorf("model returned no candidates (block reason %q)", reason)
	}
	var sb strings.Builder
	for _, p := range ar.Candidates[0].Content.Parts {
		if !p.Thought {
			sb.WriteString(p.Text)
		}
	}
	return &Response{
		Text: sb.String(), FinishReason: ar.Candidates[0].FinishReason,
		Usage: ar.UsageMetadata, Latency: time.Since(start),
	}, nil
}

// StreamEvent is one delta from the streaming endpoint.
type StreamEvent struct {
	Text  string
	Usage Usage
	Done  bool
}

// Stream calls the SSE endpoint and invokes onEvent for each delta.
//
// The callback is invoked synchronously, so a slow consumer applies
// backpressure to the read loop rather than silently buffering the whole
// answer in memory. Returning an error from the callback aborts the stream —
// that is how a disconnected HTTP client stops generation instead of paying for
// tokens nobody will read.
func (c *Client) Stream(ctx context.Context, req Request, onEvent func(StreamEvent) error) (*Response, error) {
	if c.apiKey == "" {
		return nil, errors.New("GEMINI_API_KEY is not set")
	}
	payload, err := json.Marshal(req.build())
	if err != nil {
		return nil, err
	}
	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse&key=%s", endpoint, c.model, c.apiKey)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	// Retry only while establishing the stream. Once a token has been emitted
	// the request is no longer safely repeatable — the caller has already seen
	// part of an answer — so the retry loop deliberately ends at the first
	// successful response header.
	start := time.Now()
	resp, err := c.openStream(ctx, httpReq, payload, url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	out := &Response{}
	var sb strings.Builder
	sc := bufio.NewScanner(resp.Body)
	// A single SSE frame can carry a large chunk of text; the default 64KB
	// scanner buffer is not enough and fails with "token too long" mid-answer.
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	for sc.Scan() {
		line := sc.Text()
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var chunk apiResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			// A malformed frame is not worth failing an otherwise good answer.
			continue
		}
		// Usage arrives cumulatively and is only final on the last frame.
		if chunk.UsageMetadata.TotalTokens > 0 {
			out.Usage = chunk.UsageMetadata
		}
		if len(chunk.Candidates) == 0 {
			continue
		}
		if r := chunk.Candidates[0].FinishReason; r != "" {
			out.FinishReason = r
		}
		for _, p := range chunk.Candidates[0].Content.Parts {
			if p.Thought || p.Text == "" {
				continue
			}
			sb.WriteString(p.Text)
			if err := onEvent(StreamEvent{Text: p.Text}); err != nil {
				out.Text = sb.String()
				out.Latency = time.Since(start)
				return out, err
			}
		}
	}
	if err := sc.Err(); err != nil {
		out.Text = sb.String()
		return out, fmt.Errorf("reading stream: %w", err)
	}

	out.Text = sb.String()
	out.Latency = time.Since(start)
	_ = onEvent(StreamEvent{Usage: out.Usage, Done: true})
	return out, nil
}

// openStream performs the streaming request, retrying transient failures
// before any bytes reach the caller. 503 "high demand" is common enough on
// shared capacity that a single attempt fails roughly half of all requests
// under load, which is what this exists to prevent.
func (c *Client) openStream(ctx context.Context, first *http.Request, payload []byte, url string) (*http.Response, error) {
	var lastErr error
	deadline := time.Now().Add(c.retryBudget)

	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		req := first
		if attempt > 0 {
			if time.Now().After(deadline) {
				return nil, fmt.Errorf("stream retry budget exhausted: %w", lastErr)
			}
			wait := time.Duration(1<<attempt) * time.Second
			if wait > 15*time.Second {
				wait = 15 * time.Second
			}
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(wait):
			}
			// A request body is consumed by the first attempt, so each retry
			// needs a freshly built request rather than a reused one.
			var err error
			req, err = http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
			if err != nil {
				return nil, err
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Accept", "text/event-stream")
		}

		resp, err := c.http.Do(req)
		if err != nil {
			lastErr = c.redact(err)
			continue
		}
		if resp.StatusCode == http.StatusOK {
			return resp, nil
		}

		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		resp.Body.Close()
		apiErr := fmt.Errorf("stream api %d: %s", resp.StatusCode, truncate(string(raw), 400))
		if resp.StatusCode != http.StatusTooManyRequests && resp.StatusCode < 500 {
			return nil, apiErr
		}
		lastErr = apiErr
	}
	return nil, fmt.Errorf("stream failed after %d attempts: %w", c.maxRetries+1, lastErr)
}

func (c *Client) doWithRetry(ctx context.Context, url string, payload []byte) ([]byte, error) {
	var lastErr error
	deadline := time.Now().Add(c.retryBudget)
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			if time.Now().After(deadline) {
				return nil, fmt.Errorf("retry budget exhausted: %w", lastErr)
			}
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
			lastErr = fmt.Errorf("gemini %d: %s", resp.StatusCode, truncate(string(raw), 200))
			continue
		}
		return nil, fmt.Errorf("gemini %d: %s", resp.StatusCode, truncate(string(raw), 400))
	}
	return nil, fmt.Errorf("request failed after %d attempts: %w", c.maxRetries+1, lastErr)
}

// ExtractJSON pulls the first JSON value out of a response that may be wrapped
// in prose or a code fence.
func ExtractJSON(s string) (string, bool) {
	s = strings.TrimSpace(s)
	if fence := strings.Index(s, "```"); fence >= 0 {
		rest := s[fence+3:]
		if nl := strings.IndexByte(rest, '\n'); nl >= 0 {
			rest = rest[nl+1:]
		}
		if end := strings.Index(rest, "```"); end >= 0 {
			s = strings.TrimSpace(rest[:end])
		}
	}
	start := strings.IndexAny(s, "{[")
	if start < 0 {
		return "", false
	}
	open := s[start]
	closeCh := byte('}')
	if open == '[' {
		closeCh = ']'
	}
	depth, inStr, escaped := 0, false, false
	for i := start; i < len(s); i++ {
		ch := s[i]
		switch {
		case escaped:
			escaped = false
		case ch == '\\' && inStr:
			escaped = true
		case ch == '"':
			inStr = !inStr
		case inStr:
		case ch == open:
			depth++
		case ch == closeCh:
			depth--
			if depth == 0 {
				return s[start : i+1], true
			}
		}
	}
	return "", false
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
