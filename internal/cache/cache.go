// Package cache is ragline's Redis layer. It does three jobs: rate limiting,
// caching whole answers, and caching query embeddings.
package cache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
	"time"
	"unicode"

	"github.com/redis/go-redis/v9"
)

type Cache struct{ rdb *redis.Client }

func New(addr, password string, db int) *Cache {
	return &Cache{rdb: redis.NewClient(&redis.Options{
		Addr: addr, Password: password, DB: db,
		DialTimeout: 3 * time.Second, ReadTimeout: 2 * time.Second, WriteTimeout: 2 * time.Second,
	})}
}

func (c *Cache) Ping(ctx context.Context) error { return c.rdb.Ping(ctx).Err() }
func (c *Cache) Close() error                   { return c.rdb.Close() }

var ErrMiss = errors.New("cache miss")

// ---------------------------------------------------------- rate limiting

// tokenBucketScript refills a bucket lazily and takes one token.
//
// A token bucket rather than a fixed window: a fixed window lets a client spend
// its whole minute's budget in the first second and then hammer the boundary,
// which for an LLM backend means a thundering herd every 60 seconds. The bucket
// smooths that out and still allows a genuine burst up to its capacity.
//
// The whole read-modify-write runs inside one Lua script so it is atomic
// against concurrent requests from the same client without a distributed lock.
//
// KEYS[1] bucket key
// ARGV[1] capacity, ARGV[2] refill tokens per second,
// ARGV[3] now (unix seconds, float), ARGV[4] requested tokens
var tokenBucketScript = redis.NewScript(`
local capacity   = tonumber(ARGV[1])
local rate       = tonumber(ARGV[2])
local now        = tonumber(ARGV[3])
local requested  = tonumber(ARGV[4])

local state = redis.call("HMGET", KEYS[1], "tokens", "ts")
local tokens = tonumber(state[1])
local ts     = tonumber(state[2])

if tokens == nil then
	tokens = capacity
	ts = now
end

-- Refill for the elapsed time, capped at capacity.
local delta = math.max(0, now - ts)
tokens = math.min(capacity, tokens + delta * rate)

local allowed = 0
if tokens >= requested then
	tokens = tokens - requested
	allowed = 1
end

redis.call("HMSET", KEYS[1], "tokens", tokens, "ts", now)
-- Expire an idle bucket rather than keeping a key per client forever. Two full
-- refill periods is long enough that an expiry can never grant extra budget.
redis.call("EXPIRE", KEYS[1], math.ceil(capacity / rate * 2) + 1)

local retry_after = 0
if allowed == 0 and rate > 0 then
	retry_after = (requested - tokens) / rate
end
return {allowed, tostring(tokens), tostring(retry_after)}
`)

// Allow takes one token from the client's bucket. It fails open: if Redis is
// unreachable the request proceeds, because losing the cache should degrade the
// service, not take it down.
func (c *Cache) Allow(ctx context.Context, clientID string, perMinute, burst int) (ok bool, retryAfter time.Duration, err error) {
	if perMinute <= 0 {
		return true, 0, nil
	}
	if burst <= 0 {
		burst = perMinute
	}
	rate := float64(perMinute) / 60.0
	now := float64(time.Now().UnixNano()) / 1e9

	res, err := tokenBucketScript.Run(ctx, c.rdb, []string{"rl:" + clientID},
		burst, rate, now, 1).Slice()
	if err != nil {
		return true, 0, err // fail open
	}
	allowed, _ := res[0].(int64)
	var wait time.Duration
	if s, isStr := res[2].(string); isStr {
		if secs, perr := parseFloat(s); perr == nil && secs > 0 {
			wait = time.Duration(secs * float64(time.Second))
		}
	}
	return allowed == 1, wait, nil
}

// ---------------------------------------------------------- answer caching

// CachedAnswer is a complete previous answer, replayed for a repeat question.
type CachedAnswer struct {
	Answer    string    `json:"answer"`
	ChunkIDs  []int64   `json:"chunk_ids"`
	Citations []int64   `json:"citations"`
	Model     string    `json:"model"`
	CachedAt  time.Time `json:"cached_at"`
}

// AnswerKey derives the cache key.
//
// corpusVersion is part of the key on purpose: an answer is only valid for the
// corpus it was grounded in, so ingesting a document must invalidate every
// cached answer. Deriving the key from the corpus state does that without a
// separate invalidation pass.
//
// Only the question and corpus are hashed, never a conversation id — an answer
// that depended on conversation history is not cached at all (see chat.go),
// because "what about the second one?" means something different in every
// conversation.
func AnswerKey(question, corpusVersion string) string {
	h := sha256.New()
	h.Write([]byte(normalizeQuestion(question)))
	h.Write([]byte{0})
	h.Write([]byte(corpusVersion))
	return "ans:" + hex.EncodeToString(h.Sum(nil))[:32]
}

// normalizeQuestion collapses the differences that should not produce a cache
// miss: case, surrounding whitespace, repeated spaces and trailing punctuation.
// It deliberately stops there — this is exact-match caching after tidying, not
// semantic caching, which would need an embedding lookup and can serve the
// answer to a subtly different question.
func normalizeQuestion(q string) string {
	q = strings.ToLower(strings.TrimSpace(q))
	q = strings.TrimRightFunc(q, func(r rune) bool {
		return unicode.IsPunct(r) || unicode.IsSpace(r)
	})
	return strings.Join(strings.Fields(q), " ")
}

func (c *Cache) GetAnswer(ctx context.Context, key string) (*CachedAnswer, error) {
	raw, err := c.rdb.Get(ctx, key).Result()
	if errors.Is(err, redis.Nil) {
		return nil, ErrMiss
	}
	if err != nil {
		return nil, err
	}
	var a CachedAnswer
	if err := json.Unmarshal([]byte(raw), &a); err != nil {
		return nil, ErrMiss
	}
	return &a, nil
}

func (c *Cache) PutAnswer(ctx context.Context, key string, a CachedAnswer, ttl time.Duration) error {
	raw, err := json.Marshal(a)
	if err != nil {
		return err
	}
	return c.rdb.Set(ctx, key, raw, ttl).Err()
}

// ------------------------------------------------------- embedding caching

// EmbeddingKey keys a query embedding by model and dimension as well as text,
// so switching either does not serve vectors from the wrong space.
func EmbeddingKey(text, model string, dim int) string {
	h := sha256.New()
	h.Write([]byte(normalizeQuestion(text)))
	h.Write([]byte{0})
	h.Write([]byte(model))
	h.Write([]byte{0})
	h.Write([]byte(itoa(dim)))
	return "emb:" + hex.EncodeToString(h.Sum(nil))[:32]
}

func (c *Cache) GetEmbedding(ctx context.Context, key string) ([]float32, error) {
	raw, err := c.rdb.Get(ctx, key).Result()
	if errors.Is(err, redis.Nil) {
		return nil, ErrMiss
	}
	if err != nil {
		return nil, err
	}
	var v []float32
	if err := json.Unmarshal([]byte(raw), &v); err != nil {
		return nil, ErrMiss
	}
	return v, nil
}

func (c *Cache) PutEmbedding(ctx context.Context, key string, v []float32, ttl time.Duration) error {
	raw, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return c.rdb.Set(ctx, key, raw, ttl).Err()
}

// InvalidateAnswers is the blunt instrument kept for operational use. Normal
// invalidation happens through the corpus version in the key; this exists for
// "we changed the prompt, drop everything".
//
// SCAN rather than KEYS: KEYS blocks the whole server for the duration of the
// scan, which on a shared Redis is an outage.
func (c *Cache) InvalidateAnswers(ctx context.Context) (int, error) {
	var cursor uint64
	deleted := 0
	for {
		keys, next, err := c.rdb.Scan(ctx, cursor, "ans:*", 200).Result()
		if err != nil {
			return deleted, err
		}
		if len(keys) > 0 {
			n, err := c.rdb.Del(ctx, keys...).Result()
			if err != nil {
				return deleted, err
			}
			deleted += int(n)
		}
		cursor = next
		if cursor == 0 {
			return deleted, nil
		}
	}
}
