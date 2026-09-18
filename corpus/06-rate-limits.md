# Rate limits

`edge-gateway` enforces rate limits per merchant using a token bucket. Limits
depend on the merchant's plan and on the endpoint class.

## Limits by plan

| Plan | Sustained rate | Burst capacity |
| --- | --- | --- |
| Sandbox | 10 requests/second | 20 |
| Standard | 50 requests/second | 100 |
| Growth | 200 requests/second | 400 |
| Enterprise | Negotiated, 1,000/second default | 2,000 |

The bucket refills continuously at the sustained rate. A merchant on Standard
who has been idle can therefore send 100 requests immediately, then settles to
50 per second.

Read endpoints (`GET`) draw from a separate bucket with **four times** the
sustained rate, because a dashboard refresh should never consume the budget a
checkout needs.

## Exceeding the limit

An over-limit request returns `429` with code `NW-4290` and these headers:

```
Retry-After: 2
X-RateLimit-Limit: 50
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1726704120
```

`Retry-After` is in seconds and is always present on a 429. Honour it. Clients
that ignore it and retry immediately make the situation worse for themselves,
since the retries also consume tokens.

Our SDKs retry 429s automatically with jittered exponential backoff, up to 3
attempts. The jitter matters: without it, every client that was throttled at the
same instant retries at the same instant.

## Why a token bucket

A fixed window lets a client spend its entire allowance in the first moment of
the window and then hammer the boundary, which produces a thundering herd once
per window and a load pattern that looks nothing like the configured limit. A
token bucket smooths the sustained rate while still permitting a genuine burst,
which is what real checkout traffic looks like.

The buckets live in Redis and are evaluated by a Lua script so that the
read-modify-write is atomic across gateway instances without a lock.

## Limits that are not rate limits

Two other limits are commonly confused with rate limiting:

- **Concurrent 3-D Secure sessions** are capped at 200 per merchant. Exceeding
  the cap returns `NW-4290` as well, but with the message "concurrent
  authentication limit". Raising the request rate limit does not help.
- **Webhook delivery** is not rate limited by plan. The dispatcher keeps one
  in-flight delivery per endpoint regardless of plan, so a merchant cannot
  increase webhook throughput by upgrading.

## Requesting an increase

Limit increases are handled by support and usually take two business days.
Include the endpoint, the rate you need and the traffic pattern — a sustained
increase and a burst increase are different changes and a burst increase is
almost always cheaper to grant.
