# Error codes

Every error response from the Northwind Payments API carries a stable code in
the form `NW-nnnn`. The HTTP status tells you how to react at the transport
level; the code tells you what actually happened.

```json
{
  "error": {
    "code": "NW-4012",
    "message": "The card number failed the Luhn check.",
    "type": "invalid_request",
    "param": "card.number",
    "request_id": "req_9Fk22bQx"
  }
}
```

Always log `request_id`. Support cannot trace an issue without it.

## 4xx — the request was wrong

| Code | Status | Meaning | Retry? |
| --- | --- | --- | --- |
| `NW-4001` | 400 | Malformed JSON body | No |
| `NW-4002` | 400 | Missing required field | No |
| `NW-4012` | 400 | Card number failed the Luhn check | No |
| `NW-4013` | 400 | Card expiry date is in the past | No |
| `NW-4014` | 400 | Unsupported currency for this merchant | No |
| `NW-4029` | 400 | Insufficient merchant balance for refund | No |
| `NW-4010` | 401 | Missing or malformed API key | No |
| `NW-4011` | 401 | API key has been revoked | No |
| `NW-4030` | 403 | API key lacks the required scope | No |
| `NW-4031` | 403 | Merchant account is suspended | No |
| `NW-4040` | 404 | Object not found, or belongs to another merchant | No |
| `NW-4090` | 409 | Idempotent request still in flight | Yes, after a delay |
| `NW-4093` | 409 | Payment is under dispute | No |
| `NW-4220` | 422 | Idempotency key reused with a different body | No |
| `NW-4221` | 422 | Refund would exceed the captured amount | No |
| `NW-4290` | 429 | Rate limit exceeded | Yes, honour `Retry-After` |

## 5xx — we failed

| Code | Status | Meaning | Retry? |
| --- | --- | --- | --- |
| `NW-5000` | 500 | Unhandled internal error | Yes, with backoff |
| `NW-5030` | 503 | Dependency unavailable (see `message`) | Yes, with backoff |
| `NW-5031` | 503 | Idempotency store unavailable; write rejected | Yes, with backoff |
| `NW-5040` | 504 | Acquirer did not respond within the timeout | **See below** |

## `NW-5040` deserves its own section

A 504 from the acquirer means we do not know whether the payment succeeded.
Retrying blindly can charge the customer twice.

The correct handling is:

1. Retry the request **with the same idempotency key**. If we completed the
   original, you get the original response back.
2. If the retry also returns `NW-5040`, poll `GET /v1/payments/{id}` with
   backoff for up to 5 minutes.
3. If the payment is still `processing` after 5 minutes, treat it as unresolved
   and do not re-charge. The reconciler will settle it within 24 hours and emit
   a terminal webhook.

## Declines are not errors

An acquirer decline returns HTTP **200** with a payment in state `failed` and a
`failure_code` such as `insufficient_funds`, `do_not_honour`,
`stolen_card` or `authentication_required`. It is not an API error, and
treating a decline as an exception is a common source of incorrect retry
behaviour: a `do_not_honour` will be declined identically on every retry, while
`authentication_required` means the customer must complete 3-D Secure.
