# Webhooks

Northwind Payments delivers events to merchant HTTPS endpoints through the
`webhook-dispatcher` service. This document covers delivery guarantees, the
retry schedule, signature verification and the failure modes worth knowing.

## Delivery guarantees

Delivery is **at-least-once**. A merchant endpoint will occasionally receive the
same event twice, and must deduplicate on the event `id`, which is stable across
redeliveries. Events for one payment are delivered in order under normal
operation, but ordering is **not** guaranteed across a retry: a redelivered
`payment.succeeded` can arrive after a `refund.created` for the same payment.
Consumers must treat the event body as the truth about that event, not as the
current state of the object — call the API if the current state matters.

## Retry schedule

A delivery is considered successful when the endpoint returns any 2xx status
within the timeout. Any other status, or a timeout, triggers a retry.

| Attempt | Delay after previous attempt |
| --- | --- |
| 1 | immediate |
| 2 | 30 seconds |
| 3 | 2 minutes |
| 4 | 10 minutes |
| 5 | 1 hour |
| 6 | 6 hours |
| 7 | 24 hours |

After the seventh attempt the event is marked `undeliverable` and moved to the
dead-letter topic `webhooks.dlq`. The total delivery window is therefore
slightly over **31 hours**.

An endpoint that fails 20 consecutive deliveries is automatically **disabled**,
and the merchant is emailed. A disabled endpoint stops receiving new events
entirely; re-enabling it from the dashboard resumes delivery of events emitted
from that moment onward. Events that were emitted while the endpoint was
disabled are not replayed automatically — use the events API to backfill.

The per-request timeout is **10 seconds**. An endpoint that needs longer should
acknowledge immediately and do its work asynchronously; a slow endpoint is the
most common cause of a merchant falling behind on events.

## Signature verification

Every delivery carries an `Northwind-Signature` header:

```
Northwind-Signature: t=1726704000,v1=5257a869e7ecebeda32affa62cdca3fa51cad7e77a0e56ff536d0ce8e108d8bd
```

Verify it by computing `HMAC-SHA256` over the string `{timestamp}.{raw_body}`
using the endpoint's signing secret, then comparing to `v1` in constant time.
Two rules matter:

- Use the **raw request body**, before any JSON parsing or re-serialisation.
  Re-serialising changes key order and whitespace and the signature will never
  match.
- Reject deliveries whose timestamp is more than **5 minutes** old. Without
  that check, a captured delivery can be replayed indefinitely.

Signing secrets can be rotated from the dashboard. During rotation both the old
and the new secret are valid for **24 hours**, so a deployment does not have to
be instantaneous.

## Event types

The events most merchants consume are `payment.succeeded`, `payment.failed`,
`refund.created`, `refund.succeeded`, `refund.failed`, `dispute.created` and
`payout.paid`. The full list is in the API reference.

Event payloads are additive: we add fields without notice and never remove or
repurpose one. Parsers must ignore unknown fields.

## Operational notes

`webhook-dispatcher` consumes `webhooks.outbound` and keeps one in-flight
delivery per endpoint, so a slow merchant endpoint slows only that merchant.
Queue depth per endpoint is exported as `webhook_queue_depth{endpoint_id}`; the
alert threshold is 5,000 events.
