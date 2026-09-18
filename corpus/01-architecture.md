# Northwind Payments — Platform Architecture

Northwind Payments processes card and bank-transfer payments for merchants in
India, Singapore and the United Kingdom. This document describes the services
that make up the platform and how a payment moves through them.

## Service map

The platform is eleven services. Six of them sit on the payment path and the
rest are asynchronous.

| Service | Language | Responsibility |
| --- | --- | --- |
| `edge-gateway` | Go | TLS termination, authentication, rate limiting |
| `payment-api` | Go | The public REST API merchants integrate against |
| `ledger` | Java | Double-entry ledger; the system of record for money |
| `router` | Go | Chooses an acquirer for each transaction |
| `acquirer-adapters` | Go | One adapter per acquiring bank |
| `risk-engine` | Python | Synchronous fraud scoring |
| `webhook-dispatcher` | Go | Delivers events to merchant endpoints |
| `settlement` | Java | Nightly settlement and payout files |
| `reconciler` | Java | Matches acquirer reports against the ledger |
| `notification` | Go | Merchant email and SMS |
| `reporting` | Python | Merchant dashboards and exports |

## The payment path

A card payment takes this route:

1. `edge-gateway` authenticates the request and applies the merchant's rate
   limit.
2. `payment-api` validates the request body and writes a `payment_intent` row
   in state `requires_confirmation`.
3. `risk-engine` scores the transaction. Scoring has a hard budget of 400ms;
   if it has not returned by then the payment proceeds with the merchant's
   default risk action, which is `allow` unless the merchant has configured
   `block_on_timeout`.
4. `router` selects an acquirer using the merchant's routing policy, the card
   BIN and the acquirer's current health score.
5. The selected `acquirer-adapter` sends the authorisation request.
6. `ledger` records the authorisation as a pair of entries. Money is never
   represented as a single mutable balance; a balance is always the sum of
   entries.
7. `payment-api` returns the result and `webhook-dispatcher` emits
   `payment.succeeded` or `payment.failed`.

The p99 latency budget for the whole path is 1,800ms. The largest single
contributor is the acquirer call, which is typically 600–900ms and is not under
our control.

## State machine

A payment intent moves through these states:

```
requires_confirmation ──▶ processing ──▶ succeeded
          │                   │
          │                   └────────▶ failed
          └──▶ cancelled
```

`succeeded` and `failed` are terminal for the intent itself. A succeeded
payment can still be refunded, which creates a separate `refund` object rather
than moving the payment backwards. This is deliberate: a state machine that can
run backwards cannot be reasoned about, and refunds have their own lifecycle,
their own failure modes and their own settlement timing.

## Storage

- **Postgres** is the system of record for payments, refunds and the ledger.
  The ledger runs on its own Postgres cluster, physically separated from
  everything else, because its availability requirements are stricter and its
  write pattern is append-only.
- **ScyllaDB** stores the event log that feeds webhooks and reporting.
- **Redis** holds idempotency keys, rate-limit buckets and acquirer health
  scores. Nothing in Redis is a source of truth; everything in it can be
  rebuilt from Postgres or recomputed.
- **Kafka** carries every asynchronous flow. The main topics are
  `payments.events`, `webhooks.outbound`, `settlement.batches` and
  `reconciliation.reports`.

## Multi-region

The platform runs active-active in `ap-south-1` (Mumbai) and `ap-southeast-1`
(Singapore), with the UK served from Singapore. The ledger is the exception: it
is active-passive with Mumbai as primary, because a double-entry ledger with
concurrent writers in two regions requires a consensus protocol we have chosen
not to run. Failover of the ledger is a manual operation with a documented RTO
of 15 minutes and an RPO of zero.

Merchant data residency for India is enforced at `edge-gateway`: requests
carrying an Indian merchant id are never routed out of `ap-south-1`.
