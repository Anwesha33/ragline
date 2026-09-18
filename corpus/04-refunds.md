# Refunds

A refund returns money to the customer's original payment method. Refunds are
separate objects with their own lifecycle; refunding does not move the original
payment backwards through its state machine.

## Eligibility window

A payment can be refunded for **180 days** after it succeeded. After 180 days
the original authorisation record is no longer available at most acquirers and
the refund must be handled as a bank transfer, which is a manual process
initiated through support.

Partial refunds are allowed, and a payment may be refunded multiple times as
long as the total refunded never exceeds the captured amount. The API rejects an
over-refund with error code `NW-4221`.

## Timing

How long a refund takes to reach the customer depends on the payment method and
is largely outside our control:

| Method | Typical time to customer |
| --- | --- |
| Card (India) | 5–7 business days |
| Card (UK, Singapore) | 3–5 business days |
| UPI | Under 1 hour, occasionally up to 24 hours |
| Net banking | 2–4 business days |
| Wallet | Instant to 2 hours |

The refund object reaches `succeeded` when the acquirer accepts it, which is
usually within minutes. That is **not** when the customer sees the money. The
most common support ticket we receive is a merchant reporting a "stuck" refund
that is in fact succeeded and simply has not cleared the customer's bank yet.

## Failure modes

A refund can fail after being accepted. The usual causes are a closed customer
account, a card reported lost or stolen, and an acquirer rejecting the refund
because the original authorisation has expired. A failed refund emits
`refund.failed` and the money returns to the merchant balance.

Refunds on a payment that is itself under dispute are blocked with `NW-4093`.
Refunding a disputed payment would mean paying the customer twice, since the
dispute process may already return the funds.

## Balance and funding

Refunds are funded from the merchant's available balance. If the balance is
insufficient, behaviour depends on the merchant's settings:

- `deduct_from_next_settlement` (default) — the refund is accepted and the
  shortfall is deducted from the next payout.
- `reject` — the refund is rejected with `NW-4029` until the balance covers it.

A merchant in `reject` mode with a negative balance cannot refund at all, which
surprises people during a chargeback surge.

## API

```
POST /v1/refunds
Idempotency-Key: <uuid>

{
  "payment_id": "pay_3Kd91mXq",
  "amount": 45000,
  "reason": "requested_by_customer"
}
```

Omit `amount` to refund the full remaining refundable amount. `reason` is one of
`requested_by_customer`, `duplicate`, or `fraudulent`; it is passed to the
acquirer and affects dispute outcomes, so it is worth setting accurately.
