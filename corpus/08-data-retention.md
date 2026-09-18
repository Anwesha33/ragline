# Data retention and PII

This document sets out how long Northwind Payments keeps each class of data and
what is done to personally identifiable information.

## Retention periods

| Data | Retention | Driven by |
| --- | --- | --- |
| Ledger entries | 10 years | Statutory accounting requirements |
| Payment and refund records | 7 years | Financial regulation |
| Event log (ScyllaDB) | 90 days | Operational need |
| Webhook delivery attempts | 30 days | Operational need |
| API request logs | 30 days | Operational need |
| Application logs | 14 days | Cost |
| Idempotency keys | 24 hours | Functional |
| Rate-limit buckets | Until refilled | Functional |

Ledger entries are never deleted, only archived to cold storage after 18 months.
Archived entries remain queryable through the reporting service with a latency
of minutes rather than milliseconds.

## Card data

We are PCI DSS Level 1 certified. Full card numbers ("PANs") exist only inside
the card vault, which is a separate service on a separate network segment with
its own access control. No other service ever receives a PAN.

Everything outside the vault works with a **token**, which is a reference of the
form `tok_` followed by 24 characters. Tokens are per-merchant: the same card
tokenised for two merchants produces two different tokens, so merchants cannot
correlate customers with each other.

Logs are scrubbed of anything resembling a card number before they leave the
application. The scrubber uses a Luhn check on any run of 13–19 digits, so it
catches PANs while leaving order numbers and phone numbers alone.

CVV is **never stored**, not even in the vault. It exists only in memory for the
duration of the authorisation request.

## Customer PII

Names, email addresses, phone numbers and billing addresses are stored encrypted
at rest with per-merchant data keys, wrapped by a master key in the HSM. A
merchant's data can therefore be rendered unreadable by destroying a single
key — which is how deletion requests are honoured.

## Deletion requests

Under DPDP (India) and UK GDPR, a customer may request erasure. Our obligation
is bounded by the statutory retention above: we cannot delete ledger entries or
payment records inside their retention period, and the regulation does not
require it.

What happens on an erasure request:

1. Contact details and address are redacted from the payment record.
2. The customer's per-customer encryption key is destroyed, rendering any
   remaining encrypted field unrecoverable.
3. Tokens for that customer are revoked, so the stored card can no longer be
   charged.
4. The ledger entries remain, carrying amounts and timestamps but no identity.

Requests are fulfilled within **30 days**, and the merchant receives a
confirmation record with a reference id.

## Access

Production data access requires a ticket, an approver who is not the requester,
and is logged. Access grants expire after 8 hours. Bulk export of customer data
requires approval from the data protection officer regardless of ticket status.
