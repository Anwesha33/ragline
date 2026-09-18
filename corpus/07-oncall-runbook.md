# On-call runbook

This runbook covers the alerts that page the payments on-call rotation. It
assumes access to Grafana, the `nwctl` CLI and the `#payments-incident` channel.

## Paging policy

Pages go to the primary on-call. If the primary does not acknowledge within 5
minutes the page escalates to the secondary, and after a further 10 minutes to
the engineering manager. Declare an incident in `#payments-incident` for
anything that affects payment success rate, and do it early — declaring an
incident that turns out to be minor costs nothing.

## `PaymentSuccessRateLow`

Fires when the 10-minute payment success rate drops below 92% for a single
acquirer, or below 96% overall.

1. Open the Acquirer Health dashboard. If one acquirer is degraded, shift
   traffic away from it: `nwctl router set-weight --acquirer <name> --weight 0`.
   The router picks the change up within 30 seconds through ZooKeeper; no deploy
   is needed.
2. If all acquirers are degraded, the cause is usually ours. Check `risk-engine`
   latency — when scoring exceeds its 400ms budget and the merchant is
   configured `block_on_timeout`, payments fail rather than proceeding.
3. Check for a recent deploy. `nwctl deploy history --service payment-api
   --hours 2`.

Rolling back is always acceptable without approval. Investigate afterwards.

## `WebhookQueueDepthHigh`

Fires when `webhook_queue_depth` for any endpoint exceeds 5,000.

Nearly always a single slow or failing merchant endpoint, and nearly always not
an incident. The dispatcher keeps one in-flight delivery per endpoint, so one
merchant cannot affect another.

1. Identify the endpoint from the alert labels.
2. Check its recent response codes and latencies on the Webhooks dashboard.
3. If the endpoint is timing out consistently, it will disable itself after 20
   consecutive failures. That is correct behaviour; let it happen and confirm
   the merchant was emailed.
4. Only escalate if queue depth is high across **many** endpoints, which points
   at the dispatcher itself rather than at any merchant.

## `LedgerReplicationLag`

Fires when the ledger's replica lags the primary by more than 30 seconds.

This one is serious. The ledger is active-passive with Mumbai as primary, and
replication lag is the difference between a 15-minute failover and a failover
that loses data.

1. Do **not** fail over while lag is high. The documented RPO is zero and
   failing over with lag breaks that promise.
2. Check for a long-running transaction on the primary:
   `nwctl ledger long-transactions`.
3. Page the database on-call if lag exceeds 5 minutes or is still climbing after
   15 minutes.

## `ReconciliationMismatch`

Fires when the nightly reconciler finds transactions in the acquirer report that
do not match the ledger.

Never page-worthy at night. Acknowledge and handle it during business hours
unless the mismatch value exceeds ₹50 lakh, in which case escalate to the
finance on-call as well. Most mismatches are timing differences that resolve on
the following day's run.

## Things that look like incidents and are not

- **A spike in `NW-4290`.** A merchant hitting their rate limit is the system
  working. Check whether it is a single merchant before doing anything.
- **A spike in declines.** Declines are HTTP 200 with a `failure_code`. A rise
  in `insufficient_funds` around the end of the month is seasonal.
- **Refunds "stuck" in succeeded.** A succeeded refund has been accepted by the
  acquirer; the customer's bank takes days. This is the most common false
  report.
