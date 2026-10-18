# Payments: decision and flow

Researched 2026-10-10 (Stripe docs/pricing pages, Paddle, Polar, Lemon Squeezy).
Not legal or tax advice.

## Decision: Stripe directly + our own prepaid wallet

- **Pay-as-you-go without a subscription works on Stripe.** Customers buy credit
  through Checkout (`mode=payment`). With their consent we save the card
  (`setup_future_usage=off_session`). Later "auto-reload" charges are
  off-session PaymentIntents (`off_session=true, confirm=true`). Stripe
  documents this as allowed ("unscheduled top-ups"), as long as the terms say
  when and how much we charge and we record the consent. We do: the
  auto-reload form has an explicit authorization checkbox, and the audit log
  stores the amount, threshold, cap, time and IP.
- **Why not Stripe Billing meters / credit grants?** They bill in arrears
  through a subscription, so a customer can run up usage and then the card
  fails. Live streams need a real-time balance check anyway. We keep our own
  ledger with holds, so we can never be owed money.
- **Merchant of record (Paddle, Polar, Lemon Squeezy, Stripe Managed Payments)**
  would handle global VAT, but:
  - none of them documents saved-card auto-reload for credits;
  - they cost about 5% + $0.50 (Paddle, Polar) or +3.5% (Managed Payments)
    instead of 2.9% + $0.30.
  - Revisit Managed Payments if VAT paperwork becomes the bottleneck. You
    would lose auto-reload but keep manual top-ups.

## Fees (US card: 2.9% + $0.30; +1.5% international)

| Top-up | Fee | Share |
|---|---|---|
| $5 | $0.45 | 8.9% |
| $10 | $0.59 | 5.9% |
| $20 | $0.88 | 4.4% |

The minimum top-up is $10 (`TOPUP_MIN_CENTS`). Below that the fixed fee and
the $15 dispute fee dominate.

## Credit vs debit cards, refunds, chargebacks

- **Both credit and debit cards can be charged back.** The legal basis
  differs (Reg Z vs Reg E) but we receive the same dispute. Stripe charges
  $15 per dispute, plus $15 more if we contest it (refunded only if we win).
- **3D Secure** is left to Stripe and the bank
  (`request_three_d_secure=automatic`): Radar and the issuer ask for it when
  the risk or the law (EU/UK) calls for it. Forcing it on every payment
  (`any`) made cards whose bank offers no challenge fail with "unable to
  authenticate". When 3DS does run, fraud-dispute liability usually moves to
  the card issuer, but it is not guaranteed.
  - Off-session reloads can't use 3DS.
  - EU/UK banks can still demand authentication on a reload
    (`authentication_required`). We then mark the reload failed and the
    customer tops up on-session.
  - Auto-reload switches itself off after 3 failures.
- **Disputes and early fraud warnings are handled automatically**: we suspend
  the account, release its holds and zero the balance.
  - EFWs on charges ≤ $50 are refunded straight away: a refund costs less
    than a dispute and keeps us under Visa's monitoring thresholds.
  - Visa's VAMP program flags small sellers after just a few disputes, so
    keep disputes near zero.
- **Self-serve refunds** of unused credit on payments under 14 days old (EU
  withdrawal rights; builds trust). Dashboard refunds are reflected through
  the `charge.refunded` webhook.
- **Evidence kept for disputes**: sign-up IP, terms acceptance (version +
  time + IP), auto-reload consent, per-job usage records, the 3DS result
  (in Stripe).

## Prepaid-credit legal notes

- Credits are not gift cards and are only spendable on our service; Stripe's
  policy bans stored-value/gift uses.
- The terms state: no cash value, no expiry (simplest: avoids gift-card
  expiry rules), and forfeiture on fraud or chargebacks.
- Some states treat long-unused balances as unclaimed property. Keep
  last-activity dates (we have them), and ask a lawyer before you ever
  expire balances.

## Tax and regions

**Not offered in the EU/EEA, UK or Switzerland.** That avoids VAT
registration (EU non-Union OSS, UK VAT) and EU/UK consumer-law duties, so
**Stripe Tax for the EU/UK is not needed.** Two layers enforce it:

1. **Authorize first, charge after the country check (built in).** Every
   payment (Checkout top-ups and auto-reloads) is created with
   `capture_method=manual`, so the card is only authorized. When the
   authorization arrives (`checkout.session.completed`,
   `payment_intent.amount_capturable_updated`, or the reload's own response),
   `stripe_service._settle_authorized` reads the card's issuing country:
   allowed -> the payment is captured and credited; in
   `BLOCKED_CARD_COUNTRIES` (EU: AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT
   LV LT LU MT NL PL PT RO SK SI ES SE; rest of EEA: IS LI NO; plus GB and CH)
   -> the authorization is cancelled, so the card is never charged and no card
   fee is paid. The payment is marked `refused_region` ("Not charged" on the
   Billing page, with a banner after Checkout), an audit record
   `payment_refused_region` is written and auto-reload is switched off for that
   user. Override the list with `ACW_BLOCKED_CARD_COUNTRIES` (comma-separated
   ISO codes; `none` turns the guard off). An authorization nobody captures
   (every attempt failed) expires after about 7 days without a charge.
2. **Backstop.** A blocked-country payment that was charged anyway (captured
   before this existed) is refunded in full when it would be credited.

No Radar rule is needed (custom rules need the paid Radar plan). The Terms,
Privacy page and the landing FAQ say the service is not available in those
regions and that such cards are not charged.

As a US seller of digital services:
- **US states**: economic-nexus thresholds are mostly $100k or 200
  transactions, so there's nothing to do at launch. Check your own state:
  a home server may create physical nexus there.

If you ever want to sell in the EU/UK: turn on Stripe Tax and register for UK VAT
+ non-Union OSS, or use a merchant of record (and give up auto-reload), then
empty `ACW_BLOCKED_CARD_COUNTRIES` and remove the Radar rule.

## Webhooks to configure (Stripe dashboard → Developers → Webhooks)

Endpoint `https://audiocleaner.live/api/stripe/webhook` (the address the site moved to on 2026-10-18; the older
`https://mc-multi-world.space/audio/api/stripe/webhook` is still proxied to the same handler), events:
`checkout.session.completed`, `checkout.session.async_payment_succeeded`,
`payment_intent.amount_capturable_updated`, `payment_intent.succeeded`, `payment_intent.payment_failed`, `charge.refunded`,
`charge.dispute.created`, `radar.early_fraud_warning.created`,
`payment_method.detached`.

Use a restricted API key: Write for Checkout Sessions, Customers, Payment
Intents, Charges and Refunds, Customer Portal; Read for Balance (the card fee
kept on refunds); everything else None (add Tax Calculations, Transactions:
Write only if Stripe Tax is switched on).

Live setup done 2026-10-15: webhook `we_1UQfwPRxGAWCymT90oQHeB12` (9 events),
portal configuration `bpc_1UQfwbRxGAWCymT93anjWxhJ` (the account default). That live webhook was created with the old
address: add an endpoint at the new one (same events), copy its signing secret into `acw.env`, and delete the old
endpoint once no events arrive there (docs/operations.md, "Domain").

### With the Stripe CLI (test mode; add `--live` for live mode)

Webhook (the response shows `secret` once: that is `STRIPE_WEBHOOK_SECRET`;
`api_version` matches the stripe-python version in requirements.lock):

```
stripe post /v1/webhook_endpoints -d url=https://audiocleaner.live/api/stripe/webhook -d api_version=2026-09-30.endive -d "description=Audio Cleaner" -d "enabled_events[]=checkout.session.completed" -d "enabled_events[]=checkout.session.async_payment_succeeded" -d "enabled_events[]=payment_intent.amount_capturable_updated" -d "enabled_events[]=payment_intent.succeeded" -d "enabled_events[]=payment_intent.payment_failed" -d "enabled_events[]=charge.refunded" -d "enabled_events[]=charge.dispute.created" -d "enabled_events[]=radar.early_fraud_warning.created" -d "enabled_events[]=payment_method.detached"
```

Customer portal (only "update the saved card"; its `id` is `STRIPE_PORTAL_CONFIGURATION`,
otherwise the portal settings must be saved once in the dashboard):

```
stripe post /v1/billing_portal/configurations -d "features[payment_method_update][enabled]=true" -d "business_profile[terms_of_service_url]=https://audiocleaner.live/terms" -d "business_profile[privacy_policy_url]=https://audiocleaner.live/privacy"
```

Not available through the API (dashboard only): restricted API keys, the
account's public details (support email, statement descriptor, Terms/Privacy
URLs shown on Checkout), and the Radar plan (stay on Standard while it is free,
switch to Lite before 2027-01-21, when Standard starts costing $0.05 per payment).
