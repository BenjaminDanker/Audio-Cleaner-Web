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
- **3D Secure** is requested on every Checkout payment
  (`request_three_d_secure=any`). That usually shifts fraud-dispute liability
  to the card issuer, but it is not guaranteed.
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

## Tax: the decision that is yours

As a US seller of digital services:
- **EU VAT** is due from the first B2C sale. Register for non-Union OSS,
  which Stripe Tax can file.
- **UK VAT**: register within 30 days of the first UK consumer sale.
- **US states**: economic-nexus thresholds are mostly $100k or 200
  transactions, so there's nothing to do at launch. Check your own state:
  a home server may create physical nexus there.

Options:
1. Turn on Stripe Tax and register for UK VAT + non-Union OSS before selling there.
2. Block EU/UK cards until then (a Radar rule on `:card_country:`).
3. Use Stripe Managed Payments (merchant of record) and give up auto-reload.

## Webhooks to configure (Stripe dashboard → Developers → Webhooks)

Endpoint `https://mc-multi-world.space/audio/api/stripe/webhook`, events:
`checkout.session.completed`, `checkout.session.async_payment_succeeded`,
`payment_intent.succeeded`, `payment_intent.payment_failed`, `charge.refunded`,
`charge.dispute.created`, `radar.early_fraud_warning.created`,
`payment_method.detached`.

Use a restricted API key with write access to: Customers, Checkout Sessions,
PaymentIntents, Refunds, Customer portal sessions; read access to Payment
methods, Charges and Events.
