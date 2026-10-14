# Cost audit (2026-10-14)

Question: does any product, free or paid, cost more to run than it brings in?
Short answer: **no**. Electricity is a rounding error everywhere. The real
variable costs are OpenAI (extra-accurate tiers only) and Stripe's fees.

## Inputs

| Item | Value | Source |
|---|---|---|
| Electricity | $0.151 / kWh | Kansas average residential, EIA Electric Power Monthly, May 2026 |
| CPU power | ~11 W per busy logical CPU at the wall (upper bound) | Ryzen 7 1700X: 95 W TDP for 8 cores, ~20 W idle package, ~88% PSU efficiency; no RAPL access without root |
| GPU power | 12 W idle (P8), up to ~120 W under load | `nvidia-smi` (GTX 960). **Not used by production today.** |
| OpenAI `gpt-transcribe` | $0.0045 / min of audio | OpenAI pricing page |
| OpenAI `gpt-live-transcribe` | $0.017 / min of audio sent | OpenAI pricing page (we send speech only, never silence) |
| Stripe card fee | 2.9% + $0.30 per top-up | not refunded by Stripe when we refund |
| Stripe dispute | $15 per dispute | |

CPU per minute of audio comes from our measurements (docs/operations.md,
docs/research/*): Studio clean 23.5 CPU-s (stereo), local transcription
143 CPU-s, live denoise 20 CPU-s (standard, under load) / 48 (low latency),
local live captions ~36 CPU-s.

## Per minute of audio

| Product | Price | Electricity | Other variable cost | Margin |
|---|---|---|---|---|
| Studio clean | $0.0100 | $0.00001 | - | ~100% |
| Transcription (standard) | $0.0040 | $0.00007 | - | 98% |
| Transcription, extra accurate | $0.0150 | <$0.00001 | OpenAI $0.0045 | 70% |
| Live noise removal | $0.0040 | $0.00001 | - | ~100% |
| Live noise removal, low latency | $0.0080 | $0.00002 | - | ~100% |
| Live captions (standard) | $0.0060 | $0.00002 | - | ~100% |
| Live captions, extra accurate | $0.0250 | <$0.00001 | OpenAI ≤ $0.017 | ≥ 32% |

Stripe then takes 5.9% of a $10 top-up (3.5% of $50), so the thinnest
product, extra-accurate live captions, keeps at least ~26% after fees.

## Free products

| Free thing | What it costs us |
|---|---|
| Browser denoise | Nothing per use (runs on the visitor's device). ~12 MB of VPS traffic per new visitor (cached for a year after). |
| Free transcription (60 min/day per account) | ≤ $0.004 of electricity per account per day at full use. The real cost is queue time, so paid jobs go first. |
| Desktop app | Nothing (runs on the user's computer). The update check is a tiny file on the VPS. |

## Fixed and risk items

- **GPU idle draw: ~$1.30/month** (12 W around the clock) for a card production
  doesn't use yet. It would earn its keep once the faster transcription model
  (Parakeet, docs/research/asr.md) runs on it.
- **Refunds:** Stripe keeps its fee, so refunding an unused $10 top-up costs us
  $0.59. The 14-day refund window bounds this. Consider deducting the card fee
  from refunds (state it in the Terms).
- **Disputes:** $15 each. Prepaid credit, 3-D Secure and the $10 minimum keep
  them rare (docs/payments.md).
- **Abuse of the free tier** (many accounts) costs queue time, not money.
  Paid jobs are taken first, and free jobs wait in line.
- **OpenAI price changes:** re-check docs/costs.md when OpenAI changes prices;
  `server/acw/billing/pricing.py` is the one place to change ours.
