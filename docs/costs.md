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
22 CPU-s with Parakeet (143 with Whisper), live denoise 20 CPU-s (under load),
standard live captions ~12 CPU-s.

## Profit margin per minute of audio

Variable costs only: electricity, OpenAI, and Stripe's card fee on the top-up
that paid for the minute (2.9% + $0.30: 4.4% of a $20 top-up, 5.9% of $10).
Not included: your time, the hardware itself, the VPS and domain you already
pay for.

| Product | Price | Electricity | OpenAI | Stripe ($20 top-up) | Net | Margin |
|---|---|---|---|---|---|---|
| Studio clean | $0.0100 | $0.00001 | - | $0.00044 | $0.00955 | 95% |
| Echo removal (with Denoising, a file's second track) | $0.0026 | <$0.00001 | - | $0.00011 | $0.00248 | 95% |
| Transcription (standard, Parakeet/Whisper) | $0.0040 | $0.00001 | - | $0.00018 | $0.00381 | 95% |
| Transcription, extra accurate | $0.0150 | <$0.00001 | $0.0045 | $0.00066 | $0.00984 | 66% |
| Live noise removal | $0.0040 | $0.00001 | - | $0.00018 | $0.00381 | 95% |
| Live captions (standard, English) | $0.0060 | $0.00001 | - | $0.00026 | $0.00573 | 95% |
| Live captions, extra accurate | $0.0250 | <$0.00001 | ≤ $0.017 | $0.0011 | ≥ $0.0069 | ≥ 28% (worst case: talking non-stop; ~50-60% at a typical 50-70% speaking time, since silence isn't sent) |

With $10 top-ups every margin drops by 1.5 points; with $50 top-ups it rises
by 0.9. If Stripe Tax is switched on, subtract another 0.5 points.

Echo removal has its own price, `echo` / `standard` in `server/acw/billing/pricing.py`: 2,600 micros a minute, about 29 % of
Denoising's, the ratio of the measured CPU of the two (0.034 and 0.116 CPU-seconds per second of audio). The per-part editor prices
the same step with the same constant. A job made before the price existed keeps the prices it was made with (nothing for echo).

Electricity per minute of audio (wall power, upper bound): Studio clean
23.5 CPU-s; standard transcription with Parakeet 22 CPU-s on the CPU, or ~2.4
GPU-s at ~90 W on the GTX 960; live denoise ~20 CPU-s (standard) / ~48 (low
latency); standard live captions ~12 CPU-s.

## Free products

| Free thing | What it costs us |
|---|---|
| Browser denoise | Nothing per use (runs on the visitor's device). The model and its WebAssembly (about 22 MB) come from Hugging Face; 15 kB of VPS traffic per new visitor. |
| Free subtitles | Made in the visitor's browser: no compute here. Each new browser downloads the 77 MB model once from Hugging Face and the 14 MB speech runtime (about 2.5 MB compressed) from the VPS. |
| Desktop app | Nothing (runs offline on the user's computer). |

## Fixed and risk items

- **GPU idle draw: ~$1.30/month** (12 W around the clock). It now runs
  transcription jobs (acw-worker-gpu) whenever it is free.
- **Refunds:** Stripe keeps its fee, so refunds now keep it too (deducted from
  the first refund of a payment), so a refund costs us nothing.
- **Disputes:** $15 each. Prepaid credit, 3-D Secure and the $5 minimum keep
  them rare (docs/payments.md).
- **Abuse of the free tier** (many accounts) costs queue time, not money.
  Paid jobs are taken first, and free jobs wait in line.
- **OpenAI price changes:** re-check docs/costs.md when OpenAI changes prices;
  `server/acw/billing/pricing.py` is the one place to change ours.
