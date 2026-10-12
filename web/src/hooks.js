import { useEffect, useState } from 'react';
import { api } from './api.js';

export function usePageTitle(title) {
  useEffect(() => {
    document.title = title ? `${title} · Audio Cleaner` : 'Audio Cleaner';
  }, [title]);
}

let pricingPromise = null;
let pricingCache = null;

/** Public price list (cached for the lifetime of the page). */
export function usePricing() {
  const [state, setState] = useState({ data: pricingCache, error: null, loading: !pricingCache });
  useEffect(() => {
    if (pricingCache) return undefined;
    let alive = true;
    if (!pricingPromise) pricingPromise = api('billing/pricing', { redirectOn401: false });
    pricingPromise
      .then((d) => {
        pricingCache = d;
        if (alive) setState({ data: d, error: null, loading: false });
      })
      .catch((e) => {
        pricingPromise = null; // allow retry on next mount
        if (alive) setState({ data: null, error: e.detail || 'Could not load prices.', loading: false });
      });
    return () => { alive = false; };
  }, []);
  return state;
}

export function findRate(pricing, product, tier) {
  return pricing?.rates?.find((r) => r.product === product && r.tier === tier) || null;
}

/** Cost estimate in micros, mirrors server/acw/billing/pricing.py (per started second, $0.01 minimum). */
export function estimateMicros(pricing, { kind, denoiseTier = 'standard', transcribeTier = 'standard', seconds }) {
  if (!pricing || !seconds) return null;
  const secs = Math.max(0, Math.ceil(seconds));
  const cost = (product, tier) => {
    const r = findRate(pricing, product, tier);
    if (!r) return 0;
    return Math.ceil((Math.round(r.usdPerMinute * 1e6) * secs) / 60);
  };
  let total = 0;
  if (kind === 'denoise' || kind === 'denoise_transcribe') total += cost('denoise', denoiseTier);
  if (kind === 'transcribe' || kind === 'denoise_transcribe') total += cost('transcribe', transcribeTier === 'premium' ? 'premium' : 'local');
  return total ? Math.max(total, 10000) : 0;
}

/** Re-render every `ms` (for countdowns). */
export function useNow(ms = 1000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(id);
  }, [ms]);
  return now;
}
