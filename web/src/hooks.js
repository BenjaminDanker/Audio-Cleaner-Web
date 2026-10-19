import { useEffect, useState, useSyncExternalStore } from 'react';
import { api } from './api.js';
import { SITE } from './site.js';

export function usePageTitle(title) {
  useEffect(() => {
    document.title = title ? `${title} · ${SITE.name}` : SITE.name;
    // The address search engines should list for this page (the page file itself is the same for every route).
    const link = document.querySelector('link[rel="canonical"]');
    if (link) link.href = `${location.origin}${location.pathname}`;
  }, [title]);
}

// One shared price-list store: every consumer sees the same data, and a failed load can be retried later
// (loadPricing() is called again on the next mount and by the "API offline" banner's re-check).
let pricingPromise = null;
let pricingState = { data: null, error: null, loading: true };
const pricingListeners = new Set();
const setPricing = (next) => {
  pricingState = next;
  pricingListeners.forEach((fn) => fn());
};
const subscribePricing = (fn) => {
  pricingListeners.add(fn);
  return () => pricingListeners.delete(fn);
};
const getPricing = () => pricingState;

export function loadPricing() {
  if (pricingState.data) return Promise.resolve(pricingState.data);
  if (!pricingPromise) {
    pricingPromise = api('billing/pricing', { redirectOn401: false }).then(
      (d) => {
        pricingPromise = null;
        setPricing({ data: d, error: null, loading: false });
        return d;
      },
      (e) => {
        pricingPromise = null; // allow retry
        setPricing({ data: null, error: e.detail || 'Could not load prices.', loading: false });
        throw e;
      },
    );
  }
  return pricingPromise;
}

/** Public price list (loaded once per page, shared). */
export function usePricing() {
  const state = useSyncExternalStore(subscribePricing, getPricing);
  useEffect(() => {
    if (!pricingState.data) loadPricing().catch(() => {});
  }, []);
  return state;
}

export function findRate(pricing, product, tier) {
  return pricing?.rates?.find((r) => r.product === product && r.tier === tier) || null;
}

/** Cost estimate in micros, mirrors server/acw/billing/pricing.py (per started second of audio, no minimum charge). */
export function estimateMicros(pricing, { kind, denoiseTier = 'standard', transcribeTier = 'standard', seconds, translate = 0, leveling = false }) {
  if (!pricing || !seconds) return null;
  const secs = Math.max(0, Math.ceil(seconds));
  const cost = (product, tier) => {
    const r = findRate(pricing, product, tier);
    if (!r) return 0;
    return Math.ceil((Math.round(r.usdPerMinute * 1e6) * secs) / 60);
  };
  let total = 0;
  if (kind === 'denoise' || kind === 'denoise_transcribe') {
    if (denoiseTier !== 'off') total += cost('denoise', denoiseTier); // 'off': Leveling on its own
    if (leveling) total += cost('leveling', 'standard');
  }
  if (kind === 'transcribe' || kind === 'denoise_transcribe') {
    total += cost('transcribe', transcribeTier === 'premium' ? 'premium' : 'local');
    total += translate * cost('transcribe_translate', 'standard'); // per extra language
  }
  return total;
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
