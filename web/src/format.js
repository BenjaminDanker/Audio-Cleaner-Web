// Number/time/money formatting helpers. Money is stored as micros (1 USD = 1,000,000).

const usd2 = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });
const usd4 = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 4, maximumFractionDigits: 4 });

/** $x.xx from micros. Shows 4 decimals when the amount has sub-cent precision (usage lines). */
export function money(micros, { precise = false } = {}) {
  const n = (micros || 0) / 1e6;
  if (precise || (Math.round(micros || 0) % 10000 !== 0)) return usd4.format(n);
  return usd2.format(n);
}

/** Account balances: whole cents, rounded down (never promises more than is there). */
export function balanceMoney(micros) {
  return usd2.format(Math.floor((micros || 0) / 10000) / 100);
}

export function moneyCents(cents) {
  return usd2.format((cents || 0) / 100);
}

export function moneyUsd(usd) {
  return usd2.format(usd || 0);
}

/** Per-minute rates: always 4 decimals. */
export function rate(usd) {
  return usd4.format(usd || 0);
}

export function bytes(n) {
  if (n == null) return '';
  if (n < 1024) return `${n} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v >= 100 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

export function clock(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '--:--';
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const mm = String(m).padStart(2, '0');
  const ss = String(sec).padStart(2, '0');
  return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

/** "3 min 20 s" style, for ETAs and durations in prose. */
export function human(seconds) {
  if (!Number.isFinite(seconds)) return '';
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ${String(s % 60).padStart(2, '0')} s`;
  const h = Math.floor(m / 60);
  return `${h} h ${String(m % 60).padStart(2, '0')} min`;
}

export function dateTime(epochSeconds) {
  if (!epochSeconds) return '';
  return new Date(epochSeconds * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

export function stem(filename) {
  const base = String(filename || 'audio').split(/[\\/]/).pop();
  const i = base.lastIndexOf('.');
  return (i > 0 ? base.slice(0, i) : base) || 'audio';
}

export function extension(filename) {
  const base = String(filename || '').split(/[\\/]/).pop();
  const i = base.lastIndexOf('.');
  return i > 0 ? base.slice(i + 1).toLowerCase() : '';
}
