// Words for the "Credit history" table. The server sends a short internal `kind` ("usage", "topup", ...) and a
// description written for the log ("denoise + transcribe: noisy.flac (0.2 min)"); neither is for a visitor. These turn
// them into the names the rest of the site uses. Unknown kinds and descriptions are shown as they are, never hidden.
// (A server field with the job's kind, the file name and the minutes would let us drop the pattern matching below.)

import { shownName } from './format.js';

const KIND = {
  usage: 'Used',
  topup: 'Top-up',
  grant: 'Credit added',
  refund: 'Refund',
  dispute: 'Dispute',
  adjustment: 'Adjustment',
};

export const ledgerKind = (l) => KIND[l?.kind] || l?.label || String(l?.kind || '');

const WORK = {
  denoise: 'Denoising',
  transcribe: 'Subtitles & transcript',
  'denoise + transcribe': 'Denoising and subtitles',
};

/**
 * A line of the ledger in plain words. The server (newer versions) sends `label` (what it was, in the price list's
 * words), `displayName` (the file as the visitor chose it) and `minutes` with usage entries; without them the
 * description written for the log is read instead.
 */
export function ledgerText(l) {
  if (l?.kind === 'usage' && l.label && l.product) {
    const mins = Number.isFinite(l.minutes) ? ` (${l.minutes} min)` : '';
    return l.displayName && l.product !== 'live' ? `${l.label}: ${shownName(l.displayName)}${mins}` : `${l.label}${mins}`;
  }
  const d = String(l?.description || '').trim();
  if (!d) return l?.kind === 'usage' ? 'Used' : '—';
  let m = /^(denoise \+ transcribe|denoise|transcribe): (.*) \((\d+(?:\.\d+)?) min\)$/s.exec(d);
  if (m) return `${WORK[m[1]]}: ${shownName(m[2])} (${m[3]} min)`;
  m = /^Live session \((\d+(?:\.\d+)?) min(, cut short by a restart)?\)$/.exec(d);
  if (m) return `Live session (${m[1]} min${m[2] ? ', ended early' : ''})`;
  return shownName(d);
}
