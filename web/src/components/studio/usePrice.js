import { useMemo } from 'react';
import { quoteLocal } from '../../editor/quote.js';
import { convertPrice, delta as deltaOf, priceOf } from '../../editor/tracks.js';
import { serverPictureWork } from '../../media/convert.js';

/**
 * The price of starting, worked out here (editor/tracks.js priceOf, the same rules as our side) from the price list the page loaded once,
 * and what a video conversion on our side adds (`conversion`).
 */
export function useEstimate({ session, S, T, ver, plan, info, outFile, choice, pricing, server }) {
  // The price, worked out here at once with the same rules as our side (editor/tracks.js priceOf) from the price list the page loaded
  // once: it shows as soon as a setting changes. Our side works out the same price itself when the job is made.
  const price = useMemo(() => priceOf(S.tracks, pricing, T), [ver, pricing, T]); // eslint-disable-line react-hooks/exhaustive-deps
  // A video made on our side: Video conversion when the picture is made again (media/convert.js serverPictureWork, from the video's codec
  // and size as the file says them; if its codec were ever unknown the full price is shown, and our side charges less if it can copy it).
  const work = plan.serverConvert ? serverPictureWork(info, session.fileName || outFile?.name, choice.format, choice.size) : null;
  const factor = work?.factor || 1;
  const conversion = plan.serverConvert
    ? (work?.black ? { free: true, server: true, black: true }
      : work?.encode === false ? { free: true, server: true }
        : { micros: pricing ? convertPrice(pricing, factor, T) : 0, factor })
    : plan.converting ? { free: true } : null;
  const total = price.micros + (conversion && !conversion.free ? conversion.micros : 0);
  const estMicros = !server ? 0 : !pricing ? null : total;
  return { estMicros, conversion, price };
}

/** What saving the changes of a finished job costs: worked out here at once (editor/quote.js, the same planning and pricing as our side's
 * quote) from what the job's timeline answer carried when it was opened; a card made here: starting again costs what the paid parts cost. */
export function useQuote({ session, S, T, ver, job, isLocal, canEdit, changedTracks, c, rates, pricing, balance }) {
  const quote = useMemo(() => {
    if (!changedTracks || !job || isLocal || !canEdit) return { state: 'idle', q: null, error: '' };
    const qi = session.server?.quote;
    if (!qi || !pricing) return { state: 'loading', q: null, error: '' };
    try {
      const r = quoteLocal(session.timelineOf(S.tracks), qi, session.server.rendered, pricing.stepFixedSeconds);
      if (pricing.maxRebuildSeconds && r.rebuildSeconds > pricing.maxRebuildSeconds) {
        return { state: 'error', q: null, error: `Voice rebuild takes up to ${Math.floor(pricing.maxRebuildSeconds / 60)} minutes of a recording at a time; this asks for ${Math.round(r.rebuildSeconds / 60)}.` };
      }
      const available = balance?.available_micros;
      return { state: 'ok', q: { ...r, estCostMicros: r.micros, availableMicros: available, affordable: available == null || available >= r.micros }, error: '' };
    } catch (e) {
      return { state: 'error', q: null, error: e.message || 'Couldn’t work out the price.' };
    }
  }, [changedTracks, ver, job?.id, isLocal, canEdit, session.server, pricing, balance?.available_micros]); // eslint-disable-line react-hooks/exhaustive-deps
  const deltaMicros = changedTracks ? (quote.state === 'ok' && quote.q ? quote.q.estCostMicros : isLocal ? c.total : deltaOf(S.tracks, session.snap.tracks, T, rates)) : 0;
  const affordableSave = quote.state !== 'ok' || !quote.q || quote.q.affordable !== false;
  return { quote, deltaMicros, affordableSave };
}
