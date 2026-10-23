import { useEffect, useState } from 'react';
import { delta as deltaOf, tot } from '../../editor/tracks.js';
import { serverPictureWork } from '../../media/convert.js';
import { jobKind } from '../../studio/plan.js';

/**
 * The price of starting: worked out here from the public price list when signed out, otherwise asked of our side (it prices the
 * timeline as the job will be held), and what a video conversion on our side adds (`conversion`).
 */
export function useEstimate({ session, S, T, ver, plan, info, outFile, choice, rates, stage, signedIn, c, server }) {
  const [est, setEst] = useState({ state: 'idle', micros: null, error: '' });
  const kind = jobKind(S.tracks, plan);
  // a video made on our side: Video conversion, certain at a new size (another file type only: once the file is read, so "if needed")
  const convertLine = est.data?.lines?.find((l) => l.item === 'convert') || null;
  // (signed out, the same rules worked out here from the public price list: media/convert.js serverPictureWork)
  const work = plan.serverConvert ? serverPictureWork(info, session.fileName || outFile?.name, choice.format, choice.size) : null;
  const encodes = convertLine ? (convertLine.maybe ? null : !convertLine.copied) : work?.encode ?? null;
  const factor = convertLine ? convertLine.factor || 1 : work?.factor || 1;
  const convertMicros = plan.serverConvert && rates?.cv ? tot(rates.cv * factor * Math.ceil(T)) : 0;
  const conversion = plan.serverConvert
    ? (work?.black ? { free: true, server: true, black: true }
      : encodes === false ? { free: true, server: true } : { micros: convertLine ? convertLine.micros : convertMicros, maybe: encodes == null, factor })
    : plan.converting ? { free: true } : null;
  useEffect(() => {
    if (stage !== 'ready' || !kind || !session.file || !rates) { setEst({ state: 'idle', micros: null, error: '' }); return undefined; }
    // signed out there is nobody to ask: the price list is public, so the price is worked out here from it (what the file is is never asked)
    if (!signedIn) { setEst({ state: 'ok', micros: c.total + (conversion && !conversion.free && !conversion.maybe ? convertMicros : 0), error: '', local: true }); return undefined; }
    const body = session.estimateBody({ choice, plan });
    if (!body) return undefined;
    setEst((e) => ({ ...e, state: 'loading', error: '' }));
    const ctl = new AbortController();
    const t = setTimeout(() => {
      session.estimate(body, ctl.signal).then(
        (d) => setEst({ state: 'ok', micros: d.estCostMicros, error: '', data: d }),
        (e) => { if (e?.name !== 'AbortError') setEst({ state: 'error', micros: null, error: e.detail || e.message || 'Couldn’t work out the price of these parts.' }); },
      );
    }, 400);
    return () => { clearTimeout(t); ctl.abort(); };
  }, [stage, kind, ver, rates, choice.format, choice.size, plan.serverConvert, signedIn]); // eslint-disable-line react-hooks/exhaustive-deps
  // a job on our side is priced as it will be held: its timeline's price, which counts what each run costs besides the minutes (docs/api.md section 8)
  const estMicros = !server ? 0 : !rates ? null : est.state === 'ok' ? est.micros : null;
  return { est, estMicros, conversion };
}

/** What saving the changes of a finished job costs: our side's quote for a job of parts (a card made here: starting again costs what
 * the paid parts cost). */
export function useQuote({ session, S, T, job, isLocal, canEdit, changedTracks, c, rates }) {
  const [quote, setQuote] = useState({ state: 'idle', q: null, error: '' });
  // (asked again when the timeline it prices changes, not on every redraw: a redraw would cancel the answer on its way)
  const quoteKey = changedTracks && job && !isLocal && canEdit ? JSON.stringify(session.timelineOf(S.tracks).toJSON()) : '';
  useEffect(() => {
    if (!changedTracks || !job || isLocal || !canEdit) { setQuote({ state: 'idle', q: null, error: '' }); return undefined; }
    setQuote((q) => ({ ...q, state: 'loading', error: '' }));
    const ctl = new AbortController();
    const t = setTimeout(() => {
      session.quote(ctl.signal).then(
        (q) => setQuote({ state: 'ok', q, error: '' }),
        (e) => { if (e?.name !== 'AbortError') setQuote({ state: 'error', q: null, error: e.detail || e.message || 'Couldn’t work out the price.' }); },
      );
    }, 450);
    return () => { clearTimeout(t); ctl.abort(); };
  }, [changedTracks, quoteKey, job?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const deltaMicros = changedTracks ? (quote.state === 'ok' && quote.q ? quote.q.estCostMicros : isLocal ? c.total : deltaOf(S.tracks, session.snap.tracks, T, rates)) : 0;
  const affordableSave = quote.state !== 'ok' || !quote.q || quote.q.affordable !== false;

  return { quote, deltaMicros, affordableSave };
}
