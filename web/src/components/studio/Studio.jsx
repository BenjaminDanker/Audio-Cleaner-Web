import '../../styles/studio.css';
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { Link, useNavigate } from 'react-router';
import { api, apiOffline } from '../../api.js';
import { needsAccount } from '../../access.js';
import { useAuth } from '../../auth.jsx';
import { forgetFinish, removeLocalCard } from '../../finish.js';
import { bytes } from '../../format.js';
import { findRate, loadPricing, useNow, usePricing } from '../../hooks.js';
import { jobPath } from '../../jobs.js';
import { heardSeries, pcmPeaks } from '../../editor/peaks.js';
import { dismiss as dismissSuggestion } from '../../editor/suggest.js';
import { describe, prepare } from '../../editor/mix.js';
import {
  applyToSel, cost as costOf, dbOf, echoBlocked, edit, freePatch, fmt, hasWork, needsServer, overl, pushUndo, ratesFrom, sameOut, sameTracks,
  setSel, toggleRange, toTimeline, uni,
} from '../../editor/tracks.js';
import { buildLimit } from '../../media/convert.js';
import { denoiserLoadState, subscribeDenoiserLoad } from '../../media/denoise.js';
import { speechLoadState, subscribeSpeechLoad } from '../../media/speech.js';
import { wavFits } from '../../media/wav.js';
import { tipsFor } from '../../studio/copy.js';
import { browserVerdict, outputPlan, problemsFor, subsVerdict, wantsOf } from '../../studio/plan.js';
import { keepDraft } from '../../studio/draft.js';
import { barOf, downloadText, notesOf, progressLine } from '../../studio/screen.js';
import { forgetSession, warmUpDenoiser, warmUpSpeech } from '../../studio/session.js';
import OutputFields, { useOutputChoice } from '../OutputPicker.jsx';
import { useGoSignIn } from '../SignIn.jsx';
import Bar from './Bar.jsx';
import InfoModal from './InfoModal.jsx';
import Inspector from './Inspector.jsx';
import Timeline from './Timeline.jsx';
import TopPane from './TopPane.jsx';
import { useTooltips } from './tooltips.js';
import { usePlayback } from './usePlayback.js';
import { usePointer } from './usePointer.js';
import { useEstimate, useQuote } from './usePrice.js';

const noopSub = () => () => {};
const EMPTY_FIN = { phase: 'idle', progress: 0, files: [], note: '', error: '', pending: null, strengthDb: null };
const getEmpty = () => EMPTY_FIN;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const clone = (o) => JSON.parse(JSON.stringify(o));

function useMedia(query) {
  const [m, setM] = useState(() => (typeof matchMedia === 'function' ? matchMedia(query).matches : false));
  useEffect(() => {
    const mq = matchMedia(query);
    const on = () => setM(mq.matches);
    on();
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, [query]);
  return m;
}

/* ================================================================================================ the screen */

/** The editor screen for one session: before anything runs, while it runs, and when it is done. */
export default function Studio(props) {
  const { session } = props;
  const ver = useSyncExternalStore(session.subscribe, session.getVersion);
  const job = props.job || null;
  if (job && session.job !== job) session.attach(job, props.fin); // (the job as the list has it now)
  useEffect(() => {
    // a job opened from the list (or after a reload): its tracks come from our side
    if (job && !job.local && !session.S && !session.loading) session.loadJob();
  }, [job?.id, ver]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!session.S) {
    return (
      <div className="studio"><div className="container ws"><p role="status" className="muted mb0" data-testid="opening">{session.serverError ? <span className="banner banner-error" role="alert">{session.serverError}</span> : <><span className="spinner" aria-hidden="true" /> Opening the editor…</>}</p></div></div>
    );
  }
  return <Ready {...props} job={job} ver={ver} />;
}

function Ready({ session, job, fin, ver, onNewFile, onChanged, onDeleted, welcome = '', active = 0, maxActive = null }) {
  const nav = useNavigate();
  const { balance, refresh, signedIn, known } = useAuth();
  const goSignIn = useGoSignIn();
  const offline = useSyncExternalStore(apiOffline.subscribe, apiOffline.get);
  const { data: pricing, error: pricingError } = usePricing();
  const rates = useMemo(() => ratesFrom(pricing), [pricing]);
  const tips = useMemo(() => tipsFor(pricing), [pricing]);
  const S = session.S;
  const T = S.T;
  const mobile = useMedia('(max-width: 760px)');
  const finState = useSyncExternalStore(fin ? fin.subscribe : noopSub, fin ? fin.getSnapshot : getEmpty);
  const denoiserLoad = useSyncExternalStore(subscribeDenoiserLoad, denoiserLoadState);
  const speechLoad = useSyncExternalStore(subscribeSpeechLoad, speechLoadState);
  const now = useNow(5000);

  // ---- the UI's own state ----
  const [pop, setPop] = useState(null);
  const [modal, setModal] = useState(false);
  const [more, setMore] = useState(false);
  const [sheet, setSheet] = useState(false);
  const [sgIdx, setSgIdx] = useState(-1);
  const [delAsk, setDelAsk] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [hear, setHear] = useState({ voice: true, echo: true, level: true });
  const [toast, setToast] = useState('');
  const toastT = useRef(0);
  const rootRef = useRef(null);
  const tipRef = useRef(null);
  const scrollRef = useRef(null);
  const waveRef = useRef(null);
  const live = useRef({});
  useTooltips(tipRef);

  const say = useCallback((msg) => {
    setToast(msg);
    clearTimeout(toastT.current);
    toastT.current = setTimeout(() => setToast(''), 2600);
  }, []);
  useEffect(() => () => clearTimeout(toastT.current), []);

  // ---- where it stands ----
  const isLocal = !!job?.local;
  const saving = !!job && !!session.run?.saving && (job.status !== 'done' || job.finishedAt === session.savedAt);
  const stage = job ? (job.status === 'done' && !saving ? 'done' : 'processing') : session.stage;
  const processing = stage === 'processing';
  const done = stage === 'done';
  const info = session.info;
  const hasFileHere = !!session.file;
  const reading = session.wave.state === 'reading';
  const c = useMemo(() => costOf(S.tracks, rates), [ver, rates]); // eslint-disable-line react-hooks/exhaustive-deps
  const soundFromServer = useMemo(() => (done && job && !job.local ? !!fin?.serverAudio : needsServer(S.tracks) && c.paidAny && (c.nPaid + c.rSec + c.eSec + c.lSec > 0)), [ver, done, job?.id, fin, c]); // eslint-disable-line react-hooks/exhaustive-deps

  // the output choice (OutputPicker): the file type, name and picture size
  const outFile = useMemo(() => session.file || (fin ? { name: fin.inputName } : null), [session.file, fin?.inputName]); // eslint-disable-line react-hooks/exhaustive-deps
  const outInfo = useMemo(() => info || fin?.inputInfo || null, [info, fin?.inputName, fin?.info]); // eslint-disable-line react-hooks/exhaustive-deps
  const initial = useMemo(() => (job && fin ? (() => { const w = fin.effective(); return w.sound || w.embed ? { name: w.name, format: w.format, size: w.size, asMade: !w.sound && !w.convert } : null; })() : session.draftChoice || null), [job?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const choice = useOutputChoice({ file: outFile, info: outInfo, serverAudio: soundFromServer, initial });
  const plan = useMemo(() => outputPlan({ tracks: S.tracks, info: outInfo, file: session.file || outFile, choice, out: session.out, limit: buildLimit() }), [ver, outInfo, choice.format, choice.isVideoFile, choice.size, choice.server, choice.touched, choice.chosen, choice.valid, session.out.burn]); // eslint-disable-line react-hooks/exhaustive-deps
  // our side is needed for a priced part, or for a video this browser can't make (it goes up whole: plan.serverConvert)
  const server = plan.server;
  const verdict = useMemo(() => (info ? browserVerdict(info, { wavFits }) : { ok: false, why: 'Your original file isn’t in this tab, so this can’t be done here.' }), [info]);
  const sVerdict = useMemo(() => (info ? subsVerdict(info) : { ok: false, why: 'Your original file isn’t in this tab, so this can’t be done here.' }), [info]);
  const rebuildOffered = !!findRate(pricing, 'denoise', 'rebuild');
  const rebuildPaused = rebuildOffered && !pricing?.rebuildAvailable;
  session.env = { rates, pricing, rebuildOk: !rebuildOffered || !!pricing?.rebuildAvailable, active, maxActive, signedIn };

  // ---- the price (usePrice.js) ----
  const { est, estMicros, conversion } = useEstimate({ session, S, T, ver, plan, info, outFile, choice, rates, stage, signedIn, c, server });
  const changedTracks = done && !!session.snap && !sameTracks(S.tracks, session.snap.tracks, T);
  const outNow = useMemo(() => ({ format: choice.format, name: choice.name, size: choice.size }), [choice.format, choice.name, choice.size]);
  const mediaMade = !fin || (() => { const w = fin.effective(); return !!(w.sound || w.embed); })(); // (the file type and name only matter when a media file is made)
  const changedOut = done && !!session.snap && (mediaMade && !!session.snap.choice && (session.snap.choice.format !== outNow.format || session.snap.choice.name !== outNow.name || session.snap.choice.size !== outNow.size)
    || !sameOut(session.snap.sfiles, session.out.sfiles) || session.snap.burn !== session.out.burn);
  const changed = changedTracks || changedOut;
  const canEdit = !!job && (isLocal || !!job.edit?.available || (session.S && !job.edit));
  const locked = done && !!job && !isLocal && !canEdit;
  const lockedWhy = locked ? 'The files this needs are cleared away, so strength, Leveling and the other paid choices can’t be changed any more. File type, name and subtitle files still can.' : '';

  const { quote, deltaMicros, affordableSave } = useQuote({ session, S, T, job, isLocal, canEdit, changedTracks, c, rates });

  // ---- problems that stop a start ----
  const activeCount = active;
  const problems = problemsFor({
    info, file: session.file, tracks: S.tracks, est: estMicros, estReady: est.state !== 'loading', duration: T, maxJob: pricing?.maxJobSeconds, pricing, pricingError,
    balance, active: activeCount, maxActive: maxActive || null, plan, choice, out: session.out, partsError: est.state === 'error' ? est.error : '',
  });
  const cleaning = plan.cleaning;
  // The one rule for who can start what (access.js): a plan with a priced part is done on our side and needs an account; everything else runs here.
  // Offline, the paid parts can't be started by anyone, and signing in isn't possible either.
  // (until the account check has been answered nobody is told to sign in: it may be a busy moment, not a visitor who is signed out)
  const account = needsAccount(server, signedIn) && known && !offline && (stage === 'ready' || (done && isLocal && changedTracks));
  const canStart = stage === 'ready' && !!session.file && !!info && !reading && problems.length === 0 && (!server || (estMicros != null && !offline && known))
    && (!(cleaning || plan.converting) || choice.valid) && (hasWork(S.tracks) || plan.converting);

  // ---- the picture and the sound (usePlayback.js) ----
  const tl = useMemo(() => { try { return toTimeline(S.tracks, T); } catch { return null; } }, [ver]); // eslint-disable-line react-hooks/exhaustive-deps
  const mixHear = { echo: hear.echo, noise: hear.voice, rebuild: hear.voice, leveling: hear.level };
  const cached = session.server?.cached || {};
  const model = { tl, hear: mixHear, cached, secondTrack: S.second };

  const canMix = done && !!job && !isLocal && !!session.server?.preview;
  const localBlend = done && isLocal && !!fin?.pairParts && S.tracks.voice.length === 1 && S.tracks.voice[0].mode === 1 && S.tracks.voice[0].kind === 'free';
  const localAb = done && isLocal && !!fin?.wavUrl && !localBlend;
  const { ph, playing, playInfo, fileUrl, audioUrl, mediaRef, engineRef, seek, togglePlay } = usePlayback({
    session, job, fin, info, S, T, ver, hear, canMix, localBlend, localAb, live,
  });
  live.current = { model, hear, ph, S, session };

  // ---- the heard line and the words about it ----
  const data = useMemo(() => {
    const d = session.wave.data;
    if (d) return d;
    if (session.wave.state === 'reading') return null;
    const N = 400;
    return { N, top: 1, o: Array(N).fill(0.04), m: Array(N).fill(0.04), d: Array(N).fill(0) };
  }, [session.wave, ver]); // eslint-disable-line react-hooks/exhaustive-deps
  const [localPeaks, setLocalPeaks] = useState(null);
  useEffect(() => {
    let alive = true;
    setLocalPeaks(null);
    if (done && isLocal && fin?.pcm && session.wave.data) {
      Promise.resolve().then(() => pcmPeaks(fin.pcm, session.wave.data.top)).then((p) => { if (alive) setLocalPeaks(p); }).catch(() => {});
    }
    return () => { alive = false; };
  }, [done, isLocal, fin?.pcm, session.wave.data]); // eslint-disable-line react-hooks/exhaustive-deps
  const heard = useMemo(() => {
    if (!done || !data) return null;
    if (isLocal) {
      if (!localPeaks) return null;
      return Float32Array.from(hear.voice ? localPeaks.o : data.o);
    }
    const L = session.layers;
    if (!L?.source || !tl) return null;
    const top = Math.max(1e-6, ...Array.from(L.source.a).filter(Number.isFinite));
    try {
      return heardSeries({ N: data.N, T, prep: prepare(model), layers: L, top });
    } catch { return null; }
  }, [done, data, isLocal, localPeaks, hear, session.layers, tl, ver]); // eslint-disable-line react-hooks/exhaustive-deps
  // what is heard at the playhead (the gains the mix plays), for the page's own checks
  const gains = useMemo(() => {
    if (!done || !tl || isLocal) return '';
    try {
      const d = describe(prepare(model), ph);
      return JSON.stringify(Object.fromEntries(Object.entries(d).map(([k, v]) => [k, Math.round(v * 1000) / 1000])));
    } catch { return ''; }
  }, [done, tl, hear, ph, ver]); // eslint-disable-line react-hooks/exhaustive-deps
  const hearText = useMemo(() => {
    const L = [['voice', 'Voice'], ['echo', 'Echo removal'], ['level', 'Leveling']].filter(([tr]) => tr !== 'echo' || S.second);
    const off = L.filter(([tr]) => !hear[tr]);
    return !off.length ? 'Hearing: everything' : off.length === L.length ? 'Hearing: the original' : `Hearing: without ${off.map(([, n]) => n).join(' and ')}`;
  }, [hear, S.second]);

  // ---- the job while it runs / when it is done ----
  useEffect(() => {
    if (job && job.status === 'done' && fin && !saving) fin.auto();
  }, [job?.status, job?.finishedAt, fin, saving]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    // a run that finished since the tracks were last read (a save): they are read again; the first time, edits already made here are kept
    if (job && !isLocal && job.status === 'done' && !saving) session.loadJob({ force: session.loadedAt != null && session.loadedAt !== job.finishedAt });
  }, [job?.finishedAt, job?.status, saving]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!saving && session.run?.saving && job && job.status === 'done') { session.run = null; session.stage = 'ready'; session.touch(); }
  }, [saving]); // eslint-disable-line react-hooks/exhaustive-deps
  // the page is leaving while this browser still has work to do
  useEffect(() => {
    if (!session.busy) return undefined;
    const warn = (e) => { e.preventDefault(); e.returnValue = ''; };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [session.busy, ver]); // eslint-disable-line react-hooks/exhaustive-deps

  // ================================================================================================ actions

  const refreshAll = useCallback(() => { onChanged?.(); refresh?.(); }, [onChanged, refresh]);
  const rerender = () => session.touch();
  const commit = (fn) => { session.commit(fn); };
  const apply = (tr, patch) => { if (processing || locked || !session.hasAudio) return; commit((st) => applyToSel(st, tr, patch)); };
  const selectRange = (s, e, open) => {
    // (whole seconds; a part that ends within a second of the end of a recording of a fractional length goes to the very end, else it leaves a sliver)
    s = clamp(Math.round(s), 0, T); e = clamp(Math.round(e), 0, T); if (T - e < 1) e = T;
    setSel(S, [{ s, e: Math.max(e, s + 1 > T ? T : s + 1) }]);
    if (open && mobile && S.sel) setSheet(true);
    rerender();
  };
  const selectAll = () => { S.sel = null; setMore(false); rerender(); };
  const applySuggestion = (g) => {
    pushUndo(S);
    edit(S, 'voice', g.s, g.e, g.patch);
    g.st = 'applied';
    session.touch();
  };
  const dismissSuggestionFor = (g) => {
    pushUndo(S);
    g.st = 'dismissed';
    if (g.key) dismissSuggestion(job?.id || session.id, g.key);
    session.touch();
  };
  const split = () => {
    if (processing) return;
    if (!session.split(Math.round(ph))) { say('Move the playhead inside the file to split'); return; }
    const t = Math.round(ph);
    const piece = S.tracks.voice.find((x) => x.s <= t && t < x.e);
    if (piece) selectRange(piece.s, piece.e, true);
  };

  // A plan that needs an account: keep the file and the choices in this browser, go and sign in, and they are back on return (studio/draft.js).
  const signInToStart = async () => {
    if (!session.file) return;
    try {
      await keepDraft(session, choice);
    } catch {
      if (!session.keepFailed) {
        session.keepFailed = true;
        session.error = 'Your browser couldn’t keep this file while you sign in, so you would have to choose it again afterwards. Choose Sign in to start again to go on anyway.';
        session.touch();
        return;
      }
    }
    goSignIn('/app');
  };

  const onPrimary = async () => {
    if (account && (stage === 'ready' ? canStart : true)) { await signInToStart(); return; }
    if (stage === 'ready') {
      if (!canStart) return;
      const res = await session.start({ choice, plan, refresh });
      if (res?.id) { session.touch(); nav(jobPath(res.id), { state: session.welcome ? { submitted: session.welcome } : null }); }
      else if (session.canceled) { session.canceled = false; say('Canceled. You weren’t charged.'); }
      return;
    }
    if (done) {
      if (changed) {
        const w = wantsOf({ tracks: S.tracks, plan, choice, out: session.out, file: session.file, fileName: session.fileName, local: isLocal });
        const db = reblendDb();
        if (isLocal && changedTracks && db == null) {
          // a card made here: the new tracks run again from the file
          const res = await session.start({ choice, plan, refresh });
          if (res?.id) nav(jobPath(res.id));
          return;
        }
        const keep = fin.effective();
        const wants = { ...keep, ...w, soundHere: isLocal ? keep.soundHere : false, subsHere: isLocal ? keep.subsHere : false };
        if (db != null) {
          // a new strength for the free cleanup made here: put together again from the two signals it kept (free, nothing is made twice)
          try { await fin.blendTo(db); await fin.run({ save: true, wants }); session.takeSnapshot(S.tracks, choice); session.touch(); say('Saved'); } catch (e) { session.error = e?.message || 'Couldn’t save this. Please try again.'; session.touch(); }
          return;
        }
        if (!changedTracks) { await session.saveChanges({ choice, plan, wants, quote: null, refresh }); say('Saved'); return; }
        session.savedAt = job.finishedAt;
        const r = await session.saveChanges({ choice, plan, wants, quote: quote.q, refresh, onStarted: refreshAll });
        if (r) { say('Saved'); setTimeout(() => onChanged?.(), 1200); setTimeout(() => onChanged?.(), 3500); }
        return;
      }
      if (finState.phase === 'needvideo' || (finState.phase === 'idle' && fin?.needsVideo(fin.effective()))) { rootRef.current?.querySelector('[data-testid=need-video-input]')?.click(); return; }
      fin?.again();
    }
  };

  // only the strength of a free cleanup made here changed: the new sound is a blend of the two signals the run kept
  const reblendDb = () => {
    if (!isLocal || !fin?.pairParts || !changedTracks || !session.snap) return null;
    const v = S.tracks.voice;
    const w = session.snap.tracks.voice;
    if (v.length !== 1 || w.length !== 1 || v[0].mode !== 1 || v[0].kind !== 'free' || w[0].mode !== 1 || w[0].kind !== 'free') return null;
    const flat = (t) => { const c = clone(t); c.voice.forEach((x) => { x.str = 0; x.db = null; }); return c; };
    return sameTracks(flat(S.tracks), flat(session.snap.tracks), T) ? dbOf(v[0]) : null;
  };

  const doDelete = async () => {
    if (isLocal) { removeLocalCard(job.id); forgetSession(job.id); onDeleted?.(); say('Removed'); return; }
    try {
      await api(`jobs/${job.id}`, { method: 'DELETE' });
      forgetFinish(job.id);
      forgetSession(job.id);
      onDeleted?.();
      say('Deleted');
    } catch (e) {
      session.deleteError = e.detail || e.message || 'Couldn’t delete this. Please try again.';
      setDelAsk(false);
      session.touch();
    }
  };

  const cancelJob = async () => {
    if (!job) { session.cancelRun(); return; }
    if (finState.phase === 'building' || finState.phase === 'saving') { fin.cancel(); return; }
    try { await api(`jobs/${job.id}/cancel`, { method: 'POST' }); onChanged?.(); refresh?.(); } catch (e) { session.error = e.detail || e.message; session.touch(); }
  };

  // clicks (the prototype's data-act switch)
  const onClick = (ev) => {
    if (mobile && sheet && !ev.target.closest('.ed, #insp, .bb, .pop, dialog')) setSheet(false);
    const b = ev.target.closest('[data-act]');
    if (!b) { if (pop && !ev.target.closest('.pop')) setPop(null); return; }
    const a = b.dataset.act;
    const id = b.dataset.id;
    const v = b.dataset.v;
    if (pop && a !== 'pop' && !ev.target.closest('.pop')) setPop(null);
    if (suppress.current && (a === 'sec' || a === 'sheet')) return;
    if (b.getAttribute('aria-disabled') === 'true') return; // its tooltip says why
    switch (a) {
      case 'newfile': onNewFile?.(); break;
      case 'auto': session.setAuto(!session.auto); break;
      case 'auto-undo': session.undoAuto(); break;
      case 'auto-again': session.useAgain(); break;
      case 'sg-cancel': session.cancelMoments(); break;
      case 'play': togglePlay(); break;
      case 'split': split(); break;
      case 'undo': session.undo(); break;
      case 'redo': session.redo(); break;
      case 'all': selectAll(); break;
      case 'zoom-in': setZoom((z) => Math.min(8, z * 2)); break;
      case 'zoom-out': setZoom((z) => Math.max(1, z / 2)); break;
      case 'sgchip': {
        const list = S.sugg.filter((g) => g.st === 'new');
        if (!list.length) break;
        const i = (sgIdx + 1) % list.length;
        setSgIdx(i);
        const g = list[i];
        seek(g.s);
        selectRange(g.s, g.e, true);
        break;
      }
      case 'pin': { const g = S.sugg.find((x) => x.id === +id); if (g) { seek(g.s); selectRange(g.s, g.e, true); } break; }
      case 'sec': {
        if (processing) break;
        if (lpFired.current) { lpFired.current = false; break; }
        const s0 = +b.dataset.s;
        const e0 = +b.dataset.e;
        if (ev.shiftKey || more) { toggleRange(S, s0, e0); rerender(); } else selectRange(s0, e0, true);
        break;
      }
      case 'more': setMore((m) => !m); break;
      case 'sheet': setSheet((s) => !s); break;
      case 'sg-apply': { const g = S.sugg.find((x) => x.id === +id); if (g && !processing) applySuggestion(g); break; }
      case 'sg-dismiss': { const g = S.sugg.find((x) => x.id === +id); if (g) dismissSuggestionFor(g); break; }
      case 'v-mode': apply('voice', { mode: +v }); break;
      case 'v-str': apply('voice', { str: +v, db: null }); break;
      case 'v-kind':
        if (v === 'free') {
          warmUpDenoiser();
          if (processing || locked) break;
          commit((st) => { applyToSel(st, 'voice', freePatch); applyToSel(st, 'echo', { on: 0 }); });
        } else apply('voice', { kind: 'paid' });
        break;
      case 'v-rem': apply('voice', { rem: +v }); break;
      case 'e-sw': if (!echoBlocked(S)) apply('echo', { on: uni(S, 'echo', 'on') === 1 ? 0 : 1 }); break;
      case 'l-lvl': apply('level', { lvl: +v, lufs: null }); break;
      case 's-q': if (+v === 1) warmUpSpeech(); apply('subs', { q: +v }); break;
      case 'tr-del': apply('subs', (x) => ({ tr: (x.tr || []).filter((l) => l !== v) })); break;
      case 'nudge': nudge(b.dataset.w, +b.dataset.d); break;
      case 'useph': useHere(b.dataset.w); break;
      case 'playpart': { const r = S.sel ? S.sel[0] : { s: 0, e: T }; togglePlay(r.s, r.e); break; }
      case 'info': setModal(true); break;
      case 'hear': setHear((hh) => ({ ...hh, [b.dataset.tr]: !hh[b.dataset.tr] })); break;
      case 'cancel': cancelJob(); break;
      case 'del': setDelAsk(true); break;
      case 'del-no': setDelAsk(false); break;
      case 'del-yes': doDelete(); break;
      case 'pop': setPop((p) => (p === v ? null : v)); break;
      case 'primary': onPrimary(); break;
      default: break;
    }
  };
  const onChange = (ev) => {
    const b = ev.target.closest('[data-act]');
    if (!b) return;
    const a = b.dataset.act;
    if (a === 's-lang') apply('subs', (x) => ({ lang: b.value, tr: (x.tr || []).filter((l) => l !== b.value) }));
    else if (a === 'tr-add' && b.value) {
      const l = b.value;
      apply('subs', (x) => { const t = (x.tr || []).filter((k) => k !== x.lang); return { tr: t.includes(l) || l === x.lang || t.length >= 3 ? t : t.concat(l) }; });
    } else if (a === 'o-sf') {
      session.out.sfiles = ['srt', 'vtt', 'txt'].filter((k) => (k === b.value ? b.checked : session.out.sfiles.includes(k)));
      rerender();
    } else if (a === 'o-burn') {
      session.out.burn = b.checked;
      // the cleaned video becomes an MKV, the kind of file players show these subtitles from
      if (b.checked && plan.cleaning && !plan.embedOnServer && choice.format !== 'mkv' && choice.okIds?.includes('mkv') && choice.isVideoFile) choice.pickFormat('mkv');
      rerender();
    }
  };

  const nudge = (which, d) => {
    if (processing || locked) return;
    const r = S.sel ? S.sel[0] : { s: 0, e: T };
    if (S.sel && S.sel.length > 1) return;
    if (which === 's') selectRange(Math.min(r.s + d, r.e - 1), r.e);
    else selectRange(r.s, Math.max(r.e + d, r.s + 1));
  };
  const useHere = (which) => {
    if (processing || locked) return;
    const r = S.sel ? S.sel[0] : { s: 0, e: T };
    if (which === 's') selectRange(Math.min(ph, r.e - 1), r.e);
    else selectRange(r.s, Math.max(ph, r.s + 1));
  };

  // the slider: one undo step for a whole drag
  const sliding = useRef(false);
  const slideStart = () => { if (!sliding.current && !processing && !locked) { pushUndo(S); sliding.current = true; } };
  const slideEnd = () => { sliding.current = false; };
  const onDb = (db) => {
    if (processing || locked) return;
    slideStart();
    const lim = overl(S, 'voice').some((x) => x.mode === 1 && x.kind === 'free') ? 40 : 100;
    const v = Math.min(db, lim);
    const nearest = [12, 24, 40, 100].reduce((p, q) => (Math.abs(q - v) < Math.abs(p - v) ? q : p), 12);
    applyToSel(S, 'voice', { db: Math.abs(nearest - v) < 0.4 ? null : v, str: Math.max(1, [12, 24, 40, 100].indexOf(nearest) + 1) });
    session.touch();
  };

  // ---- the pointer: drag to choose, handles, long-press, the sheet's header (usePointer.js) ----
  const { suppress, lpFired } = usePointer({ rootRef, waveRef, T, live });
  Object.assign(live.current, { mobile, processing, setMore, setSheet, setPop, pop, modal, more, seek, togglePlay, session });

  // the bar's height, for the room under the page
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return undefined;
    const set = () => {
      const bb = root.querySelector('.bb');
      const ins = root.querySelector('#insp');
      root.style.setProperty('--bar-h', `${bb ? bb.offsetHeight : 0}px`);
      root.style.setProperty('--sel-h', mobile && ins ? `${ins.offsetHeight + 8}px` : '0px');
    };
    set();
    const ro = window.ResizeObserver ? new ResizeObserver(set) : null;
    root.querySelectorAll('.bb, #insp').forEach((n) => ro?.observe(n));
    window.addEventListener('resize', set);
    return () => { ro?.disconnect(); window.removeEventListener('resize', set); };
  }, [mobile, stage, sheet, ver, pop]);

  // keep the playhead in view when the timeline is zoomed
  useEffect(() => {
    const sc = scrollRef.current;
    if (!sc || zoom <= 1) { if (sc) sc.scrollLeft = 0; return; }
    const wave = waveRef.current;
    const x = (ph / T) * (wave?.offsetWidth || 0);
    if (x < sc.scrollLeft + 40 || x > sc.scrollLeft + sc.clientWidth - 40) sc.scrollLeft = Math.max(0, x - sc.clientWidth / 2);
  }, [zoom]); // eslint-disable-line react-hooks/exhaustive-deps

  // ================================================================================================ what is shown

  const pend = session.auto ? S.sugg.filter((g) => g.st === 'new').length : 0;
  const pins = session.auto && !processing && !reading ? S.sugg.filter((g) => g.st !== 'dismissed') : [];
  const fileMeta = `${session.lengthUnknown ? 'about ' : ''}${fmt(T)} · ${session.kind === 'video' ? 'Video' : 'Audio'}${session.file ? ` · ${bytes(session.file.size)}` : ''}`;
  const reco = session.reco;

  // processing: what the bar says
  const procLine = useMemo(() => progressLine({ processing, job, isLocal, saving, session, denoiserLoad, speechLoad }), [processing, job, ver, saving, denoiserLoad, speechLoad, now]); // eslint-disable-line react-hooks/exhaustive-deps
  const keepOpen = processing && !job && Object.keys(session.run?.tasks || {}).some((k) => /^(clean|words)/.test(k));
  const building = done && ['waiting', 'building', 'saving'].includes(finState.phase) && !changed;
  const needVideo = done && !!fin && (finState.phase === 'needvideo' || (finState.phase === 'idle' && fin.needsVideo(fin.effective())));

  // the files: what will be made, or what was
  const files = done && fin ? (finState.files.length ? finState.files : fin.planned(fin.effective())) : [];
  const previewText = fin?.subs && !fin.subs.empty ? fin.subs.txt.split(/\n+/).filter(Boolean).slice(0, 3).join(' ') : '';
  const expiresIn = job?.expiresAt ? job.expiresAt - now / 1000 : null;

  const willDownload = downloadText({ plan, choice, session });
  const tooBigNote = plan.wantsText && plan.videoResult && plan.tooBigHere ? `This browser can’t rebuild a video this big (over ${Math.round((buildLimit() / 1024 ** 3) * 10) / 10} GB); Chrome, Edge or Firefox can. You’ll get the subtitle files to load in your player or video editor.` : '';
  const mkvNote = !plan.cleaning && !plan.converting && plan.embedFormat && plan.embedFormat !== 'mkv' && plan.embedFormat !== 'webm' ? plan.embedFormat.toUpperCase() : '';

  // the long-recording estimate of the Free cleanup
  const loadLeft = denoiserLoad.done ? 0 : denoiserLoad.secondsLeft;
  const eta = (() => {
    if (!info?.duration) return null;
    const channels = info.channels && info.channels >= 2 ? 2 : 1;
    const cores = navigator.hardwareConcurrency || 2;
    return info.duration * 0.13 * (channels === 2 && cores < 3 ? 2 : 1) + (loadLeft || 0);
  })();

  const bar = barOf({ stage, account, estMicros, server, processing, saving, procLine, building, finState, changed, deltaMicros, needVideo });
  const primaryDisabled = stage === 'ready' ? !canStart : processing || building ? true
    : changed ? (!choice.valid && plan.cleaning && !locked ? true : (changedTracks && (quote.state !== 'ok' && !isLocal ? true : !affordableSave)) || (isLocal && changedTracks && !hasFileHere) || (plan.wantsText && !session.out.sfiles.length))
      : false;

  // notes at the top
  const notes = notesOf({
    offline, session, stage, signedIn, problems, keepOpen, welcome, job, isLocal, done, finState, needVideo, fin, playInfo, changedTracks, quote, affordableSave,
  });

  const ctx = {
    S, session, tips, stage, sheet, mobile, info, verdict, sVerdict, rebuildPaused, locked, lockedWhy, eta, loadLeft, ph, zoom, pins, hear, hearText, heard, data,
    scrollRef, waveRef, reading, prog: processing ? (procLine?.value ?? 0) : 0, cost: c, total: estMicros, rates: rates || { std: 0, ext: 0 }, deltaMicros, changed: changedTracks, serverTotal: !server ? 0 : est.state === 'ok' ? est.micros : null,
    partsNote: needsServer(S.tracks) && est.state === 'ok' && !est.local, localPrice: server && est.state === 'ok' && !!est.local, conversion, pop, choice, plan, out: session.out, files, preview: previewText, job, expiresIn, delAsk, paid: job?.costMicros || session.paid || 0, finState, fin, bar, needVideo,
    disabled: processing, willDownload, tooBigNote, mkvNote, primaryDisabled,
    h: { onDb, slideStart, slideEnd, onVideo: (f) => fin?.setVideo(f) },
  };

  return (
    <div
      className="studio" ref={rootRef} data-testid="studio" data-job-id={job?.id} data-stage={stage} data-phase={done ? finState.phase : undefined} onClick={onClick} onChange={onChange}
      {...(done && fin?.pcm ? { 'data-rms': fin.pcm.rms.toFixed(6), 'data-peak': fin.pcm.peak.toFixed(4), 'data-frames': fin.pcm.frames, 'data-channels': fin.pcm.channels } : {})}
      {...(done && fin?.subs ? { 'data-cues': fin.subs.cues.length, 'data-language': fin.subs.language } : {})}
      {...(done && finState.strengthDb != null ? { 'data-strength-db': finState.strengthDb.toFixed(2) } : {})}
      {...(done && gains ? { 'data-gains': gains } : {})}
      data-time={ph.toFixed(2)} data-playing={playing ? 'yes' : 'no'} data-held={playInfo.held ?? 0} data-buffering={playInfo.buffering ? 'yes' : 'no'} data-seconds={T} data-changed={done ? (changedTracks ? 'tracks' : changedOut ? 'output' : 'no') : undefined}
    >
      <div className="container ws" data-stage={stage}>
        {notes.length > 0 && (
          <div className="notes" data-testid="notes">
            {notes.map((n) => (
              <div key={n.key} className={n.cls} role={n.cls.startsWith('banner') ? 'alert' : undefined} data-testid={n.testid}>
                {n.text}{' '}
                {n.kind === 'credit' && <Link to="/app/billing">Add credit</Link>}
                {n.kind === 'prices' && <button type="button" className="btn btn-ghost btn-sm" onClick={() => loadPricing().catch(() => {})}>Try again</button>}
              </div>
            ))}
          </div>
        )}
        <div className="ed">
          <div className="pane main" id="edmain">
            <TopPane {...{ session, job, S, T, info, fileUrl, audioUrl, mediaRef, engineRef, done, localAb, ph, playing, signedIn, pend, reco, zoom, more, fileMeta }} />
            <Timeline ctx={ctx} />
          </div>
          <aside className="pane insp lk" id="insp" aria-label="Selection" data-testid="inspector" data-sheet={sheet ? 'open' : 'min'} ref={(el) => el && el.classList.toggle('min', !sheet)}>
            <Inspector ctx={{ ...ctx, h: { ...ctx.h } }} />
          </aside>
        </div>
      </div>
      <Bar ctx={ctx} />
      <div className="toast" role="status" hidden={!toast} data-testid="toast">{toast}</div>
      <div className="tip" role="tooltip" hidden ref={tipRef} />
      <InfoModal open={modal} onClose={() => setModal(false)} pricing={pricing} />
    </div>
  );
}

export { OutputFields };
