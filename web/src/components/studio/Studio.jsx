import '../../styles/studio.css';
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { Link, useNavigate } from 'react-router';
import { api } from '../../api.js';
import { useAuth } from '../../auth.jsx';
import { etaSeconds, etaText } from '../../eta.js';
import { forgetFinish, removeLocalCard } from '../../finish.js';
import { balanceMoney, bytes, human } from '../../format.js';
import { findRate, loadPricing, useNow, usePricing } from '../../hooks.js';
import { ACTIVE, jobPath, queueText, stageText } from '../../jobs.js';
import { heardSeries } from '../../editor/peaks.js';
import { dismiss as dismissSuggestion } from '../../editor/suggest.js';
import { MixPlayer, describe, prepare } from '../../editor/mix.js';
import {
  applyToSel, cost as costOf, covered, dbOf, delta as deltaOf, echoBlocked, edit, freePatch, fmt, money, needsServer, overl, pushUndo, ranges, ratesFrom, sameOut, sameTracks,
  selTitle, setSel, splitAll, toggleRange, toTimeline, uni, trOf,
} from '../../editor/tracks.js';
import { BlendPlayer, gainsFor, matchGain, memorySource, previewFits } from '../../media/blend.js';
import { buildLimit } from '../../media/convert.js';
import { denoiserLoadState, subscribeDenoiserLoad } from '../../media/denoise.js';
import { speechLoadState, subscribeSpeechLoad } from '../../media/speech.js';
import { wavFits } from '../../media/wav.js';
import { AbEngine, ElementEngine, syncPicture } from '../../studio/engine.js';
import { tipsFor } from '../../studio/copy.js';
import { browserVerdict, LONG_SECONDS, outputPlan, problemsFor, subsVerdict, supported, wantsOf, jobKind } from '../../studio/plan.js';
import { forgetSession, warmUpDenoiser, warmUpSpeech } from '../../studio/session.js';
import OutputFields, { useOutputChoice } from '../OutputPicker.jsx';
import Bar from './Bar.jsx';
import { IcAll, IcPause, IcPlay, IcRedo, IcMore, IcSplit, IcUndo, IcZoomIn, IcZoomOut, IcSpark } from './icons.jsx';
import InfoModal from './InfoModal.jsx';
import Inspector from './Inspector.jsx';
import Timeline from './Timeline.jsx';
import { useTooltips } from './tooltips.js';

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
  const { balance, refresh, contact } = useAuth();
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
  const [ph, setPh] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [pop, setPop] = useState(null);
  const [modal, setModal] = useState(false);
  const [more, setMore] = useState(false);
  const [sheet, setSheet] = useState(false);
  const [sgIdx, setSgIdx] = useState(-1);
  const [delAsk, setDelAsk] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [hear, setHear] = useState({ voice: true, echo: true, level: true });
  const [toast, setToast] = useState('');
  const [quote, setQuote] = useState({ state: 'idle', q: null, error: '' });
  const [est, setEst] = useState({ state: 'idle', micros: null, error: '' });
  const [playInfo, setPlayInfo] = useState({ buffering: false, error: '' });
  const [fileUrl, setFileUrl] = useState('');
  const toastT = useRef(0);
  const rootRef = useRef(null);
  const tipRef = useRef(null);
  const scrollRef = useRef(null);
  const waveRef = useRef(null);
  const mediaRef = useRef(null);
  const engineRef = useRef(null);
  const stopAt = useRef(null);
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
  const server = needsServer(S.tracks);
  const c = useMemo(() => costOf(S.tracks, rates), [ver, rates]); // eslint-disable-line react-hooks/exhaustive-deps
  const soundFromServer = useMemo(() => (done && job && !job.local ? !!fin?.serverAudio : needsServer(S.tracks) && c.paidAny && (c.nPaid + c.rSec + c.eSec + c.lSec > 0)), [ver, done, job?.id, fin, c]); // eslint-disable-line react-hooks/exhaustive-deps

  // the output choice (OutputPicker): the file type, name and picture size
  const outFile = useMemo(() => session.file || (fin ? { name: fin.inputName } : null), [session.file, fin?.inputName]); // eslint-disable-line react-hooks/exhaustive-deps
  const outInfo = useMemo(() => info || fin?.inputInfo || null, [info, fin?.inputName, fin?.info]); // eslint-disable-line react-hooks/exhaustive-deps
  const initial = useMemo(() => (job && fin ? (() => { const w = fin.effective(); return w.sound || w.embed ? { name: w.name, format: w.format, size: w.size } : null; })() : null), [job?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const choice = useOutputChoice({ file: outFile, info: outInfo, serverAudio: soundFromServer, initial });
  const plan = useMemo(() => outputPlan({ tracks: S.tracks, info: outInfo, file: session.file || outFile, choice, out: session.out, limit: buildLimit() }), [ver, outInfo, choice.format, choice.isVideoFile, session.out.burn]); // eslint-disable-line react-hooks/exhaustive-deps
  const verdict = useMemo(() => (info ? browserVerdict(info, { wavFits }) : { ok: false, why: 'Your original file isn’t in this tab, so this can’t be done here.' }), [info]);
  const sVerdict = useMemo(() => (info ? subsVerdict(info) : { ok: false, why: 'Your original file isn’t in this tab, so this can’t be done here.' }), [info]);
  const rebuildOffered = !!findRate(pricing, 'denoise', 'rebuild');
  const rebuildPaused = rebuildOffered && !pricing?.rebuildAvailable;
  session.env = { rates, pricing, rebuildOk: !rebuildOffered || !!pricing?.rebuildAvailable, active, maxActive };

  // ---- the price ----
  const multi = Object.values(S.tracks).some((L) => L.length > 1);
  const kind = jobKind(S.tracks);
  useEffect(() => {
    if (stage !== 'ready' || !kind || !multi || !session.file || !rates) { setEst({ state: 'idle', micros: null, error: '' }); return undefined; }
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
  }, [stage, kind, multi, ver, rates, choice.format]); // eslint-disable-line react-hooks/exhaustive-deps
  const estMicros = !server ? 0 : !rates ? null : multi ? (est.state === 'ok' ? est.micros : null) : c.total;
  const changedTracks = done && !!session.snap && !sameTracks(S.tracks, session.snap.tracks, T);
  const outNow = useMemo(() => ({ format: choice.format, name: choice.name, size: choice.size }), [choice.format, choice.name, choice.size]);
  const changedOut = done && !!session.snap && (!!session.snap.choice && (session.snap.choice.format !== outNow.format || session.snap.choice.name !== outNow.name || session.snap.choice.size !== outNow.size)
    || !sameOut(session.snap.sfiles, session.out.sfiles) || session.snap.burn !== session.out.burn);
  const changed = changedTracks || changedOut;
  const canEdit = !!job && (isLocal || !!job.edit?.available || (session.S && !job.edit));
  const locked = done && !!job && !isLocal && !canEdit;
  const lockedWhy = locked ? 'The files this needs are cleared away, so strength, Leveling and the other paid choices can’t be changed any more. File type, name and subtitle files still can.' : '';

  // what saving the changes costs: our side's quote for a job of parts (a card made here: starting again costs what the paid parts cost)
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
  }, [changedTracks, ver, job?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const deltaMicros = changedTracks ? (quote.state === 'ok' && quote.q ? quote.q.estCostMicros : isLocal ? c.total : deltaOf(S.tracks, session.snap.tracks, T, rates)) : 0;
  const affordableSave = quote.state !== 'ok' || !quote.q || quote.q.affordable !== false;

  // ---- problems that stop a start ----
  const activeCount = active;
  const problems = problemsFor({
    info, file: session.file, tracks: S.tracks, est: estMicros, estReady: est.state !== 'loading', duration: T, maxJob: pricing?.maxJobSeconds, pricing, pricingError,
    balance, active: activeCount, maxActive: maxActive || null, plan, choice, out: session.out, partsError: est.state === 'error' ? est.error : '',
  });
  const cleaning = plan.cleaning;
  const canStart = stage === 'ready' && !!session.file && !!info && !reading && problems.length === 0 && (!server || estMicros != null) && (!cleaning || choice.valid)
    && (server || S.tracks.voice.some((x) => x.mode === 1 && x.kind === 'free') || S.tracks.subs.some((x) => x.q === 1));

  // ---- the picture and the sound ----
  useEffect(() => {
    if (!session.file || !info?.hasVideo) { setFileUrl(''); return undefined; }
    const u = URL.createObjectURL(session.file);
    setFileUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [session.file, info?.hasVideo]);
  const [audioUrl, setAudioUrl] = useState('');
  useEffect(() => {
    if (!session.file || info?.hasVideo) { setAudioUrl(''); return undefined; }
    const u = URL.createObjectURL(session.file);
    setAudioUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [session.file, info?.hasVideo]);

  const tl = useMemo(() => { try { return toTimeline(S.tracks, T); } catch { return null; } }, [ver]); // eslint-disable-line react-hooks/exhaustive-deps
  const mixHear = { echo: hear.echo, noise: hear.voice, rebuild: hear.voice, leveling: hear.level };
  const cached = session.server?.cached || {};
  const model = { tl, hear: mixHear, cached, secondTrack: S.second };
  live.current = { model, hear, ph, S, session };

  const canMix = done && !!job && !isLocal && !!session.server?.preview;
  const localBlend = done && isLocal && !!fin?.pairParts && S.tracks.voice.length === 1 && S.tracks.voice[0].mode === 1 && S.tracks.voice[0].kind === 'free';
  const localAb = done && isLocal && !!fin?.wavUrl && !localBlend;
  const [abUrl, setAbUrl] = useState('');
  useEffect(() => { setAbUrl(localAb ? fin.wavUrl : ''); }, [localAb, fin?.wavUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    // the engine that plays: the mix of what is made (a job), the kept signals (a card made here), or the file itself
    let eng = null;
    let abEl = null;
    const onEnd = () => { setPlaying(false); setPh((p) => p); };
    const el = mediaRef.current;
    try {
      if (canMix) {
        eng = new MixPlayer({ jobId: job.id, preview: session.server.preview, getModel: () => live.current.model, onState: setPlayInfo, onEnd });
        eng.prepare();
      } else if (localBlend && fin.pairParts) {
        fin.ensurePair().then((pair) => {
          if (engineRef.current !== eng2) return;
          const bp = new BlendPlayer(memorySource(pair), { onEnd, onState: setPlayInfo });
          engineRef.current = bp;
          eng = bp;
          bp.prepare();
          const v0 = live.current.S.tracks.voice[0];
      const g0 = gainsFor(v0 && v0.mode === 1 ? dbOf(v0) : fin.currentStrength());
      bp.setGains(g0.full, g0.input);
          setEngineVer((n) => n + 1);
        }).catch(() => {});
      } else if (localAb && el && abUrl) {
        abEl = new Audio(abUrl);
        abEl.preload = 'auto';
        eng = new AbEngine(el, abEl, { onEnd });
      } else if (el) {
        eng = new ElementEngine(el, { onEnd });
      }
    } catch (e) {
      setPlayInfo({ buffering: false, error: 'Couldn’t get the sound to listen with. Please try again.' });
    }
    const eng2 = eng;
    engineRef.current = eng;
    setPlaying(false);
    return () => { try { (engineRef.current || eng)?.close(); } catch { /* closed */ } engineRef.current = null; abEl?.pause(); };
  }, [canMix, localBlend, localAb, abUrl, job?.id, job?.finishedAt, session.server?.preview, fileUrl, audioUrl]); // eslint-disable-line react-hooks/exhaustive-deps
  const [, setEngineVer] = useState(0);

  // the speaker buttons and the tracks change what is heard at once
  useEffect(() => {
    const eng = engineRef.current;
    if (!eng) return;
    if (eng instanceof MixPlayer) eng.remix();
    else if (eng instanceof AbEngine) eng.setHearCleaned(hear.voice);
    else if (eng instanceof BlendPlayer && fin) {
      const v0 = S.tracks.voice[0];
      const db = v0 && v0.mode === 1 ? dbOf(v0) : fin.currentStrength(); // (the strength being tried, not only the one saved)
      const sums = eng.source.sums ? eng.source.sums() : null;
      const g = hear.voice ? gainsFor(db, { sums }) : { full: 0, input: sums ? matchGain(sums, 1) : 1 };
      eng.setGains(g.full, g.input);
    }
  }, [ver, hear, playing]); // eslint-disable-line react-hooks/exhaustive-deps

  // the playhead follows what plays; the picture follows the sound
  useEffect(() => {
    if (!playing) return undefined;
    const t = setInterval(() => {
      const eng = engineRef.current;
      if (!eng) return;
      const tt = eng.time;
      const end = stopAt.current != null ? stopAt.current : T;
      if (tt >= end - 0.02) { eng.pause(); stopAt.current = null; setPlaying(false); setPh(Math.min(end, T)); return; }
      setPh(tt);
      if (mediaRef.current && !(eng instanceof ElementEngine) && !(eng instanceof AbEngine)) syncPicture(mediaRef.current, eng, true);
    }, 100);
    return () => clearInterval(t);
  }, [playing, T]);
  useEffect(() => {
    const el = mediaRef.current;
    const eng = engineRef.current;
    if (el && eng && !(eng instanceof ElementEngine) && !(eng instanceof AbEngine)) { el.muted = true; syncPicture(el, eng, playing); }
  }, [playing]);

  const seek = useCallback((t) => {
    const x = clamp(t, 0, T);
    setPh(x);
    engineRef.current?.seek(x);
    if (mediaRef.current && !(engineRef.current instanceof ElementEngine) && !(engineRef.current instanceof AbEngine)) mediaRef.current.currentTime = x;
  }, [T]);
  const togglePlay = useCallback(async (from = null, stop = null) => {
    const eng = engineRef.current;
    if (!eng) return;
    if (from != null) { await eng.seek(from); setPh(from); }
    stopAt.current = stop;
    if (eng.playing && from == null) { eng.pause(); setPlaying(false); return; }
    setPlaying(true);
    try { await eng.play(); } catch { setPlaying(false); }
  }, []);

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
      import('../../editor/peaks.js').then(({ pcmPeaks }) => pcmPeaks(fin.pcm, session.wave.data.top)).then((p) => { if (alive) setLocalPeaks(p); }).catch(() => {});
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
    if (job && !isLocal && job.status === 'done' && !saving) session.loadJob({ force: !!session.server });
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
    s = clamp(Math.round(s), 0, T); e = clamp(Math.round(e), 0, T);
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

  const onPrimary = async () => {
    if (stage === 'ready') {
      if (!canStart) return;
      const res = await session.start({ choice, plan, refresh });
      if (res?.id) { session.touch(); nav(jobPath(res.id), { state: session.welcome ? { submitted: session.welcome } : null }); }
      else if (session.canceled) { session.canceled = false; say('Canceled. You weren’t charged.'); }
      return;
    }
    if (done) {
      if (changed) {
        const w = wantsOf({ tracks: S.tracks, plan, choice, out: session.out, file: session.file, local: isLocal });
        if (isLocal && changedTracks) {
          // a card made here: the new tracks run again from the file
          const res = await session.start({ choice, plan, refresh });
          if (res?.id) nav(jobPath(res.id));
          return;
        }
        const keep = fin.effective();
        const wants = { ...keep, ...w, soundHere: isLocal ? keep.soundHere : false, subsHere: isLocal ? keep.subsHere : false };
        const db = reblendDb();
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
    if (b.getAttribute('aria-disabled') === 'true') {
      const why = b.dataset.tip;
      if (why && !['hear'].includes(a)) say(why.length > 160 ? why.slice(-160) : why);
      return;
    }
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

  // ---- the pointer: drag to choose, handles, long-press, the sheet's header ----
  const suppress = useRef(false);
  const lpFired = useRef(false);
  useEffect(() => {
    let drag = null;
    let sheetDrag = null;
    let lp = null;
    const noClick = () => { suppress.current = true; setTimeout(() => { suppress.current = false; }, 0); };
    const rawT = (x) => { const r = waveRef.current.getBoundingClientRect(); return clamp(((x - r.left) / r.width) * T, 0, T); };
    const edgeT = (t) => (t < T * 0.01 ? 0 : t > T * 0.99 ? T : Math.round(t));
    const L = () => live.current;
    const down = (ev) => {
      lpFired.current = false;
      clearTimeout(lp);
      const root = rootRef.current;
      if (!root || !root.contains(ev.target)) return;
      const ih = ev.target.closest('.ih');
      if (ih && L().mobile) { sheetDrag = { y: ev.clientY, moved: false }; return; }
      if (L().processing) return;
      const Sx = L().S;
      const h = ev.target.closest('.hnd');
      if (h && (!Sx.sel || Sx.sel.length === 1)) {
        if (!Sx.sel) Sx.sel = [{ s: 0, e: T }];
        drag = { mode: h.dataset.h };
        h.setPointerCapture(ev.pointerId);
        ev.preventDefault();
        return;
      }
      const sc = ev.target.closest('.sec');
      if (ev.pointerType === 'touch') {
        if (sc) {
          const s0 = +sc.dataset.s;
          const e0 = +sc.dataset.e;
          lp = setTimeout(() => { lpFired.current = true; L().setMore(true); toggleRange(Sx, s0, e0); L().session.touch(); }, 450);
        }
        return;
      }
      const area = ev.target.closest('.main');
      if (!area || ev.target.closest('.top')) return;
      const btn = ev.target.closest('button');
      if (btn && !btn.classList.contains('sec')) return;
      drag = { mode: 'new', x0: ev.clientX, t0: edgeT(rawT(ev.clientX)), moved: false, trk: area, pid: ev.pointerId, add: ev.shiftKey, base: Sx.sel ? Sx.sel.map((r) => ({ s: r.s, e: r.e })) : [] };
    };
    const move = (ev) => {
      if (lp && Math.abs(ev.movementX) + Math.abs(ev.movementY) > 6) clearTimeout(lp);
      if (sheetDrag) {
        const dy = ev.clientY - sheetDrag.y;
        if (Math.abs(dy) > 22 && !sheetDrag.moved) { sheetDrag.moved = true; L().setSheet(dy < 0); noClick(); }
        return;
      }
      if (!drag) return;
      const Sx = L().S;
      const t = edgeT(rawT(ev.clientX));
      if (drag.mode === 's' || drag.mode === 'e') {
        const r = Sx.sel[0];
        if (drag.mode === 's') r.s = clamp(t, 0, r.e - 1); else r.e = clamp(t, r.s + 1, T);
      } else {
        if (!drag.moved) {
          if (Math.abs(ev.clientX - drag.x0) < 4) return;
          drag.moved = true;
          try { drag.trk.setPointerCapture(drag.pid); } catch { /* ignore */ }
        }
        let s = Math.min(drag.t0, t);
        let e = Math.max(drag.t0, t);
        if (e - s < 1) { if (s + 1 <= T) e = s + 1; else s = e - 1; }
        setSel(Sx, (drag.add ? drag.base : []).concat([{ s, e }]));
      }
      L().session.touch();
    };
    const end = () => {
      clearTimeout(lp);
      if (sheetDrag) { sheetDrag = null; return; }
      if (!drag) return;
      const moved = drag.mode !== 'new' || drag.moved;
      drag = null;
      if (moved) {
        noClick();
        const Sx = L().S;
        setSel(Sx, Sx.sel || []);
        if (Sx.sel && L().mobile) L().setSheet(true);
        L().session.touch();
      }
    };
    const click = (ev) => {
      const wf = ev.target.closest('#wf');
      if (wf && rootRef.current?.contains(wf) && !suppress.current && !ev.target.closest('.hnd')) L().seek(rawT(ev.clientX));
    };
    const key = (ev) => {
      const tg = ev.target;
      const typing = tg && (tg.tagName === 'INPUT' || tg.tagName === 'SELECT' || tg.tagName === 'TEXTAREA');
      const Lx = L();
      if (ev.key === 'Escape') {
        if (Lx.modal) return;
        if (Lx.pop) { Lx.setPop(null); } else if (Lx.S.sel || Lx.more) { Lx.S.sel = null; Lx.setMore(false); Lx.session.touch(); }
        return;
      }
      if (typing || !rootRef.current?.contains(tg) && tg !== document.body) return;
      if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 'z') { ev.preventDefault(); if (ev.shiftKey) Lx.session.redo(); else Lx.session.undo(); return; }
      if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 'y') { ev.preventDefault(); Lx.session.redo(); return; }
      if (ev.key === ' ' && tg.tagName !== 'BUTTON') { ev.preventDefault(); Lx.togglePlay(); }
      const h = tg.closest && tg.closest('.hnd');
      if (h && (!Lx.S.sel || Lx.S.sel.length === 1) && (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight')) {
        ev.preventDefault();
        const d = ev.key === 'ArrowLeft' ? -1 : 1;
        const r = Lx.S.sel ? Lx.S.sel[0] : { s: 0, e: T };
        if (h.dataset.h === 's') r.s = clamp(r.s + d, 0, r.e - 1); else r.e = clamp(r.e + d, r.s + 1, T);
        setSel(Lx.S, [r]);
        Lx.session.touch();
      }
    };
    const outside = (ev) => { if (L().pop && !ev.target.closest('.pop, .bb')) L().setPop(null); };
    document.addEventListener('pointerdown', down);
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', end);
    document.addEventListener('pointercancel', end);
    document.addEventListener('click', click);
    document.addEventListener('click', outside);
    document.addEventListener('keydown', key);
    return () => {
      clearTimeout(lp);
      document.removeEventListener('pointerdown', down);
      document.removeEventListener('pointermove', move);
      document.removeEventListener('pointerup', end);
      document.removeEventListener('pointercancel', end);
      document.removeEventListener('click', click);
      document.removeEventListener('click', outside);
      document.removeEventListener('keydown', key);
    };
  }, [T]); // eslint-disable-line react-hooks/exhaustive-deps
  Object.assign(live.current, { mobile, processing, setMore, setSheet, setPop, pop, modal, more, seek, togglePlay, session });

  // the bar's height, for the room under the page
  const barRef = useRef(null);
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

  const R = ranges(S);
  const pend = session.auto ? S.sugg.filter((g) => g.st === 'new').length : 0;
  const pins = session.auto && !processing && !reading ? S.sugg.filter((g) => g.st !== 'dismissed') : [];
  const fileMeta = `${session.lengthUnknown ? 'about ' : ''}${fmt(T)} · ${session.kind === 'video' ? 'Video' : 'Audio'}${session.file ? ` · ${bytes(session.file.size)}` : ''}`;
  const reco = session.reco;

  // processing: what the bar says
  const procLine = useMemo(() => {
    if (!processing) return null;
    if (job && !isLocal) {
      const pct = typeof job.progress === 'number' ? job.progress : 0;
      const step = stageText(job);
      const eta = etaText(etaSeconds(job, Date.now()));
      let text;
      if (saving && (job.status === 'done' || job.status === 'uploading')) text = 'Making the new parts';
      else if (job.status === 'uploading') text = 'Waiting for the upload to finish';
      else if (job.status === 'queued') text = [queueText(job) || 'In line, you’re up soon', eta].filter(Boolean).join(' · ');
      else text = [step ? `${step}${pct ? ` · ${Math.round(pct)}%` : ''}` : pct ? `${Math.round(pct)}% done` : 'Just a moment…', eta].filter(Boolean).join(' · ');
      const value = job.status === 'running' || job.status === 'cloud_asr' ? pct : null;
      const canCancel = job.status === 'uploading' || job.status === 'queued';
      return { text, value, cancel: canCancel ? 'Cancel job' : '' };
    }
    const tasks = Object.entries(session.run?.tasks || {});
    if (!tasks.length) return { text: 'Getting things ready…', value: null, cancel: 'Cancel job' };
    const [, t] = tasks.sort((a, b) => (a[1].frac ?? 0) - (b[1].frac ?? 0))[0];
    const pct = t.frac != null ? Math.floor(t.frac * 100) : null;
    const loading = /ready…$/.test(t.text || '');
    let text = t.text;
    if (/noise remover ready/.test(t.text) && denoiserLoad.total) text = `${t.text} ${(denoiserLoad.loaded / 1048576).toFixed(1)} of ${(denoiserLoad.total / 1048576).toFixed(1)} MB`;
    if (/subtitle maker ready/.test(t.text) && speechLoad.total) text = `${t.text} ${bytes(speechLoad.loaded)} of ${bytes(speechLoad.total)} (only the first time)`;
    return {
      text: [text, !loading && pct != null ? `${pct}%` : '', t.eta ? `about ${human(t.eta)} left` : '', t.rate > 0 ? `${t.rate.toFixed(1)}× real time` : ''].filter(Boolean).join(' · '),
      value: pct,
      cancel: session.run?.cancel === false ? '' : 'Cancel job',
    };
  }, [processing, job, ver, saving, denoiserLoad, speechLoad, now]); // eslint-disable-line react-hooks/exhaustive-deps
  const keepOpen = processing && !job && Object.keys(session.run?.tasks || {}).some((k) => /^(clean|words)/.test(k));
  const building = done && ['waiting', 'building', 'saving'].includes(finState.phase) && !changed;
  const needVideo = done && !!fin && (finState.phase === 'needvideo' || (finState.phase === 'idle' && fin.needsVideo(fin.effective())));

  // the files: what will be made, or what was
  const files = done && fin ? (finState.files.length ? finState.files : fin.planned(fin.effective())) : [];
  const previewText = fin?.subs && !fin.subs.empty ? fin.subs.txt.split(/\n+/).filter(Boolean).slice(0, 3).join(' ') : '';
  const expiresIn = job?.expiresAt ? job.expiresAt - now / 1000 : null;

  const willDownload = (() => {
    const parts = [];
    if (plan.cleaning) parts.push(`${choice.name.trim()}${plan.embedNow ? ' (with the subtitles inside)' : ''}`);
    else if (plan.embedNow) parts.push(`${(session.fileName || 'audio').replace(/\.[^.]+$/, '')}-subtitled.mkv (the video with the subtitles inside)`);
    if (plan.wantsText) {
      const f = [['srt', 'SRT'], ['vtt', 'VTT'], ['txt', 'TXT']].filter(([k]) => session.out.sfiles.includes(k)).map(([, l]) => l);
      if (f.length) parts.push(`the subtitles (${f.join(', ')})`);
    }
    return parts.join(' and ');
  })();
  const tooBigNote = plan.wantsText && plan.videoResult && plan.tooBigHere ? `This browser can’t rebuild a video this big (over ${Math.round((buildLimit() / 1024 ** 3) * 10) / 10} GB); Chrome, Edge or Firefox can. You’ll get the subtitle files to load in your player or video editor.` : '';
  const mkvNote = !plan.cleaning && plan.embedFormat && plan.embedFormat !== 'mkv' && plan.embedFormat !== 'webm' ? plan.embedFormat.toUpperCase() : '';

  // the long-recording estimate of the Free cleanup
  const loadLeft = denoiserLoad.done ? 0 : denoiserLoad.secondsLeft;
  const eta = (() => {
    if (!info?.duration) return null;
    const channels = info.channels && info.channels >= 2 ? 2 : 1;
    const cores = navigator.hardwareConcurrency || 2;
    return info.duration * 0.13 * (channels === 2 && cores < 3 ? 2 : 1) + (loadLeft || 0);
  })();

  const bar = (() => {
    if (stage === 'ready') {
      return {
        mid: { kind: 'details' }, building: false,
        button: { label: estMicros != null || !server ? `Start · ${money(estMicros ?? 0)}` : 'Start', testid: 'start', state: 'start' },
      };
    }
    if (processing) {
      const lbl = (saving ? 'Saving' : 'Processing') + (procLine?.value != null ? ` · ${Math.round(procLine.value)}%` : '');
      return { mid: { kind: 'progress', ...procLine }, building: false, button: { label: lbl, testid: 'processing', state: 'processing' } };
    }
    if (building) {
      const p = Math.round((finState.progress || 0) * 100);
      return {
        mid: { kind: 'progress', text: finState.phase === 'waiting' ? 'Waiting for your browser to finish…' : 'Putting it together…', value: finState.phase === 'waiting' ? null : p, cancel: finState.phase === 'building' ? 'Cancel' : '' },
        building: true, button: { label: finState.phase === 'saving' ? 'Downloading…' : `Processing · ${p}%`, testid: 'processing', state: 'building' },
      };
    }
    const label = changed ? `Save changes · ${money(deltaMicros)}` : finState.phase === 'error' ? 'Try again' : needVideo ? 'Choose the original video' : 'Download again';
    return { mid: { kind: 'files' }, building: false, button: { label, testid: changed ? 'save-changes' : 'download-again', state: changed ? 'save' : 'download' } };
  })();
  const primaryDisabled = stage === 'ready' ? !canStart : processing || building ? true
    : changed ? (!choice.valid && plan.cleaning && !locked ? true : (changedTracks && (quote.state !== 'ok' && !isLocal ? true : !affordableSave)) || (isLocal && changedTracks && !hasFileHere) || (plan.wantsText && !session.out.sfiles.length))
      : false;

  // notes at the top
  const notes = [];
  if (session.error) notes.push({ key: 'error', cls: session.errorKind === 'storage' ? 'banner banner-warn' : 'banner banner-error', text: session.error, testid: 'job-error' });
  if (stage === 'ready') problems.forEach((p) => notes.push({ key: p.text, cls: 'banner banner-warn', text: p.text, kind: p.kind }));
  if (stage === 'ready' && session.lengthUnknown) notes.push({ key: 'len', cls: 'hint', text: 'Your browser can’t read the length of this file type, so the numbers here are a rough guess from its size. We measure it after the upload and you pay for the real length.', testid: 'length-unknown' });
  if (keepOpen) notes.push({ key: 'open', cls: 'hint', text: 'Keep this tab open while it works.', testid: 'keep-open' });
  if (welcome && job && !isLocal) notes.push({ key: 'welcome', cls: 'banner banner-success', text: welcome, testid: 'submitted' });
  if (job?.notice) notes.push({ key: 'notice', cls: 'banner banner-info small', text: job.notice, testid: 'job-notice' });
  if (done && finState.error) notes.push({ key: 'fin', cls: 'banner banner-error small', text: finState.error, testid: 'made-error' });
  if (done && needVideo && !finState.error) notes.push({ key: 'nv', cls: 'hint', text: 'To put the picture back, choose your original video once. It stays on your device.', testid: 'need-video' });
  if (done && fin?.pcm && fin.pcm.rms < 1e-5) notes.push({ key: 'silent', cls: 'banner banner-warn', text: 'The cleaned audio is nearly silent. The original may only contain noise, or try a gentler setting.' });
  if (session.serverError) notes.push({ key: 'serr', cls: 'banner banner-error', text: session.serverError, testid: 'editor-error' });
  if (playInfo.error) notes.push({ key: 'perr', cls: 'banner banner-error small', text: playInfo.error, testid: 'play-error' });
  if (done && changedTracks && quote.state === 'error') notes.push({ key: 'qerr', cls: 'banner banner-warn small', text: quote.error, testid: 'quote-error' });
  if (done && changedTracks && quote.state === 'ok' && !affordableSave) notes.push({ key: 'poor', cls: 'banner banner-warn small', text: `This costs about ${money(quote.q.estCostMicros)}, but your balance is ${balanceMoney(quote.q.availableMicros)}.`, kind: 'credit' });

  // ---- the header lines under the file name ----
  const autoLine = session.listening
    ? <span data-testid="reco-listening"><span className="spinner" aria-hidden="true" /> Listening to a little of your file to pick the best settings…</span>
    : session.applied && reco
      ? (
        <>
          <span className="auto-text"><b data-testid="reco-status">{session.custom ? 'Set automatically, then changed by you.' : 'Set automatically for this file.'}</b></span>
          {session.custom
            ? <button type="button" className="link" data-act="auto-again" data-testid="reco-apply">Use them again</button>
            : <button type="button" className="link" data-act="auto-undo" data-testid="reco-undo">Undo</button>}
        </>
      )
      : null;
  const sgLine = session.sg.phase === 'running'
    ? (
      <span className="autol" data-testid="sg-progress" role="status">
        <span className="spinner" aria-hidden="true" /> Listening to your file for moments worth a look… {Math.round(session.sg.progress * 100)}% <button type="button" className="link" data-act="sg-cancel" data-testid="sg-cancel">Cancel</button>
      </span>
    )
    : session.auto && session.sg.phase === 'unavailable' && !job ? <span className="autol muted" data-testid="sg-unavailable">{session.sg.why}</span>
      : session.sg.phase === 'canceled' ? <span className="autol muted" data-testid="sg-canceled">Stopped looking. You can still choose parts yourself.</span>
        : session.sg.phase === 'error' ? <span className="autol muted" data-testid="sg-error">Couldn’t look for suggestions in this file. You can still choose parts yourself.</span> : null;

  const ctx = {
    S, session, tips, stage, sheet, mobile, info, verdict, sVerdict, rebuildPaused, locked, lockedWhy, eta, loadLeft, ph, zoom, pins, hear, hearText, heard, data,
    scrollRef, waveRef, reading, prog: processing ? (procLine?.value ?? 0) : 0, cost: c, total: estMicros, rates: rates || { std: 0, ext: 0 }, deltaMicros, changed: changedTracks, serverTotal: !server ? 0 : est.state === 'ok' && multi ? est.micros : null,
    partsNote: multi && est.state === 'ok', pop, choice, plan, out: session.out, files, preview: previewText, job, expiresIn, delAsk, paid: job?.costMicros || session.paid || 0, finState, fin, bar, needVideo,
    disabled: processing, willDownload, tooBigNote, mkvNote, primaryDisabled,
    h: { onDb, slideStart, slideEnd, onVideo: (f) => fin?.setVideo(f) },
  };

  return (
    <div
      className="studio" ref={rootRef} data-testid="studio" data-stage={stage} data-phase={done ? finState.phase : undefined} onClick={onClick} onChange={onChange}
      {...(done && fin?.pcm ? { 'data-rms': fin.pcm.rms.toFixed(6), 'data-peak': fin.pcm.peak.toFixed(4), 'data-frames': fin.pcm.frames, 'data-channels': fin.pcm.channels } : {})}
      {...(done && fin?.subs ? { 'data-cues': fin.subs.cues.length, 'data-language': fin.subs.language } : {})}
      {...(done && finState.strengthDb != null ? { 'data-strength-db': finState.strengthDb.toFixed(2) } : {})}
      {...(done && gains ? { 'data-gains': gains } : {})}
      data-time={ph.toFixed(2)} data-playing={playing ? 'yes' : 'no'} data-seconds={T} data-changed={done ? (changedTracks ? 'tracks' : changedOut ? 'output' : 'no') : undefined}
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
            <div className="top" id="top">
              <div className={`vid${info?.hasVideo ? '' : ' aud'}`}>
                {info?.hasVideo && fileUrl ? <video ref={mediaRef} src={fileUrl} playsInline preload="auto" muted={done && !localAb} data-testid="video" /> : null}
                {!info?.hasVideo || !fileUrl ? <span data-testid="poster">{session.kind === 'video' ? 'Video' : 'Audio'}</span> : null}
                {info?.hasVideo && <span className="vt js-ph">{fmt(ph)}</span>}
                {!info?.hasVideo && audioUrl && <audio ref={mediaRef} src={audioUrl} preload="auto" />}
              </div>
              <div className="tinfo">
                <div className="tname">
                  <h1 className="fname" data-testid="file-name">{session.fileName}</h1>
                  <p className="fmeta" data-testid="file-meta">{fileMeta} · <button type="button" className="link" data-act="newfile" data-testid="different-file">Different file</button></p>
                </div>
                <div className="autoc">
                  <div className="autow" data-testid="auto-box">
                    <label className="auto" data-testid="auto-settings">
                      <button type="button" className="sw" role="switch" aria-checked={session.auto} aria-labelledby="auto-l" data-act="auto" disabled={!session.hasAudio} />
                      <span id="auto-l">Automatic</span>
                    </label>
                    {pend > 0 && (
                      <button type="button" className="chip lk-b" data-act="sgchip" aria-label={`${pend} suggestions. Show the next one`} data-testid="sg-chip">{pend}<IcSpark /></button>
                    )}
                  </div>
                  {session.auto && autoLine && <p className="autol" aria-live="polite" data-testid="reco">{autoLine}</p>}
                  {session.auto && session.applied && reco && (
                    <details className="auto-why" data-testid="reco-details">
                      <summary>Why these choices</summary>
                      <p className="muted small mb0"><span data-testid="reco-line">{reco.line}</span> <span data-testid="reco-pick">{reco.pick}</span></p>
                      {reco.reasons?.length > 0 && <ul className="reco-why small muted" data-testid="reco-why">{reco.reasons.map((r) => <li key={r}>{r}</li>)}</ul>}
                    </details>
                  )}
                  {sgLine}
                </div>
              </div>
              <div className="tools">
                <button type="button" className="btn btn-primary tp o0" data-act="play" aria-label={playing ? 'Pause' : 'Play'} data-testid="play" disabled={!engineRef.current && !mediaRef.current}>{playing ? <IcPause /> : <IcPlay />}</button>
                <span className="tt o0"><span className="js-ph" data-testid="time">{fmt(ph)}</span><span className="lg"> / {fmt(T)}</span></span>
                <button type="button" className="btn btn-secondary btn-sm tb lk-b o3" data-act="split" data-testid="split"><IcSplit /><span className="tl"><span className="lg">Split at playhead</span><span className="sm">Split</span></span></button>
                <button type="button" className="btn btn-secondary btn-sm tb lk-b o1" data-act="undo" aria-label="Undo" title="Undo" disabled={!S.past.length} data-testid="undo"><IcUndo /></button>
                <button type="button" className="btn btn-secondary btn-sm tb lk-b o1" data-act="redo" aria-label="Redo" title="Redo" disabled={!S.future.length} data-testid="redo"><IcRedo /></button>
                <button type="button" className="btn btn-secondary btn-sm tb o1" data-act="zoom-out" aria-label="Zoom out" title="Zoom out" disabled={zoom <= 1} data-testid="zoom-out"><IcZoomOut /></button>
                <button type="button" className="btn btn-secondary btn-sm tb o1" data-act="zoom-in" aria-label="Zoom in" title="Zoom in" disabled={zoom >= 8} data-testid="zoom-in"><IcZoomIn /></button>
                <button type="button" className="btn btn-secondary btn-sm tb lk-b o2 push" data-act="all" aria-pressed={!S.sel ? 'true' : undefined} data-testid="select-all"><IcAll /><span className="tl"><span className="lg">Select all</span><span className="sm">All</span></span></button>
                <span className="brk" />
                <button type="button" className="btn btn-secondary btn-sm tb lk-b o3 tb-more" data-act="more" aria-pressed={more ? 'true' : undefined} data-testid="select-more"><IcMore /><span className="tl">Select more</span></button>
              </div>
            </div>
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
