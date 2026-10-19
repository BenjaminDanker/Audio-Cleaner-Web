import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { Link, useNavigate } from 'react-router';
import { GoodToKnow } from '../alternatives.jsx';
import { useAuth } from '../auth.jsx';
import { addLocalCard, beginFinish, finishSeconds, getFinish } from '../finish.js';
import { balanceMoney, BILLING_NOTE, bytes, clock, human, money, rate, shownName, stem } from '../format.js';
import { estimateMicros, findRate, loadPricing, usePricing } from '../hooks.js';
import { highsHint, rebuildHint } from '../hints.js';
import { jobPath, LANGUAGES, LOUDNESS, NOISE, submitJob, tooManyText } from '../jobs.js';
import { buildLimit, canHoldSubtitles, defaultFormatId, serverAudioFormat } from '../media/convert.js';
import { previewFits } from '../media/blend.js';
import { denoiserLoadState, preloadDenoiser, subscribeDenoiserLoad } from '../media/denoise.js';
import { analyseFile } from '../media/analyse.js';
import { probeMedia } from '../media/mb.js';
import { denoiseFile } from '../media/pipeline.js';
import { preloadSpeech, speechLoadState, subscribeSpeechLoad, subtitleFiles, transcribeFile } from '../media/speech.js';
import { buildWav, wavFits } from '../media/wav.js';
import { FinishProgress, ProgressPanel, SubtitlesProgress } from './Progress.jsx';
import OutputFields, { useOutputChoice } from './OutputPicker.jsx';
import FileDrop, { SERVER_FORMATS } from './FileDrop.jsx';
import StrengthPicker from './StrengthPicker.jsx';
import { strengthDb } from '../strengths.js';
import { recommendForFile } from '../recommend.js';


const LONG_SECONDS = 2 * 3600; // in the browser, a recording this long takes a while
const AUTO_KEY = 'acw.autoSettings'; // this browser's choice for "Automatic best settings" (on unless switched off)

function storedAuto() {
  try {
    return localStorage.getItem(AUTO_KEY) !== 'off';
  } catch {
    return true;
  }
}
const MAX_SECONDS = 6 * 3600; // and this is the most the browser will take

// The most a single upload can be, as the server's default allows it (the price list may say otherwise: maxUploadBytes).
const DEFAULT_MAX_UPLOAD_BYTES = 4 * 1024 ** 3;
const gigabytes = (n) => `${Math.round((n / 1024 ** 3) * 10) / 10} GB`;

/** A file that cannot hold any sound: nothing to read, nothing to clean. */
const TINY_FILE_BYTES = 100;

const supported = typeof Worker !== 'undefined' && typeof WebAssembly !== 'undefined';

/** Start fetching the noise remover (about 22 MB, once, low priority) once the free in-browser cleanup is on the table for a file.
 *  Not for visitors whose browser asks to save data: for them it is fetched when cleaning starts. A failure is reported then. */
function warmUpDenoiser() {
  if (!supported || navigator.connection?.saveData) return;
  preloadDenoiser().catch(() => {});
}

/** "Files must be under 4096 MB." -> "Files must be under 4 GB." */
const inGigabytes = (text) => text.replace(/\b(\d{4,})\s*MB\b/g, (_, n) => gigabytes(Number(n) * 1024 ** 2));

const compressedNote = (c) => `Compressed for upload: ${bytes(c.from)} → ${bytes(c.to)}`;

/** "Got it, we're on it." plus what was done to the file before it went up (shown once, at the top of the job's page). */
function startedNote(out) {
  if (!out) return 'Got it.';
  return `Got it, we’re on it.${out.extracted ? ` Only the sound was uploaded (${bytes(out.uploadBytes)} instead of ${bytes(out.originalBytes)}).` : ''}${out.compressed ? ` ${compressedNote(out.compressed)}.` : ''}`;
}


/** Start fetching the subtitle maker (about 80 MB, once; the browser keeps it) as soon as free subtitles are chosen. */
function warmUpSpeech() {
  if (!supported || navigator.connection?.saveData) return;
  preloadSpeech().catch(() => {});
}

/** Can this browser make the subtitles for this file itself? */
function subsVerdict(info) {
  const paid = 'Standard can (paid).';
  if (!supported) return { ok: false, why: `This browser can’t run the subtitle maker. ${paid}` };
  if (info.hasAudio === false) return { ok: false, why: 'This file has no audio track.' };
  const readable = info.canDecode !== false && (info.viaMediabunny || info.viaLibav || info.duration != null);
  if (!readable) return { ok: false, why: `Your browser can’t open this file type. ${paid}` };
  if (info.duration && info.duration > MAX_SECONDS) return { ok: false, why: `This is longer than ${MAX_SECONDS / 3600} hours, which is too long for your browser. ${paid}` };
  return { ok: true, why: '' };
}

/** Can this browser do the noise removal for this file? `why` is the plain-words reason when it can't. */
function browserVerdict(info) {
  const paid = 'Paid Denoising can do it.';
  if (!supported) return { ok: false, why: `This browser can’t run the noise remover. ${paid}` };
  if (info.hasAudio === false) return { ok: false, why: 'This file has no audio track.' };
  const readable = info.canDecode !== false && (info.viaMediabunny || info.viaLibav || info.duration != null);
  if (!readable) return { ok: false, why: `Your browser can’t open this file type. ${paid}` };
  if (info.duration && info.duration > MAX_SECONDS) return { ok: false, why: `This is longer than ${MAX_SECONDS / 3600} hours, which is too long for your browser. ${paid}` };
  // The cleaned sound is offered as a WAV file, and a WAV file can't hold more than about 4 GB: say so now, not after hours of work.
  if (info.duration && !wavFits(info.duration, info.channels >= 2 ? 2 : 1, 16)) return { ok: false, why: `The cleaned sound would be too big for one WAV file (4 GB). ${paid}` };
  return { ok: true, why: '' };
}

/** One radio row of a list. A row that can't be used stays visible, greyed out, with the reason where its hint would be. */
function Choice({ name, value, current, onPick, disabled, title, hint, reason }) {
  return (
    <label className={`choice${disabled ? ' is-disabled' : ''}`}>
      <input type="radio" name={name} value={value} checked={current === value} onChange={() => onPick(value)} disabled={disabled} />
      <span><b>{title}</b><small>{disabled && reason ? reason : hint}</small></span>
    </label>
  );
}

/** One step of the form: a title (and a line under it), then what belongs to it. */
function Step({ title, lead, children, testId }) {
  return (
    <section className="step" aria-label={title} data-testid={testId}>
      <h3 className="step-h">{title}</h3>
      {lead && <p className="hint step-lead">{lead}</p>}
      {children}
    </section>
  );
}

/** A tick box with its explanation under it (every on/off choice looks like this). */
function Toggle({ name, checked, onChange, disabled, title, hint, hintId, muted }) {
  return (
    <label className={`check${muted ? ' muted' : ''}`}>
      <input type="checkbox" name={name} checked={checked} onChange={onChange} disabled={disabled} />
      <span><b>{title}</b>{hint && <span className="hint block" data-testid={hintId}>{hint}</span>}</span>
    </label>
  );
}

/**
 * The Cleanup form: drop one file, then choose what to do with it. Only what is possible for that file is offered.
 *   Noise removal:  Off | Free, in your browser (nothing uploaded) | Denoising (paid, on our side)
 *   Subtitles:      Off | Free, in your browser (nothing uploaded) | Standard (paid) | Extra accurate (paid)
 * Any combination works: the browser part runs here, in parallel with the job (if any) for the rest.
 * onSubmitted({id, ...}) once a job on our side exists (its page finishes it: see finish.js). When everything is started (and the
 * parts made here are done) the page of the job opens: /app/job/<id>. `active`: how many files are being processed on our side
 * and `maxActive` the most there may be at once (the server says the same, with the code too_many_active_jobs).
 */
export default function JobForm({ onSubmitted, active = 0, maxActive = null }) {
  const nav = useNavigate();
  const { balance, refresh } = useAuth();
  const { data: pricing, error: pricingError } = usePricing();
  const load = useSyncExternalStore(subscribeDenoiserLoad, denoiserLoadState);
  const speechLoad = useSyncExternalStore(subscribeSpeechLoad, speechLoadState);

  const [file, setFile] = useState(null);
  const [info, setInfo] = useState(null);
  const [probing, setProbing] = useState(false);
  const [noise, setNoise] = useState('denoise'); // 'off' | 'browser' | 'denoise'
  const [subs, setSubs] = useState('standard'); // 'off' | 'browser' | 'standard' | 'accurate'
  const [embed, setEmbed] = useState(false);
  const [echo, setEcho] = useState(false); // Denoising: remove speaker echo using the second audio track
  const [loudness, setLoudness] = useState('-14'); // the YouTube target
  const [noiseKind, setNoiseKind] = useState('auto');
  const [leveling, setLeveling] = useState(true); // Leveling (paid, on our side): even out the voice, set the loudness. With noise removal Off it runs on its own.
  const [restoreHighs, setRestoreHighs] = useState(false); // Restore highs (paid, off unless asked for): bring back the high sounds a recording lost; with Denoising or on its own
  const [rebuild, setRebuild] = useState(false); // Voice rebuild (its own price)
  const [strength, setStrength] = useState('max'); // Denoising
  const [localStrength, setLocalStrength] = useState('max'); // in the browser
  const [caps, setCaps] = useState({ srt: true, vtt: true, txt: true });
  const [language, setLanguage] = useState('');
  const [targets, setTargets] = useState(['', '', '']); // translations (made on our side, with Standard or Extra accurate)
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState(null); // upload: {phase, frac, loaded, total}
  const [localRunning, setLocalRunning] = useState(false);
  const [localProg, setLocalProg] = useState(null);
  const [localStage, setLocalStage] = useState('');
  const [finProg, setFinProg] = useState(null); // the last step (putting the files together), while it runs here: 0 to 1
  const [subsRunning, setSubsRunning] = useState(false);
  const [subsProg, setSubsProg] = useState(null);
  const [compressedInfo, setCompressedInfo] = useState(null); // {from, to} once the input was encoded to FLAC
  const [stats, setStats] = useState(null); // what the quick listen to the file found (media/analysis.js); null: not (yet) measured
  const [listening, setListening] = useState(false);
  const [auto, setAuto] = useState(storedAuto); // "Automatic best settings": listen to the file and pick the settings for it
  const autoRef = useRef(auto);
  autoRef.current = auto;
  const manual = useRef(new Set()); // the settings the person changed themselves: what the quick listen finds leaves them alone
  const [custom, setCustom] = useState(false); // ... the same, for showing the line that goes back to the automatic choices
  const [applied, setApplied] = useState(false); // the automatic choices were set for this file
  const appliedRef = useRef(false);
  const undoTo = useRef(null); // the choices from before they were set automatically
  const now = useRef({}); // the current choices, for the listen that finishes later
  const rebuildOk = useRef(false); // is the rebuild offered (the price list has it)?
  const listen = useRef(null);
  const [error, setError] = useState('');
  const choice = useOutputChoice({ file, info, serverAudio: noise === 'denoise' || (noise === 'off' && (leveling || restoreHighs)) });
  const [errorKind, setErrorKind] = useState(''); // 'storage' while the server is short on space
  const abort = useRef(null);

  useEffect(() => () => { abort.current?.abort(); listen.current?.abort(); }, []);
  useEffect(() => {
    if (!busy) return undefined;
    const warn = (e) => { e.preventDefault(); e.returnValue = ''; };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [busy]);

  const verdict = info ? browserVerdict(info) : null;
  const sVerdict = info ? subsVerdict(info) : null;
  const wantsBrowser = noise === 'browser';
  const wantsDenoise = noise === 'denoise'; // Denoising (or Voice rebuild) on our side
  const soundOnly = noise === 'off' && (leveling || restoreHighs) && info?.hasAudio !== false; // no noise removal: Leveling and/or Restore highs on their own, on our side
  const serverClean = wantsDenoise || soundOnly; // a cleaned file comes from our side
  const cleaning = serverClean || wantsBrowser; // the cleaned sound is part of the result (the file type and name are chosen)
  const wantsText = subs !== 'off'; // any subtitles: the language, formats and "into the video" choices apply
  const localSubs = subs === 'browser'; // made here, free
  const serverText = wantsText && !localSubs; // made on our side, paid
  const tier = subs === 'accurate' ? 'premium' : 'local'; // transcription level ids stay internal
  const kind = serverClean && serverText ? 'denoise_transcribe' : serverClean ? 'denoise' : serverText ? 'transcribe' : null;
  const translations = serverText ? targets.filter((t) => t && t !== language) : [];
  const isVideo = !!info?.hasVideo;
  const hasSecondTrack = (info?.audioTracks || 0) >= 2; // OBS and others can record mic and desktop sound as two tracks
  // Subtitles in the video: put in here, in the browser, once the job is done (nothing extra is uploaded). Only a video the
  // browser can't open goes up whole, and then our side puts them in. A recording with no picture can get them in the
  // plain black video made here when a video file type is picked.
  const videoResult = isVideo ? !cleaning || choice.isVideoFile : cleaning && choice.isVideoFile;
  const embedOnServer = serverText && isVideo && !info?.viaMediabunny;
  const limit = buildLimit(); // null: this browser writes the result to disk, so any size works
  const tooBigHere = isVideo && !embedOnServer && file && limit != null && file.size > limit;
  // subtitles made here go into a video this browser can rebuild (one it opens itself)
  const canEmbed = wantsText && videoResult && !tooBigHere && (serverText || !isVideo || !!info?.viaMediabunny);
  const embedNow = canEmbed && embed;
  const embedHere = embedNow && !embedOnServer;
  // A subtitles-only video keeps its own file type when that can carry them (WebM), else it becomes MKV.
  const embedFormat = cleaning ? choice.format : isVideo ? defaultFormatId(file, info, true) : null;

  // Some formats (WMA, AMR, AVI, MPEG...) the browser can't read, but the server can. Then the length is a rough
  // guess from the file size (128 kbit/s) until the server measures it; the price is corrected to the real length.
  const lengthUnknown = !!(file && info && !probing && !info.duration && info.hasAudio !== false);
  const maxJob = pricing?.maxJobSeconds;
  const duration = info?.duration || (lengthUnknown ? Math.min(Math.max(30, (file.size * 8) / 128000), maxJob || 4 * 3600) : null);
  const denoiseRate = findRate(pricing, 'denoise', 'standard');
  const rebuildOffered = findRate(pricing, 'denoise', 'rebuild');
  const rebuildRate = pricing?.rebuildAvailable ? rebuildOffered : null; // paused while its machine is unavailable
  const denoiseTier = soundOnly ? 'off' : rebuild && rebuildRate ? 'rebuild' : 'standard';
  const levelingRate = findRate(pricing, 'leveling', 'standard');
  const highsRate = findRate(pricing, 'highs', 'standard');
  const withLeveling = serverClean && leveling; // Leveling is charged on its own line
  const withHighs = serverClean && restoreHighs && denoiseTier !== 'rebuild'; // so is Restore highs (the voice rebuild makes its own)
  const standardRate = findRate(pricing, 'transcribe', 'local');
  const translateRate = findRate(pricing, 'transcribe_translate', 'standard');
  const fileReco = useMemo(() => (stats ? recommendForFile(stats, { language }) : null), [stats, language]);
  const reco = auto ? fileReco : null; // used only with "Automatic best settings" on
  rebuildOk.current = !!rebuildRate;
  now.current = { noise, strength, localStrength, rebuild, restoreHighs, leveling, subs, noiseKind };

  const procEta = useMemo(() => {
    if (!info?.duration) return null;
    const channels = info.channels && info.channels >= 2 ? 2 : 1;
    const cores = navigator.hardwareConcurrency || 2;
    return info.duration * 0.13 * (channels === 2 && cores < 3 ? 2 : 1);
  }, [info]);
  // Whatever is still downloading counts too, so the estimate is honest if you start straight away.
  const loadLeft = load.done ? 0 : load.secondsLeft;
  // ... and so does the last step, putting the files together (the cleaned sound back into the video, the chosen file type)
  const finishEta = wantsBrowser ? finishSeconds(buildWants(), { info, file }) : 0;
  const eta = procEta == null ? null : procEta + (loadLeft || 0) + finishEta;
  const tail = wantsBrowser && procEta && finishEta ? Math.min(0.4, finishEta / (procEta + finishEta)) : 0; // its share of the progress bar

  const est = useMemo(() => {
    if (!kind) return 0;
    return estimateMicros(pricing, { kind, denoiseTier, transcribeTier: tier, seconds: duration, translate: translations.length, leveling: withLeveling, highs: withHighs });
  }, [pricing, kind, tier, duration, denoiseTier, translations.length, withLeveling, withHighs]);
  const breakdown = useMemo(() => {
    if (kind !== 'denoise_transcribe' || !duration) return null;
    return [estimateMicros(pricing, { kind: 'denoise', denoiseTier, seconds: duration, leveling: withLeveling, highs: withHighs }), estimateMicros(pricing, { kind: 'transcribe', transcribeTier: tier, seconds: duration, translate: translations.length })];
  }, [pricing, kind, tier, duration, denoiseTier, translations.length, withLeveling, withHighs]);

  async function choose(f) {
    if (f.size < TINY_FILE_BYTES) {
      // Nothing to read in there: say so instead of offering options with made-up numbers.
      setFile(null);
      setInfo(null);
      setError(f.size === 0 ? 'That file is empty (0 bytes). Please pick another one.' : 'That file is too small to hold any sound. Please pick another one.');
      return;
    }
    setFile(f);
    setInfo(null);
    setError('');
    setEmbed(false);
    setEcho(false);
    // every file starts from the usual choices: Denoising, Standard subtitles, Leveling, no rebuild
    setSubs('standard');
    setNoise('denoise');
    setStrength('max');
    setLocalStrength('max');
    setRebuild(false);
    setRestoreHighs(false);
    setLeveling(true);
    setNoiseKind('auto');
    setStats(null);
    setListening(false);
    manual.current = new Set();
    appliedRef.current = false;
    undoTo.current = null;
    setApplied(false);
    setCustom(false);
    listen.current?.abort();
    setProbing(true);
    const i = await probeMedia(f);
    // Denoising and Standard subtitles are the usual start, with the price right there; a file with no sound has nothing to do.
    const canBrowser = browserVerdict(i).ok;
    if (i.hasAudio === false) {
      setNoise('off');
      setSubs('off');
    }
    setInfo(i);
    setProbing(false);
    // The quick listen needs the noise remover too, so a browser that asks to save data skips it: nothing is downloaded before Start.
    if (i.hasAudio !== false && autoRef.current && !navigator.connection?.saveData) listenTo(f, i, canBrowser);
  }

  /** A quick listen to the file (about a minute of it, spread over the whole), so the settings can start out right. */
  async function listenTo(f, i, canBrowser) {
    const ctl = new AbortController();
    listen.current = ctl;
    setListening(true);
    try {
      const found = await analyseFile(f, i, { signal: ctl.signal });
      if (ctl.signal.aborted) return;
      setStats(found);
      const r = found ? recommendForFile(found, { language: '' }) : null;
      if (r && autoRef.current) applyReco(r, i);
      else if (!r && canBrowser) warmUpDenoiser();
    } catch {
      /* canceled, or unreadable: the form just keeps its usual starting choices */
    } finally {
      if (listen.current === ctl) setListening(false);
    }
  }

  /** Set the best choices for the file, except the ones the person changed themselves (`again`: those too). */
  function applyReco(r, i = info, again = false) {
    if (again) manual.current = new Set();
    const skip = manual.current;
    const canBrowser = i ? browserVerdict(i).ok : false;
    const next = r.noise === 'browser' && !canBrowser ? 'denoise' : r.noise;
    if (!undoTo.current) undoTo.current = { ...now.current };
    if (!skip.has('noise')) setNoise(next);
    const noiseNow = skip.has('noise') ? now.current.noise : next;
    if (noiseNow === 'browser') warmUpDenoiser();
    if (!skip.has('strength')) {
      setStrength(r.strength);
      setLocalStrength(r.strength);
    }
    if (!skip.has('rebuild')) setRebuild(!!(r.rebuild && rebuildOk.current && next === 'denoise'));
    if (!skip.has('highs')) setRestoreHighs(!!r.restoreHighs && !(r.rebuild && rebuildOk.current && next === 'denoise')); // on only where the automatic settings ask for it
    if (!skip.has('leveling')) setLeveling(!!r.leveling);
    if (!skip.has('subs') && i?.hasAudio !== false) setSubs(r.subs || 'off');
    setNoiseKind('auto');
    appliedRef.current = true;
    setApplied(true);
    setCustom(skip.size > 0);
  }

  // another language for the subtitles can change which kind of subtitles is best (unless the person chose)
  useEffect(() => {
    if (auto && fileReco && appliedRef.current && !manual.current.has('subs') && info?.hasAudio !== false) setSubs(fileReco.subs || 'off');
  }, [fileReco]); // eslint-disable-line react-hooks/exhaustive-deps

  function pickSubs(v) {
    touch('subs');
    setSubs(v);
    if (v === 'browser') warmUpSpeech();
  }

  /** "Automatic best settings" on or off (remembered in this browser). Switched on with a file chosen, the file is listened
   * to (if it wasn't yet) and its settings are picked; switched off, the current choices stay and nothing is listened to. */
  function setAutoSettings(on) {
    setAuto(on);
    autoRef.current = on;
    try {
      localStorage.setItem(AUTO_KEY, on ? 'on' : 'off');
    } catch {
      /* not remembered: private window or storage blocked */
    }
    if (!on) {
      listen.current?.abort();
      setListening(false);
      return;
    }
    manual.current = new Set();
    if (!file || !info || info.hasAudio === false) return;
    if (fileReco) applyReco(fileReco, info, true);
    else if (!listening) listenTo(file, info, browserVerdict(info).ok);
  }

  /** "Undo": the choices go back to what they were before they were set automatically, and the automatic setting is switched off. */
  function undoAuto() {
    const u = undoTo.current;
    if (u) {
      setNoise(u.noise); setStrength(u.strength); setLocalStrength(u.localStrength); setRebuild(u.rebuild);
      setRestoreHighs(u.restoreHighs); setLeveling(u.leveling); setSubs(u.subs); setNoiseKind(u.noiseKind);
    }
    undoTo.current = null;
    appliedRef.current = false;
    setApplied(false);
    setCustom(false);
    setAutoSettings(false);
  }

  /** The person chose something themselves: what the quick listen finds later no longer changes it. */
  function touch(key) {
    manual.current.add(key);
    if (appliedRef.current) setCustom(true);
  }

  function pickNoise(v) {
    touch('noise');
    setNoise(v);
    if (v === 'browser') warmUpDenoiser();
  }

  function reset() {
    abort.current?.abort();
    listen.current?.abort();
    setStats(null);
    setListening(false);
    appliedRef.current = false;
    undoTo.current = null;
    setApplied(false);
    setCustom(false);
    setFile(null);
    setInfo(null);
    setError('');
  }

  const problems = [];
  if (info?.hasAudio === false) problems.push('This file has no audio track.');
  if (kind && duration && maxJob && duration > maxJob) problems.push(`Files can be at most ${Math.floor(maxJob / 3600)} hours long.`);
  const maxUpload = pricing?.maxUploadBytes || DEFAULT_MAX_UPLOAD_BYTES;
  // A video's picture is not uploaded and uncompressed WAV/AIFF is packed smaller first; anything else goes up as it is.
  const canShrink = (info?.hasVideo && info?.viaMediabunny) || /^pcm/.test(info?.audioCodec || '');
  if (kind && file && file.size > maxUpload && !canShrink) problems.push(`This file is ${gigabytes(file.size)}. The most we can take for upload is ${gigabytes(maxUpload)}. Cleaning in your browser doesn’t upload the file, so this limit doesn’t apply there.`);
  if (kind && maxActive && active >= maxActive) problems.push(tooManyText(active));
  if (kind && est == null && duration && pricingError) problems.push('We couldn’t load the prices just now, so this can’t start yet.');
  if (kind && est != null && balance && est > balance.available_micros) problems.push(`This costs about ${money(est)}, but your balance is ${balanceMoney(balance.available_micros)}. You can also choose the free options.`);
  if (info && !wantsBrowser && !localSubs && !kind) problems.push('Choose at least one thing to do: noise removal, subtitles, or both.');
  if (wantsText && !Object.values(caps).some(Boolean)) problems.push('Choose at least one caption or transcript format.');
  if (embedHere && cleaning && !canHoldSubtitles(choice.format)) problems.push(`Subtitles put into the video need an MKV file (most players can’t show them from ${choice.format.toUpperCase()}). Pick MKV, or untick “Put the subtitles into the video”.`);
  const canStart = (wantsBrowser || localSubs || kind) && file && info && !probing && !busy && problems.length === 0 && (!kind || est != null) && (!cleaning || choice.valid);

  /** What the person asked for, as it stands now: the files that download by themselves when everything is done. */
  function buildWants(f = file) {
    const w = {
      sound: cleaning, format: choice.format, name: choice.name.trim(), size: choice.size,
      subs: wantsText, caps: { ...caps }, embed: embedNow, soundHere: wantsBrowser, subsHere: localSubs,
    };
    // subtitles into the video with no cleaning: the video keeps its own sound, as an MKV
    if (!cleaning && embedNow) Object.assign(w, { format: 'mkv', size: 'same', name: `${stem(f?.name)}-subtitled.mkv` });
    return w;
  }

  /** "When it's done, these download by themselves: ..." */
  function willDownload() {
    const parts = [];
    if (cleaning) parts.push(`${choice.name.trim()}${embedNow ? ' (with the subtitles inside)' : ''}`);
    else if (embedNow) parts.push(`${stem(file?.name)}-subtitled.mkv (the video with the subtitles inside)`);
    if (wantsText) {
      const f = [['srt', 'SRT'], ['vtt', 'VTT'], ['txt', 'TXT']].filter(([id]) => caps[id]).map(([, l]) => l);
      if (f.length) parts.push(`the subtitles (${f.join(', ')})`);
    }
    return parts.length ? parts.join(' and ') : '';
  }

  async function runLocal(f, i, signal) {
    setLocalRunning(true);
    setLocalProg({ done: 0, total: i?.duration || null, rate: 0, eta: null, elapsed: 0 });
    try {
      const attenDb = strengthDb(localStrength);
      // the full-strength sound and the recording are kept (in memory) while they fit: the strength can be changed afterwards
      const pcm = await denoiseFile(f, {
        attenDb, durationHint: i?.duration || null, signal, onProgress: setLocalProg, onStage: setLocalStage,
        keepPair: !!i?.duration && previewFits(i.duration, i.channels >= 2 ? 2 : 1),
      });
      setLocalStage('Preparing the preview…');
      const wav = await buildWav(pcm, 16);
      if (signal.aborted) throw new DOMException('Canceled', 'AbortError');
      return { pcm, wav, url: URL.createObjectURL(wav), strength: localStrength, attenDb };
    } finally {
      setLocalRunning(false);
    }
  }

  async function runLocalSubs(f, i, signal) {
    setSubsRunning(true);
    setSubsProg({ done: 0, total: i?.duration || null, eta: null, stage: '' });
    try {
      return await transcribeFile(f, {
        language: language || null, durationHint: i?.duration || null, signal,
        onProgress: (p) => setSubsProg((x) => ({ ...x, ...p })),
        onStage: (s) => setSubsProg((x) => ({ ...x, stage: s })),
      });
    } finally {
      setSubsRunning(false);
    }
  }

  async function runJob(f, i, signal, entry) {
    setStep({ phase: 'create', frac: 0 });
    const body = {
      kind,
      denoiseTier: serverClean ? denoiseTier : null,
      transcribeTier: serverText ? tier : null,
      durationSeconds: duration || 1,
      options: {
        audio_format: serverAudioFormat(choice.format, f, i),
        // putting subtitles into the video here needs a subtitle file, even when only the transcript was ticked
        caption_formats: serverText ? ['srt', 'vtt', 'txt'].filter((c) => caps[c] || (c === 'vtt' && embedHere && !caps.srt)) : [],
        embed_subtitles: !!(embedNow && embedOnServer),
        language: language || null,
        translate: translations,
        loudness_lufs: LOUDNESS.find((l) => l.id === loudness)?.value ?? null,
        noise: noiseKind,
        leveling,
        restore_highs: restoreHighs && denoiseTier !== 'rebuild',
        atten_lim_db: strengthDb(strength),
        echo_reference_track: !!(wantsDenoise && echo && hasSecondTrack),
      },
    };
    try {
      const out = await submitJob({
        file: f, info: i, body, signal,
        onStage: (s) => setStep((p) => ({ ...p, phase: s, frac: 0 })),
        onProgress: (p) => setStep(p),
        onCompressed: setCompressedInfo,
      });
      beginFinish({ ...out, displayName: f.name, outputs: [] }, entry); // from here on the job's card finishes it by itself
      onSubmitted?.(out);
      return out;
    } finally {
      setStep(null);
    }
  }

  async function submit(e) {
    e.preventDefault();
    if (!canStart) return;
    listen.current?.abort(); // whatever the quick listen finds from here on must not change a run that has started
    manual.current = new Set(['noise', 'strength', 'rebuild', 'highs', 'leveling', 'subs']);
    const f = file;
    const i = info;
    const ctl = new AbortController();
    abort.current = ctl;
    setBusy(true);
    setError('');
    setErrorKind('');
    setCompressedInfo(null);
    const wants = buildWants(f);
    let doneBrowser;
    const entry = { file: f, info: i, wants, browser: new Promise((r) => { doneBrowser = r; }) };

    // The browser parts and the job (if there is one) run side by side; each reports on its own.
    const outcomes = await Promise.allSettled([
      wantsBrowser ? runLocal(f, i, ctl.signal) : Promise.resolve(null),
      localSubs ? runLocalSubs(f, i, ctl.signal) : Promise.resolve(null),
      kind ? runJob(f, i, ctl.signal, entry) : Promise.resolve(null),
    ]);
    const [local, subsOut, job] = outcomes;
    const canceled = ctl.signal.aborted || outcomes.some((o) => o.status === 'rejected' && o.reason?.name === 'AbortError');
    const subsFiles = subsOut.status === 'fulfilled' && subsOut.value ? subtitleFiles(subsOut.value) : null;
    const cleaned = local.status === 'fulfilled' && local.value ? local.value : null;
    const madeJob = job.status === 'fulfilled' && job.value ? job.value : null;
    let failed = false;
    const failedParts = {};
    if (!subsFiles && localSubs) {
      failedParts.subs = true;
      if (subsOut.status === 'rejected' && subsOut.reason?.name !== 'AbortError') {
        failed = true;
        setError(subsOut.reason?.message || 'Something went wrong while making the subtitles. Please try again.');
      }
    }
    if (!cleaned && wantsBrowser) {
      failedParts.sound = true;
      if (local.status === 'rejected' && local.reason?.name !== 'AbortError') {
        failed = true;
        setError(local.reason?.message || 'Something went wrong while cleaning this file. Please try again.');
      }
    }
    if (kind && !madeJob) {
      if (serverClean) failedParts.sound = true;
      if (serverText) failedParts.subs = true;
      if (job.status === 'rejected' && job.reason?.name !== 'AbortError') {
        failed = true;
        const err = job.reason;
        setError(inGigabytes(err.detail || err.message || 'Could not start the job.'));
        setErrorKind(err?.status === 507 ? 'storage' : '');
      }
    }
    // what this browser made now goes to the finishing step: the job's card (it waits for this), or a card of its own for a free run
    const parts = {
      pcm: cleaned?.pcm || null, wav: cleaned?.wav || null, wavUrl: cleaned?.url || '', subs: subsFiles, failed: canceled ? { sound: true, subs: true } : failedParts,
      ...(cleaned ? { pairParts: cleaned.pcm.pair, pairChannels: cleaned.pcm.channels, strengthDb: cleaned.attenDb } : {}),
    };
    let fin = madeJob ? getFinish(madeJob.id) : null;
    if (!fin && !canceled) {
      const made = { sound: !!cleaned, subs: !!subsFiles };
      if (made.sound || made.subs) {
        const label = made.sound && made.subs ? 'Free noise removal and subtitles' : made.sound ? 'Free noise removal' : 'Free subtitles';
        fin = beginFinish({
          id: `local-${Date.now().toString(36)}`, local: true, free: true, status: 'done', kind: 'denoise', label, displayName: f.name, inputName: f.name,
          createdAt: Math.floor(Date.now() / 1000), durationSeconds: i?.duration || 0, outputs: [], costMicros: 0,
        }, { file: f, info: i, wants: { ...wants, sound: made.sound, subs: made.subs }, browser: null });
      }
    }
    if (fin) {
      Object.assign(fin, parts);
      if (fin.job.local) {
        // a free run: the last step happens here, in the form, and then its card joins the list
        const off = fin.subscribe(() => setFinProg(fin.state.progress));
        setFinProg(0);
        await fin.run({ save: true, signal: ctl.signal });
        off();
        setFinProg(null);
        addLocalCard(fin.job);
      }
    }
    doneBrowser();
    if (kind) await refresh(); // the balance changes as soon as a job exists
    setBusy(false);
    setLocalRunning(false);
    setStep(null);
    if (!failed && !canceled) {
      setFile(null);
      setInfo(null);
      // the job's own page: its progress, then its result
      const id = madeJob?.id || fin?.job?.id;
      if (id) nav(jobPath(id), { state: madeJob ? { submitted: startedNote(madeJob) } : null });
    }
  }

  const pct = step?.frac != null ? Math.round(step.frac * 100) : null;
  const startLabel = !(wantsBrowser || localSubs || kind) ? 'Start' : !kind ? 'Start – free' : est ? `Start – about ${money(est)}` : 'Start';
  const showLeveling = info && info.hasAudio !== false && noise !== 'browser';
  const rebuilding = rebuild && !!rebuildRate;
  const autoLine = listening
    ? 'Listening to a little of your file to pick the best settings…'
    : applied && reco ? (custom ? 'Set automatically, then changed by you.' : 'Set automatically for this file.') : '';
  return (
    <>
      <form className="card stack job-form" onSubmit={submit} aria-labelledby="newjob-h">
        <h2 id="newjob-h" className="mb0">Add a file</h2>

        {!file && !busy && <FileDrop onFile={choose} accept={SERVER_FORMATS} hint="or click to pick one. Almost any audio or video file: MP3, WAV, M4A, FLAC, WMA, AMR, MP4, MOV, MKV, AVI, WMV, MPEG and more." />}
        {probing && <p role="status" className="muted"><span className="spinner" aria-hidden="true" /> Reading {shownName(file?.name)}…</p>}

        {file && info && (
          <div className="file-chip">
            <div className="grow">
              <div className="name">{shownName(file.name)}</div>
              <div className="meta">
                {bytes(file.size)}
                {duration ? ` · ${lengthUnknown ? 'about ' : ''}${clock(duration)}` : ''}
                {info.channels ? ` · ${info.channels === 1 ? 'mono' : info.channels === 2 ? 'stereo' : `${info.channels} channels`}` : ''}
                {info.sampleRate ? ` · ${(info.sampleRate / 1000).toFixed(info.sampleRate % 1000 ? 1 : 0)} kHz` : ''}
                {isVideo ? ' · video' : ''}
              </div>
            </div>
            {!busy && <button type="button" className="btn btn-ghost btn-sm" onClick={reset}>Remove</button>}
          </div>
        )}
        {info?.hasVideo && info.viaMediabunny && <p className="hint mb0">This is a video, so only its sound is uploaded when we do the work on our side. That keeps it small and quick, and the picture never leaves your device.</p>}

        {file && info && !probing && (
          <>
            {info.hasAudio !== false && (
              <div className="auto-bar" data-testid="auto-box">
                <label className="switch" data-testid="auto-settings">
                  <input type="checkbox" role="switch" name="auto" checked={auto} onChange={(e) => setAutoSettings(e.target.checked)} disabled={busy} />
                  <span><b>Automatic best settings</b></span>
                </label>
                {auto ? (
                  <p className="auto-line" aria-live="polite" data-testid="reco">
                    {listening ? (
                      <span data-testid="reco-listening"><span className="spinner" aria-hidden="true" /> {autoLine}</span>
                    ) : applied && reco ? (
                      <>
                        <span className="auto-text"><b data-testid="reco-status">{autoLine}</b> <span className="muted" data-testid="reco-line">{reco.line}</span></span>
                        {custom
                          ? <button type="button" className="btn btn-ghost btn-sm" onClick={() => applyReco(reco, info, true)} disabled={busy} data-testid="reco-apply">Use them again</button>
                          : <button type="button" className="btn btn-ghost btn-sm" onClick={undoAuto} disabled={busy} data-testid="reco-undo">Undo</button>}
                      </>
                    ) : (
                      <span className="muted">Listens to a little of your file and sets the best choices for it. You can still change anything.</span>
                    )}
                  </p>
                ) : (
                  <p className="auto-line muted">Off. Nothing is changed for you.</p>
                )}
                {auto && applied && reco?.reasons?.length > 0 && (
                  <details className="auto-why">
                    <summary>Why these choices</summary>
                    <ul className="reco-why small muted" data-testid="reco-why">
                      {reco.reasons.map((r) => <li key={r}>{r}</li>)}
                    </ul>
                  </details>
                )}
              </div>
            )}

            <Step title="Noise and voice" lead="Take the noise out of the sound.">
              <fieldset className="field" disabled={busy}>
                <legend className="sr-only">Noise removal</legend>
                <div className="choice-cards">
                  <Choice name="noise" value="off" current={noise} onPick={pickNoise} title="Off" hint="Leave the sound as it is" />
                  <Choice
                    name="noise" value="browser" current={noise} onPick={pickNoise} disabled={!verdict.ok}
                    title="Free, in your browser" hint="Done on your own device. Nothing is uploaded." reason={verdict.why}
                  />
                  <Choice
                    name="noise" value="denoise" current={noise} onPick={pickNoise} disabled={info.hasAudio === false}
                    title="Denoising" hint={`Paid, done on our side: the noise removed${denoiseRate ? `. ${rate(denoiseRate.usdPerMinute)} a minute` : ''}`}
                    reason="This file has no audio track."
                  />
                </div>
              </fieldset>

              {wantsBrowser && (
                <div className="step-sub">
                  <StrengthPicker name="lstrength" value={localStrength} onChange={(v) => { touch('strength'); setLocalStrength(v); }} disabled={busy} />
                  {info.channels > 2 && <p className="hint mb0">This file has {info.channels} channels; they will be mixed down to mono.</p>}
                  {info.duration > LONG_SECONDS && (
                    <div className="banner banner-warn">
                      This is a long recording ({human(info.duration)}). Cleaning it here could take about {human(eta)} and needs a lot of memory.
                      Keep this tab open, or choose Denoising for long files.
                    </div>
                  )}
                  {eta && <p className="hint mb0" data-testid="eta">Should take about {human(eta)} on this device{loadLeft ? ' (including getting the noise remover ready)' : ''}.</p>}
                  <GoodToKnow k="browserDenoise" />
                  {hasSecondTrack && <p className="hint mb0" data-testid="echo-browser-note">This file has a second audio track (your desktop sound). Removing speaker echo with it needs Denoising; the free cleanup in your browser can&rsquo;t do it.</p>}
                  {isVideo && info.viaLibav && <p className="hint mb0" data-testid="audio-only-note">Your browser can clean the sound of this video, but can&rsquo;t rebuild this video type, so you&rsquo;ll get the cleaned sound as a WAV file. Denoising gives you the whole video back.</p>}
                </div>
              )}

              {noise === 'off' && info.hasAudio !== false && (
                <div className="step-sub">
                  <Toggle
                    name="highs-alone" checked={restoreHighs} onChange={(e) => { touch('highs'); setRestoreHighs(e.target.checked); }} disabled={busy}
                    title="Restore highs" hintId="highs-alone-hint" hint={highsHint(highsRate)}
                  />
                </div>
              )}

              {wantsDenoise && (
                <div className="step-sub">
                  {!rebuilding && <StrengthPicker name="strength" value={strength} onChange={(v) => { touch('strength'); setStrength(v); }} disabled={busy} />}
                  {!rebuilding && (
                    <div className="field">
                      <label htmlFor="noisekind">Background noise</label>
                      <select id="noisekind" className="select" value={noiseKind} onChange={(e) => setNoiseKind(e.target.value)} disabled={busy}>
                        {NOISE.map((n) => <option key={n.id} value={n.id}>{n.label}</option>)}
                      </select>
                      <span className="hint">{NOISE.find((n) => n.id === noiseKind).hint}</span>
                    </div>
                  )}
                  {!rebuilding && (
                    <Toggle
                      name="highs" checked={restoreHighs} onChange={(e) => { touch('highs'); setRestoreHighs(e.target.checked); }} disabled={busy}
                      title="Restore highs" hintId="highs-hint" hint={highsHint(highsRate)}
                    />
                  )}
                  {rebuildOffered && !rebuildRate && (
                    <span data-testid="rebuild-paused">
                      <Toggle name="rebuild" checked={false} disabled muted title="Rebuild the voice" hint="Paused right now. Try again later." />
                    </span>
                  )}
                  {rebuildRate && (
                    <Toggle
                      name="rebuild" checked={rebuild} onChange={(e) => { touch('rebuild'); setRebuild(e.target.checked); }} disabled={busy}
                      title="Rebuild the voice" hintId="rebuild-hint" hint={rebuildHint(rebuildRate)}
                    />
                  )}
                  {hasSecondTrack && (
                    <Toggle
                      name="echo" checked={echo} onChange={(e) => setEcho(e.target.checked)} disabled={busy}
                      title="Remove speaker echo using the second audio track (your desktop sound)"
                      hint="OBS can record your mic and your desktop audio on separate tracks. Put the mic on track 1 and the desktop sound on track 2."
                    />
                  )}
                  <GoodToKnow k={rebuilding ? 'rebuild' : 'denoise'} />
                </div>
              )}
            </Step>

            {showLeveling && (
              <Step title="Leveling and loudness" lead="Even out the volume and set how loud the result is.">
                <Toggle
                  name={noise === 'off' ? 'leveling-alone' : 'leveling'} checked={leveling} disabled={busy}
                  onChange={(e) => { touch('leveling'); setLeveling(e.target.checked); }}
                  title="Leveling" hintId={noise === 'off' ? 'leveling-alone-hint' : 'leveling-hint'}
                  hint={`Paid, done on our side: evens out the volume and sets the loudness${noise === 'off' ? ', without removing any noise' : ''}${levelingRate ? `. ${rate(levelingRate.usdPerMinute)} a minute${noise === 'off' ? '' : ' more'}` : ''}.${noise === 'off' ? '' : ' Turn it off to keep the voice as it was recorded, just without the noise.'}`}
                />
                <div className="step-sub">
                  <div className="field">
                    <label htmlFor={noise === 'off' ? 'loud-alone' : 'loud'}>Loudness target</label>
                    <select id={noise === 'off' ? 'loud-alone' : 'loud'} className="select" value={loudness} onChange={(e) => setLoudness(e.target.value)} disabled={busy || !leveling}>
                      {LOUDNESS.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
                    </select>
                  </div>
                </div>
              </Step>
            )}

            <Step title="Subtitles and transcript" lead="Write down what is said, in the language you choose.">
              <fieldset className="field" disabled={busy}>
                <legend className="sr-only">Subtitles and transcript</legend>
                <div className="choice-cards">
                  <Choice name="subs" value="off" current={subs} onPick={pickSubs} title="Off" hint="No subtitles or transcript" />
                  <Choice
                    name="subs" value="browser" current={subs} onPick={pickSubs} disabled={!sVerdict.ok}
                    title="Free, in your browser" reason={sVerdict.why}
                    hint="Made on your own device, nothing is uploaded. Fine for clear speech. The first time, it downloads about 80 MB."
                  />
                  <Choice
                    name="subs" value="standard" current={subs} onPick={pickSubs}
                    title="Standard" hint={`Paid. Good for most recordings, in many languages${standardRate ? `. ${rate(standardRate.usdPerMinute)} a minute` : ''}`}
                  />
                  <Choice
                    name="subs" value="accurate" current={subs} onPick={pickSubs}
                    title="Extra accurate" hint="Paid. Best for accents, noisy rooms and specialist words. Works in any language."
                  />
                </div>
              </fieldset>

              {wantsText && (
                <div className="step-sub">
                  <div className="field">
                    <label htmlFor="lang">Language</label>
                    <select id="lang" className="select" value={language} onChange={(e) => setLanguage(e.target.value)} disabled={busy}>
                      {LANGUAGES.map(([c, n]) => <option key={c} value={c}>{n}</option>)}
                    </select>
                  </div>
                  <fieldset className="field" disabled={busy || !serverText} data-testid="translate-to">
                    <legend className="label">Also translate into</legend>
                    <div className="sel-row">
                      {targets.map((t, i) => (
                        <select
                          key={i} className="select" aria-label={`Translation language ${i + 1}`} value={t}
                          onChange={(e) => setTargets(targets.map((x, j) => (j === i ? e.target.value : x)))}
                        >
                          <option value="">{i === 0 ? 'No translation' : 'No other language'}</option>
                          {LANGUAGES.filter(([c]) => c && c !== language && (c === t || !targets.includes(c))).map(([c, n]) => <option key={c} value={c}>{n}</option>)}
                        </select>
                      ))}
                    </div>
                    <span className="hint" data-testid="translate-hint">
                      {serverText
                        ? `One set of subtitle files per language${translateRate ? `, ${rate(translateRate.usdPerMinute)} a minute each` : ''}. Made from the transcript.`
                        : 'Translations come with Standard or Extra accurate subtitles.'}
                    </span>
                  </fieldset>
                  <fieldset className="field" disabled={busy}>
                    <legend className="label">Formats</legend>
                    <div className="checks">
                      {[['srt', 'SRT'], ['vtt', 'VTT'], ['txt', 'TXT']].map(([id, label]) => (
                        <label className="check" key={id}>
                          <input type="checkbox" checked={caps[id]} onChange={(e) => setCaps({ ...caps, [id]: e.target.checked })} />
                          <span>{label}</span>
                        </label>
                      ))}
                    </div>
                  </fieldset>
                  {canEmbed && (
                    <Toggle
                      name="embed" checked={embed} disabled={busy} title="Put the subtitles into the video" hintId="embed-hint"
                      onChange={(e) => {
                        setEmbed(e.target.checked);
                        // the cleaned video becomes an MKV, the kind of file players show these subtitles from
                        if (e.target.checked && cleaning && !embedOnServer && !canHoldSubtitles(choice.format) && choice.okIds?.includes('mkv')) choice.pickFormat('mkv');
                      }}
                      hint={embedOnServer
                        ? 'Your browser can’t open this video type, so it’s uploaded whole and we put them in on our side.'
                        : `Done on your device ${kind ? 'when the job is finished' : 'once the subtitles are made'}, so nothing extra is uploaded. Players can turn them on and off. They go into an MKV file, the kind players show subtitles from${!cleaning && embedFormat && !canHoldSubtitles(embedFormat) ? `, so you’ll get an MKV instead of ${embedFormat.toUpperCase()}` : ''}. Video editors usually take the .srt file instead.`}
                    />
                  )}
                  {wantsText && videoResult && tooBigHere && <p className="hint mb0" data-testid="embed-too-big">This browser can’t rebuild a video this big (over {gigabytes(limit)}); Chrome, Edge or Firefox can. You’ll get the subtitle files to load in your player or video editor.</p>}
                  {subs === 'accurate' && <span className="hint" data-testid="accurate-note">Extra accurate sends your audio to a speech partner we work with, only for this option. <Link to="/privacy">Privacy details</Link></span>}
                  {serverClean && serverText && <span className="hint" data-testid="both-note">We write the transcript from your original recording (it comes out more accurate), and clean the sound separately.</span>}
                  {wantsBrowser && serverText && <span className="hint" data-testid="browser-subs-note">The noise removal happens on your device. For the subtitles, the original sound is uploaded.</span>}
                  {serverClean && localSubs && <span className="hint" data-testid="denoise-local-subs-note">The cleanup is done on our side; the subtitles are made here on your device, from your original recording.</span>}
                  {subs === 'browser' && <GoodToKnow k="browserSubs" />}
                  {subs === 'standard' && <GoodToKnow k="standardSubs" />}
                  {subs === 'accurate' && <GoodToKnow k="accurateSubs" />}
                  {translations.length > 0 && <GoodToKnow k="translations" />}
                </div>
              )}
            </Step>

            {cleaning && (
              <Step title="The result file" lead="The cleaned sound as the file you choose. It downloads by itself when it’s done.">
                <OutputFields choice={choice} info={info} idPrefix="job" disabled={busy} />
              </Step>
            )}

            <div className="step step-end">
              {willDownload() && <p className="mb0" data-testid="will-download">When it’s done, these download by themselves: <b>{willDownload()}</b>.</p>}
              {kind && duration && est != null && (
                <p className="mb0" data-testid="estimate">
                  <b>Estimated cost: {money(est)}</b>{' '}
                  <span className="hint">for {human(duration)} of audio{breakdown ? ` (cleaning ${money(breakdown[0])} + subtitles ${money(breakdown[1])})` : localSubs ? '; the subtitles made on your device are free' : ''}. You only pay for what gets processed. {BILLING_NOTE}</span>
                </p>
              )}
              {lengthUnknown && <p className="hint mb0" data-testid="length-unknown">Your browser can&rsquo;t read the length of this file type, so the numbers here are a rough guess from its size. We measure it after the upload and you pay for the real length.</p>}
              {problems.map((p) => (
                <div key={p} className="banner banner-warn" role="alert">
                  {p} {p.includes('balance') && <Link to="/app/billing">Add credit</Link>}
                  {p.includes('load the prices') && <button type="button" className="btn btn-ghost btn-sm" onClick={() => loadPricing().catch(() => {})}>Try again</button>}
                </div>
              ))}
              <div className="row">
                <button type="submit" className="btn btn-primary btn-lg" disabled={!canStart || busy} data-testid="start">{startLabel}</button>
              </div>
            </div>
          </>
        )}

        {busy && (
          <div className="stack" data-testid="run-progress">
            {localRunning && <ProgressPanel prog={localProg} stage={localStage} load={load} procEta={procEta} tail={tail} finishEta={finishEta} />}
            {subsRunning && <SubtitlesProgress prog={subsProg} load={speechLoad} />}
            {finProg != null && <FinishProgress progress={finProg} from={1 - tail} />}
            {step && (
              <div className="stack" data-testid="upload-progress">
                <div className="row-between">
                  <strong>
                    {step.phase === 'extract' && 'Pulling the sound out of your video…'}
                    {step.phase === 'compress' && 'Making it smaller for upload…'}
                    {step.phase === 'create' && 'Getting things ready…'}
                    {step.phase === 'upload' && 'Uploading…'}
                  </strong>
                  <span className="muted small">
                    {pct != null ? `${pct}%` : ''}
                    {step.phase === 'upload' && step.total ? ` · ${bytes(step.loaded)} of ${bytes(step.total)}` : ''}
                  </span>
                </div>
                <progress className="bar" max={100} value={pct ?? undefined} aria-label="Upload progress" />
                {compressedInfo && <p className="small muted mb0" data-testid="compressed-note">{compressedNote(compressedInfo)}</p>}
              </div>
            )}
            <div className="row">
              <button type="button" className="btn btn-secondary" onClick={() => abort.current?.abort()}>Cancel</button>
              {(wantsBrowser || localSubs) && <span className="muted small">Keep this tab open while it works.</span>}
            </div>
          </div>
        )}

        {error && (
          <div className={errorKind === 'storage' ? 'banner banner-warn' : 'banner banner-error'} role="alert" data-testid="job-error">
            {error}
          </div>
        )}

      </form>

    </>
  );
}
