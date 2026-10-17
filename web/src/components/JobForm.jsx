import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { Link } from 'react-router';
import { GoodToKnow } from '../alternatives.jsx';
import { useAuth } from '../auth.jsx';
import { balanceMoney, BILLING_NOTE, bytes, clock, human, money, rate, shownName } from '../format.js';
import { estimateMicros, findRate, loadPricing, usePricing } from '../hooks.js';
import { LANGUAGES, LOUDNESS, NOISE, submitJob } from '../jobs.js';
import { buildLimit, canHoldSubtitles, defaultFormatId, serverAudioFormat } from '../media/convert.js';
import { denoiserLoadState, preloadDenoiser, subscribeDenoiserLoad } from '../media/denoise.js';
import { analyseFile } from '../media/analyse.js';
import { probeMedia } from '../media/mb.js';
import { denoiseFile } from '../media/pipeline.js';
import { preloadSpeech, speechLoadState, subscribeSpeechLoad, transcribeFile } from '../media/speech.js';
import { buildWav, wavFits } from '../media/wav.js';
import { BrowserResult, ProgressPanel } from './BrowserClean.jsx';
import SubtitlesResult, { SubtitlesProgress, subtitleFiles } from './SubtitlesResult.jsx';
import OutputFields, { useOutputChoice } from './OutputPicker.jsx';
import FileDrop, { SERVER_FORMATS } from './FileDrop.jsx';
import Recommended from './Recommended.jsx';
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

/** One radio card. A card that can't be used stays visible, greyed out, with the reason where its hint would be. */
function Choice({ name, value, current, onPick, disabled, title, hint, reason, recommended }) {
  return (
    <label className={`choice${disabled ? ' is-disabled' : ''}`}>
      <input type="radio" name={name} value={value} checked={current === value} onChange={() => onPick(value)} disabled={disabled} />
      <span>{recommended && !disabled && <em className="reco-badge" data-testid={`reco-${name}-${value}`}>Recommended</em>}<b>{title}</b><small>{disabled && reason ? reason : hint}</small></span>
    </label>
  );
}

/**
 * The Cleanup form: drop one file, then choose what to do with it. Only what is possible for that file is offered.
 *   Noise removal:  Off | Free, in your browser (nothing uploaded) | Denoising (paid, on our side)
 *   Subtitles:      Off | Free, in your browser (nothing uploaded) | Standard (paid) | Extra accurate (paid)
 * Any combination works: the browser part runs here, in parallel with the job (if any) for the rest.
 * onSubmitted({id, ...}, {file, info, extracted, keepVideo})
 */
export default function JobForm({ onSubmitted }) {
  const { balance, refresh } = useAuth();
  const { data: pricing, error: pricingError } = usePricing();
  const load = useSyncExternalStore(subscribeDenoiserLoad, denoiserLoadState);
  const speechLoad = useSyncExternalStore(subscribeSpeechLoad, speechLoadState);

  const [file, setFile] = useState(null);
  const [info, setInfo] = useState(null);
  const [probing, setProbing] = useState(false);
  const [noise, setNoise] = useState('off'); // 'off' | 'browser' | 'denoise'
  const [subs, setSubs] = useState('off'); // 'off' | 'browser' | 'standard' | 'accurate'
  const [keepVideo, setKeepVideo] = useState(true);
  const [embed, setEmbed] = useState(false);
  const [echo, setEcho] = useState(false); // Denoising: remove speaker echo using the second audio track
  const [loudness, setLoudness] = useState('-16');
  const [noiseKind, setNoiseKind] = useState('auto');
  const [leveling, setLeveling] = useState(true); // Leveling with Denoising or Voice rebuild: even out the voice, set the loudness
  const [levelingAlone, setLevelingAlone] = useState(false); // Leveling on its own, with noise removal Off (paid)
  const [restoreHighs, setRestoreHighs] = useState(true); // Denoising: bring back the highs of dull phone-quality recordings
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
  const [localResult, setLocalResult] = useState(null); // {file, info, isVideo, keepVideo, result}
  const [subsRunning, setSubsRunning] = useState(false);
  const [subsProg, setSubsProg] = useState(null);
  const [subsResult, setSubsResult] = useState(null); // {file, info, files, embed, cleanPcm} from the free in-browser subtitles
  const [compressedInfo, setCompressedInfo] = useState(null); // {from, to} once the input was encoded to FLAC
  const [stats, setStats] = useState(null); // what the quick listen to the file found (media/analysis.js); null: not (yet) measured
  const [listening, setListening] = useState(false);
  const [auto, setAuto] = useState(storedAuto); // "Automatic best settings": listen to the file and pick the settings for it
  const autoRef = useRef(auto);
  autoRef.current = auto;
  const touched = useRef(false); // the person changed one of the recommended settings themselves: later findings leave them alone
  const [custom, setCustom] = useState(false); // ... the same, for showing the button that goes back to the recommendation
  const rebuildOk = useRef(false); // is the rebuild offered (the price list has it)?
  const listen = useRef(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const choice = useOutputChoice({ file, info, keepVideo, serverAudio: noise === 'denoise' || (noise === 'off' && levelingAlone) });
  const [errorKind, setErrorKind] = useState(''); // 'storage' while the server is short on space
  const abort = useRef(null);
  const jobLocal = useRef(null); // {out, entry} of the job just started, so the sound cleaned here can join its subtitles

  useEffect(() => () => { abort.current?.abort(); listen.current?.abort(); }, []);
  useEffect(() => () => { if (localResult?.result?.url) URL.revokeObjectURL(localResult.result.url); }, [localResult]);
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
  const levelingOnly = noise === 'off' && levelingAlone && info?.hasAudio !== false; // Leveling on its own, on our side
  const serverClean = wantsDenoise || levelingOnly; // a cleaned file comes from our side
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
  const videoResult = isVideo ? !serverClean || choice.isVideoFile : serverClean && choice.isVideoFile;
  const embedOnServer = serverText && isVideo && !info?.viaMediabunny;
  const limit = buildLimit(); // null: this browser writes the result to disk, so any size works
  const tooBigHere = isVideo && !embedOnServer && file && limit != null && file.size > limit;
  // subtitles made here go into a video this browser can rebuild (one it opens itself)
  const canEmbed = wantsText && videoResult && !tooBigHere && (serverText || !isVideo || !!info?.viaMediabunny);
  const embedNow = canEmbed && embed;
  const embedHere = embedNow && !embedOnServer;
  // A subtitles-only video keeps its own file type when that can carry them (WebM), else it becomes MKV.
  const embedFormat = serverClean ? choice.format : isVideo ? defaultFormatId(file, info, true) : null;

  // Some formats (WMA, AMR, AVI, MPEG...) the browser can't read, but the server can. Then the length is a rough
  // guess from the file size (128 kbit/s) until the server measures it; the price is corrected to the real length.
  const lengthUnknown = !!(file && info && !probing && !info.duration && info.hasAudio !== false);
  const maxJob = pricing?.maxJobSeconds;
  const duration = info?.duration || (lengthUnknown ? Math.min(Math.max(30, (file.size * 8) / 128000), maxJob || 4 * 3600) : null);
  const denoiseRate = findRate(pricing, 'denoise', 'standard');
  const rebuildOffered = findRate(pricing, 'denoise', 'rebuild');
  const rebuildRate = pricing?.rebuildAvailable ? rebuildOffered : null; // paused while its machine is unavailable
  const denoiseTier = levelingOnly ? 'off' : rebuild && rebuildRate ? 'rebuild' : 'standard';
  const levelingRate = findRate(pricing, 'leveling', 'standard');
  const withLeveling = serverClean && (levelingOnly || leveling); // Leveling is charged on its own line
  const standardRate = findRate(pricing, 'transcribe', 'local');
  const translateRate = findRate(pricing, 'transcribe_translate', 'standard');
  const fileReco = useMemo(() => (stats ? recommendForFile(stats, { language }) : null), [stats, language]);
  const reco = auto ? fileReco : null; // shown and marked only with "Automatic best settings" on
  rebuildOk.current = !!rebuildRate;
  const recNoise = reco ? (reco.noise === 'browser' && verdict && !verdict.ok ? 'denoise' : reco.noise) : null;

  const procEta = useMemo(() => {
    if (!info?.duration) return null;
    const channels = info.channels && info.channels >= 2 ? 2 : 1;
    const cores = navigator.hardwareConcurrency || 2;
    return info.duration * 0.13 * (channels === 2 && cores < 3 ? 2 : 1);
  }, [info]);
  // Whatever is still downloading counts too, so the estimate is honest if you start straight away.
  const loadLeft = load.done ? 0 : load.secondsLeft;
  const eta = procEta == null ? null : procEta + (loadLeft || 0);

  // For Denoising the file type in the list and "Keep the video" are the same decision: a video type keeps the picture.
  useEffect(() => {
    if (serverClean && isVideo && !info?.viaLibav && choice.outputs) setKeepVideo(choice.isVideoFile);
  }, [serverClean, isVideo, info, choice.outputs, choice.isVideoFile]);

  const est = useMemo(() => {
    if (!kind) return 0;
    return estimateMicros(pricing, { kind, denoiseTier, transcribeTier: tier, seconds: duration, translate: translations.length, leveling: withLeveling });
  }, [pricing, kind, tier, duration, denoiseTier, translations.length, withLeveling]);
  const breakdown = useMemo(() => {
    if (kind !== 'denoise_transcribe' || !duration) return null;
    return [estimateMicros(pricing, { kind: 'denoise', denoiseTier, seconds: duration, leveling: withLeveling }), estimateMicros(pricing, { kind: 'transcribe', transcribeTier: tier, seconds: duration, translate: translations.length })];
  }, [pricing, kind, tier, duration, denoiseTier, translations.length, withLeveling]);

  async function choose(f) {
    if (f.size < TINY_FILE_BYTES) {
      // Nothing to read in there: say so instead of offering options with made-up numbers.
      setFile(null);
      setInfo(null);
      setResult(null);
      setError(f.size === 0 ? 'That file is empty (0 bytes). Please pick another one.' : 'That file is too small to hold any sound. Please pick another one.');
      return;
    }
    setFile(f);
    setInfo(null);
    setResult(null);
    setError('');
    setEmbed(false);
    setEcho(false);
    setKeepVideo(true);
    setSubs('off');
    setNoise('off');
    setStats(null);
    setListening(false);
    touched.current = false;
    setCustom(false);
    listen.current?.abort();
    setProbing(true);
    const i = await probeMedia(f);
    // The best choice that works for this file; a paid one is never switched on without the price being right there.
    const canBrowser = browserVerdict(i).ok;
    setNoise(canBrowser ? 'browser' : i.hasAudio === false ? 'off' : 'denoise');
    setInfo(i);
    setProbing(false);
    // The noise remover is only fetched once it is clear that this file wants it (the quick listen below says), or when it can't be told.
    if (i.hasAudio !== false && autoRef.current) listenTo(f, i, canBrowser);
    else if (canBrowser) warmUpDenoiser();
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
      if (r && !touched.current && autoRef.current) applyReco(r, i);
      else if (!r && canBrowser && !touched.current) warmUpDenoiser();
    } catch {
      /* canceled, or unreadable: the form just keeps its usual starting choices */
      if (canBrowser && !ctl.signal.aborted && !touched.current) warmUpDenoiser();
    } finally {
      if (listen.current === ctl) setListening(false);
    }
  }

  /** Set the recommended choices (never anything paid that the file doesn't need). */
  function applyReco(r, i = info) {
    const canBrowser = i ? browserVerdict(i).ok : false;
    const next = r.noise === 'browser' && !canBrowser ? 'denoise' : r.noise;
    setNoise(next);
    if (next === 'browser') warmUpDenoiser();
    touched.current = false;
    setCustom(false);
    if (next === 'browser') setLocalStrength(r.strength);
    else if (next === 'denoise') {
      setStrength(r.strength);
      setRebuild(!!(r.rebuild && rebuildOk.current));
      setNoiseKind('auto');
      if (r.restoreHighs) setRestoreHighs(true);
    }
  }

  function pickSubs(v) {
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
    if (!file || !info || info.hasAudio === false) return;
    if (fileReco) applyReco(fileReco);
    else if (!listening) listenTo(file, info, browserVerdict(info).ok);
  }

  /** The person chose something themselves: what the quick listen finds later no longer changes it. */
  function touch() {
    touched.current = true;
    setCustom(true);
  }

  function pickNoise(v) {
    touch();
    setNoise(v);
    if (v === 'browser') warmUpDenoiser();
  }

  function reset() {
    abort.current?.abort();
    listen.current?.abort();
    setStats(null);
    setListening(false);
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
  if (kind && est == null && duration && pricingError) problems.push('We couldn’t load the prices just now, so this can’t start yet.');
  if (kind && est != null && balance && est > balance.available_micros) problems.push(`This costs about ${money(est)}, but your balance is ${balanceMoney(balance.available_micros)}.`);
  if (info && !wantsBrowser && !localSubs && !kind) problems.push('Choose at least one thing to do: noise removal, subtitles, or both.');
  if (wantsText && !Object.values(caps).some(Boolean)) problems.push('Choose at least one caption or transcript format.');
  if (embedHere && serverClean && !canHoldSubtitles(choice.format)) problems.push(`Subtitles put into the video need an MKV file (most players can’t show them from ${choice.format.toUpperCase()}). Pick MKV, or untick “Put the subtitles into the video”.`);
  const canStart = (wantsBrowser || localSubs || kind) && file && info && !probing && !busy && problems.length === 0 && (!kind || est != null) && (!serverClean || choice.valid);

  async function runLocal(f, i, signal) {
    setLocalRunning(true);
    setLocalProg({ done: 0, total: i?.duration || null, rate: 0, eta: null, elapsed: 0 });
    try {
      const attenDb = strengthDb(localStrength);
      const pcm = await denoiseFile(f, {
        attenDb, durationHint: i?.duration || null, signal, onProgress: setLocalProg, onStage: setLocalStage,
      });
      setLocalStage('Preparing the preview…');
      const wav = await buildWav(pcm, 16);
      if (signal.aborted) throw new DOMException('Canceled', 'AbortError');
      return { pcm, wav, url: URL.createObjectURL(wav), strength: localStrength };
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

  async function runJob(f, i, signal) {
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
        leveling: levelingOnly || leveling,
        restore_highs: restoreHighs,
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
      setResult(out);
      const entry = {
        file: f, info: i, extracted: out.extracted, keepVideo, embed: embedHere, browserClean: wantsBrowser, browserSubs: localSubs,
        output: serverClean ? { name: choice.name.trim(), format: choice.format, size: choice.size } : null,
      };
      jobLocal.current = { out, entry };
      onSubmitted?.(out, entry);
      return out;
    } finally {
      setStep(null);
    }
  }

  async function submit(e) {
    e.preventDefault();
    if (!canStart) return;
    touched.current = true; // whatever the quick listen finds from here on must not change a run that has started
    const f = file;
    const i = info;
    const ctl = new AbortController();
    abort.current = ctl;
    setBusy(true);
    setError('');
    setErrorKind('');
    setResult(null);
    setCompressedInfo(null);
    if (wantsBrowser) setLocalResult(null);
    if (localSubs) setSubsResult(null);
    const subsCaps = { ...caps };
    const subsEmbed = embedHere && !kind; // with a job, the job's card puts them into the video

    // The browser parts and the job (if there is one) run side by side; each reports on its own.
    const outcomes = await Promise.allSettled([
      wantsBrowser ? runLocal(f, i, ctl.signal) : Promise.resolve(null),
      localSubs ? runLocalSubs(f, i, ctl.signal) : Promise.resolve(null),
      kind ? runJob(f, i, ctl.signal) : Promise.resolve(null),
    ]);
    const [local, subsOut, job] = outcomes;
    const subsFiles = subsOut.status === 'fulfilled' && subsOut.value ? subtitleFiles(subsOut.value) : null;
    const started = jobLocal.current;
    jobLocal.current = null;
    if (started?.entry.embed && (wantsBrowser || localSubs)) {
      // The video the job's card makes uses the sound cleaned here and the subtitles made here, when they worked out.
      if (wantsBrowser) {
        if (local.status === 'fulfilled' && local.value) started.entry.cleanPcm = local.value.pcm;
        else started.entry.browserClean = false;
      }
      if (localSubs) {
        if (subsFiles) { started.entry.localVtt = subsFiles.vtt; started.entry.localLang = subsFiles.language; }
        else started.entry.browserSubs = false;
      }
      onSubmitted?.(started.out, started.entry);
    }
    const canceled = ctl.signal.aborted || outcomes.some((o) => o.status === 'rejected' && o.reason?.name === 'AbortError');
    let failed = false;
    if (subsFiles) {
      setSubsResult({ file: f, info: i, files: subsFiles, caps: subsCaps, embed: subsEmbed, cleanPcm: local.status === 'fulfilled' && local.value ? local.value.pcm : null });
    } else if (subsOut.status === 'rejected' && subsOut.reason?.name !== 'AbortError') {
      failed = true;
      setError(subsOut.reason?.message || 'Something went wrong while making the subtitles. Please try again.');
    }
    if (local.status === 'fulfilled' && local.value) {
      setLocalResult({ file: f, info: i, isVideo: !!i?.hasVideo, keepVideo: keepVideo && !i?.viaLibav, result: local.value });
    } else if (local.status === 'rejected' && local.reason?.name !== 'AbortError') {
      failed = true;
      setError(local.reason?.message || 'Something went wrong while cleaning this file. Please try again.');
    }
    if (job.status === 'rejected' && job.reason?.name !== 'AbortError') {
      failed = true;
      const err = job.reason;
      setError(inGigabytes(err.detail || err.message || 'Could not start the job.'));
      setErrorKind(err?.status === 507 ? 'storage' : '');
    }
    if (kind) await refresh(); // the balance changes as soon as a job exists
    setBusy(false);
    setLocalRunning(false);
    setStep(null);
    if (!failed && !canceled) {
      setFile(null);
      setInfo(null);
    }
  }

  const pct = step?.frac != null ? Math.round(step.frac * 100) : null;
  const startLabel = !(wantsBrowser || localSubs || kind) ? 'Start' : !kind ? 'Start – free' : est ? `Start – about ${money(est)}` : 'Start';
  return (
    <>
      <form className="card stack" onSubmit={submit} aria-labelledby="newjob-h">
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
        {info?.hasVideo && info.viaMediabunny && <p className="small muted mb0">This is a video, so only its sound is uploaded when we do the work on our side. That keeps it small and quick, and the picture never leaves your device.</p>}

        {file && info && !probing && (
          <>
            {info.hasAudio !== false && (
              <label className="switch" data-testid="auto-settings">
                <input type="checkbox" role="switch" name="auto" checked={auto} onChange={(e) => setAutoSettings(e.target.checked)} disabled={busy} />
                <span><b>Automatic best settings</b>
                  <span className="hint block">Listens to a little of your file and picks the settings it needs. You can still change anything.</span>
                </span>
              </label>
            )}
            {auto && (
              <Recommended
                title="Recommended for this file" line={reco?.line} reasons={reco?.reasons} listening={listening}
                listeningText="Listening to a little of your file to suggest the best settings…"
                onApply={custom && reco ? () => applyReco(reco) : null}
              />
            )}
            <fieldset className="field" disabled={busy}>
              <legend className="label">Noise removal</legend>
              <div className="choice-cards">
                <Choice name="noise" value="off" current={noise} onPick={pickNoise} title="Off" hint="Leave the sound as it is" recommended={recNoise === 'off'} />
                <Choice
                  name="noise" value="browser" current={noise} onPick={pickNoise} disabled={!verdict.ok}
                  title="Free, in your browser" hint="Done on your own device. Nothing is uploaded." reason={verdict.why} recommended={recNoise === 'browser'}
                />
                <Choice
                  name="noise" value="denoise" current={noise} onPick={pickNoise} disabled={info.hasAudio === false}
                  title="Denoising" hint={`Paid, done on our side: the noise removed${denoiseRate ? `. ${rate(denoiseRate.usdPerMinute)} a minute` : ''}`}
                  reason="This file has no audio track." recommended={recNoise === 'denoise'}
                />
              </div>
            </fieldset>
            {noise === 'off' && info.hasAudio !== false && (
              <div className="stack">
                <label className="check">
                  <input type="checkbox" name="leveling-alone" checked={levelingAlone} onChange={(e) => setLevelingAlone(e.target.checked)} disabled={busy} />
                  <span>Leveling
                    <span className="hint block" data-testid="leveling-alone-hint">
                      Paid, done on our side: evens out the volume and sets the loudness, without removing any noise.
                      {levelingRate ? ` ${rate(levelingRate.usdPerMinute)} a minute.` : ''}
                    </span>
                  </span>
                </label>
                {levelingOnly && (
                  <>
                    <div className="field">
                      <label htmlFor="loud-alone">Loudness target</label>
                      <select id="loud-alone" className="select" value={loudness} onChange={(e) => setLoudness(e.target.value)} disabled={busy}>
                        {LOUDNESS.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
                      </select>
                    </div>
                    <OutputFields choice={choice} info={info} idPrefix="job" disabled={busy} />
                  </>
                )}
              </div>
            )}
            {wantsBrowser && <GoodToKnow k="browserDenoise" />}
            {wantsDenoise && <GoodToKnow k={rebuild && rebuildRate ? 'rebuild' : 'denoise'} />}

            {wantsBrowser && (
              <div className="stack">
                <StrengthPicker name="lstrength" value={localStrength} onChange={(v) => { touch(); setLocalStrength(v); }} disabled={busy} />
                {info.channels > 2 && <p className="small muted mb0">This file has {info.channels} channels; they will be mixed down to mono.</p>}
                {info.duration > LONG_SECONDS && (
                  <div className="banner banner-warn">
                    This is a long recording ({human(info.duration)}). Cleaning it here could take about {human(eta)} and needs a lot of memory.
                    Keep this tab open, or choose Denoising for long files.
                  </div>
                )}
                {info.duration && !wavFits(info.duration, info.channels >= 2 ? 2 : 1, 24) && <p className="muted small mb0" data-testid="wav24-warn">For a recording this long, only the 16-bit WAV can be saved (a WAV file can’t be bigger than 4 GB).</p>}
                {eta && <p className="muted small mb0" data-testid="eta">Should take about {human(eta)} on this device{loadLeft ? ' (including getting the noise remover ready)' : ''}.</p>}
              </div>
            )}

            {wantsDenoise && (
              <div className="stack">
              {rebuildOffered && !rebuildRate && (
                <label className="check muted" data-testid="rebuild-paused">
                  <input type="checkbox" name="rebuild" checked={false} disabled />
                  <span>Rebuild the voice
                    <span className="hint block">Paused right now. Try again later.</span>
                  </span>
                </label>
              )}
              {rebuildRate && (
                <label className="check">
                  <input type="checkbox" name="rebuild" checked={rebuild} onChange={(e) => { touch(); setRebuild(e.target.checked); }} disabled={busy} />
                  <span>Rebuild the voice
                    <span className="hint block" data-testid="rebuild-hint">
                      For recordings made in echoey rooms: removes the room echo and brings back dull, muffled sound. Slower,
                      {' '}{rate(rebuildRate.usdPerMinute)} a minute.
                      On very echoey recordings it can change a word, so listen before you publish.
                    </span>
                  </span>
                </label>
              )}
              {!(rebuild && rebuildRate) && <StrengthPicker name="strength" value={strength} onChange={(v) => { touch(); setStrength(v); }} disabled={busy} />}
              <div className="form-grid">
                {!(rebuild && rebuildRate) && (
                <div className="field">
                  <label htmlFor="noisekind">Background noise</label>
                  <select id="noisekind" className="select" value={noiseKind} onChange={(e) => setNoiseKind(e.target.value)} disabled={busy}>
                    {NOISE.map((n) => <option key={n.id} value={n.id}>{n.label}</option>)}
                  </select>
                  <span className="hint">{NOISE.find((n) => n.id === noiseKind).hint}</span>
                </div>
                )}
                <div className="field">
                  <label htmlFor="loud">Loudness target</label>
                  <select id="loud" className="select" value={loudness} onChange={(e) => setLoudness(e.target.value)} disabled={busy || !leveling}>
                    {LOUDNESS.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
                  </select>
                </div>
              </div>
              <label className="check">
                <input type="checkbox" name="leveling" checked={leveling} onChange={(e) => setLeveling(e.target.checked)} disabled={busy} />
                <span>Leveling
                  <span className="hint block" data-testid="leveling-hint">Evens out the volume and sets the loudness above{levelingRate ? `, ${rate(levelingRate.usdPerMinute)} a minute more` : ''}. Turn it off to keep the voice as it was recorded, just without the noise.</span>
                </span>
              </label>
              {!(rebuild && rebuildRate) && (
                <label className="check">
                  <input type="checkbox" name="highs" checked={restoreHighs} onChange={(e) => { touch(); setRestoreHighs(e.target.checked); }} disabled={busy} />
                  <span>Restore missing highs
                    <span className="hint block" data-testid="highs-hint">For phone calls and other recordings that sound dull or muffled. Clear recordings are left as they are.</span>
                  </span>
                </label>
              )}
              <OutputFields choice={choice} info={info} idPrefix="job" disabled={busy} />
              {hasSecondTrack && (
                <label className="check">
                  <input type="checkbox" name="echo" checked={echo} onChange={(e) => setEcho(e.target.checked)} disabled={busy} />
                  <span>
                    Remove speaker echo using the second audio track (your desktop sound)
                    <span className="hint block">OBS can record your mic and your desktop audio on separate tracks. Put the mic on track 1 and the desktop sound on track 2.</span>
                  </span>
                </label>
              )}
              </div>
            )}

            {hasSecondTrack && wantsBrowser && <p className="small muted mb0" data-testid="echo-browser-note">This file has a second audio track (your desktop sound). Removing speaker echo with it needs Denoising; the free cleanup in your browser can&rsquo;t do it.</p>}

            {isVideo && wantsBrowser && info.viaLibav && <p className="small muted mb0" data-testid="audio-only-note">Your browser can clean the sound of this video, but can&rsquo;t rebuild this video type, so you&rsquo;ll get the cleaned sound as a WAV file. Denoising gives you the whole video back.</p>}

            <fieldset className="field" disabled={busy}>
              <legend className="label">Subtitles &amp; transcript</legend>
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
                  recommended={reco?.subs === 'standard'}
                />
                <Choice
                  name="subs" value="accurate" current={subs} onPick={pickSubs}
                  title="Extra accurate" hint="Paid. Best for accents, noisy rooms and specialist words. Works in any language."
                  recommended={reco?.subs === 'accurate'}
                />
              </div>
              {subs === 'accurate' && <span className="hint" data-testid="accurate-note">Extra accurate sends your audio to a speech partner we work with, only for this option. <Link to="/privacy">Privacy details</Link></span>}
              {serverClean && serverText && <span className="hint" data-testid="both-note">We write the transcript from your original recording (it comes out more accurate), and clean the sound separately.</span>}
              {wantsBrowser && serverText && <span className="hint" data-testid="browser-subs-note">The noise removal happens on your device. For the subtitles, the original sound is uploaded.</span>}
              {serverClean && localSubs && <span className="hint" data-testid="denoise-local-subs-note">The cleanup is done on our side; the subtitles are made here on your device, from your original recording.</span>}
            </fieldset>
            {subs === 'browser' && <GoodToKnow k="browserSubs" />}
            {subs === 'standard' && <GoodToKnow k="standardSubs" />}
            {subs === 'accurate' && <GoodToKnow k="accurateSubs" />}
            {translations.length > 0 && <GoodToKnow k="translations" />}

            {wantsText && (
              <div className="stack">
                <div className="form-grid">
                  <div className="field">
                    <label htmlFor="lang">Language</label>
                    <select id="lang" className="select" value={language} onChange={(e) => setLanguage(e.target.value)} disabled={busy}>
                      {LANGUAGES.map(([c, n]) => <option key={c} value={c}>{n}</option>)}
                    </select>
                  </div>
                  <fieldset className="field" disabled={busy || !serverText} data-testid="translate-to">
                    <legend className="label">Also translate into</legend>
                    <div className="checks">
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
                </div>
                {canEmbed && (
                  <label className="check">
                    <input
                      type="checkbox" name="embed" checked={embed} disabled={busy}
                      onChange={(e) => {
                        setEmbed(e.target.checked);
                        // the cleaned video becomes an MKV, the kind of file players show these subtitles from
                        if (e.target.checked && serverClean && !embedOnServer && !canHoldSubtitles(choice.format) && choice.okIds?.includes('mkv')) choice.pickFormat('mkv');
                      }}
                    />
                    <span>Put the subtitles into the video
                      <span className="hint block" data-testid="embed-hint">
                        {embedOnServer
                          ? 'Your browser can’t open this video type, so it’s uploaded whole and we put them in on our side.'
                          : `Done on your device ${kind ? 'when the job is finished' : 'once the subtitles are made'}, so nothing extra is uploaded. Players can turn them on and off. They go into an MKV file, the kind players show subtitles from${!serverClean && embedFormat && !canHoldSubtitles(embedFormat) ? `, so you’ll get an MKV instead of ${embedFormat.toUpperCase()}` : ''}. Video editors usually take the .srt file instead.`}
                      </span>
                    </span>
                  </label>
                )}
                {wantsText && videoResult && tooBigHere && <p className="small muted mb0" data-testid="embed-too-big">This browser can’t rebuild a video this big (over {gigabytes(limit)}); Chrome, Edge or Firefox can. You’ll get the subtitle files to load in your player or video editor.</p>}
              </div>
            )}

            {kind && duration && est != null && (
              <p className="mb0" data-testid="estimate">
                <b>Estimated cost: {money(est)}</b>{' '}
                <span className="muted">for {human(duration)} of audio{breakdown ? ` (cleaning ${money(breakdown[0])} + subtitles ${money(breakdown[1])})` : localSubs ? '; the subtitles made on your device are free' : ''}. You only pay for what gets processed. {BILLING_NOTE}</span>
              </p>
            )}
            {lengthUnknown && <p className="mb0 muted small" data-testid="length-unknown">Your browser can&rsquo;t read the length of this file type, so the numbers here are a rough guess from its size. We measure it after the upload and you pay for the real length.</p>}
            {problems.map((p) => (
              <div key={p} className="banner banner-warn" role="alert">
                {p} {p.includes('balance') && <Link to="/app/billing">Add credit</Link>}
                {p.includes('load the prices') && <button type="button" className="btn btn-ghost btn-sm" onClick={() => loadPricing().catch(() => {})}>Try again</button>}
              </div>
            ))}
          </>
        )}

        {busy && (
          <div className="stack" data-testid="run-progress">
            {localRunning && <ProgressPanel prog={localProg} stage={localStage} load={load} procEta={procEta} />}
            {subsRunning && <SubtitlesProgress prog={subsProg} load={speechLoad} />}
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
        {result && !busy && (
          <div className="banner banner-success" role="status" data-testid="submitted">
            Got it, we’re on it.{result.extracted ? ` Only the sound was uploaded (${bytes(result.uploadBytes)} instead of ${bytes(result.originalBytes)}).` : ''}{result.compressed ? ` ${compressedNote(result.compressed)}.` : ''} You can follow along below.
          </div>
        )}

        {file && info && !probing && (
          <div className="row">
            <button type="submit" className="btn btn-primary btn-lg" disabled={!canStart} data-testid="start">{startLabel}</button>
          </div>
        )}
      </form>

      {subsResult && (
        <SubtitlesResult
          file={subsResult.file} info={subsResult.info} files={subsResult.files} caps={subsResult.caps} embed={subsResult.embed}
          cleanPcm={subsResult.cleanPcm} onClose={() => setSubsResult(null)}
        />
      )}
      {localResult && (
        <BrowserResult
          file={localResult.file} info={localResult.info} result={localResult.result} isVideo={localResult.isVideo}
          keepVideo={localResult.keepVideo} onClose={() => setLocalResult(null)}
        />
      )}
    </>
  );
}
