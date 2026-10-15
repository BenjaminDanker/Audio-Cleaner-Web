import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../auth.jsx';
import { balanceMoney, BILLING_NOTE, bytes, clock, extension, human, money, moneyUsd } from '../format.js';
import { estimateMicros, findRate, usePricing } from '../hooks.js';
import { AUDIO_FORMATS, LANGUAGES, LOUDNESS, NOISE, submitJob } from '../jobs.js';
import { denoiserLoadState, preloadDenoiser, subscribeDenoiserLoad } from '../media/denoise.js';
import { probeMedia } from '../media/mb.js';
import { denoiseFile } from '../media/pipeline.js';
import { buildWav } from '../media/wav.js';
import { BrowserResult, ProgressPanel } from './BrowserClean.jsx';
import FileDrop, { SERVER_FORMATS } from './FileDrop.jsx';
import StrengthPicker from './StrengthPicker.jsx';
import { strengthDb } from '../strengths.js';


const LONG_SECONDS = 2 * 3600; // in the browser, a recording this long takes a while
const MAX_SECONDS = 6 * 3600; // and this is the most the browser will take

const supported = typeof Worker !== 'undefined' && typeof WebAssembly !== 'undefined';

const compressedNote = (c) => `Compressed for upload: ${bytes(c.from)} → ${bytes(c.to)}`;

function defaultAudioFormat(file, info) {
  if (!info?.hasVideo) return 'wav';
  return extension(file.name) === 'webm' ? 'opus' : 'm4a';
}

const mins = (sec) => Math.max(1, Math.ceil(sec / 60));
const minutesWord = (n) => `${n} ${n === 1 ? 'minute' : 'minutes'}`;

/** Can this browser do the noise removal for this file? `why` is the plain-words reason when it can't. */
function browserVerdict(info) {
  const paid = 'Clean & polish can (paid).';
  if (!supported) return { ok: false, why: `This browser can’t run the noise remover. ${paid}` };
  if (info.hasAudio === false) return { ok: false, why: 'This file has no audio track.' };
  const readable = info.canDecode !== false && (info.viaMediabunny || info.viaLibav || info.duration != null);
  if (!readable) return { ok: false, why: `Your browser can’t open this file type. ${paid}` };
  if (info.duration && info.duration > MAX_SECONDS) return { ok: false, why: `This is longer than ${MAX_SECONDS / 3600} hours, which is too long for your browser. ${paid}` };
  return { ok: true, why: '' };
}

/** One radio card. A card that can't be used stays visible, greyed out, with the reason where its hint would be. */
function Choice({ name, value, current, onPick, disabled, title, hint, reason }) {
  return (
    <label className={`choice${disabled ? ' is-disabled' : ''}`}>
      <input type="radio" name={name} value={value} checked={current === value} onChange={() => onPick(value)} disabled={disabled} />
      <span><b>{title}</b><small>{disabled && reason ? reason : hint}</small></span>
    </label>
  );
}

/**
 * The Cleanup form: drop one file, then choose what to do with it. Only what is possible for that file is offered.
 *   Noise removal:  Off | Free, in your browser (nothing uploaded) | Clean & polish (paid, on our side)
 *   Subtitles:      Off | Free (standard) | Standard | Extra accurate
 * Any combination works: the browser part runs here, in parallel with the job (if any) for the rest.
 * onSubmitted({id, ...}, {file, info, extracted, keepVideo})
 */
export default function JobForm({ onSubmitted }) {
  const { balance, refresh, freeSecondsLeft } = useAuth();
  const { data: pricing } = usePricing();
  const load = useSyncExternalStore(subscribeDenoiserLoad, denoiserLoadState);

  const [file, setFile] = useState(null);
  const [info, setInfo] = useState(null);
  const [probing, setProbing] = useState(false);
  const [noise, setNoise] = useState('off'); // 'off' | 'browser' | 'polish'
  const [subs, setSubs] = useState('off'); // 'off' | 'free' | 'standard' | 'accurate'
  const [keepVideo, setKeepVideo] = useState(true);
  const [embed, setEmbed] = useState(false);
  const [echo, setEcho] = useState(false); // Clean & polish: remove speaker echo using the second audio track
  const [format, setFormat] = useState('wav');
  const [formatTouched, setFormatTouched] = useState(false);
  const [loudness, setLoudness] = useState('-16');
  const [noiseKind, setNoiseKind] = useState('auto');
  const [strength, setStrength] = useState('max'); // Clean & polish
  const [localStrength, setLocalStrength] = useState('max'); // in the browser
  const [caps, setCaps] = useState({ srt: true, vtt: true, txt: true });
  const [language, setLanguage] = useState('');
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState(null); // upload: {phase, frac, loaded, total}
  const [localRunning, setLocalRunning] = useState(false);
  const [localProg, setLocalProg] = useState(null);
  const [localStage, setLocalStage] = useState('');
  const [localResult, setLocalResult] = useState(null); // {file, info, isVideo, keepVideo, result}
  const [compressedInfo, setCompressedInfo] = useState(null); // {from, to} once the input was encoded to FLAC
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [errorKind, setErrorKind] = useState(''); // 'storage' while the server is short on space
  const abort = useRef(null);

  // Start fetching the noise remover (about 22 MB, once, at low priority) as soon as this page opens, so it is usually
  // ready by the time a file has been chosen. Nothing is downloaded on any other page.
  useEffect(() => {
    if (supported) preloadDenoiser().catch(() => {}); // a failure is reported when cleaning starts
  }, []);
  useEffect(() => () => abort.current?.abort(), []);
  useEffect(() => () => { if (localResult?.result?.url) URL.revokeObjectURL(localResult.result.url); }, [localResult]);
  useEffect(() => {
    if (!busy) return undefined;
    const warn = (e) => { e.preventDefault(); e.returnValue = ''; };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [busy]);

  const verdict = info ? browserVerdict(info) : null;
  const wantsBrowser = noise === 'browser';
  const wantsPolish = noise === 'polish';
  const wantsText = subs !== 'off';
  const tier = subs === 'accurate' ? 'premium' : 'local'; // transcription level ids stay internal
  const kind = wantsPolish && wantsText ? 'denoise_transcribe' : wantsPolish ? 'denoise' : wantsText ? 'transcribe' : null;
  const free = subs === 'free' && !wantsPolish;
  const isVideo = !!info?.hasVideo;
  const hasSecondTrack = (info?.audioTracks || 0) >= 2; // OBS and others can record mic and desktop sound as two tracks

  // Some formats (WMA, AMR, AVI, MPEG...) the browser can't read, but the server can. Then the length is a rough
  // guess from the file size (128 kbit/s) until the server measures it; the price is corrected to the real length.
  const lengthUnknown = !!(file && info && !probing && !info.duration && info.hasAudio !== false);
  const maxJob = pricing?.maxJobSeconds;
  const duration = info?.duration || (lengthUnknown ? Math.min(Math.max(30, (file.size * 8) / 128000), maxJob || 4 * 3600) : null);
  const freeMax = pricing?.free?.transcribeMaxFileSeconds;
  const freeDay = pricing?.free?.transcribeSecondsPerDay;
  const freeLeft = freeSecondsLeft || 0;
  const polishRate = findRate(pricing, 'denoise', 'standard');
  const standardRate = findRate(pricing, 'transcribe', 'local');

  // Free subtitles: only for a file within the per-file limit and today's free minutes, and not together with Clean & polish.
  const freeSubs = useMemo(() => {
    if (!duration) return { ok: false, why: 'We need the length of the file first.' };
    if (!freeMax) return { ok: false, why: 'We couldn’t load the free limits just now.' };
    if (duration > freeMax) return { ok: false, why: `This file is longer than the ${mins(freeMax)}-minute free limit.` };
    if (duration > freeLeft) {
      return { ok: false, why: freeLeft > 0 ? `This is more than the ${minutesWord(mins(freeLeft))} of free time you have left today.` : 'You’ve used all of today’s free minutes.' };
    }
    if (wantsPolish) return { ok: false, why: 'Free subtitles can’t be added to Clean & polish. Standard can.' };
    return { ok: true, why: '' };
  }, [duration, freeMax, freeLeft, wantsPolish]);

  const procEta = useMemo(() => {
    if (!info?.duration) return null;
    const channels = info.channels && info.channels >= 2 ? 2 : 1;
    const cores = navigator.hardwareConcurrency || 2;
    return info.duration * 0.13 * (channels === 2 && cores < 3 ? 2 : 1);
  }, [info]);
  // Whatever is still downloading counts too, so the estimate is honest if you start straight away.
  const loadLeft = load.done ? 0 : load.secondsLeft;
  const eta = procEta == null ? null : procEta + (loadLeft || 0);

  // Free subtitles that stop being possible (another file, the free minutes ran out) are switched off, never silently charged.
  useEffect(() => { if (subs === 'free' && info && !freeSubs.ok) setSubs('off'); }, [subs, info, freeSubs]);
  useEffect(() => { if (file && info && !formatTouched) setFormat(defaultAudioFormat(file, info)); }, [file, info, formatTouched]);

  const est = useMemo(() => {
    if (!kind) return 0;
    if (free) return 0;
    return estimateMicros(pricing, { kind, transcribeTier: tier, seconds: duration });
  }, [pricing, kind, tier, duration, free]);
  const breakdown = useMemo(() => {
    if (free || kind !== 'denoise_transcribe' || !duration) return null;
    return [estimateMicros(pricing, { kind: 'denoise', seconds: duration }), estimateMicros(pricing, { kind: 'transcribe', transcribeTier: tier, seconds: duration })];
  }, [pricing, kind, tier, duration, free]);

  async function choose(f) {
    setFile(f);
    setInfo(null);
    setResult(null);
    setError('');
    setFormatTouched(false);
    setEmbed(false);
    setEcho(false);
    setKeepVideo(true);
    setSubs('off');
    setNoise('off');
    setProbing(true);
    const i = await probeMedia(f);
    // The best choice that works for this file; a paid one is never switched on without the price being right there.
    setNoise(browserVerdict(i).ok ? 'browser' : i.hasAudio === false ? 'off' : 'polish');
    setInfo(i);
    setProbing(false);
  }

  function pickNoise(v) {
    setNoise(v);
    if (v === 'polish' && subs === 'free') setSubs('standard'); // free subtitles can't go with Clean & polish
  }

  function reset() {
    abort.current?.abort();
    setFile(null);
    setInfo(null);
    setError('');
  }

  const problems = [];
  if (info?.hasAudio === false) problems.push('This file has no audio track.');
  if (kind && duration && maxJob && duration > maxJob) problems.push(`Files can be at most ${Math.floor(maxJob / 3600)} hours long.`);
  if (kind && !free && est != null && balance && est > balance.available_micros) problems.push(`This costs about ${money(est)}, but your balance is ${balanceMoney(balance.available_micros)}.`);
  if (info && !wantsBrowser && !kind) problems.push('Choose at least one thing to do: noise removal, subtitles, or both.');
  if (wantsText && !Object.values(caps).some(Boolean)) problems.push('Choose at least one caption or transcript format.');
  const canStart = (wantsBrowser || kind) && file && info && !probing && !busy && problems.length === 0 && (!kind || free || est != null);

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

  async function runJob(f, i, signal) {
    setStep({ phase: 'create', frac: 0 });
    const body = {
      kind,
      denoiseTier: wantsPolish ? 'standard' : null,
      transcribeTier: wantsText ? tier : null,
      durationSeconds: duration || 1,
      free,
      options: {
        audio_format: format,
        caption_formats: wantsText ? ['srt', 'vtt', 'txt'].filter((c) => caps[c]) : [],
        embed_subtitles: !!(wantsText && embed && i?.hasVideo),
        language: language || null,
        loudness_lufs: LOUDNESS.find((l) => l.id === loudness)?.value ?? null,
        noise: noiseKind,
        atten_lim_db: strengthDb(strength),
        echo_reference_track: !!(wantsPolish && echo && hasSecondTrack),
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
      onSubmitted?.(out, { file: f, info: i, extracted: out.extracted, keepVideo });
      return out;
    } finally {
      setStep(null);
    }
  }

  async function submit(e) {
    e.preventDefault();
    if (!canStart) return;
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

    // The browser part and the job (if there is one) run side by side; each reports on its own.
    const outcomes = await Promise.allSettled([
      wantsBrowser ? runLocal(f, i, ctl.signal) : Promise.resolve(null),
      kind ? runJob(f, i, ctl.signal) : Promise.resolve(null),
    ]);
    const [local, job] = outcomes;
    const canceled = ctl.signal.aborted || outcomes.some((o) => o.status === 'rejected' && o.reason?.name === 'AbortError');
    let failed = false;
    if (local.status === 'fulfilled' && local.value) {
      setLocalResult({ file: f, info: i, isVideo: !!i?.hasVideo, keepVideo: keepVideo && !i?.viaLibav, result: local.value });
    } else if (local.status === 'rejected' && local.reason?.name !== 'AbortError') {
      failed = true;
      setError(local.reason?.message || 'Something went wrong while cleaning this file. Please try again.');
    }
    if (job.status === 'rejected' && job.reason?.name !== 'AbortError') {
      failed = true;
      const err = job.reason;
      setError(err.detail || err.message || 'Could not start the job.');
      setErrorKind(err?.status === 507 ? 'storage' : '');
    }
    if (kind) await refresh(); // the free minutes left (and the balance) change as soon as a job exists
    setBusy(false);
    setLocalRunning(false);
    setStep(null);
    if (!failed && !canceled) {
      setFile(null);
      setInfo(null);
    }
  }

  const pct = step?.frac != null ? Math.round(step.frac * 100) : null;
  const left = Math.floor(freeLeft / 60);
  const startLabel = !(wantsBrowser || kind) ? 'Start' : free || !kind ? 'Start - free' : est ? `Start - about ${money(est)}` : 'Start';
  return (
    <>
      <form className="card stack" onSubmit={submit} aria-labelledby="newjob-h">
        <h2 id="newjob-h" className="mb0">Add a file</h2>

        {!file && !busy && <FileDrop onFile={choose} accept={SERVER_FORMATS} hint="or click to pick one. Almost any audio or video file: MP3, WAV, M4A, FLAC, WMA, AMR, MP4, MOV, MKV, AVI, WMV, MPEG and more." />}
        {probing && <p role="status" className="muted"><span className="spinner" aria-hidden="true" /> Reading {file?.name}…</p>}

        {file && info && (
          <div className="file-chip">
            <div className="grow">
              <div className="name">{file.name}</div>
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
            <fieldset className="field" disabled={busy}>
              <legend className="label">Noise removal</legend>
              <div className="choice-cards">
                <Choice name="noise" value="off" current={noise} onPick={pickNoise} title="Off" hint="Leave the sound as it is" />
                <Choice
                  name="noise" value="browser" current={noise} onPick={pickNoise} disabled={!verdict.ok}
                  title="Free, in your browser" hint="Done on your own device. Nothing is uploaded." reason={verdict.why}
                />
                <Choice
                  name="noise" value="polish" current={noise} onPick={pickNoise} disabled={info.hasAudio === false}
                  title="Clean &amp; polish" hint={`Paid, done on our side: noise removed, voice evened out, loudness set${polishRate ? `. From ${moneyUsd(polishRate.usdPerHour)} an hour` : ''}`}
                  reason="This file has no audio track."
                />
              </div>
            </fieldset>

            {wantsBrowser && (
              <div className="stack">
                <StrengthPicker name="lstrength" value={localStrength} onChange={setLocalStrength} disabled={busy} />
                {info.channels > 2 && <p className="small muted mb0">This file has {info.channels} channels; they will be mixed down to mono.</p>}
                {info.duration > LONG_SECONDS && (
                  <div className="banner banner-warn">
                    This is a long recording ({human(info.duration)}). Cleaning it here could take about {human(eta)} and needs a lot of memory.
                    Keep this tab open, or choose Clean &amp; polish for long files.
                  </div>
                )}
                {eta && <p className="muted small mb0" data-testid="eta">Should take about {human(eta)} on this device{loadLeft ? ' (including getting the noise remover ready)' : ''}.</p>}
              </div>
            )}

            {wantsPolish && (
              <div className="stack">
              <StrengthPicker name="strength" value={strength} onChange={setStrength} disabled={busy} />
              <div className="form-grid">
                <div className="field">
                  <label htmlFor="noisekind">Background noise</label>
                  <select id="noisekind" className="select" value={noiseKind} onChange={(e) => setNoiseKind(e.target.value)} disabled={busy}>
                    {NOISE.map((n) => <option key={n.id} value={n.id}>{n.label}</option>)}
                  </select>
                  <span className="hint">{NOISE.find((n) => n.id === noiseKind).hint}</span>
                </div>
                <div className="field">
                  <label htmlFor="afmt">Cleaned audio format</label>
                  <select id="afmt" className="select" value={format} onChange={(e) => { setFormat(e.target.value); setFormatTouched(true); }} disabled={busy}>
                    {AUDIO_FORMATS.map((f) => <option key={f.id} value={f.id}>{f.label}</option>)}
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="loud">Loudness target</label>
                  <select id="loud" className="select" value={loudness} onChange={(e) => setLoudness(e.target.value)} disabled={busy}>
                    {LOUDNESS.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
                  </select>
                </div>
              </div>
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

            {hasSecondTrack && wantsBrowser && <p className="small muted mb0" data-testid="echo-browser-note">This file has a second audio track (your desktop sound). Removing speaker echo with it needs Clean &amp; polish; the free cleanup in your browser can&rsquo;t do it.</p>}

            {isVideo && wantsBrowser && info.viaLibav && <p className="small muted mb0" data-testid="audio-only-note">Your browser can clean the sound of this video, but can&rsquo;t rebuild this video type, so you&rsquo;ll get the cleaned sound as a WAV file. Clean &amp; polish gives you the whole video back.</p>}
            {isVideo && noise !== 'off' && !(wantsBrowser && info.viaLibav) && (
              <label className="check">
                <input type="checkbox" name="keepvideo" checked={keepVideo} onChange={(e) => setKeepVideo(e.target.checked)} disabled={busy} />
                <span>Keep the video, with the cleaned sound{wantsPolish ? ' (put back together on your device afterwards)' : ''}</span>
              </label>
            )}

            <fieldset className="field" disabled={busy}>
              <legend className="label">Subtitles &amp; transcript</legend>
              <div className="choice-cards">
                <Choice name="subs" value="off" current={subs} onPick={setSubs} title="Off" hint="No subtitles or transcript" />
                <Choice
                  name="subs" value="free" current={subs} onPick={setSubs} disabled={!freeSubs.ok}
                  title="Free (standard)" reason={freeSubs.why}
                  hint={`Good for most recordings. Free for files up to ${freeMax ? minutesWord(mins(freeMax)) : 'a few minutes'}; you have ${minutesWord(left)} left today.`}
                />
                <Choice
                  name="subs" value="standard" current={subs} onPick={setSubs}
                  title="Standard" hint={`Paid. Good for most recordings${standardRate ? `, ${moneyUsd(standardRate.usdPerHour)} an hour` : ''}`}
                />
                <Choice
                  name="subs" value="accurate" current={subs} onPick={setSubs}
                  title="Extra accurate" hint="Paid. Best for accents, noisy rooms and specialist words. Works in any language."
                />
              </div>
              {subs === 'accurate' && <span className="hint" data-testid="accurate-note">Extra accurate sends your audio to a speech partner we work with, only for this option. <Link to="/privacy">Privacy details</Link></span>}
              {wantsPolish && wantsText && <span className="hint" data-testid="both-note">We write the transcript from your original recording (it comes out more accurate), and clean the sound separately.</span>}
              {wantsBrowser && wantsText && <span className="hint" data-testid="browser-subs-note">The noise removal happens on your device. For the subtitles, the original sound is uploaded.</span>}
            </fieldset>

            {wantsText && (
              <div className="stack">
                <div className="form-grid">
                  <div className="field">
                    <label htmlFor="lang">Language</label>
                    <select id="lang" className="select" value={language} onChange={(e) => setLanguage(e.target.value)} disabled={busy}>
                      {LANGUAGES.map(([c, n]) => <option key={c} value={c}>{n}</option>)}
                    </select>
                  </div>
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
                {isVideo && (
                  <label className="check">
                    <input type="checkbox" name="embed" checked={embed} onChange={(e) => setEmbed(e.target.checked)} disabled={busy} />
                    <span>Embed subtitles in the video{embed ? ' (your whole video is uploaded, so it takes longer)' : ''}</span>
                  </label>
                )}
              </div>
            )}

            {kind && !free && duration && est != null && (
              <p className="mb0" data-testid="estimate">
                <b>Estimated cost: {money(est)}</b>{' '}
                <span className="muted">for {human(duration)} of audio{breakdown ? ` (clean & polish ${money(breakdown[0])} + subtitles ${money(breakdown[1])})` : ''}. You only pay for what gets processed. {BILLING_NOTE}</span>
              </p>
            )}
            {free && (
              <p className="mb0" data-testid="free-note">
                <b>Free: uses {lengthUnknown ? 'about ' : ''}{mins(duration)} of your {left} free minutes left today</b>
              </p>
            )}
            {lengthUnknown && <p className="mb0 muted small" data-testid="length-unknown">Your browser can&rsquo;t read the length of this file type, so the numbers here are a rough guess from its size. We measure it after the upload and you pay for the real length.</p>}
            {problems.map((p) => <div key={p} className="banner banner-warn" role="alert">{p} {p.includes('balance') && <Link to="/app/billing">Add credit</Link>}</div>)}
          </>
        )}

        {busy && (
          <div className="stack" data-testid="run-progress">
            {localRunning && <ProgressPanel prog={localProg} stage={localStage} load={load} procEta={procEta} />}
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
              {wantsBrowser && <span className="muted small">Keep this tab open while it works.</span>}
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

      {localResult && (
        <BrowserResult
          file={localResult.file} info={localResult.info} result={localResult.result} isVideo={localResult.isVideo}
          keepVideo={localResult.keepVideo} onClose={() => setLocalResult(null)}
        />
      )}
    </>
  );
}
