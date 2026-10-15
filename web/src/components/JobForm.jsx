import { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../auth.jsx';
import { balanceMoney, BILLING_NOTE, bytes, clock, extension, human, money } from '../format.js';
import { estimateMicros, usePricing } from '../hooks.js';
import { AUDIO_FORMATS, LANGUAGES, LOUDNESS, submitJob } from '../jobs.js';
import { probeMedia } from '../media/mb.js';
import FileDrop from './FileDrop.jsx';

const STRENGTHS = [
  { id: '18', label: 'Natural', db: 18 },
  { id: '40', label: 'Strong', db: 40 },
  { id: '100', label: 'Maximum', db: 100 },
];

const compressedNote = (c) => `Compressed for upload: ${bytes(c.from)} → ${bytes(c.to)}`;

function defaultAudioFormat(file, info) {
  if (!info?.hasVideo) return 'wav';
  return extension(file.name) === 'webm' ? 'opus' : 'm4a';
}

const mins = (sec) => Math.max(1, Math.ceil(sec / 60));

/**
 * New-job form for the Files page: one file, then "Clean & polish" and/or "Subtitles & transcript".
 * Subtitles only + Standard + a file within the free limits is free (comes out of the daily free minutes).
 * onSubmitted({id, ...}, {file, info, extracted})
 */
export default function JobForm({ onSubmitted }) {
  const { balance, ensureTos, refresh, freeSecondsLeft } = useAuth();
  const { data: pricing } = usePricing();

  const [file, setFile] = useState(null);
  const [info, setInfo] = useState(null);
  const [probing, setProbing] = useState(false);
  const [polish, setPolish] = useState(true);
  const [text, setText] = useState(false);
  const [embed, setEmbed] = useState(false);
  const [tier, setTier] = useState('local'); // transcription: 'local' = Standard, 'premium' = Extra accurate (ids stay internal)
  const [format, setFormat] = useState('wav');
  const [formatTouched, setFormatTouched] = useState(false);
  const [loudness, setLoudness] = useState('-16');
  const [strength, setStrength] = useState('100');
  const [caps, setCaps] = useState({ srt: true, vtt: true, txt: true });
  const [language, setLanguage] = useState('');
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState(null); // {phase, frac, loaded, total}
  const [compressedInfo, setCompressedInfo] = useState(null); // {from, to} once the input was encoded to FLAC
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [errorKind, setErrorKind] = useState(''); // 'storage' while the server is short on space
  const abort = useRef(null);

  const wantsDenoise = polish;
  const wantsText = text;
  const kind = polish && text ? 'denoise_transcribe' : polish ? 'denoise' : text ? 'transcribe' : null;
  const duration = info?.duration || null;
  const maxJob = pricing?.maxJobSeconds;
  const freeMax = pricing?.free?.transcribeMaxFileSeconds;
  const freeDay = pricing?.free?.transcribeSecondsPerDay;
  const freeLeft = freeSecondsLeft || 0;
  const textOnlyStandard = text && !polish && tier === 'local';
  const free = !!(textOnlyStandard && duration && freeMax && duration <= freeMax && duration <= freeLeft);
  // Why a Standard subtitles-only job is not free (shown only when it is close to being free).
  let notFreeReason = '';
  if (textOnlyStandard && duration && !free) {
    if (freeMax && duration > freeMax) notFreeReason = `longer than the ${mins(freeMax)}-minute free limit`;
    else if (duration > freeLeft) notFreeReason = freeLeft > 0 ? `more than the ${mins(freeLeft)} free ${mins(freeLeft) === 1 ? 'minute' : 'minutes'} you have left today` : 'you have used all of today’s free minutes';
  }

  useEffect(() => { if (file && info && !formatTouched) setFormat(defaultAudioFormat(file, info)); }, [file, info, formatTouched]);
  useEffect(() => () => abort.current?.abort(), []);

  const est = useMemo(() => {
    if (free) return 0;
    if (!kind) return null;
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
    setProbing(true);
    setInfo(await probeMedia(f));
    setProbing(false);
  }

  const problems = [];
  if (info && !probing && !duration) problems.push('We couldn’t read the length of this file. Please choose a normal audio or video file (MP3, WAV, M4A, MP4, MOV, WebM, MKV...).');
  if (info?.hasAudio === false) problems.push('This file has no audio track.');
  if (duration && maxJob && duration > maxJob) problems.push(`Files can be at most ${Math.floor(maxJob / 3600)} hours long.`);
  if (!free && est != null && balance && est > balance.available_micros) problems.push(`This job costs about ${money(est)}, but your balance is ${balanceMoney(balance.available_micros)}.`);
  if (!kind) problems.push('Choose at least one: Clean & polish, or Subtitles & transcript.');
  if (wantsText && !Object.values(caps).some(Boolean)) problems.push('Choose at least one caption or transcript format.');
  const canStart = kind && file && info && !probing && !busy && problems.length === 0 && (free || est != null);

  async function submit(e) {
    e.preventDefault();
    if (!canStart) return;
    if (!free && !(await ensureTos())) {
      setError('Please accept the Terms of Service first, then we can get going.');
      return;
    }
    setBusy(true);
    setError('');
    setErrorKind('');
    setResult(null);
    setStep({ phase: 'create', frac: 0 });
    setCompressedInfo(null);
    const ctl = new AbortController();
    abort.current = ctl;
    const body = {
      kind,
      denoiseTier: wantsDenoise ? 'standard' : null,
      transcribeTier: wantsText ? tier : null,
      durationSeconds: duration || 1,
      free,
      options: {
        audio_format: format,
        caption_formats: wantsText ? ['srt', 'vtt', 'txt'].filter((c) => caps[c]) : [],
        embed_subtitles: !!(wantsText && embed && info?.hasVideo),
        language: language || null,
        loudness_lufs: LOUDNESS.find((l) => l.id === loudness)?.value ?? null,
        atten_lim_db: STRENGTHS.find((s) => s.id === strength).db,
      },
    };
    try {
      const out = await submitJob({
        file, info, body, signal: ctl.signal,
        onStage: (s) => setStep((p) => ({ ...p, phase: s, frac: 0 })),
        onProgress: (p) => setStep(p),
        onCompressed: setCompressedInfo,
      });
      setResult(out);
      onSubmitted?.(out, { file, info, extracted: out.extracted });
      await refresh(); // the free minutes left (and the balance) change as soon as a job exists
      setFile(null);
      setInfo(null);
    } catch (err) {
      if (err?.name !== 'AbortError') {
        setError(err.detail || err.message || 'Could not start the job.');
        setErrorKind(err?.status === 507 ? 'storage' : '');
      }
    } finally {
      setBusy(false);
      setStep(null);
    }
  }

  const pct = step?.frac != null ? Math.round(step.frac * 100) : null;
  return (
    <form className="card stack" onSubmit={submit} aria-labelledby="newjob-h">
      <h2 id="newjob-h" className="mb0">Add a file</h2>

      {!file && !busy && <FileDrop onFile={choose} />}
      {probing && <p role="status" className="muted"><span className="spinner" aria-hidden="true" /> Reading file…</p>}

      {file && (
        <div className="file-chip">
          <div className="grow">
            <div className="name">{file.name}</div>
            <div className="meta">{bytes(file.size)}{duration ? ` · ${clock(duration)}` : ''}{info?.hasVideo ? ' · video' : ''}</div>
          </div>
          {!busy && <button type="button" className="btn btn-ghost btn-sm" onClick={() => { setFile(null); setInfo(null); }}>Remove</button>}
        </div>
      )}
      {info?.hasVideo && info.viaMediabunny && <p className="small muted mb0">This is a video, so only its sound is uploaded. That keeps it small and quick, and the picture never leaves your device.</p>}

      <fieldset className="field" disabled={busy}>
        <legend className="label">What should we do?</legend>
        <div className="choice-cards">
          <label className="choice">
            <input type="checkbox" name="polish" checked={polish} onChange={(e) => setPolish(e.target.checked)} />
            <span><b>Clean &amp; polish</b><small>Noise removed, voice evened out, loudness set</small></span>
          </label>
          <label className="choice">
            <input type="checkbox" name="text" checked={text} onChange={(e) => setText(e.target.checked)} />
            <span><b>Subtitles &amp; transcript</b><small>Captions (SRT, VTT) and a text transcript</small></span>
          </label>
        </div>
        {polish && text && <span className="hint" data-testid="both-note">We write the transcript from your original recording (it comes out more accurate), and clean the sound separately.</span>}
      </fieldset>

      {file && info && (
        <>
          {wantsDenoise && (
            <div className="form-grid">
              <div className="field">
                <label htmlFor="strength">Noise reduction</label>
                <select id="strength" className="select" value={strength} onChange={(e) => setStrength(e.target.value)} disabled={busy}>
                  {STRENGTHS.map((s) => <option key={s.id} value={s.id}>{s.label} ({s.db} dB)</option>)}
                </select>
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
          )}

          {wantsText && (
            <div className="stack">
              <fieldset className="field" disabled={busy}>
                  <legend className="label">How accurate?</legend>
                  <div className="choice-cards">
                    <label className="choice">
                      <input type="radio" name="tier" value="local" checked={tier === 'local'} onChange={() => setTier('local')} />
                      <span><b>Standard</b><small>Good for most recordings{freeDay && !polish ? `, with ${mins(freeDay)} free minutes a day` : ''}</small></span>
                    </label>
                    <label className="choice">
                      <input type="radio" name="tier" value="premium" checked={tier === 'premium'} onChange={() => setTier('premium')} />
                      <span><b>Extra accurate</b><small>Best for accents, noisy rooms and specialist words</small></span>
                    </label>
                  </div>
                  {tier === 'premium' && <span className="hint" data-testid="accurate-note">Extra accurate sends your audio to a speech partner we work with, only for this option. <Link to="/privacy">Privacy details</Link></span>}
                </fieldset>
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
              {info?.hasVideo && (
                <label className="check">
                  <input type="checkbox" name="embed" checked={embed} onChange={(e) => setEmbed(e.target.checked)} disabled={busy} />
                  <span>Embed the subtitles in the video{embed ? ' (your whole video is uploaded, so it takes longer)' : ''}</span>
                </label>
              )}
            </div>
          )}

          {!free && kind && duration && est != null && (
            <p className="mb0" data-testid="estimate">
              <b>Estimated cost: {money(est)}</b>{' '}
              <span className="muted">for {human(duration)} of audio{breakdown ? ` (clean & polish ${money(breakdown[0])} + subtitles ${money(breakdown[1])})` : ''}. You only pay for what gets processed. {BILLING_NOTE}</span>
            </p>
          )}
          {free && (
            <p className="mb0" data-testid="free-note">
              <b>Free: uses {mins(duration)} of your {Math.floor(freeLeft / 60)} free minutes left today</b>
            </p>
          )}
          {notFreeReason && <p className="mb0 muted small" data-testid="not-free-note">This one isn’t free because it’s {notFreeReason}, so it’s charged at the usual rate.</p>}
          {problems.map((p) => <div key={p} className="banner banner-warn" role="alert">{p} {p.includes('balance') && <Link to="/app/billing">Add credit</Link>}</div>)}
        </>
      )}

      {busy && step && (
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
          <div><button type="button" className="btn btn-secondary btn-sm" onClick={() => abort.current?.abort()}>Cancel</button></div>
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

      <div className="row">
        <button type="submit" className="btn btn-primary btn-lg" disabled={!canStart}>
          {free ? 'Start (free)' : est ? `Start job (${money(est)})` : 'Start job'}
        </button>
      </div>
    </form>
  );
}
