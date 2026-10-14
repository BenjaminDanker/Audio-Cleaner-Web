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

const KINDS = [
  { id: 'denoise', title: 'Studio clean', sub: 'Remove noise and level the voice' },
  { id: 'transcribe', title: 'Transcription', sub: 'Text and subtitles' },
  { id: 'denoise_transcribe', title: 'Both', sub: 'Clean first, then transcribe the clean audio' },
];

const compressedNote = (c) => `Compressed for upload: ${bytes(c.from)} → ${bytes(c.to)}`;

function defaultAudioFormat(file, info) {
  if (!info?.hasVideo) return 'wav';
  return extension(file.name) === 'webm' ? 'opus' : 'm4a';
}

/**
 * New-job form. variant="studio": paid jobs. variant="free": free standard transcription (daily allowance).
 * onSubmitted({id, ...}, {file, info, extracted})
 */
export default function JobForm({ variant = 'studio', onSubmitted }) {
  const free = variant === 'free';
  const { balance, ensureTos, refresh } = useAuth();
  const { data: pricing } = usePricing();

  const [file, setFile] = useState(null);
  const [info, setInfo] = useState(null);
  const [probing, setProbing] = useState(false);
  const [kind, setKind] = useState(free ? 'transcribe' : 'denoise');
  const [tier, setTier] = useState('local'); // transcription: local = standard, premium
  const [format, setFormat] = useState('wav');
  const [formatTouched, setFormatTouched] = useState(false);
  const [loudness, setLoudness] = useState('-16');
  const [strength, setStrength] = useState('100');
  const [caps, setCaps] = useState({ srt: true, vtt: true, txt: true, json: false });
  const [language, setLanguage] = useState('');
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState(null); // {phase, frac, loaded, total}
  const [compressedInfo, setCompressedInfo] = useState(null); // {from, to} once the input was encoded to FLAC
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const abort = useRef(null);

  const wantsDenoise = kind === 'denoise' || kind === 'denoise_transcribe';
  const wantsText = kind === 'transcribe' || kind === 'denoise_transcribe';
  const duration = info?.duration || null;
  const maxJob = pricing?.maxJobSeconds;
  const freeMax = pricing?.free?.transcribeMaxFileSeconds;

  useEffect(() => { if (file && info && !formatTouched) setFormat(defaultAudioFormat(file, info)); }, [file, info, formatTouched]);
  useEffect(() => () => abort.current?.abort(), []);

  const est = useMemo(() => {
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
    setProbing(true);
    setInfo(await probeMedia(f));
    setProbing(false);
  }

  const problems = [];
  if (info && !probing && !duration) problems.push('We could not read the length of this file. Please choose a normal audio or video file (MP3, WAV, M4A, MP4, MOV, WebM, MKV...).');
  if (info?.hasAudio === false) problems.push('This file has no audio track.');
  if (duration && maxJob && duration > maxJob) problems.push(`Files can be at most ${Math.floor(maxJob / 3600)} hours long.`);
  if (free && duration && freeMax && duration > freeMax) problems.push(`Free transcription is limited to ${Math.round(freeMax / 60)} minutes per file. Use a shorter file, or paid transcription in Studio.`);
  if (!free && est != null && balance && est > balance.available_micros) problems.push(`This job costs about ${money(est)} but your available balance is ${balanceMoney(balance.available_micros)}.`);
  if (wantsText && !Object.values(caps).some(Boolean)) problems.push('Choose at least one caption or transcript format.');
  const canStart = file && info && !probing && !busy && problems.length === 0 && (free || est != null);

  async function submit(e) {
    e.preventDefault();
    if (!canStart) return;
    if (!free && !(await ensureTos())) {
      setError('Please accept the Terms of Service to continue.');
      return;
    }
    setBusy(true);
    setError('');
    setResult(null);
    setStep({ phase: 'create', frac: 0 });
    setCompressedInfo(null);
    const ctl = new AbortController();
    abort.current = ctl;
    const body = {
      kind,
      denoiseTier: wantsDenoise ? 'standard' : null,
      transcribeTier: wantsText ? (free ? 'local' : tier) : null,
      durationSeconds: duration || 1,
      free,
      options: {
        audio_format: format,
        caption_formats: ['srt', 'vtt', 'txt', 'json'].filter((c) => caps[c]),
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
      refresh();
      setFile(null);
      setInfo(null);
    } catch (err) {
      if (err?.name !== 'AbortError') setError(err.detail || err.message || 'Could not start the job.');
    } finally {
      setBusy(false);
      setStep(null);
    }
  }

  const pct = step?.frac != null ? Math.round(step.frac * 100) : null;
  return (
    <form className="card stack" onSubmit={submit} aria-labelledby="newjob-h">
      <h2 id="newjob-h" className="mb0">{free ? 'Transcribe a file for free' : 'New job'}</h2>

      {!free && (
        <fieldset className="field" disabled={busy}>
          <legend className="label">What do you want?</legend>
          <div className="choice-cards">
            {KINDS.map((k) => (
              <label className="choice" key={k.id}>
                <input type="radio" name="kind" value={k.id} checked={kind === k.id} onChange={() => setKind(k.id)} />
                <span><b>{k.title}</b><small>{k.sub}</small></span>
              </label>
            ))}
          </div>
        </fieldset>
      )}

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
      {info?.hasVideo && info.viaMediabunny && <p className="small muted mb0">This is a video: only its audio track will be uploaded, which keeps the upload small. Your video never leaves this device.</p>}

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
              {!free && (
                <fieldset className="field" disabled={busy}>
                  <legend className="label">Transcription quality</legend>
                  <div className="choice-cards">
                    <label className="choice">
                      <input type="radio" name="tier" value="local" checked={tier === 'local'} onChange={() => setTier('local')} />
                      <span><b>Standard</b><small>On-prem model. Private, fast, affordable.</small></span>
                    </label>
                    <label className="choice">
                      <input type="radio" name="tier" value="premium" checked={tier === 'premium'} onChange={() => setTier('premium')} />
                      <span><b>Premium</b><small>Higher accuracy. Audio is sent to OpenAI as our sub-processor.</small></span>
                    </label>
                  </div>
                </fieldset>
              )}
              <div className="form-grid">
                <div className="field">
                  <label htmlFor="lang">Language</label>
                  <select id="lang" className="select" value={language} onChange={(e) => setLanguage(e.target.value)} disabled={busy}>
                    {LANGUAGES.map(([c, n]) => <option key={c} value={c}>{n}</option>)}
                  </select>
                </div>
                <fieldset className="field" disabled={busy}>
                  <legend className="label">Caption and transcript formats</legend>
                  <div className="checks">
                    {[['srt', 'SRT'], ['vtt', 'VTT'], ['txt', 'Text'], ['json', 'JSON']].map(([id, label]) => (
                      <label className="check" key={id}>
                        <input type="checkbox" checked={caps[id]} onChange={(e) => setCaps({ ...caps, [id]: e.target.checked })} />
                        <span>{label}</span>
                      </label>
                    ))}
                  </div>
                </fieldset>
              </div>
            </div>
          )}

          {!free && duration && est != null && (
            <p className="mb0" data-testid="estimate">
              <b>Estimated cost: {money(est)}</b>{' '}
              <span className="muted">for {human(duration)} of audio{breakdown ? ` (clean ${money(breakdown[0])} + transcription ${money(breakdown[1])})` : ''}. You are only charged for what is processed. {BILLING_NOTE}</span>
            </p>
          )}
          {free && <p className="mb0 muted small">Free: counts against your daily allowance{pricing?.free ? ` of ${Math.round(pricing.free.transcribeSecondsPerDay / 60)} minutes` : ''}. Longer files and the premium model are in <Link to="/app">Studio</Link>.</p>}
          {problems.map((p) => <div key={p} className="banner banner-warn" role="alert">{p} {p.includes('balance') && <Link to="/app/billing">Add credit</Link>}</div>)}
        </>
      )}

      {busy && step && (
        <div className="stack" data-testid="upload-progress">
          <div className="row-between">
            <strong>
              {step.phase === 'extract' && 'Extracting the audio from your video…'}
              {step.phase === 'compress' && 'Compressing for upload…'}
              {step.phase === 'create' && 'Creating job…'}
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

      {error && <div className="banner banner-error" role="alert">{error}</div>}
      {result && !busy && (
        <div className="banner banner-success" role="status" data-testid="submitted">
          Job started.{result.extracted ? ` Only the audio was uploaded (${bytes(result.uploadBytes)} instead of ${bytes(result.originalBytes)}).` : ''}{result.compressed ? ` ${compressedNote(result.compressed)}.` : ''} Follow it below.
        </div>
      )}

      <div className="row">
        <button type="submit" className="btn btn-primary btn-lg" disabled={!canStart}>
          {free ? 'Start free transcription' : est ? `Start job (${money(est)})` : 'Start job'}
        </button>
      </div>
    </form>
  );
}
