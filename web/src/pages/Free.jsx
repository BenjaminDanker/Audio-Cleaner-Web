import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { Link } from 'react-router-dom';
import AbPlayer from '../components/AbPlayer.jsx';
import FileDrop from '../components/FileDrop.jsx';
import { IconDownload, IconLock } from '../components/Icons.jsx';
import { saveBlob } from '../download.js';
import { bytes, clock, human, moneyUsd, stem } from '../format.js';
import { findRate, usePageTitle, usePricing } from '../hooks.js';
import { denoiserLoadState, preloadDenoiser, STRENGTHS, subscribeDenoiserLoad } from '../media/denoise.js';
import { probeMedia } from '../media/mb.js';
import { denoiseFile, STAGE_LOADING } from '../media/pipeline.js';
import { buildWav } from '../media/wav.js';

const LONG_SECONDS = 2 * 3600;
const MAX_SECONDS = 6 * 3600;

const supported = typeof Worker !== 'undefined' && typeof WebAssembly !== 'undefined';

export default function Free() {
  usePageTitle('Clean up a recording');
  const { data: pricing } = usePricing();
  const polish = findRate(pricing, 'denoise', 'standard');

  const [file, setFile] = useState(null);
  const [info, setInfo] = useState(null);
  const [phase, setPhase] = useState('idle'); // idle | probing | ready | working | done
  const [strength, setStrength] = useState('strong');
  const [prog, setProg] = useState(null);
  const [stage, setStage] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const abort = useRef(null);
  const load = useSyncExternalStore(subscribeDenoiserLoad, denoiserLoadState);

  // Start fetching the noise remover (about 22 MB, once, at low priority) as soon as this page opens, so it is usually
  // ready by the time a file has been chosen. Nothing is downloaded on any other page.
  useEffect(() => {
    if (supported) preloadDenoiser().catch(() => {}); // a failure is reported when cleaning starts
  }, []);

  const isVideo = !!info?.hasVideo;

  const choose = useCallback(async (f) => {
    abort.current?.abort();
    setFile(f);
    setInfo(null);
    setResult(null);
    setError('');
    setProg(null);
    setPhase('probing');
    const i = await probeMedia(f);
    setInfo(i);
    setPhase('ready');
  }, []);

  const reset = () => {
    abort.current?.abort();
    setFile(null);
    setInfo(null);
    setResult(null);
    setError('');
    setProg(null);
    setPhase('idle');
  };

  async function start() {
    const ctl = new AbortController();
    abort.current = ctl;
    setError('');
    setResult(null);
    setProg({ done: 0, total: info?.duration || null, rate: 0, eta: null, elapsed: 0 });
    setPhase('working');
    try {
      const atten = STRENGTHS.find((s) => s.id === strength).db;
      const pcm = await denoiseFile(file, {
        attenDb: atten,
        durationHint: info?.duration || null,
        signal: ctl.signal,
        onProgress: setProg,
        onStage: setStage,
      });
      setStage('Preparing the preview…');
      const wav = await buildWav(pcm, 16);
      if (ctl.signal.aborted) throw new DOMException('Canceled', 'AbortError');
      setResult({ pcm, wav, url: URL.createObjectURL(wav), strength });
      setPhase('done');
    } catch (e) {
      if (e?.name === 'AbortError' || ctl.signal.aborted) {
        setPhase('ready');
        return;
      }
      setError(e?.message || 'Something went wrong while cleaning this file. Please try again.');
      setPhase('ready');
    }
  }

  useEffect(() => () => abort.current?.abort(), []);
  useEffect(() => () => { if (result?.url) URL.revokeObjectURL(result.url); }, [result]);
  useEffect(() => {
    if (phase !== 'working') return undefined;
    const warn = (e) => { e.preventDefault(); e.returnValue = ''; };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [phase]);

  const duration = info?.duration;
  const tooLong = duration && duration > MAX_SECONDS;
  const procEta = useMemo(() => {
    if (!duration) return null;
    const channels = info?.channels && info.channels >= 2 ? 2 : 1;
    const cores = navigator.hardwareConcurrency || 2;
    return duration * 0.13 * (channels === 2 && cores < 3 ? 2 : 1);
  }, [duration, info]);
  // Whatever is still downloading counts too, so the estimate is honest if you start straight away.
  const loadLeft = load.done ? 0 : load.secondsLeft;
  const eta = procEta == null ? null : procEta + (loadLeft || 0);

  return (
    <div className="container narrow page stack-lg">
      <header>
        <h1>Clean up a recording</h1>
        <p className="lead muted">Take the background noise out of a voice recording or video. It happens on your own device, so your file is never uploaded.</p>
        <p className="row small muted mb0"><IconLock width={18} height={18} /> Private by design: nothing is uploaded and nothing is tracked. Your free account just unlocks the noise remover.</p>
      </header>

      {!supported && <div className="banner banner-error" role="alert">This browser can&rsquo;t run the noise remover. Please try a current version of Chrome, Edge, Firefox or Safari.</div>}

      {phase === 'idle' && supported && <FileDrop onFile={choose} />}

      {phase === 'probing' && <p role="status" className="card"><span className="spinner" aria-hidden="true" /> Reading {file?.name}…</p>}

      {file && (phase === 'ready' || phase === 'working' || phase === 'done') && (
        <section className="card stack" aria-labelledby="file-h">
          <div className="file-chip">
            <div className="grow">
              <div className="name" id="file-h">{file.name}</div>
              <div className="meta">
                {bytes(file.size)}
                {duration ? ` · ${clock(duration)}` : ''}
                {info?.channels ? ` · ${info.channels === 1 ? 'mono' : info.channels === 2 ? 'stereo' : `${info.channels} channels`}` : ''}
                {info?.sampleRate ? ` · ${(info.sampleRate / 1000).toFixed(info.sampleRate % 1000 ? 1 : 0)} kHz` : ''}
                {isVideo ? ' · video' : ''}
              </div>
            </div>
            {phase !== 'working' && <button type="button" className="btn btn-ghost btn-sm" onClick={reset}>Choose another</button>}
          </div>

          {info?.hasAudio === false && <div className="banner banner-error" role="alert">This file has no audio track, so there is nothing to clean.</div>}
          {info?.canDecode === false && <div className="banner banner-warn">Your browser may not be able to decode the audio format ({info.audioCodec}) of this file. We will try anyway.</div>}
          {duration > LONG_SECONDS && !tooLong && (
            <div className="banner banner-warn">
              This is a long recording ({human(duration)}). Cleaning it here could take about {human(eta)} and needs a lot of memory.
              Keep this tab open, or use <Link to="/app">Clean &amp; polish</Link> for long files.
            </div>
          )}
          {tooLong && <div className="banner banner-error" role="alert">Recordings longer than {MAX_SECONDS / 3600} hours cannot be cleaned in the browser. Please split the file or use <Link to="/app">Clean &amp; polish</Link>.</div>}
          {info?.channels > 2 && <p className="small muted mb0">This file has {info.channels} channels; they will be mixed down to mono.</p>}
          {error && <div className="banner banner-error" role="alert">{error}</div>}

          {(phase === 'ready' || phase === 'done') && (
            <>
              <fieldset className="field" disabled={phase === 'working'}>
                <legend className="label">Noise reduction strength</legend>
                <div className="choice-cards">
                  {STRENGTHS.map((s) => (
                    <label className="choice" key={s.id}>
                      <input type="radio" name="strength" value={s.id} checked={strength === s.id} onChange={() => setStrength(s.id)} />
                      <span><b>{s.label}</b><small>{s.db} dB limit. {s.hint}</small></span>
                    </label>
                  ))}
                </div>
              </fieldset>
              <div className="row">
                <button type="button" className="btn btn-primary btn-lg" onClick={start} disabled={info?.hasAudio === false || tooLong || !supported}>
                  {phase === 'done' ? 'Clean again with these settings' : 'Clean audio'}
                </button>
                {eta && <span className="muted small" data-testid="eta">Should take about {human(eta)} on this device{loadLeft ? ' (including getting the noise remover ready)' : ''}.</span>}
              </div>
            </>
          )}

          {phase === 'working' && <ProgressPanel prog={prog} stage={stage} load={load} procEta={procEta} onCancel={() => abort.current?.abort()} />}
        </section>
      )}

      {phase === 'done' && result && (
        <Result file={file} info={info} result={result} isVideo={isVideo} />
      )}

      {phase === 'done' && (
        <aside className="card stack card-flat">
          <h2 className="mb0">Got a long recording, or want subtitles?</h2>
          <p className="mb0 muted">
            Clean &amp; polish looks after long files without tying up your computer. It also evens out the loudness, and you can add subtitles and a transcript in the same go{polish ? ` (from ${moneyUsd(polish.usdPerHour)} an hour)` : ''}.
          </p>
          <div className="row"><Link to="/app" className="btn btn-primary">Open Files</Link></div>
        </aside>
      )}
    </div>
  );
}

function ProgressPanel({ prog, stage, load, procEta, onCancel }) {
  const loading = stage === STAGE_LOADING;
  const pct = loading ? Math.min(100, (load.loaded / load.total) * 100) : prog?.total ? Math.min(100, (prog.done / prog.total) * 100) : null;
  const [announce, setAnnounce] = useState('');
  const last = useRef(-1);
  useEffect(() => {
    if (pct == null || loading) return;
    const bucket = Math.floor(pct / 10);
    if (bucket !== last.current) {
      last.current = bucket;
      setAnnounce(`${bucket * 10} percent done`);
    }
  }, [pct, loading]);
  const mb = (n) => (n / 1048576).toFixed(1);
  // While the download is still running the whole wait is "what is left to download" plus the cleaning itself.
  const loadEta = loading && load.secondsLeft != null ? load.secondsLeft + (procEta || 0) : null;
  const eta = loading ? loadEta : prog?.eta;
  return (
    <div className="stack" data-testid="progress">
      <div className="row-between">
        <strong data-testid="stage">{loading ? `${STAGE_LOADING} ${mb(load.loaded)} of ${mb(load.total)} MB` : stage || 'Working…'}</strong>
        <span className="muted small" data-testid="progress-text">
          {loading ? `${Math.floor(pct)}%` : pct != null ? `${Math.floor(pct)}%` : `${clock(prog?.done || 0)} done`}
          {eta != null && (loading || prog.done > 0) ? ` · about ${human(eta)} left` : ''}
          {!loading && prog?.rate > 0 ? ` · ${prog.rate.toFixed(1)}× real time` : ''}
        </span>
      </div>
      {pct != null ? <progress className="bar" max={100} value={pct} aria-label={loading ? 'Getting the noise remover ready' : 'Cleaning progress'} /> : <progress className="bar" aria-label="Cleaning progress" />}
      <p className="sr-only" role="status" aria-live="polite">{announce}</p>
      <div className="row">
        <button type="button" className="btn btn-secondary" onClick={onCancel}>Cancel</button>
        <span className="muted small">Keep this tab open while it works.</span>
      </div>
    </div>
  );
}

function Result({ file, info, result, isVideo }) {
  const { pcm, wav, url } = result;
  const base = stem(file.name);
  const [busy, setBusy] = useState('');
  const [err, setErr] = useState('');
  const [note, setNote] = useState('');
  const [encoder, setEncoder] = useState(undefined); // undefined: unknown yet, null: none
  const [remuxProg, setRemuxProg] = useState(0);

  useEffect(() => {
    if (!isVideo) return undefined;
    let alive = true;
    import('../media/remux.js').then((m) => m.videoAudioEncoder()).then((c) => { if (alive) setEncoder(c); });
    return () => { alive = false; };
  }, [isVideo]);

  async function dl24() {
    setBusy('wav24'); setErr('');
    try {
      saveBlob(await buildWav(pcm, 24), `${base}-cleaned-24bit.wav`);
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy('');
    }
  }

  async function dlVideo() {
    setBusy('video'); setErr(''); setNote(''); setRemuxProg(0);
    try {
      const { remuxWithCleanAudio } = await import('../media/remux.js');
      const out = await remuxWithCleanAudio({ videoFile: file, pcm, onProgress: setRemuxProg });
      saveBlob(out.blob, `${base}-cleaned${out.ext}`);
      if (out.note) setNote(out.note);
    } catch (e) {
      setErr(e.message || 'Could not build the video.');
    } finally {
      setBusy('');
    }
  }

  const silent = pcm.rms < 1e-5;
  return (
    <section className="card stack" aria-labelledby="res-h" data-testid="result" data-rms={pcm.rms.toFixed(6)} data-peak={pcm.peak.toFixed(4)} data-frames={pcm.frames} data-channels={pcm.channels}>
      <h2 id="res-h" className="mb0">All done. Have a listen, then save it</h2>
      {silent && <div className="banner banner-warn">The cleaned audio is nearly silent. The original may only contain noise, or try a gentler strength.</div>}
      <AbPlayer original={file} cleanedUrl={url} isVideo={isVideo} />
      <p className="stat-row mb0">
        <span>Length <b>{clock(pcm.seconds)}</b></span>
        <span>Format <b>48 kHz {pcm.channels === 1 ? 'mono' : 'stereo'}</b></span>
        <span>Took <b>{human(pcm.elapsed)}</b></span>
        {info?.sampleRate && info.sampleRate !== 48000 && <span>Converted from <b>{(info.sampleRate / 1000).toFixed(1)} kHz</b></span>}
      </p>
      <div className="row">
        <a className="btn btn-primary" href={url} download={`${base}-cleaned.wav`}><IconDownload width={18} height={18} /> Cleaned WAV (16-bit)</a>
        <button type="button" className="btn btn-secondary" onClick={dl24} disabled={!!busy}>{busy === 'wav24' ? 'Preparing…' : 'WAV (24-bit)'}</button>
        {isVideo && (
          <button type="button" className="btn btn-secondary" onClick={dlVideo} disabled={!!busy || encoder === null} data-testid="dl-video">
            {busy === 'video' ? `Building video… ${Math.round(remuxProg * 100)}%` : 'Video with cleaned audio'}
          </button>
        )}
      </div>
      {isVideo && encoder === null && <p className="banner banner-info small mb0">This browser cannot encode AAC or Opus audio, so a rebuilt video is not available here. Download the cleaned WAV and combine it with your video, or try a Chromium-based browser.</p>}
      {isVideo && encoder && <p className="small muted mb0">The video picture is copied unchanged; only the audio is replaced ({encoder === 'aac' ? 'AAC' : 'Opus'}).</p>}
      {note && <p className="banner banner-info small mb0">{note}</p>}
      {err && <p className="banner banner-error small mb0" role="alert">{err}</p>}
    </section>
  );
}
