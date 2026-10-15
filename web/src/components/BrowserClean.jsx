import { useEffect, useRef, useState } from 'react';
import AbPlayer from './AbPlayer.jsx';
import { IconDownload } from './Icons.jsx';
import { saveBlob } from '../download.js';
import { clock, human, stem } from '../format.js';
import { STAGE_LOADING } from '../media/pipeline.js';
import { buildWav, wavFits } from '../media/wav.js';

// The two pieces of the in-browser noise removal that the Cleanup page shows: progress while it runs, and the result
// (listen, then download). Nothing here uploads anything.

export function ProgressPanel({ prog, stage, load, procEta }) {
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
    </div>
  );
}

export function BrowserResult({ file, info, result, isVideo, keepVideo = true, onClose }) {
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
  const fits24 = wavFits(pcm.seconds, pcm.channels, 24); // a WAV file cannot hold more than about 4 GB
  return (
    <section className="card stack" aria-labelledby="res-h" data-testid="result" data-rms={pcm.rms.toFixed(6)} data-peak={pcm.peak.toFixed(4)} data-frames={pcm.frames} data-channels={pcm.channels}>
      <div className="row-between">
        <h2 id="res-h" className="mb0">Noise removed. Have a listen, then save it</h2>
        {onClose && <button type="button" className="btn btn-ghost btn-sm" onClick={onClose}>Close</button>}
      </div>
      {silent && <div className="banner banner-warn">The cleaned audio is nearly silent. The original may only contain noise, or try a gentler setting.</div>}
      <AbPlayer original={file} cleanedUrl={url} isVideo={isVideo} />
      <p className="stat-row mb0">
        <span>Length <b>{clock(pcm.seconds)}</b></span>
        <span>Format <b>48 kHz {pcm.channels === 1 ? 'mono' : 'stereo'}</b></span>
        <span>Took <b>{human(pcm.elapsed)}</b></span>
        {info?.sampleRate && info.sampleRate !== 48000 && <span>Converted from <b>{(info.sampleRate / 1000).toFixed(1)} kHz</b></span>}
      </p>
      <div className="row">
        <a className="btn btn-primary" href={url} download={`${base}-cleaned.wav`}><IconDownload width={18} height={18} /> Cleaned WAV (16-bit)</a>
        <button type="button" className="btn btn-secondary" onClick={dl24} disabled={!!busy || !fits24} title={fits24 ? undefined : 'Too long for a 24-bit WAV file (the format stops at 4 GB). The 16-bit WAV is available.'}>{busy === 'wav24' ? 'Preparing…' : 'WAV (24-bit)'}</button>
        {isVideo && keepVideo && (
          <button type="button" className="btn btn-secondary" onClick={dlVideo} disabled={!!busy || encoder === null} data-testid="dl-video">
            {busy === 'video' ? `Building video… ${Math.round(remuxProg * 100)}%` : 'Video with cleaned audio'}
          </button>
        )}
      </div>
      {!fits24 && <p className="small muted mb0" data-testid="wav24-note">This recording is too long for a 24-bit WAV file (a WAV file can’t be bigger than 4 GB), so only the 16-bit WAV is offered.</p>}
      {isVideo && keepVideo && encoder === null && <p className="banner banner-info small mb0">This browser cannot encode AAC or Opus audio, so a rebuilt video is not available here. Download the cleaned WAV and combine it with your video, or try a Chromium-based browser.</p>}
      {isVideo && keepVideo && encoder && <p className="small muted mb0">The video picture is copied unchanged; only the audio is replaced ({encoder === 'aac' ? 'AAC' : 'Opus'}).</p>}
      {note && <p className="banner banner-info small mb0">{note}</p>}
      {err && <p className="banner banner-error small mb0" role="alert">{err}</p>}
    </section>
  );
}
