import { useEffect, useRef, useState } from 'react';
import { saveBlob } from '../download.js';
import { bytes, human, stem } from '../format.js';
import { buildMedia } from '../media/convert.js';
import { buildCues, toSrt, toText, toVtt } from '../media/speech.js';
import { IconDownload } from './Icons.jsx';

/** The subtitle files of a finished in-browser transcription: { srt, vtt, txt, cues, segments, language, empty }. */
export function subtitleFiles(result) {
  const cues = buildCues(result.segments);
  return { srt: toSrt(cues), vtt: toVtt(cues), txt: toText(result.segments), cues, segments: result.segments, language: result.language, empty: cues.length === 0 };
}

/** Progress of the free in-browser subtitles: the one-time model download first, then the writing. */
export function SubtitlesProgress({ prog, load }) {
  const loading = !load.done;
  const pct = loading
    ? (load.total ? Math.round((load.loaded / load.total) * 100) : null)
    : prog?.total ? Math.min(100, Math.round((prog.done / prog.total) * 100)) : null;
  return (
    <div className="stack" data-testid="subs-progress">
      <div className="row-between">
        <strong>{loading ? 'Getting the subtitle maker ready…' : 'Writing the subtitles…'}</strong>
        <span className="muted small">
          {loading ? `${bytes(load.loaded)} of ${bytes(load.total)} (only the first time)` : `${pct != null ? `${pct}%` : ''}${prog?.eta ? ` · about ${human(prog.eta)} left` : ''}`}
        </span>
      </div>
      {pct != null ? <progress className="bar" max={100} value={pct} aria-label="Subtitle progress" /> : <progress className="bar" aria-label="Subtitle progress" />}
    </div>
  );
}

const FILES = [
  ['srt', 'SRT', (b) => `${b}.srt`, 'application/x-subrip'],
  ['vtt', 'VTT', (b) => `${b}.vtt`, 'text/vtt'],
  ['txt', 'TXT', (b) => `${b}-transcript.txt`, 'text/plain'],
];

/**
 * The subtitles made in this browser: their files to download, a peek at the text, and (with `embed`) the video with
 * them inside, made here as an MKV, with the sound cleaned here when there is some (`cleanPcm`), else the video's own.
 */
export default function SubtitlesResult({ file, info, files, caps, embed, cleanPcm, onClose }) {
  const base = stem(file.name);
  const [busy, setBusy] = useState(false);
  const [prog, setProg] = useState(0);
  const [err, setErr] = useState('');
  const [note, setNote] = useState('');
  const ctl = useRef(null);
  useEffect(() => () => ctl.current?.abort(), []);
  const wanted = FILES.filter(([id]) => caps?.[id] !== false);
  const preview = files.txt.split(/\n+/).filter(Boolean).slice(0, 3).join(' ');

  async function saveVideo() {
    const abort = new AbortController();
    ctl.current = abort;
    setBusy(true); setErr(''); setNote(''); setProg(0);
    try {
      const out = await buildMedia({
        format: 'mkv', videoFile: file, info, pcm: cleanPcm || null, audioFile: cleanPcm ? null : file, audioName: file.name,
        subtitles: files.vtt, onProgress: setProg, signal: abort.signal,
      });
      saveBlob(out.blob, `${base}-${cleanPcm ? 'cleaned' : 'subtitled'}.mkv`);
      setNote(out.note || 'Saved.');
    } catch (e) {
      if (e?.name !== 'AbortError') setErr(e.message || 'Couldn’t make the video. Please try again.');
    } finally {
      ctl.current = null;
      setBusy(false);
    }
  }

  return (
    <section className="card stack" aria-labelledby="subs-res-h" data-testid="subs-result" data-cues={files.cues.length} data-language={files.language}>
      <div className="row-between">
        <h2 id="subs-res-h" className="mb0">Your subtitles are ready</h2>
        {onClose && <button type="button" className="btn btn-ghost btn-sm" onClick={onClose}>Close</button>}
      </div>
      {files.empty ? (
        <p className="banner banner-warn mb0" data-testid="subs-empty">We didn&rsquo;t hear any speech in this recording, so there are no subtitles.</p>
      ) : (
        <>
          <p className="muted mb0" data-testid="subs-preview">&ldquo;{preview.length > 220 ? `${preview.slice(0, 220)}…` : preview}&rdquo;</p>
          <div className="row">
            {wanted.map(([id, label, name, type]) => (
              <button key={id} type="button" className="btn btn-secondary" data-testid={`subs-dl-${id}`} onClick={() => saveBlob(new Blob([files[id]], { type }), name(base))}>
                <IconDownload width={18} height={18} /> {label}
              </button>
            ))}
          </div>
          {embed && info?.hasVideo && (
            <div className="row">
              <button type="button" className="btn btn-primary" onClick={saveVideo} disabled={busy} data-testid="subs-video">
                <IconDownload width={18} height={18} /> {busy ? `Making the video… ${Math.round(prog * 100)}%` : `Video with the subtitles${cleanPcm ? ' and the cleaned sound' : ''} (MKV)`}
              </button>
              {busy && <button type="button" className="btn btn-secondary btn-sm" onClick={() => ctl.current?.abort()}>Cancel</button>}
              <span className="small muted">Made on your device; nothing is uploaded.</span>
            </div>
          )}
        </>
      )}
      {note && <p className="small muted mb0" role="status">{note}</p>}
      {err && <p className="banner banner-error small mb0" role="alert">{err}</p>}
    </section>
  );
}
