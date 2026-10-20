import { useRef } from 'react';
import { bytes, shownName } from '../format.js';
import { useFinish } from '../finish.js';
import { FinishProgress } from './Progress.jsx';
import { IconDownload } from './Icons.jsx';

const BUSY = new Set(['waiting', 'building', 'saving']);

/**
 * What a finished job the editor can't open shows (a job made through the API with no audio steps): the files it made, one "Download
 * again" for all of them, and what it has to say. The editor (components/studio) is the page of every other job.
 */
export default function ResultPanel({ job, fin }) {
  const st = useFinish(fin);
  const picker = useRef(null);
  const w = fin.effective();
  const files = st.files.length ? st.files : fin.planned(w);
  const busy = BUSY.has(st.phase);
  const needVideo = st.phase === 'needvideo' || (st.phase === 'idle' && fin.needsVideo(w));
  const silent = fin.pcm && fin.pcm.rms < 1e-5;
  const preview = fin.subs && !fin.subs.empty ? fin.subs.txt.split(/\n+/).filter(Boolean).slice(0, 3).join(' ') : '';
  const pcm = fin.pcm;

  const onDownload = () => {
    if (needVideo) picker.current?.click();
    else fin.again();
  };

  return (
    <div
      className="result stack" data-testid={st.phase === 'ready' ? 'result' : 'result-pending'}
      {...(pcm ? { 'data-rms': pcm.rms.toFixed(6), 'data-peak': pcm.peak.toFixed(4), 'data-frames': pcm.frames, 'data-channels': pcm.channels } : {})}
      {...(fin.subs ? { 'data-cues': fin.subs.cues.length, 'data-language': fin.subs.language } : {})}
    >
      {(st.phase === 'building' || st.phase === 'waiting') && (
        st.phase === 'waiting'
          ? <p className="mb0" role="status" data-testid="waiting"><span className="spinner" aria-hidden="true" /> Waiting for your browser to finish…</p>
          : <FinishProgress progress={st.progress} onCancel={() => fin.cancel()} />
      )}
      {silent && <div className="banner banner-warn">The cleaned audio is nearly silent. The original may only contain noise, or try a gentler setting.</div>}
      {preview && <p className="muted mb0" data-testid="subs-preview">&ldquo;{preview.length > 220 ? `${preview.slice(0, 220)}…` : preview}&rdquo;</p>}

      {files.length > 0 && (
        <ul className="outputs" aria-label="Your files" data-testid="made">
          {files.map((f) => (
            <li key={f.name} data-testid="made-file">
              <span className="oname">{shownName(f.name)}</span>
              <span className="muted small">{f.label}{f.bytes != null ? ` · ${bytes(f.bytes)}` : ''}</span>
            </li>
          ))}
        </ul>
      )}
      {st.note && <p className="small banner banner-info mb0" role="status" data-testid="made-note">{st.note}</p>}
      {job.notice && <p className="banner banner-info small mb0" role="status" data-testid="job-notice">{job.notice}</p>}
      {st.error && <p className="small banner banner-error mb0" role="alert" data-testid="made-error">{st.error}</p>}
      {needVideo && !st.error && <p className="small muted mb0" data-testid="need-video">To put the picture back, choose your original video once. It stays on your device.</p>}

      {files.length > 0 && (
        <div className="row">
          <button type="button" className="btn btn-primary" onClick={onDownload} disabled={busy} data-testid="download-again">
            <IconDownload width={18} height={18} /> {st.phase === 'saving' ? 'Downloading…' : st.phase === 'error' ? 'Try again' : needVideo ? 'Choose the original video' : 'Download again'}
          </button>
          <input
            ref={picker} type="file" accept="video/*,.mkv,.mov,.mp4,.webm" className="sr-only" tabIndex={-1}
            aria-label="Choose the original video"
            onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ''; if (f) fin.setVideo(f); }}
          />
          {files.length > 1 && <span className="small muted">Downloads all {files.length} files. If your browser asks, allow it.</span>}
        </div>
      )}
    </div>
  );
}
