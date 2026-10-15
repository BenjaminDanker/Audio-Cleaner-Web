import { useRef, useState } from 'react';
import { api, outputUrl } from '../api.js';
import { saveBlob } from '../download.js';
import { bytes, clock, dateTime, human, money, shownName, stem } from '../format.js';
import { useNow } from '../hooks.js';
import { ACTIVE, jobTitle, kindLabel, queueText, stageText, statusLabel } from '../jobs.js';
import { IconDownload } from './Icons.jsx';

const OUT_LABEL = { audio: 'Cleaned audio', video: 'Video', captions: 'Captions', transcript: 'Transcript', segments: 'Segments' };

function badgeClass(status) {
  if (status === 'done') return 'badge badge-ok';
  if (status === 'failed' || status === 'canceled') return 'badge badge-bad';
  if (ACTIVE.has(status)) return 'badge badge-run';
  return 'badge';
}

function Countdown({ expiresAt }) {
  const now = useNow(30_000);
  if (!expiresAt) return null;
  const left = expiresAt - now / 1000;
  if (left <= 0) return <span className="muted small">Your files are being cleared away.</span>;
  return <span className="muted small" data-testid="expiry">Files are cleared away in {human(Math.max(60, Math.round(left / 60) * 60))}</span>;
}

/** Add the cleaned audio back into the original video, locally. */
function RemuxPanel({ job, local }) {
  const [file, setFile] = useState(local?.file || null);
  const [busy, setBusy] = useState(false);
  const [prog, setProg] = useState(0);
  const [err, setErr] = useState('');
  const [note, setNote] = useState('');
  const picker = useRef(null);
  const audioOut = job.outputs.find((o) => o.kind === 'audio');
  if (!audioOut) return null;

  async function run(videoFile) {
    setBusy(true); setErr(''); setNote(''); setProg(0);
    try {
      const res = await fetch(outputUrl(job.id, audioOut.name), { credentials: 'same-origin' });
      if (!res.ok) throw new Error('Couldn’t download the cleaned audio. Please try again.');
      const audioFile = await res.blob();
      const { remuxWithCleanAudio } = await import('../media/remux.js');
      const out = await remuxWithCleanAudio({ videoFile, audioFile, onProgress: setProg });
      saveBlob(out.blob, `${stem(videoFile.name)}-cleaned${out.ext}`);
      if (out.note) setNote(out.note);
    } catch (e) {
      setErr(e.message || 'Couldn’t build the video. Please try again.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack" data-testid="remux">
      <div className="row">
        {file ? (
          <button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={() => run(file)} data-testid="remux-btn">
            {busy ? `Building video… ${Math.round(prog * 100)}%` : `Put cleaned audio into ${shownName(file.name)}`}
          </button>
        ) : (
          <>
            <button type="button" className="btn btn-secondary btn-sm" onClick={() => picker.current?.click()}>Put it back into your video…</button>
            <input
              ref={picker} type="file" accept="video/*,.mkv,.mov,.mp4,.webm" className="sr-only" tabIndex={-1}
              aria-label="Choose the original video"
              onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ''; if (f) { setFile(f); run(f); } }}
            />
          </>
        )}
        <span className="small muted">This happens on your device; the video isn’t uploaded.</span>
      </div>
      {note && <p className="small banner banner-info mb0">{note}</p>}
      {err && <p className="small banner banner-error mb0" role="alert">{err}</p>}
    </div>
  );
}

/** Failed and canceled jobs cost nothing: say so once (the server's own messages often say it already). */
const withNotCharged = (text) => (/not charged|weren.t charged/i.test(text) ? text : `${text} You weren’t charged.`);

export default function JobCard({ job, local, onChanged }) {
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const active = ACTIVE.has(job.status);
  const canCancel = job.status === 'uploading' || job.status === 'queued';
  const hasVideoInput = local?.info?.hasVideo || /\.(mp4|mov|m4v|mkv|webm)$/i.test(job.inputName || '');
  const showRemux = job.status === 'done' && (hasVideoInput && local?.extracted !== false && local?.keepVideo !== false) && !!job.outputs.find((o) => o.kind === 'audio') && !job.outputs.find((o) => o.kind === 'video');
  const hasVideoOut = job.outputs.some((o) => o.kind === 'video');

  async function act(fn) {
    setBusy(true); setErr('');
    try {
      await fn();
      onChanged?.();
    } catch (e) {
      setErr(e.detail || e.message);
    } finally {
      setBusy(false);
    }
  }

  const pct = typeof job.progress === 'number' ? job.progress : 0;
  const step = stageText(job); // '' when the server's step name is not one we know: then only the percentage is shown
  const stepLine = step ? `${step}${pct ? ` · ${Math.round(pct)}%` : ''}` : pct ? `${Math.round(pct)}% done` : 'Just a moment…';
  return (
    <article className="card job" data-testid="job" data-status={job.status} data-job-id={job.id}>
      <div className="job-head">
        <div>
          <div className="job-title">{jobTitle(job, local)}</div>
          <div className="small muted">{kindLabel(job)} · {dateTime(job.createdAt)} · {clock(job.durationSeconds)}</div>
        </div>
        <span className="row">
          {job.free && <span className="badge badge-free" data-testid="free-badge">Free</span>}
          <span className={badgeClass(job.status)}>{statusLabel(job.status)}</span>
        </span>
      </div>

      {job.status === 'queued' && (
        <p className="queue-note mb0" role="status" data-testid="queue-note">
          <span className="spinner" aria-hidden="true" /> {queueText(job) || 'In line, you’re up soon'}
        </p>
      )}
      {(job.status === 'running' || job.status === 'cloud_asr') && (
        <div>
          <progress className="bar" max={100} value={pct} aria-label={`Job progress, ${step || statusLabel(job.status)}`} />
          <p className="small muted mb0" role="status">{stepLine}</p>
        </div>
      )}

      {job.status === 'failed' && <p className="banner banner-error mb0" role="alert">{withNotCharged(job.error || 'This one didn’t work out.')}</p>}
      {job.status === 'canceled' && <p className="muted mb0">{withNotCharged(job.error && job.error !== 'Canceled.' ? job.error : 'Canceled.')}</p>}

      {job.status === 'done' && (
        <>
          <ul className="outputs" aria-label="Downloads">
            {job.outputs.map((o) => (
              <li key={o.name}>
                <span><span className="oname">{o.name}</span> <span className="muted small">· {OUT_LABEL[o.kind] || o.kind} · {bytes(o.bytes)}</span></span>
                <a className="btn btn-primary btn-sm" href={outputUrl(job.id, o.name)} download={o.name} data-testid="download">
                  <IconDownload width={16} height={16} /> Download
                </a>
              </li>
            ))}
          </ul>
          {job.notice && <p className="banner banner-info small mb0" role="status" data-testid="job-notice">{job.notice}</p>}
          {showRemux && <RemuxPanel job={job} local={local} />}
          {hasVideoOut && <p className="small muted mb0">The video file above already has the clean sound in it.</p>}
          <div className="row-between">
            <span className="small muted">Cost <b>{money(job.costMicros || 0)}</b> · <Countdown expiresAt={job.expiresAt} /></span>
            {confirmDelete ? (
              <span className="row" data-testid="delete-confirm">
                <span className="small">Delete these files now? They can’t be brought back.</span>
                <button type="button" className="btn btn-danger btn-sm" disabled={busy} onClick={() => act(() => api(`jobs/${job.id}`, { method: 'DELETE' }))}>Yes, delete</button>
                <button type="button" className="btn btn-ghost btn-sm" disabled={busy} onClick={() => setConfirmDelete(false)}>Keep them</button>
              </span>
            ) : (
              <button type="button" className="btn btn-danger-outline btn-sm" disabled={busy} onClick={() => setConfirmDelete(true)}>Delete now</button>
            )}
          </div>
        </>
      )}

      {canCancel && (
        <div><button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={() => act(() => api(`jobs/${job.id}/cancel`, { method: 'POST' }))}>Cancel job</button></div>
      )}
      {err && <p className="banner banner-error small mb0" role="alert">{err}</p>}
    </article>
  );
}

