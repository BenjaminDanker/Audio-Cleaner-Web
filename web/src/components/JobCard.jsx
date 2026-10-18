import { useEffect, useState } from 'react';
import { Link } from 'react-router';
import { api } from '../api.js';
import { useAuth } from '../auth.jsx';
import { finishOf, forgetFinish, removeLocalCard, useFinish } from '../finish.js';
import { clock, dateTime, human, money } from '../format.js';
import { useNow } from '../hooks.js';
import { ACTIVE, jobTitle, kindLabel, queueText, stageText, statusLabel } from '../jobs.js';
import ResultPanel from './Result.jsx';

function badgeClass(status, finishing) {
  if (finishing) return 'badge badge-run';
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

/** Failed and canceled jobs cost nothing: say so once (the server's own messages often say it already). */
const withNotCharged = (text) => (/not charged|weren.t charged/i.test(text) ? text : `${text} You weren’t charged.`);

/**
 * One job in the list: what it is doing while it runs, and, once it is done, its files (see Result.jsx). A free run made
 * in this tab (job.local) has no job on our side; it shows the same way.
 */
export default function JobCard({ job, onChanged }) {
  const { contact } = useAuth();
  const fin = finishOf(job);
  const st = useFinish(fin);
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const canCancel = job.status === 'uploading' || job.status === 'queued';
  const finishing = job.status === 'done' && (st.phase === 'waiting' || st.phase === 'building' || st.phase === 'saving');

  // The job finished while this tab was open: its last step and the downloads happen by themselves.
  useEffect(() => {
    if (job.status === 'done') fin.auto();
  }, [job.status, fin]);

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

  // The last step is part of the wait: the bar fills to nine tenths while we work, and the last tenth is putting the files together.
  const tail = fin.thisTab && fin.willBuild() ? 0.1 : 0;
  const pct = (typeof job.progress === 'number' ? job.progress : 0) * (1 - tail);
  const step = stageText(job); // '' when the server's step name is not one we know: then only the percentage is shown
  const stepLine = step ? `${step}${pct ? ` · ${Math.round(pct)}%` : ''}` : pct ? `${Math.round(pct)}% done` : 'Just a moment…';
  return (
    <article className="card job" data-testid="job" data-status={job.status} data-job-id={job.id}>
      <div className="job-head">
        <div>
          <div className="job-title">{jobTitle(job, fin.file ? { file: fin.file } : null)}</div>
          <div className="small muted">{kindLabel(job)} · {dateTime(job.createdAt)} · {clock(job.durationSeconds)}</div>
        </div>
        <span className="row">
          {job.free && <span className="badge badge-free" data-testid="free-badge">{job.kind === 'transcribe' || job.local ? 'Free' : 'Free subtitles'}</span>}
          <span className={badgeClass(job.status, finishing)}>{finishing ? 'Putting it together' : statusLabel(job.status)}</span>
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

      {job.status === 'failed' && (
        <>
          <p className="banner banner-error mb0" role="alert">{withNotCharged(job.error || 'This one didn’t work out.')}</p>
          {contact && <p className="small muted mb0">Still stuck? <Link to={`/contact?topic=job&job=${job.id}`} data-testid="job-contact">Tell us about this job</Link>.</p>}
        </>
      )}
      {job.status === 'canceled' && <p className="muted mb0">{withNotCharged(job.error && job.error !== 'Canceled.' ? job.error : 'Canceled.')}</p>}

      {job.status === 'done' && (
        <>
          <ResultPanel job={job} fin={fin} />
          <div className="row-between">
            {job.local ? (
              <span className="small muted">Kept in this tab only, until you leave or reload the page.</span>
            ) : (
              <span className="small muted">Cost <b>{money(job.costMicros || 0)}</b> · <Countdown expiresAt={job.expiresAt} /></span>
            )}
            {job.local ? (
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => removeLocalCard(job.id)}>Remove</button>
            ) : confirmDelete ? (
              <span className="row" data-testid="delete-confirm">
                <span className="small">Delete these files now? They can’t be brought back.</span>
                <button type="button" className="btn btn-danger btn-sm" disabled={busy} onClick={() => act(async () => { await api(`jobs/${job.id}`, { method: 'DELETE' }); forgetFinish(job.id); })}>Yes, delete</button>
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
