import { useEffect } from 'react';
import { Link } from 'react-router';
import { etaSeconds, etaText } from '../eta.js';
import { finishOf } from '../finish.js';
import { clock, dateTime, human } from '../format.js';
import { useNow } from '../hooks.js';
import { ACTIVE, jobPath, jobTitle, kindLabel, stageText, statusLabel, withNotCharged } from '../jobs.js';

function badgeClass(status) {
  if (status === 'done') return 'badge badge-ok';
  if (status === 'failed' || status === 'canceled') return 'badge badge-bad';
  if (ACTIVE.has(status)) return 'badge badge-run';
  return 'badge';
}

/**
 * One job of "Your jobs": its name, what it is doing and, while it is active, a progress bar with the time left, so several
 * can be watched at once. "Open" goes to the job's own page (its progress or result, Download again, Change and save again,
 * Delete now). A job that finishes while this list is on screen is finished here too (its files download by themselves).
 */
export default function JobRow({ job }) {
  const fin = finishOf(job);
  const now = useNow(5000);
  const active = ACTIVE.has(job.status);
  useEffect(() => {
    if (job.status === 'done') fin.auto();
  }, [job.status, fin]);

  const pct = typeof job.progress === 'number' ? job.progress : 0;
  const eta = etaText(etaSeconds(job, Date.now()));
  const step = stageText(job);
  const left = job.expiresAt ? job.expiresAt - now / 1000 : null;
  let line = '';
  if (job.status === 'uploading') line = 'Waiting for the upload to finish';
  else if (job.status === 'queued') line = ['In line', eta].filter(Boolean).join(' · ');
  else if (active) line = [step || 'Working on it', pct ? `${Math.round(pct)}%` : '', eta].filter(Boolean).join(' · ');
  return (
    <article className="card job job-row" data-testid="job-row" data-status={job.status} data-job-id={job.id}>
      <div className="job-head">
        <div>
          <div className="job-title"><Link to={jobPath(job.id)} data-testid="job-open-title">{jobTitle(job, fin.file ? { file: fin.file } : null)}</Link></div>
          <div className="small muted">{kindLabel(job)} · {dateTime(job.createdAt)} · {clock(job.durationSeconds)}</div>
        </div>
        <span className="row">
          <span className={badgeClass(job.status)}>{statusLabel(job.status)}</span>
          <Link to={jobPath(job.id)} className="btn btn-secondary btn-sm" data-testid="job-open">Open</Link>
        </span>
      </div>
      {active && (
        <div>
          <progress className="bar" max={100} value={job.status === 'queued' || job.status === 'uploading' ? 0 : pct} aria-label={`Progress of ${jobTitle(job)}`} data-testid="row-progress" />
          <p className="small muted mb0" role="status" data-testid="row-line">{line}</p>
        </div>
      )}
      {job.status === 'done' && job.local && <p className="small muted mb0" data-testid="row-local">Kept in this tab only, until you leave or reload the page.</p>}
      {job.status === 'done' && !job.local && left != null && (
        <p className="small muted mb0" data-testid="row-expiry">
          {left <= 0 ? 'Your files are being cleared away.' : `Files are cleared away in ${human(Math.max(60, Math.round(left / 60) * 60))}`}
        </p>
      )}
      {job.status === 'failed' && <p className="small muted mb0">{withNotCharged(job.error || 'This one didn’t work out.')}</p>}
      {job.status === 'canceled' && <p className="small muted mb0">{withNotCharged('Canceled.')}</p>}
    </article>
  );
}
