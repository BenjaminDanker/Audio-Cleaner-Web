// How long is left on a file job, in plain words: worked out from how fast its progress moves (the server's progress events,
// one reading per poll of the job list), and, before progress has moved enough to say, from the server's own rough length
// of the job (`estimateSeconds`). The readings of every job are kept here for as long as the tab is, so the job list and the job
// view show the same figure and moving between them doesn't start the estimate over.

const readings = new Map(); // job id -> {first: {t, p}, moved: {t, p} | null, last: {t, p}}

const MIN_PROGRESS = 2; // percent the bar must have moved before its speed is trusted
const MIN_SECONDS = 5;
const TRUST_PROGRESS = 15; // percent of movement after which the measured speed counts fully

/** Take a reading of every job in the list (call it once per fetch of the list, with the time in ms). */
export function noteJobs(jobs, now = Date.now()) {
  const seen = new Set();
  for (const j of jobs || []) {
    if (j.status !== 'running' && j.status !== 'cloud_asr') continue;
    seen.add(j.id);
    const p = Number.isFinite(j.progress) ? j.progress : 0;
    const r = readings.get(j.id);
    if (!r) {
      readings.set(j.id, { first: { t: now, p }, moved: null, last: { t: now, p } });
      continue;
    }
    if (p < r.last.p) { // it started over (a restart put it back in the queue)
      readings.set(j.id, { first: { t: now, p }, moved: null, last: { t: now, p } });
      continue;
    }
    if (!r.moved && p > r.first.p) r.moved = { t: now, p };
    r.last = { t: now, p };
  }
  for (const id of [...readings.keys()]) if (!seen.has(id)) readings.delete(id);
}

export function forgetReadings() { readings.clear(); }

/** Seconds left on a job, or null when nothing can be said (still uploading, or finished). */
export function etaSeconds(job, now = Date.now()) {
  const est = Number.isFinite(job.estimateSeconds) && job.estimateSeconds > 0 ? job.estimateSeconds : null;
  if (job.status === 'queued') {
    const wait = Number.isFinite(job.queueWaitSeconds) ? job.queueWaitSeconds : 0;
    return est == null && !wait ? null : wait + (est || 0);
  }
  if (job.status !== 'running' && job.status !== 'cloud_asr') return null;
  const p = Number.isFinite(job.progress) ? Math.max(0, Math.min(100, job.progress)) : 0;
  const byEstimate = est == null ? null : est * (1 - p / 100);
  const r = readings.get(job.id);
  if (r?.moved) {
    const dp = r.last.p - r.moved.p;
    const dt = (Math.max(now, r.last.t) - r.moved.t) / 1000;
    // The speed since the first movement: the bar jumps over the steps that take no time, so it is measured from where it began to move.
    if (dp >= MIN_PROGRESS && dt >= MIN_SECONDS) {
      const byRate = (100 - p) / (dp / dt);
      if (byEstimate == null) return byRate;
      const w = Math.min(1, dp / TRUST_PROGRESS);
      return w * byRate + (1 - w) * byEstimate;
    }
  }
  return byEstimate;
}

/** "about 5 min left", "less than a minute left", "about 1 h 20 min left". */
export function etaText(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return '';
  const m = Math.round(seconds / 60);
  if (m < 1) return 'less than a minute left';
  if (m < 60) return `about ${m} min left`;
  const h = Math.floor(m / 60);
  return `about ${h} h ${String(m % 60).padStart(2, '0')} min left`;
}
