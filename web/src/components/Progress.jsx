import { useEffect, useRef, useState } from 'react';
import { bytes, clock, human } from '../format.js';
import { STAGE_LOADING } from '../media/pipeline.js';

// What the form shows while the free parts run in this browser: the noise removal, the subtitles, and the last step
// that puts the files together. Nothing here uploads anything.

/** The noise removal. `tail` (0 to 1): the share of the bar kept for putting the files together afterwards. */
export function ProgressPanel({ prog, stage, load, procEta, tail = 0, finishEta = 0 }) {
  const loading = stage === STAGE_LOADING;
  const raw = loading ? Math.min(100, (load.loaded / load.total) * 100) : prog?.total ? Math.min(100, (prog.done / prog.total) * 100) : null;
  const pct = raw == null || loading ? raw : raw * (1 - tail);
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
  const loadEta = loading && load.secondsLeft != null ? load.secondsLeft + (procEta || 0) + finishEta : null;
  const eta = loading ? loadEta : prog?.eta != null ? prog.eta + finishEta : null;
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

/** The free in-browser subtitles: the one-time model download first, then the writing. */
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

/** The last step: the files are put together on this device. `from` (0 to 1): where the bar started, when the earlier work used the first part of it. */
export function FinishProgress({ progress, from = 0, onCancel, testId = 'finish-progress' }) {
  const pct = Math.round((from + (1 - from) * progress) * 100);
  return (
    <div className="stack" data-testid={testId}>
      <div className="row-between">
        <strong>Putting it together…</strong>
        <span className="muted small">{pct}%</span>
      </div>
      <progress className="bar" max={100} value={pct} aria-label="Putting your files together" />
      {onCancel && <div><button type="button" className="btn btn-ghost btn-sm" onClick={onCancel}>Cancel</button></div>}
    </div>
  );
}
