import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api.js';
import { useAuth } from './auth.jsx';
import { noteJobs } from './eta.js';
import { ACTIVE } from './jobs.js';

const POLL_MS = 1500; // while a job is uploading or running (its progress moves)
const SLOW_POLL_MS = 5000; // while jobs only wait in the queue or are with our speech partner: nothing but the queue position changes

const HIDDEN_POLL_MS = 15000; // while the tab is in the background: nobody watches the progress, the page refreshes when it is shown again

const MOVING = new Set(['uploading', 'running', 'canceling']);

/** Milliseconds until the list is fetched again: quick while some job's progress moves, slow while jobs only wait. */
export function pollDelay(list, hidden = false) {
  if (hidden) return HIDDEN_POLL_MS;
  return (list || []).some((j) => MOVING.has(j.status)) ? POLL_MS : SLOW_POLL_MS;
}

const tabHidden = () => typeof document !== 'undefined' && document.hidden;

/**
 * Job list for the signed-in user. While any job is active the list itself is fetched again (one request,
 * whatever the number of jobs; GET /api/jobs carries each job's status, stage and progress): every 1.5 s while one is
 * uploading or running, every 5 s while they only wait (queued, or in the cloud stage), every 15 s while the tab is in the
 * background, and once at once when the tab is shown again. If the server asks us to wait
 * (HTTP 429, Retry-After) or cannot be reached, polling slows down instead of hammering it, and a short note says so. A
 * sign-in that ran out (401) takes the visitor to the sign-in page like every other call. `limits`: what the server says about
 * jobs ({maxActiveJobs, retentionSeconds}); every fetch is also a reading for the time-left estimate (eta.js).
 * Signed out there is no list to ask for (the free tools make no job on our side): nothing is fetched and the list is empty.
 */
export function useJobs({ onFinished } = {}) {
  const { ready, signedIn } = useAuth();
  const [jobs, setJobs] = useState(null);
  const [limits, setLimits] = useState(null);
  const [error, setError] = useState('');
  const jobsRef = useRef(null);
  jobsRef.current = jobs;
  const finished = useRef(onFinished);
  finished.current = onFinished;

  const load = useCallback(async () => {
    if (!signedIn) return;
    try {
      const r = await api('jobs');
      noteJobs(r.jobs);
      setJobs(r.jobs);
      if (r.limits) setLimits(r.limits);
      setError('');
    } catch (e) {
      setError(e.detail || 'Could not load your jobs.');
      setJobs((j) => j || []);
    }
  }, [signedIn]);

  useEffect(() => { load(); }, [load]);

  const hasActive = signedIn && !!jobs?.some((j) => ACTIVE.has(j.status));
  useEffect(() => {
    if (!hasActive) return undefined;
    let stop = false;
    let timer;
    let failures = 0;
    let running = false;
    const tick = async () => {
      if (running || stop) return;
      running = true;
      clearTimeout(timer);
      let wait = tabHidden() ? HIDDEN_POLL_MS : POLL_MS;
      try {
        const r = await api('jobs');
        if (stop) return;
        const before = jobsRef.current || [];
        const now = new Map(r.jobs.map((j) => [j.id, j]));
        const ended = before.some((j) => ACTIVE.has(j.status) && now.has(j.id) && !ACTIVE.has(now.get(j.id).status));
        failures = 0;
        wait = pollDelay(r.jobs, tabHidden());
        noteJobs(r.jobs);
        setJobs(r.jobs);
        if (r.limits) setLimits(r.limits);
        setError('');
        if (ended) finished.current?.();
      } catch (e) {
        if (stop || e?.name === 'AbortError') { running = false; return; }
        if (e.status === 401) {
          setError('Your sign-in ended. Please sign in again to see your jobs.');
          running = false;
          return; // the api layer is already taking the visitor to the sign-in page
        }
        failures += 1;
        if (e.status === 429) {
          wait = Math.min(120_000, Math.max(5000, (e.retryAfter || 5) * 1000));
          setError(e.detail || 'Lots of requests just now. Updates will be a little slower for a moment.');
        } else {
          wait = Math.min(15_000, POLL_MS * 2 ** Math.min(failures, 4));
          if (failures >= 3) setError('We lost touch with the server for a moment. Still trying…');
          if (tabHidden()) wait = Math.max(wait, HIDDEN_POLL_MS);
        }
      }
      running = false;
      if (!stop) timer = setTimeout(tick, wait);
    };
    // Back in view: look right away instead of waiting out the slow background wait.
    const onVisible = () => { if (!document.hidden && failures === 0) tick(); };
    document.addEventListener('visibilitychange', onVisible);
    timer = setTimeout(tick, tabHidden() ? HIDDEN_POLL_MS : 1000);
    return () => { stop = true; clearTimeout(timer); document.removeEventListener('visibilitychange', onVisible); };
  }, [hasActive]);

  if (!signedIn) return { jobs: ready ? [] : null, limits: null, error: '', reload: load };
  return { jobs, limits, error, reload: load };
}
