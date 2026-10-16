import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api.js';
import { ACTIVE } from './jobs.js';

const POLL_MS = 1500; // while a job is uploading or running (its progress moves)
const SLOW_POLL_MS = 5000; // while jobs only wait in the queue or are with our speech partner: nothing but the queue position changes

const MOVING = new Set(['uploading', 'running', 'canceling']);

/** Milliseconds until the list is fetched again: quick while some job's progress moves, slow while jobs only wait. */
export function pollDelay(list) {
  return (list || []).some((j) => MOVING.has(j.status)) ? POLL_MS : SLOW_POLL_MS;
}

/**
 * Job list for the signed-in user. While any job is active the list itself is fetched again (one request,
 * whatever the number of jobs; GET /api/jobs carries each job's status, stage and progress): every 1.5 s while one is
 * uploading or running, every 5 s while they only wait (queued, or in the cloud stage). If the server asks us to wait
 * (HTTP 429, Retry-After) or cannot be reached, polling slows down instead of hammering it, and a short note says so. A
 * sign-in that ran out (401) takes the visitor to the sign-in page like every other call.
 */
export function useJobs({ onFinished } = {}) {
  const [jobs, setJobs] = useState(null);
  const [error, setError] = useState('');
  const jobsRef = useRef(null);
  jobsRef.current = jobs;
  const finished = useRef(onFinished);
  finished.current = onFinished;

  const load = useCallback(async () => {
    try {
      const r = await api('jobs');
      setJobs(r.jobs);
      setError('');
    } catch (e) {
      setError(e.detail || 'Could not load your jobs.');
      setJobs((j) => j || []);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const hasActive = !!jobs?.some((j) => ACTIVE.has(j.status));
  useEffect(() => {
    if (!hasActive) return undefined;
    let stop = false;
    let timer;
    let failures = 0;
    const tick = async () => {
      let wait = POLL_MS;
      try {
        const r = await api('jobs');
        if (stop) return;
        const before = jobsRef.current || [];
        const now = new Map(r.jobs.map((j) => [j.id, j]));
        const ended = before.some((j) => ACTIVE.has(j.status) && now.has(j.id) && !ACTIVE.has(now.get(j.id).status));
        failures = 0;
        wait = pollDelay(r.jobs);
        setJobs(r.jobs);
        setError('');
        if (ended) finished.current?.();
      } catch (e) {
        if (stop || e?.name === 'AbortError') return;
        if (e.status === 401) {
          setError('Your sign-in ended. Please sign in again to see your jobs.');
          return; // the api layer is already taking the visitor to the sign-in page
        }
        failures += 1;
        if (e.status === 429) {
          wait = Math.min(120_000, Math.max(5000, (e.retryAfter || 5) * 1000));
          setError(e.detail || 'Lots of requests just now. Updates will be a little slower for a moment.');
        } else {
          wait = Math.min(15_000, POLL_MS * 2 ** Math.min(failures, 4));
          if (failures >= 3) setError('We lost touch with the server for a moment. Still trying…');
        }
      }
      if (!stop) timer = setTimeout(tick, wait);
    };
    timer = setTimeout(tick, 1000);
    return () => { stop = true; clearTimeout(timer); };
  }, [hasActive]);

  return { jobs, error, reload: load };
}
