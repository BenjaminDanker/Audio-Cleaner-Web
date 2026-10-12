import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api.js';
import { ACTIVE } from './jobs.js';

/** Job list for the signed-in user; polls each active job every 1.5 s (GET /api/jobs/{id}). */
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
    const tick = async () => {
      const active = (jobsRef.current || []).filter((j) => ACTIVE.has(j.status));
      const updates = await Promise.all(active.map((j) => api(`jobs/${j.id}`, { redirectOn401: false }).catch(() => null)));
      if (stop) return;
      const byId = new Map(updates.filter(Boolean).map((j) => [j.id, j]));
      const cur = jobsRef.current || [];
      let ended = false;
      const next = cur.map((j) => {
        const n = byId.get(j.id);
        if (n && ACTIVE.has(j.status) && !ACTIVE.has(n.status)) ended = true;
        return n || j;
      });
      setJobs(next);
      if (ended) finished.current?.();
      timer = setTimeout(tick, 1500);
    };
    timer = setTimeout(tick, 1000);
    return () => { stop = true; clearTimeout(timer); };
  }, [hasActive]);

  return { jobs, error, reload: load };
}
