// The editor's calls to the server (docs/api.md section 8).

import { api } from '../api.js';
import { Timeline } from './timeline.js';

/** GET /api/jobs/{id}/timeline, with the timelines read into Timeline objects. */
export async function loadEditor(jobId, signal) {
  const r = await api(`jobs/${jobId}/timeline`, { signal, redirectOn401: false });
  const seconds = r.seconds;
  return {
    ...r,
    base: Timeline.fromJSON(r.timeline, seconds),
    rendered: r.rendered ? Timeline.fromJSON(r.rendered, seconds) : null,
    cached: r.cached || {},
    secondTrack: !!r.secondTrack,
  };
}

export const quoteTimeline = (jobId, tl, signal) => api(`jobs/${jobId}/quote`, { method: 'POST', body: { timeline: tl.toJSON() }, signal });

/** Process (paid, only what is missing is made) or export (free, from the cache alone). */
export function saveTimeline(jobId, tl, { free, maxCostMicros }) {
  return api(`jobs/${jobId}/${free ? 'export' : 'process'}`, {
    method: 'POST',
    body: { timeline: tl.toJSON(), ...(free ? {} : { maxCostMicros }) },
  });
}

/** Estimate for a job made with parts (the upload form): the timeline replaces the sound options. */
export function estimateWithTimeline({ kind, transcribeTier, durationSeconds, options, timeline }, signal) {
  return api('jobs/estimate', {
    method: 'POST',
    body: { kind, ...(transcribeTier ? { transcribeTier } : {}), durationSeconds, options, timeline: timeline.toJSON() },
    signal,
  });
}
