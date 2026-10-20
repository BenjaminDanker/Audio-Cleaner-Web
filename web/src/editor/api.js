// The editor's calls to the server (docs/api.md section 8).

import { api, uploadBlob } from '../api.js';
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
export function saveTimeline(jobId, tl, { free, maxCostMicros, audioFormat }) {
  return api(`jobs/${jobId}/${free ? 'export' : 'process'}`, {
    method: 'POST',
    body: { timeline: tl.toJSON(), ...(free ? {} : { maxCostMicros }), ...(audioFormat ? { audioFormat } : {}) },
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

/* ------------------------------------------------------------------------------------------------ timeline version 2 (docs/api.md section 8) */

/** The peaks of one layer of the job for the waveform: GET /api/jobs/{id}/peaks/{source|echo|noise|rebuild} -> {rate: 4, seconds, peaks: [number|null]}. */
export const loadLayerPeaks = (jobId, layer, signal) => api(`jobs/${jobId}/peaks/${layer}`, { signal, redirectOn401: false });

/** The cleaned audio of a range the browser did itself (FLAC, 48 kHz, the job's channels): PUT /api/jobs/{id}/browser/noise?start&end. */
export function uploadBrowserNoise(jobId, { start, end }, blob, { signal, onProgress } = {}) {
  return uploadBlob(`api/jobs/${jobId}/browser/noise?start=${start}&end=${end}`, blob, { signal, onProgress });
}

/** The cues of a range the browser transcribed: PUT /api/jobs/{id}/browser/cues?start&end with [{start, end, text}] (absolute seconds). */
export const uploadBrowserCues = (jobId, { start, end }, cues, signal) => api(`jobs/${jobId}/browser/cues?start=${start}&end=${end}`, { method: 'PUT', body: cues, signal });

/** What the browser still has to send before the job goes on: the 409 `needs_uploads` of process / export, or the `needs` of create. */
export const needsOf = (e) => (e?.code === 'needs_uploads' ? e.needs || e.body?.needs || [] : null);

/** The price of a job's timeline before it is made (nothing is held): kind, durations and the timeline. */
export const estimateTimeline = (body, signal) => api('jobs/estimate', { method: 'POST', body, signal });
