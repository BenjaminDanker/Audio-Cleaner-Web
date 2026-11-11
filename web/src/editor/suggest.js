// The suggestions of a job, used by the editor (its pins and count): fetched once per job (GET /api/jobs/{id}/suggestions,
// made by the server during the first run where the job asked for them), with the ones the person dismissed remembered in this
// browser. A job whose suggestions this browser found itself before the upload did not ask: those are kept here with it.

import { useCallback, useSyncExternalStore } from 'react';
import { api } from '../api.js';
import { LAYER_NAMES, NOISE_LEVELS, levelOf, rangeText } from './timeline.js';

const KEY = 'acw.sg.dismissed.';
const OWN = 'acw.sg.own.';
const stores = new Map(); // job id -> {state, listeners, promise}
const EMPTY = { loaded: false, list: [], dismissed: [] };

export const keyOf = (s) => `${s.kind}:${Math.round(s.start * 10)}-${Math.round(s.end * 10)}`;

function readDismissed(id) {
  try { return JSON.parse(localStorage.getItem(KEY + id) || '[]'); } catch { return []; }
}
function saveDismissed(id, list) {
  try { localStorage.setItem(KEY + id, JSON.stringify(list)); } catch { /* not remembered: private window */ }
}

function store(id) {
  let s = stores.get(id);
  if (!s) {
    s = { state: { ...EMPTY, dismissed: readDismissed(id) }, listeners: new Set(), promise: null };
    stores.set(id, s);
  }
  return s;
}
const set = (id, patch) => {
  const s = store(id);
  s.state = { ...s.state, ...patch };
  for (const fn of s.listeners) fn();
};

/** Ask for the suggestions of a job once (a job without any answers an empty list). */
export function loadSuggestions(id) {
  const s = store(id);
  if (!s.promise) {
    s.promise = api(`jobs/${id}/suggestions`, { redirectOn401: false }).then(
      (r) => set(id, { loaded: true, list: (r?.suggestions?.suggestions || []).map((x) => ({ ...x, key: keyOf(x) })) }),
      () => set(id, { loaded: true, list: [] }),
    );
  }
  return s.promise;
}

/** Use the suggestions found in this browser before the upload (the upload form) under a name of their own. */
export function setSuggestions(id, list) {
  store(id).promise = Promise.resolve();
  set(id, { loaded: true, list: list.map((x) => ({ ...x, key: keyOf(x) })) });
}

/** Keep what this browser found before the upload with the job it became (our side does not look again for a job that came with them). */
export function keepOwnSuggestions(id, list) {
  try { localStorage.setItem(OWN + id, JSON.stringify(list)); } catch { /* not kept: the tab that made the job still has them */ }
}
/** The ones kept for a job, or null. */
export function ownSuggestions(id) {
  try {
    const list = JSON.parse(localStorage.getItem(OWN + id) || 'null');
    return Array.isArray(list) ? list : null;
  } catch {
    return null;
  }
}
/** Forget the ones of jobs that are not among `keep` (a Set of job ids) any more. */
export function pruneOwnSuggestions(keep) {
  try {
    for (const k of Object.keys(localStorage)) if (k.startsWith(OWN) && !keep.has(k.slice(OWN.length))) localStorage.removeItem(k);
  } catch { /* nothing to clean */ }
}

export function dismiss(id, key) {
  const d = [...new Set([...store(id).state.dismissed, key])];
  saveDismissed(id, d);
  set(id, { dismissed: d });
}

/** What is known about a job's suggestions now (not reactive): {loaded, list, dismissed}. */
export const suggestionState = (id) => store(id).state;

export function forgetSuggestions(id) {
  stores.delete(id);
}

/** {loaded, list, dismissed}: the suggestions of the job (nothing is fetched while `enabled` is false). */
export function useSuggestions(id, enabled = true) {
  const s = store(id);
  const sub = useCallback((fn) => { s.listeners.add(fn); if (enabled) loadSuggestions(id); return () => s.listeners.delete(fn); }, [s, id, enabled]);
  return useSyncExternalStore(sub, () => s.state);
}

/* ------------------------------------------------------------------------------------------------ words */

const STRENGTH_WORD = (db) => NOISE_LEVELS[levelOf(db)].label;

/** "sounds distorted." / "is noisier and quiet." */
export function sayWhat(s) {
  const r = (s.reasons || []).filter(Boolean);
  if (!r.length) return 'sounds different from the rest.';
  return `${r.length > 1 ? `${r.slice(0, -1).join(', ')} and ${r[r.length - 1]}` : r[0]}.`;
}

/** The question a suggestion asks, and the label of its button. */
export function sayAsk(s) {
  if (s.rebuild) return { ask: 'Rebuild just this part?', act: 'Rebuild this part', does: 'Voice rebuild' };
  if (s.noise) {
    const w = STRENGTH_WORD(s.noise.strength_db);
    return { ask: `${w} here?`, act: `Use ${w} here`, does: LAYER_NAMES.noise };
  }
  return { ask: 'Change this part?', act: 'Use this', does: 'this part' };
}

export const rangeOf = (s) => rangeText(s.start, s.end);

/** The operations that apply a suggestion to a timeline (suggest.apply): Voice rebuild where it rebuilds; else no rebuild and its noise removal. */
export function suggestionOps(s, tag) {
  if (s.rebuild) return [{ layer: 'rebuild', start: s.start, end: s.end, settings: {}, tag }];
  const ops = [{ layer: 'rebuild', start: s.start, end: s.end, settings: null, tag }];
  if (s.noise) ops.push({ layer: 'noise', start: s.start, end: s.end, settings: { strength_db: s.noise.strength_db, remover: s.noise.remover || 'auto' }, tag });
  return ops;
}
