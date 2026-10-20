// The timeline of the per-part editor in the browser: what is done to which part of a recording. The same format and rules as
// common/acw_common/timeline.py and the server's API (docs/api.md section 8): layers over seconds, each part with its own settings;
// parts of a layer never overlap (a part painted over old ones replaces them), neighbours with the same settings are one part, times
// are kept to the millisecond and clamped to the recording. Pure functions and one class: the editor, its tests and the upload form
// all use them.

export const LAYERS = ['echo', 'noise', 'rebuild', 'leveling'];
/** Version 2 adds a layer that is not part of the sound (the subtitles) and noise parts the browser cleaned itself. */
export const SUBTITLES = 'subtitles';
export const ALL_LAYERS = [...LAYERS, SUBTITLES];
export const QUALITIES = ['free', 'standard', 'accurate'];
export const MAX_TRANSLATIONS = 3;
export const MAX_PARTS = 500;
export const DEFAULT_LOUDNESS = -16;

/** The names a person sees (the price list's words). */
export const LAYER_NAMES = { echo: 'Echo removal', noise: 'Noise removal', rebuild: 'Voice rebuild', leveling: 'Leveling', subtitles: 'Subtitles' };

/** Noise removal strengths: dB of the attenuation limit, as the server takes them. `off` is no part at all. */
export const NOISE_LEVELS = [
  { id: 'off', label: 'Off', db: 0 },
  { id: 'gentle', label: 'Gentle', db: 12 },
  { id: 'balanced', label: 'Balanced', db: 24 },
  { id: 'strong', label: 'Strong', db: 40 },
  { id: 'max', label: 'Maximum', db: 100 },
];

/** The level (index into NOISE_LEVELS) of a strength in dB: the nearest named one. */
export function levelOf(db) {
  if (!(db > 0)) return 0;
  let best = 1;
  for (let i = 1; i < NOISE_LEVELS.length; i++) if (Math.abs(NOISE_LEVELS[i].db - db) < Math.abs(NOISE_LEVELS[best].db - db)) best = i;
  return best;
}

export const ms = (t) => Math.round(Number(t) * 1000) / 1000;

export class TimelineError extends Error {}

/** The checked settings of a part of `layer` (defaults filled in). */
export function settingsOf(layer, raw = {}) {
  const r = { ...(raw || {}) };
  let out;
  if (layer === 'noise' && r.browser === true) {
    // the browser cleaned this range itself and uploaded it: the strength it used is kept for showing, the remover is not used
    out = { browser: true };
    const db = Number(r.strength_db ?? r.strength);
    if (Number.isFinite(db) && db > 0 && db <= 100) out.strength_db = db;
  } else if (layer === 'noise') {
    let db = r.strength_db ?? r.strength ?? 100;
    if (typeof db === 'string') {
      const named = NOISE_LEVELS.find((l) => l.id === db.toLowerCase()) || (db.toLowerCase() === 'maximum' ? NOISE_LEVELS[4] : null);
      if (!named || named.id === 'off') throw new TimelineError(`unknown strength ${db}`);
      db = named.db;
    }
    db = Number(db);
    if (!Number.isFinite(db) || db <= 0 || db > 100) throw new TimelineError('strength_db must be above 0 and at most 100');
    const remover = r.remover ?? 'auto';
    if (!['auto', 'light', 'heavy'].includes(remover)) throw new TimelineError('remover must be auto, light or heavy');
    out = { strength_db: db, remover };
  } else if (layer === 'leveling') {
    let lufs = r.loudness_lufs === undefined ? DEFAULT_LOUDNESS : r.loudness_lufs;
    if (lufs !== null) {
      lufs = Number(lufs);
      if (!Number.isFinite(lufs) || lufs < -70 || lufs > 0) throw new TimelineError('loudness_lufs must be between -70 and 0 (or null)');
    }
    out = { loudness_lufs: lufs };
  } else if (layer === SUBTITLES) {
    const quality = r.quality;
    if (!QUALITIES.includes(quality)) throw new TimelineError('subtitles need a quality: free, standard or accurate');
    const language = r.language === undefined || r.language === null || r.language === '' ? null : String(r.language).trim().toLowerCase();
    if (language !== null && !/^[a-z]{2,3}$/.test(language)) throw new TimelineError('language must be a code such as en, or null to detect it');
    const translate = [...new Set((r.translate || []).map((c) => String(c).trim().toLowerCase()).filter(Boolean))];
    if (translate.length > MAX_TRANSLATIONS) throw new TimelineError(`at most ${MAX_TRANSLATIONS} translation languages for a part`);
    if (translate.includes(language)) throw new TimelineError('a translation language is the same as the spoken language of the part');
    if (translate.length && quality === 'free') throw new TimelineError('Translations need Standard or Accurate subtitles: free subtitles are made in the browser and can’t be translated');
    out = { quality, language, translate };
  } else if (layer === 'echo' || layer === 'rebuild') {
    out = {};
  } else {
    throw new TimelineError(`unknown layer ${layer}`);
  }
  return out;
}

const sameSettings = (a, b) => {
  const ka = Object.keys(a);
  return ka.length === Object.keys(b).length && ka.every((k) => (Array.isArray(a[k]) ? JSON.stringify(a[k]) === JSON.stringify(b[k]) : a[k] === b[k]));
};
const cloneSettings = (s) => Object.fromEntries(Object.entries(s).map(([k, v]) => [k, Array.isArray(v) ? [...v] : v]));

export class Timeline {
  constructor(seconds) {
    this.seconds = ms(Math.max(0, Number(seconds) || 0));
    this.layers = { echo: [], noise: [], rebuild: [], leveling: [], subtitles: [] };
  }

  /** Every layer asked for on the whole file. `noise` / `leveling`: true for the defaults, an object for these settings. */
  static wholeFile(seconds, { echo = false, noise = true, rebuild = false, leveling = true, subtitles = false } = {}) {
    const t = new Timeline(seconds);
    for (const [name, on] of [['echo', echo], ['noise', noise], ['rebuild', rebuild], ['leveling', leveling], ['subtitles', subtitles]]) {
      if (on) t.paint(name, 0, t.seconds, typeof on === 'object' ? on : undefined);
    }
    return t;
  }

  copy() {
    const t = new Timeline(this.seconds);
    for (const n of ALL_LAYERS) t.layers[n] = this.layers[n].map((p) => ({ start: p.start, end: p.end, settings: cloneSettings(p.settings) }));
    return t;
  }

  parts(layer) {
    return this.layers[layer];
  }

  /** The part of `layer` that covers time `t` (start inclusive, end exclusive), or null. */
  partAt(layer, t) {
    return this.layers[layer].find((p) => p.start <= t && t < p.end) || null;
  }

  erase(layer, start, end) {
    const a = ms(Math.max(0, start));
    const b = ms(Math.min(this.seconds, end));
    if (!(b > a)) return this;
    const out = [];
    for (const p of this.layers[layer]) {
      if (p.end <= a || p.start >= b) { out.push(p); continue; }
      if (p.start < a) out.push({ start: p.start, end: a, settings: cloneSettings(p.settings) });
      if (p.end > b) out.push({ start: b, end: p.end, settings: cloneSettings(p.settings) });
    }
    this.layers[layer] = out;
    return this;
  }

  /** Put a part on the layer, replacing whatever it covers. */
  paint(layer, start, end, settings) {
    if (!(Number(end) >= Number(start))) throw new TimelineError('a part must not end before it starts');
    const s = settingsOf(layer, settings);
    const a = ms(Math.max(0, start));
    const b = ms(Math.min(this.seconds, end));
    if (!(b > a)) return this;
    this.erase(layer, a, b);
    const parts = this.layers[layer];
    parts.push({ start: a, end: b, settings: s });
    parts.sort((x, y) => x.start - y.start);
    const merged = [];
    for (const p of parts) {
      const last = merged[merged.length - 1];
      if (last && last.end === p.start && sameSettings(last.settings, p.settings)) last.end = p.end;
      else merged.push(p);
    }
    if (merged.length > MAX_PARTS) throw new TimelineError(`at most ${MAX_PARTS} parts per layer`);
    this.layers[layer] = merged;
    return this;
  }

  /** The settings of `layer` over [a, b) if they are all the same (a part or a stretch of one), `null` if they differ, `false` if off. */
  uniform(layer, a, b) {
    const parts = this.layers[layer];
    const inside = parts.filter((p) => p.end > a && p.start < b);
    if (!inside.length) return false;
    let cover = 0;
    for (const p of inside) cover += Math.min(p.end, b) - Math.max(p.start, a);
    if (cover < b - a - 0.0005) return null;
    return inside.every((p) => sameSettings(p.settings, inside[0].settings)) ? inside[0].settings : null;
  }

  isEmpty() {
    return ALL_LAYERS.every((n) => !this.layers[n].length);
  }

  /** Does this timeline use what version 2 added (subtitles, or noise cleaned by the browser)? */
  usesV2() {
    return this.layers.subtitles.length > 0 || this.layers.noise.some((p) => p.settings.browser === true);
  }

  /** The server's JSON. Version 2 only for a timeline that needs it, so the rest is written exactly as before. */
  toJSON() {
    const layers = {};
    for (const n of ALL_LAYERS) if (this.layers[n].length) layers[n] = this.layers[n].map((p) => ({ start: p.start, end: p.end, ...cloneSettings(p.settings) }));
    return { version: this.usesV2() ? 2 : 1, seconds: this.seconds, layers };
  }

  equals(other) {
    return !!other && this.seconds === other.seconds && JSON.stringify(this.toJSON().layers) === JSON.stringify(other.toJSON().layers);
  }

  static fromJSON(obj, seconds = null) {
    if (!obj || typeof obj !== 'object' || ![1, 2].includes(obj.version)) throw new TimelineError('unsupported timeline');
    const t = new Timeline(seconds ?? obj.seconds);
    for (const [name, parts] of Object.entries(obj.layers || {})) {
      if (!ALL_LAYERS.includes(name)) throw new TimelineError(`unknown layer ${name}`);
      if (obj.version === 1 && name === SUBTITLES) throw new TimelineError('subtitles need version 2');
      for (const p of parts) {
        const { start, end, ...rest } = p;
        if (obj.version === 1 && rest.browser) throw new TimelineError('browser parts need version 2');
        t.paint(name, start, end, rest);
      }
    }
    return t;
  }
}

/* ------------------------------------------------------------------------------------------------ ranges */

/** Merge and sort [s, e] ranges. */
export function mergeRanges(ranges) {
  const out = [];
  for (const [s, e] of [...ranges].filter((r) => r[1] > r[0]).sort((a, b) => a[0] - b[0])) {
    const last = out[out.length - 1];
    if (last && s <= last[1]) last[1] = Math.max(last[1], e);
    else out.push([s, e]);
  }
  return out;
}

/** The parts of [a, b] ranges that lie inside `within`. */
export function intersect(ranges, within) {
  const out = [];
  for (const [s, e] of ranges) {
    for (const [ws, we] of within) {
      const x = Math.max(s, ws);
      const y = Math.min(e, we);
      if (y > x) out.push([x, y]);
    }
  }
  return out;
}

/** The parts of `ranges` that `cut` does not cover. */
export function subtract(ranges, cut) {
  let cur = ranges.map((r) => [...r]);
  for (const [cs, ce] of mergeRanges(cut)) {
    const next = [];
    for (const [s, e] of cur) {
      if (ce <= s || cs >= e) { next.push([s, e]); continue; }
      if (cs > s) next.push([s, cs]);
      if (ce < e) next.push([ce, e]);
    }
    cur = next;
  }
  return cur;
}

/** "0:15–0:25" (whole seconds; a stretch under a second is shown to the tenth). */
export function clock(t) {
  const s = Math.max(0, Math.round(t));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}` : `${m}:${String(s % 60).padStart(2, '0')}`;
}
export const rangeText = (a, b) => `${clock(a)}–${clock(b)}`;

/** The short name of a part of a layer for its chip: "Gentle · 0:40–0:45", "whole file", "Maximum · whole file". */
export function partLabel(layer, p, seconds) {
  const whole = p.start <= 0.0005 && p.end >= seconds - 0.0005;
  const where = whole ? 'whole file' : rangeText(p.start, p.end);
  if (layer === 'noise') return `${NOISE_LEVELS[levelOf(p.settings.strength_db)].label} · ${where}`;
  return where;
}

/* ------------------------------------------------------------------------------------------------ edits as a list */

/**
 * The editor keeps the person's changes as a list of operations over a base timeline, so any of them can be taken back and the base can
 * change underneath (the upload form's choices): [{layer, start, end, settings | null (erase), tag?}]. The timeline is base + ops.
 */
export function replay(base, ops) {
  const t = base.copy();
  for (const o of ops) {
    if (o.settings === null) t.erase(o.layer, o.start, o.end);
    else t.paint(o.layer, o.start, o.end, o.settings);
  }
  return t;
}

/** The operations that make [a, b) of `target` what it is in `source` (undo the changes inside a part). */
export function restoreOps(source, a, b) {
  const ops = [];
  for (const layer of LAYERS) {
    ops.push({ layer, start: a, end: b, settings: null });
    for (const p of source.parts(layer)) {
      const s = Math.max(p.start, a);
      const e = Math.min(p.end, b);
      if (e > s) ops.push({ layer, start: s, end: e, settings: { ...p.settings } });
    }
  }
  return ops;
}

/** The stretches where `tl` and `other` differ in any layer (a part added, taken away or with other settings): [[start, end], ...]. */
export function differences(tl, other) {
  const edges = new Set([0, tl.seconds]);
  for (const t of [tl, other]) for (const layer of LAYERS) for (const p of t.parts(layer)) { edges.add(p.start); edges.add(p.end); }
  const xs = [...edges].filter((t) => t >= 0 && t <= tl.seconds).sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i + 1 < xs.length; i++) {
    const mid = (xs[i] + xs[i + 1]) / 2;
    const differ = LAYERS.some((layer) => JSON.stringify(tl.partAt(layer, mid)?.settings ?? null) !== JSON.stringify(other ? other.partAt(layer, mid)?.settings ?? null : null));
    if (differ) out.push([xs[i], xs[i + 1]]);
  }
  return mergeRanges(out);
}

/** The parts of a layer inside [a, b), cut to it: [[start, end, settings], ...]. */
export function clipped(tl, layer, a, b) {
  return tl.parts(layer).filter((p) => p.end > a && p.start < b).map((p) => [Math.max(p.start, a), Math.min(p.end, b), p.settings]);
}

/** Does `tl` differ from `base` inside [a, b)? */
export function changedIn(tl, base, a, b) {
  return LAYERS.some((layer) => JSON.stringify(clipped(tl, layer, a, b)) !== JSON.stringify(clipped(base, layer, a, b)));
}
