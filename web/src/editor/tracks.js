// The editor's model: four tracks (Voice, Echo, Leveling, Subtitles), each a list of contiguous sections that cover the recording, and
// what the person is working on (the selection, the suggestions, undo). Pure functions on one plain object `S`, so the page and the
// unit tests (tests/studio-unit.mjs) use the same code. The rules are the approved prototype's (edit / splitAll / normalize):
//   - every change applies to the selection (the whole file when nothing is selected), cuts the sections at its edges, and
//     neighbours that end up the same join again by themselves; a split stays until a change touches it
//   - prices come from the live price list (ratesFrom), billed per second of audio, no minimum charge
// and the mapping to the server's timeline (editor/timeline.js, docs/api.md section 8): Voice -> noise / rebuild, Echo -> echo,
// Leveling -> leveling, Subtitles -> subtitles; Free noise removal -> a noise part the browser cleaned itself (`browser: true`).

import { STRENGTHS } from '../strengths.js';
import { Timeline, levelOf } from './timeline.js';

/* ------------------------------------------------------------------------------------------------ words and constants */

export const STR = ['', 'Gentle', 'Balanced', 'Strong', 'Max'];
export const STR_DB = [0, 12, 24, 40, 100];
export const REM = ['Auto', 'Light', 'Heavy'];
export const REMOVERS = ['auto', 'light', 'heavy'];
export const SUBQ = ['Off', 'Free', 'Standard', 'Extra accurate'];
export const QUALITY = [null, 'free', 'standard', 'accurate'];
export const TARGETS = ['Podcast', 'YouTube', 'Broadcast', 'Keep loudness'];
export const LUFS = [null, -16, -14, -23, null]; // by lvl; 0 is off and 4 keeps the level
export const TRACKS = [['voice', 'Voice'], ['echo', 'Echo'], ['level', 'Leveling'], ['subs', 'Subtitles']];
export const MAX_TRANSLATIONS = 3;
/** The most a Free (in the browser) strength can be: Gentle to Strong. */
export const FREE_MAX_STR = 3;

export const p2 = (n) => String(n).padStart(2, '0');
export const fmt = (t) => { t = Math.max(0, Math.round(t)); const h = Math.floor(t / 3600); const m = Math.floor((t % 3600) / 60); const s = t % 60; return h ? `${h}:${p2(m)}:${p2(s)}` : `${m}:${p2(s)}`; };
export const lenTxt = (n) => {
  n = Math.round(n);
  if (n < 60) return `${n} s`;
  if (n < 3600) return `${Math.floor(n / 60)} min${n % 60 ? ` ${n % 60} s` : ''}`;
  const m = Math.floor((n % 3600) / 60);
  return `${Math.floor(n / 3600)} h${m ? ` ${m} min` : ''}`;
};
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const EPS = 1e-6;

/** The prototype's price words: "Free", "<$0.001", "$0.024", "$1.50". Micros of a dollar. */
export const money = (m) => (m <= 0 ? 'Free' : m < 1000 ? '<$0.001' : `$${m >= 100000 || m % 10000 === 0 ? (m / 1e6).toFixed(2) : (m / 1e6).toFixed(3)}`);

/* ------------------------------------------------------------------------------------------------ sections and the state */

export const sec = (s, e, o) => ({ s, e, hard: false, auto: false, ...o });

/** The sections a file starts with: the whole recording, as the usual choices say (Automatic may change them after listening). */
export function freshTracks(T, { second = false, auto = true } = {}) {
  return {
    voice: [sec(0, T, { mode: 1, str: 2, kind: 'paid', rem: 0, db: null, auto })],
    echo: [sec(0, T, { on: second ? 1 : 0, auto: !!second && auto })],
    level: [sec(0, T, { lvl: 2, lufs: null, auto })],
    subs: [sec(0, T, { q: 0, lang: 'auto', tr: [], auto })],
  };
}

/** A new state for a recording `T` seconds long. `second`: the file has a second audio track (echo removal can use it). */
export function makeState(T, { second = false, auto = true } = {}) {
  return { T, second: !!second, tracks: freshTracks(T, { second, auto }), sel: null, sugg: [], past: [], future: [] };
}

/** The translations that count: only with Standard or Extra accurate, never the spoken language itself. */
export const trOf = (x) => (x.q >= 2 ? (x.tr || []).filter((l) => l !== x.lang) : []);

/** The strength of a Voice section in dB (a custom strength from the slider, else the named one). */
export const dbOf = (x) => (x.db != null ? x.db : STR_DB[x.str]);
/** The named strength nearest to a dB value (1 to 4). */
export const strOfDb = (db) => levelOf(db) || 1;

/** A string that is the same exactly when two sections do the same thing. */
export function vkey(tr, x) {
  if (tr === 'voice') return x.mode === 1 ? `1:${x.str}:${x.kind}:${x.kind === 'paid' ? x.rem : 0}:${x.db != null ? Math.round(x.db * 100) : ''}` : String(x.mode);
  if (tr === 'subs') return x.q ? `${x.q}:${x.lang}:${trOf(x).join('.')}` : '0';
  if (tr === 'level') return x.lvl ? `${x.lvl}:${x.lufs ?? ''}` : '0';
  return String(x.on);
}

export const ranges = (S) => S.sel || [{ s: 0, e: S.T }];
export const hit = (S, x) => ranges(S).some((r) => x.e > r.s && x.s < r.e);
export const overl = (S, tr) => S.tracks[tr].filter((x) => hit(S, x));
export const uni = (S, tr, key, list) => { const L = list || overl(S, tr); return L.every((x) => x[key] === L[0][key]) ? L[0][key] : null; };

/** Set the selection: merged, sorted; nothing, or the whole file, is no selection (the whole file is the default). */
export function setSel(S, list) {
  const a = list.filter((r) => r.e > r.s).map((r) => ({ s: r.s, e: r.e })).sort((x, y) => x.s - y.s);
  const m = [];
  a.forEach((r) => { const l = m[m.length - 1]; if (l && r.s <= l.e) l.e = Math.max(l.e, r.e); else m.push(r); });
  S.sel = !m.length || (m.length === 1 && m[0].s <= 0 && m[0].e >= S.T) ? null : m;
}
export const covered = (S, s, e) => (S.sel || []).some((r) => r.s <= s && r.e >= e);
/** Add a range to the selection, or take it out when it is all inside already (Shift-click, Select more, long-press). */
export function toggleRange(S, s, e) {
  const cur = S.sel ? S.sel.map((r) => ({ s: r.s, e: r.e })) : [];
  if (S.sel && covered(S, s, e)) {
    const out = [];
    cur.forEach((r) => { if (r.e <= s || r.s >= e) out.push(r); else { if (r.s < s) out.push({ s: r.s, e: s }); if (r.e > e) out.push({ s: e, e: r.e }); } });
    setSel(S, out);
  } else { cur.push({ s, e }); setSel(S, cur); }
}
/** "Whole file · 3:00", "0:15–0:25 · 10 s", "2 ranges · 1:10 total". */
export function selTitle(S) {
  const R = ranges(S);
  if (!S.sel) return `Whole file · ${fmt(S.T)}`;
  if (R.length === 1) return `${fmt(R[0].s)}–${fmt(R[0].e)} · ${lenTxt(R[0].e - R[0].s)}`;
  return `${R.length} ranges · ${fmt(R.reduce((a, r) => a + r.e - r.s, 0))} total`;
}

export const at = (tracks, tr, t) => tracks[tr].find((x) => t >= x.s && t < x.e) || tracks[tr][tracks[tr].length - 1];

export function cut(S, L, t) {
  if (t <= 0 || t >= S.T) return;
  const i = L.findIndex((x) => x.s < t && t < x.e);
  if (i < 0) return;
  const a = L[i];
  const b = { ...a, s: t, hard: false };
  a.e = t;
  L.splice(i + 1, 0, b);
}
export function normalize(S, tr) {
  const L = S.tracks[tr];
  for (let i = 1; i < L.length;) {
    const a = L[i - 1];
    const b = L[i];
    if (!b.hard && vkey(tr, a) === vkey(tr, b)) { a.e = b.e; a.auto = a.auto && b.auto; L.splice(i, 1); } else i++;
  }
}
/** Change what a track does between s and e. `patch`: an object, or a function of the section that returns one. */
export function edit(S, tr, s, e, patch) {
  const L = S.tracks[tr];
  cut(S, L, s); cut(S, L, e);
  L.forEach((x) => { if (x.s >= s - EPS && x.e <= e + EPS) Object.assign(x, typeof patch === 'function' ? patch(x) : patch, { auto: false }); });
  // everything merges by itself: a split stays until a change touches it, then neighbours that are the same join again
  Object.keys(S.tracks).forEach((k) => { S.tracks[k].forEach((x) => { if (x.hard && x.s >= s - EPS && x.s <= e + EPS) x.hard = false; }); normalize(S, k); });
}
/** Split every track at time t (a split stays until a change touches it). false when t is not inside the file. */
export function splitAll(S, t) {
  t = Math.round(t);
  if (t <= 0 || t >= S.T) return false;
  let did = false;
  Object.keys(S.tracks).forEach((tr) => {
    const L = S.tracks[tr];
    const i = L.findIndex((x) => x.s < t && t < x.e);
    if (i >= 0) { cut(S, L, t); L[i + 1].hard = true; did = true; }
  });
  return did;
}

/** Apply a change to the selection (or the whole file): the prototype's apply(). */
export function applyToSel(S, tr, patch) {
  ranges(S).map((r) => ({ s: r.s, e: r.e })).forEach((r) => edit(S, tr, r.s, r.e, patch));
}

export const snapJ = (S) => JSON.stringify({ tracks: S.tracks, sugg: S.sugg });
export function restore(S, j) { const o = JSON.parse(j); S.tracks = o.tracks; S.sugg = o.sugg; }
export function pushUndo(S) { S.past.push(snapJ(S)); if (S.past.length > 100) S.past.shift(); S.future = []; }
export function undo(S) { if (!S.past.length) return false; S.future.push(snapJ(S)); restore(S, S.past.pop()); return true; }
export function redo(S) { if (!S.future.length) return false; S.past.push(snapJ(S)); restore(S, S.future.pop()); return true; }

/** A suggestion whose change is all there on the Voice track is done. */
export function autoDone(S) {
  S.sugg.forEach((g) => {
    if (g.st !== 'new') return;
    const L = S.tracks.voice.filter((x) => x.e > g.s && x.s < g.e);
    if (L.length && L.every((x) => Object.keys(g.patch).every((k) => x[k] === g.patch[k]))) g.st = 'applied';
  });
}

/* ------------------------------------------------------------------------------------------------ Free and Standard */

/** What Free (in the browser) can do for a section: Gentle to Strong, no choice of remover. */
export const freePatch = (x) => ({ kind: 'free', str: Math.min(x.str, FREE_MAX_STR), db: x.db != null ? Math.min(x.db, STR_DB[FREE_MAX_STR]) : null });

/** The Free noise removal and echo removal can't be on the same stretch (the free cleanup can't use the second track). */
export function echoBlocked(S, list) {
  return (list || overl(S, 'voice')).some((x) => x.mode === 1 && x.kind === 'free');
}

/* ------------------------------------------------------------------------------------------------ prices */

const findRate = (pricing, product, tier) => pricing?.rates?.find((r) => r.product === product && r.tier === tier) || null;

/** The price list as micros of a dollar for one second of audio: {noise, rebuild, echo, level, std, ext, tr}. null: the list is not here. */
export function ratesFrom(pricing) {
  if (!pricing?.rates) return null;
  const per = (product, tier) => {
    const r = findRate(pricing, product, tier);
    return r ? (Math.round(r.usdPerMinute * 1e6)) / 60 : 0;
  };
  return {
    noise: per('denoise', 'standard'), rebuild: per('denoise', 'rebuild'), echo: per('echo', 'standard'), level: per('leveling', 'standard'),
    std: per('transcribe', 'local'), ext: per('transcribe', 'premium'), tr: per('transcribe_translate', 'standard'),
    cv: per('convert', 'standard'), // Video conversion on our side (a video this browser can't make), when the picture is made again
  };
}
/** Micros a minute, for the words: noise 9000 -> "$0.009 a minute". */
export const perMinute = (secRate) => Math.round(secRate * 60);

export const tot = (x) => Math.max(0, Math.ceil(x - EPS));

/** What the tracks need and cost: seconds per kind, micros per line, the total, and whether anything paid is on. */
export function cost(tracks, R) {
  const rate = R || { noise: 0, rebuild: 0, echo: 0, level: 0, std: 0, ext: 0, tr: 0 };
  const c = { nPaid: 0, nFree: 0, rSec: 0, eSec: 0, lSec: 0, sStd: 0, sExt: 0, sFree: 0, trSec: 0, trLangs: [] };
  // billed per second of audio: a started second counts (only the last section of a recording can end inside a second)
  const secs = (x) => Math.ceil(x.e - x.s - EPS);
  tracks.voice.forEach((x) => { const d = secs(x); if (x.mode === 1) { if (x.kind === 'free') c.nFree += d; else c.nPaid += d; } else if (x.mode === 2) c.rSec += d; });
  tracks.echo.forEach((x) => { if (x.on) c.eSec += secs(x); });
  tracks.level.forEach((x) => { if (x.lvl) c.lSec += secs(x); });
  tracks.subs.forEach((x) => {
    const d = secs(x);
    const t = trOf(x);
    if (x.q === 1) c.sFree += d; else if (x.q === 2) c.sStd += d; else if (x.q === 3) c.sExt += d;
    c.trSec += d * t.length;
    t.forEach((l) => { if (!c.trLangs.includes(l)) c.trLangs.push(l); });
  });
  c.noise = c.nPaid * rate.noise; c.rebuild = c.rSec * rate.rebuild; c.echo = c.eSec * rate.echo; c.level = c.lSec * rate.level; c.tr = c.trSec * rate.tr;
  c.subs = c.sStd * rate.std + c.sExt * rate.ext + c.tr;
  c.sum = c.noise + c.rebuild + c.echo + c.level + c.subs;
  c.paidAny = c.nPaid > 0 || c.rSec > 0 || c.eSec > 0 || c.lSec > 0 || c.sStd > 0 || c.sExt > 0; // (a price list that says $0 doesn't make a paid part free)
  c.total = tot(c.sum);
  return c;
}

/** The segments of the file where the tracks of `a` and `b` are constant: [[start, end], ...]. */
function segments(T, ...sets) {
  const edges = new Set([0, T]);
  sets.forEach((tracks) => Object.values(tracks).forEach((L) => L.forEach((x) => { edges.add(x.s); edges.add(x.e); })));
  const xs = [...edges].filter((t) => t >= 0 && t <= T).sort((p, q) => p - q);
  const out = [];
  for (let i = 0; i + 1 < xs.length; i++) if (xs[i + 1] - xs[i] > EPS) out.push([xs[i], xs[i + 1]]);
  return out;
}

/** What saving the changes of a finished recording costs: only what is new (a part that wasn't paid for before), micros rounded up. */
export function delta(now, old, T, R) {
  const rate = R || { noise: 0, rebuild: 0, echo: 0, level: 0, std: 0, ext: 0, tr: 0 };
  const sr = [0, 0, rate.std, rate.ext];
  let x = 0;
  for (const [a, b] of segments(T, now, old)) {
    const m = (a + b) / 2;
    const d = Math.ceil(b - a - EPS);
    const vn = at(now, 'voice', m);
    const vo = at(old, 'voice', m);
    if (vn.mode === 1 && vn.kind === 'paid' && !(vo.mode === 1 && vo.kind === 'paid')) x += rate.noise * d;
    if (vn.mode === 2 && vo.mode !== 2) x += rate.rebuild * d;
    if (at(now, 'echo', m).on && !at(old, 'echo', m).on) x += rate.echo * d;
    if (at(now, 'level', m).lvl && !at(old, 'level', m).lvl) x += rate.level * d;
    const sn = at(now, 'subs', m);
    const so = at(old, 'subs', m);
    x += Math.max(0, sr[sn.q] - sr[so.q]) * d;
    x += trOf(sn).filter((l) => !trOf(so).includes(l)).length * rate.tr * d;
  }
  return tot(x);
}

/** A signature of what the tracks do, second by second: equal when two sets of tracks do the same. */
export function valSig(tracks, T) {
  return JSON.stringify(TRACKS.map(([tr]) => {
    const parts = [];
    for (const [a, b] of segments(T, tracks)) {
      const k = vkey(tr, at(tracks, tr, (a + b) / 2));
      const last = parts[parts.length - 1];
      if (last && last[2] === k) last[1] = b; else parts.push([a, b, k]);
    }
    return parts;
  }));
}
export const sameTracks = (a, b, T) => valSig(a, T) === valSig(b, T);
export const sameOut = (a, b) => same(a, b);

/* ------------------------------------------------------------------------------------------------ what needs the server */

/** Does any part need our side (paid noise removal, Voice rebuild, echo, Leveling, Standard or Extra accurate subtitles)? */
export const needsServer = (tracks) => cost(tracks, null).paidAny;
/** Does a paid part change the sound (so a cleaned audio file comes from our side)? */
export function serverSound(tracks) {
  return tracks.voice.some((x) => x.mode === 2 || (x.mode === 1 && x.kind === 'paid')) || tracks.echo.some((x) => x.on) || tracks.level.some((x) => x.lvl);
}
/** Is the sound changed at all (by us or in the browser)? Then a cleaned file is part of the result. */
export function changesSound(tracks) {
  return tracks.voice.some((x) => x.mode > 0) || tracks.echo.some((x) => x.on) || tracks.level.some((x) => x.lvl);
}
/** Does any track do anything at all (so there is something to start)? Whatever runs in the browser or on our side counts alike. */
export const hasWork = (tracks) => changesSound(tracks) || tracks.subs.some((x) => x.q > 0);
export const anyFreeNoise = (tracks) => tracks.voice.some((x) => x.mode === 1 && x.kind === 'free');
export const anySubs = (tracks) => tracks.subs.some((x) => x.q > 0);
export const anyFreeSubs = (tracks) => tracks.subs.some((x) => x.q === 1);
export const anyPaidSubs = (tracks) => tracks.subs.some((x) => x.q >= 2);

/** Ranges (merged) where the Free noise removal is chosen, with the strength of each: [{start, end, db}]. Neighbours with the same strength join. */
export function freeNoiseRanges(tracks) {
  const out = [];
  for (const x of tracks.voice) {
    if (x.mode !== 1 || x.kind !== 'free') continue;
    const db = dbOf(x);
    const last = out[out.length - 1];
    if (last && Math.abs(last.end - x.s) < EPS && last.db === db) last.end = x.e; else out.push({ start: x.s, end: x.e, db });
  }
  return out;
}
/** Ranges where Free subtitles are chosen, with their language ('' = detect): [{start, end, language}]. */
export function freeSubRanges(tracks) {
  const out = [];
  for (const x of tracks.subs) {
    if (x.q !== 1) continue;
    const language = x.lang === 'auto' ? '' : x.lang;
    const last = out[out.length - 1];
    if (last && Math.abs(last.end - x.s) < EPS && last.language === language) last.end = x.e; else out.push({ start: x.s, end: x.e, language });
  }
  return out;
}

/* ------------------------------------------------------------------------------------------------ the server's timeline */

const lufsId = (x) => LUFS[x.lvl];

/**
 * The timeline the server takes (docs/api.md section 8) from the tracks: Standard noise removal -> `noise` {strength_db, remover},
 * Free -> `noise` {browser: true}, Voice rebuild -> `rebuild`, Echo -> `echo`, Leveling -> `leveling` (Keep: loudness null), Subtitles ->
 * `subtitles`. `seconds`: the recording's length (the last part ends there).
 */
export function toTimeline(tracks, seconds) {
  const tl = new Timeline(seconds);
  const end = (x) => seconds && x.e >= seconds - 0.0005 ? seconds : x.e;
  for (const x of tracks.voice) {
    if (x.mode === 1 && x.kind === 'free') tl.paint('noise', x.s, end(x), { browser: true, strength_db: dbOf(x) });
    else if (x.mode === 1) tl.paint('noise', x.s, end(x), { strength_db: dbOf(x), remover: REMOVERS[x.rem || 0] });
    else if (x.mode === 2) tl.paint('rebuild', x.s, end(x), {});
  }
  for (const x of tracks.echo) if (x.on) tl.paint('echo', x.s, end(x), {});
  for (const x of tracks.level) {
    if (!x.lvl) continue;
    tl.paint('leveling', x.s, end(x), { loudness_lufs: x.lvl === 9 ? x.lufs : lufsId(x) });
  }
  for (const x of tracks.subs) {
    if (!x.q) continue;
    tl.paint('subtitles', x.s, end(x), { quality: QUALITY[x.q], language: x.lang === 'auto' ? null : x.lang, translate: trOf(x) });
  }
  return tl;
}

/** The loudness choice (1 to 4, or 9 for a value of its own) of a loudness in LUFS. */
export function lvlOfLufs(v) {
  if (v === null || v === undefined) return 4;
  const i = LUFS.indexOf(v);
  return i > 0 && i < 4 ? i : 9;
}

/**
 * The tracks of a timeline (a job that is loaded: what it did, or what it was last saved with). `seconds`: the recording's length.
 * `subs`: where the timeline has no subtitles layer, the job's own whole-file choice {quality, language, translate} (a job made without a timeline).
 */
export function fromTimeline(tl, seconds, { subs = null } = {}) {
  const T = seconds;
  const edges = new Set([0, T]);
  Object.values(tl.layers).forEach((L) => L.forEach((p) => { edges.add(Math.min(p.start, T)); edges.add(Math.min(p.end, T)); }));
  const xs = [...edges].sort((a, b) => a - b);
  const tracks = { voice: [], echo: [], level: [], subs: [] };
  const push = (tr, a, b, o) => {
    const L = tracks[tr];
    const x = sec(a, b, { ...o, auto: false });
    const last = L[L.length - 1];
    if (last && vkey(tr, last) === vkey(tr, x)) last.e = b; else L.push(x);
  };
  for (let i = 0; i + 1 < xs.length; i++) {
    const a = xs[i];
    const b = xs[i + 1];
    if (b - a < EPS) continue;
    const m = (a + b) / 2;
    const rb = tl.partAt('rebuild', m);
    const nz = tl.partAt('noise', m);
    const base = { str: 2, kind: 'paid', rem: 0, db: null };
    if (rb) push('voice', a, b, { ...base, mode: 2 });
    else if (nz && nz.settings.browser) {
      const db = nz.settings.strength_db;
      const lvl = db ? Math.min(3, levelOf(db) || 3) : 3;
      push('voice', a, b, { ...base, mode: 1, kind: 'free', str: lvl, db: db && Math.abs(STR_DB[lvl] - db) > 0.01 ? Math.min(db, 40) : null });
    }
    else if (nz) {
      const db = nz.settings.strength_db;
      const lvl = levelOf(db) || 1;
      push('voice', a, b, { ...base, mode: 1, str: lvl, db: Math.abs(STR_DB[lvl] - db) < 0.01 ? null : db, rem: Math.max(0, REMOVERS.indexOf(nz.settings.remover)) });
    } else push('voice', a, b, { ...base, mode: 0 });
    push('echo', a, b, { on: tl.partAt('echo', m) ? 1 : 0 });
    const lv = tl.partAt('leveling', m);
    push('level', a, b, lv ? { lvl: lvlOfLufs(lv.settings.loudness_lufs), lufs: lvlOfLufs(lv.settings.loudness_lufs) === 9 ? lv.settings.loudness_lufs : null } : { lvl: 0, lufs: null });
    const sb = tl.partAt('subtitles', m);
    if (sb) push('subs', a, b, { q: QUALITY.indexOf(sb.settings.quality), lang: sb.settings.language || 'auto', tr: [...sb.settings.translate] });
    else if (subs) push('subs', a, b, { q: QUALITY.indexOf(subs.quality), lang: subs.language || 'auto', tr: [...(subs.translate || [])] });
    else push('subs', a, b, { q: 0, lang: 'auto', tr: [] });
  }
  return tracks;
}

/* ------------------------------------------------------------------------------------------------ Automatic */

/**
 * What "Automatic best settings" sets for a recording (recommend.js): one patch per track, for the whole file.
 * `rebuildOk`: Voice rebuild is on offer. `canBrowser`: this browser could do the Free noise removal (never chosen automatically).
 */
export function recoPatches(r, { rebuildOk = true, freeOnly = false } = {}) {
  const str = Math.max(1, STRENGTHS.findIndex((s) => s.id === r.strength) + 1);
  // Without an account only the free options can be chosen: the same judgement, kept to them (a rebuild or the heavy remover
  // becomes the strongest Free strength, Leveling stays off, any subtitles are the Free ones)
  if (freeOnly) {
    return {
      voice: { mode: 1, kind: 'free', str: r.rebuild || r.heavy ? 3 : Math.min(str, 3), rem: 0, db: null },
      level: { lvl: 0, lufs: null },
      subs: { q: r.subs ? 1 : 0 },
    };
  }
  return {
    voice: r.rebuild && rebuildOk ? { mode: 2 } : { mode: 1, kind: 'paid', str, rem: 0, db: null },
    level: r.leveling ? { lvl: 2, lufs: null } : { lvl: 0, lufs: null },
    subs: r.subs === 'accurate' ? { q: 3 } : r.subs === 'standard' ? { q: 2 } : { q: 0 },
  };
}
/** A suggestion's change kept to the free options (without an account): Voice rebuild or a Standard strength becomes the nearest
 *  Free strength (Free has no Max). */
export function freeOnlySuggestion(g) {
  if (g.patch.mode === 2) return { ...g, todo: STR[3], patch: { mode: 1, str: 3, kind: 'free', rem: 0, db: null } };
  if (g.patch.mode === 1) { const str = Math.min(g.patch.str || 2, 3); return { ...g, todo: STR[str], patch: { ...g.patch, kind: 'free', str, rem: 0, db: null } }; }
  return g;
}
/** Set the automatic choices on the sections that are still automatic (or on every section: `all`). They stay marked automatic. */
export function applyReco(S, patches, { all = false } = {}) {
  for (const tr of ['voice', 'level', 'subs']) {
    if (!patches[tr]) continue;
    S.tracks[tr].forEach((x) => { if (all || x.auto) Object.assign(x, patches[tr], { auto: true }); });
    normalize(S, tr);
  }
}

/* ------------------------------------------------------------------------------------------------ suggestions */

/** The prototype's suggestion from one of the server's / the browser's (editor/suggest.js): whole seconds, a plain "why", what to do. */
export function toSuggestion(s, T, id) {
  const a = clamp(Math.floor(s.start), 0, Math.max(0, T - 1));
  const b = clamp(Math.ceil(s.end), a + 1, T);
  const reasons = (s.reasons || []).filter(Boolean);
  const why = reasons.length ? reasons.join(', ').replace(/^./, (c) => c.toUpperCase()) : 'Sounds different from the rest';
  if (s.rebuild) return { id, key: s.key, s: a, e: b, why, todo: 'Voice rebuild', patch: { mode: 2 }, st: 'new' };
  if (!s.noise) return { id, key: s.key, s: a, e: b, why, todo: 'Noise removal', patch: { mode: 1 }, st: 'new' };
  const str = Math.max(1, levelOf(s.noise.strength_db));
  return {
    id, key: s.key, s: a, e: b, why, todo: STR[str],
    patch: { mode: 1, str, kind: 'paid', rem: Math.max(0, REMOVERS.indexOf(s.noise.remover || 'auto')), db: null }, st: 'new',
  };
}
