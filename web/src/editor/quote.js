/*
 * What saving an edit of a finished job costs, worked out here at once: the same planning and pricing as our side's quote
 * (server/acw/editing.py make_quote over common/acw_common/editor.py plan / billable and edit_subtitles.subtitle_plan), from what the
 * job's timeline answer carries once (`quote`: editing.quote_inputs). The cache is the sha1 names of the job's blocks
 * (blocks.FlacCache.name), so a block is "there" when the sha1 of its key is in that list. Kept the same as the Python by
 * web/tests (quote agreement).
 */
import { Timeline } from './timeline.js';

const SR = 48000;
const STEPS = ['echo', 'light', 'heavy', 'rebuild', 'leveling'];
const RAMP_FLOOR_S = 0.1;
const MARGIN_A = 2 * Math.trunc(RAMP_FLOOR_S * SR);
const MARGIN_B = Math.trunc((0.3 + 0.25 / 2) * SR) + 2 * 960 + 4800;
const PAD = 2 * Math.trunc(RAMP_FLOOR_S * SR);
const LEVEL_CONTEXT_S = 2.0;
const LEVEL_MIN_SIDE_S = 10.0;
const GRIDS = { light: [30, 16, 0.25], heavy: [30, 24, 0.25], echo: [30, 10, 0.01] };
const MODEL_TAGS = { echo: 'aec3', light: 'dfn3', heavy: 'dpdfnet8', rebuild: 'sidon-v0.1' };
// Voice rebuild's blocks (restore.py): 16 s windows with 2 s of overlap at 50 feature frames a second, 960 samples a frame.
const RB_WIN = 800;
const RB_STEP = 700;
const RB_BLOCK = RB_STEP * 960;
const RB_TAG = 'w16o2';

// ---- sha1 (the cache's file names): small and synchronous; keys are short ASCII
function sha1(str) {
  const bytes = new TextEncoder().encode(str);
  const len = bytes.length;
  const words = new Uint32Array((((len + 8) >> 6) + 1) * 16);
  for (let i = 0; i < len; i++) words[i >> 2] |= bytes[i] << (24 - (i % 4) * 8);
  words[len >> 2] |= 0x80 << (24 - (len % 4) * 8);
  words[words.length - 1] = len * 8;
  let [h0, h1, h2, h3, h4] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0];
  const w = new Uint32Array(80);
  const rol = (x, n) => (x << n) | (x >>> (32 - n));
  for (let o = 0; o < words.length; o += 16) {
    for (let i = 0; i < 16; i++) w[i] = words[o + i];
    for (let i = 16; i < 80; i++) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    let [a, b, c, d, e] = [h0, h1, h2, h3, h4];
    for (let i = 0; i < 80; i++) {
      const f = i < 20 ? (b & c) | (~b & d) : i < 40 ? b ^ c ^ d : i < 60 ? (b & c) | (b & d) | (c & d) : b ^ c ^ d;
      const k = i < 20 ? 0x5a827999 : i < 40 ? 0x6ed9eba1 : i < 60 ? 0x8f1bbcdc : 0xca62c1d6;
      const t = (rol(a, 5) + f + e + k + w[i]) >>> 0;
      e = d; d = c; c = rol(b, 30) >>> 0; b = a; a = t;
    }
    h0 = (h0 + a) >>> 0; h1 = (h1 + b) >>> 0; h2 = (h2 + c) >>> 0; h3 = (h3 + d) >>> 0; h4 = (h4 + e) >>> 0;
  }
  return [h0, h1, h2, h3, h4].map((x) => x.toString(16).padStart(8, '0')).join('');
}

// ---- ranges (blocks.py)
const unionSeconds = (ranges) => {
  let total = 0;
  let end = -1;
  [...ranges].filter((r) => r[1] > r[0]).sort((p, q) => p[0] - q[0] || p[1] - q[1]).forEach(([a, b]) => {
    if (a >= end) { total += b - a; end = b; } else if (b > end) { total += b - end; end = b; }
  });
  return total / SR;
};
const intersect = (ranges, a, b) => ranges.filter(([x, y]) => Math.min(b, y) > Math.max(a, x)).map(([x, y]) => [Math.max(a, x), Math.min(b, y)]);
const ceilSecs = (s) => (s > 0 ? Math.ceil(s - 1e-9) : 0);

function gridWindows(a, b, step, ra = null, rb = null) {
  const [secs, before, after] = GRIDS[step];
  ra = ra == null ? a : Math.max(a, ra);
  rb = rb == null ? b : Math.min(b, rb);
  if (rb <= ra) return [];
  const size = Math.round(secs * SR);
  const pre = Math.round(before * SR);
  const post = Math.round(after * SR);
  const out = [];
  for (let k = Math.floor(ra / size); k <= Math.floor((rb - 1) / size); k++) {
    const s = Math.max(a, k * size);
    const e = Math.min(b, (k + 1) * size);
    if (e > s) out.push([Math.max(a, s - pre), s, e, Math.min(b, e + post)]);
  }
  return out;
}
const blockKey = (step, inputId, ...parts) => ['acwb1', step, MODEL_TAGS[step], inputId, ...parts.map(String)].join('|');

// ---- the timeline as the planner reads it
const settingsKey = (st) => JSON.stringify(Object.keys(st || {}).sort().map((k) => [k, st[k]]));
const partKey = (a, b) => `${a.toFixed(3)}-${b.toFixed(3)}`;
function samples(tl, layer, n) {
  const out = [];
  for (const p of tl.parts(layer)) {
    const a = Math.max(0, Math.min(n, Math.round(p.start * SR)));
    const b = Math.max(0, Math.min(n, Math.round(p.end * SR)));
    if (b > a) out.push([a, b, p.settings]);
  }
  return out;
}
/** editing.with_removers: the parts still on "auto" get what was decided for exactly that part; `provisional` for the rest. */
function withRemovers(tl, resolved, provisional = null) {
  const out = tl.copy();
  let left = false;
  for (const p of out.layers.noise) {
    if (p.settings.remover !== 'auto') continue;
    const r = (resolved || {})[partKey(p.start, p.end)];
    if (r === 'light' || r === 'heavy') p.settings.remover = r;
    else { left = true; if (provisional) p.settings.remover = provisional; }
  }
  return [out, left];
}
/** Timeline.audio(): what the sound depends on (no subtitles; a browser part without the strength the page keeps for it). */
function audioOf(tl) {
  const t = new Timeline(tl.seconds);
  for (const name of ['echo', 'noise', 'rebuild', 'leveling']) {
    for (const p of tl.parts(name)) t.paint(name, p.start, p.end, p.settings.browser ? { browser: true } : p.settings);
  }
  return JSON.stringify(t.toJSON().layers);
}

function fill(parts, n, on, off) {
  const segs = [];
  let pos = 0;
  for (const [a, b, st] of parts) {
    if (a > pos) segs.push({ a: pos, b: a, kind: off, st: {} });
    segs.push({ a, b, kind: on, st: { ...st } });
    pos = b;
  }
  if (pos < n || !segs.length) segs.push({ a: pos, b: n, kind: off, st: {} });
  return segs;
}
function mainSegments(noise, rebuild, n) {
  const pts = [...new Set([0, n, ...noise.flatMap(([a, b]) => [a, b]), ...rebuild.flatMap(([a, b]) => [a, b])])].sort((p, q) => p - q);
  const segs = [];
  for (let i = 0; i + 1 < pts.length; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    if (b <= a) continue;
    let kind;
    let st;
    if (rebuild.some(([x, y]) => x <= a && a < y)) { kind = 'rebuild'; st = {}; } else {
      const nz = noise.find(([x, y]) => x <= a && a < y);
      if (!nz) { kind = 'pass'; st = {}; } else if (nz[2].browser) { kind = 'browser'; st = { browser: true }; } else {
        if (nz[2].remover !== 'light' && nz[2].remover !== 'heavy') throw new Error('the noise remover of every part must be light or heavy');
        kind = 'noise'; st = { remover: nz[2].remover, strength_db: nz[2].strength_db };
      }
    }
    const last = segs[segs.length - 1];
    if (last && last.kind === kind && settingsKey(last.st) === settingsKey(st)) last.b = b;
    else segs.push({ a, b, kind, st });
  }
  return segs;
}
function layout(tl, n, reference) {
  const echo = samples(tl, 'echo', n);
  if (echo.length && !reference) throw new Error('echo removal needs the file’s second audio track');
  const level = [];
  for (const [a, b, s] of samples(tl, 'leveling', n)) {
    const last = level[level.length - 1];
    if (last && last[1] === a && settingsKey(last[2]) === settingsKey(s)) last[1] = b; else level.push([a, b, s]);
  }
  const lay = {
    n, echoParts: echo.map(([a, b]) => [a, b]), echo: fill(echo, n, 'echo', 'off'),
    main: mainSegments(samples(tl, 'noise', n), samples(tl, 'rebuild', n), n), levelParts: level,
  };
  lay.asked = { echo: lay.echoParts.slice(), light: [], heavy: [], rebuild: [], leveling: level.map(([a, b]) => [a, b]) };
  for (const s of lay.main) {
    if (s.kind === 'noise') lay.asked[s.st.remover].push([s.a, s.b]);
    else if (s.kind === 'rebuild') lay.asked.rebuild.push([s.a, s.b]);
  }
  return lay;
}
function levelSignature(lay, a, b) {
  const e = Math.max(b - a, Math.trunc(LEVEL_MIN_SIDE_S * SR));
  const ma = Math.max(0, Math.max(0, a - e) - Math.trunc(LEVEL_CONTEXT_S * SR));
  const mb = Math.min(lay.n, Math.min(lay.n, b + e) + Math.trunc(LEVEL_CONTEXT_S * SR));
  const cut = (segs) => segs.filter(([x, y]) => y > ma && x < mb).map(([x, y, st]) => [Math.max(x, ma), Math.min(y, mb), st]);
  return JSON.stringify([
    ['echo', cut(lay.echo.filter((s) => s.kind === 'echo').map((s) => [s.a, s.b, ''])) ],
    ['main', cut(lay.main.filter((s) => s.kind !== 'pass').map((s) => [s.a, s.b, `${s.kind}${settingsKey(s.st)}`]))],
  ]);
}

/** editor.plan + billable: the blocks the timeline needs, which of them the cache lacks, and the billable seconds per step. */
function plan(tl, q, previous) {
  const f = q.facts;
  const n = f.n;
  const lay = layout(tl, n, f.reference);
  const refId = f.reference ? 'r' : null;
  const ident = (lo, hi) => {
    const ps = lay.echoParts.filter(([a, b]) => b > lo - PAD && a < hi + PAD).map(([a, b]) => [Math.max(a, lo - PAD), Math.min(b, hi + PAD)]);
    return ps.length ? `${q.sourceId}~${refId}~e${ps.map(([a, b]) => `${a}-${b}`).join(',')}` : q.sourceId;
  };
  const todo = [...new Set(f.plan.filter((c) => c !== null))].sort((p, r) => p - r);
  const spans = (ch) => (f.spans[String(ch)] || []).map(([a, b]) => [a, b]);
  const jobs = f.rebuildMid ? ['mid'] : [...Array(f.channels).keys()];
  const extend = (a, b, ma, mb) => [Math.max(0, a - ma), Math.min(n, b + mb)];
  const tasks = new Map();
  const add = (step, key, s, e, asked) => {
    const old = tasks.get(key);
    if (!old) tasks.set(key, { step, key, s, e, asked });
    else old.asked = [...old.asked, ...asked];
  };
  for (let ch = 0; ch < (lay.echoParts.length ? f.channels : 0); ch++) {
    const seen = new Set();
    for (const [a, b] of lay.echoParts) {
      const [ra, rb] = extend(a, b, MARGIN_A, MARGIN_A);
      for (const [lo, s, e, hi] of gridWindows(0, n, 'echo', ra, rb)) {
        if (seen.has(`${s},${e}`)) continue;
        seen.add(`${s},${e}`);
        add('echo', blockKey('echo', `${q.sourceId}~${refId}`, ch, lo, s, e, hi), s, e, intersect(lay.asked.echo, s, e));
      }
    }
  }
  lay.main.forEach((seg, i) => {
    if (seg.kind !== 'noise' && seg.kind !== 'rebuild') return;
    const prev = i > 0 ? lay.main[i - 1].kind : null;
    const next = i + 1 < lay.main.length ? lay.main[i + 1].kind : null;
    const ma = (i > 0 && prev === 'rebuild') || seg.kind === 'rebuild' ? MARGIN_B : (i > 0 ? MARGIN_A : 0);
    const mb = (next !== null && next === 'rebuild') || seg.kind === 'rebuild' ? MARGIN_B : (next !== null ? MARGIN_A : 0);
    const [ra, rb] = extend(seg.a, seg.b, ma, mb);
    if (seg.kind === 'noise') {
      const step = seg.st.remover;
      for (const ch of todo) {
        for (const [sa, sb] of spans(ch)) {
          for (const [lo, s, e, hi] of gridWindows(sa, sb, step, ra, rb)) add(step, blockKey(step, ident(lo, hi), ch, lo, s, e, hi), s, e, intersect(lay.asked[step], s, e));
        }
      }
    } else {
      for (const job of jobs) {
        for (let b = Math.floor(ra / RB_BLOCK); b <= Math.floor((rb - 1) / RB_BLOCK); b++) {
          const s = b * RB_BLOCK;
          const e = Math.min(n, (b + 1) * RB_BLOCK);
          const lo = Math.max(0, (b > 0 ? (b - 1) * RB_STEP : 0) * 960 - 2 * SR);
          const hi = Math.min(n, ((b + 1) * RB_STEP + RB_WIN) * 960 + 2 * SR);
          add('rebuild', blockKey('rebuild', ident(lo, hi), job, RB_TAG, b, n), s, e, intersect(lay.asked.rebuild, s, e));
        }
      }
    }
  });
  const cached = q.cachedSet || (q.cachedSet = new Set(q.cached));
  const misses = [...tasks.values()].filter((t) => !cached.has(sha1(t.key)));
  let level = lay.levelParts.map(([a, b]) => [a, b]);
  if (previous && n) {
    let old = null;
    try { old = layout(previous, n, f.reference); } catch { old = null; }
    if (old) {
      const same = new Set(old.levelParts.filter(([a, b]) => levelSignature(old, a, b) === levelSignature(lay, a, b)).map(([a, b, s]) => `${a},${b},${settingsKey(s)}`));
      level = lay.levelParts.filter(([a, b, s]) => !same.has(`${a},${b},${settingsKey(s)}`)).map(([a, b]) => [a, b]);
    }
  }
  const billable = {};
  for (const step of STEPS) {
    const secs = step === 'leveling'
      ? ceilSecs(level.reduce((x, [a, b]) => x + (b - a), 0) / SR)
      : ceilSecs(unionSeconds(misses.filter((t) => t.step === step).flatMap((t) => t.asked)));
    if (secs > 0) billable[step] = secs;
  }
  const rebuildSeconds = unionSeconds(lay.asked.rebuild);
  return { billable, misses: misses.length, rebuildSeconds };
}

const atRate = (rateMin, secs, f = 0) => Math.ceil((rateMin * (secs + f)) / 60 - 1e-9);
const STEP_LINE = { echo: ['echo', 'Echo removal'], light: ['denoise', 'Noise removal'], heavy: ['denoise', 'Noise removal'], rebuild: ['rebuild', 'Voice rebuild'], leveling: ['leveling', 'Leveling'] };

/**
 * The quote of saving `tl` on the job (editing.make_quote): {micros, lines: [{item, label, seconds, fixedSeconds, micros}], free, unchanged,
 * approximate, rebuildSeconds}. `q`: the timeline answer's `quote`; `rendered`: the timeline the result files were made from (Timeline);
 * `stepFixed`: the price list's stepFixedSeconds. Throws an Error with the reason for a timeline this recording can't have.
 */
export function quoteLocal(tl, q, rendered, stepFixed) {
  const [planned, approximate] = withRemovers(tl, q.resolved, 'light');
  const previous = rendered ? withRemovers(rendered, q.resolved, 'light')[0] : null;
  const p = plan(planned, q, previous);
  const audioSame = !!rendered && audioOf(withRemovers(tl, q.resolved)[0]) === audioOf(rendered);
  const lines = [];
  let micros = 0;
  if (!audioSame) {
    const byItem = {};
    for (const step of STEPS) {
      const secs = p.billable[step];
      if (!secs) continue;
      const fixed = stepFixed?.[step] || 0;
      const m = atRate(q.rates[step], secs, fixed);
      const [item, label] = STEP_LINE[step];
      const line = byItem[item] || (byItem[item] = { item, label, seconds: 0, fixedSeconds: 0, micros: 0 });
      line.seconds += secs; line.fixedSeconds += fixed; line.micros += m;
      micros += m;
    }
    lines.push(...Object.values(byItem));
  }
  // the subtitle parts (edit_subtitles.subtitle_plan with the job's folder): what is made already is not made or charged again
  const sub = { standard: [0, 0], accurate: [0, 0] };
  let tr = [0, 0];
  const langs = new Set();
  let work = 0;
  for (const part of tl.parts('subtitles')) {
    const st = part.settings;
    if (st.quality === 'free') continue;
    const have = q.subtitles[`${Math.round(part.start * 1000)}-${Math.round(part.end * 1000)}-${st.quality[0]}-${st.language || 'auto'}`];
    const secs = Math.max(1, Math.ceil(part.end - part.start - 1e-9));
    const transcribe = !have?.done;
    const toTranslate = (st.translate || []).filter((t) => t !== have?.language && !(have?.translations || []).includes(t));
    if (transcribe) {
      const m = atRate(q.subtitleRates[`transcribe_${st.quality}`], secs);
      sub[st.quality][0] += secs; sub[st.quality][1] += m; micros += m;
    }
    for (const t of toTranslate) {
      const m = atRate(q.subtitleRates.translate, secs);
      tr = [tr[0] + secs, tr[1] + m]; micros += m; langs.add(t);
    }
    if (transcribe || toTranslate.length) work += 1;
  }
  if (sub.standard[0]) lines.push({ item: 'transcribe', label: 'Subtitles, Standard', seconds: sub.standard[0], fixedSeconds: 0, micros: sub.standard[1] });
  if (sub.accurate[0]) lines.push({ item: 'transcribe_accurate', label: 'Subtitles, Extra accurate', seconds: sub.accurate[0], fixedSeconds: 0, micros: sub.accurate[1] });
  if (tr[0]) lines.push({ item: 'translate', label: 'Translations', seconds: tr[0], fixedSeconds: 0, micros: tr[1], languages: [...langs] });
  const subsJSON = (t) => JSON.stringify(t ? t.toJSON().layers.subtitles || [] : []);
  const subsSame = !!rendered && subsJSON(tl) === subsJSON(rendered) && !work;
  // a video made on our side is made again with a new sound, or new subtitles that go inside it
  if (q.video.makes && (!audioSame || (!subsSame && q.video.embedded)) && q.video.convertMicros) {
    micros += q.video.convertMicros;
    lines.push({ item: 'convert', label: 'Video conversion', seconds: q.video.convertLine?.seconds ?? null, fixedSeconds: 0, micros: q.video.convertMicros });
  }
  return { micros, lines, free: micros === 0, unchanged: audioSame && subsSame, approximate, rebuildSeconds: p.rebuildSeconds };
}

export { sha1 };
