import { useCallback, useMemo, useState } from 'react';
import { DEFAULT_LOUDNESS, NOISE_LEVELS, Timeline, changedIn, levelOf, replay, restoreOps } from '../../editor/timeline.js';

const MAX_OPS = 300;

/**
 * The editor's state: the timeline is `base` plus the person's changes as a list of operations (editor/timeline.js replay), so any of
 * them can be taken back and the base can change underneath it (the upload form's choices). `sel` is the part being worked on.
 */
export function useEditorModel({ seconds, base, secondTrack }) {
  const [ops, setOps] = useState([]);
  const [sel, setSelRaw] = useState(null); // {s, e} | null
  const [remover, setRemover] = useState('auto');

  const tl = useMemo(() => {
    try { return replay(base, ops); } catch { return base; }
  }, [base, ops]);
  const dirty = useMemo(() => !tl.equals(base), [tl, base]);

  const loudnessNow = tl.parts('leveling')[0]?.settings.loudness_lufs;
  const loudness = loudnessNow === undefined ? DEFAULT_LOUDNESS : loudnessNow;

  const push = useCallback((more) => setOps((o) => [...o, ...more].slice(-MAX_OPS)), []);

  const setSel = useCallback((s, e) => {
    if (s == null) { setSelRaw(null); return; }
    let a = Math.max(0, Math.min(seconds, Math.min(s, e)));
    let b = Math.max(0, Math.min(seconds, Math.max(s, e)));
    if (b - a < 1) { if (b >= seconds) a = Math.max(0, b - 1); else b = Math.min(seconds, a + 1); }
    setSelRaw({ s: Math.round(a * 10) / 10, e: Math.round(b * 10) / 10 });
  }, [seconds]);

  const act = useMemo(() => {
    const ranged = (fn) => () => { if (sel) push(fn(sel.s, sel.e)); };
    const here = (layer) => {
      const u = sel ? tl.uniform(layer, sel.s, sel.e) : false;
      return u && u.remover ? u.remover : remover;
    };
    return {
      noiseLevel: (i) => {
        if (!sel) return;
        const { s, e } = sel;
        if (i === 0) { push([{ layer: 'noise', start: s, end: e, settings: null }]); return; }
        push([{ layer: 'noise', start: s, end: e, settings: { strength_db: NOISE_LEVELS[i].db, remover: here('noise') } }]);
      },
      remover: (r) => {
        setRemover(r);
        if (!sel) return;
        const { s, e } = sel;
        const ops2 = [];
        for (const p of tl.parts('noise')) {
          const a = Math.max(p.start, s);
          const b = Math.min(p.end, e);
          if (b > a) ops2.push({ layer: 'noise', start: a, end: b, settings: { ...p.settings, remover: r } });
        }
        if (ops2.length) push(ops2);
      },
      rebuild: (on) => { if (sel) push([{ layer: 'rebuild', start: sel.s, end: sel.e, settings: on ? {} : null }]); },
      echo: (on) => { if (sel) push([{ layer: 'echo', start: sel.s, end: sel.e, settings: on ? {} : null }]); },
      leveling: (on) => { if (sel) push([{ layer: 'leveling', start: sel.s, end: sel.e, settings: on ? { loudness_lufs: loudness } : null }]); },
      loudness: (lufs) => {
        const more = tl.parts('leveling').map((p) => ({ layer: 'leveling', start: p.start, end: p.end, settings: { loudness_lufs: lufs } }));
        if (more.length) push(more);
      },
      restore: ranged((s, e) => restoreOps(base, s, e)),
      noiseIndex: () => {
        const u = sel ? tl.uniform('noise', sel.s, sel.e) : false;
        return u === null ? -1 : u ? levelOf(u.strength_db) : 0;
      },
    };
  }, [sel, tl, remover, loudness, base, push]);

  const paintWhole = useCallback((layer, settings) => push([{ layer, start: 0, end: seconds, settings }]), [push, seconds]);
  const apply = useCallback((more) => push(more), [push]);
  const undoTag = useCallback((tag) => setOps((o) => o.filter((x) => x.tag !== tag)), []);
  const undoLast = useCallback(() => setOps((o) => o.slice(0, -1)), []);
  const reset = useCallback(() => setOps([]), []);
  const changedHere = sel ? changedIn(tl, base, sel.s, sel.e) : false;

  return { tl, ops, dirty, sel, setSel, act, apply, undoTag, undoLast, reset, loudness, paintWhole, changedHere, remover, secondTrack };
}

export { Timeline };
