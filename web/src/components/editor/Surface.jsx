import { useCallback, useEffect, useRef } from 'react';
import { clock, rangeText } from '../../editor/timeline.js';

/**
 * Choosing a part on a time surface (the scrub bar, a layer row): drag with the mouse, drag sideways with a finger, drag the handles
 * (any pointer), tap or click to move the playhead. Times are kept to a tenth of a second and a part is at least a second long.
 */
export function useSurface({ seconds, getSel, onSelect, onSeek }) {
  const ref = useRef(null);
  const drag = useRef(null);
  const live = useRef({});
  live.current = { seconds, getSel, onSelect, onSeek };

  const at = useCallback((ev) => {
    const r = ref.current.getBoundingClientRect();
    const t = ((ev.clientX - r.left) / Math.max(1, r.width)) * live.current.seconds;
    return Math.max(0, Math.min(live.current.seconds, Math.round(t * 10) / 10));
  }, []);

  const set = (a, b) => {
    const { seconds: total, onSelect: pick } = live.current;
    let s = Math.min(a, b);
    let e = Math.max(a, b);
    if (e - s < 1) { if (e >= total) s = Math.max(0, e - 1); else e = Math.min(total, s + 1); }
    pick(s, e);
  };

  const onPointerDown = (ev) => {
    if (ev.button > 0 || ev.target.closest('.ed-pin')) return;
    const hd = ev.target.closest('[data-edge]');
    const t = at(ev);
    if (hd) drag.current = { kind: 'edge', edge: hd.dataset.edge };
    else if (ev.pointerType === 'mouse' || ev.pointerType === 'pen') drag.current = { kind: 'new', t0: t, x0: ev.clientX, moved: false, mouse: true };
    else drag.current = { kind: 'new', t0: t, x0: ev.clientX, moved: false, mouse: false, pending: true };
    if (!drag.current.pending) { ref.current.setPointerCapture?.(ev.pointerId); ev.preventDefault(); }
  };
  const onPointerMove = (ev) => {
    const d = drag.current;
    if (!d) return;
    const t = at(ev);
    if (d.kind === 'edge') {
      const sel = live.current.getSel();
      if (!sel) return;
      if (d.edge === 's') set(Math.min(t, sel.e - 1), sel.e);
      else set(sel.s, Math.max(t, sel.s + 1));
    } else if (d.pending) {
      if (Math.abs(ev.clientX - d.x0) > 10) { d.pending = false; ref.current.setPointerCapture?.(ev.pointerId); d.moved = true; set(d.t0, t); }
    } else if (Math.abs(ev.clientX - d.x0) > 3 || Math.abs(t - d.t0) >= 1) {
      d.moved = true;
      set(d.t0, t);
    }
  };
  const end = (ev) => {
    const d = drag.current;
    drag.current = null;
    if (d && d.kind === 'new' && !d.moved && ev.type === 'pointerup') live.current.onSeek(d.t0);
  };
  return { ref, handlers: { onPointerDown, onPointerMove, onPointerUp: end, onPointerCancel: end } };
}

/** The selection (with its handles and label) and the playhead over a time surface. */
export function Overlay({ seconds, sel, playhead, handles = false, label = false }) {
  const pct = (t) => `${(t / seconds) * 100}%`;
  return (
    <>
      {sel && (
        <div className="ed-sel" style={{ left: pct(sel.s), width: `${((sel.e - sel.s) / seconds) * 100}%` }} data-testid="sel-box">
          {handles && <i className="ed-hd l" data-edge="s" data-testid="handle-start" role="slider" aria-label="Start of the part" aria-valuetext={clock(sel.s)} />}
          {handles && <i className="ed-hd r" data-edge="e" data-testid="handle-end" role="slider" aria-label="End of the part" aria-valuetext={clock(sel.e)} />}
          {label && <span className="ed-lab">{rangeText(sel.s, sel.e)}</span>}
        </div>
      )}
      <div className="ed-ph" style={{ left: pct(playhead) }} data-testid="playhead" />
    </>
  );
}

/** The time marks over a bar: 0:00 ... end, with fewer labels when there is less room. */
export function Ruler({ seconds, pins = [], onPin, focus }) {
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  const step = steps.find((s) => seconds / s <= 8) || 3600;
  const ticks = [];
  for (let t = 0; t <= seconds + 1e-6; t += step) ticks.push(t);
  return (
    <div className="ed-ruler" aria-hidden={pins.length === 0 ? 'true' : undefined}>
      {ticks.map((t, i) => (
        <span key={t} className={`tl${i === 0 ? ' first' : ''}`} style={{ left: `${(t / seconds) * 100}%` }}>{clock(t)}</span>
      ))}
      {pins.map((p) => (
        <button
          key={p.key} type="button" className="ed-pin" data-on={focus === p.key} data-testid="pin" data-key={p.key}
          style={{ left: `${(((p.start + p.end) / 2) / seconds) * 100}%` }} aria-label={`Suggestion at ${rangeText(p.start, p.end)}`}
          onClick={() => onPin(p)}
        >
          <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2l2.2 7.8L22 12l-7.8 2.2L12 22l-2.2-7.8L2 12l7.8-2.2z" /></svg>
        </button>
      ))}
    </div>
  );
}

/** The waveform: one bar per pixel column from `peaks` (a peak per quarter second; NaN where it is not known yet). */
export function Wave({ seconds, peaks, version }) {
  const cv = useRef(null);
  useEffect(() => {
    const el = cv.current;
    if (!el) return undefined;
    const draw = () => {
      const w = Math.max(1, Math.round(el.clientWidth * (window.devicePixelRatio || 1)));
      const h = Math.max(1, Math.round(el.clientHeight * (window.devicePixelRatio || 1)));
      if (el.width !== w) el.width = w;
      if (el.height !== h) el.height = h;
      const g = el.getContext('2d');
      g.clearRect(0, 0, w, h);
      const css = getComputedStyle(el);
      g.fillStyle = css.color;
      const per = 4;
      let known = false;
      for (let x = 0; x < w; x++) {
        const a = Math.floor(((x / w) * seconds) * per);
        const b = Math.max(a + 1, Math.ceil((((x + 1) / w) * seconds) * per));
        let m = NaN;
        for (let i = a; i < b && peaks && i < peaks.length; i++) if (!Number.isNaN(peaks[i])) m = Number.isNaN(m) ? peaks[i] : Math.max(m, peaks[i]);
        if (Number.isNaN(m)) { g.globalAlpha = 0.25; g.fillRect(x, h / 2 - 0.5, 1, 1); continue; }
        known = true;
        g.globalAlpha = 0.85;
        const half = Math.max(1, Math.min(1, Math.sqrt(m)) * (h / 2 - 2));
        g.fillRect(x, h / 2 - half, 1, half * 2);
      }
      el.dataset.known = known ? 'yes' : 'no';
    };
    draw();
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(draw) : null;
    ro?.observe(el);
    return () => ro?.disconnect();
  }, [seconds, peaks, version]);
  return <canvas ref={cv} className="ed-wave" aria-hidden="true" data-testid="waveform" />;
}
