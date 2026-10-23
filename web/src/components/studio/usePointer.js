import { useEffect, useRef } from 'react';
import { setSel, toggleRange } from '../../editor/tracks.js';

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/**
 * The pointer and the keys on the editor screen: drag across the waveform or a row to choose a part, the part's handles, a long press
 * (touch) to choose more, the mobile sheet's header, a click on the waveform to move the playhead, Escape, undo / redo, space to play.
 * Everything it needs as it is now is read from `live.current` (S, session, mobile, processing, pop, modal, more, seek, togglePlay and the
 * setters). Returns `suppress` (true for the click that ends a drag) and `lpFired` (a long press chose a part).
 */
export function usePointer({ rootRef, waveRef, T, live }) {
  const suppress = useRef(false);
  const lpFired = useRef(false);
  useEffect(() => {
    let drag = null;
    let sheetDrag = null;
    let lp = null;
    const noClick = () => { suppress.current = true; setTimeout(() => { suppress.current = false; }, 0); };
    const rawT = (x) => { const r = waveRef.current.getBoundingClientRect(); return clamp(((x - r.left) / r.width) * T, 0, T); };
    const edgeT = (t) => (t < T * 0.01 ? 0 : t > T * 0.99 ? T : Math.round(t));
    const L = () => live.current;
    const down = (ev) => {
      lpFired.current = false;
      clearTimeout(lp);
      const root = rootRef.current;
      if (!root || !root.contains(ev.target)) return;
      const ih = ev.target.closest('.ih');
      if (ih && L().mobile) { sheetDrag = { y: ev.clientY, moved: false }; return; }
      if (L().processing) return;
      const Sx = L().S;
      const h = ev.target.closest('.hnd');
      if (h && (!Sx.sel || Sx.sel.length === 1)) {
        if (!Sx.sel) Sx.sel = [{ s: 0, e: T }];
        drag = { mode: h.dataset.h };
        h.setPointerCapture(ev.pointerId);
        ev.preventDefault();
        return;
      }
      const sc = ev.target.closest('.sec');
      if (ev.pointerType === 'touch') {
        if (sc) {
          const s0 = +sc.dataset.s;
          const e0 = +sc.dataset.e;
          lp = setTimeout(() => { lpFired.current = true; L().setMore(true); toggleRange(Sx, s0, e0); L().session.touch(); }, 450);
        }
        return;
      }
      const area = ev.target.closest('.main');
      if (!area || ev.target.closest('#top')) return; // (not the top part: the heard line over the waveform is an svg with the class "top" too)
      const btn = ev.target.closest('button');
      if (btn && !btn.classList.contains('sec')) return;
      drag = { mode: 'new', x0: ev.clientX, t0: edgeT(rawT(ev.clientX)), moved: false, trk: area, pid: ev.pointerId, add: ev.shiftKey, base: Sx.sel ? Sx.sel.map((r) => ({ s: r.s, e: r.e })) : [] };
    };
    const move = (ev) => {
      if (lp && Math.abs(ev.movementX) + Math.abs(ev.movementY) > 6) clearTimeout(lp);
      if (sheetDrag) {
        const dy = ev.clientY - sheetDrag.y;
        if (Math.abs(dy) > 22 && !sheetDrag.moved) { sheetDrag.moved = true; L().setSheet(dy < 0); noClick(); }
        return;
      }
      if (!drag) return;
      const Sx = L().S;
      const t = edgeT(rawT(ev.clientX));
      if (drag.mode === 's' || drag.mode === 'e') {
        const r = Sx.sel[0];
        if (drag.mode === 's') r.s = clamp(t, 0, r.e - 1); else r.e = clamp(t, r.s + 1, T);
      } else {
        if (!drag.moved) {
          if (Math.abs(ev.clientX - drag.x0) < 4) return;
          drag.moved = true;
          try { drag.trk.setPointerCapture(drag.pid); } catch { /* ignore */ }
        }
        let s = Math.min(drag.t0, t);
        let e = Math.max(drag.t0, t);
        if (e - s < 1) { if (s + 1 <= T) e = s + 1; else s = e - 1; }
        setSel(Sx, (drag.add ? drag.base : []).concat([{ s, e }]));
      }
      L().session.touch();
    };
    const end = () => {
      clearTimeout(lp);
      if (sheetDrag) { sheetDrag = null; return; }
      if (!drag) return;
      const moved = drag.mode !== 'new' || drag.moved;
      drag = null;
      if (moved) {
        noClick();
        const Sx = L().S;
        setSel(Sx, Sx.sel || []);
        if (Sx.sel && L().mobile) L().setSheet(true);
        L().session.touch();
      }
    };
    const click = (ev) => {
      const wf = ev.target.closest('#wf');
      if (wf && rootRef.current?.contains(wf) && !suppress.current && !ev.target.closest('.hnd')) L().seek(rawT(ev.clientX));
    };
    const key = (ev) => {
      const tg = ev.target;
      const typing = tg && (tg.tagName === 'INPUT' || tg.tagName === 'SELECT' || tg.tagName === 'TEXTAREA');
      const Lx = L();
      if (ev.key === 'Escape') {
        if (Lx.modal) return;
        if (Lx.pop) { Lx.setPop(null); } else if (Lx.S.sel || Lx.more) { Lx.S.sel = null; Lx.setMore(false); Lx.session.touch(); }
        return;
      }
      if (typing || !rootRef.current?.contains(tg) && tg !== document.body) return;
      if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 'z') { ev.preventDefault(); if (ev.shiftKey) Lx.session.redo(); else Lx.session.undo(); return; }
      if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 'y') { ev.preventDefault(); Lx.session.redo(); return; }
      if (ev.key === ' ' && tg.tagName !== 'BUTTON') { ev.preventDefault(); Lx.togglePlay(); }
      const h = tg.closest && tg.closest('.hnd');
      if (h && (!Lx.S.sel || Lx.S.sel.length === 1) && (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight')) {
        ev.preventDefault();
        const d = ev.key === 'ArrowLeft' ? -1 : 1;
        const r = Lx.S.sel ? Lx.S.sel[0] : { s: 0, e: T };
        if (h.dataset.h === 's') r.s = clamp(r.s + d, 0, r.e - 1); else r.e = clamp(r.e + d, r.s + 1, T);
        setSel(Lx.S, [r]);
        Lx.session.touch();
      }
    };
    // (the path is read from the event: what was clicked may be gone from the page by now, drawn again by the click itself)
    const outside = (ev) => { if (L().pop && !ev.composedPath().some((n) => n.classList && (n.classList.contains('pop') || n.classList.contains('bb')))) L().setPop(null); };
    document.addEventListener('pointerdown', down);
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', end);
    document.addEventListener('pointercancel', end);
    document.addEventListener('click', click);
    document.addEventListener('click', outside);
    document.addEventListener('keydown', key);
    return () => {
      clearTimeout(lp);
      document.removeEventListener('pointerdown', down);
      document.removeEventListener('pointermove', move);
      document.removeEventListener('pointerup', end);
      document.removeEventListener('pointercancel', end);
      document.removeEventListener('click', click);
      document.removeEventListener('click', outside);
      document.removeEventListener('keydown', key);
    };
  }, [T]); // eslint-disable-line react-hooks/exhaustive-deps
  return { suppress, lpFired };
}
