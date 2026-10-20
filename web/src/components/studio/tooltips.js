import { useEffect } from 'react';

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/**
 * Tooltips for everything with a `data-tip`: after about a second of hover (mouse), on keyboard focus, or on a long-press (touch).
 * `tip`: a ref to the tooltip element (position: fixed). The listeners live on the document for as long as the editor does.
 */
export function useTooltips(tip) {
  useEffect(() => {
    let tipT = 0;
    let tipFor = null;
    let tipLong = false;
    let tipXY = null;
    const el0 = () => tip.current;
    const hideTip = () => { clearTimeout(tipT); const t = el0(); if (t) t.hidden = true; tipFor = null; };
    const showTip = (el) => {
      const t = el0();
      if (!t || !el || !el.isConnected || !el.dataset.tip) return;
      tipFor = el;
      t.textContent = el.dataset.tip;
      t.hidden = false;
      const r = el.getBoundingClientRect();
      const w = t.offsetWidth;
      const h = t.offsetHeight;
      let y = r.top - h - 8;
      if (y < 8) y = r.bottom + 8;
      t.style.left = `${clamp(r.left + r.width / 2 - w / 2, 8, innerWidth - w - 8)}px`;
      t.style.top = `${y}px`;
    };
    const near = (ev) => ev.target.closest && ev.target.closest('[data-tip]');
    const over = (ev) => {
      if (ev.pointerType === 'touch') return;
      const el = near(ev);
      if (el === tipFor) return;
      hideTip();
      if (el) tipT = setTimeout(() => showTip(el), 1000);
    };
    const out = (ev) => {
      if (ev.pointerType === 'touch') return;
      const el = near(ev);
      if (el && !el.contains(ev.relatedTarget)) hideTip();
    };
    const focusin = (ev) => {
      const el = near(ev);
      hideTip();
      if (el && ev.target.matches(':focus-visible')) tipT = setTimeout(() => showTip(el), 700);
    };
    const down = (ev) => {
      hideTip();
      tipLong = false;
      if (ev.pointerType !== 'touch') return;
      const el = near(ev);
      if (!el) return;
      tipXY = [ev.clientX, ev.clientY];
      tipT = setTimeout(() => { showTip(el); tipLong = true; }, 500);
    };
    const move = (ev) => {
      if (tipXY && Math.abs(ev.clientX - tipXY[0]) + Math.abs(ev.clientY - tipXY[1]) > 8) { tipXY = null; if (!tipLong) clearTimeout(tipT); }
    };
    const up = () => { tipXY = null; if (!tipLong) clearTimeout(tipT); };
    const menu = (ev) => { if (tipLong) ev.preventDefault(); };
    // a long-press shows the tooltip instead of pressing the button
    const click = (ev) => { if (tipLong) { tipLong = false; ev.preventDefault(); ev.stopPropagation(); } };
    const key = (ev) => { if (ev.key === 'Escape') hideTip(); };
    document.addEventListener('pointerover', over);
    document.addEventListener('pointerout', out);
    document.addEventListener('focusin', focusin);
    document.addEventListener('focusout', hideTip);
    document.addEventListener('scroll', hideTip, { passive: true, capture: true });
    document.addEventListener('pointerdown', down, true);
    document.addEventListener('pointermove', move, true);
    document.addEventListener('pointerup', up, true);
    document.addEventListener('contextmenu', menu);
    document.addEventListener('click', click, true);
    document.addEventListener('keydown', key);
    return () => {
      clearTimeout(tipT);
      document.removeEventListener('pointerover', over);
      document.removeEventListener('pointerout', out);
      document.removeEventListener('focusin', focusin);
      document.removeEventListener('focusout', hideTip);
      document.removeEventListener('scroll', hideTip, { capture: true });
      document.removeEventListener('pointerdown', down, true);
      document.removeEventListener('pointermove', move, true);
      document.removeEventListener('pointerup', up, true);
      document.removeEventListener('contextmenu', menu);
      document.removeEventListener('click', click, true);
      document.removeEventListener('keydown', key);
    };
  }, [tip]);
}
