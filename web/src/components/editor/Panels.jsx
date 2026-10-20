import { useMemo } from 'react';
import { STRENGTHS } from '../../strengths.js';
import { LAYER_NAMES, NOISE_LEVELS, clock, mergeRanges, partLabel, rangeText, subtract } from '../../editor/timeline.js';
import { sayAsk, sayWhat } from '../../editor/suggest.js';
import { LOUDNESS } from '../../jobs.js';
import WhichOne from '../WhichOne.jsx';

/** The Leveling hint for the part [s, e): a Leveling part that covers the whole file is measured over the whole file (so any
 * change of the sound redoes it there); a shorter one is measured over the part plus max(its length, 10 s) on each side. */
function levelHint(tl, s, e) {
  const p = tl.parts('leveling').find((x) => x.start <= s && x.end >= e);
  if (p && p.start <= 0 && p.end >= tl.seconds) return 'Evens out the volume. It covers the whole file, so when the sound changes anywhere it is done again for the whole file (it costs very little).';
  const pad = Math.max(e - s, 10);
  return `Evens out the volume. It measures the loudness over ${rangeText(Math.max(0, s - pad), Math.min(tl.seconds, e + pad))} to match what is next to this part.`;
}

const LEVEL_HINT = ['', ...STRENGTHS.map((s) => s.hint)];

/** Is every part of `layer` inside [a, b) made already (on the server)? */
function ready(tl, cached, layer, a, b) {
  const have = mergeRanges(cached[layer === 'noise' ? 'noise' : layer] || []);
  const want = tl.parts(layer).filter((p) => p.end > a && p.start < b).map((p) => [Math.max(p.start, a), Math.min(p.end, b)]);
  return subtract(want, have).every(([s, e]) => e - s < 0.05);
}

/* ------------------------------------------------------------------------------------------------ suggestions */

export function SuggestionCards({ items, applied, focus, noteOf, onApply, onShow, onDismiss, onUndo, emptyText }) {
  if (!items.length) return <p className="ed-none" data-testid="sg-none">{emptyText}</p>;
  return (
    <ul className="ed-sgs" data-testid="sg-list">
      {items.map((s) => {
        const { ask, act } = sayAsk(s);
        const done = applied.has(s.key);
        const note = noteOf ? noteOf(s) : '';
        return done ? (
          <li key={s.key} className="ed-sg applied" data-testid="sg-card" data-state="applied" data-key={s.key}>
            <span className="ed-sg-i" aria-hidden="true">✓</span>
            <div className="ed-sg-b"><p className="mb0"><b className="mono">{rangeText(s.start, s.end)}</b> done: {act.toLowerCase()}. <button type="button" className="btn btn-ghost btn-sm" data-testid="sg-undo" onClick={() => onUndo(s)}>Undo</button></p></div>
          </li>
        ) : (
          <li key={s.key} className={`ed-sg${focus === s.key ? ' hl' : ''}`} data-testid="sg-card" data-state="open" data-key={s.key}>
            <span className="ed-sg-i" aria-hidden="true"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l2.2 7.8L22 12l-7.8 2.2L12 22l-2.2-7.8L2 12l7.8-2.2z" /></svg></span>
            <div className="ed-sg-b">
              <p><b className="mono">{rangeText(s.start, s.end)}</b> {sayWhat(s)} {ask}</p>
              <div className="row ed-sg-row">
                <button type="button" className="btn btn-primary btn-sm" data-testid="sg-apply" onClick={() => onApply(s)}>{act}</button>
                <button type="button" className="btn btn-secondary btn-sm" data-testid="sg-show" onClick={() => onShow(s)}>Show me</button>
                <button type="button" className="btn btn-ghost btn-sm" data-testid="sg-dismiss" onClick={() => onDismiss(s)}>Dismiss</button>
                {note && <span className="hint">{note}</span>}
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

/* ------------------------------------------------------------------------------------------------ the layers, as chips */

export function LayerChips({ tl, cached, mode, secondTrack, seconds, needs, onPick }) {
  const layers = ['noise', 'rebuild', ...(secondTrack ? ['echo'] : []), 'leveling'];
  const lacks = (layer, p) => needs.some((n) => n.layer === layer && n.end > p.start && n.start < p.end);
  return (
    <div className="ed-layers" data-testid="layer-chips">
      {layers.map((layer) => {
        const parts = tl.parts(layer);
        return (
          <div key={layer} className={`ed-crow l-${layer}`} data-testid={`crow-${layer}`}>
            <b>{LAYER_NAMES[layer]}</b>
            <div className="ed-rchips">
              {parts.length ? parts.map((p) => {
                const pend = mode === 'edit' && lacks(layer, p);
                const whole = p.start <= 0.0005 && p.end >= seconds - 0.0005;
                return (
                  <button
                    key={`${p.start}-${p.end}`} type="button" className={`ed-rchip${pend ? ' pend' : ''}`} data-testid="layer-chip" data-layer={layer}
                    data-range={`${p.start}-${p.end}`} data-whole={whole ? 'yes' : 'no'} data-pending={pend ? 'yes' : 'no'}
                    title={pend ? 'Needs processing' : undefined} onClick={() => onPick(layer, p)}
                  >
                    {partLabel(layer, p, seconds)}{pend ? <span className="sr-only"> (needs processing)</span> : null}
                  </button>
                );
              }) : <span className="ed-rchip off">Off</span>}
            </div>
          </div>
        );
      })}
      <p className="hint mb0">Tap a part to change it. Where Voice rebuild and noise removal overlap, Voice rebuild is used.</p>
    </div>
  );
}

/* ------------------------------------------------------------------------------------------------ step 2: which part */

export function RangeStep({ sel, seconds, playhead, onSet, onAround, onWhole, onPlay, onNext, onCancel }) {
  const { s, e } = sel;
  const nudge = (which, d) => (which === 's' ? onSet(Math.min(s + d, e - 1), e) : onSet(s, Math.max(e + d, s + 1)));
  const box = (which, label, v) => (
    <div className="ed-ct">
      <span className="lbl">{label}</span>
      <div className="ctr">
        <button type="button" className="btn btn-secondary btn-sm" data-testid={`nudge-${which}-minus`} aria-label={`${label} one second earlier`} onClick={() => nudge(which, -1)}>−</button>
        <b data-testid={`time-${which}`}>{clock(v)}</b>
        <button type="button" className="btn btn-secondary btn-sm" data-testid={`nudge-${which}-plus`} aria-label={`${label} one second later`} onClick={() => nudge(which, 1)}>+</button>
      </div>
      <button type="button" className="btn btn-ghost btn-sm" data-testid={`use-playhead-${which}`} onClick={() => (which === 's' ? onSet(Math.min(playhead, e - 1), e) : onSet(s, Math.max(playhead, s + 1)))}>Use playhead</button>
    </div>
  );
  return (
    <div data-testid="range-step">
      <h3>Which part? <span className="mono" data-testid="range-text">{rangeText(s, e)}</span></h3>
      <p className="hint">Drag the handles above the waveform, or set the start and end here.</p>
      <div className="ed-times">{box('s', 'Start', s)}{box('e', 'End', e)}</div>
      <div className="row ed-tight">
        <button type="button" className="btn btn-secondary btn-sm" data-testid="around" onClick={onAround}>Around playhead</button>
        <button type="button" className="btn btn-secondary btn-sm" data-testid="whole-file" onClick={onWhole}>Whole file</button>
        <button type="button" className="btn btn-secondary btn-sm" data-testid="play-part" onClick={onPlay}>Play part</button>
      </div>
      <div className="ed-nav">
        <button type="button" className="btn btn-ghost" onClick={onCancel}>Cancel</button>
        <button type="button" className="btn btn-primary" data-testid="next-change" onClick={onNext}>Next: what to change</button>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------------------------------------ step 3: what to change */

export function PartPanel({ model, cached, mode, secondTrack, rebuildPaused, onPlay, onDone }) {
  const { tl, sel, act, changedHere } = model;
  const { s, e } = sel;
  const ni = act.noiseIndex();
  const rb = tl.uniform('rebuild', s, e);
  const lv = tl.uniform('leveling', s, e);
  const ec = tl.uniform('echo', s, e);
  const nz = tl.uniform('noise', s, e);
  const badge = (layer, state) => {
    if (mode !== 'edit' || !state) return null;
    const ok = layer === 'leveling' ? !changedHere : ready(tl, cached, layer, s, e);
    return ok
      ? <span className="badge badge-free" data-testid={`badge-${layer}`} data-state="ready">Ready</span>
      : <span className="badge badge-soon" data-testid={`badge-${layer}`} data-state="needs">Needs processing</span>;
  };
  const both = rb && nz !== false;
  const toggle = (layer, state, on, hint, extra = null) => (
    <div className={`ed-pl l-${layer}`} data-testid={`pl-${layer}`}>
      <label className="ed-prow">
        <b>{LAYER_NAMES[layer]} {badge(layer, state)}</b>
        <span className="switch"><input type="checkbox" role="switch" data-testid={`toggle-${layer}`} checked={state !== false && state !== null} ref={(el) => { if (el) el.indeterminate = state === null; }} onChange={(ev) => on(ev.target.checked)} aria-label={`${LAYER_NAMES[layer]} for this part`} disabled={layer === 'rebuild' && rebuildPaused && !rb} /></span>
      </label>
      <p className="hint">{hint}</p>
      {extra}
    </div>
  );
  const removerNow = nz && nz.remover ? nz.remover : model.remover;
  return (
    <div data-testid="part-panel" className="ed-pp">
      <div className="ed-pp-top">
        <span className="ed-pp-range"><span className="small muted dsk">Part</span> <span className="mono" data-testid="pp-range">{rangeText(s, e)}</span> <span className="small muted dsk">· {Math.round((e - s) * 10) / 10} s</span></span>
        <span className="ed-pp-actions">
          <button type="button" className="btn btn-secondary btn-sm" data-testid="pp-play" onClick={onPlay}>Play</button>
          {changedHere && <button type="button" className="btn btn-ghost btn-sm" data-testid="pp-undo" onClick={act.restore}><span className="dsk">Undo changes</span><span className="mob">Undo</span></button>}
          <button type="button" className="btn btn-ghost btn-sm" data-testid="pp-done" onClick={onDone}>Done</button>
        </span>
      </div>
      <div className="ed-pls">
        <div className="ed-pl l-noise" data-testid="pl-noise">
          <div className="ed-prow"><b>{LAYER_NAMES.noise} {badge('noise', nz !== false)}</b></div>
          <div className="ed-seg5" role="group" aria-label="Noise removal strength">
            {NOISE_LEVELS.map((l, i) => (
              <button key={l.id} type="button" data-testid={`noise-${l.id}`} aria-pressed={ni === i} onClick={() => act.noiseLevel(i)}>{l.label}</button>
            ))}
          </div>
          <p className="hint">{ni > 0 ? LEVEL_HINT[ni] : ni === 0 ? 'Leaves this part as it is.' : 'This part has different strengths.'}</p>
          <details className="ed-more">
            <summary>More</summary>
            <label className="field">
              <span className="label">Noise remover</span>
              <select className="select" data-testid="remover" value={removerNow} onChange={(ev) => act.remover(ev.target.value)}>
                <option value="auto">Automatic</option>
                <option value="light">Light noise</option>
                <option value="heavy">Heavy noise</option>
              </select>
              <span className="hint">Automatic listens to this part and picks. Light suits hiss, hum and a quiet room; heavy suits a café, traffic or a crowd.</span>
            </label>
          </details>
        </div>
        {toggle('rebuild', rb === null ? null : !!rb, (on) => act.rebuild(on), rebuildPaused && !rb ? 'Paused right now. Try again later.' : 'Rebuilds the voice in this part only. Good for a muffled, echoey or distorted stretch.', <WhichOne id="which-one-part" className="ed-which" />)}
        {secondTrack && toggle('echo', ec === null ? null : !!ec, (on) => act.echo(on), 'Takes the speaker echo out by using the second audio track (your desktop sound).')}
        {toggle('leveling', lv === null ? null : !!lv, (on) => act.leveling(on), levelHint(model.tl, s, e))}
      </div>
      {both && <p className="hint ed-both" data-testid="rebuild-wins">Voice rebuild is used in this part. The noise removal setting stays, but it is not run where the voice is rebuilt.</p>}
    </div>
  );
}

/* ------------------------------------------------------------------------------------------------ the whole-job setting */

export function LoudnessSelect({ value, onChange, disabled }) {
  const id = LOUDNESS.find((l) => l.value === value)?.id || String(value);
  const known = LOUDNESS.some((l) => l.id === id);
  return (
    <label className="field ed-loud">
      <span className="label">Loudness target for Leveling (the whole file)</span>
      <select className="select" data-testid="loudness" value={id} disabled={disabled} onChange={(ev) => { const l = LOUDNESS.find((x) => x.id === ev.target.value); onChange(l ? l.value : Number(ev.target.value)); }}>
        {LOUDNESS.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
        {!known && <option value={id}>{value} LUFS (as before)</option>}
      </select>
    </label>
  );
}

export function useNeedsText(needs) {
  return useMemo(() => {
    const by = {};
    for (const n of needs) (by[n.layer] ||= []).push([n.start, n.end]);
    return by;
  }, [needs]);
}
