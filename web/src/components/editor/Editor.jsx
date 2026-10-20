import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { HEAR_ALL, MixPlayer, describe, fetchStretch, needsProcessing, prepare } from '../../editor/mix.js';
import { LAYER_NAMES, clock, rangeText } from '../../editor/timeline.js';
import { keyOf, suggestionOps } from '../../editor/suggest.js';
import { useEditorModel } from './model.js';
import { LayerChips, LoudnessSelect, PartPanel, RangeStep, SuggestionCards } from './Panels.jsx';
import LayerRows from './Rows.jsx';
import { Overlay, Ruler, Wave, useSurface } from './Surface.jsx';
import { usePricing } from '../../hooks.js';

const PLAY = <svg className="ed-ico" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 4.5v15l12-7.5z" /></svg>;
const PAUSE = <svg className="ed-ico" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M6 4h4v16H6zM14 4h4v16h-4z" /></svg>;
const STEPS = [['home', 'Moment'], ['range', 'Part'], ['change', 'Change']];

/* ------------------------------------------------------------------------------------------------ the sound and the picture */

/** The quarter-second peaks of the recording, filled in from wherever they come (the overview, the stretches that were played, the first look). */
function usePeaks(seconds, ext) {
  const peaks = useRef(null);
  const [version, setVersion] = useState(0);
  if (!peaks.current || peaks.current.length !== Math.ceil(seconds * 4)) peaks.current = new Float32Array(Math.ceil(seconds * 4)).fill(NaN);
  const timer = useRef(0);
  const put = useCallback((from, values) => {
    const p = peaks.current;
    for (let i = 0; i < values.length && from + i < p.length; i++) p[from + i] = values[i];
    if (!timer.current) timer.current = setTimeout(() => { timer.current = 0; setVersion((v) => v + 1); }, 150);
  }, []);
  useEffect(() => () => clearTimeout(timer.current), []);
  if (ext) return { peaks: ext.arr, version: ext.version, put };
  return { peaks: peaks.current, version, put };
}

/** A few tiny stretches of the recording, spread over it: a cheap picture of where it is loud and where it is quiet. */
async function overview(jobId, seconds, channels, put, signal) {
  const points = Math.max(12, Math.min(20, Math.round(seconds / 10))); // (every stretch is a request: the account may make 120 at once, then two a second)
  const step = seconds / points;
  const todo = Array.from({ length: points }, (_, i) => Math.min(Math.max(0, seconds - 0.3), i * step + step / 2 - 0.15));
  let next = 0;
  const run = async () => {
    while (next < todo.length && !signal.aborted) {
      const i = next++;
      try {
        const got = await fetchStretch(jobId, 'source', todo[i], Math.min(0.3, seconds), { channels }, signal);
        if (!got) continue;
        let m = 0;
        for (const a of got.channels) for (let j = 0; j < a.length; j += 4) m = Math.max(m, Math.abs(a[j]));
        const from = Math.floor(i * step * 4);
        const to = Math.min(Math.ceil(seconds * 4), Math.floor((i + 1) * step * 4));
        put(from, new Float32Array(Math.max(1, to - from)).fill(m));
      } catch (e) {
        if (signal.aborted) return;
      }
    }
  };
  await Promise.all([run(), run()]);
}

/** Playing the original file in the upload form (nothing is made yet, so there is nothing else to hear). */
function useFilePlayer(file, enabled) {
  const [url, setUrl] = useState('');
  const el = useRef(null);
  useEffect(() => {
    if (!file || !enabled) return undefined;
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file, enabled]);
  return { url, el };
}

/* ------------------------------------------------------------------------------------------------ the editor */

/**
 * "Fix a moment". `mode` 'edit' is a finished job's page (listening to the mix of what the job holds, a price and Save); 'plan' is the
 * upload form (choose parts before anything is made; the price is the Start button's). The three steps (Moment, Part, Change), the layers
 * as chips, and "Show all layers" for rows. `cost` is the bar at the bottom; `onTimeline(tl)` tells the page the timeline as it is now.
 */
export default function Editor({
  mode, seconds, base, cached = {}, secondTrack = false, jobId = null, preview = null, file = null, info = null, fileName = '',
  suggestions = [], dismissed = [], onDismiss, suggestNote = null, sgAction = null, planControl = null, peaksExt = null,
  cost = null, onTimeline, active = true,
}) {
  const edit = mode === 'edit';
  const model = useEditorModel({ seconds, base, secondTrack });
  const { tl, sel, setSel } = model;
  const { data: pricing } = usePricing();
  const rebuildPaused = pricing?.rebuildAvailable === false;
  const [view, setView] = useState('home');
  const [focus, setFocus] = useState(null);
  const [allLayers, setAllLayers] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [hear, setHear] = useState(HEAR_ALL);
  const [applied, setApplied] = useState(() => new Set());
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playInfo, setPlayInfo] = useState({ buffering: false, error: '' });
  const { peaks, version: waveVersion, put } = usePeaks(seconds, peaksExt);
  const media = useRef(null);
  const playerRef = useRef(null);
  const modelRef = useRef(null);
  modelRef.current = { tl, hear, cached, secondTrack };
  const needs = useMemo(() => (edit ? needsProcessing(tl, cached, secondTrack) : []), [edit, tl, cached, secondTrack]);

  useEffect(() => { onTimeline?.(tl, model.dirty); }, [tl]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- the mix (a job) ----
  const canMix = edit && !!preview && !!jobId;
  useEffect(() => {
    if (!active || !canMix) return undefined;
    const p = new MixPlayer({
      jobId, preview, getModel: () => modelRef.current, onState: setPlayInfo,
      onEnd: () => { setPlaying(false); },
    });
    p.onPeaks = (k, pk) => put(k * 40, pk);
    playerRef.current = p;
    p.prepare();
    const ctl = new AbortController();
    overview(jobId, seconds, preview.channels?.length ?? preview.channels ?? 2, put, ctl.signal).catch(() => {});
    return () => { ctl.abort(); p.close(); playerRef.current = null; setPlaying(false); };
  }, [active, canMix, jobId, preview?.samples]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { playerRef.current?.remix(); }, [tl, hear, cached, secondTrack]);
  useEffect(() => {
    if (!playing) return undefined;
    const t = setInterval(() => {
      const p = playerRef.current;
      if (p) setPlayhead(p.time);
      else if (media.current) setPlayhead(media.current.currentTime);
    }, 100);
    return () => clearInterval(t);
  }, [playing]);

  // ---- the file itself (the picture, and the sound in the upload form) ----
  const hasVideo = !!(file && info?.hasVideo);
  const { url: fileUrl } = useFilePlayer(file, active && (hasVideo || !edit));
  const mediaMode = !edit; // the element is what plays
  useEffect(() => {
    const v = media.current;
    if (!v || !hasVideo || mediaMode) return;
    if (playing && v.paused) v.play().catch(() => {});
    if (!playing && !v.paused) v.pause();
    if (Math.abs(v.currentTime - playhead) > 0.6) v.currentTime = playhead;
  }, [playing, playhead, hasVideo, mediaMode, fileUrl]);

  const seek = useCallback((t) => {
    const x = Math.max(0, Math.min(seconds, t));
    setPlayhead(x);
    playerRef.current?.seek(x);
    if (media.current && (mediaMode || hasVideo)) media.current.currentTime = x;
  }, [seconds, mediaMode, hasVideo]);
  const toggle = useCallback(() => {
    if (mediaMode && media.current) {
      const v = media.current;
      if (v.paused) { v.play().then(() => setPlaying(true), () => {}); } else { v.pause(); setPlaying(false); }
      return;
    }
    const p = playerRef.current;
    if (!p) return;
    if (playing) { p.pause(); setPlaying(false); } else { setPlaying(true); p.play().catch(() => setPlaying(false)); }
  }, [mediaMode, playing]);
  const playFrom = useCallback((t) => {
    seek(t);
    if (mediaMode && media.current) { media.current.play().then(() => setPlaying(true), () => {}); return; }
    const p = playerRef.current;
    if (p && !p.playing) { setPlaying(true); p.play().catch(() => setPlaying(false)); }
  }, [seek, mediaMode]);

  // ---- choosing a part ----
  const scrubWrap = useRef(null);
  const choose = useCallback((s, e) => {
    setSel(s, e);
    setView((v) => (v === 'home' ? 'range' : v));
  }, [setSel]);
  // on a phone the part settings are a sheet over the bottom: keep the recording in view above it
  const hadSel = useRef(false);
  useEffect(() => {
    const has = !!sel;
    if (has && !hadSel.current && typeof window !== 'undefined' && window.innerWidth <= 760 && active) scrubWrap.current?.scrollIntoView({ block: 'start', behavior: 'smooth' });
    hadSel.current = has;
  }, [sel, active]);
  const surface = useSurface({ seconds, getSel: () => model.sel, onSelect: choose, onSeek: seek });
  const done = () => { setSel(null); setView('home'); setFocus(null); };
  const around = () => { const t = Math.round(playhead); choose(Math.max(0, Math.min(seconds - 10, t - 5)), Math.min(seconds, Math.max(10, t + 5))); };

  // ---- suggestions ----
  const items = suggestions.filter((s) => !dismissed.includes(s.key));
  const open = items.filter((s) => !applied.has(s.key));
  const apply = (s) => {
    model.apply(suggestionOps(s, s.key));
    setApplied((a) => new Set(a).add(s.key));
    setFocus(s.key);
    choose(s.start, s.end);
    setView('change');
  };
  const undo = (s) => { model.undoTag(s.key); setApplied((a) => { const n = new Set(a); n.delete(s.key); return n; }); };
  const show = (s) => { setFocus(s.key); choose(s.start, s.end); seek(s.start); setView('change'); };
  const dismissIt = (s) => { onDismiss?.(s); if (focus === s.key) setFocus(null); };
  // "You hear it right away" only where the suggestion keeps the remover that is made for that stretch and needs no Voice rebuild; the bar is the truth
  const noteOf = (s) => {
    if (!edit) return '';
    if (s.rebuild || !s.noise) return 'Needs processing';
    const have = s.noise.remover === 'heavy' ? cached.heavy : s.noise.remover === 'light' ? cached.light : cached.noise;
    const covered = (have || []).some(([a, b]) => a <= s.start + 0.05 && b >= s.end - 0.05);
    return covered ? 'You hear it right away' : 'Needs processing';
  };
  const focused = focus ? items.find((s) => s.key === focus) : null;

  // ---- what is heard ----
  const gains = useMemo(() => {
    const d = describe(prepare({ tl, hear, cached, secondTrack }), playhead);
    return JSON.stringify(Object.fromEntries(Object.entries(d).map(([k, v]) => [k, Math.round(v * 1000) / 1000])));
  }, [tl, hear, cached, secondTrack, playhead]);
  const allOn = ['noise', 'rebuild', 'leveling', ...(secondTrack ? ['echo'] : [])].every((l) => hear[l]);
  const allOff = ['noise', 'rebuild', 'leveling', ...(secondTrack ? ['echo'] : [])].every((l) => !hear[l]);
  const setHearAll = (v) => setHear({ noise: v, rebuild: v, leveling: v, echo: v });
  const toggleHear = (l) => setHear((h) => ({ ...h, [l]: !h[l] }));
  const hearText = allOn ? 'Hearing: all my edits' : allOff ? 'Hearing: the original'
    : `Hearing: ${['noise', 'rebuild', 'echo', 'leveling'].filter((l) => hear[l] && (l !== 'echo' || secondTrack)).map((l) => LAYER_NAMES[l]).join(', ')}`;

  // ---- the parts, under the waveform ----
  const bands = [];
  for (const layer of ['noise', 'rebuild', 'echo', 'leveling']) {
    if (layer === 'echo' && !secondTrack) continue;
    for (const p of tl.parts(layer)) {
      if (p.start <= 0.0005 && p.end >= seconds - 0.0005) continue; // (the whole file is the default: no mark)
      bands.push({ layer, p, pend: needs.some((n) => n.layer === layer && n.end > p.start && n.start < p.end) });
    }
  }

  // ---- steps (what shows when) ----
  const stepNow = !sel ? 'home' : view;
  const goStep = (v) => {
    if (v !== 'home' && !sel) around();
    setView(v);
    if (v === 'home') { setSel(null); setFocus(null); }
  };
  const hearChips = edit && (
    <div className="ed-hear" role="group" aria-label="Hear these layers" data-testid="hear-chips">
      <span className="lbl">Hear:</span>
      {['noise', 'rebuild', ...(secondTrack ? ['echo'] : []), 'leveling'].map((l) => (
        <label key={l} className={`ed-hchip l-${l}`}>
          <input type="checkbox" data-testid={`hear-${l}`} checked={!!hear[l]} onChange={() => toggleHear(l)} />
          <span>{LAYER_NAMES[l]}</span>
        </label>
      ))}
    </div>
  );
  const home = (
    <div data-testid="view-home">
      <div className="ed-sec">
        <h3>{edit ? 'Moments worth a look' : 'We listened to your file'} {open.length > 0 && <span className="badge" data-testid="sg-count">{open.length} new</span>}</h3>
        {suggestNote}
        <SuggestionCards
          items={items} applied={applied} focus={focus} noteOf={noteOf} onApply={apply} onShow={show} onDismiss={dismissIt} onUndo={undo}
          emptyText={suggestNote ? '' : 'No suggestions for this file.'}
        />
        {sgAction}
      </div>
      <div className="ed-sec">
        <h3>Layers</h3>
        <LayerChips
          tl={tl} cached={cached} mode={mode} secondTrack={secondTrack} seconds={seconds} needs={needs}
          onPick={(layer, p) => { setFocus(null); setSel(p.start, p.end); setView('change'); }}
        />
        <LoudnessSelect value={model.loudness} onChange={model.act.loudness} disabled={tl.parts('leveling').length === 0} />
      </div>
      <div className="ed-nav">
        <button type="button" className="btn btn-secondary" data-testid="pick-part" onClick={() => { around(); setView('range'); }}>Pick a part myself</button>
        {model.ops.length > 0 && <button type="button" className="btn btn-ghost" data-testid="reset-all" onClick={() => { model.reset(); setApplied(new Set()); }}>Undo all changes</button>}
      </div>
    </div>
  );

  return (
    <div className={`ed-core${edit ? ' is-edit' : ' is-plan'}`} data-testid="editor" data-mode={mode} data-dirty={model.dirty ? 'yes' : 'no'} data-ops={model.ops.length} data-gains={gains} data-time={playhead.toFixed(2)} data-playing={playing ? 'yes' : 'no'}>
      {planControl && <div className="ed-plan-bar" data-testid="plan-bar"><span className="small muted">What to process:</span> {planControl}</div>}
      <div className="ed-grid">
        <div className="ed-left">
          <div className="ed-vid">
            {hasVideo && fileUrl
              ? <video ref={media} className="ed-video" src={fileUrl} muted={edit} playsInline preload="auto" data-testid="video" onEnded={() => setPlaying(false)} />
              : (
                <div className="ed-video ed-poster" data-testid="poster">
                  {mediaMode && fileUrl && <audio ref={media} src={fileUrl} preload="auto" onEnded={() => setPlaying(false)} />}
                  <svg viewBox="0 0 64 40" aria-hidden="true"><g stroke="currentColor" strokeWidth="3" strokeLinecap="round"><path d="M10 16v8M18 10v20M26 5v30M34 12v16M42 8v24M50 14v12" /></g></svg>
                  <span className="ed-poster-name">{fileName}</span>
                </div>
              )}
            <span className="ed-vt">{clock(playhead)}</span>
          </div>
          <div className="ed-tp" data-testid="transport">
            <button type="button" className="btn btn-primary btn-sm ed-play" data-testid="play" aria-label={playing ? 'Pause' : 'Play'} disabled={edit && !canMix} onClick={toggle}>{playing ? PAUSE : PLAY}</button>
            <span className="ed-time mono" data-testid="time">{clock(playhead)} / {clock(seconds)}</span>
            {edit && (
              <span className="ed-bseg" role="group" aria-label="Listen to">
                <button type="button" data-testid="orig" aria-pressed={allOff} onClick={() => setHearAll(false)}>Original</button>
                <button type="button" data-testid="edited" aria-pressed={allOn} onClick={() => setHearAll(true)}>Edited</button>
              </span>
            )}
          </div>
          {edit && <p className="ed-hearing hint" role="status" data-testid="hearing">{canMix ? hearText : 'The sound for listening is not ready yet.'}{playInfo.buffering && playing ? ' · getting the next part…' : ''}</p>}
          {playInfo.error && <p className="banner banner-error small mb0" role="alert">{playInfo.error} <button type="button" className="btn btn-ghost btn-sm" onClick={() => playerRef.current?.retry()}>Try again</button></p>}
          {!allLayers && (
            <div className="ed-scrub" data-testid="scrub-wrap" ref={scrubWrap}>
              <Ruler seconds={seconds} pins={open} focus={focus} onPin={(s) => show(s)} />
              <div className="ed-wv" ref={surface.ref} {...surface.handlers} data-testid="scrub" role="group" aria-label="Recording: drag to choose a part, tap to move the playhead">
                <Wave seconds={seconds} peaks={peaks} version={waveVersion} />
                <div className="ed-pbands">
                  {bands.map(({ layer, p, pend }) => (
                    <i key={`${layer}-${p.start}`} className={`l-${layer}${pend ? ' p' : ''}`} style={{ left: `${(p.start / seconds) * 100}%`, width: `${((p.end - p.start) / seconds) * 100}%` }} />
                  ))}
                </div>
                <Overlay seconds={seconds} sel={sel} playhead={playhead} handles label />
              </div>
            </div>
          )}
          {allLayers && (
            <LayerRows
              tl={tl} seconds={seconds} secondTrack={secondTrack} sel={sel} playhead={playhead} peaks={peaks} waveVersion={waveVersion}
              hear={hear} onHear={toggleHear} onSelect={choose} onSeek={seek} zoom={zoom} onZoom={setZoom} needs={needs} mode={mode}
            />
          )}
          {hearChips}
          {edit && hear.leveling && tl.parts('leveling').length > 0 && <p className="hint mb0" data-testid="level-note">Leveling is applied when you save. Here it is only matched in loudness.</p>}
          <button type="button" className="btn btn-ghost btn-sm ed-alllayers" data-testid="all-layers" aria-pressed={allLayers} onClick={() => setAllLayers((v) => !v)}>
            {allLayers ? 'Back to the simple view' : 'Show all layers'}
          </button>
        </div>
        <div className={`ed-host${stepNow !== 'home' ? ' sheet' : ''}`} data-testid="host" data-view={stepNow}>
          <ol className="ed-st3" aria-label="Steps">
            {STEPS.map(([id, label]) => (
              <li key={id}>
                <button type="button" className={stepNow === id ? 'on' : ''} data-testid={`step-${id}`} aria-current={stepNow === id ? 'step' : undefined} onClick={() => goStep(id)}>{label}</button>
              </li>
            ))}
          </ol>
          {stepNow === 'home' && home}
          {stepNow === 'range' && sel && (
            <RangeStep
              sel={sel} seconds={seconds} playhead={playhead} onSet={choose} onAround={around} onWhole={() => choose(0, seconds)}
              onPlay={() => playFrom(sel.s)} onNext={() => setView('change')} onCancel={done}
            />
          )}
          {stepNow === 'change' && sel && (
            <>
              {focused && (
                <SuggestionCards items={[focused]} applied={applied} focus={focus} noteOf={noteOf} onApply={apply} onShow={show} onDismiss={dismissIt} onUndo={undo} emptyText="" />
              )}
              {!focused && focus && applied.has(focus) && (
                <SuggestionCards items={suggestions.filter((s) => s.key === focus)} applied={applied} focus={focus} noteOf={noteOf} onApply={apply} onShow={show} onDismiss={dismissIt} onUndo={undo} emptyText="" />
              )}
              <PartPanel model={model} cached={cached} mode={mode} secondTrack={secondTrack} rebuildPaused={rebuildPaused} onPlay={() => playFrom(sel.s)} onDone={done} />
            </>
          )}
        </div>
      </div>
      {cost}
    </div>
  );
}

export { keyOf, rangeText };
