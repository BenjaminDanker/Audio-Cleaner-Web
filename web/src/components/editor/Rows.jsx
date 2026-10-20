import { LAYER_NAMES, NOISE_LEVELS, clock, levelOf, partLabel } from '../../editor/timeline.js';
import { Overlay, Ruler, Wave, useSurface } from './Surface.jsx';

const CLIP_TEXT = {
  noise: (p) => NOISE_LEVELS[levelOf(p.settings.strength_db)].label,
  rebuild: () => 'Rebuild',
  echo: () => 'Echo',
  leveling: () => 'Leveling',
};

function Lane({ children, seconds, sel, playhead, onSelect, onSeek, label, testId }) {
  const { ref, handlers } = useSurface({ seconds, getSel: () => sel, onSelect, onSeek });
  return (
    <div className="ed-lane" ref={ref} {...handlers} data-testid={testId} aria-label={label}>
      {children}
      <Overlay seconds={seconds} sel={sel} playhead={playhead} handles />
    </div>
  );
}

/**
 * "Show all layers": one row for the recording and one for each layer, with every part as a clip. Drag across any row to choose a part,
 * tap a clip to choose exactly it. For heavy editing; the panel beside it is the same.
 */
export default function LayerRows({ tl, seconds, secondTrack, sel, playhead, peaks, waveVersion, hear, onHear, onSelect, onSeek, zoom, onZoom, needs, mode }) {
  const layers = ['noise', 'rebuild', ...(secondTrack ? ['echo'] : []), 'leveling'];
  const pct = (t) => `${(t / seconds) * 100}%`;
  const lacks = (layer, p) => needs.some((n) => n.layer === layer && n.end > p.start && n.start < p.end);
  return (
    <div className="ed-rows" data-testid="rows">
      <div className="ed-rows-bar">
        <span className="small muted" data-testid="zoom-label">{zoom === 1 ? 'Whole file' : `${clock(seconds / zoom)} visible`}</span>
        <span className="row ed-tight">
          <button type="button" className="btn btn-secondary btn-sm" data-testid="zoom-out" aria-label="Zoom out" disabled={zoom <= 1} onClick={() => onZoom(Math.max(1, zoom / 2))}>−</button>
          <button type="button" className="btn btn-secondary btn-sm" data-testid="zoom-in" aria-label="Zoom in" disabled={zoom >= 8} onClick={() => onZoom(Math.min(8, zoom * 2))}>+</button>
        </span>
      </div>
      <div className="ed-rows-grid">
        <div className="ed-heads">
          <div className="ed-rh ed-rh-rec"><b>Recording</b></div>
          {layers.map((layer) => (
            <div key={layer} className={`ed-rh l-${layer}`} data-testid={`row-head-${layer}`}>
              <b>{LAYER_NAMES[layer]}</b>
              {mode === 'edit' && (
                <label className="ed-rh-hear">
                  <span className="switch"><input type="checkbox" role="switch" data-testid={`row-hear-${layer}`} checked={!!hear[layer]} onChange={() => onHear(layer)} aria-label={`Hear ${LAYER_NAMES[layer]}`} /></span>
                  <span className="small muted">Hear</span>
                </label>
              )}
            </div>
          ))}
        </div>
        <div className="ed-scroll">
          <div className="ed-inner" style={{ width: `${zoom * 100}%` }}>
            <Lane seconds={seconds} sel={sel} playhead={playhead} onSelect={onSelect} onSeek={onSeek} label="Recording" testId="row-recording">
              <Ruler seconds={seconds} />
              <div className="ed-wv ed-wv-rows"><Wave seconds={seconds} peaks={peaks} version={waveVersion} /></div>
            </Lane>
            {layers.map((layer) => (
              <Lane key={layer} seconds={seconds} sel={sel} playhead={playhead} onSelect={onSelect} onSeek={onSeek} label={LAYER_NAMES[layer]} testId={`row-${layer}`}>
                {tl.parts(layer).map((p) => (
                  <button
                    key={`${p.start}-${p.end}`} type="button" className={`ed-clip l-${layer}${mode === 'edit' && lacks(layer, p) ? ' pend' : ''}`}
                    style={{ left: pct(p.start), width: `${((p.end - p.start) / seconds) * 100}%` }} data-testid="clip" data-layer={layer}
                    data-range={`${p.start}-${p.end}`} title={partLabel(layer, p, seconds)} onClick={() => onSelect(p.start, p.end)}
                  >
                    <i>{CLIP_TEXT[layer](p)}</i>
                  </button>
                ))}
              </Lane>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
