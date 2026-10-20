import { useEffect, useRef, useState } from 'react';
import { setSuggestions, useSuggestions, dismiss, forgetSuggestions } from '../../editor/suggest.js';
import { startSuggestions, suggestVerdict } from '../../media/suggest-run.js';
import Editor from './Editor.jsx';

/**
 * "Choose parts (optional)" in the upload form: the same editor, in plan mode. The suggestions are found in this browser while the person
 * looks (files up to 30 minutes this browser can open; longer ones get theirs after the first run). `base` is the whole file as the form's
 * choices say; `onPlan({tl, dirty})` tells the form what the person has made of it.
 */
export default function PlanParts({ file, info, seconds, base, secondTrack, planControl, onPlan }) {
  const id = useRef(`plan-${Math.random().toString(36).slice(2)}`).current;
  const verdict = suggestVerdict(info, seconds);
  const [run, setRun] = useState({ phase: verdict.ok ? 'running' : 'unavailable', progress: 0, error: '' });
  const [peaks, setPeaks] = useState(() => ({ arr: new Float32Array(Math.ceil(seconds * 4)).fill(NaN), version: 0 }));
  const task = useRef(null);
  const sg = useSuggestions(id, false);

  useEffect(() => {
    if (!verdict.ok) return undefined;
    let alive = true;
    const arr = new Float32Array(Math.ceil(seconds * 4)).fill(NaN);
    let version = 0;
    let timer = 0;
    const bump = () => { timer = 0; if (alive) setPeaks({ arr, version: ++version }); };
    const t = startSuggestions(file, info, {
      seconds,
      onProgress: (f) => alive && setRun((r) => (r.phase === 'running' ? { ...r, progress: f } : r)),
      onPeaks: (from, pk) => { for (let i = 0; i < pk.length && from + i < arr.length; i++) arr[from + i] = pk[i]; if (!timer) timer = setTimeout(bump, 200); },
    });
    task.current = t;
    t.done.then(
      (r) => { if (alive) { setSuggestions(id, r.suggestions); setRun({ phase: 'done', progress: 1, error: '' }); bump(); } },
      (e) => { if (alive) setRun({ phase: e?.name === 'AbortError' ? 'canceled' : 'error', progress: 0, error: e?.message || '' }); },
    );
    return () => { alive = false; clearTimeout(timer); t.cancel(); forgetSuggestions(id); };
  }, [file, info, seconds]); // eslint-disable-line react-hooks/exhaustive-deps

  const note = (
    <>
      {run.phase === 'running' && (
        <div className="ed-progress" data-testid="sg-progress" role="status">
          <span className="spinner" aria-hidden="true" />
          <span>Listening to your file for moments worth a look… {Math.round(run.progress * 100)}%</span>
          <progress max={100} value={Math.round(run.progress * 100)} aria-label="Looking for suggestions" />
          <button type="button" className="btn btn-ghost btn-sm" data-testid="sg-cancel" onClick={() => task.current?.cancel()}>Cancel</button>
        </div>
      )}
      {run.phase === 'unavailable' && <p className="ed-none" data-testid="sg-unavailable">{verdict.why}</p>}
      {run.phase === 'canceled' && <p className="ed-none" data-testid="sg-canceled">Stopped looking. You can still choose parts yourself.</p>}
      {run.phase === 'error' && <p className="ed-none" data-testid="sg-error">Couldn’t look for suggestions in this file. You can still choose parts yourself.</p>}
    </>
  );
  return (
    <Editor
      mode="plan" seconds={seconds} base={base} secondTrack={secondTrack} file={file} info={info} fileName={file?.name}
      suggestions={sg.list} dismissed={sg.dismissed} onDismiss={(s) => dismiss(id, s.key)} suggestNote={run.phase === 'done' ? null : note}
      planControl={planControl} peaksExt={peaks} onTimeline={(tl, dirty) => onPlan({ tl, dirty })}
    />
  );
}
