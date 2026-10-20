import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router';
import { loadEditor, quoteTimeline, saveTimeline } from '../../editor/api.js';
import { dismiss, forgetSuggestions, keyOf, loadSuggestions, useSuggestions } from '../../editor/suggest.js';
import { LAYER_NAMES, differences, intersect, mergeRanges, rangeText, subtract } from '../../editor/timeline.js';
import { api } from '../../api.js';
import { balanceMoney, money } from '../../format.js';
import { tooManyText } from '../../jobs.js';
import Editor from './Editor.jsx';

/** How far before and after a part each step also listens (the block grids of common/acw_common/blocks.py and editor.py): not billed. */
const LISTENS = { light: [16, 0.25], heavy: [24, 0.25], echo: [10, 0.01], rebuild: [14, 14], leveling: [2, 2] };
const LAYER_OF = { light: 'noise', heavy: 'noise', echo: 'echo', rebuild: 'rebuild', leveling: 'leveling' };
const ITEM_STEPS = { denoise: ['light', 'heavy'], echo: ['echo'], rebuild: ['rebuild'], leveling: ['leveling'] };

/** Per line of a quote: the part(s) that have to be made and the stretch each is made from (the part with what the step also listens to). */
export function quoteDetails(quote, tl, cached, rendered, seconds) {
  if (!quote || quote.unchanged) return [];
  return (quote.lines || []).map((line) => {
    const steps = ITEM_STEPS[line.item] || [];
    const key = steps.find((s) => quote.steps?.[s]) || steps[0];
    const layer = LAYER_OF[key];
    let cores;
    if (layer === 'leveling') {
      // Leveling is made again where the sound under it or its own settings changed since the files were made
      const lev = tl.parts('leveling').map((p) => [p.start, p.end]);
      cores = rendered ? intersect(differences(tl, rendered), lev) : lev;
      if (!cores.length) cores = lev;
    } else {
      cores = subtract(tl.parts(layer).map((p) => [p.start, p.end]), mergeRanges(cached[layer] || []));
    }
    cores = mergeRanges(cores.filter(([s, e]) => e - s > 0.05));
    const [pre, post] = LISTENS[key] || [0, 0];
    const parts = cores.slice(0, 4).map(([s, e]) => ({ core: [s, e], read: [Math.max(0, s - pre), Math.min(seconds, e + post)] }));
    // the stretch it says it processes is what the server bills; where that is more than the part and what it listens to around it, say how much
    const spanned = parts.reduce((n, p) => n + (p.read[1] - p.read[0]), 0);
    return { ...line, parts, more: Math.max(0, cores.length - parts.length), wider: line.seconds > spanned + 2 };
  });
}

function CostBar({ q, dirty, state, error, affordable, available, busy, onSave, details, onDetails, showDetails, saveError, saved }) {
  let title;
  let sub;
  let button = null;
  if (saved) {
    title = 'Saved. Your new files download by themselves when they are ready.';
    sub = '';
  } else if (!dirty) {
    title = 'No changes yet';
    sub = 'Pick a part to change it. A new strength and switching layers are often free.';
    button = <button type="button" className="btn btn-primary" disabled data-testid="cost-save">Save</button>;
  } else if (state === 'loading' && !q) {
    title = 'Working out the price…';
    sub = '';
  } else if (state === 'error' && !q) {
    title = 'Couldn’t work out the price';
    sub = error;
  } else if (q?.unchanged) {
    title = 'No changes';
    sub = 'The files you have already hold this.';
    button = <button type="button" className="btn btn-primary" disabled data-testid="cost-save">Save</button>;
  } else if (q?.free) {
    title = <>Changes: <span className="free" data-testid="cost-free">Free</span></>;
    sub = 'Uses what is already processed. Nothing runs again.';
    button = <button type="button" className="btn btn-primary" disabled={busy || !affordable} onClick={onSave} data-testid="cost-save">{busy ? 'Saving…' : 'Save'}</button>;
  } else if (q) {
    title = `Process changes · ${money(q.estCostMicros)}`;
    sub = q.approximate ? 'About this much: the noise remover is chosen when it runs, and you pay for what runs.' : '';
    button = <button type="button" className="btn btn-primary" disabled={busy || !affordable} onClick={onSave} data-testid="cost-save">{busy ? 'Saving…' : 'Save'}</button>;
  }
  const detailRows = q && !q.free && !q.unchanged ? details : [];
  // on a phone the part settings sheet sits right above this bar: tell the style sheet how tall the bar is now
  const bar = useRef(null);
  useEffect(() => {
    const el = bar.current;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const set = () => document.documentElement.style.setProperty('--ed-cbh', `${Math.ceil(el.getBoundingClientRect().height)}px`);
    set();
    const ro = new ResizeObserver(set);
    ro.observe(el);
    return () => { ro.disconnect(); document.documentElement.style.removeProperty('--ed-cbh'); };
  }, []);
  return (
    <div className="ed-costbar" ref={bar} data-testid="cost-bar" data-state={saved ? 'saved' : !dirty ? 'idle' : q?.unchanged ? 'unchanged' : q?.free ? 'free' : q ? 'paid' : state} role="region" aria-label="Price of the changes">
      <div className="cb-m">
        <div className="cb-t" data-testid="cost-title" aria-live="polite">{title}</div>
        {sub && <p className="cb-d" data-testid="cost-sub">{sub}</p>}
        {detailRows.length > 0 && (
          <p className="cb-d">
            {detailRows.length === 1 && detailRows[0].parts.length === 1
              ? `${detailRows[0].label}: processes ${detailRows[0].wider ? `${detailRows[0].seconds} s` : rangeText(...detailRows[0].parts[0].read)} to do ${rangeText(...detailRows[0].parts[0].core)} well. `
              : `${detailRows.length} steps run. `}
            <button type="button" className="cb-more" data-testid="cost-details" aria-expanded={showDetails} onClick={onDetails}>{showDetails ? 'Hide' : 'Details'}</button>
          </p>
        )}
        {showDetails && detailRows.length > 0 && (
          <ul className="cb-list" data-testid="cost-detail-list">
            {detailRows.map((r) => (
              <li key={r.item}>
                <b>{r.label}</b> {money(r.micros)}:{' '}
                {r.parts.length
                  ? r.parts.map((p) => `processes ${r.wider && r.parts.length === 1 ? `${r.seconds} s` : rangeText(...p.read)} to do ${rangeText(...p.core)} well`).join('; ') + (r.more ? ` and ${r.more} more` : '')
                  : `${r.seconds} s`}
                {' '}<span className="muted">({r.seconds} s billed)</span>
              </li>
            ))}
            <li>Includes a small amount for each run. Parts are rounded up to whole seconds.</li>
          </ul>
        )}
        {!affordable && q && (
          <p className="cb-d cb-warn" role="alert" data-testid="cost-poor">This costs {money(q.estCostMicros)}, but your balance is {balanceMoney(available)}. <Link to="/app/billing">Add credit</Link></p>
        )}
        {saveError && <p className="cb-d cb-warn" role="alert" data-testid="cost-error">{saveError}</p>}
      </div>
      {button}
    </div>
  );
}

/**
 * The "Fix a moment" row under a finished job's result, and the editor it opens (collapsed to begin with: the simple flow is unchanged).
 * `fin`: the job's finishing (finish.js), told to download the new files by themselves when a save has been processed.
 */
export default function FixAMoment({ job, fin, onChanged }) {
  const [open, setOpen] = useState(false);
  const [everOpened, setEverOpened] = useState(false);
  const wants = !!job.edit?.available && !job.local;
  const sg = useSuggestions(job.id, wants);
  const openCount = sg.list.filter((s) => !sg.dismissed.includes(s.key)).length;

  useEffect(() => {
    document.body.classList.toggle('editing', open);
    return () => document.body.classList.remove('editing');
  }, [open]);
  // "Fix a moment" can be opened from Change and save again
  useEffect(() => {
    const on = (e) => {
      if (e.detail !== job.id) return;
      setOpen(true);
      setEverOpened(true);
      setTimeout(() => document.getElementById(`fix-entry-${job.id}`)?.scrollIntoView({ block: 'center', behavior: 'smooth' }), 50);
    };
    window.addEventListener('acw:open-editor', on);
    return () => window.removeEventListener('acw:open-editor', on);
  }, [job.id]);

  if (!wants) return null;
  return (
    <div className="fix-moment" data-testid="fix-moment">
      <button type="button" className="ed-entry" id={`fix-entry-${job.id}`} aria-expanded={open} aria-controls={`ed-${job.id}`} data-testid="fix-entry" onClick={() => { setOpen((o) => !o); setEverOpened(true); }}>
        <span className="entry-t"><b>Fix a moment</b><span className="hint">Pick a moment, or a part, and change just that.</span></span>
        {openCount > 0 && <span className="badge ed-badge" data-testid="fix-badge">{openCount} suggestion{openCount === 1 ? '' : 's'}</span>}
      </button>
      {everOpened && (
        <section className="ed" id={`ed-${job.id}`} hidden={!open} aria-label="Fix a moment" data-testid="fix-panel">
          <div className="ed-head">
            <h2>Fix a moment</h2>
            <span className="small muted ed-head-note dsk">Subtitles stay as they are: set them under Change and save again.</span>
            <button type="button" className="btn btn-secondary btn-sm" data-testid="fix-close" onClick={() => setOpen(false)}>Close</button>
          </div>
          <JobEditor job={job} fin={fin} active={open} onChanged={onChanged} suggestions={sg} />
        </section>
      )}
    </div>
  );
}

function JobEditor({ job, fin, active, onChanged, suggestions }) {
  const [data, setData] = useState(null); // {base, rendered, cached, seconds, preview, secondTrack, ...}
  const [loadError, setLoadError] = useState('');
  const reloadData = useCallback(async () => {
    try {
      setData(await loadEditor(job.id));
      setLoadError('');
    } catch (e) {
      setLoadError(e.detail || e.message || 'Couldn’t open the editor for this file.');
    }
  }, [job.id]);
  useEffect(() => { reloadData(); }, [reloadData]);

  const [tl, setTl] = useState(null);
  const [dirty, setDirty] = useState(false);
  const [quote, setQuote] = useState({ state: 'idle', q: null, error: '', key: '' });
  const [showDetails, setShowDetails] = useState(false);
  const [busy, setBusy] = useState(false);
  const [saveError, setSaveError] = useState('');
  const [saved, setSaved] = useState(false);
  const key = useMemo(() => (tl ? JSON.stringify(tl.toJSON().layers) : ''), [tl]);
  const [again, setAgain] = useState(0); // a price the server asked us to wait for is asked again after the wait

  // the price of the timeline as it is now, asked of the server a moment after the last change
  useEffect(() => {
    if (!dirty || !tl || !data) { setQuote({ state: 'idle', q: null, error: '', key }); return undefined; }
    setSaved(false);
    setSaveError('');
    setQuote((q) => ({ ...q, state: 'loading', error: '' }));
    const ctl = new AbortController();
    const t = setTimeout(() => {
      quoteTimeline(job.id, tl, ctl.signal).then(
        (q) => setQuote({ state: 'ok', q, error: '', key }),
        (e) => {
          if (e?.name === 'AbortError') return;
          setQuote({ state: 'error', q: null, error: e.detail || e.message || 'Couldn’t work out the price.', key });
          if (e.status === 429 && e.retryAfter && e.retryAfter <= 60) retry = setTimeout(() => setAgain((n) => n + 1), (e.retryAfter + 0.5) * 1000);
        },
      );
    }, 450);
    let retry = 0;
    return () => { clearTimeout(t); clearTimeout(retry); ctl.abort(); };
  }, [key, dirty, job.id, data, again]); // eslint-disable-line react-hooks/exhaustive-deps

  const q = quote.key === key ? quote.q : null; // (a price for an older state of the timeline is not shown for this one)
  const affordable = !q || q.affordable !== false;
  const details = useMemo(() => (q && tl && data ? quoteDetails(q, tl, data.cached, data.rendered, data.seconds) : []), [q, tl, data]);

  async function save() {
    if (!q || !tl) return;
    setBusy(true);
    setSaveError('');
    try {
      await saveTimeline(job.id, tl, { free: q.free, maxCostMicros: q.estCostMicros });
      fin.rearm(); // the new files come down by themselves when the job is done again
      forgetSuggestions(job.id);
      setSaved(true);
      onChanged?.();
      // a free save is over in a second: look again soon, so the page sees the job go back through the queue and finish
      setTimeout(() => onChanged?.(), 1200);
      setTimeout(() => onChanged?.(), 3500);
    } catch (e) {
      if (e.code === 'price_changed') setSaveError('The price changed while you were deciding. The new price is shown; save again if it is fine.');
      else if (e.code === 'too_many_active_jobs') setSaveError(tooManyText(3));
      else if (e.status === 402) setSaveError(e.detail || 'There isn’t enough credit for this.');
      else setSaveError(e.detail || e.message || 'Couldn’t save this. Please try again.');
      if (e.code === 'price_changed') setQuote((x) => ({ ...x, state: 'loading', q: null }));
      if (e.code === 'price_changed') quoteTimeline(job.id, tl).then((nq) => setQuote({ state: 'ok', q: nq, error: '', key }), () => {});
    } finally {
      setBusy(false);
    }
  }

  const list = suggestions.list;
  const lookFor = suggestions.loaded && list.length === 0 && job.edit?.suggestions === false && !job.edit?.run;
  const [looking, setLooking] = useState('');
  async function lookForSuggestions() {
    setLooking('starting');
    try {
      await api(`jobs/${job.id}/suggestions`, { method: 'POST' });
      setLooking('running');
      onChanged?.();
    } catch (e) {
      setLooking(e.detail || 'Couldn’t start looking.');
    }
  }

  if (loadError) return <div className="ed-body"><p className="banner banner-error mb0" role="alert" data-testid="editor-error">{loadError}</p></div>;
  if (!data) return <div className="ed-body"><p role="status" className="muted mb0"><span className="spinner" aria-hidden="true" /> Opening the editor…</p></div>;
  const file = fin.file;
  return (
    <Editor
      mode="edit" active={active} seconds={data.seconds} base={data.base} cached={data.cached} secondTrack={data.secondTrack} jobId={job.id}
      preview={data.preview} file={file} info={fin.info} fileName={job.displayName || job.inputName}
      suggestions={list} dismissed={suggestions.dismissed} onDismiss={(s) => dismiss(job.id, s.key)}
      sgAction={lookFor ? (
        <p className="mb0" data-testid="look-for">
          {looking === 'running' ? 'Looking for suggestions… they appear here when this file is done again.' : <button type="button" className="btn btn-secondary btn-sm" onClick={lookForSuggestions} disabled={looking === 'starting'}>Look for suggestions</button>}
          {looking && looking !== 'running' && looking !== 'starting' && <span className="hint"> {looking}</span>}
        </p>
      ) : null}
      onTimeline={(t, d) => { setTl(t); setDirty(d); }}
      cost={(
        <CostBar
          q={q} dirty={dirty} state={quote.state} error={quote.error} affordable={affordable} available={q?.availableMicros} busy={busy} onSave={save}
          details={details} showDetails={showDetails} onDetails={() => setShowDetails((v) => !v)} saveError={saveError} saved={saved && !dirty}
        />
      )}
    />
  );
}

export { LAYER_NAMES, keyOf, loadSuggestions };
