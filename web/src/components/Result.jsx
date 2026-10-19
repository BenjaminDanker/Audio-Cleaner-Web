import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { api, outputUrl } from '../api.js';
import { useAuth } from '../auth.jsx';
import { balanceMoney, bytes, clock, money, shownName } from '../format.js';
import { beginFinish, useFinish } from '../finish.js';
import { findRate, usePricing } from '../hooks.js';
import { highsHint, rebuildHint } from '../hints.js';
import { jobPath, LOUDNESS } from '../jobs.js';
import { BlendPlayer, dbAt, energySums, gainsFor, matchGain, mixOf, posOf, strengthName } from '../media/blend.js';
import { canHoldSubtitles, FORMAT } from '../media/convert.js';
import { STRENGTHS } from '../strengths.js';
import AbPlayer from './AbPlayer.jsx';
import { FinishProgress } from './Progress.jsx';
import { IconDownload } from './Icons.jsx';
import OutputFields, { useOutputChoice } from './OutputPicker.jsx';
import StrengthPicker from './StrengthPicker.jsx';

const BUSY = new Set(['waiting', 'building', 'saving']);
const CAPS = [['srt', 'SRT'], ['vtt', 'VTT'], ['txt', 'TXT']];
const SAME_DB = 0.05; // two strengths this close are the same

/* ------------------------------------------------------------------------------------------------ the paid choices */

/** The id of the loudness choice for a value in LUFS (null: keep the level). */
const loudnessId = (v) => (v == null ? 'keep' : LOUDNESS.find((l) => l.value === v)?.id || String(v));

/**
 * The choices of "Change and save again" that can be redone without starting over (strength, Voice rebuild, restoring highs,
 * Leveling and its loudness, the kind of subtitles), and what saving them does. The slider of the listen section and the
 * panel are two views of the same state:
 *   nothing  no paid choice changed: only the free ones (file type, name, subtitle files) are made again, in the browser
 *   local    the new sound is the noise removal at the chosen strength and nothing after it: blended here from the two copies
 *            the job kept (or the two this browser made), free, no job
 *   server   a re-run on our side of the steps that changed (rerun.py): the price comes from the server before it is confirmed
 */
function useRedo(fin, job) {
  const nav = useNavigate();
  const { refresh } = useAuth();
  const { data: pricing } = usePricing();
  const server = !job.local && !!job.redo?.available;
  const opts = job.options || {};
  const tier = job.denoiseTier || null;
  const baseSubs = job.transcribeTier || 'off';
  // what the job on our side was made with
  const made = useMemo(() => ({
    atten: Number.isFinite(opts.atten_lim_db) ? opts.atten_lim_db : 100,
    // the free cleanup in this browser does the noise removal and nothing after it
    highs: job.local ? false : opts.restore_highs === true,
    leveling: job.local ? false : opts.leveling !== false,
    loudness: opts.loudness_lufs === undefined ? -16 : opts.loudness_lufs,
    rebuild: tier === 'rebuild',
    subs: baseSubs,
  }), [job.local, opts.atten_lim_db, opts.restore_highs, opts.leveling, opts.loudness_lufs, tier, baseSubs]);
  const start = useCallback(() => ({ ...made, atten: fin.currentStrength() }), [made, fin]);
  const [c, setC] = useState(start);
  const set = useCallback((patch) => setC((p) => ({ ...p, ...patch })), []);
  useEffect(() => { setC(start()); }, [job.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const canDenoise = server && (tier === 'standard' || tier === 'rebuild');
  const canHighs = server && (tier === 'standard' || tier === 'rebuild' || tier === 'off'); // Restore highs: not with Voice rebuild on
  const canSlide = !!fin.sliderSource && tier !== 'rebuild' && (job.local || tier === 'standard');
  const here = fin.sliderSource === 'here'; // a free run made in this browser
  const rebuilding = c.rebuild && canDenoise;
  const noiseLast = !c.highs && !c.leveling && !rebuilding; // noise removal is the last step of these choices

  // what differs from what the job was made with, as the server wants it (rerun.py: only the differences)
  const body = useMemo(() => {
    if (!server) return null;
    const options = {};
    const out = { options };
    if (canDenoise && c.rebuild !== made.rebuild) out.denoiseTier = c.rebuild ? 'rebuild' : 'standard';
    if (canDenoise && !c.rebuild && Math.abs(c.atten - made.atten) > SAME_DB) options.atten_lim_db = Math.round(c.atten * 100) / 100;
    if (canHighs && !c.rebuild && c.highs !== made.highs) options.restore_highs = c.highs;
    if (server && (tier === 'standard' || tier === 'rebuild' || tier === 'off')) {
      if (c.leveling !== made.leveling) options.leveling = c.leveling;
      if (c.leveling && c.loudness !== made.loudness) options.loudness_lufs = c.loudness;
    }
    if (c.subs !== made.subs) out.transcribeTier = c.subs;
    const empty = !out.denoiseTier && !out.transcribeTier && Object.keys(options).length === 0;
    return empty ? null : out;
  }, [server, canDenoise, canHighs, tier, c, made]);

  const strengthChanged = Math.abs(c.atten - fin.currentStrength()) > SAME_DB;
  // The result is the blend of the two copies, made here, when noise removal is the last step and nothing else is paid for.
  const localBlend = canSlide && noiseLast && !c.rebuild && c.subs === made.subs && (strengthChanged || (!here && !!body));
  const mode = here ? (strengthChanged ? 'local' : 'nothing') : localBlend ? 'local' : body ? 'server' : 'nothing';

  // the price of a run on our side, asked of the server (debounced): nothing is held or stored by asking
  const [quote, setQuote] = useState({ state: 'idle', data: null, error: '' });
  const bodyKey = JSON.stringify(body);
  useEffect(() => {
    if (mode !== 'server') { setQuote({ state: 'idle', data: null, error: '' }); return undefined; }
    setQuote((q) => ({ ...q, state: 'loading', error: '' }));
    const ctl = new AbortController();
    const t = setTimeout(() => {
      api(`jobs/${job.id}/rerun/estimate`, { method: 'POST', body, signal: ctl.signal }).then(
        (data) => setQuote({ state: 'ok', data, error: '' }),
        (e) => { if (e?.name !== 'AbortError') setQuote({ state: 'error', data: null, error: e.detail || e.message || 'Couldn’t work out the price.' }); },
      );
    }, 350);
    return () => { clearTimeout(t); ctl.abort(); };
  }, [mode, bodyKey, job.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const price = mode === 'local' ? { free: true, text: 'Free' } : mode === 'server' && quote.state === 'ok'
    ? { free: !!quote.data.free, text: quote.data.free ? 'Free' : `about ${money(quote.data.estCostMicros)}`, micros: quote.data.estCostMicros, lines: quote.data.lines || [] }
    : null;
  const affordable = mode !== 'server' || quote.state !== 'ok' || quote.data.affordable !== false;
  const canSave = !busy && (mode !== 'server' || (quote.state === 'ok' && affordable));
  const rebuildOffered = !!pricing?.rebuildAvailable && canDenoise;
  const rates = { highs: findRate(pricing, 'highs', 'standard'), rebuild: findRate(pricing, 'denoise', 'rebuild') };

  /** Save with these choices. `wants`: the free choices (file type, name, subtitle files) to make the files with. */
  async function save(wants) {
    setBusy(true);
    setError('');
    try {
      if (mode === 'server') {
        const out = await api(`jobs/${job.id}/rerun`, { method: 'POST', body: { ...body, maxCostMicros: quote.data?.estCostMicros ?? undefined } });
        // the new job is its own job with its own page; its files download by themselves when it is done
        const w = { ...wants, soundHere: false, blend: undefined };
        beginFinish({ ...out, id: out.id, status: 'queued', displayName: job.displayName || job.inputName, inputName: job.inputName, outputs: [], options: {}, durationSeconds: job.durationSeconds },
          { file: fin.file, info: fin.info, wants: w, browser: null });
        refresh();
        nav(jobPath(out.id), { state: { submitted: out.free ? 'Got it. This one is free; your files download by themselves when it’s done.' : `Got it. This costs about ${money(out.estCostMicros)}; your files download by themselves when it’s done.` } });
        return;
      }
      if (mode === 'local') {
        const db = Math.round(c.atten * 100) / 100;
        await fin.blendTo(db);
        const keepBlend = !here && !(Math.abs(db - made.atten) <= SAME_DB && made.highs === c.highs && made.leveling === c.leveling);
        await fin.run({ save: true, wants: { ...wants, blend: keepBlend ? { db } : undefined } });
        return;
      }
      await fin.run({ save: true, wants });
    } catch (e) {
      setError(e?.detail || e?.message || 'Couldn’t save this. Please try again.');
    } finally {
      setBusy(false);
    }
  }

  return {
    c, set, mode, price, quote, busy, error, canSave, affordable, save, server, canDenoise, canSlide, here, noiseLast, rebuildOffered,
    made, tier, exact: noiseLast, baseSubs, available: server || here, canHighs, rates,
  };
}

/* ------------------------------------------------------------------------------------------------ the strength slider */

const PRESETS = STRENGTHS.map((s) => ({ ...s, pos: posOf(s.db) }));

/**
 * Listen with the strength changing live: the original and the cleaned sound are two signals played together, and the strength
 * is how much of each you hear (media/blend.js). "Original" plays only the recording.
 */
function BlendListen({ fin, job, redo }) {
  const st = useFinish(fin);
  const [player, setPlayer] = useState(null);
  const [mode, setMode] = useState('cleaned');
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const pair = fin.pair;
  const exact = redo.exact;
  const sums = useMemo(() => (pair && !exact ? energySums(pair.full, pair.input) : null), [pair, exact]);
  const db = redo.c.atten;
  const gains = useMemo(() => (mode === 'original'
    ? { a: 1, g: sums ? matchGain(sums, 1) : 1, full: 0, input: sums ? matchGain(sums, 1) : 1 }
    : gainsFor(db, { sums })), [mode, db, sums]);

  useEffect(() => {
    if (!pair) return undefined;
    const p = new BlendPlayer(pair, { onEnd: () => setPlaying(false) });
    setPlayer(p);
    return () => { p.close(); setPlayer(null); setPlaying(false); setTime(0); };
  }, [pair]);
  useEffect(() => { player?.setGains(gains.full, gains.input); }, [player, gains]);
  useEffect(() => {
    if (!player || !playing) return undefined;
    const t = setInterval(() => setTime(player.time), 200);
    return () => clearInterval(t);
  }, [player, playing]);

  if (st.pairPhase === 'error') {
    return (
      <div className="stack" data-testid="blend-error">
        <p className="banner banner-error small mb0" role="alert">{st.pairError}</p>
        <div><button type="button" className="btn btn-secondary btn-sm" onClick={() => fin.ensurePair().catch(() => {})}>Try again</button></div>
      </div>
    );
  }
  if (!pair) {
    const size = job.redo?.preview?.bytes ? job.redo.preview.bytes.noise + job.redo.preview.bytes.input : 0;
    return (
      <div className="stack" data-testid="blend-prepare">
        {st.pairPhase === 'loading' ? (
          <>
            <p className="mb0" role="status"><span className="spinner" aria-hidden="true" /> Getting the sound ready to try other strengths{fin.sliderSource === 'server' ? ` (${Math.round(st.pairProgress * 100)}%)` : ''}…</p>
            <progress className="bar" max={100} value={Math.round(st.pairProgress * 100)} aria-label="Getting the sound ready" />
          </>
        ) : (
          <>
            <p className="mb0">Hear other strengths while it plays: move the slider and the sound changes at once.</p>
            <div className="row">
              <button type="button" className="btn btn-secondary" onClick={() => fin.ensurePair().catch(() => {})} data-testid="blend-load">Try other strengths</button>
              {size > 0 && <span className="small muted">Gets {bytes(size)} once, to this device.</span>}
            </div>
          </>
        )}
      </div>
    );
  }

  const dur = player?.duration || 0;
  const toggle = () => {
    if (!player) return;
    if (playing) { player.pause(); setPlaying(false); } else { player.play().then(() => setPlaying(true)).catch(() => {}); }
  };
  const seek = (v) => { setTime(v); player?.seek(v); };
  const name = strengthName(db);
  const sayPrice = redo.price ? redo.price.text : redo.mode === 'server' && redo.quote.state === 'loading' ? 'working out the price…' : '';
  return (
    <div
      className="ab blend" data-testid="blend-player" data-db={db.toFixed(2)} data-mode={mode} data-exact={exact ? 'yes' : 'no'}
      data-gain-full={gains.full.toFixed(5)} data-gain-input={gains.input.toFixed(5)} data-share={mixOf(db).toFixed(5)}
    >
      <div className="ab-modes" role="radiogroup" aria-label="What you hear">
        <div className="seg">
          <label>
            <input type="radio" name={`blend-mode-${job.id}`} value="original" checked={mode === 'original'} onChange={() => setMode('original')} data-testid="blend-original" />
            <span>Original</span>
          </label>
          <label>
            <input type="radio" name={`blend-mode-${job.id}`} value="cleaned" checked={mode === 'cleaned'} onChange={() => setMode('cleaned')} data-testid="blend-cleaned" />
            <span>Cleaned</span>
          </label>
        </div>
        <span className="muted small" aria-live="polite">Listening to the {mode} sound</span>
      </div>
      <div className="transport">
        <button type="button" className="btn btn-primary" onClick={toggle} aria-label={playing ? 'Pause' : 'Play'} data-testid="blend-play">{playing ? 'Pause' : 'Play'}</button>
        <input type="range" min={0} max={dur || 0} step="0.01" value={Math.min(time, dur || 0)} onChange={(e) => seek(Number(e.target.value))} aria-label="Seek" aria-valuetext={`${clock(time)} of ${clock(dur)}`} />
        <span className="time">{clock(time)} / {clock(dur)}</span>
      </div>
      <div className="field strength-slider">
        <label htmlFor={`strength-range-${job.id}`} className="label">Strength: <b data-testid="strength-name">{name}</b> <span className="muted small">({Math.round(db)} dB)</span></label>
        <input
          id={`strength-range-${job.id}`} type="range" min={0} max={3} step="0.01" value={posOf(db)} data-testid="strength-slider"
          onChange={(e) => redo.set({ atten: dbAt(Number(e.target.value)) })} aria-valuetext={name}
          disabled={mode === 'original'}
        />
        <div className="strength-marks" aria-hidden="true">
          {PRESETS.map((s) => <span key={s.id} style={{ left: `${(s.pos / 3) * 100}%` }}>{s.label}</span>)}
        </div>
        <div className="seg" role="group" aria-label="Strength names">
          {PRESETS.map((s) => (
            <button key={s.id} type="button" className={`btn btn-ghost btn-sm${Math.abs(s.db - db) < 0.5 ? ' is-on' : ''}`} onClick={() => { redo.set({ atten: s.db }); setMode('cleaned'); }} data-testid={`strength-set-${s.id}`}>{s.label}</button>
          ))}
        </div>
      </div>
      <p className="hint mb0" data-testid="blend-note">
        {exact
          ? 'What you hear is what you save.'
          : 'This is the noise removal only, at an even loudness. Leveling and restoring highs are added when you save.'}
      </p>
      {redo.available && redo.mode !== 'nothing' && (
        <div className="row">
          <button type="button" className="btn btn-primary" onClick={() => redo.save(fin.effective())} disabled={!redo.canSave} data-testid="blend-save">
            <IconDownload width={18} height={18} /> {redo.busy ? 'Saving…' : 'Save this strength'}{sayPrice ? ` · ${sayPrice}` : ''}
          </button>
          {redo.mode === 'local' && <span className="small muted">Made on this device. Nothing is uploaded.</span>}
        </div>
      )}
      {redo.error && <p className="banner banner-error small mb0" role="alert">{redo.error}</p>}
    </div>
  );
}

/* ------------------------------------------------------------------------------------------------ change and save again */

/** "Change and save again": the paid choices that can be redone (with the price), next to the free ones, and Save. */
function SaveAgain({ fin, job, redo }) {
  const st = useFinish(fin);
  const w = fin.effective();
  const avail = fin.availableCaps();
  const anyCaps = Object.values(avail).some(Boolean);
  const inputName = fin.inputName;
  const info = useMemo(() => fin.info || { hasVideo: /\.(mp4|mov|m4v|mkv|webm)$/i.test(inputName), duration: job.durationSeconds || null }, [fin.info, inputName, job.durationSeconds]);
  const file = useMemo(() => ({ name: inputName }), [inputName]);
  const hasMedia = w.sound && !fin.useServerVideo; // the file type and name can still change
  const choice = useOutputChoice({ file: hasMedia ? file : null, info: hasMedia ? info : null, initial: hasMedia ? { name: w.name, format: w.format, size: w.size } : null });
  const [caps, setCaps] = useState({ srt: !!w.caps.srt, vtt: !!w.caps.vtt, txt: !!w.caps.txt });
  const [embed, setEmbed] = useState(!!w.embed);
  const busy = BUSY.has(st.phase) || redo.busy;
  const videoFile = hasMedia ? FORMAT[choice.format]?.kind === 'video' : true;
  const canEmbed = w.subs && anyCaps && info.hasVideo && !fin.useServerVideo && info.viaMediabunny !== false && info.viaLibav !== true && videoFile;
  const embedOn = canEmbed && embed;
  const problem = embedOn && hasMedia && !canHoldSubtitles(choice.format) ? 'Subtitles put into the video need an MKV file. Pick MKV, or untick “Put the subtitles into the video”.' : '';
  const picked = Object.entries(caps).some(([id, on]) => on && avail[id]);
  const something = hasMedia || embedOn || (w.subs && picked) || redo.mode !== 'nothing';
  const valid = !busy && !problem && something && (!hasMedia || choice.valid) && redo.canSave;
  const c = redo.c;
  const named = STRENGTHS.find((s) => Math.abs(s.db - c.atten) < 0.5);
  const showPaid = redo.available;
  const jobId = job.id;

  function save() {
    redo.save({
      ...w,
      format: hasMedia ? choice.format : w.format,
      name: hasMedia ? choice.name.trim() : w.name,
      size: hasMedia ? choice.size : w.size,
      caps: { srt: caps.srt && avail.srt, vtt: caps.vtt && avail.vtt, txt: caps.txt && avail.txt },
      embed: embedOn,
    });
  }

  return (
    <div className="stack again" data-testid="again-fields">
      {showPaid && (
        <div className="stack again-paid" data-testid="again-paid">
          {redo.canSlide && (
            <StrengthPicker
              name={`again-${jobId}`} value={named?.id || ''} label="Strength" disabled={busy || c.rebuild}
              onChange={(id) => redo.set({ atten: STRENGTHS.find((s) => s.id === id).db })}
              customText={`${strengthName(c.atten)} (${Math.round(c.atten)} dB). Try it in the listen section above.`}
            />
          )}
          {redo.canHighs && !c.rebuild && (
            <label className="check">
              <input type="checkbox" checked={c.highs} disabled={busy} onChange={(e) => redo.set({ highs: e.target.checked })} data-testid="again-highs" />
              <span><b>Restore highs</b><span className="hint block">{highsHint(redo.rates.highs)}</span></span>
            </label>
          )}
          {redo.canDenoise && redo.rebuildOffered && (
            <label className="check">
              <input type="checkbox" checked={c.rebuild} disabled={busy} onChange={(e) => redo.set({ rebuild: e.target.checked })} data-testid="again-rebuild" />
              <span><b>Rebuild the voice</b><span className="hint block">{rebuildHint(redo.rates.rebuild)}</span></span>
            </label>
          )}
          {redo.server && redo.tier && (
            <div className="stack">
              <label className="check">
                <input type="checkbox" checked={c.leveling} disabled={busy} onChange={(e) => redo.set({ leveling: e.target.checked })} data-testid="again-leveling" />
                <span><b>Leveling</b><span className="hint block">Evens out the volume and sets the loudness. It is done again if it stays on, and has its own price.</span></span>
              </label>
              <div className="field">
                <label htmlFor={`again-loud-${jobId}`}>Loudness target</label>
                <select
                  id={`again-loud-${jobId}`} className="select" disabled={busy || !c.leveling} data-testid="again-loudness"
                  value={loudnessId(c.loudness)}
                  onChange={(e) => redo.set({ loudness: e.target.value === 'keep' ? null : (LOUDNESS.find((l) => l.id === e.target.value)?.value ?? Number(e.target.value)) })}
                >
                  {LOUDNESS.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
                  {!LOUDNESS.some((l) => l.id === loudnessId(c.loudness)) && <option value={loudnessId(c.loudness)}>{c.loudness} LUFS (as before)</option>}
                </select>
              </div>
            </div>
          )}
          {redo.server && (
            <div className="field">
              <label htmlFor={`again-subs-${jobId}`}>Subtitles and transcript</label>
              <select id={`again-subs-${jobId}`} className="select" value={c.subs} disabled={busy} onChange={(e) => redo.set({ subs: e.target.value })} data-testid="again-subs">
                <option value="off">None</option>
                <option value="local">Standard</option>
                <option value="premium">Extra accurate</option>
              </select>
              {c.subs !== redo.baseSubs && <span className="hint">Written again from your original recording, at the price of that kind of subtitles.</span>}
            </div>
          )}
          <p className="mb0" data-testid="again-price" data-state={redo.mode === 'local' ? 'free' : redo.price ? (redo.price.free ? 'free' : 'paid') : redo.quote.state}>
            {redo.mode === 'nothing' && <span className="muted">Nothing paid is changed. Saving makes the files again, free.</span>}
            {redo.mode === 'local' && <span><b>Free</b> <span className="hint">Made on this device from the sound we kept. Nothing is uploaded.</span></span>}
            {redo.mode === 'server' && redo.quote.state === 'loading' && <span className="muted"><span className="spinner" aria-hidden="true" /> Working out the price…</span>}
            {redo.mode === 'server' && redo.price && (
              <span>
                <b>{redo.price.free ? 'Free' : `Costs ${redo.price.text}`}</b>{' '}
                <span className="hint">
                  {redo.price.free ? 'Only steps that cost nothing run again.' : `${redo.price.lines.map((l) => `${l.label}${l.quantity > 1 ? ` × ${l.quantity}` : ''} ${money(l.micros)}`).join(' + ')}.`}
                  {' '}You pay only for the steps that run again.
                </span>
              </span>
            )}
            {redo.mode === 'server' && redo.quote.state === 'error' && <span className="banner banner-warn small" role="alert">{redo.quote.error}</span>}
          </p>
          {!redo.affordable && redo.quote.data && (
            <div className="banner banner-warn small" role="alert">
              This costs about {money(redo.quote.data.estCostMicros)}, but your balance is {balanceMoney(redo.quote.data.availableMicros)}. <Link to="/app/billing">Add credit</Link>
            </div>
          )}
        </div>
      )}
      {!showPaid && !job.local && job.redo && !job.redo.available && (
        <p className="hint mb0" data-testid="again-gone">The files this needs are cleared away, so strength, Leveling and the other paid choices can’t be changed any more. File type, name and subtitle files still can.</p>
      )}
      {hasMedia && <OutputFields choice={choice} info={info} idPrefix={`again-${job.id}`} disabled={busy} />}
      {w.subs && anyCaps && (
        <fieldset className="field" disabled={busy}>
          <legend className="label">Subtitle files</legend>
          <div className="checks">
            {CAPS.filter(([id]) => avail[id]).map(([id, label]) => (
              <label className="check" key={id}>
                <input type="checkbox" checked={caps[id]} onChange={(e) => setCaps({ ...caps, [id]: e.target.checked })} data-testid={`again-cap-${id}`} />
                <span>{label}</span>
              </label>
            ))}
          </div>
        </fieldset>
      )}
      {canEmbed && (
        <label className="check">
          <input
            type="checkbox" checked={embed} disabled={busy} data-testid="again-embed"
            onChange={(e) => {
              setEmbed(e.target.checked);
              if (e.target.checked && hasMedia && !canHoldSubtitles(choice.format) && choice.okIds?.includes('mkv')) choice.pickFormat('mkv');
            }}
          />
          <span><b>Put the subtitles into the video</b></span>
        </label>
      )}
      {problem && <p className="hint hint-bad mb0" role="alert">{problem}</p>}
      <div className="row">
        <button type="button" className="btn btn-primary" onClick={save} disabled={!valid} data-testid="again-save">
          <IconDownload width={18} height={18} /> {busy ? (redo.mode === 'server' ? 'Starting…' : 'Putting it together…') : 'Save'}
          {!busy && redo.mode === 'server' && redo.price ? ` · ${redo.price.text}` : ''}
        </button>
        <span className="small muted">{redo.mode === 'server' ? 'Starts a new run of the steps that changed. Your files download by themselves when it is done.' : 'Saves the files again with these choices.'}</span>
      </div>
      {redo.error && <p className="banner banner-error small mb0" role="alert" data-testid="again-error">{redo.error}</p>}
    </div>
  );
}

/**
 * Everything about a finished job, at the bottom of its card: the files it made, one "Download again" for all of them,
 * the listen toggle (with the strength slider where the job has the sound for it), and the choices that can still change.
 * The same for a job done on our side and a free one made here.
 */
export default function ResultPanel({ job, fin }) {
  const st = useFinish(fin);
  const picker = useRef(null);
  const [listen, setListen] = useState(false);
  const [again, setAgain] = useState(false);
  const redo = useRedo(fin, job);
  const w = fin.effective();
  const files = st.files.length ? st.files : fin.planned(w);
  const busy = BUSY.has(st.phase);
  const needVideo = st.phase === 'needvideo' || (st.phase === 'idle' && fin.needsVideo(w));
  const listenUrl = fin.wavUrl || (fin.serverAudio && w.sound && !w.soundHere ? outputUrl(job.id, fin.serverAudio.name) : '');
  const slide = redo.canSlide && w.sound; // the slider takes the place of the plain original / cleaned player
  const canListen = slide || (!!fin.file && !!listenUrl);
  const silent = fin.pcm && fin.pcm.rms < 1e-5;
  const preview = fin.subs && !fin.subs.empty ? fin.subs.txt.split(/\n+/).filter(Boolean).slice(0, 3).join(' ') : '';
  const pcm = fin.pcm;

  // a free run made in this browser has its two signals ready: the slider is there at once
  useEffect(() => {
    if (listen && slide && fin.sliderSource === 'here') fin.ensurePair().catch(() => {});
  }, [listen, slide, fin]);

  const onDownload = () => {
    if (needVideo) picker.current?.click();
    else fin.again();
  };

  return (
    <div
      className="result stack" data-testid={st.phase === 'ready' ? 'result' : 'result-pending'}
      {...(pcm ? { 'data-rms': pcm.rms.toFixed(6), 'data-peak': pcm.peak.toFixed(4), 'data-frames': pcm.frames, 'data-channels': pcm.channels } : {})}
      {...(fin.subs ? { 'data-cues': fin.subs.cues.length, 'data-language': fin.subs.language } : {})}
      {...(st.strengthDb != null ? { 'data-strength-db': st.strengthDb.toFixed(2) } : {})}
    >
      {(st.phase === 'building' || st.phase === 'waiting') && (
        st.phase === 'waiting'
          ? <p className="mb0" role="status" data-testid="waiting"><span className="spinner" aria-hidden="true" /> Waiting for your browser to finish…</p>
          : <FinishProgress progress={st.progress} onCancel={() => fin.cancel()} />
      )}
      {silent && <div className="banner banner-warn">The cleaned audio is nearly silent. The original may only contain noise, or try a gentler setting.</div>}
      {preview && <p className="muted mb0" data-testid="subs-preview">&ldquo;{preview.length > 220 ? `${preview.slice(0, 220)}…` : preview}&rdquo;</p>}

      {files.length > 0 && (
        <ul className="outputs" aria-label="Your files" data-testid="made">
          {files.map((f) => (
            <li key={f.name} data-testid="made-file">
              <span className="oname">{shownName(f.name)}</span>
              <span className="muted small">{f.label}{f.bytes != null ? ` · ${bytes(f.bytes)}` : ''}</span>
            </li>
          ))}
        </ul>
      )}
      {st.note && <p className="small banner banner-info mb0" role="status" data-testid="made-note">{st.note}</p>}
      {job.notice && <p className="banner banner-info small mb0" role="status" data-testid="job-notice">{job.notice}</p>}
      {st.error && <p className="small banner banner-error mb0" role="alert" data-testid="made-error">{st.error}</p>}
      {needVideo && !st.error && <p className="small muted mb0" data-testid="need-video">To put the picture back, choose your original video once. It stays on your device.</p>}

      {files.length > 0 && (
        <div className="row">
          <button type="button" className="btn btn-primary" onClick={onDownload} disabled={busy} data-testid="download-again">
            <IconDownload width={18} height={18} /> {st.phase === 'saving' ? 'Downloading…' : st.phase === 'error' ? 'Try again' : needVideo ? 'Choose the original video' : 'Download again'}
          </button>
          <input
            ref={picker} type="file" accept="video/*,.mkv,.mov,.mp4,.webm" className="sr-only" tabIndex={-1}
            aria-label="Choose the original video"
            onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ''; if (f) fin.setVideo(f); }}
          />
          {files.length > 1 && <span className="small muted">Downloads all {files.length} files. If your browser asks, allow it.</span>}
        </div>
      )}

      {canListen && (
        <details className="more" onToggle={(e) => setListen(e.currentTarget.open)} data-testid="listen">
          <summary>{slide ? 'Listen: original or cleaned, any strength' : 'Listen: original or cleaned'}</summary>
          {listen && (slide
            ? <BlendListen fin={fin} job={job} redo={redo} />
            : <AbPlayer original={fin.file} cleanedUrl={listenUrl} isVideo={!!fin.info?.hasVideo} />)}
        </details>
      )}
      <details className="more" onToggle={(e) => setAgain(e.currentTarget.open)} data-testid="again">
        <summary>Change and save again</summary>
        {again && <SaveAgain fin={fin} job={job} redo={redo} />}
      </details>
    </div>
  );
}
