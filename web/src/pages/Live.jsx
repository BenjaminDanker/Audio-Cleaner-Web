import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api, streamUrl } from '../api.js';
import { useAuth } from '../auth.jsx';
import { IconBroadcast } from '../components/Icons.jsx';
import { balanceMoney, clock, dateTime, rate, moneyUsd } from '../format.js';
import { findRate, usePageTitle, usePricing } from '../hooks.js';
import { LANGUAGES } from '../jobs.js';
import { applyCaption, startLive } from '../live/session.js';

export default function Live() {
  usePageTitle('Live & API keys');
  const { data: pricing } = usePricing();
  return (
    <div className="container page stack-lg">
      <header className="row">
        <IconBroadcast width={36} height={36} />
        <div>
          <h1 className="mb0">Live for streamers</h1>
          <p className="muted mb0">Low-latency noise removal and live captions for OBS, billed only while you are live.</p>
        </div>
      </header>
      <LivePrices pricing={pricing} />
      <div className="callout-grid">
        <ApiKeys />
        <ObsSetup />
      </div>
      <LiveDemo pricing={pricing} />
    </div>
  );
}

function LivePrices({ pricing }) {
  const rows = (pricing?.rates || []).filter((r) => r.product.startsWith('stream_'));
  return (
    <section className="card stack" aria-labelledby="lp-h">
      <h2 id="lp-h" className="mb0">Live pricing</h2>
      {!pricing && <p className="muted mb0" role="status">Loading…</p>}
      {pricing && (
        <div className="table-wrap">
          <table data-testid="live-prices">
            <thead><tr><th>Service</th><th className="num">Per minute</th><th className="num">Per hour</th><th>Status</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.product}/${r.tier}`} className={r.available ? undefined : 'soon'}>
                  <th scope="row">{r.label}</th>
                  <td className="num">{rate(r.usdPerMinute)}</td>
                  <td className="num">{moneyUsd(r.usdPerHour)}</td>
                  <td>{r.available ? <span className="badge badge-ok">Available</span> : <span className="badge badge-soon">Coming soon</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="small muted mb0">Billed per second of audio processed. A session reserves about five minutes of credit up front and releases what it does not use. Denoise and captions add up when both are on.</p>
    </section>
  );
}

function ObsSetup() {
  return (
    <section className="card stack" aria-labelledby="obs-h">
      <h2 id="obs-h" className="mb0">OBS plugin setup</h2>
      <ol className="ol-steps mb0">
        <li>Create an API key on this page with the <b>stream</b> scope and copy it.</li>
        <li>Install the Audio Cleaner OBS plugin. <span className="badge badge-soon">Coming soon</span></li>
        <li>In OBS, add the <b>Audio Cleaner</b> filter to your microphone source and paste the key.</li>
        <li>Pick a latency preset (80, 150 or 250 ms), choose captions if you want them, and go live.</li>
      </ol>
      <div className="row"><button type="button" className="btn btn-secondary" disabled aria-disabled="true">Download plugin (coming soon)</button></div>
      <p className="small muted mb0">Server address for the plugin: <span className="mono">{streamUrl().replace(/^ws/, 'ws')}</span></p>
    </section>
  );
}

/* ------------------------------------------------------------------ API keys */

function ApiKeys() {
  const [keys, setKeys] = useState(null);
  const [error, setError] = useState('');
  const [name, setName] = useState('OBS on my PC');
  const [scopes, setScopes] = useState({ stream: true, jobs: false });
  const [busy, setBusy] = useState(false);
  const [fresh, setFresh] = useState(null);
  const [copied, setCopied] = useState(false);
  const [confirm, setConfirm] = useState(null);

  const load = useCallback(async () => {
    try {
      setKeys((await api('keys')).keys);
      setError('');
    } catch (e) {
      setError(e.detail);
    }
  }, []);
  useEffect(() => { load(); }, [load]);

  async function create(e) {
    e.preventDefault();
    setBusy(true); setError(''); setFresh(null); setCopied(false);
    try {
      const r = await api('keys', { method: 'POST', body: { name: name.trim(), scopes: Object.keys(scopes).filter((s) => scopes[s]) } });
      setFresh(r);
      load();
    } catch (err) {
      setError(err.detail);
    } finally {
      setBusy(false);
    }
  }

  async function revoke(id) {
    setBusy(true);
    try {
      await api(`keys/${id}`, { method: 'DELETE' });
      setConfirm(null);
      if (fresh?.id === id) setFresh(null);
      await load();
    } catch (err) {
      setError(err.detail);
    } finally {
      setBusy(false);
    }
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(fresh.key);
      setCopied(true);
    } catch {
      setError('Copy failed. Select the key and copy it manually.');
    }
  }

  const noScope = !scopes.stream && !scopes.jobs;
  return (
    <section className="card stack" aria-labelledby="keys-h">
      <h2 id="keys-h" className="mb0">API keys</h2>
      <p className="muted mb0 small">Keys let programs (like the OBS plugin) use your balance. Treat them like passwords. Up to 10 active keys.</p>
      {fresh && (
        <div className="stack" data-testid="new-key">
          <div className="banner banner-success"><b>Your new key.</b> Copy it now. For your security it is shown only once.</div>
          <div className="key-box">
            <code data-testid="key-value">{fresh.key}</code>
            <button type="button" className="btn btn-primary btn-sm" onClick={copy}>{copied ? 'Copied' : 'Copy'}</button>
          </div>
        </div>
      )}
      <form className="stack" onSubmit={create}>
        <div className="field">
          <label htmlFor="key-name">Name</label>
          <input id="key-name" className="input" value={name} maxLength={60} onChange={(e) => setName(e.target.value)} required />
        </div>
        <fieldset className="field">
          <legend className="label">Allowed to</legend>
          <div className="checks">
            <label className="check"><input type="checkbox" checked={scopes.stream} onChange={(e) => setScopes({ ...scopes, stream: e.target.checked })} /><span>Stream (live)</span></label>
            <label className="check"><input type="checkbox" checked={scopes.jobs} onChange={(e) => setScopes({ ...scopes, jobs: e.target.checked })} /><span>Jobs (files)</span></label>
          </div>
        </fieldset>
        <div><button type="submit" className="btn btn-primary" disabled={busy || noScope || !name.trim()}>Create key</button></div>
      </form>
      {error && <div className="banner banner-error" role="alert">{error}</div>}
      {keys === null && !error && <p role="status" className="muted mb0">Loading…</p>}
      {keys && keys.length === 0 && <p className="muted mb0">No active keys.</p>}
      {keys && keys.length > 0 && (
        <ul className="outputs" aria-label="Active keys" data-testid="keys">
          {keys.map((k) => (
            <li key={k.id}>
              <span>
                <span className="oname">{k.name}</span>{' '}
                <span className="mono muted">{k.prefix}…</span>
                <div className="small muted">{k.scopes.join(', ')} · created {dateTime(k.createdAt)} · {k.lastUsedAt ? `last used ${dateTime(k.lastUsedAt)}` : 'never used'}</div>
              </span>
              {confirm === k.id ? (
                <span className="row">
                  <span className="small">Revoke this key?</span>
                  <button type="button" className="btn btn-danger btn-sm" disabled={busy} onClick={() => revoke(k.id)}>Revoke</button>
                  <button type="button" className="btn btn-ghost btn-sm" onClick={() => setConfirm(null)}>Keep</button>
                </span>
              ) : (
                <button type="button" className="btn btn-danger-outline btn-sm" onClick={() => setConfirm(k.id)} aria-label={`Revoke key ${k.name}`}>Revoke</button>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/* ------------------------------------------------------------------ browser demo */

const STRENGTH = [['18', 'Natural (18 dB)'], ['40', 'Strong (40 dB)'], ['100', 'Maximum (100 dB)']];
const ERROR_TEXT = {
  insufficient_funds: null, // server message is clear
  tos: 'Please accept the Terms of Service first.',
};

function LiveDemo({ pricing }) {
  const { balance, ensureTos, refresh } = useAuth();
  const [denoise, setDenoise] = useState(true);
  const [captions, setCaptions] = useState('local');
  const [language, setLanguage] = useState('');
  const [strength, setStrength] = useState('100');
  const [monitor, setMonitor] = useState('processed');
  const [state, setState] = useState('idle'); // idle | starting | live | stopping
  const [error, setError] = useState('');
  const [info, setInfo] = useState(null);
  const [stats, setStats] = useState(null);
  const [lines, setLines] = useState([]); // caption lines {key, id, text, final}; partials are updated in place by id
  const [warning, setWarning] = useState(null); // {text, credit}
  const [stopped, setStopped] = useState('');
  const session = useRef(null);
  const logRef = useRef(null);

  // Price of exactly what is selected: stream_denoise/light + stream_captions/<tier> from /api/billing/pricing.
  const den = findRate(pricing, 'stream_denoise', 'light');
  const capRates = { local: findRate(pricing, 'stream_captions', 'local'), premium: findRate(pricing, 'stream_captions', 'premium') };
  const cap = captions ? capRates[captions] : null;
  const perMin = (denoise && den ? den.usdPerMinute : 0) + (cap ? cap.usdPerMinute : 0);
  const priceKnown = !!pricing && (!denoise || !!den) && (!captions || !!cap);
  const offered = (!denoise || den?.available !== false) && (!captions || cap?.available !== false);
  const premiumListed = capRates.premium?.available !== false;
  const mediaOk = !!(navigator.mediaDevices?.getUserMedia && window.AudioWorklet && window.WebSocket);

  useEffect(() => () => session.current?.stop(), []);
  useEffect(() => { logRef.current?.scrollTo({ top: logRef.current.scrollHeight }); }, [lines]);

  const onEvent = useCallback((ev) => {
    if (ev.type === 'ready') setInfo(ev);
    else if (ev.type === 'caption') {
      setLines((l) => applyCaption(l, ev));
    } else if (ev.type === 'warning') {
      if (ev.code === 'low_balance') setWarning({ text: `Your balance covers about ${Math.max(0, Math.round(ev.secondsLeft))} more seconds. Add credit to keep going.`, credit: true });
      else setWarning({ text: ev.message || 'Warning from the live service.', credit: false });
    } else if (ev.type === 'stopped') {
      setStopped(ev.reason === 'insufficient_funds' ? 'The session stopped because your balance ran out.' : ev.reason === 'max_duration' ? 'The session reached its maximum length.' : 'The session stopped.');
    } else if (ev.type === 'error') {
      setError(ERROR_TEXT[ev.code] ?? ev.message ?? 'The live service reported an error.');
    } else if (ev.type === 'closed') {
      setState('idle');
      session.current = null;
      refresh();
    }
  }, [refresh]);

  async function start() {
    setError(''); setWarning(null); setStopped(''); setLines([]); setStats(null); setInfo(null);
    if (!(await ensureTos())) { setError('Please accept the Terms of Service to use live processing.'); return; }
    setState('starting');
    try {
      session.current = await startLive({
        denoise, captions: captions || null, language: language || null, attenDb: Number(strength), monitor,
        onEvent, onStats: setStats,
      });
      setState('live');
    } catch (e) {
      session.current = null;
      setState('idle');
      if (e?.name === 'NotAllowedError') setError('Microphone access was blocked. Allow it in your browser and try again.');
      else if (e?.name === 'NotFoundError') setError('No microphone was found.');
      else setError(e.message || 'Could not start.');
    }
  }

  function stop() {
    setState('stopping');
    session.current?.stop();
  }

  const live = state === 'live';
  const busy = state !== 'idle';
  return (
    <section className="card stack" aria-labelledby="demo-h">
      <h2 id="demo-h" className="mb0">Test in your browser</h2>
      <p className="muted mb0">Speak into your microphone and hear the live result, the same service the OBS plugin uses.</p>
      <div className="banner banner-warn"><b>This uses credit.</b> Your balance: <b>{balanceMoney(balance?.available_micros ?? 0)}</b>. <b>Use headphones</b> to avoid feedback.</div>
      <p className="mb0" data-testid="live-price" data-usd-per-minute={priceKnown ? perMin : ''} aria-live="polite">
        {!pricing && <span className="muted">Loading prices…</span>}
        {pricing && perMin <= 0 && priceKnown && <span className="muted">Choose at least one option.</span>}
        {pricing && !priceKnown && <span className="muted">The price for this combination is not available.</span>}
        {pricing && priceKnown && perMin > 0 && (
          <>
            <b>Your choice costs {rate(perMin)} per minute</b> <span className="muted">({moneyUsd(perMin * 60)} per hour
            {denoise && den ? `; denoise ${rate(den.usdPerMinute)}` : ''}{cap ? `${denoise && den ? ' + ' : '; '}${captions === 'premium' ? 'premium' : 'standard'} captions ${rate(cap.usdPerMinute)}` : ''}).
            {' '}A session reserves about five minutes ({moneyUsd(perMin * 5)}) up front and releases what it does not use.</span>
          </>
        )}
      </p>
      {!mediaOk && <div className="banner banner-error" role="alert">This browser does not support the features the live test needs (microphone, AudioWorklet, WebSocket).</div>}

      <div className="form-grid">
        <fieldset className="field" disabled={busy}>
          <legend className="label">Process</legend>
          <label className="check"><input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} /><span>Remove noise (light){den ? <span className="muted small"> {rate(den.usdPerMinute)}/min</span> : null}</span></label>
          <div className="field">
            <label htmlFor="live-cap">Captions</label>
            <select id="live-cap" className="select" value={captions} onChange={(e) => setCaptions(e.target.value)}>
              <option value="">None</option>
              <option value="local" disabled={capRates.local?.available === false}>Standard (on-prem model){capRates.local ? ` · +${rate(capRates.local.usdPerMinute)}/min` : ''}</option>
              {(!pricing || capRates.premium) && <option value="premium" disabled={!premiumListed}>Premium (OpenAI){capRates.premium ? ` · +${rate(capRates.premium.usdPerMinute)}/min` : ''}{premiumListed ? '' : ' · coming soon'}</option>}
            </select>
            {captions === 'premium' && <span className="hint" data-testid="premium-note">Premium captions send your cleaned audio to OpenAI, our sub-processor, and show text as it is spoken. Higher accuracy, higher price.</span>}
          </div>
        </fieldset>
        <fieldset className="field" disabled={state === 'starting'}>
          <legend className="label">Noise reduction strength</legend>
          <select id="live-str" className="select" aria-label="Noise reduction strength" value={strength} onChange={(e) => { setStrength(e.target.value); session.current?.setAtten(Number(e.target.value)); }} disabled={!denoise && !live}>
            {STRENGTH.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
          <label htmlFor="live-lang" className="label">Caption language</label>
          <select id="live-lang" className="select" value={language} onChange={(e) => setLanguage(e.target.value)} disabled={busy}>
            {LANGUAGES.map(([c, n]) => <option key={c} value={c}>{n}</option>)}
          </select>
        </fieldset>
        <fieldset className="field">
          <legend className="label">You hear</legend>
          <div className="seg" role="radiogroup" aria-label="Monitoring">
            {[['off', 'Nothing'], ['original', 'Original'], ['processed', 'Processed']].map(([v, l]) => (
              <label key={v}>
                <input type="radio" name="monitor" value={v} checked={monitor === v} onChange={() => { setMonitor(v); session.current?.setMonitor(v); }} />
                <span>{l}</span>
              </label>
            ))}
          </div>
          <span className="hint">Switch while you talk to compare.</span>
        </fieldset>
      </div>

      <div className="row">
        {!live && state !== 'stopping' && <button type="button" className="btn btn-primary btn-lg" onClick={start} disabled={busy || !mediaOk || (!denoise && !captions) || !priceKnown || !offered} data-testid="live-start">{state === 'starting' ? 'Connecting…' : 'Start test'}</button>}
        {(live || state === 'stopping') && <button type="button" className="btn btn-danger btn-lg" onClick={stop} disabled={state === 'stopping'} data-testid="live-stop">{state === 'stopping' ? 'Stopping…' : 'Stop'}</button>}
        {live && <span className="badge badge-ok" role="status">Live</span>}
      </div>

      {error && <div className="banner banner-error" role="alert" data-testid="live-error">{error} {/balance|credit|reserves/i.test(error) && <Link to="/app/billing">Add credit</Link>}</div>}
      {warning && <div className="banner banner-warn" role="alert">{warning.text} {warning.credit && <Link to="/app/billing">Add credit</Link>}</div>}
      {stopped && <div className="banner banner-info" role="status">{stopped}</div>}

      {(live || stats) && (
        <div className="grid grid-3" data-testid="live-stats" data-rtt={stats ? stats.rttAvg.toFixed(1) : ''} data-recv={stats?.recv ?? 0} data-sent={stats?.sent ?? 0}>
          <div className="stack">
            <b>Round trip</b>
            <div className="big-number">{stats ? `${Math.round(stats.rttAvg)} ms` : '…'}</div>
            <span className="small muted">+ {info ? `${Math.round(info.modelLatencyMs)} ms` : '30 ms'} model delay. Playback buffer {stats ? Math.round(stats.bufferMs) : 0} ms{stats?.underruns ? `, ${stats.underruns} gaps` : ''}.</span>
          </div>
          <div className="stack">
            <b id="in-l">Microphone level</b>
            <meter className="level" min={0} max={1} value={stats?.inLevel ?? 0} aria-labelledby="in-l" />
            <b id="out-l">Processed level</b>
            <meter className="level" min={0} max={1} value={stats?.outLevel ?? 0} aria-labelledby="out-l" />
          </div>
          <div className="stack">
            <b>Session</b>
            <div className="small muted">
              {stats ? `${clock(stats.elapsed)} · ${stats.sent} frames sent, ${stats.recv} received` : ''}
              {info && stats ? <><br />Cost so far about {moneyUsd((stats.elapsed / 60) * info.usdPerMinute)} at {rate(info.usdPerMinute)} per minute</> : null}
            </div>
          </div>
        </div>
      )}

      {captions && (
        <div className="stack">
          <b id="cap-h">Live captions</b>
          <div className="caption-log" ref={logRef} role="log" aria-live="polite" aria-labelledby="cap-h" data-testid="captions">
            {lines.length === 0 && <p className="muted">{live ? (captions === 'premium' ? 'Speak, and captions appear here as you talk.' : 'Speak, and captions appear here a moment after each pause.') : 'Captions appear here while the test runs.'}</p>}
            {lines.map((l) => (l.final
              ? <p key={l.key} data-testid="caption-line">{l.text}</p>
              : <p key={l.key} className="partial" aria-hidden="true" data-testid="caption-partial">{l.text}</p>))}
          </div>
        </div>
      )}
    </section>
  );
}
