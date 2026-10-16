import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router';
import { api, streamUrl } from '../api.js';
import { GoodToKnow } from '../alternatives.jsx';
import { useAuth } from '../auth.jsx';
import { IconBroadcast } from '../components/Icons.jsx';
import PluginInstall from '../components/PluginInstall.jsx';
import PriceTable from '../components/PriceTable.jsx';
import { balanceMoney, clock, dateTime, rate, moneyUsd, trialLeft } from '../format.js';
import { findRate, usePageTitle, usePricing } from '../hooks.js';
import { LANGUAGES } from '../jobs.js';
import { startLocalLive } from '../live/local.js';
import { applyCaption, startLive } from '../live/session.js';
import StrengthPicker from '../components/StrengthPicker.jsx';
import { strengthDb } from '../strengths.js';

export default function Live() {
  usePageTitle('Live & API keys');
  const { data: pricing } = usePricing();
  return (
    <div className="container page stack-lg">
      <header className="row">
        <IconBroadcast width={36} height={36} />
        <div>
          <h1 className="mb0">Live for streamers</h1>
          <p className="muted mb0">Clean mic audio and live captions for OBS. Your first two hours of live are free; after that you only pay while you’re live.</p>
        </div>
      </header>
      <LivePrices />
      <div className="callout-grid">
        <ApiKeys />
        <ObsSetup />
      </div>
      <LiveDemo pricing={pricing} />
    </div>
  );
}

function LivePrices() {
  return (
    <section className="card stack" aria-labelledby="lp-h">
      <h2 id="lp-h" className="mb0">What live costs</h2>
      <PriceTable only={['live']} testId="live-prices" showNote={false} />
      <p className="small muted mb0">Counted by the second of audio. When you start, we set aside about five minutes of credit and give back whatever you don&rsquo;t use. If you turn on both noise removal and captions, the two add up.</p>
    </section>
  );
}

function ObsSetup() {
  return (
    <section className="card stack" aria-labelledby="obs-h">
      <h2 id="obs-h" className="mb0">OBS plugin setup</h2>
      <ol className="ol-steps mb0">
        <li>Create an API key on this page with <b>OBS plugin and live</b> switched on, and copy it.</li>
        <li>Download and install the Audio Cleaner OBS plugin below. You need OBS 31 or newer; restart OBS afterwards.</li>
        <li>In OBS, add the <b>Audio Cleaner</b> filter to your microphone source and paste the key.</li>
        <li>Pick how much delay you can live with (80, 150 or 250 ms), turn on captions if you want them, and go live.</li>
      </ol>
      <PluginInstall />
      <p className="small muted mb0">Address for the plugin: <span className="mono">{streamUrl()}</span></p>
    </section>
  );
}

const LANGUAGE_NAME = Object.fromEntries(LANGUAGES.filter(([c]) => c));

/* ------------------------------------------------------------------ API keys */

const SCOPE_LABEL = { stream: 'OBS plugin and live', jobs: 'File API for scripts' };

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
      setError('Copying didn\'t work. Select the key and copy it by hand.');
    }
  }

  const noScope = !scopes.stream && !scopes.jobs;
  return (
    <section className="card stack" aria-labelledby="keys-h">
      <h2 id="keys-h" className="mb0">API keys</h2>
      <p className="muted mb0 small">Keys let programs (like the OBS plugin or your own scripts) use your balance, so treat them like passwords. You can have up to 10.</p>
      {fresh && (
        <div className="stack" data-testid="new-key">
          <div className="banner banner-success"><b>Here’s your new key.</b> Copy it now, because we only show it once.</div>
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
          <legend className="label">What can this key do?</legend>
          <div className="stack-sm">
            <label className="check">
              <input type="checkbox" checked={scopes.stream} onChange={(e) => setScopes({ ...scopes, stream: e.target.checked })} />
              <span><b>{SCOPE_LABEL.stream}</b><span className="hint block">Lets the OBS plugin and live captions use your balance while you stream.</span></span>
            </label>
            <label className="check">
              <input type="checkbox" checked={scopes.jobs} onChange={(e) => setScopes({ ...scopes, jobs: e.target.checked })} />
              <span><b>{SCOPE_LABEL.jobs}</b><span className="hint block">Lets your own scripts send files in for noise removal and subtitles, charged like the Cleanup page.</span></span>
            </label>
          </div>
        </fieldset>
        <div><button type="submit" className="btn btn-primary" disabled={busy || noScope || !name.trim()}>Create key</button></div>
      </form>
      {error && <div className="banner banner-error" role="alert">{error}</div>}
      {keys === null && !error && <p role="status" className="muted mb0">Loading…</p>}
      {keys && keys.length === 0 && <p className="muted mb0">No keys yet.</p>}
      {keys && keys.length > 0 && (
        <ul className="outputs" aria-label="Active keys" data-testid="keys">
          {keys.map((k) => (
            <li key={k.id}>
              <span>
                <span className="oname">{k.name}</span>{' '}
                <span className="mono muted">{k.prefix}…</span>
                <div className="small muted">{k.scopes.map((x) => SCOPE_LABEL[x] || x).join(', ')} · created {dateTime(k.createdAt)} · {k.lastUsedAt ? `last used ${dateTime(k.lastUsedAt)}` : 'never used'}</div>
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

const ERROR_TEXT = {
  insufficient_funds: null, // the server's message is clear
  tos: 'Please agree to the Terms of Service first.',
  bad_config: null, // the server's message says what to change
  trial_busy: null, // the server's message says what to do
  too_many_sessions: null,
  rate_limited: 'The connection was closed because this page sent too many messages. Please start again.',
};
const DENOISE_NAME = { light: 'standard noise removal' };
const CAPTION_NAME = { local: 'standard captions', premium: 'extra accurate captions' };

/**
 * Each option the server offers when live is full is a change to the setup that was asked for: keys it names
 * replace those values (`denoise: null` = no noise removal), everything else stays as it was.
 */
function applyOption(opt, was) {
  return {
    denoise: 'denoise' in opt ? opt.denoise || null : was.denoise,
    captions: 'captions' in opt ? opt.captions || '' : was.captions,
  };
}

/** "Try standard noise removal + extra accurate captions": one-click retry for an option the server says fits. */
function optionLabel(opt, was) {
  const next = applyOption(opt, was);
  const parts = [];
  if (next.denoise) parts.push(DENOISE_NAME[next.denoise] || 'noise removal');
  if (next.captions) parts.push(CAPTION_NAME[next.captions] || 'captions');
  return `Try ${parts.join(' + ')}${!next.denoise && was.denoise ? ' (without noise removal)' : ''}`;
}

/** One labelled group of controls in the test panel; all groups share a row and line up at the top. */
function Group({ id, title, children }) {
  return (
    <div className="opt-group" role="group" aria-labelledby={id} data-testid={`group-${id.replace(/^grp-/, '')}`}>
      <h3 id={id} className="opt-title">{title}</h3>
      {children}
    </div>
  );
}

const STOPPED_TEXT = {
  insufficient_funds: 'The session stopped because your balance ran out.',
  max_duration: 'The session reached its maximum length.',
  idle_timeout: 'The session ended because no sound reached us for a while. Start it again when you’re ready.',
  no_audio: 'The session ended because not enough sound reached us. Check your microphone and connection, then start again.',
  captions_unavailable: 'The session ended because extra-accurate captions aren’t available right now. You weren’t charged for them.',
  trial_over: 'Your free trial is used up, so the session stopped.',
};

function LiveDemo({ pricing }) {
  const { balance, refresh, trialLiveSecondsLeft } = useAuth();
  const [mode, setMode] = useState('free'); // 'free': everything in this browser | 'paid': on our servers, like the OBS plugin
  const [freeCaps, setFreeCaps] = useState(true);
  const [denoise, setDenoise] = useState(true);
  const [captions, setCaptions] = useState('local'); // '' | 'local' (standard, English only) | 'premium' (extra accurate, any language)
  const [language, setLanguage] = useState('');
  const [strength, setStrength] = useState('max');
  const [monitor, setMonitor] = useState('processed');
  const [state, setState] = useState('idle'); // idle | starting | live | stopping
  const [error, setError] = useState('');
  const [full, setFull] = useState(null); // {message, available:[{denoise?, captions?}], was:{denoise, captions}} when live is at capacity
  const [info, setInfo] = useState(null);
  const [stats, setStats] = useState(null);
  const [lines, setLines] = useState([]); // caption lines {key, id, text, final}; partials are updated in place by id
  const [targets, setTargets] = useState(['', '', '']); // translations (paid test)
  const [translations, setTranslations] = useState({}); // caption id -> {lang: text}
  const [warning, setWarning] = useState(null); // {text, credit}
  const [stopped, setStopped] = useState(null); // {text, credit}
  const session = useRef(null);
  const logRef = useRef(null);
  const free = mode === 'free';
  const showCaptions = free ? freeCaps : !!captions;

  // Price of exactly what is selected: stream_denoise/light + stream_captions/<level> from /api/billing/pricing.
  const den = findRate(pricing, 'stream_denoise', 'light');
  const capRates = { local: findRate(pricing, 'stream_captions', 'local'), premium: findRate(pricing, 'stream_captions', 'premium') };
  const cap = captions ? capRates[captions] : null;
  const perMin = (denoise && den ? den.usdPerMinute : 0) + (cap ? cap.usdPerMinute : 0);
  const priceKnown = !!pricing && (!denoise || !!den) && (!captions || !!cap);
  const offered = (!denoise || den?.available !== false) && (!captions || cap?.available !== false);
  const accurateListed = capRates.premium?.available !== false;
  const spoken = captions === 'premium' ? language : 'en';
  const liveTargets = captions ? targets.filter((t) => t && t !== spoken) : [];
  const onTrial = trialLiveSecondsLeft > 0 && captions !== 'premium' && liveTargets.length === 0; // the free trial covers Standard live only
  const translateRate = findRate(pricing, 'stream_translate', 'standard');
  const mediaOk = !!(navigator.mediaDevices?.getUserMedia && window.AudioWorklet && window.WebSocket);

  useEffect(() => () => session.current?.stop(), []);
  useEffect(() => { logRef.current?.scrollTo({ top: logRef.current.scrollHeight }); }, [lines]);

  const onEvent = useCallback((ev) => {
    if (ev.type === 'ready') setInfo(ev);
    else if (ev.type === 'caption') {
      setLines((l) => applyCaption(l, ev));
    } else if (ev.type === 'translation') {
      setTranslations((t) => ({ ...t, [ev.of]: { ...t[ev.of], [ev.lang]: ev.text } }));
    } else if (ev.type === 'warning') {
      if (ev.code === 'low_balance') setWarning({ text: `Your balance covers about ${Math.max(0, Math.round(ev.secondsLeft))} more seconds. Add credit to keep going.`, credit: true });
      else if (ev.code === 'trial_low') setWarning({ text: `Your free trial has about ${trialLeft(ev.secondsLeft)} of live left. Add credit to keep going after that.`, credit: true });
      else if (ev.code === 'trial_used_up_now_paid') setWarning({ text: 'Your free trial is used up. The session carries on and now uses your credit.', credit: false });
      else setWarning({ text: ev.message || 'A heads-up from the live service.', credit: false });
    } else if (ev.type === 'stopped') {
      setStopped({ text: STOPPED_TEXT[ev.reason] ?? 'The session stopped.', credit: ev.reason === 'insufficient_funds' || ev.reason === 'trial_over' });
    } else if (ev.type === 'error') {
      setError(ERROR_TEXT[ev.code] ?? ev.message ?? 'The live service ran into a problem.');
    } else if (ev.type === 'closed') {
      setState('idle');
      session.current = null;
      refresh();
    }
  }, [refresh]);

  /** cfg: {denoise: tier id | null, captions: '' | 'local' | 'premium'} (defaults to what is selected right now). */
  async function start(cfg = { denoise: denoise ? 'light' : null, captions }) {
    setError(''); setFull(null); setWarning(null); setStopped(null); setLines([]); setTranslations({}); setStats(null); setInfo(null);
    setState('starting');
    if (free) {
      try {
        session.current = await startLocalLive({
          denoise, captions: freeCaps, language: language || null, attenDb: strengthDb(strength), monitor, onEvent, onStats: setStats,
        });
        setState('live');
      } catch (e) {
        session.current = null;
        setState('idle');
        if (e?.name === 'NotAllowedError') setError('Microphone access was blocked. Allow it in your browser and try again.');
        else if (e?.name === 'NotFoundError') setError('We couldn’t find a microphone.');
        else setError(e?.message || 'Couldn’t start.');
      }
      return;
    }
    try {
      session.current = await startLive({
        denoise: cfg.denoise, captions: cfg.captions || null,
        language: cfg.captions === 'premium' ? language || null : null, // standard captions are English; the language only matters for extra accurate
        translate: cfg.captions ? liveTargets : [],
        attenDb: strengthDb(strength), monitor,
        onEvent, onStats: setStats,
      });
      setState('live');
    } catch (e) {
      session.current = null;
      setState('idle');
      if (e?.code === 'full') {
        // Live is at capacity right now. The server lists what it can still take; offer each as a one-click retry.
        setFull({
          message: e.message || 'Live is busy right now.',
          available: Array.isArray(e.available) ? e.available : [],
          was: { denoise: cfg.denoise || null, captions: cfg.captions || '' },
        });
      } else if (e?.name === 'NotAllowedError') setError('Microphone access was blocked. Allow it in your browser and try again.');
      else if (e?.name === 'NotFoundError') setError('We couldn’t find a microphone.');
      else setError(ERROR_TEXT[e?.code] ?? e.message ?? 'Couldn\'t start.');
    }
  }

  function tryOption(opt) {
    const cfg = applyOption(opt, full.was);
    setDenoise(!!cfg.denoise);
    setCaptions(cfg.captions);
    start(cfg);
  }

  function stop() {
    setState('stopping');
    session.current?.stop();
  }

  const live = state === 'live';
  const busy = state !== 'idle';
  const perHour = (r) => (r ? ` · ${moneyUsd(r.usdPerHour)} an hour` : '');
  return (
    <section className="card stack" aria-labelledby="demo-h">
      <h2 id="demo-h" className="mb0">Try it with your microphone</h2>
      <p className="muted mb0">Talk into your mic and hear the result as it happens.</p>
      {trialLiveSecondsLeft > 0 && <p className="mb0" data-testid="live-trial-left"><b>Free trial: {trialLeft(trialLiveSecondsLeft)} of live left.</b> <span className="muted">It covers noise removal and Standard captions.</span></p>}
      <div className="seg" role="radiogroup" aria-label="Which test" data-testid="live-mode">
        {[['free', 'Free, in your browser'], ['paid', 'Paid, done on our side']].map(([v, l]) => (
          <label key={v}>
            <input type="radio" name="live-mode" value={v} checked={mode === v} disabled={busy} onChange={() => setMode(v)} />
            <span>{l}</span>
          </label>
        ))}
      </div>
      {free ? (
        <>
          <div className="banner banner-info" data-testid="live-free-banner"><b>Free:</b> this test runs on your own device, nothing is sent anywhere and it uses no credit. <b>Please wear headphones</b>, or you’ll get feedback.</div>
          <p className="mb0" data-testid="live-plugin-note">The OBS plugin works like the paid test: the work happens on our side, so your computer stays free for your stream, and its captions get about 40% fewer words wrong than these free ones on clear English speech.</p>
        </>
      ) : (
        <>
          {onTrial
            ? <div className="banner banner-info" data-testid="live-trial-banner"><b>Covered by your free trial.</b> No credit is used until the trial runs out. <b>Please wear headphones</b>, or you’ll get feedback.</div>
            : <div className="banner banner-warn"><b>This uses credit.</b> Your balance: <b>{balanceMoney(balance?.available_micros ?? 0)}</b>. <b>Please wear headphones</b>, or you’ll get feedback.</div>}
          <p className="mb0" data-testid="live-free-hint">This is how the OBS plugin works. Want to try without spending credit? Switch to the free test above.</p>
        </>
      )}
      {!mediaOk && <div className="banner banner-error" role="alert">This browser can’t do what the live test needs (microphone, AudioWorklet, WebSocket). A current Chrome, Edge, Firefox or Safari will.</div>}

      <div className="opt-grid" data-testid="live-options">
        <Group id="grp-noise" title="Noise removal">
          <label className="check">
            <input type="checkbox" checked={denoise} disabled={busy} onChange={(e) => setDenoise(e.target.checked)} />
            <span>Remove noise from my voice<span className="hint block">{free ? 'Free, done on your device.' : den ? `${moneyUsd(den.usdPerHour)} an hour` : 'Takes out hiss, hum and room noise.'}</span></span>
          </label>
          <StrengthPicker
            name="live-str" value={strength} disabled={!denoise || state === 'starting'}
            onChange={(id) => { setStrength(id); session.current?.setAtten(strengthDb(id)); }}
          />
          <span className="hint">You can change it while you talk.</span>
          {!free && <span className="hint">Echo from your speakers is removed automatically. With headphones you don’t need it.</span>}
          {!free && denoise && <GoodToKnow k="liveDenoise" />}
        </Group>

        <Group id="grp-captions" title="Captions">
          {free ? (
            <>
              <div className="choice-stack" role="radiogroup" aria-labelledby="grp-captions">
                <label className="choice">
                  <input type="radio" name="live-free-captions" value="off" checked={!freeCaps} disabled={busy} onChange={() => setFreeCaps(false)} />
                  <span><b>Off</b><small>No captions</small></span>
                </label>
                <label className="choice">
                  <input type="radio" name="live-free-captions" value="on" checked={freeCaps} disabled={busy} onChange={() => setFreeCaps(true)} />
                  <span><b>On</b><small>Free, made on your device, in many languages</small></span>
                </label>
              </div>
              {freeCaps && (
                <>
                  <div className="field">
                    <label htmlFor="live-free-lang">Language you speak</label>
                    <select id="live-free-lang" className="select" value={language} onChange={(e) => setLanguage(e.target.value)} disabled={busy}>
                      {LANGUAGES.map(([c, n]) => <option key={c} value={c}>{n}</option>)}
                    </select>
                  </div>
                  <span className="hint" data-testid="free-captions-note">Each line shows after you pause. The first time, the caption maker downloads about 80 MB.</span>
                </>
              )}
            </>
          ) : (
          <>
          <div className="choice-stack" role="radiogroup" aria-labelledby="grp-captions">
            <label className="choice">
              <input type="radio" name="live-captions" value="" checked={captions === ''} disabled={busy} onChange={() => setCaptions('')} />
              <span><b>Off</b><small>No captions</small></span>
            </label>
            <label className="choice">
              <input type="radio" name="live-captions" value="local" checked={captions === 'local'} disabled={busy || capRates.local?.available === false} onChange={() => setCaptions('local')} />
              <span><b>Standard</b><small>English only{perHour(capRates.local)}</small></span>
            </label>
            {(!pricing || capRates.premium) && (
              <label className="choice">
                <input type="radio" name="live-captions" value="premium" checked={captions === 'premium'} disabled={busy || !accurateListed} onChange={() => setCaptions('premium')} />
                <span><b>Extra accurate</b><small>Works in any language{perHour(capRates.premium)}{accurateListed ? '' : ' · coming soon'}</small></span>
              </label>
            )}
          </div>
          {captions === 'premium' && (
            <>
              <div className="field">
                <label htmlFor="live-lang">Language you speak</label>
                <select id="live-lang" className="select" value={language} onChange={(e) => setLanguage(e.target.value)} disabled={busy}>
                  {LANGUAGES.map(([c, n]) => <option key={c} value={c}>{n}</option>)}
                </select>
              </div>
              <span className="hint" data-testid="premium-note">Extra accurate captions send your audio to a speech partner we work with, and show text as you speak. <Link to="/privacy">Privacy details</Link></span>
            </>
          )}
          {captions === 'local' && <span className="hint">Standard captions are English only. For another language, choose Extra accurate.</span>}
          {captions === 'local' && <GoodToKnow k="liveCaptions" />}
          {captions === 'premium' && <GoodToKnow k="liveAccurate" />}
          {captions && (
            <div className="field" data-testid="live-translate">
              <span className="label">Also translate into</span>
              {targets.map((t, i) => (
                <select
                  key={i} className="select" aria-label={`Live translation language ${i + 1}`} value={t} disabled={busy}
                  onChange={(e) => setTargets(targets.map((x, j) => (j === i ? e.target.value : x)))}
                >
                  <option value="">{i === 0 ? 'No translation' : 'No other language'}</option>
                  {LANGUAGES.filter(([c]) => c && c !== spoken && (c === t || !targets.includes(c))).map(([c, n]) => <option key={c} value={c}>{n}</option>)}
                </select>
              ))}
              {liveTargets.length > 0 && <GoodToKnow k="liveTranslations" />}
              <span className="hint">Each line shows under the caption a moment later{translateRate ? `. ${moneyUsd(translateRate.usdPerHour)} an hour per language, counted by the speech translated` : ''}{onTrial || !liveTargets.length ? '' : '. Not covered by the free trial'}.</span>
            </div>
          )}
          </>
          )}
        </Group>

        <Group id="grp-hear" title="You hear">
          <div className="seg" role="radiogroup" aria-label="What you hear">
            {[['off', 'Nothing'], ['original', 'Original'], ['processed', 'Cleaned']].map(([v, l]) => (
              <label key={v}>
                <input type="radio" name="monitor" value={v} checked={monitor === v} onChange={() => { setMonitor(v); session.current?.setMonitor(v); }} />
                <span>{l}</span>
              </label>
            ))}
          </div>
          <span className="hint">Switch while you talk to compare your mic with the cleaned sound.</span>
        </Group>
      </div>

      {free ? (
        <p className="mb0" data-testid="live-price" data-usd-per-minute="0"><b>Cost: free.</b> <span className="muted">Nothing is sent to us, so nothing is charged.</span></p>
      ) : (
      <p className="mb0" data-testid="live-price" data-usd-per-minute={priceKnown ? perMin : ''} aria-live="polite">
        {!pricing && <span className="muted">Loading prices…</span>}
        {pricing && perMin <= 0 && priceKnown && <span className="muted">Turn on noise removal or captions to start.</span>}
        {pricing && !priceKnown && <span className="muted">We don’t have a price for that combination.</span>}
        {pricing && priceKnown && perMin > 0 && onTrial && (
          <>
            <b>Cost: free while your trial lasts</b> <span className="muted">({trialLeft(trialLiveSecondsLeft)} left). After that it’s {rate(perMin)} a minute from your credit.</span>
          </>
        )}
        {pricing && priceKnown && perMin > 0 && !onTrial && (
          <>
            <b>Cost: {rate(perMin)} a minute</b> <span className="muted">({moneyUsd(perMin * 60)} an hour).
            {denoise && den && cap ? ` Noise removal ${rate(den.usdPerMinute)} + ${captions === 'premium' ? 'extra accurate' : 'standard'} captions ${rate(cap.usdPerMinute)} a minute.` : ''}
            {' '}We set aside about five minutes ({moneyUsd(perMin * 5)}) when you start and give back whatever you don’t use.</span>
          </>
        )}
      </p>
      )}

      <div className="row">
        {!live && state !== 'stopping' && <button type="button" className="btn btn-primary btn-lg" onClick={() => start()} disabled={busy || !mediaOk || (free ? !denoise && !freeCaps : (!denoise && !captions) || !priceKnown || !offered)} data-testid="live-start">{state === 'starting' ? 'Connecting…' : 'Start test'}</button>}
        {(live || state === 'stopping') && <button type="button" className="btn btn-danger btn-lg" onClick={stop} disabled={state === 'stopping'} data-testid="live-stop">{state === 'stopping' ? 'Stopping…' : 'Stop'}</button>}
        {live && <span className="badge badge-ok" role="status">Live</span>}
      </div>

      {full && (
        <div className="banner banner-warn stack" role="alert" data-testid="live-full">
          <p className="mb0">{full.message}</p>
          {full.available.length > 0 && (
            <div className="row">
              {full.available.map((opt) => (
                <button key={JSON.stringify(opt)} type="button" className="btn btn-secondary btn-sm" onClick={() => tryOption(opt)} data-testid="live-full-option">{optionLabel(opt, full.was)}</button>
              ))}
            </div>
          )}
        </div>
      )}
      {error && <div className="banner banner-error" role="alert" data-testid="live-error">{error} {/balance|credit|reserves|set aside/i.test(error) && <Link to="/app/billing">Add credit</Link>}</div>}
      {warning && <div className="banner banner-warn" role="alert">{warning.text} {warning.credit && <Link to="/app/billing">Add credit</Link>}</div>}
      {stopped && <div className="banner banner-info" role="status">{stopped.text} {stopped.credit && <Link to="/app/billing">Add credit</Link>}</div>}

      {(live || stats) && (
        <div className="grid grid-3" data-testid="live-stats" data-rtt={stats ? stats.rttAvg.toFixed(1) : ''} data-recv={stats?.recv ?? 0} data-sent={stats?.sent ?? 0}>
          <div className="stack">
            <b>{free ? 'Processing delay' : 'Round trip'}</b>
            <div className="big-number">{stats ? `${Math.round(stats.rttAvg)} ms` : '…'}</div>
            <span className="small muted">+ {info ? `${Math.round(info.modelLatencyMs)} ms` : '30 ms'} {free ? 'model' : 'processing'} delay. Playback buffer {stats ? Math.round(stats.bufferMs) : 0} ms{stats?.underruns ? `, ${stats.underruns} gaps` : ''}.</span>
          </div>
          <div className="stack">
            <b id="in-l">Microphone level</b>
            <meter className="level" min={0} max={1} value={stats?.inLevel ?? 0} aria-labelledby="in-l" />
            <b id="out-l">Cleaned level</b>
            <meter className="level" min={0} max={1} value={stats?.outLevel ?? 0} aria-labelledby="out-l" />
          </div>
          <div className="stack">
            <b>Session</b>
            <div className="small muted">
              {stats ? `${clock(stats.elapsed)} · ${stats.sent} frames ${free ? 'in' : 'sent'}, ${stats.recv} ${free ? 'cleaned' : 'received'}` : ''}
              {info && stats && !info.local ? <><br />So far about {moneyUsd((stats.elapsed / 60) * info.usdPerMinute)} at {rate(info.usdPerMinute)} per minute</> : null}
            </div>
          </div>
        </div>
      )}

      {showCaptions && (
        <div className="stack">
          <b id="cap-h">Live captions</b>
          <div className="caption-log" ref={logRef} role="log" aria-live="polite" aria-labelledby="cap-h" data-testid="captions">
            {lines.length === 0 && <p className="muted">{live ? (!free && captions === 'premium' ? 'Say something and the captions show up here as you talk.' : 'Say something and the captions show up here as you talk and pause.') : 'Captions show up here while the test runs.'}</p>}
            {lines.map((l) => (
              // one node per line: a partial turns into its final line in place
              <div key={l.key}>
                {l.final
                  ? <p className="mb0" data-testid="caption-line">{l.text}</p>
                  : <p className="partial mb0" aria-hidden="true" data-testid="caption-partial">{l.text}</p>}
                {l.final && Object.entries(translations[String(l.id || '').replace(/^id:/, '')] || {}).map(([lang, text]) => (
                  <p key={lang} className="small muted mb0" lang={lang} data-testid="caption-translation">{LANGUAGE_NAME[lang] || lang}: {text}</p>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
