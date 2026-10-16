import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router';
import { api } from '../api.js';
import { dateTime, money } from '../format.js';

const SCOPE_LABEL = { stream: 'OBS plugin and live', jobs: 'File API for scripts' };
const MAX_CAP_DOLLARS = 10000;

/** "12.50" -> 1250 cents; '' -> null (no limit); anything else -> undefined (not a usable amount). */
function dollarsToCents(text) {
  const t = String(text).trim().replace(/^\$/, '');
  if (t === '') return null;
  if (!/^\d+(\.\d{1,2})?$/.test(t)) return undefined;
  const cents = Math.round(parseFloat(t) * 100);
  return cents >= 1 && cents <= MAX_CAP_DOLLARS * 100 ? cents : undefined;
}

const CAP_PROBLEM = `Use an amount from $0.01 to $${MAX_CAP_DOLLARS.toLocaleString('en-US')}, like 5 or 12.50, or leave it empty for no limit.`;

/** The API keys card: create and revoke keys, give each a monthly spending limit, and the secret that signs finish notifications. */
export default function ApiKeys() {
  const [keys, setKeys] = useState(null);
  const [error, setError] = useState('');
  const [name, setName] = useState('OBS on my PC');
  const [scopes, setScopes] = useState({ stream: true, jobs: false });
  const [cap, setCap] = useState('');
  const [busy, setBusy] = useState(false);
  const [fresh, setFresh] = useState(null);
  const [copied, setCopied] = useState(false);
  const [confirm, setConfirm] = useState(null);
  const [editing, setEditing] = useState(null); // { id, text } while a key's limit is being changed
  const [editError, setEditError] = useState('');

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
    const cents = dollarsToCents(cap);
    if (cents === undefined) { setError(CAP_PROBLEM); return; }
    setBusy(true); setError(''); setFresh(null); setCopied(false);
    try {
      const body = { name: name.trim(), scopes: Object.keys(scopes).filter((s) => scopes[s]) };
      if (cents !== null) body.monthlyCapCents = cents;
      const r = await api('keys', { method: 'POST', body });
      setFresh(r);
      setCap('');
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

  async function saveLimit(id, cents) {
    setBusy(true); setEditError('');
    try {
      await api(`keys/${id}`, { method: 'PATCH', body: { monthlyCapCents: cents } });
      setEditing(null);
      await load();
    } catch (err) {
      setEditError(err.detail);
    } finally {
      setBusy(false);
    }
  }

  function submitLimit(e, id) {
    e.preventDefault();
    const cents = dollarsToCents(editing.text);
    if (cents === undefined) { setEditError(CAP_PROBLEM); return; }
    saveLimit(id, cents);
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
      <p className="muted mb0 small">Keys let programs (like the OBS plugin or your own scripts) use your balance, so treat them like passwords. You can have up to 10. Writing your own? See the <Link to="/developers">developer guide</Link>.</p>
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
        <div className="field">
          <label htmlFor="key-cap">Spending limit per month (optional)</label>
          <div className="row">
            <span aria-hidden="true">$</span>
            <input id="key-cap" className="input input-money" inputMode="decimal" placeholder="No limit" value={cap} onChange={(e) => setCap(e.target.value)} aria-describedby="key-cap-hint" />
          </div>
          <span id="key-cap-hint" className="hint">The most this key can spend in a calendar month. When it is used up, this key is refused until the month starts over; your balance and your other keys are not affected.</span>
        </div>
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
                <KeyUsage k={k} editing={editing?.id === k.id ? editing : null} setEditing={setEditing} editError={editing?.id === k.id ? editError : ''} busy={busy}
                  onSubmit={(e) => submitLimit(e, k.id)} onRemove={() => saveLimit(k.id, null)} onStart={() => { setEditError(''); setEditing({ id: k.id, text: k.monthlyCapCents ? String(k.monthlyCapCents / 100) : '' }); }} />
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
      <NotificationSecret />
    </section>
  );
}

/** What this key has spent this month, its limit, and the way to change it. */
function KeyUsage({ k, editing, setEditing, editError, busy, onSubmit, onRemove, onStart }) {
  const capMicros = k.monthlyCapCents == null ? null : k.monthlyCapCents * 10000;
  const used = k.usedThisMonthMicros || 0;
  const reached = capMicros != null && used >= capMicros;
  const resets = k.capResetsAt ? new Date(k.capResetsAt * 1000).toLocaleDateString(undefined, { dateStyle: 'medium' }) : '';
  return (
    <div className="small" data-testid="key-usage">
      {editing ? (
        <form className="cap-form" onSubmit={onSubmit}>
          <div className="field">
            <label htmlFor={`cap-${k.id}`}>Spending limit per month</label>
            <div className="row">
              <span aria-hidden="true">$</span>
              <input id={`cap-${k.id}`} className="input" inputMode="decimal" placeholder="No limit" value={editing.text} onChange={(e) => setEditing({ id: k.id, text: e.target.value })} />
            </div>
          </div>
          <button type="submit" className="btn btn-primary btn-sm" disabled={busy}>Save</button>
          {k.monthlyCapCents != null && <button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={onRemove}>Remove limit</button>}
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => setEditing(null)}>Cancel</button>
        </form>
      ) : (
        <span className={reached ? 'neg' : 'muted'}>
          {capMicros == null
            ? `Used ${money(used)} this month · no spending limit. `
            : reached
              ? `Limit reached: ${money(used)} of ${money(capMicros)} used this month. Starts over ${resets}. `
              : `Used ${money(used)} of ${money(capMicros)} this month. `}
          <button type="button" className="btn btn-ghost btn-sm" onClick={onStart} aria-label={`Change spending limit of key ${k.name}`}>{capMicros == null ? 'Set a limit' : 'Change limit'}</button>
        </span>
      )}
      {editError && <div className="banner banner-error" role="alert">{editError}</div>}
    </div>
  );
}

/** The account's secret for signing finish notifications (callbackUrl): shown on request, replaceable. */
function NotificationSecret() {
  const [secret, setSecret] = useState(undefined); // undefined: loading, null: none yet, string: the secret
  const [shown, setShown] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let live = true;
    api('keys/webhook-secret').then((r) => { if (live) setSecret(r.secret); }).catch((e) => { if (live) { setSecret(null); setError(e.detail); } });
    return () => { live = false; };
  }, []);

  async function make() {
    setBusy(true); setError(''); setCopied(false);
    try {
      const r = await api('keys/webhook-secret', { method: 'POST' });
      setSecret(r.secret); setShown(true); setConfirm(false);
    } catch (e) {
      setError(e.detail);
    } finally {
      setBusy(false);
    }
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(secret);
      setCopied(true);
    } catch {
      setError('Copying didn\'t work. Select the secret and copy it by hand.');
    }
  }

  return (
    <div className="key-extra" data-testid="notify-secret">
      <h3 className="mb0">Finish notifications</h3>
      <p className="small muted mb0">Give a <code>callbackUrl</code> when your script creates a job and we tell that address when the job is done. Every message is signed with this secret, so you can check it came from us. It is the same for all your keys. <Link to="/developers#callbacks">How it works</Link></p>
      {secret === undefined && <p role="status" className="muted small mb0">Loading…</p>}
      {secret === null && !error && <p className="small mb0">You don’t have a secret yet. It is made the first time you use a notification, or you can make it now.</p>}
      {secret && (
        <div className="secret-box">
          <code data-testid="notify-secret-value" aria-label="Signing secret">{shown ? secret : `${secret.slice(0, 6)}${'•'.repeat(12)}`}</code>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => setShown((v) => !v)} aria-pressed={shown}>{shown ? 'Hide' : 'Show'}</button>
          <button type="button" className="btn btn-secondary btn-sm" onClick={copy}>{copied ? 'Copied' : 'Copy'}</button>
        </div>
      )}
      {error && <div className="banner banner-error" role="alert">{error}</div>}
      {secret === null && <div><button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={make}>Make a secret</button></div>}
      {secret && !confirm && <div><button type="button" className="btn btn-danger-outline btn-sm" onClick={() => setConfirm(true)}>Replace secret</button></div>}
      {secret && confirm && (
        <div className="row">
          <span className="small">Replace it? Your scripts must start using the new one.</span>
          <button type="button" className="btn btn-danger btn-sm" disabled={busy} onClick={make}>Replace</button>
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => setConfirm(false)}>Keep</button>
        </div>
      )}
    </div>
  );
}
