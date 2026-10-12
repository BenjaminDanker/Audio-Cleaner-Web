import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '../api.js';
import { useAuth } from '../auth.jsx';
import { saveBlob } from '../download.js';
import { balanceMoney, dateTime } from '../format.js';
import { usePageTitle } from '../hooks.js';

export default function Account() {
  usePageTitle('Account');
  const { user, balance, refresh, logout } = useAuth();
  const navigate = useNavigate();
  const [msg, setMsg] = useState(null);
  const [busy, setBusy] = useState('');
  const [confirm, setConfirm] = useState('');
  const [delErr, setDelErr] = useState('');

  const emailOk = !!user && confirm.trim().toLowerCase() === user.email.toLowerCase();

  async function signOutEverywhere() {
    setBusy('all'); setMsg(null);
    try {
      await api('auth/logout-all', { method: 'POST' });
      await refresh();
      navigate('/login', { replace: true });
    } catch (e) {
      setMsg({ kind: 'error', text: e.detail });
      setBusy('');
    }
  }

  async function download() {
    setBusy('export'); setMsg(null);
    try {
      const data = await api('account/export');
      saveBlob(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }), 'acw-account-export.json');
      setMsg({ kind: 'success', text: 'Your data was downloaded as acw-account-export.json.' });
    } catch (e) {
      setMsg({ kind: 'error', text: e.detail });
    } finally {
      setBusy('');
    }
  }

  async function deleteAccount(e) {
    e.preventDefault();
    if (!emailOk) return;
    setBusy('delete'); setDelErr('');
    try {
      await api('account/delete', { method: 'POST', body: { confirm: confirm.trim() } });
      await refresh();
      navigate('/', { replace: true });
    } catch (err) {
      setDelErr(err.detail);
      setBusy('');
    }
  }

  return (
    <div className="container narrow page stack-lg">
      <header>
        <h1 className="mb0">Account</h1>
        <p className="muted mb0">Your profile, your data, and the exit door.</p>
      </header>

      <section className="card stack" aria-labelledby="prof-h">
        <h2 id="prof-h" className="mb0">Profile</h2>
        <dl className="kv mb0">
          <dt>Email</dt><dd data-testid="acct-email">{user?.email}</dd>
          {user?.displayName && (<><dt>Name</dt><dd>{user.displayName}</dd></>)}
          <dt>Member since</dt><dd>{dateTime(user?.createdAt)}</dd>
          <dt>Terms accepted</dt><dd>{user?.tosAccepted ? 'Yes, current version' : 'Not yet'}</dd>
        </dl>
        <div className="row">
          <button type="button" className="btn btn-secondary" disabled={!!busy} onClick={logout}>Sign out</button>
          <button type="button" className="btn btn-secondary" disabled={!!busy} onClick={signOutEverywhere}>{busy === 'all' ? 'Signing out…' : 'Sign out everywhere'}</button>
        </div>
        <p className="small muted mb0">&ldquo;Everywhere&rdquo; ends every browser session of this account, including this one. API keys keep working until you revoke them on the <Link to="/app/live">Live page</Link>.</p>
      </section>

      <section className="card stack" aria-labelledby="data-h">
        <h2 id="data-h" className="mb0">Your data</h2>
        <p className="mb0 muted">Download everything we store about your account (profile, sign-in methods, sessions, API key metadata, ledger, payments, job and live-session history) as a JSON file. Your files are never part of it: uploads are deleted after processing and results after 24 hours.</p>
        <div><button type="button" className="btn btn-secondary" disabled={!!busy} onClick={download}>{busy === 'export' ? 'Preparing…' : 'Download my data'}</button></div>
        {msg && <div className={`banner banner-${msg.kind}`} role={msg.kind === 'error' ? 'alert' : 'status'}>{msg.text}</div>}
      </section>

      <section className="card stack danger-zone" aria-labelledby="del-h">
        <h2 id="del-h" className="mb0">Delete account</h2>
        <p className="mb0">This permanently deletes your account: your sign-in methods, API keys, saved card link and any stored results. Payment and ledger records are kept in anonymised form because tax and dispute rules require it.</p>
        <div className="banner banner-warn">
          <p><b>Your remaining credit ({balanceMoney(balance?.balance_micros ?? 0)}) is forfeited</b> and cannot be restored or refunded after deletion.</p>
          <p className="mb0">Want money back first? You can <Link to="/app/billing">request a refund of the unused part of recent payments</Link> (payments from the last 14 days) before you delete.</p>
        </div>
        <p className="small muted mb0">Deletion is blocked while jobs are uploading or running. Wait for them to finish or cancel them on the Studio page.</p>
        <form className="stack" onSubmit={deleteAccount}>
          <div className="field">
            <label htmlFor="del-confirm">Type your email address ({user?.email}) to confirm</label>
            <input id="del-confirm" className="input" type="email" autoComplete="off" value={confirm} onChange={(e) => setConfirm(e.target.value)} placeholder={user?.email} />
          </div>
          {delErr && <div className="banner banner-error" role="alert" data-testid="delete-error">{delErr}</div>}
          <div><button type="submit" className="btn btn-danger" disabled={!emailOk || !!busy}>{busy === 'delete' ? 'Deleting…' : 'Delete my account'}</button></div>
        </form>
      </section>
    </div>
  );
}
