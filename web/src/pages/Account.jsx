import { useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router';
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
  const delAlert = useRef(null);

  const emailOk = !!user && confirm.trim().toLowerCase() === user.email.toLowerCase();

  // The reason for a refused deletion (for example 409 "Wait for your running jobs to finish first.") must not go unnoticed.
  useEffect(() => {
    if (!delErr || !delAlert.current) return;
    delAlert.current.scrollIntoView({ block: 'center' });
    delAlert.current.focus({ preventScroll: true });
  }, [delErr]);

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
      setMsg({ kind: 'success', text: 'Done. It’s saved as acw-account-export.json.' });
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
    } catch (err) {
      setDelErr(err.detail || 'We couldn’t delete the account. Please try again.');
      setBusy('');
      return;
    }
    // The session is gone on the server. Leave the signed-in area first (with a one-time notice for the landing page);
    // only then refresh the auth state, otherwise the auth guard on this page would bounce us to /login.
    navigate('/', { replace: true, state: { notice: 'account-deleted' } });
  }

  return (
    <div className="container narrow page stack-lg">
      <header>
        <h1 className="mb0">Account</h1>
        <p className="muted mb0">Your details, your data, and the way out.</p>
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
        <p className="mb0 muted">Download everything we keep about your account (profile, sign-in methods, sessions, key details, credit history, payments, job and live history) as a JSON file. Your audio isn’t part of it, because we only keep it for a short while: uploads and results are deleted 2 hours after a file is done.</p>
        <div><button type="button" className="btn btn-secondary" disabled={!!busy} onClick={download}>{busy === 'export' ? 'Preparing…' : 'Download my data'}</button></div>
        {msg && <div className={`banner banner-${msg.kind}`} role={msg.kind === 'error' ? 'alert' : 'status'}>{msg.text}</div>}
      </section>

      <section className="card stack danger-zone" aria-labelledby="del-h">
        <h2 id="del-h" className="mb0">Delete account</h2>
        <p className="mb0">This deletes your account for good: your sign-in methods, keys, saved card link and any results we still hold. Payment and credit records are kept without your name, because tax and dispute rules ask us to.</p>
        <div className="banner banner-warn">
          <p><b>Any credit you have left ({balanceMoney(balance?.balance_micros ?? 0)}) is lost</b> and can’t be brought back or refunded afterwards.</p>
          <p className="mb0">Want money back first? You can <Link to="/app/billing">ask for a refund of the unused part of a recent payment</Link> (the last 14 days) before you delete.</p>
        </div>
        <p className="small muted mb0">You can’t delete while a job is uploading or running. Let it finish, or cancel it on the Cleanup page.</p>
        <form className="stack" onSubmit={deleteAccount}>
          <div className="field">
            <label htmlFor="del-confirm">Type your email address ({user?.email}) to confirm</label>
            <input id="del-confirm" className="input" type="email" autoComplete="off" value={confirm} onChange={(e) => { setConfirm(e.target.value); setDelErr(''); }} placeholder={user?.email} />
          </div>
          {delErr && <div className="banner banner-error" role="alert" tabIndex={-1} ref={delAlert} data-testid="delete-error"><b>Your account is still here.</b> {delErr}</div>}
          <div><button type="submit" className="btn btn-danger" disabled={!emailOk || !!busy}>{busy === 'delete' ? 'Deleting…' : 'Delete my account'}</button></div>
        </form>
      </section>
    </div>
  );
}
