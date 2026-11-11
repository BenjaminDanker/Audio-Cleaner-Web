import { useCallback, useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { api } from '../api.js';
import { useAuth } from '../auth.jsx';
import { usePageTitle } from '../hooks.js';
import { takeReturn } from '../returnTo.js';

// The token must be used exactly once even if React mounts this effect twice (StrictMode in dev).
let inflight = null;

function takeToken() {
  const m = /[#&]token=([^&]+)/.exec(window.location.hash || '');
  let token = null;
  if (m) {
    try {
      token = decodeURIComponent(m[1]);
    } catch { /* a link cut off in the middle of a %-code: treated like a link with no token */ }
  }
  // Remove the secret from the address bar and browser history right away.
  window.history.replaceState(null, '', window.location.pathname + window.location.search);
  return token;
}

// A link opened in another browser than the one that asked for it is not used up: the server names the address and waits for
// the person's yes (`confirm`). `dead`: the answer was about the link itself (invalid, used, expired), so only a new one helps.
const verify = (token, confirm) =>
  api('auth/email/verify', { method: 'POST', body: confirm ? { token, confirm: true } : { token }, redirectOn401: false }).then(
    () => ({ ok: true }),
    (e) => (e.code === 'confirm_sign_in' && e.body?.email ? { ok: false, ask: { token, email: e.body.email } } : { ok: false, message: e.detail, dead: e.status === 400 }),
  );

export default function LoginVerify() {
  usePageTitle('Signing in');
  const navigate = useNavigate();
  const { refresh } = useAuth();
  const [error, setError] = useState('');
  const [ask, setAsk] = useState(null); // { token, email }: "Sign in as …?"
  const [busy, setBusy] = useState(false);

  const finish = useCallback(async (r) => {
    if (r.ok) {
      await refresh();
      navigate(takeReturn() || '/app', { replace: true });
    } else if (r.ask) {
      setAsk(r.ask);
    } else {
      // (a yes that did not get through, or came too fast, leaves the link good: the question stays, to press again)
      if (r.dead) setAsk(null);
      setError(r.message);
    }
  }, [navigate, refresh]);

  useEffect(() => {
    if (!inflight) {
      const token = takeToken();
      inflight = token ? verify(token, false) : Promise.resolve({ ok: false, message: 'That sign-in link looks cut off. Ask for a new one.' });
    }
    let alive = true;
    inflight.then((r) => { if (alive) finish(r); });
    return () => { alive = false; };
  }, [finish]);

  const confirm = async () => {
    setBusy(true);
    setError('');
    const r = await verify(ask.token, true);
    setBusy(false);
    finish(r);
  };

  return (
    <div className="container page">
      <div className="card login-card stack center">
        {!error && !ask && <p role="status"><span className="spinner" aria-hidden="true" /> Signing you in…</p>}
        {ask && (
          <>
            <h1 className="mb0">Sign in as {ask.email}?</h1>
            <p className="muted mb0">This link was opened in a different browser or device than the one that asked for it. Go on only if this is your own address: what you upload here goes into that account.</p>
            {error && <div className="banner banner-error" role="alert">{error}</div>}
            <button type="button" className="btn btn-primary" onClick={confirm} disabled={busy}>
              {busy ? <><span className="spinner" aria-hidden="true" /> Signing you in…</> : 'Sign in'}
            </button>
            <Link to="/login" className="btn btn-secondary" onClick={() => { inflight = null; }}>That’s not my address</Link>
          </>
        )}
        {error && !ask && (
          <>
            <div className="banner banner-error" role="alert">{error}</div>
            <Link to="/login" className="btn btn-primary" onClick={() => { inflight = null; }}>Back to sign in</Link>
          </>
        )}
      </div>
    </div>
  );
}
