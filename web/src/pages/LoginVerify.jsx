import { useEffect, useState } from 'react';
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

export default function LoginVerify() {
  usePageTitle('Signing in');
  const navigate = useNavigate();
  const { refresh } = useAuth();
  const [error, setError] = useState('');

  useEffect(() => {
    if (!inflight) {
      const token = takeToken();
      inflight = token
        ? api('auth/email/verify', { method: 'POST', body: { token }, redirectOn401: false }).then(() => ({ ok: true }), (e) => ({ ok: false, message: e.detail }))
        : Promise.resolve({ ok: false, message: 'That sign-in link looks cut off. Ask for a new one.' });
    }
    let alive = true;
    inflight.then(async (r) => {
      if (!alive) return;
      if (r.ok) {
        await refresh();
        navigate(takeReturn() || '/app', { replace: true });
      } else {
        setError(r.message);
      }
    });
    return () => { alive = false; };
  }, [navigate, refresh]);

  return (
    <div className="container page">
      <div className="card login-card stack center">
        {!error && <p role="status"><span className="spinner" aria-hidden="true" /> Signing you in…</p>}
        {error && (
          <>
            <div className="banner banner-error" role="alert">{error}</div>
            <Link to="/login" className="btn btn-primary" onClick={() => { inflight = null; }}>Back to sign in</Link>
          </>
        )}
      </div>
    </div>
  );
}
