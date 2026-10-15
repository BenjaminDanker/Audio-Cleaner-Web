import { useEffect, useState } from 'react';
import { Link, Navigate, useSearchParams } from 'react-router-dom';
import { api, oauthStartUrl } from '../api.js';
import { useAuth } from '../auth.jsx';
import { IconGoogle, IconMail } from '../components/Icons.jsx';
import Turnstile from '../components/Turnstile.jsx';
import { usePageTitle } from '../hooks.js';

const ERRORS = {
  cancelled: 'You stopped the sign-in. Whenever you’re ready, you can try again.',
  state: 'We couldn’t verify that sign-in (it may have been opened in another browser). Please try again.',
  expired: 'That sign-in link ran out of time. Please try again.',
  provider: 'The sign-in service had a hiccup. Please try again in a moment.',
  suspended: 'This account is paused. Please get in touch and we’ll sort it out.',
};

export default function Login() {
  usePageTitle('Sign in');
  const { ready, signedIn } = useAuth();
  const [params] = useSearchParams();
  const errCode = params.get('error');
  const [providers, setProviders] = useState(null);
  const [loadErr, setLoadErr] = useState('');
  const [email, setEmail] = useState('');
  const [token, setToken] = useState(null);
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [resetKey, setResetKey] = useState(0);

  useEffect(() => {
    api('auth/providers', { redirectOn401: false }).then(setProviders).catch((e) => setLoadErr(e.detail));
  }, []);

  if (ready && signedIn) return <Navigate to="/app" replace />;

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError('');
    try {
      await api('auth/email', { method: 'POST', body: { email: email.trim(), turnstileToken: token || undefined }, redirectOn401: false });
      setSent(true);
    } catch (err) {
      setError(err.detail);
      setToken(null);
      setResetKey((k) => k + 1);
    } finally {
      setBusy(false);
    }
  }

  const none = providers && !providers.google && !providers.email;
  const needToken = providers?.turnstileSiteKey && !token;
  return (
    <div className="container page">
      <div className="card login-card stack">
        <div>
          <h1 className="mb0">Sign in</h1>
          <p className="muted mb0">One account for Studio, transcripts and live. New here? Signing in makes your account, it’s free. (The browser tool needs no account.)</p>
        </div>
        {errCode && <div className="banner banner-error" role="alert">{ERRORS[errCode] || 'Sign-in failed. Please try again.'}</div>}
        {loadErr && <div className="banner banner-error" role="alert">{loadErr}</div>}
        {!providers && !loadErr && <p role="status" className="muted"><span className="spinner" aria-hidden="true" /> Loading…</p>}
        {none && <div className="banner banner-warn">Sign-in isn’t switched on yet.</div>}

        {providers?.google && (
          <div className="stack">
            {providers.google && <a className="btn btn-oauth" href={oauthStartUrl('google')}><IconGoogle width={20} height={20} /> Continue with Google</a>}
          </div>
        )}
        {providers?.google && providers?.email && <div className="divider">or</div>}

        {providers?.email && !sent && (
          <form className="stack" onSubmit={submit} noValidate>
            <div className="field">
              <label htmlFor="email">Email address</label>
              <input id="email" className="input" type="email" autoComplete="email" inputMode="email" required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
              <span className="hint">We’ll email you a link that signs you in. No password.</span>
            </div>
            {providers.turnstileSiteKey && <Turnstile siteKey={providers.turnstileSiteKey} onToken={setToken} onError={setError} resetKey={resetKey} />}
            {error && <div className="banner banner-error" role="alert">{error}</div>}
            <button className="btn btn-primary" type="submit" disabled={busy || !email.includes('@') || needToken}>
              <IconMail width={18} height={18} /> {busy ? 'Sending…' : 'Email me a sign-in link'}
            </button>
          </form>
        )}
        {sent && (
          <div className="banner banner-success" role="status">
            <p><b>Check your inbox.</b> If {email} can sign in, a link is on its way. It works once and runs out in 15 minutes.</p>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => { setSent(false); setToken(null); setResetKey((k) => k + 1); }}>Use a different email</button>
          </div>
        )}
        <p className="small muted mb0">By signing in you agree to the <Link to="/terms">Terms</Link> and <Link to="/privacy">Privacy Policy</Link>.</p>
      </div>
    </div>
  );
}
