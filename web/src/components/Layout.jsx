import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { apiOffline, checkHealth } from '../api.js';
import { useAuth } from '../auth.jsx';
import { balanceMoney } from '../format.js';
import { loadPricing } from '../hooks.js';
import { SITE } from '../site.js';
import { IconMenu, Logo } from './Icons.jsx';

function AccountMenu() {
  const { user, logout } = useAuth();
  const ref = useRef(null);
  const close = () => ref.current?.removeAttribute('open');
  useEffect(() => {
    const onDoc = (e) => {
      if (ref.current?.open && !ref.current.contains(e.target)) close();
    };
    const onKey = (e) => {
      if (e.key === 'Escape' && ref.current?.open) {
        close();
        ref.current.querySelector('summary')?.focus();
      }
    };
    document.addEventListener('click', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('click', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, []);
  const initial = (user.displayName || user.email || '?').trim().charAt(0).toUpperCase();
  return (
    <details className="menu" ref={ref}>
      <summary aria-label="Account menu">
        <span className="avatar" aria-hidden="true">{initial}</span>
        <span className="email">{user.displayName || user.email}</span>
      </summary>
      <div className="menu-panel">
        <div className="menu-email">{user.email}</div>
        <Link to="/app" onClick={close}>Studio</Link>
        <Link to="/app/billing" onClick={close}>Billing</Link>
        <Link to="/app/live" onClick={close}>Live &amp; keys</Link>
        <Link to="/app/account" onClick={close}>Your account</Link>
        <Link to="/terms" onClick={close}>Terms</Link>
        <button type="button" onClick={() => { close(); logout(); }}>Sign out</button>
      </div>
    </details>
  );
}

function Header() {
  const { ready, signedIn, balance } = useAuth();
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => setOpen(false), [loc.pathname, loc.hash]);
  return (
    <header className="site-header">
      <div className="container header-inner">
        <Link to="/" className="brand" aria-label={`${SITE.name} home`}>
          <Logo />
          <span>{SITE.name}</span>
        </Link>
        <nav className="nav" id="main-nav" data-open={open} aria-label="Main">
          <NavLink to="/free">Free denoise</NavLink>
          <NavLink to="/app" end>Studio</NavLink>
          <NavLink to="/transcribe">Transcribe</NavLink>
          <NavLink to="/app/live">Live</NavLink>
          <NavLink to="/desktop">Desktop app</NavLink>
          <Link to={{ pathname: '/', hash: '#pricing' }}>Pricing</Link>
        </nav>
        <div className="header-right">
          {ready && signedIn && balance && (
            <Link to="/app/billing" className="balance-chip" title="Available balance" aria-label={`Available balance ${balanceMoney(balance.available_micros)}. Open billing.`}>
              {balanceMoney(balance.available_micros)}
            </Link>
          )}
          {ready && signedIn && <AccountMenu />}
          {ready && !signedIn && <Link to="/login" className="btn btn-primary btn-sm">Sign in</Link>}
        </div>
        <button type="button" className="btn btn-secondary btn-sm nav-toggle" aria-expanded={open} aria-controls="main-nav" onClick={() => setOpen((v) => !v)}>
          <IconMenu width={18} height={18} /> <span className="sr-only">Menu</span>
        </button>
      </div>
    </header>
  );
}

/**
 * Re-checks /api/health every 30 s while the server is on its short break, and picks everything up again (who is signed
 * in, the price list) as soon as it answers. Mounted once, in the Layout, so the banner and the "right back" panel
 * share one timer.
 */
function useRecovery(offline) {
  const { refresh } = useAuth();
  useEffect(() => {
    if (!offline) return undefined;
    const id = setInterval(async () => {
      if (await checkHealth()) {
        refresh();
        loadPricing().catch(() => {});
      }
    }, 30000);
    return () => clearInterval(id);
  }, [offline, refresh]);
}

/** Site-wide notice while the paid side is unreachable. The free browser tool never depends on it. */
function OfflineBanner() {
  const offline = useSyncExternalStore(apiOffline.subscribe, apiOffline.get);
  const [dismissed, setDismissed] = useState(false);
  useEffect(() => { if (!offline) setDismissed(false); }, [offline]); // a later break shows the notice again
  useRecovery(offline);
  if (!offline || dismissed) return null;
  return (
    <div className="offline-banner" role="status" data-testid="api-offline">
      <div className="container row-between">
        <span>{SITE.name} is taking a short break. <Link to="/free">Free tools in your browser</Link> still work.</span>
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => setDismissed(true)}>Dismiss</button>
      </div>
    </div>
  );
}

function Footer() {
  const email = SITE.supportEmail.trim();
  const legalName = SITE.legalName.trim();
  return (
    <footer className="site-footer">
      <div className="container footer-grid">
        <div>
          <Link to="/" className="brand"><Logo size={26} /><span>{SITE.name}</span></Link>
          <p className="muted small mt0">A small tool for cleaning up voices. Your files stay yours.</p>
          {email && <p className="muted small">Contact: <a href={`mailto:${email}`}>{email}</a></p>}
          {legalName && <p className="muted small">&copy; {new Date().getFullYear()} {legalName}</p>}
        </div>
        <div>
          <h4>Product</h4>
          <ul>
            <li><Link to="/free">Free browser denoise</Link></li>
            <li><Link to="/app">Studio clean</Link></li>
            <li><Link to="/transcribe">Free transcription</Link></li>
            <li><Link to="/app/live">Live for streamers</Link></li>
            <li><Link to="/desktop">Desktop app</Link></li>
            <li><Link to={{ pathname: '/', hash: '#pricing' }}>Pricing</Link></li>
          </ul>
        </div>
        <div>
          <h4>Legal</h4>
          <ul>
            <li><Link to="/terms">Terms of Service</Link></li>
            <li><Link to="/privacy">Privacy Policy</Link></li>
            <li><Link to="/licenses">Open-source licences</Link></li>
          </ul>
        </div>
      </div>
    </footer>
  );
}

export default function Layout() {
  const loc = useLocation();
  const main = useRef(null);
  const first = useRef(true);
  // After in-app navigation move focus to the page (keyboard and screen-reader users), unless there is a hash
  // target. Not on the initial load, so Tab still starts at the skip link.
  useEffect(() => {
    if (first.current) { first.current = false; return; }
    if (loc.hash) return;
    window.scrollTo(0, 0);
    main.current?.focus({ preventScroll: true });
  }, [loc.pathname, loc.hash]);
  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <Header />
      <OfflineBanner />
      <main id="main" ref={main} tabIndex={-1}>
        <Outlet />
      </main>
      <Footer />
    </>
  );
}
