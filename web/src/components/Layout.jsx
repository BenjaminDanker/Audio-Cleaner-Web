import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { apiOffline, checkHealth } from '../api.js';
import { useAuth } from '../auth.jsx';
import { balanceMoney } from '../format.js';
import { loadPricing } from '../hooks.js';
import { forgetReturn } from '../returnTo.js';
import { SITE } from '../site.js';
import ErrorBoundary from './ErrorBoundary.jsx';
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
  const shown = user.displayName || user.email;
  const initial = (shown || '?').trim().charAt(0).toUpperCase();
  return (
    <details className="menu" ref={ref}>
      <summary aria-label={`${shown}, account menu`}>
        <span className="avatar" aria-hidden="true">{initial}</span>
        <span className="email">{shown}</span>
      </summary>
      <div className="menu-panel">
        <div className="menu-email">{user.email}</div>
        <Link to="/app" onClick={close}>Cleanup</Link>
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
  const { ready, known, signedIn, balance } = useAuth();
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
          <NavLink to="/app" end>Cleanup</NavLink>
          <NavLink to="/app/live">Live</NavLink>
          <Link to={{ pathname: '/', hash: '#pricing' }}>Pricing</Link>
        </nav>
        <div className="header-right">
          {ready && signedIn && balance && (
            <Link to="/app/billing" className="balance-chip" title="Available balance" aria-label={`Available balance ${balanceMoney(balance.available_micros)}. Open billing.`}>
              {balanceMoney(balance.available_micros)}
            </Link>
          )}
          {ready && signedIn && <AccountMenu />}
          {ready && known && !signedIn && <Link to="/login" className="btn btn-primary btn-sm">Sign in</Link>}
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

/** Site-wide notice while the server is unreachable (sign-in, files and live all need it). */
function OfflineBanner() {
  const offline = useSyncExternalStore(apiOffline.subscribe, apiOffline.get);
  const [dismissed, setDismissed] = useState(false);
  useEffect(() => { if (!offline) setDismissed(false); }, [offline]); // a later break shows the notice again
  useRecovery(offline);
  if (!offline || dismissed) return null;
  return (
    <div className="offline-banner" role="status" data-testid="api-offline">
      <div className="container row-between">
        <span>{SITE.name} is taking a short break. Signing in and everything that needs an account will be back shortly.</span>
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
          <h2 className="footer-h">Product</h2>
          <ul>
            <li><Link to="/app">Cleanup, subtitles</Link></li>
            <li><Link to="/app/live">Live for streamers</Link></li>
            <li><Link to={{ pathname: '/', hash: '#pricing' }}>Pricing</Link></li>
          </ul>
        </div>
        <div>
          <h2 className="footer-h">Legal</h2>
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

// Shown only to someone whose session began before the Terms last changed (signing in records agreement, so new
// accounts never see it). Using the service records agreement to the new version, and the notice goes away.
function TermsNotice() {
  const { user } = useAuth();
  const [hidden, setHidden] = useState(false);
  if (!user || user.tosAccepted || hidden) return null;
  return (
    <div className="container" data-testid="terms-notice">
      <div className="banner banner-info row-between" role="status">
        <span>We&rsquo;ve updated our <Link to="/terms">Terms of Service</Link>. Using the service from now on means you agree to the new version.</span>
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => setHidden(true)}>OK</button>
      </div>
    </div>
  );
}

// Scrollable tables (wide ones on a phone) must be reachable by keyboard and have a name. Done in one place, for every
// page, because the wrapper is drawn by several of them: the name is the heading just above it.
function useScrollableTables() {
  useEffect(() => {
    const fix = () => {
      for (const w of document.querySelectorAll('.table-wrap:not([data-scroll-fixed])')) {
        w.setAttribute('data-scroll-fixed', '');
        w.setAttribute('tabindex', '0');
        w.setAttribute('role', 'region');
        const prev = w.previousElementSibling;
        const label = prev && /^H[1-6]$/.test(prev.tagName) ? prev.textContent.trim() : '';
        w.setAttribute('aria-label', label ? `${label} (table)` : 'Table');
      }
    };
    fix();
    let queued = 0;
    const mo = new MutationObserver(() => {
      if (!queued) queued = requestAnimationFrame(() => { queued = 0; fix(); }); // at most once per frame
    });
    mo.observe(document.body, { childList: true, subtree: true });
    return () => { mo.disconnect(); if (queued) cancelAnimationFrame(queued); };
  }, []);
}

export default function Layout() {
  const loc = useLocation();
  const main = useRef(null);
  const first = useRef(true);
  useScrollableTables();
  // A remembered "go here after signing in" only lives while the visitor stays on the sign-in path.
  useEffect(() => {
    if (!/^\/(login|app)(\/|$)/.test(loc.pathname)) forgetReturn();
  }, [loc.pathname]);
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
      <TermsNotice />
      <main id="main" ref={main} tabIndex={-1}>
        <ErrorBoundary resetKey={loc.pathname}>
          <Outlet />
        </ErrorBoundary>
      </main>
      <Footer />
    </>
  );
}
