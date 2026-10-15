import { useState, useSyncExternalStore } from 'react';
import { apiOffline, checkHealth } from '../api.js';
import { useAuth } from '../auth.jsx';
import { loadPricing, usePageTitle } from '../hooks.js';
import { SITE } from '../site.js';

/** Calm full-page notice shown instead of pages that need the server while it is on its short break. */
export function MaintenancePanel() {
  usePageTitle('We’ll be right back');
  const { refresh } = useAuth();
  const [checking, setChecking] = useState(false);
  async function again() {
    setChecking(true);
    if (await checkHealth()) {
      refresh();
      loadPricing().catch(() => {});
    }
    setChecking(false);
  }
  return (
    <div className="container page">
      <div className="card maintenance stack center" data-testid="maintenance">
        <h1 className="mb0">We&rsquo;ll be right back</h1>
        <p className="lead muted mb0">{SITE.name} is taking a short break. This page will pick up by itself as soon as we&rsquo;re back.</p>
        <div className="row row-center">
          <button type="button" className="btn btn-primary" onClick={again} disabled={checking}>{checking ? 'Checking…' : 'Check now'}</button>
        </div>
      </div>
    </div>
  );
}

/** Wraps pages that cannot work without the server (Cleanup, Live, Billing, Account, sign-in). */
export default function NeedsServer({ children }) {
  const offline = useSyncExternalStore(apiOffline.subscribe, apiOffline.get);
  return offline ? <MaintenancePanel /> : children;
}
