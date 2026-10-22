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

/** The words for a page that still works with the server down: the free tools run in the browser, the paid options need our side. */
export const PAID_OFFLINE = 'The paid options are unavailable right now. The free ones still work.';

/** A short notice for the pages whose free tools run without the server (the editor, Live): shown instead of the calm full-page panel. */
export function PaidOffline() {
  return <div className="banner banner-warn" role="status" data-testid="paid-offline">{PAID_OFFLINE}</div>;
}

/** Wraps pages that cannot work without the server (Billing, Account, sign-in, a job on our side). The free tools do not use it. */
export default function NeedsServer({ children }) {
  const offline = useSyncExternalStore(apiOffline.subscribe, apiOffline.get);
  return offline ? <MaintenancePanel /> : children;
}
