import { useEffect, useRef } from 'react';

const SRC = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
let loader = null;

function loadScript() {
  if (window.turnstile) return Promise.resolve(window.turnstile);
  if (!loader) {
    loader = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = SRC;
      s.async = true;
      s.defer = true;
      s.onload = () => resolve(window.turnstile);
      s.onerror = () => { loader = null; reject(new Error('Could not load the human check.')); };
      document.head.appendChild(s);
    });
  }
  return loader;
}

/** Cloudflare Turnstile widget (the only third-party script we allow, and only when a site key is configured). */
export default function Turnstile({ siteKey, onToken, onError, resetKey }) {
  const box = useRef(null);
  const widget = useRef(null);
  useEffect(() => {
    let alive = true;
    loadScript()
      .then((ts) => {
        if (!alive || !box.current) return;
        if (widget.current != null) ts.remove(widget.current);
        widget.current = ts.render(box.current, {
          sitekey: siteKey,
          callback: (t) => onToken(t),
          'expired-callback': () => onToken(null),
          'error-callback': () => { onToken(null); onError?.('The human check failed to load. Please reload the page.'); },
        });
      })
      .catch((e) => onError?.(e.message));
    return () => {
      alive = false;
      if (widget.current != null && window.turnstile) {
        try { window.turnstile.remove(widget.current); } catch { /* ignore */ }
        widget.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [siteKey, resetKey]);
  return <div ref={box} data-testid="turnstile" />;
}
