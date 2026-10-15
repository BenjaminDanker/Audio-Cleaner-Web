import { lazy } from 'react';

// After we publish a new version the old hashed files are gone, so a tab that was opened before the update can fail
// to load the next page. Such a load is tried once more, then the page is reloaded (which fetches the new version).
// To be safe against a reload loop (the files really are missing), the reload happens once: the flag is cleared again
// only after a page loaded successfully.
const FLAG = 'acw-chunk-reload';

function flag(value) {
  try {
    if (value === undefined) return sessionStorage.getItem(FLAG) === '1';
    if (value) sessionStorage.setItem(FLAG, '1');
    else sessionStorage.removeItem(FLAG);
  } catch { /* storage blocked: no reload guard, so no automatic reload either */ }
  return value;
}

const never = () => new Promise(() => {}); // keeps the loading spinner up while the page reloads

/** Like React.lazy, but recovers from a chunk that cannot be fetched. */
export function lazyRetry(factory) {
  return lazy(async () => {
    try {
      const mod = await factory().catch(() => new Promise((resolve) => setTimeout(resolve, 600)).then(factory));
      flag(false);
      return mod;
    } catch (err) {
      let canReload = false;
      try {
        canReload = !flag();
        if (canReload) flag(true);
      } catch { /* ignore */ }
      if (canReload) {
        window.location.reload();
        return never();
      }
      throw err;
    }
  });
}
