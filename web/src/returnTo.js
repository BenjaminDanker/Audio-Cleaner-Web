// Where to go after signing in. Sign-in can leave the site (Google) and comes back to /app, so the page the visitor
// wanted is kept in this tab's sessionStorage. Only same-site paths are accepted.
const KEY = 'acw-return-to';

const safe = (path) => typeof path === 'string' && path.startsWith('/') && !path.startsWith('//') && !path.startsWith('/login');

export function rememberReturn(path) {
  try {
    if (safe(path)) sessionStorage.setItem(KEY, path);
  } catch { /* storage blocked: land on the default page */ }
}

/** The remembered path (once), or null. */
export function takeReturn() {
  try {
    const path = sessionStorage.getItem(KEY);
    sessionStorage.removeItem(KEY);
    return safe(path) ? path : null;
  } catch {
    return null;
  }
}
