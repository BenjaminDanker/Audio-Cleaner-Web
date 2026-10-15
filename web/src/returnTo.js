// Where to go after signing in. Sign-in can leave the site (Google) and comes back to /app, so the page the visitor
// wanted is kept in this tab's sessionStorage. Only same-site paths are accepted: a path that starts with "/", is not
// "//host", has no backslash (some browsers read "/\host" as "//host") or control characters, and is not a sign-in page.
const KEY = 'acw-return-to';

const safe = (path) => typeof path === 'string' && path.startsWith('/') && !path.startsWith('//') && !path.startsWith('/login')
  && !/[\\\u0000-\u001f\u007f]/.test(path);

export function rememberReturn(path) {
  try {
    if (safe(path)) sessionStorage.setItem(KEY, path);
  } catch { /* storage blocked: land on the default page */ }
}

/** The remembered path, or null. Does not clear it (safe to call while drawing a page); see forgetReturn. */
export function peekReturn() {
  try {
    const path = sessionStorage.getItem(KEY);
    return safe(path) ? path : null;
  } catch {
    return null;
  }
}

export function forgetReturn() {
  try {
    sessionStorage.removeItem(KEY);
  } catch { /* nothing to clear */ }
}

/** The remembered path (once), or null. */
export function takeReturn() {
  const path = peekReturn();
  forgetReturn();
  return path;
}

export const isSafeReturn = safe;
