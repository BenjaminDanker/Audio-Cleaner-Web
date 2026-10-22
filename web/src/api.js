// Small fetch/XHR wrapper for the Audio Cleaner HTTP API (docs/api.md).
// - same-origin cookie auth, JSON in/out
// - X-CSRF-Token on every non-GET request (token comes from GET /api/me)
// - 401 -> sign-in page; 403 CSRF -> refresh /api/me once and retry

export const BASE = import.meta.env.BASE_URL; // always ends with "/"
export const API = `${BASE}api/`;

let csrfToken = null;
let unauthorizedHandler = null;

export function setCsrfToken(token) {
  csrfToken = token || null;
}

export function onUnauthorized(fn) {
  unauthorizedHandler = fn;
}

// ---- API availability -------------------------------------------------------------------------------------
// When the home server is down, the front door answers every /api/* call with HTTP 503 {"detail":"maintenance"}
// (or 502/504 from a proxy, or the connection fails outright). All of these mean the same thing: "taking a short
// break". /api/me and /api/billing/pricing are always watched (every page asks for them); any other endpoint only
// counts when it says "maintenance" or answers 502/504, because a plain 503 elsewhere can be meaningful ("payments are
// not available yet"). Recovery is a separate poll of /api/health (see checkHealth / the Layout).
const downBy = { me: false, pricing: false, other: false };
const offlineListeners = new Set();
const DOWN_STATUS = new Set([502, 503, 504]);

function reportDown(key, down) {
  if (downBy[key] === down) return;
  downBy[key] = down;
  offlineListeners.forEach((fn) => fn());
}

/** For useSyncExternalStore: `apiOffline.subscribe`, `apiOffline.get`. */
export const apiOffline = {
  subscribe(fn) {
    offlineListeners.add(fn);
    return () => offlineListeners.delete(fn);
  },
  get: () => downBy.me || downBy.pricing || downBy.other,
};

/**
 * Asks /api/health whether the server is back. Returns true (and clears the "taking a break" state) when it answers
 * with a success status, false otherwise. Never throws.
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${API}health`, { credentials: 'same-origin', cache: 'no-store', headers: { Accept: 'application/json' } });
    if (!res.ok) return false;
    downBy.me = false;
    downBy.pricing = false;
    downBy.other = false;
    offlineListeners.forEach((fn) => fn());
    return true;
  } catch {
    return false;
  }
}

/** The code of the answer to starting a file job while the account already has the most it may have at once. */
export const TOO_MANY_JOBS = 'too_many_active_jobs';

export class ApiError extends Error {
  constructor(message, status, fields, retryAfter, code) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = message;
    this.fields = fields;
    this.code = code || null; // the server's error code (X-Error-Code / "code"), for the few errors the page treats specially
    this.retryAfter = retryAfter; // seconds the server asked us to wait (HTTP 429 with Retry-After), if it did
  }
}

/** "about 20 seconds", "about 12 minutes", "about 2 hours". */
export function waitText(seconds) {
  const s = Math.max(1, Math.round(seconds));
  if (s < 90) return `${s} ${s === 1 ? 'second' : 'seconds'}`;
  const m = Math.round(s / 60);
  if (m < 90) return `about ${m} minutes`;
  const h = Math.round(s / 3600);
  return `about ${h} ${h === 1 ? 'hour' : 'hours'}`;
}

/** The wait (seconds) from a Retry-After header: whole seconds, or an HTTP date. null when absent or unusable. */
function retryAfterSeconds(value) {
  if (!value) return null;
  const n = Number(value);
  if (Number.isFinite(n)) return n > 0 ? Math.min(n, 86400) : null;
  const t = Date.parse(value);
  return Number.isFinite(t) ? Math.min(Math.max(0, (t - Date.now()) / 1000), 86400) || null : null;
}

/** A "too many requests" answer says how long to wait, once, in plain words (the server's own message may already do so). */
function withWait(detail, seconds) {
  if (!seconds || /\b(try again|wait|minute|second|hour)/i.test(detail)) return detail;
  return `${detail.replace(/[.!\s]+$/, '')}. Please try again in ${waitText(seconds)}.`;
}

async function parse(res) {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return { detail: res.ok ? text : `Something went wrong on our side (${res.status}).` };
  }
}

function errorFrom(res, data) {
  let detail = data && typeof data.detail === 'string' ? data.detail : null;
  if (!detail) {
    if (res.status === 507) detail = 'We are a bit short on storage space right now, so new files cannot be taken. Please try again in a little while.';
    else detail = res.status >= 500 ? 'Something went wrong on our side. Please try again in a moment.' : `Request failed (${res.status}).`;
  }
  let wait = null;
  const code = (typeof data?.code === 'string' && data.code) || res.headers?.get?.('X-Error-Code') || null;
  // "too many files at once" says what to do by itself; a time to wait would only confuse it
  if (res.status === 429 && code !== TOO_MANY_JOBS) {
    wait = retryAfterSeconds(res.headers?.get?.('Retry-After'));
    if (wait) detail = withWait(detail, wait);
  }
  const err = new ApiError(detail, res.status, data?.fields, wait, code);
  err.body = data && typeof data === 'object' ? data : null; // the rest of the answer (e.g. the `needs` of 409 needs_uploads)
  return err;
}

export async function fetchMe() {
  let res;
  try {
    res = await fetch(`${API}me`, { credentials: 'same-origin', headers: { Accept: 'application/json' } });
  } catch (e) {
    if (e?.name !== 'AbortError') reportDown('me', true);
    throw e;
  }
  reportDown('me', DOWN_STATUS.has(res.status));
  const data = await parse(res);
  if (!res.ok) throw errorFrom(res, data);
  setCsrfToken(data?.csrfToken);
  return data;
}

/**
 * api('billing/pricing') / api('jobs', { method: 'POST', body })
 * Options: method, body (object), signal, redirectOn401 (default true).
 */
export async function api(path, { method = 'GET', body, signal, redirectOn401 = true } = {}, _retried = false) {
  const headers = { Accept: 'application/json' };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (method !== 'GET' && csrfToken) headers['X-CSRF-Token'] = csrfToken;
  let res;
  try {
    res = await fetch(`${API}${path}`, {
      method,
      credentials: 'same-origin',
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if (e?.name === 'AbortError') throw e;
    if (path === 'billing/pricing') reportDown('pricing', true);
    else reportDown('other', true);
    throw new ApiError('Could not reach Audio Cleaner. Check your connection and try again.', 0);
  }
  if (path === 'billing/pricing') reportDown('pricing', DOWN_STATUS.has(res.status));
  const data = await parse(res);
  if (path !== 'billing/pricing') {
    const maintenance = res.status === 502 || res.status === 504 || (res.status === 503 && data?.detail === 'maintenance');
    reportDown('other', maintenance);
  }
  if (res.ok) return data;
  if (res.status === 403 && !_retried && /csrf/i.test(data?.detail || '')) {
    await fetchMe().catch(() => {});
    return api(path, { method, body, signal, redirectOn401 }, true);
  }
  if (res.status === 401 && redirectOn401 && unauthorizedHandler) unauthorizedHandler();
  throw errorFrom(res, data);
}

/**
 * PUT a Blob/File as the raw request body with upload progress. Resolves with the parsed JSON.
 * `url` is relative to the app base (the job response's `uploadUrl`).
 */
export function uploadBlob(url, blob, { onProgress, signal } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', `${BASE}${url}`);
    xhr.withCredentials = true;
    if (csrfToken) xhr.setRequestHeader('X-CSRF-Token', csrfToken);
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded, e.total);
    };
    xhr.onload = () => {
      let data = null;
      try {
        data = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        /* ignore */
      }
      if (xhr.status >= 200 && xhr.status < 300) return resolve(data);
      if (xhr.status === 401 && unauthorizedHandler) unauthorizedHandler();
      const detail = typeof data?.detail === 'string' ? data.detail : xhr.status === 507 ? 'We are a bit short on storage space right now. Please try again in a little while.' : `Upload failed (${xhr.status}).`;
      const wait = xhr.status === 429 ? retryAfterSeconds(xhr.getResponseHeader('Retry-After')) : null;
      reject(new ApiError(wait ? withWait(detail, wait) : detail, xhr.status, undefined, wait));
    };
    xhr.onerror = () => reject(new ApiError('The upload was interrupted. Check your connection and try again.', 0));
    xhr.onabort = () => reject(new DOMException('Upload canceled', 'AbortError'));
    if (signal) {
      if (signal.aborted) return reject(new DOMException('Upload canceled', 'AbortError'));
      signal.addEventListener('abort', () => xhr.abort(), { once: true });
    }
    xhr.send(blob);
  });
}

/** URL for a job output (browser navigation / <a download>). */
export function outputUrl(jobId, name) {
  return `${API}jobs/${encodeURIComponent(jobId)}/outputs/${encodeURIComponent(name)}`;
}

/** wss:// or ws:// URL for the live stream endpoint, same origin as the page. */
export function streamUrl() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}${API}stream/ws`;
}

export function oauthStartUrl(provider) {
  return `${API}auth/oauth/${provider}/start`;
}
