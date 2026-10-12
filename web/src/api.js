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

export class ApiError extends Error {
  constructor(message, status, fields) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = message;
    this.fields = fields;
  }
}

async function parse(res) {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return { detail: res.ok ? text : `Server error (${res.status}).` };
  }
}

function errorFrom(res, data) {
  let detail = data && typeof data.detail === 'string' ? data.detail : null;
  if (!detail) detail = res.status >= 500 ? 'The server had a problem. Please try again in a moment.' : `Request failed (${res.status}).`;
  return new ApiError(detail, res.status, data?.fields);
}

export async function fetchMe() {
  const res = await fetch(`${API}me`, { credentials: 'same-origin', headers: { Accept: 'application/json' } });
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
    throw new ApiError('Could not reach the server. Check your connection and try again.', 0);
  }
  const data = await parse(res);
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
      const detail = typeof data?.detail === 'string' ? data.detail : `Upload failed (${xhr.status}).`;
      reject(new ApiError(detail, xhr.status));
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
