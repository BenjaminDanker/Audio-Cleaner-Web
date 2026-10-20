// Shared helpers for the Playwright checks of the website (see the suites in this folder and web/package.json).
// The browser suites run against the production layout: the site at the root of its own domain (base "/", ACW_BASE_PATH empty), the live
// WebSocket on a second process behind tests/front.mjs (what Caddy does in production). BASE is the origin; the API is /api/...
import { execFileSync } from 'node:child_process';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { startMirror } from './mirror.mjs';

export const ORIGIN = process.env.E2E_ORIGIN || 'http://127.0.0.1:8792';
export const BASE = ORIGIN; // the site is served from the root of its own domain (audiocleaner.live)
export const FILES = process.env.E2E_FILES || join(homedir(), 'audio-cleaner', 'dev', 'webtest-files');
export const SHOTS = process.env.E2E_SHOTS || join(homedir(), 'audio-cleaner', 'dev', 'webtest-shots');
const SERVER = process.env.ACW_SERVER_DIR || join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'server');
const PY = process.env.ACW_PYTHON || join(homedir(), 'audio-cleaner', 'venv', 'bin', 'python');

const adminEnv = () => ({
  ...process.env,
  ACW_DATA_DIR: process.env.ACW_DATA_DIR || join(homedir(), 'audio-cleaner', 'dev', 'webtest'),
  ACW_BASE_PATH: '',
  ACW_PUBLIC_ORIGIN: ORIGIN,
});

export function admin(...args) {
  return execFileSync(PY, ['-m', 'acw.admin', ...args], { cwd: SERVER, env: adminEnv(), encoding: 'utf8' });
}

export function loginLink(email) {
  return admin('login-link', email).split('\n')[0].trim();
}

/** The local copy of the free tools' models (mirror.mjs): the site under test is built to download them from here. */
export const mirror = () => startMirror();

export async function launch(extraArgs = [], opts = {}) {
  await startMirror().catch((e) => console.log(`WARNING ${e.message}`)); // only the suites that run the free tools need it
  const browser = await chromium.launch({ headless: true, args: ['--autoplay-policy=no-user-gesture-required', ...extraArgs] });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, permissions: ['microphone'], ...opts });
  const page = await context.newPage();
  const logs = [];
  page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) logs.push(`[${m.type()}] ${m.text()}`); });
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  return { browser, context, page, logs };
}

export async function signIn(page, email) {
  const link = loginLink(email);
  await page.goto(link);
  // Lands on /app, or on the page the visitor was sent to sign in from.
  await page.waitForURL((u) => !u.pathname.startsWith('/login'), { timeout: 15000 });
}

const results = [];
export async function check(name, fn) {
  const t0 = Date.now();
  try {
    const detail = await fn();
    results.push({ name, ok: true, detail });
    console.log(`PASS ${name}${detail ? ` -- ${detail}` : ''} (${Date.now() - t0} ms)`);
  } catch (e) {
    results.push({ name, ok: false, detail: e.message });
    console.log(`FAIL ${name} -- ${e.message.split('\n')[0]}`);
    if (e.message.includes('\n')) console.log(e.message.split('\n').slice(1, 12).map((l) => `     ${l}`).join('\n').slice(0, 1500));
  }
}
/** A check that cannot pass on the current product because of a known product bug: printed, never silently dropped. */
export function skip(name, reason) {
  console.log(`SKIP ${name} -- KNOWN PRODUCT BUG: ${reason}`);
}
export const summary = () => results;

/**
 * Waits until the files of the editor's recording are made (and handed over): the editor says data-phase="ready" once its last step is
 * done, or fails at once with the page's own error text instead of waiting out the timeout. `_testid` is kept for the older calls.
 */
export async function waitResult(page, _testid = 'result', timeout = 300000) {
  const either = page.locator('[data-testid=studio][data-phase=ready], [data-testid=job-error], [data-testid=made-error]').first();
  await either.waitFor({ timeout });
  if (await page.getByTestId('job-error').count()) throw new Error(`the page says: ${await page.getByTestId('job-error').first().innerText()}`);
  if (await page.getByTestId('made-error').count()) throw new Error(`the page says: ${await page.getByTestId('made-error').first().innerText()}`);
}

/** What the finished editor says about the sound and words made in the browser (data-rms, data-peak, data-frames, data-channels, data-cues, data-language, data-strength-db). */
export const resultAttr = (page, name) => page.getByTestId('studio').getAttribute(`data-${name}`);

// ---- the editor (the Cleanup page once a file is chosen, and a job's page) ----
export const studio = (page) => page.getByTestId('studio');
const LEVELS = { off: 0, podcast: 1, youtube: 2, broadcast: 3, keep: 4 };
const QUALITIES = { off: 0, free: 1, standard: 2, accurate: 3 };
const STRENGTH = { gentle: 1, balanced: 2, strong: 3, max: 4 };

/** The quick listen that sets the best choices ("Automatic") is over, and the waveform has been read. */
export async function settleRecommendation(page) {
  await page.waitForFunction(() => !document.querySelector('[data-testid=reco-listening]') && document.querySelector('[data-testid=waveform]')?.dataset.reading === 'no', null, { timeout: 60000 });
}
/** Automatic on or off (the switch under the file's name). */
export async function setAuto(page, on) {
  const sw = page.locator('.studio button[data-act=auto]');
  if ((await sw.getAttribute('aria-checked')) !== String(on)) await sw.click();
}
export const setVoice = (page, mode) => page.getByTestId(`voice-${mode}`).click(); // 'off' | 'noise' | 'rebuild'
/** Free (in the browser) or Standard noise removal for the selection (Noise removal is switched on first). */
export async function setKind(page, kind) {
  if ((await page.getByTestId('voice-noise').getAttribute('aria-checked')) !== 'true') await setVoice(page, 'noise');
  await page.locator(`.studio button[data-act=v-kind][data-v=${kind}]`).click();
}
export async function setStrength(page, name) { await page.locator(`.studio button[data-act=v-str][data-v="${STRENGTH[name]}"]`).click(); }
export async function setLeveling(page, name) { await page.locator(`.studio button[data-act=l-lvl][data-v="${LEVELS[name]}"]`).click(); }
export async function setSubs(page, name) { await page.locator(`.studio button[data-act=s-q][data-v="${QUALITIES[name]}"]`).click(); }

/**
 * The old Cleanup form's two groups on the editor: group 'noise' (off | browser | denoise) and 'subs' (off | browser | standard | accurate).
 * Leveling is its own track, paid and done on our side: picking noise Off or Free also turns it off (else a free job would be a paid one, as Leveling was
 * not offered with the free cleanup on the old form).
 */
export async function pick(page, group, value) {
  if (group === 'noise') {
    if (value === 'off') { await setVoice(page, 'off'); await setLeveling(page, 'off'); } else if (value === 'browser') { await setKind(page, 'free'); await setLeveling(page, 'off'); } else await setKind(page, 'paid');
  } else await setSubs(page, value === 'browser' ? 'free' : value);
}
/**
 * Puts the choices back to what they were before the paid defaults: the free in-browser cleanup when the browser can do it (else
 * Standard), Maximum strength where there is a choice, Leveling off, no Voice rebuild, subtitles Off. The checks that are about something
 * else than the automatic settings start from there.
 */
export async function restoreOldDefaults(page) {
  await setAuto(page, false);
  const free = page.locator('.studio button[data-act=v-kind][data-v=free]');
  await setVoice(page, 'noise');
  if ((await free.count()) && (await free.getAttribute('aria-disabled')) !== 'true') await setKind(page, 'free'); else await setKind(page, 'paid');
  await setSubs(page, 'off');
  await setLeveling(page, 'off');
}
/**
 * Choose a file on the Cleanup page and wait until it has been looked at (the editor shows its waveform and the quick listen is over).
 * The choices are then put back to the old defaults (restoreOldDefaults); pass { recommended: true } to keep what was recommended.
 */
export async function dropFile(page, path, { recommended = false } = {}) {
  await page.setInputFiles('input[type=file]', path);
  await page.getByTestId('start').waitFor({ timeout: 30000 });
  await settleRecommendation(page);
  if (!recommended) await restoreOldDefaults(page);
  await page.waitForTimeout(300);
}
/** Drag across the waveform from second `s` to second `e` (the editor takes whole seconds). */
export async function dragRange(page, s, e) {
  const wf = page.getByTestId('waveform');
  const box = await wf.boundingBox();
  const T = Number(await page.getByTestId('wave').getAttribute('data-seconds'));
  const x = (t) => box.x + (t / T) * box.width;
  const y = box.y + box.height / 2;
  await page.mouse.move(x(s), y);
  await page.mouse.down();
  await page.mouse.move(x((s + e) / 2), y, { steps: 4 });
  await page.mouse.move(x(e), y, { steps: 4 });
  await page.mouse.up();
}
/** The selection's title ("Whole file · 0:11", "0:03–0:08 · 5 s"). */
export const selectionTitle = (page) => page.locator('.studio .js-title').innerText();
/** The price on the main button ("Start · $0.003", "Save changes · Free"). */
export const mainButton = (page) => page.locator('.studio .bb .btn-lg').innerText();

// ---- helpers for the suites about job pages and changing a job again (jobs.mjs, redo.mjs) ----
export const FFMPEG = process.env.FFMPEG || join(homedir(), 'audio-cleaner', 'bin', 'ffmpeg');
export const JOB_PATH = /\/app\/job\/[0-9a-f]{32}$/;

/** Runs a few lines of Python against the test backend's data folder, with the server's modules importable (what admin() does). */
export function pyRun(code) {
  return execFileSync(PY, ['-c', code], { cwd: SERVER, env: adminEnv(), encoding: 'utf8' });
}

/** fetch() with the CSRF token, run inside the page (same origin, the page's own sign-in). Answers {status, body, headers}. */
export const apiCall = (page, method, path, body) => page.evaluate(async ({ method, path, body }) => {
  const me = await (await fetch('/api/me', { credentials: 'same-origin' })).json();
  const res = await fetch(`/api/${path}`, {
    method, credentials: 'same-origin',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json', 'X-CSRF-Token': me.csrfToken },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  let parsed = null;
  try { parsed = text ? JSON.parse(text) : null; } catch { parsed = text; }
  return { status: res.status, body: parsed, headers: Object.fromEntries(res.headers.entries()) };
}, { method, path, body });

/** The job ids the page's account has now (the same list the Cleanup page shows), newest first. */
export async function listJobs(page) {
  const r = await apiCall(page, 'GET', 'jobs');
  if (r.status !== 200) throw new Error(`GET /api/jobs answered ${r.status}`);
  return r.body;
}

/** Waits until the job (by its API id) has ended; answers its final form. */
export async function waitJobDone(page, id, { timeout = 300000 } = {}) {
  const t0 = Date.now();
  for (;;) {
    const j = await apiCall(page, 'GET', `jobs/${id}`);
    if (j.status === 200 && ['done', 'failed', 'canceled', 'expired'].includes(j.body.status)) return j.body;
    if (Date.now() - t0 > timeout) throw new Error(`job ${id} did not finish in ${timeout / 1000} s (${j.body?.status} ${j.body?.stage || ''} ${j.body?.progress ?? ''})`);
    await page.waitForTimeout(1500);
  }
}

/** The id of the job whose page the page is on. */
export const jobIdOf = (page) => /\/app\/job\/([0-9a-f]{32}|local-[0-9a-z]+)/.exec(new URL(page.url()).pathname)?.[1] || null;

/**
 * On a job's page: waits until the editor shows the job done (returns the editor's locator), or throws with the page's text when the job
 * failed or was canceled (those show as a card).
 */
export async function waitCardDone(page, timeout = 240000) {
  const t0 = Date.now();
  for (;;) {
    const st = await page.locator('[data-testid=studio]').first().getAttribute('data-stage', { timeout: 500 }).catch(() => null);
    if (st === 'done') return studio(page);
    const cs = await page.getByTestId('job').first().getAttribute('data-status', { timeout: 500 }).catch(() => null);
    if (cs === 'failed' || cs === 'canceled') throw new Error(`the job ${cs}: ${(await page.getByTestId('job').first().innerText()).slice(0, 300)}`);
    if (Date.now() - t0 > timeout) throw new Error(`the job is still ${st || cs} after ${timeout / 1000} s`);
    await page.waitForTimeout(500);
  }
}
