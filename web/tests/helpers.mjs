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

/** Waits for a result (data-testid), or fails at once with the page's own error text instead of waiting out the timeout. */
export async function waitResult(page, testid, timeout = 300000) {
  const either = page.locator(`[data-testid=${testid}], [data-testid=job-error]`).first();
  await either.waitFor({ timeout });
  if (await page.getByTestId('job-error').count()) throw new Error(`the page says: ${await page.getByTestId('job-error').first().innerText()}`);
}

// ---- the Cleanup page's options (drop a file, then pick what to do) ----
/**
 * Click one of the option cards: group 'noise' (off | browser | denoise) or 'subs' (off | browser | standard | accurate).
 * Leveling is a plain checkbox that no noise choice changes, so picking noise Off also unticks "Leveling on its own" (else a free job would be a paid one).
 */
export async function pick(page, group, value) {
  await page.locator(`label.choice:has(input[name=${group}][value=${value}])`).click();
  if (group === 'noise' && value === 'off') await page.locator('input[name=leveling-alone]').uncheck();
  // Restore highs is a paid option of its own that starts off; the automatic settings may have ticked it for this file, and a test
  // that is about something else must not pay for it (tests about Restore highs tick it themselves, after this)
  if (group === 'noise' && value === 'denoise') await untick(page, 'input[name=highs]');
  if (group === 'noise' && value === 'off') await untick(page, 'input[name=highs-alone]');
}
/** Unticks a checkbox that is on the page and ticked (nothing happens when it is not there). */
async function untick(page, selector) {
  const box = page.locator(selector);
  if ((await box.count()) && (await box.isChecked())) await box.uncheck();
}
/** The quick listen that sets the best choices ("Automatic best settings") is over. */
export async function settleRecommendation(page) {
  await page.waitForFunction(() => !document.querySelector('[data-testid=reco-listening]'), null, { timeout: 30000 });
}
/**
 * Puts the options back to what they were before the paid defaults: the free in-browser cleanup when the browser can do it (else
 * Denoising), Maximum strength, no rebuild, no Restore highs, subtitles Off. The checks that are about something else than the automatic settings start from there.
 */
export async function restoreOldDefaults(page) {
  const card = (v) => page.locator(`label.choice:has(input[name=noise][value=${v}]):not(.is-disabled)`);
  const want = (await card('browser').count()) ? 'browser' : 'denoise';
  if (!(await page.locator(`input[name=noise][value=${want}]`).isChecked())) await card(want).click();
  const max = page.locator(`input[name=${want === 'browser' ? 'lstrength' : 'strength'}][value=max]`);
  if ((await max.count()) && !(await max.isChecked())) await page.locator(`label:has(input[name=${want === 'browser' ? 'lstrength' : 'strength'}][value=max])`).click();
  const rebuild = page.locator('input[name=rebuild]');
  if ((await rebuild.count()) && (await rebuild.isChecked())) await rebuild.uncheck();
  await untick(page, 'input[name=highs]'); // Restore highs (paid) starts off for the tests that are about something else
  // the subtitles start out Off, as before the paid defaults
  if (!(await page.locator('input[name=subs][value=off]').isChecked())) await page.locator('label.choice:has(input[name=subs][value=off])').click();
}
/**
 * Drop a file on the Cleanup page and wait until it has been looked at (the Start button shows up and the recommendation is in).
 * The options are then put back to the old defaults (restoreOldDefaults); pass { recommended: true } to keep what was recommended.
 */
export async function dropFile(page, path, { recommended = false } = {}) {
  await page.setInputFiles('input[type=file]', path);
  await page.getByTestId('start').waitFor({ timeout: 20000 });
  await settleRecommendation(page);
  if (!recommended) await restoreOldDefaults(page);
  await page.waitForTimeout(300);
}

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

/** On a job's page: waits until the job card says done (throws with the card's text when the job failed or was canceled). */
export async function waitCardDone(page, timeout = 240000) {
  const card = page.getByTestId('job').first();
  const t0 = Date.now();
  for (;;) {
    const st = await card.getAttribute('data-status', { timeout: 3000 }).catch(() => null);
    if (st === 'done') return card;
    if (st === 'failed' || st === 'canceled') throw new Error(`the job ${st}: ${(await card.innerText()).slice(0, 300)}`);
    if (Date.now() - t0 > timeout) throw new Error(`the job is still ${st} after ${timeout / 1000} s`);
    await page.waitForTimeout(500);
  }
}
