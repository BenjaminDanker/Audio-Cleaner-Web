// Shared helpers for the Playwright checks of the website (see the suites in this folder and web/package.json).
// The browser suites run against the production layout: the site at the root of its own domain (base "/", ACW_BASE_PATH empty), the live
// WebSocket on a second process behind tests/front.mjs (what Caddy does in production). BASE is the origin; the API is /api/...
import { execFileSync } from 'node:child_process';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { chromium } from 'playwright';
import { startMirror } from './mirror.mjs';

export const ORIGIN = process.env.E2E_ORIGIN || 'http://127.0.0.1:8792';
export const BASE = ORIGIN; // the site is served from the root of its own domain (audiocleaner.live)
export const FILES = process.env.E2E_FILES || join(homedir(), 'audio-cleaner', 'dev', 'webtest-files');
export const SHOTS = process.env.E2E_SHOTS || join(homedir(), 'audio-cleaner', 'dev', 'webtest-shots');
const SERVER = process.env.ACW_SERVER_DIR || join(homedir(), 'repos', 'Audio-Cleaner-Web', 'server');
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
/** Click one of the option cards: group 'noise' (off | browser | denoise) or 'subs' (off | browser | standard | accurate). */
export async function pick(page, group, value) {
  await page.locator(`label.choice:has(input[name=${group}][value=${value}])`).click();
}
/** The quick listen that picks the recommended choices ("Recommended for this file") is over. */
export async function settleRecommendation(page) {
  await page.waitForFunction(() => !document.querySelector('[data-testid=reco-listening]'), null, { timeout: 30000 });
}
/**
 * Puts the options back to what they were before files were measured: the free in-browser cleanup when the browser can do it (else
 * Denoising), Maximum strength, no rebuild. The checks that are about something else than the recommendation start from there.
 */
export async function restoreOldDefaults(page) {
  const card = (v) => page.locator(`label.choice:has(input[name=noise][value=${v}]):not(.is-disabled)`);
  const want = (await card('browser').count()) ? 'browser' : 'denoise';
  if (!(await page.locator(`input[name=noise][value=${want}]`).isChecked())) await card(want).click();
  const max = page.locator(`input[name=${want === 'browser' ? 'lstrength' : 'strength'}][value=max]`);
  if ((await max.count()) && !(await max.isChecked())) await page.locator(`label:has(input[name=${want === 'browser' ? 'lstrength' : 'strength'}][value=max])`).click();
  const rebuild = page.locator('input[name=rebuild]');
  if ((await rebuild.count()) && (await rebuild.isChecked())) await rebuild.uncheck();
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
