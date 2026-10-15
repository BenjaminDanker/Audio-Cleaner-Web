// Shared helpers for the Playwright end-to-end checks (see tests/e2e.mjs).
import { execFileSync } from 'node:child_process';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { chromium } from 'playwright';

export const ORIGIN = process.env.E2E_ORIGIN || 'http://127.0.0.1:8792';
export const BASE = `${ORIGIN}/audio`;
export const FILES = process.env.E2E_FILES || join(homedir(), 'audio-cleaner', 'dev', 'webtest-files');
export const SHOTS = process.env.E2E_SHOTS || join(homedir(), 'audio-cleaner', 'dev', 'webtest-shots');
const SERVER = process.env.ACW_SERVER_DIR || join(homedir(), 'repos', 'Audio-Cleaner-Web', 'server');
const PY = process.env.ACW_PYTHON || join(homedir(), 'audio-cleaner', 'venv', 'bin', 'python');

const adminEnv = () => ({
  ...process.env,
  ACW_DATA_DIR: process.env.ACW_DATA_DIR || join(homedir(), 'audio-cleaner', 'dev', 'webtest'),
  ACW_BASE_PATH: '/audio',
  ACW_PUBLIC_ORIGIN: ORIGIN,
});

export function admin(...args) {
  return execFileSync(PY, ['-m', 'acw.admin', ...args], { cwd: SERVER, env: adminEnv(), encoding: 'utf8' });
}

export function loginLink(email) {
  return admin('login-link', email).split('\n')[0].trim();
}

export async function launch(extraArgs = [], opts = {}) {
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
  await page.waitForURL((u) => u.pathname.startsWith('/audio/') && !u.pathname.startsWith('/audio/login'), { timeout: 15000 });
}

let results = [];
export async function check(name, fn) {
  const t0 = Date.now();
  try {
    const detail = await fn();
    results.push({ name, ok: true, detail });
    console.log(`PASS ${name}${detail ? ` -- ${detail}` : ''} (${Date.now() - t0} ms)`);
  } catch (e) {
    results.push({ name, ok: false, detail: e.message });
    console.log(`FAIL ${name} -- ${e.message.split('\n')[0]}`);
  }
}
export const summary = () => results;

// ---- the Cleanup page's options (drop a file, then pick what to do) ----
/** Click one of the option cards: group 'noise' (off | browser | polish) or 'subs' (off | free | standard | accurate). */
export async function pick(page, group, value) {
  await page.locator(`label.choice:has(input[name=${group}][value=${value}])`).click();
}
/** Drop a file on the Cleanup page and wait until it has been looked at (the Start button shows up). */
export async function dropFile(page, path) {
  await page.setInputFiles('input[type=file]', path);
  await page.getByTestId('start').waitFor({ timeout: 20000 });
  await page.waitForTimeout(300);
}
export const startButton = (page) => page.getByTestId('start');
