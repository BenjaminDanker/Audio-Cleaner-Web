// Copy check: visitors must not see how the service is built or what is inside it. Loads every public page (and, with the
// local test backend, the signed-in ones) and fails if the visible text mentions model names, vendors, infrastructure
// words or the words the site does not use ("premium", "upgrade", ...).
// Not checked on purpose: /licenses (open-source attributions are required by those licences), /privacy and /terms
// (they must name the kind of sub-processor), and the invisible bits of the page source.
// Needs `npm run build` served by the test backend on 127.0.0.1:8792 (see helpers.mjs); fixtures are not needed.
// Usage: node tests/copy.mjs
import { BASE, check, launch, signIn, summary } from './helpers.mjs';

const EMAIL = process.env.E2E_EMAIL || 'e2e-main@example.com';

const BANNED = [
  /DeepFilterNet/i, /\bDFN/i, /Whisper/i, /Parakeet/i, /FastConformer/i, /sherpa/i, /OpenAI/i, /gpt-/i, /WebAssembly/i, /\bwasm\b/i,
  /on-prem/i, /self-hosted/i, /home server/i, /server-side/i, /\bour servers?\b/i,
  /premium/i,
  /\bupgrade/i, /\benterprise\b/i, /\bsolutions?\b/i, /\btiers?\b/i, /\bplans?\b/i, /\bpro\b/i, /professional/i, /seamless/i, /leverage/i,
  /Create a free account/i, /Get started today/i,
  /\bStudio\b/i, /transcribe page/i, /free transcription/i, /no (account|sign-in)/i,
];

/** Everything a visitor can read or hear on the page: the text, option labels, and the labels screen readers announce. */
async function visibleText(page) {
  return page.evaluate(() => {
    const parts = [document.title, document.body.innerText];
    for (const o of document.querySelectorAll('option')) parts.push(o.textContent);
    for (const d of document.querySelectorAll('details')) parts.push(d.textContent); // closed answers are not in innerText
    for (const e of document.querySelectorAll('[aria-label],[title],[placeholder],img[alt]')) {
      for (const a of ['aria-label', 'title', 'placeholder', 'alt']) if (e.getAttribute(a)) parts.push(e.getAttribute(a));
    }
    return parts.join('\n');
  });
}

function findings(text) {
  const out = [];
  for (const re of BANNED) {
    const m = re.exec(text);
    if (m) out.push(`"${m[0]}" in: …${text.slice(Math.max(0, m.index - 40), m.index + 60).replace(/\s+/g, ' ')}…`);
  }
  return out;
}

async function scan(page, path, waitFor) {
  await page.goto(`${BASE}${path}`);
  await page.waitForSelector('main h1', { timeout: 15000 });
  if (waitFor) await page.waitForSelector(waitFor, { timeout: 15000 });
  await page.waitForTimeout(500);
  return findings(await visibleText(page)).map((f) => `${path}: ${f}`);
}

const { browser, page } = await launch();

await check('copy: public pages show no internals and none of the avoided words', async () => {
  const bad = [];
  bad.push(...(await scan(page, '/', 'table tbody tr')));
  bad.push(...(await scan(page, '/#pricing', 'table tbody tr')));
  bad.push(...(await scan(page, '/transcribe'))); // gone: the not-found page
  bad.push(...(await scan(page, '/free'))); // signed out: lands on the sign-in page
  if (!/\/login/.test(page.url())) throw new Error('signed-out /free did not go to /login: ' + page.url());
  bad.push(...(await scan(page, '/app/live'))); // signed out: lands on the sign-in page
  if (!/\/login/.test(page.url())) throw new Error('signed-out /app/live did not go to /login: ' + page.url());
  bad.push(...(await scan(page, '/login')));
  bad.push(...(await scan(page, '/nope')));
  if (bad.length) throw new Error(bad.join(' | '));
  return '8 pages clean';
});

await check('landing: no "privacy promise" section; retention stays in the FAQ; Privacy and licences only in the footer', async () => {
  await page.goto(`${BASE}/`);
  await page.waitForSelector('table tbody tr');
  const t = await page.locator('main').textContent();
  if (/privacy promise/i.test(t) || (await page.locator('#privacy').count())) throw new Error('the privacy promise section is still on the landing page');
  if (!/Results stay for 24 hours/i.test(t) || !/train/i.test(t)) throw new Error('retention answer missing from the FAQ');
  if (await page.locator('main a[href$="/privacy"]').count()) throw new Error('landing page still links to the privacy policy outside the footer');
  if (await page.locator('main a[href$="/licenses"]').count()) throw new Error('/licenses linked outside the footer');
  if (!(await page.locator('footer a[href$="/privacy"]').count()) || !(await page.locator('footer a[href$="/licenses"]').count())) throw new Error('footer lost its Privacy / licences links');
  return 'ok';
});

await check('copy: signed-in pages (Free cleanup, Files, Billing, Live, Account) show no internals', async () => {
  await signIn(page, EMAIL);
  const bad = [];
  for (const p of ['/free', '/app', '/app/billing', '/app/live', '/app/account']) bad.push(...(await scan(page, p)));
  if (bad.length) throw new Error(bad.join(' | '));
  return '5 pages clean';
});

await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
