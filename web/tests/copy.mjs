// The words and facts visitors read: the public and signed-in pages show no internals and none of the avoided words (the list
// below is also what tests/reco-unit.mjs checks its sentences against); the legal pages have no unfinished placeholders, the
// licences are real texts, and they say the service is not offered in the EU/EEA, the UK or Switzerland (and that such a card
// is not charged); the privacy page's retention answers and the landing FAQ match what the server does; the job step names and the
// ledger wording are plain. Needs the test backend (port 8792) with `npm run build` output; the signed-in parts use a fresh account.
// Usage: node tests/copy.mjs
import { join } from 'node:path';
import { BASE, FILES, check, dropFile, launch, pick, signIn, summary } from './helpers.mjs';

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
async function visibleText(page, { skipCode = false } = {}) {
  return page.evaluate((skipCode) => {
    // The developer pages show the API's own field values (for example "premium") in code samples; those have to be spelled
    // exactly, so code is left out of the scan there. Everything else on the page is checked.
    let body = document.body.innerText;
    if (skipCode) {
      const clone = document.body.cloneNode(true);
      clone.querySelectorAll('code, pre, script, style').forEach((e) => e.remove());
      body = clone.innerText;
    }
    const parts = [document.title, body];
    for (const o of document.querySelectorAll('option')) parts.push(o.textContent);
    for (const d of document.querySelectorAll('details')) parts.push(d.textContent); // closed answers are not in innerText
    for (const m of document.querySelectorAll('meta[name=description],meta[property^="og:"],meta[name^="twitter:"]')) parts.push(m.getAttribute('content') || '');
    for (const e of document.querySelectorAll('[aria-label],[title],[placeholder],img[alt]')) {
      for (const a of ['aria-label', 'title', 'placeholder', 'alt']) if (e.getAttribute(a)) parts.push(e.getAttribute(a));
    }
    return parts.join('\n');
  }, skipCode);
}

function findings(text) {
  const out = [];
  for (const re of BANNED) {
    const m = re.exec(text);
    if (m) out.push(`"${m[0]}" in: …${text.slice(Math.max(0, m.index - 40), m.index + 60).replace(/\s+/g, ' ')}…`);
  }
  return out;
}

async function scan(page, path, waitFor, opts) {
  await page.goto(`${BASE}${path}`);
  await page.waitForSelector('main h1', { timeout: 15000 });
  if (waitFor) await page.waitForSelector(waitFor, { timeout: 15000 });
  await page.waitForTimeout(500);
  return findings(await visibleText(page, opts)).map((f) => `${path}: ${f}`);
}

const { browser, page } = await launch();

await check('copy: public pages (signed out, the sign-in page for /app and /app/live) show no internals and none of the avoided words', async () => {
  const bad = [];
  bad.push(...(await scan(page, '/', 'table tbody tr')));
  bad.push(...(await scan(page, '/#pricing', 'table tbody tr')));
  bad.push(...(await scan(page, '/app')));
  if (!/\/login/.test(page.url())) throw new Error('signed-out /app did not go to /login: ' + page.url());
  bad.push(...(await scan(page, '/app/live')));
  if (!/\/login/.test(page.url())) throw new Error('signed-out /app/live did not go to /login: ' + page.url());
  bad.push(...(await scan(page, '/login')));
  bad.push(...(await scan(page, '/contact')));
  bad.push(...(await scan(page, '/nope')));
  bad.push(...(await scan(page, '/developers', 'pre.code', { skipCode: true }))); // the API guide: code samples are exempt
  bad.push(...(await scan(page, '/developers/live', 'pre.code', { skipCode: true })));
  if (bad.length) throw new Error(bad.join(' | '));
  return '10 pages clean';
});

await check('copy: signed-in pages (Cleanup with a file and its options, Billing, Live, Account) show no internals', async () => {
  await signIn(page, EMAIL);
  const bad = [];
  for (const p of ['/app', '/app/billing', '/app/live', '/app/account']) bad.push(...(await scan(page, p)));
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'noisy.wav'));
  await pick(page, 'subs', 'accurate');
  bad.push(...findings(await visibleText(page)).map((x) => `/app with a file: ${x}`));
  if (bad.length) throw new Error(bad.join(' | '));
  return '4 pages + the options for a file clean';
});

await check('legal: Terms and Privacy have no unfinished placeholders ("to be confirmed", [brackets], "undefined") and no empty contact links', async () => {
  const bad = [];
  for (const path of ['/terms', '/privacy']) {
    await page.goto(`${BASE}${path}`);
    await page.waitForSelector('main h1');
    const text = await visibleText(page);
    for (const re of [/to be confirmed/i, /\bTBD\b/, /\[[^\]\n]*\]/, /lorem ipsum/i, /\bundefined\b/, /\bnull\b/, /legal review|draft/i]) {
      const m = re.exec(text);
      if (m) bad.push(`${path}: "${m[0]}" in …${text.slice(Math.max(0, m.index - 50), m.index + 60).replace(/\s+/g, ' ')}…`);
    }
    for (const href of await page.locator('main a[href^="mailto:"], footer a[href^="mailto:"]').evaluateAll((as) => as.map((a) => a.getAttribute('href')))) {
      if (!/^mailto:[^@\s]+@[^@\s]+\.[^@\s]+$/.test(href)) bad.push(`${path}: contact link ${href}`);
    }
  }
  if (bad.length) throw new Error(bad.join(' | '));
  return 'clean';
});

await check('legal: Terms and Privacy say the service is not offered in the EU/EEA, the UK or Switzerland, and that a card from there is cancelled before it is charged', async () => {
  for (const p of ['/terms', '/privacy']) {
    await page.goto(`${BASE}${p}`);
    const t = (await page.locator('main').innerText()).replace(/\s+/g, ' ');
    if (!/not available to people in the (European Union or European Economic Area \(EU\/EEA\)|EU\/EEA), the (United Kingdom|UK) or Switzerland/.test(t)) throw new Error(p + ': availability sentence missing');
    if (!/card issued in one of those places is used, we cancel the payment before it is charged/.test(t)) throw new Error(p + ': not-charged sentence missing');
  }
  return 'ok';
});

await check('legal: every licence in the list links to a real licence text (not the app page), and the Creative Commons text is linked, not shipped', async () => {
  await page.goto(`${BASE}/licenses`);
  await page.waitForSelector('[data-testid=licenses] tbody tr');
  const links = await page.locator('[data-testid=licenses] a[href*="/licenses/"]:not([href^="http"])').evaluateAll((as) => as.map((a) => a.getAttribute('href')));
  if (links.length < 12) throw new Error(`only ${links.length} licence links`);
  const seen = [...new Set(links)];
  for (const href of seen) {
    const res = await page.request.get(new URL(href, BASE + '/').href);
    const body = await res.text();
    if (!res.ok() || body.length < 500) throw new Error(`${href}: ${res.status()} ${body.length} bytes`);
    if (/^\s*<!doctype html/i.test(body)) throw new Error(`${href}: got the app page`);
    if (!/MIT|Apache|Mozilla Public|Lesser General Public|Permission is hereby granted|Creative Commons Attribution/.test(body.slice(0, 4000))) throw new Error(`${href}: does not look like a licence`);
  }
  await page.goto(`${BASE}/licenses`);
  if (!(await page.locator('[data-testid=licenses] a[href^="https://creativecommons.org/licenses/by/4.0/"]').count())) throw new Error('no link to the CC BY 4.0 licence');
  const main = await page.locator('main').innerText();
  for (const need of ['mediabunny', 'React', 'React Router', 'Vite', 'Rust crates', 'sherpa-onnx', 'CC BY 4.0']) if (!main.includes(need)) throw new Error('missing ' + need);
  if (!/models .*are downloaded by your browser from Hugging Face/.test(main) || /dfn\//.test(main)) throw new Error('the page does not say the free tools\' models come from Hugging Face');
  return `${seen.length} licence files`;
});

await check('retention: the privacy page states the server\'s retention periods, the landing FAQ says results stay 24 hours, and there is no privacy promise section on the landing page', async () => {
  await page.goto(`${BASE}/`);
  await page.waitForSelector('table tbody tr');
  const t = await page.locator('main').textContent();
  if (/privacy promise/i.test(t) || (await page.locator('#privacy').count())) throw new Error('the privacy promise section is still on the landing page');
  if (!/Results stay for 24 hours/i.test(t) || !/train/i.test(t)) throw new Error('retention answer missing from the FAQ');
  await page.goto(`${BASE}/privacy`);
  const pr = await page.locator('#retention ~ ul').first().innerText();
  for (const need of [/7 years/, /12 months/, /90 days/, /30 days/, /contact form/, /free live trial/]) if (!need.test(pr)) throw new Error(`Privacy retention lacks ${need}`);
  return 'ok';
});

await check('running jobs: the server\'s step names are shown in plain words ("premium" and "cloud asr" never appear), an unknown step shows only the percentage', async () => {
  const now = Math.floor(Date.now() / 1000);
  const job = (id, extra) => ({ id, kind: 'denoise', free: false, status: 'running', inputName: `${id}.wav`, createdAt: now, durationSeconds: 60, progress: 0, stage: '', outputs: [], costMicros: 0, expiresAt: null, queuePosition: null, queueWaitSeconds: null, ...extra });
  const list = [
    job('j1', { status: 'cloud_asr', stage: 'cloud asr running', progress: 85 }),
    job('j2', { stage: 'transcribing (premium)', progress: 40 }),
    job('j3', { stage: 'Cleaning', progress: 41 }),
    job('j4', { stage: 'some brand new step', progress: 7 }),
  ];
  await page.route('**/api/jobs', (r) => (r.request().method() === 'GET' ? r.fulfill({ json: { jobs: list } }) : r.continue()));
  try {
    await page.goto(`${BASE}/app`);
    await page.getByTestId('job').first().waitFor();
    const lines = (await page.locator('[data-testid=job] p[role=status]').allInnerTexts()).map((l) => l.trim());
    const all = lines.join('\n');
    if (/premium|cloud asr|some brand new/i.test(all)) throw new Error('raw step shown: ' + all.slice(0, 400));
    for (const w of ['Writing the transcript · 85%', 'Writing the transcript · 40%', 'Cleaning the sound · 41%', '7% done']) if (!lines.includes(w)) throw new Error(`missing "${w}" in ${JSON.stringify(lines)}`);
    return lines.join(' | ');
  } finally {
    await page.unroute('**/api/jobs');
  }
});

await check('ledger: the entries read as what happened (no "usage", "grant" or "weird" words), with the file name and minutes; a live session that was cut short says so', async () => {
  const { ledgerKind, ledgerText } = await import('../src/ledger.js');
  if (ledgerText({ kind: 'usage', description: 'denoise + transcribe: noisy.flac (0.2 min)' }) !== 'Denoising and subtitles: noisy.flac (0.2 min)') throw new Error('denoise + transcribe');
  if (ledgerText({ kind: 'usage', description: 'Live session (3.0 min, cut short by a restart)' }) !== 'Live session (3.0 min, ended early)') throw new Error('live');
  if (ledgerText({ kind: 'topup', description: 'Top-up $20.00' }) !== 'Top-up $20.00') throw new Error('topup');
  if (ledgerKind({ kind: 'grant' }) !== 'Credit added' || ledgerKind({ kind: 'usage' }) !== 'Used') throw new Error('kinds');
  return 'ok';
});

await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
