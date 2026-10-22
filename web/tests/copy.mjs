// The words and facts visitors read: the public and signed-in pages show no internals and none of the avoided words (the list
// below is also what tests/reco-unit.mjs checks its sentences against); the legal pages have no unfinished placeholders, the
// licences are real texts, and they say the service is not offered in the EU/EEA, the UK or Switzerland (and that such a card
// is not charged); the privacy page's retention answers and the landing FAQ match what the server does; the job step names and the
// ledger wording are plain. The signed-in scan covers the editor with a file chosen (its tooltips too, and its Output menu, Details,
// Your files and All the options window). Needs the test backend (port 8792) with `npm run build` output; the signed-in parts use a fresh account.
// Usage: node tests/copy.mjs
import { join } from 'node:path';
import { BASE, FILES, check, dropFile, launch, pick, setVoice, signIn, summary } from './helpers.mjs';

const EMAIL = process.env.E2E_EMAIL || 'e2e-main@example.com';

const BANNED = [
  /DeepFilterNet/i, /\bDFN/i, /\bWhisper\b/, /Parakeet/i, /FastConformer/i, /sherpa/i, /OpenAI/i, /gpt-/i, /WebAssembly/i, /\bwasm\b/i,
  /on-prem/i, /self-hosted/i, /home server/i, /server-side/i, /\bour servers?\b/i,
  /premium/i,
  /\bupgrade/i, /\benterprise\b/i, /\bsolutions?\b/i, /\btiers?\b/i, /\bplans?\b/i, /\bpro\b/i, /professional/i, /seamless/i, /leverage/i,
  /Create a free account/i, /Get started today/i,
  /\bStudio\b/i, /transcribe page/i, /free transcription/i, /no (account|sign-in)/i,
  /\bpolish(ed|es|ing)?\b|\bPolish (your|the|it|up|this|that|my)\b/, // never "polish"
  /Restore (missing )?highs/i, /missing highs/i, /AP-BWE/i, // (restoring highs is not offered any more: a recording that lost them is rebuilt)
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
    for (const e of document.querySelectorAll('[aria-label],[title],[placeholder],img[alt],[data-tip]')) {
      for (const a of ['aria-label', 'title', 'placeholder', 'alt', 'data-tip']) if (e.getAttribute(a)) parts.push(e.getAttribute(a)); // (data-tip: the editor's tooltips)
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

await check('copy: public pages (signed out, with the Cleanup and Live pages, which open without an account) show no internals and none of the avoided words', async () => {
  const bad = [];
  bad.push(...(await scan(page, '/', 'table tbody tr')));
  bad.push(...(await scan(page, '/#pricing', 'table tbody tr')));
  bad.push(...(await scan(page, '/app', '[data-testid=free-no-account]')));
  if (new URL(page.url()).pathname !== '/app') throw new Error('signed-out /app did not stay: ' + page.url());
  bad.push(...(await scan(page, '/app/live', '[data-testid=keys-signin]')));
  if (new URL(page.url()).pathname !== '/app/live') throw new Error('signed-out /app/live did not stay: ' + page.url());
  bad.push(...(await scan(page, '/login')));
  bad.push(...(await scan(page, '/contact')));
  bad.push(...(await scan(page, '/nope')));
  bad.push(...(await scan(page, '/developers', 'pre.code', { skipCode: true }))); // the API guide: code samples are exempt
  bad.push(...(await scan(page, '/developers/live', 'pre.code', { skipCode: true })));
  if (bad.length) throw new Error(bad.join(' | '));
  return '10 pages clean (the sign-in page too)';
});

await check('copy: signed-in pages (Cleanup, Billing, Live, Account) and the editor with a file chosen (its tracks, inspector, tooltips, Output menu, Details, Your files and the All the options window) show no internals', async () => {
  await signIn(page, EMAIL);
  const bad = [];
  for (const p of ['/app', '/app/billing', '/app/live', '/app/account']) bad.push(...(await scan(page, p)));
  const scanHere = async (where) => {
    const text = await visibleText(page);
    bad.push(...findings(text).map((x) => `${where}: ${x}`));
    if (/\bhighs\b/i.test(text)) bad.push(`${where}: "highs" is still on the page`);
    if (await page.locator('input[name=highs], input[name=highs-alone]').count()) bad.push(`${where}: a Restore highs box is still there`);
  };
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'noisy.wav'));
  await scanHere('/app with a file');
  await pick(page, 'subs', 'accurate');
  await scanHere('/app with a file, Extra accurate subtitles');
  // the paid options of each noise choice (Voice rebuild too: it is paused on this backend, which says so), and no Restore highs in any of them (the step is gone)
  for (const noise of ['denoise', 'off', 'browser']) {
    await pick(page, 'noise', noise);
    await scanHere(`/app, noise ${noise}`);
  }
  await setVoice(page, 'rebuild').catch(() => {});
  await scanHere('/app, Voice rebuild');
  for (const [id, popId, name] of [['output-button', 'output-pop', 'the Output menu'], ['bar-det', 'details', 'Details']]) {
    await page.getByTestId(id).click();
    await page.getByTestId(popId).waitFor();
    await scanHere(`/app, ${name}`);
    await page.getByTestId(id).click();
  }
  await page.getByTestId('info-open').click();
  await page.getByTestId('info-modal').waitFor();
  await scanHere('/app, All the options');
  await page.getByTestId('info-close').click();
  // a finished job's page (stubbed, with its timeline): Your files and the finished bar
  const now = Math.floor(Date.now() / 1000);
  const done = { id: 'copydone', kind: 'denoise', free: false, status: 'done', inputName: 'copydone.wav', displayName: 'copydone.wav', createdAt: now - 60, finishedAt: now - 30, durationSeconds: 60, progress: 100, stage: '', costMicros: 1500, expiresAt: now + 7000, queuePosition: null, queueWaitSeconds: null,
    outputs: [{ name: 'copydone-cleaned.flac', kind: 'audio', bytes: 1000 }], transcribeTier: 'local', edit: { available: true, hasTimeline: false, run: null, lastRun: null, suggestions: false } };
  await page.route('**/api/jobs', (r) => (r.request().method() === 'GET' ? r.fulfill({ json: { jobs: [done], limits: { maxActiveJobs: 3, retentionSeconds: 7200 } } }) : r.continue()));
  await page.route('**/api/jobs/copydone/timeline', (r) => r.fulfill({ json: {
    timeline: { version: 1, seconds: 60, layers: { noise: [{ start: 0, end: 60, strength_db: 100, remover: 'auto' }] } }, rendered: null, seconds: 60, channels: 1,
    secondTrack: false, cached: {}, suggestions: null, run: null, lastRun: null, preview: null } }));
  await page.route('**/api/jobs/copydone/peaks/**', (r) => r.fulfill({ status: 404, json: { detail: 'Nothing is cached for this job yet.', code: 'not_cached' } }));
  try {
    await page.goto(`${BASE}/app/job/copydone`);
    await page.getByTestId('download-again').waitFor({ timeout: 20000 });
    await scanHere('a finished job page');
    for (const [id, popId, name] of [['bar-files', 'files', 'Your files'], ['output-button', 'output-pop', 'the Output menu']]) {
      await page.getByTestId(id).click();
      await page.getByTestId(popId).waitFor();
      await scanHere(`a finished job page, ${name}`);
      await page.getByTestId(id).click();
    }
    await page.getByTestId('info-open').click();
    await page.getByTestId('info-modal').waitFor();
    await scanHere('a finished job page, All the options');
    await page.getByTestId('info-close').click();
  } finally {
    await page.unroute('**/api/jobs');
    await page.unroute('**/api/jobs/copydone/timeline');
    await page.unroute('**/api/jobs/copydone/peaks/**');
  }
  if (bad.length) throw new Error(bad.join(' | '));
  return '4 pages + the editor (a file chosen, the menus, All the options, a finished job) clean';
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

/** Anything that says uploads, copies or results go with 24 hours, or that the upload goes as soon as it is processed (the old rules). */
const OLD_RETENTION = [
  /(upload|result|file|cop(y|ies))[^.]{0,100}24 hours?/i, /24 hours?[^.]{0,100}(upload|result|file|cop(y|ies))/i,
  /deleted as soon as/i, /as soon as (it is|it.s|they are|they.re) processed/i, /Results stay for/i,
];
const oldRetention = (t) => OLD_RETENTION.filter((re) => re.test(t)).map((re) => `${re}: …${t.slice(Math.max(0, t.search(re) - 40), t.search(re) + 120).replace(/\s+/g, ' ')}…`);

await check('retention: the landing FAQ, the Privacy page (section 4) and the Terms (section 8) say uploads, kept copies and results are kept together for 2 hours and then deleted together, with no trace of "24 hours" or "deleted as soon as"; no privacy promise section on the landing page', async () => {
  await page.goto(`${BASE}/`);
  await page.waitForSelector('table tbody tr');
  const t = (await page.locator('main').textContent()).replace(/\s+/g, ' ');
  if (/privacy promise/i.test(t) || (await page.locator('#privacy').count())) throw new Error('the privacy promise section is still on the landing page');
  if (!/(upload|file)s? [^.]*(and|,) [^.]*results? are kept together for 2 hours/i.test(t) || !/deleted together/i.test(t)) throw new Error('the FAQ does not say files and results are kept together for 2 hours and then deleted together: ' + (/What happens to my files[^?]*\?.{0,400}/i.exec(t)?.[0] || t.slice(0, 400)));
  if (!/train/i.test(t)) throw new Error('the FAQ no longer says nothing is trained on the audio');
  const bad = oldRetention(t).map((x) => `landing: ${x}`);

  await page.goto(`${BASE}/privacy`);
  await page.waitForSelector('main h1');
  const section4 = (await page.locator('#files ~ ul').first().innerText()).replace(/\s+/g, ' ');
  if (!/kept for 2 hours/.test(section4) || !/Results[^.]*available for download for 2 hours/.test(section4) || !/deleted together with the upload/.test(section4)) throw new Error('Privacy section 4: ' + section4.slice(0, 500));
  // the editor's rule: 2 hours after the last change, and never later than 6 hours after the upload
  if (!/after your last change to the job/.test(section4) || !/deleted 6 hours after the upload/.test(section4) || !/6 hours after the upload at the latest/.test(section4)) throw new Error('Privacy section 4 lacks the last-change / 6 hours rule: ' + section4.slice(0, 700));
  // (the one sentence about the copy this browser keeps of the file while the person signs in is about their own device, not about uploads: it is deleted once restored, or after a day)
  const KEPT_COPY = /That copy stays on your device, is never uploaded by this step, and is deleted as soon as it is restored, or after a day\./;
  const privacy = (await page.locator('main').innerText()).replace(/\s+/g, ' ');
  if (!KEPT_COPY.test(privacy) || !/You don.t need an account for them\./.test(privacy)) throw new Error('Privacy lacks the sentences about the free tools without an account and the copy the browser keeps while signing in');
  bad.push(...oldRetention(privacy.replace(KEPT_COPY, '')).map((x) => `privacy: ${x}`));
  const pr = await page.locator('#retention ~ ul').first().innerText();
  for (const need of [/7 years/, /12 months/, /90 days/, /30 days/, /contact form/, /free live trial/]) if (!need.test(pr)) throw new Error(`Privacy retention lacks ${need}`);

  await page.goto(`${BASE}/terms`);
  await page.waitForSelector('main h1');
  const section8 = (await page.locator('#data ~ ul').first().innerText()).replace(/\s+/g, ' ');
  if (!/kept together for 2 hours/.test(section8) || !/deleted together/.test(section8)) throw new Error('Terms section 8: ' + section8.slice(0, 500));
  if (!/after the last change to a job/.test(section8) || !/no more than 6 hours after the upload/.test(section8)) throw new Error('Terms section 8 lacks the last-change / 6 hours rule: ' + section8.slice(0, 500));
  const terms = (await page.locator('main').innerText()).replace(/\s+/g, ' ');
  if (!/including the free tools without an account/.test(terms) || !/need no account and no credit/.test(terms)) throw new Error('Terms lack the sentences about the free tools without an account');
  bad.push(...oldRetention(terms).map((x) => `terms: ${x}`));
  if (bad.length) throw new Error(bad.join(' | '));
  return '2 hours, together, on the landing page, Privacy and Terms';
});

await check('retention: the signed-in Cleanup page (the list note and "Good to know"), the Account page and the Developers page say 2 hours too, and none of them keeps the old 24 hours', async () => {
  const bad = [];
  await page.goto(`${BASE}/app`);
  await page.getByTestId('jobs-note').waitFor();
  const note = (await page.getByTestId('jobs-note').innerText()).replace(/\s+/g, ' ');
  if (!/for 2 hours after your last change to it, and at most 6 hours after the upload/.test(note)) throw new Error('the list note: ' + note);
  const good = (await page.locator('main').innerText()).replace(/\s+/g, ' ');
  if (!/upload and your results are kept together for 2 hours after your last change to a file \(at most 6 hours after the upload\)[^.]*deleted together/.test(good)) throw new Error('Cleanup "Good to know": ' + good.slice(-700));
  bad.push(...oldRetention(good).map((x) => `cleanup: ${x}`));
  await page.goto(`${BASE}/app/account`);
  await page.waitForSelector('main h1');
  const acct = (await page.locator('main').innerText()).replace(/\s+/g, ' ');
  if (!/uploads and results are deleted 2 hours after your last change to a job, and at most 6 hours after the upload/.test(acct)) throw new Error('Account page: ' + acct.slice(0, 600));
  bad.push(...oldRetention(acct).map((x) => `account: ${x}`));
  await page.goto(`${BASE}/developers`);
  await page.waitForSelector('pre.code');
  const dev = (await page.locator('main').innerText()).replace(/\s+/g, ' ');
  if (!/kept together for 2 hours after the job.s last change[^.]*never more than 6 hours after the upload, then deleted together/.test(dev)) throw new Error('Developers page: no 2 hours / 6 hours answer for the files');
  for (const need of [/Editing part by part/, /\/api\/jobs\/\{id\}\/timeline/, /\/api\/jobs\/\{id\}\/quote/, /\/api\/jobs\/\{id\}\/process/, /\/api\/jobs\/\{id\}\/export/, /preview\/\{layer\}/, /suggestions/, /bad_timeline/, /needs_processing/, /not_cached/, /2 hours after the last edit/, /Voice rebuild wins/, /no minimum/]) {
    if (!need.test(dev)) throw new Error(`Developers page lacks ${need}`);
  }
  // the API's idempotency keys are remembered for 24 hours: that is not about files
  bad.push(...oldRetention(dev.replace(/within 24 hours/g, '').replace(/Keys are remembered for 24 hours, per account\./g, '')).map((x) => `developers: ${x}`));
  if (bad.length) throw new Error(bad.join(' | '));
  return 'ok';
});

await check('file conversion: the Terms, the Privacy Policy and the price table say changing a file type or video size in the browser is free, needs no account and uploads nothing, and that a video the browser cannot convert is converted on our side (uploaded whole, copying free, making the picture again paid as Video conversion)', async () => {
  const flat = async (path, wait) => {
    await page.goto(`${BASE}${path}`);
    await page.waitForSelector(wait);
    return (await page.locator('main').innerText()).replace(/\s+/g, ' ');
  };
  const need = (what, text, res) => { for (const re of res) if (!re.test(text)) throw new Error(`${what} lacks ${re}`); };
  const terms = await flat('/terms', 'main h1');
  need('Terms', terms, [
    /The free noise removal, the free subtitles and changing a file.s type or video size in your browser run on your own device and need no account and no credit\./,
    /File type and video size\. Free in your browser, with no account; nothing is uploaded\. A video your browser can.t convert can be converted on our side: the whole video is uploaded, copying its picture into another file type is free, and making the picture again \(a new size, or a file type that can.t hold it as it is\) is paid, as Video conversion\./,
    /The free noise removal and changing a file.s type or video size in your browser process files locally; nothing is uploaded\./,
  ]);
  const privacy = await flat('/privacy', 'main h1');
  need('Privacy', privacy, [
    /2\. Free noise removal, subtitles and file conversion in your browser/,
    /The free noise removal, the free subtitles and changing a file.s type or video size on the Cleanup page process your file entirely on your device\. Your audio and video are not uploaded/,
    /A video your browser can.t convert itself is uploaded whole when you have it converted on our side \(another file type or video size\), and is kept and deleted like any upload \(below\)\./,
  ]);
  // (the old sentences about the free tools are still there)
  need('Privacy', privacy, [/You don.t need an account for them\./, /That copy stays on your device, is never uploaded by this step/]);
  need('Terms', terms, [/including the free tools without an account/]);
  // the price table: the free row, and Video conversion among the Files (its price is the server's list's, not pinned here)
  const rate = (await (await fetch(`${BASE}/api/billing/pricing`)).json()).rates.find((r) => r.product === 'convert');
  if (!rate || rate.label !== 'Video conversion') throw new Error('price list: ' + JSON.stringify(rate));
  await page.goto(`${BASE}/#pricing`);
  await page.waitForSelector('table.price-table tbody tr');
  const rows = (await page.locator('table.price-table tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' ').trim());
  const free = rows.find((r) => /^File type and video size in your browser/.test(r));
  if (!free || !/Another file type \(MP4, MKV, WebM, MP3, …\) or video size, on its own or with the rest\. Made on your own device, nothing is uploaded\. Free$/.test(free)) throw new Error('free row: ' + free);
  const paid = rows.find((r) => /^Video conversion/.test(r));
  if (!paid || !/Another file type or video size for a video your browser can.t convert itself, made on our side\. Only when the picture has to be made again; copying it into another file type is free\b.*\/ min/.test(paid)) throw new Error('paid row: ' + paid);
  const order = rows.map((r) => r.split(' ')[0] + ' ' + r.split(' ')[1]);
  const iFree = rows.indexOf(free), iPaid = rows.indexOf(paid), iFiles = rows.findIndex((r) => r === 'Files'), iLive = rows.findIndex((r) => r === 'Live streaming');
  if (!(iFree < iFiles && iFiles < iPaid && iPaid < iLive)) throw new Error(`rows are not in the groups Free / Files / Live: ${order.join(' | ')}`);
  return `${free.slice(0, 60)}… | ${paid.slice(-24)}`;
});

await check('running jobs: the server\'s step names are shown in plain words on the list rows and on the job\'s page ("premium" and "cloud asr" never appear), an unknown step shows only a plain "Working on it" with the percentage on the row', async () => {
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
    await page.getByTestId('job-row').first().waitFor();
    if (await page.getByTestId('job').count()) throw new Error('full job cards on the list');
    const lines = (await page.locator('[data-testid=job-row] [data-testid=row-line]').allInnerTexts()).map((l) => l.trim());
    const all = lines.join('\n');
    if (/premium|cloud asr|some brand new/i.test(all)) throw new Error('raw step shown: ' + all.slice(0, 400));
    for (const w of ['Writing the transcript · 85%', 'Writing the transcript · 40%', 'Cleaning the sound · 41%', 'Working on it · 7%']) if (!lines.includes(w)) throw new Error(`missing "${w}" in ${JSON.stringify(lines)}`);
    // the job's own page, same words
    const onPages = [];
    for (const [id, want] of [['j1', 'Writing the transcript · 85%'], ['j2', 'Writing the transcript · 40%'], ['j3', 'Cleaning the sound · 41%'], ['j4', '7% done']]) {
      await page.goto(`${BASE}/app/job/${id}`);
      const line = (await page.getByTestId('job-line').innerText()).trim();
      if (line !== want) throw new Error(`job page ${id}: "${line}", wanted "${want}"`);
      const text = await page.getByTestId('job-page').innerText();
      if (/premium|cloud asr|some brand new/i.test(text)) throw new Error(`job page ${id} shows a raw step: ` + text.slice(0, 300));
      onPages.push(line);
    }
    return `rows: ${lines.join(' | ')}; pages: ${onPages.join(' | ')}`;
  } finally {
    await page.unroute('**/api/jobs');
  }
});

await check('time left: a running row and a queued row say how long it takes in plain words ("about 2 min left", "In line · about 3 min left"), never raw seconds', async () => {
  const now = Math.floor(Date.now() / 1000);
  const job = (id, extra) => ({ id, kind: 'denoise', free: false, status: 'running', inputName: `${id}.wav`, createdAt: now, durationSeconds: 60, progress: 50, stage: 'Cleaning', outputs: [], costMicros: 0, expiresAt: null, queuePosition: null, queueWaitSeconds: null, ...extra });
  const list = [
    job('t1', { estimateSeconds: 240 }), // half of 4 minutes is left
    job('t2', { status: 'queued', progress: 0, stage: '', queuePosition: 2, queueWaitSeconds: 60, estimateSeconds: 120 }),
    job('t3', { estimateSeconds: 20 }),
  ];
  await page.route('**/api/jobs', (r) => (r.request().method() === 'GET' ? r.fulfill({ json: { jobs: list } }) : r.continue()));
  try {
    await page.goto(`${BASE}/app`);
    await page.getByTestId('job-row').first().waitFor();
    const line = (id) => page.locator(`[data-testid=job-row][data-job-id=${id}] [data-testid=row-line]`).innerText().then((t) => t.trim());
    const got = { t1: await line('t1'), t2: await line('t2'), t3: await line('t3') };
    if (got.t1 !== 'Cleaning the sound · 50% · about 2 min left') throw new Error('running row: ' + got.t1);
    if (got.t2 !== 'In line · about 3 min left') throw new Error('queued row: ' + got.t2);
    if (got.t3 !== 'Cleaning the sound · 50% · less than a minute left') throw new Error('short row: ' + got.t3);
    await page.goto(`${BASE}/app/job/t1`);
    const card = (await page.getByTestId('job-line').innerText()).trim();
    if (card !== 'Cleaning the sound · 50% · about 2 min left') throw new Error('job page: ' + card);
    await page.goto(`${BASE}/app/job/t2`);
    const queued = (await page.getByTestId('queue-note').innerText()).trim();
    if (!/^In line: 1 ahead of you, about 1 min · about 3 min left$/.test(queued)) throw new Error('queued job page: ' + queued);
    return Object.values(got).join(' | ');
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
