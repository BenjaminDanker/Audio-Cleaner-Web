// The Files page and the sign-in rule for the free tool: signed-out /free fetches nothing from /dfn/; the form asks for a
// file and then "Clean & polish" and/or "Subtitles & transcript"; subtitles-only + Standard + a short file is free
// (sends free:true, shows the free minutes, needs no credit); both boxes send denoise_transcribe and say where the transcript
// comes from; /transcribe no longer exists.
// Needs the local backend (port 8792, see helpers.mjs) with `npm run build` output, and the fixtures.
// Usage: node tests/files.mjs
import { join } from 'node:path';
import { BASE, FILES, SHOTS, admin, check, launch, loginLink, signIn, summary } from './helpers.mjs';

const EMAIL = `files-${Date.now()}@example.com`;
admin('login-link', EMAIL); // creates the user (no credit: free jobs must not need any)

const apiCall = (page, method, path, body) => page.evaluate(async ({ method, path, body }) => {
  const me = await (await fetch('/audio/api/me', { credentials: 'same-origin' })).json();
  const res = await fetch(`/audio/api/${path}`, {
    method, credentials: 'same-origin',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json', 'X-CSRF-Token': me.csrfToken },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  return { status: res.status, body: text ? JSON.parse(text) : null };
}, { method, path, body });

{
  const { browser, page } = await launch();
  const dfn = [];
  page.on('request', (r) => { if (new URL(r.url()).pathname.includes('/dfn/')) dfn.push(r.url()); });
  await check('signed-out /free shows the sign-in prompt and fetches nothing from /dfn/', async () => {
    await page.goto(`${BASE}/free`);
    await page.getByRole('heading', { name: 'Sign in' }).waitFor({ timeout: 10000 });
    if (!/\/login/.test(page.url())) throw new Error('not on /login: ' + page.url());
    if (await page.locator('input[type=file]').count()) throw new Error('file picker visible while signed out');
    await page.waitForTimeout(1500);
    if (dfn.length) throw new Error('fetched: ' + dfn.join(', '));
    return 'sign-in page, 0 requests to /dfn/';
  });
  await check('signing in from /free comes back to /free (email link in the same tab)', async () => {
    await page.goto(loginLink(`return-${Date.now()}@example.com`));
    await page.waitForURL(/\/audio\/free$/, { timeout: 15000 });
    await page.locator('input[type=file]').first().waitFor({ state: 'attached', timeout: 10000 });
    return page.url();
  });
  await check('/transcribe is gone: the not-found page, no redirect', async () => {
    await page.goto(`${BASE}/transcribe`);
    await page.getByRole('heading', { name: /couldn.t find that page/ }).waitFor({ timeout: 10000 });
    if (!/\/transcribe$/.test(page.url())) throw new Error('redirected to ' + page.url());
    const links = await page.locator('a[href$="/transcribe"]').count();
    if (links) throw new Error('a link to /transcribe is still on the page');
    return '404 page';
  });
  await check('nav: Free cleanup, Files, Live, Pricing (no Studio, no Transcribe)', async () => {
    await page.goto(`${BASE}/`);
    const nav = (await page.locator('#main-nav').innerText()).split('\n').map((x) => x.trim()).filter(Boolean);
    if (JSON.stringify(nav) !== JSON.stringify(['Free cleanup', 'Files', 'Live', 'Pricing'])) throw new Error(JSON.stringify(nav));
    return nav.join(' | ');
  });
  await check('landing: product cards point to /free, /app, /app/live, /app; hero line', async () => {
    await page.goto(`${BASE}/`);
    await page.waitForSelector('.product-card');
    const cards = await page.locator('.product-card').evaluateAll((els) => els.map((e) => [e.querySelector('h3').textContent.replace(/\s+/g, ' ').trim(), new URL(e.querySelector('a').href).pathname]));
    const want = [['Free cleanup Free', '/audio/free'], ['Clean & polish', '/audio/app'], ['Live for streamers', '/audio/app/live'], ['Subtitles & transcripts Free every day', '/audio/app']];
    if (JSON.stringify(cards) !== JSON.stringify(want)) throw new Error(JSON.stringify(cards));
    if (!(await page.locator('.hero').innerText()).includes('Nothing is uploaded. No watermark.')) throw new Error('hero line');
    return 'ok';
  });
  await browser.close();
}

{
  const { browser, page } = await launch();
  await signIn(page, EMAIL);
  await page.goto(`${BASE}/app`);
  await page.getByRole('heading', { name: 'Your files' }).waitFor();
  await apiCall(page, 'POST', 'account/tos', { version: '2026-10-10' });
  const posts = [];
  // Record every job creation and stop it there with an error, so no job is made (except in the one real run further down).
  let passThrough = false;
  await page.route('**/api/jobs', (r) => {
    if (r.request().method() !== 'POST' || passThrough) return r.continue();
    posts.push(r.request().postDataJSON());
    return r.fulfill({ status: 400, json: { detail: 'Stopped by the test.' } });
  });
  // The boxes are drawn as cards (the input itself is invisible), so a person clicks the card; the tests do the same.
  const card = (name) => {
    const box = page.getByRole('checkbox', { name });
    const lab = page.locator('label.choice', { has: box });
    return {
      isChecked: () => box.isChecked(),
      check: async () => { if (!(await box.isChecked())) await lab.click(); },
      uncheck: async () => { if (await box.isChecked()) await lab.click(); },
    };
  };
  const polish = () => card(/Clean & polish/);
  const subs = () => card(/Subtitles & transcript/);
  const choose = async (f) => { await page.setInputFiles('input[type=file]', join(FILES, f)); await page.getByRole('button', { name: 'Remove' }).waitFor(); await page.waitForTimeout(400); };

  await check('Files page: title, intro, no Studio, no link to a separate transcription page, no "Free transcription" button', async () => {
    const main = await page.locator('main').innerText();
    if (!main.includes('Clean & polish recordings, and get subtitles and transcripts.')) throw new Error('intro missing');
    if (/Studio|Free transcription/.test(main)) throw new Error('leftover wording');
    if (await page.locator('a[href$="/transcribe"]').count()) throw new Error('link to /transcribe');
    if (!(await polish().isChecked()) || (await subs().isChecked())) throw new Error('defaults: Clean & polish on, subtitles off');
    return 'ok';
  });

  await check('subtitles only + Standard on a short file: "Free: uses N of your M free minutes left today", sends free:true, needs no credit', async () => {
    await choose('noisy.wav');
    await polish().uncheck();
    await subs().check();
    const note = page.getByTestId('free-note');
    await note.waitFor();
    const t = await note.innerText();
    if (!/^Free: uses 1 of your 60 free minutes left today$/.test(t.trim())) throw new Error(t);
    if (await page.getByTestId('estimate').count()) throw new Error('a price is shown for a free job');
    const btn = page.getByRole('button', { name: 'Start (free)' });
    if (!(await btn.isEnabled())) throw new Error('start disabled although the job is free');
    await page.screenshot({ path: join(SHOTS, 'files-free.png'), fullPage: true });
    await btn.click();
    await page.getByTestId('job-error').waitFor();
    const b = posts.at(-1);
    if (b.free !== true || b.kind !== 'transcribe' || b.transcribeTier !== 'local' || b.denoiseTier) throw new Error(JSON.stringify(b));
    return t.trim();
  });

  await check('both boxes: sends kind denoise_transcribe (paid, estimate shown) and says the transcript comes from the original recording', async () => {
    await page.route('**/api/me', async (r) => { const res = await r.fetch(); const j = await res.json(); await r.fulfill({ response: res, json: { ...j, balance: { ...j.balance, available_micros: 5_000_000, balance_micros: 5_000_000 } } }); });
    await page.goto(`${BASE}/app`);
    await choose('noisy.wav');
    await subs().check();
    const sentence = 'We write the transcript from your original recording (it comes out more accurate), and clean the sound separately.';
    await page.getByText(sentence, { exact: true }).waitFor();
    await page.getByTestId('estimate').waitFor();
    if (await page.getByTestId('free-note').count()) throw new Error('a paid job is shown as free');
    await page.screenshot({ path: join(SHOTS, 'files-both.png'), fullPage: true });
    await page.getByRole('button', { name: /Start job/ }).click();
    await page.getByTestId('job-error').waitFor();
    const b = posts.at(-1);
    if (b.free !== false || b.kind !== 'denoise_transcribe' || b.denoiseTier !== 'standard' || b.transcribeTier !== 'local') throw new Error(JSON.stringify(b));
    return 'denoise_transcribe, free:false';
  });

  await check('Clean & polish alone sends kind denoise; at least one box is required; Extra accurate is paid', async () => {
    await page.reload();
    await choose('noisy.wav');
    await page.getByRole('button', { name: /Start job/ }).click();
    await page.getByTestId('job-error').waitFor();
    if (posts.at(-1).kind !== 'denoise' || posts.at(-1).free !== false || posts.at(-1).transcribeTier) throw new Error(JSON.stringify(posts.at(-1)));
    await polish().uncheck();
    await page.getByText(/Choose at least one/).waitFor();
    if (await page.getByRole('button', { name: /Start/ }).isEnabled()) throw new Error('start enabled with nothing chosen');
    await subs().check();
    await page.locator('label.choice', { hasText: 'Extra accurate' }).click();
    await page.getByTestId('estimate').waitFor();
    await page.getByRole('button', { name: /Start job/ }).click();
    await page.getByTestId('job-error').waitFor();
    if (posts.at(-1).free !== false || posts.at(-1).kind !== 'transcribe' || posts.at(-1).transcribeTier !== 'premium') throw new Error(JSON.stringify(posts.at(-1)));
    return 'ok';
  });

  await check('not free when close: shows why (free minutes left today) and charges the usual rate', async () => {
    await page.unroute('**/api/me');
    await page.route('**/api/me', async (r) => { const res = await r.fetch(); const j = await res.json(); await r.fulfill({ response: res, json: { ...j, freeTranscribeSecondsLeft: 5 } }); });
    await page.goto(`${BASE}/app`);
    await choose('noisy.wav');
    await polish().uncheck();
    await subs().check();
    const note = await page.getByTestId('not-free-note').innerText();
    if (!/isn.t free because it.s more than the 1 free minute you have left today/.test(note)) throw new Error(note);
    if (await page.getByTestId('free-note').count()) throw new Error('shown as free');
    return note.trim();
  });

  await check('video file: "Embed the subtitles in the video" is offered; an audio file does not get it', async () => {
    await page.unroute('**/api/me');
    await page.goto(`${BASE}/app`);
    await choose('noisy.wav');
    await subs().check();
    if (await page.getByRole('checkbox', { name: /Embed the subtitles/ }).count()) throw new Error('offered for audio');
    await page.getByRole('button', { name: 'Remove' }).click();
    await choose('test.mp4');
    const box = page.getByRole('checkbox', { name: /Embed the subtitles/ });
    await box.waitFor();
    return 'ok';
  });

  await check('a real free job: Free badge on the card, and the free minutes left go down afterwards', async () => {
    passThrough = true;
    await page.goto(`${BASE}/app`);
    await choose('noisy.wav');
    await polish().uncheck();
    await subs().check();
    await page.getByTestId('free-note').waitFor();
    await page.getByRole('button', { name: 'Start (free)' }).click();
    await page.getByTestId('submitted').waitFor({ timeout: 60000 });
    await page.getByTestId('free-badge').first().waitFor();
    await choose('noisy.wav');
    await polish().uncheck();
    await subs().check();
    const t = (await page.getByTestId('free-note').innerText()).trim();
    if (!/^Free: uses 1 of your 59 free minutes left today$/.test(t)) throw new Error(t);
    await page.screenshot({ path: join(SHOTS, 'files-after-free.png'), fullPage: true });
    // let the job finish or cancel it so the worker is not left busy for the next suite
    const id = await page.locator('[data-testid=job]').first().getAttribute('data-job-id');
    await apiCall(page, 'POST', `jobs/${id}/cancel`).catch(() => {});
    return t;
  });

  await check('a format the browser cannot open (WMA): accepted, honest "rough guess" note, and the server cleans it', async () => {
    passThrough = true;
    admin('grant', EMAIL, '5', 'files test: WMA');
    await page.goto(`${BASE}/app`);
    await choose('noisy.wma');
    await page.getByTestId('length-unknown').waitFor({ timeout: 15000 });
    await polish().check();
    await subs().uncheck();
    await page.getByRole('button', { name: /^Start/ }).click();
    await page.getByTestId('submitted').waitFor({ timeout: 60000 });
    const id = await page.locator('[data-testid=job]').first().getAttribute('data-job-id');
    for (let i = 0; i < 90; i++) {
      const j = await apiCall(page, 'GET', `jobs/${id}`);
      if (j.body?.status === "done") return `done: ${j.body.outputs.map((o) => o.name).join(", ")} (${Math.round(j.body.durationSeconds)} s measured)`;
      if (["failed", "canceled"].includes(j.body?.status)) throw new Error(JSON.stringify(j.body));
      await page.waitForTimeout(1000);
    }
    throw new Error('job did not finish');
  });

  await browser.close();
}

process.exit(summary().some((r) => !r.ok) ? 1 : 0);
