// The Cleanup page: signed-out /app fetches nothing from /dfn/; after a file is dropped only the options that are possible
// for it are offered (with a short reason for the others): noise removal Off | Free, in your browser | Clean & polish, and
// subtitles Off | Free (standard) | Standard | Extra accurate. Subtitles-only + Free + a short file sends free:true and needs no
// credit; Clean & polish + a subtitles level sends denoise_transcribe; one Start button says what it costs; Balance and
// "Good to know" sit below "Your jobs"; /transcribe and /free no longer exist.
// Needs the local backend (port 8792, see helpers.mjs) with `npm run build` output, and the fixtures.
// Usage: node tests/files.mjs
import { join } from 'node:path';
import { BASE, FILES, SHOTS, admin, check, dropFile, launch, loginLink, pick, signIn, summary } from './helpers.mjs';

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
  await check('signed-out /app shows the sign-in prompt and fetches nothing from /dfn/', async () => {
    await page.goto(`${BASE}/app`);
    await page.getByRole('heading', { name: 'Sign in' }).waitFor({ timeout: 10000 });
    if (!/\/login/.test(page.url())) throw new Error('not on /login: ' + page.url());
    if (await page.locator('input[type=file]').count()) throw new Error('file picker visible while signed out');
    await page.waitForTimeout(1500);
    if (dfn.length) throw new Error('fetched: ' + dfn.join(', '));
    return 'sign-in page, 0 requests to /dfn/';
  });
  await check('signing in from /app comes back to /app (email link in the same tab)', async () => {
    await page.goto(loginLink(`return-${Date.now()}@example.com`));
    await page.waitForURL(/\/audio\/app$/, { timeout: 15000 });
    await page.locator('input[type=file]').first().waitFor({ state: 'attached', timeout: 10000 });
    return page.url();
  });
  for (const gone of ['transcribe', 'free']) {
    await check(`/${gone} is gone: the not-found page, no redirect, no link to it`, async () => {
      await page.goto(`${BASE}/${gone}`);
      await page.getByRole('heading', { name: /couldn.t find that page/ }).waitFor({ timeout: 10000 });
      if (!new RegExp(`/${gone}$`).test(page.url())) throw new Error('redirected to ' + page.url());
      if (await page.locator(`a[href$="/${gone}"]`).count()) throw new Error(`a link to /${gone} is still on the page`);
      return '404 page';
    });
  }
  await check('nav: Cleanup, Live, Pricing (no Free cleanup tab, no Files, no Studio)', async () => {
    await page.goto(`${BASE}/`);
    const nav = (await page.locator('#main-nav').innerText()).split('\n').map((x) => x.trim()).filter(Boolean);
    if (JSON.stringify(nav) !== JSON.stringify(['Cleanup', 'Live', 'Pricing'])) throw new Error(JSON.stringify(nav));
    return nav.join(' | ');
  });
  await check('landing: product cards point to /app, /app, /app/live, /app (nothing to /free); hero line; availability answer', async () => {
    await page.goto(`${BASE}/`);
    await page.waitForSelector('.product-card');
    const cards = await page.locator('.product-card').evaluateAll((els) => els.map((e) => [e.querySelector('h3').textContent.replace(/\s+/g, ' ').trim(), new URL(e.querySelector('a').href).pathname]));
    const want = [['Free, in your browser Free', '/audio/app'], ['Clean & polish', '/audio/app'], ['Live for streamers', '/audio/app/live'], ['Subtitles & transcripts Free every day', '/audio/app']];
    if (JSON.stringify(cards) !== JSON.stringify(want)) throw new Error(JSON.stringify(cards));
    if (await page.locator('a[href$="/free"]').count()) throw new Error('a link to /free on the landing page');
    if (!(await page.locator('.hero').innerText()).includes('Nothing is uploaded. No watermark.')) throw new Error('hero line');
    const faq = await page.locator('.faq').textContent();
    if (!/isn.t offered in the EU\/EEA, the UK or Switzerland/.test(faq)) throw new Error('availability answer missing');
    return 'ok';
  });
  await check('Terms and Privacy say the service is not available in the EU/EEA, the UK or Switzerland, and that those cards are not charged', async () => {
    for (const p of ['/terms', '/privacy']) {
      await page.goto(`${BASE}${p}`);
      const t = (await page.locator('main').innerText()).replace(/\s+/g, ' ');
      if (!/not available to people in the (European Union or European Economic Area \(EU\/EEA\)|EU\/EEA), the (United Kingdom|UK) or Switzerland/.test(t)) throw new Error(p + ': availability sentence missing');
      if (!/card issued in one of those places is used, we cancel the payment before it is charged/.test(t)) throw new Error(p + ': not-charged sentence missing');
      if (/\/free\b/.test(await page.locator('main').innerHTML())) throw new Error(p + ' links to /free');
    }
    return 'ok';
  });
  await browser.close();
}

{
  const { browser, page } = await launch();
  await signIn(page, EMAIL);
  await page.goto(`${BASE}/app`);
  await page.getByRole('heading', { name: 'Cleanup', exact: true }).waitFor();
  await apiCall(page, 'POST', 'account/tos', { version: '2026-10-10' });
  const posts = [];
  // Record every job creation and stop it there with an error, so no job is made (except in the real runs further down).
  let passThrough = false;
  let forceEcho = false; // a real run where the form (rightly) never asks for echo removal: add it on the way out
  await page.route('**/api/jobs', (r) => {
    if (r.request().method() === 'POST' && passThrough && forceEcho) {
      const body = r.request().postDataJSON();
      return r.continue({ postData: JSON.stringify({ ...body, options: { ...body.options, echo_reference_track: true } }) });
    }
    if (r.request().method() !== 'POST' || passThrough) return r.continue();
    posts.push(r.request().postDataJSON());
    return r.fulfill({ status: 400, json: { detail: 'Stopped by the test.' } });
  });
  const radio = (group, value) => page.locator(`input[name=${group}][value=${value}]`);
  const card = (group, value) => page.locator(`label.choice:has(input[name=${group}][value=${value}])`);
  const start = () => page.getByTestId('start');
  const choose = async (f) => { await dropFile(page, join(FILES, f)); };
  const freshApp = async () => { await page.goto(`${BASE}/app`); await page.getByTestId('available').waitFor(); };
  const withMe = (patch) => page.route('**/api/me', async (r) => { const res = await r.fetch(); const j = await res.json(); await r.fulfill({ response: res, json: patch(j) }); });
  const withPricing = (patch) => page.route('**/api/billing/pricing', async (r) => { const res = await r.fetch(); const j = await res.json(); await r.fulfill({ response: res, json: patch(j) }); });

  await check('Cleanup page: title, no tab for free cleanup, Balance and "Good to know" come AFTER "Your jobs", the old privacy line is gone', async () => {
    if ((await page.title()).split(' · ')[0] !== 'Cleanup') throw new Error('title: ' + (await page.title()));
    const main = await page.locator('main').innerText();
    if (/Private by design/.test(main)) throw new Error('the old "Private by design" line is still there');
    const y = async (name) => (await page.getByRole('heading', { name, exact: true }).boundingBox()).y;
    const [form, jobs, bal, tips] = [await y('Add a file'), await y('Your jobs'), await y('Balance'), await y('Good to know')];
    if (!(form < jobs && jobs < bal && jobs < tips)) throw new Error(`order: form ${form}, jobs ${jobs}, balance ${bal}, tips ${tips}`);
    if (await page.locator('main input[type=checkbox][name=polish], main input[name=text]').count()) throw new Error('the two coarse boxes are back');
    if (await page.locator('input[name=noise]').count()) throw new Error('options are shown before a file is dropped');
    return `order: form < jobs < balance, tips`;
  });

  await check('a file the browser can open (WAV): noise removal Off | Free, in your browser (picked) | Clean & polish; subtitles Off | Free (standard) | Standard | Extra accurate; Start – free', async () => {
    await choose('noisy.wav');
    const titles = async (g) => (await page.locator(`label.choice:has(input[name=${g}]) b`).allInnerTexts()).map((x) => x.replace(/\s*\u2713$/, ''));
    const n = await titles('noise'); const t = await titles('subs');
    if (JSON.stringify(n) !== JSON.stringify(['Off', 'Free, in your browser', 'Clean & polish'])) throw new Error(JSON.stringify(n));
    if (JSON.stringify(t) !== JSON.stringify(['Off', 'Free (standard)', 'Standard', 'Extra accurate'])) throw new Error(JSON.stringify(t));
    for (const [g, v] of [['noise', 'off'], ['noise', 'browser'], ['noise', 'polish'], ['subs', 'free'], ['subs', 'standard'], ['subs', 'accurate']]) if (await radio(g, v).isDisabled()) throw new Error(`${g}/${v} disabled`);
    if (!(await radio('noise', 'browser').isChecked()) || !(await radio('subs', 'off').isChecked())) throw new Error('defaults: free noise removal on, subtitles off');
    if ((await start().innerText()).trim() !== 'Start – free') throw new Error('label: ' + (await start().innerText()));
    if (!/Extra accurate[\s\S]*any language/.test(await card('subs', 'accurate').innerText())) throw new Error('"any language" missing on Extra accurate');
    await page.screenshot({ path: join(SHOTS, 'cleanup-options.png'), fullPage: true });
    return 'ok';
  });

  await check('noise strength: the same four buttons as the desktop app (Gentle, Balanced, Strong, Maximum), Maximum to start with; Clean & polish with Gentle sends 12 dB, Background noise defaults to Automatic and Heavy is sent; Live has the same four', async () => {
    const labels = async (name) => page.getByTestId(`strength-${name}`).locator('.seg span').allInnerTexts();
    const four = JSON.stringify(['Gentle', 'Balanced', 'Strong', 'Maximum']);
    if (JSON.stringify(await labels('lstrength')) !== four) throw new Error('browser: ' + JSON.stringify(await labels('lstrength')));
    if (!(await radio('lstrength', 'max').isChecked())) throw new Error('browser: Maximum should be picked first');
    await withMe((j) => ({ ...j, balance: { ...j.balance, available_micros: 5_000_000, balance_micros: 5_000_000 } })); // enough for the estimate; the job itself is stopped by the test
    await freshApp();
    await choose('noisy.wav');
    await pick(page, 'noise', 'polish');
    if (JSON.stringify(await labels('strength')) !== four) throw new Error('polish: ' + JSON.stringify(await labels('strength')));
    await page.getByTestId('strength-strength').getByText('Gentle', { exact: true }).click();
    if (!/room sound/.test(await page.getByTestId('strength-strength').innerText())) throw new Error('no hint for Gentle');
    posts.length = 0;
    await start().click();
    await page.getByTestId('job-error').waitFor();
    const sent = posts.at(-1)?.options?.atten_lim_db;
    if (sent !== 12) throw new Error('atten_lim_db sent: ' + sent);
    if (posts.at(-1)?.options?.noise !== 'auto') throw new Error('noise should default to auto: ' + posts.at(-1)?.options?.noise);
    await page.locator('#noisekind').selectOption('heavy');
    if (!/café|traffic/.test(await page.locator('#noisekind').locator('xpath=..').innerText())) throw new Error('no hint for Heavy noise');
    posts.length = 0;
    await start().click();
    await page.getByTestId('job-error').waitFor();
    if (posts.at(-1)?.options?.noise !== 'heavy') throw new Error('noise sent: ' + posts.at(-1)?.options?.noise);
    // "Rebuild the voice": its own price tier; strength and Background noise do not apply then
    await page.getByRole('checkbox', { name: /Rebuild the voice/ }).check();
    if (await page.locator('#noisekind').count() || await page.getByTestId('strength-strength').count()) throw new Error('strength/noise still shown with rebuild');
    if (!/an hour/.test(await page.getByTestId('rebuild-hint').innerText())) throw new Error('rebuild price missing');
    posts.length = 0;
    const [req] = await Promise.all([
      page.waitForRequest((r) => r.url().endsWith('/api/jobs') && r.method() === 'POST'),
      start().click(),
    ]);
    if (req.postDataJSON()?.denoiseTier !== 'rebuild') throw new Error('denoiseTier sent: ' + req.postDataJSON()?.denoiseTier);
    await page.getByRole('checkbox', { name: /Rebuild the voice/ }).uncheck();
    await page.unroute('**/api/me');
    await page.goto(`${BASE}/app/live`);
    await page.getByTestId('strength-live-str').waitFor();
    if (JSON.stringify(await labels('live-str')) !== four) throw new Error('live: ' + JSON.stringify(await labels('live-str')));
    await freshApp();
    await choose('noisy.wav'); // the next checks continue with a WAV in the form
    return 'Gentle -> 12 dB';
  });

  await check('subtitles only + Free on a short file: "Free: uses N of your M free minutes left today", Start – free, sends free:true, needs no credit', async () => {
    await pick(page, 'noise', 'off');
    await pick(page, 'subs', 'free');
    const note = page.getByTestId('free-note');
    await note.waitFor();
    const t = await note.innerText();
    if (!/^Free: uses 1 of your 60 free minutes left today$/.test(t.trim())) throw new Error(t);
    if (await page.getByTestId('estimate').count()) throw new Error('a price is shown for a free job');
    if ((await start().innerText()).trim() !== 'Start – free' || !(await start().isEnabled())) throw new Error('start: ' + (await start().innerText()));
    await start().click();
    await page.getByTestId('job-error').waitFor();
    const b = posts.at(-1);
    if (b.free !== true || b.kind !== 'transcribe' || b.transcribeTier !== 'local' || b.denoiseTier) throw new Error(JSON.stringify(b));
    return t.trim();
  });

  await check('Clean & polish + Standard: one denoise_transcribe job (paid, estimate, "about $" on Start); Free subtitles is greyed out with a reason', async () => {
    await withMe((j) => ({ ...j, balance: { ...j.balance, available_micros: 5_000_000, balance_micros: 5_000_000 } }));
    await freshApp();
    await choose('noisy.wav');
    await pick(page, 'noise', 'polish');
    await pick(page, 'subs', 'standard');
    const sentence = 'We write the transcript from your original recording (it comes out more accurate), and clean the sound separately.';
    await page.getByText(sentence, { exact: true }).waitFor();
    await page.getByTestId('estimate').waitFor();
    if (await page.getByTestId('free-note').count()) throw new Error('a paid job is shown as free');
    if (!(await radio('subs', 'free').isDisabled())) throw new Error('free subtitles should be disabled together with Clean & polish');
    const why = await card('subs', 'free').locator('small').innerText();
    if (!/can.t be added to Clean & polish/.test(why)) throw new Error('reason: ' + why);
    const label = (await start().innerText()).trim();
    if (!/^Start – about \$0\.00\d+$/.test(label)) throw new Error('label: ' + label);
    await page.screenshot({ path: join(SHOTS, 'cleanup-both.png'), fullPage: true });
    await start().click();
    await page.getByTestId('job-error').waitFor();
    const b = posts.at(-1);
    if (b.free !== false || b.kind !== 'denoise_transcribe' || b.denoiseTier !== 'standard' || b.transcribeTier !== 'local') throw new Error(JSON.stringify(b));
    return label;
  });

  await check('Clean & polish alone sends kind denoise; Extra accurate alone sends premium (paid); nothing chosen disables Start with the reason', async () => {
    await page.reload();
    await choose('noisy.wav');
    await pick(page, 'noise', 'polish');
    await start().click();
    await page.getByTestId('job-error').waitFor();
    if (posts.at(-1).kind !== 'denoise' || posts.at(-1).free !== false || posts.at(-1).transcribeTier) throw new Error(JSON.stringify(posts.at(-1)));
    await pick(page, 'noise', 'off');
    await page.getByText(/Choose at least one thing/).waitFor();
    if (await start().isEnabled()) throw new Error('start enabled with nothing chosen');
    await pick(page, 'subs', 'accurate');
    await page.getByTestId('accurate-note').waitFor();
    await page.getByTestId('estimate').waitFor();
    await start().click();
    await page.getByTestId('job-error').waitFor();
    if (posts.at(-1).free !== false || posts.at(-1).kind !== 'transcribe' || posts.at(-1).transcribeTier !== 'premium') throw new Error(JSON.stringify(posts.at(-1)));
    return 'ok';
  });

  await check('Free noise removal in the browser + Standard subtitles: the job is a paid transcribe-only job (the browser part is free)', async () => {
    await freshApp();
    await choose('noisy.wav');
    await pick(page, 'subs', 'standard');
    await page.getByTestId('estimate').waitFor();
    await start().click();
    await page.getByTestId('job-error').waitFor();
    const b = posts.at(-1);
    if (b.free !== false || b.kind !== 'transcribe' || b.denoiseTier || b.transcribeTier !== 'local') throw new Error(JSON.stringify(b));
    await page.waitForSelector('[data-testid=result]', { timeout: 60000 }); // and the cleaned sound is ready to download
    return 'transcribe job + browser result';
  });

  await check('balance too low: Start is disabled, the reason is shown with an "Add credit" link', async () => {
    await page.unroute('**/api/me');
    await freshApp();
    await choose('noisy.wav');
    await pick(page, 'noise', 'polish');
    const msg = page.locator('.banner-warn', { hasText: /your balance is/ });
    await msg.waitFor();
    if (!(await msg.getByRole('link', { name: 'Add credit' }).count())) throw new Error('no Add credit link');
    if (await start().isEnabled()) throw new Error('start enabled with too little balance');
    return (await msg.innerText()).trim();
  });

  await check('Free subtitles greyed out with a reason when the file is over the free limit, and when today\'s free minutes are too few', async () => {
    await page.unroute('**/api/billing/pricing').catch(() => {});
    await withPricing((j) => ({ ...j, free: { ...j.free, transcribeMaxFileSeconds: 5 } }));
    await freshApp();
    await choose('noisy.wav');
    if (!(await radio('subs', 'free').isDisabled())) throw new Error('free subtitles enabled for a file over the limit');
    const why = await card('subs', 'free').locator('small').innerText();
    if (!/longer than the 1-minute free limit/.test(why)) throw new Error('reason: ' + why);
    await page.unroute('**/api/billing/pricing');
    await withMe((j) => ({ ...j, freeTranscribeSecondsLeft: 5 }));
    await freshApp();
    await choose('noisy.wav');
    if (!(await radio('subs', 'free').isDisabled())) throw new Error('free subtitles enabled with too few minutes left');
    const why2 = await card('subs', 'free').locator('small').innerText();
    if (!/more than the 1 minute of free time you have left today/.test(why2)) throw new Error('reason: ' + why2);
    await pick(page, 'noise', 'off');
    await pick(page, 'subs', 'standard'); // still available at the usual price
    await page.getByTestId('estimate').waitFor();
    await page.unroute('**/api/me');
    return `${why} | ${why2}`;
  });

  await check('a file the browser cannot open at all: Free noise removal is greyed out with the reason, Clean & polish is chosen, honest "rough guess" note', async () => {
    await freshApp();
    await choose('garbage.mp3');
    if (!(await radio('noise', 'browser').isDisabled())) throw new Error('free noise removal offered for a file the browser cannot open');
    const why = (await card('noise', 'browser').locator('small').innerText()).trim();
    if (why !== 'Your browser can’t open this file type. Clean & polish can (paid).') throw new Error('reason: ' + why);
    if (!(await radio('noise', 'polish').isChecked())) throw new Error('Clean & polish should be the starting choice');
    await page.getByTestId('length-unknown').waitFor({ timeout: 15000 });
    await page.screenshot({ path: join(SHOTS, 'cleanup-unreadable.png'), fullPage: true });
    return why;
  });

  await check('video file: "Keep the video, with the cleaned sound" and "Embed subtitles in the video" are offered; an audio file gets neither', async () => {
    await freshApp();
    await choose('noisy.wav');
    await pick(page, 'subs', 'standard');
    if (await page.getByRole('checkbox', { name: /Embed subtitles/ }).count()) throw new Error('embed offered for audio');
    if (await page.getByRole('checkbox', { name: /Keep the video/ }).count()) throw new Error('keep-video offered for audio');
    await page.getByRole('button', { name: 'Remove' }).click();
    await choose('test.mp4');
    await page.getByRole('checkbox', { name: /Keep the video, with the cleaned sound/ }).waitFor();
    await pick(page, 'subs', 'standard');
    await page.getByRole('checkbox', { name: /Embed subtitles in the video/ }).waitFor();
    return 'ok';
  });

  await check('two audio tracks (mic + desktop sound): the echo box appears only for Clean & polish; free cleanup says it cannot do it; one-track files never see it', async () => {
    await withMe((j) => ({ ...j, balance: { ...j.balance, available_micros: 5_000_000, balance_micros: 5_000_000 } }));
    const box = page.getByRole('checkbox', { name: /Remove speaker echo using the second audio track \(your desktop sound\)/ });
    await freshApp();
    await choose('noisy.wav');
    await pick(page, 'noise', 'polish');
    if (await box.count()) throw new Error('echo box offered for a one-track file');
    await page.getByRole('button', { name: 'Remove' }).click();
    for (const f of ['echo2.mp4', 'echo2.mka']) {
      await choose(f);
      if (!(await radio('noise', 'browser').isChecked())) throw new Error(`${f}: free cleanup should be the starting choice`);
      const note = (await page.getByTestId('echo-browser-note').innerText()).replace(/\s+/g, ' ');
      if (!/second audio track[\s\S]*needs Clean & polish[\s\S]*free cleanup in your browser can.t do it/.test(note)) throw new Error('note: ' + note);
      if (await box.count()) throw new Error(`${f}: the echo box shows for free cleanup`);
      await pick(page, 'noise', 'polish');
      await box.waitFor();
      if (await box.isChecked()) throw new Error('echo box should start unticked');
      if (!/OBS can record your mic and your desktop audio on separate tracks/.test(await page.locator('label.check:has(input[name=echo])').innerText())) throw new Error('hint missing');
      if (await page.getByTestId('echo-browser-note').count()) throw new Error('browser note still shown for Clean & polish');
      if (f === 'echo2.mp4') await page.screenshot({ path: join(SHOTS, 'cleanup-echo.png'), fullPage: true });
      await page.getByRole('button', { name: 'Remove' }).click();
    }
    const text = await page.locator('main').innerText();
    if (/AEC|WebRTC|LiveKit|echo cancel/i.test(text)) throw new Error('internals on the page');
    return 'ok';
  });

  await check('echo box ticked: the job asks for echo_reference_track, and a video goes up as ONE small file that keeps both audio tracks (.mka); unticked or free cleanup sends false', async () => {
    const box = page.getByRole('checkbox', { name: /Remove speaker echo/ });
    const size = (await import('node:fs')).statSync(join(FILES, 'echo2.mp4')).size;
    await choose('echo2.mp4');
    await pick(page, 'noise', 'polish');
    await start().click();
    await page.getByTestId('job-error').waitFor();
    let b = posts.at(-1);
    if (b.options.echo_reference_track !== false) throw new Error('unticked should send false: ' + JSON.stringify(b.options));
    await box.check();
    await start().click();
    await page.getByTestId('job-error').waitFor();
    b = posts.at(-1);
    if (b.kind !== 'denoise' || b.options.echo_reference_track !== true) throw new Error(JSON.stringify(b));
    if (!/\.mka$/.test(b.filename) || b.sizeBytes >= size) throw new Error(`upload: ${b.filename} ${b.sizeBytes} (original ${size})`);
    await pick(page, 'noise', 'browser'); // ticked, then switched away from Clean & polish: not sent
    await pick(page, 'subs', 'standard');
    await start().click();
    await page.getByTestId('job-error').waitFor();
    if (posts.at(-1).options.echo_reference_track !== false) throw new Error('echo requested without Clean & polish');
    await page.unroute('**/api/me');
    return `${b.filename}, ${b.sizeBytes} bytes of ${size}`;
  });

  await check('a real free job: Free badge on the card, and the free minutes left go down afterwards', async () => {
    passThrough = true;
    await freshApp();
    await choose('noisy.wav');
    await pick(page, 'noise', 'off');
    await pick(page, 'subs', 'free');
    await page.getByTestId('free-note').waitFor();
    await start().click();
    await page.getByTestId('submitted').waitFor({ timeout: 60000 });
    await page.getByTestId('free-badge').first().waitFor();
    await choose('noisy.wav');
    await pick(page, 'noise', 'off');
    await pick(page, 'subs', 'free');
    const t = (await page.getByTestId('free-note').innerText()).trim();
    if (!/^Free: uses 1 of your 59 free minutes left today$/.test(t)) throw new Error(t);
    await page.screenshot({ path: join(SHOTS, 'cleanup-after-free.png'), fullPage: true });
    // let the job finish or cancel it so the worker is not left busy for the next suite
    const id = await page.locator('[data-testid=job]').first().getAttribute('data-job-id');
    await apiCall(page, 'POST', `jobs/${id}/cancel`).catch(() => {});
    return t;
  });

  await check('WMA (the browser reads it through libav.js): its real length is shown, and Clean & polish cleans it on the server', async () => {
    passThrough = true;
    admin('grant', EMAIL, '5', 'files test: WMA');
    await freshApp();
    await choose('noisy.wma');
    if (await page.getByTestId('length-unknown').count()) throw new Error('length should be known');
    if (!(await radio('noise', 'browser').isChecked())) throw new Error('free noise removal should be the starting choice');
    await pick(page, 'noise', 'polish');
    await page.getByTestId('estimate').waitFor();
    await start().click();
    await page.getByTestId('submitted').waitFor({ timeout: 60000 });
    const id = await page.locator('[data-testid=job]').first().getAttribute('data-job-id');
    for (let i = 0; i < 90; i++) {
      const j = await apiCall(page, 'GET', `jobs/${id}`);
      if (j.body?.status === 'done') return `done: ${j.body.outputs.map((o) => o.name).join(', ')} (${Math.round(j.body.durationSeconds)} s measured)`;
      if (['failed', 'canceled'].includes(j.body?.status)) throw new Error(JSON.stringify(j.body));
      await page.waitForTimeout(1000);
    }
    throw new Error('job did not finish');
  });

  await check('a real two-track job with echo removal: the server finds both tracks (no "only one track" note) and finishes', async () => {
    passThrough = true;
    admin('grant', EMAIL, '5', 'files test: echo');
    await freshApp();
    await choose('echo2.mp4');
    await pick(page, 'noise', 'polish');
    await page.getByRole('checkbox', { name: /Remove speaker echo/ }).check();
    await start().click();
    await page.getByTestId('submitted').waitFor({ timeout: 60000 });
    const id = await page.locator('[data-testid=job]').first().getAttribute('data-job-id');
    for (let i = 0; i < 90; i++) {
      const j = await apiCall(page, 'GET', `jobs/${id}`);
      if (j.body?.status === 'done') {
        if (j.body.options.echo_reference_track !== true || j.body.notice) throw new Error(JSON.stringify(j.body));
        if (!j.body.outputs.some((o) => /-cleaned\.\w+$/.test(o.name))) throw new Error(JSON.stringify(j.body.outputs));
        await page.screenshot({ path: join(SHOTS, 'cleanup-echo-done.png'), fullPage: true });
        return `done: ${j.body.outputs.map((o) => o.name).join(', ')}`;
      }
      if (['failed', 'canceled'].includes(j.body?.status)) throw new Error(JSON.stringify(j.body));
      await page.waitForTimeout(1000);
    }
    throw new Error('job did not finish');
  });

  await check('echo removal asked for a one-track file: the job still finishes, and its card says there was no second track to use', async () => {
    forceEcho = true;
    await freshApp();
    await choose('noisy.wav');
    await pick(page, 'noise', 'polish');
    await start().click();
    await page.getByTestId('submitted').waitFor({ timeout: 60000 });
    const notice = page.locator('[data-testid=job]').first().getByTestId('job-notice');
    await notice.waitFor({ timeout: 90000 });
    forceEcho = false;
    const t = (await notice.innerText()).trim();
    if (!/only one audio track/.test(t)) throw new Error(t);
    return t;
  });

  await browser.close();
}

process.exit(summary().some((r) => !r.ok) ? 1 : 0);
