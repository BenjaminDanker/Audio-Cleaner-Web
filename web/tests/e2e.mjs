// End-to-end checks against a running backend (see report). Usage: node tests/e2e.mjs [section ...]
import { readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { BASE, FILES, SHOTS, admin, check, launch, signIn, summary } from './helpers.mjs';

const only = process.argv.slice(2);
const want = (s) => only.length === 0 || only.includes(s);
const EMAIL = process.env.E2E_EMAIL || `e2e-${Date.now()}@example.com`;

const { browser, context, page, logs } = await launch();
const gotoApp = async (path) => { await page.goto(`${BASE}${path}`); await page.waitForSelector('main h1', { timeout: 15000 }); };

async function waitJob(name, timeout = 240000) {
  const card = page.locator('[data-testid=job]', { hasText: name }).first();
  await card.waitFor({ timeout: 30000 });
  await page.waitForFunction((n) => {
    const c = [...document.querySelectorAll('[data-testid=job]')].find((e) => e.textContent.includes(n));
    return c && ['done', 'failed', 'canceled'].includes(c.getAttribute('data-status'));
  }, name, { timeout });
  return card;
}

if (want('basic')) {
  await check('landing: pricing table from API, tiers, soon badges', async () => {
    await page.goto(`${BASE}/`);
    await page.waitForSelector('table tbody tr');
    const txt = await page.locator('table').innerText();
    for (const s of ['$0.0100', '$0.60', 'Coming soon', 'Studio clean', '$0.0040']) if (!txt.includes(s)) throw new Error('missing ' + s);
    const soon = await page.locator('.badge-soon').count();
    return `${await page.locator('table tbody tr').count()} rows, ${soon} coming-soon`;
  });
  await check('404 page + legal pages', async () => {
    await page.goto(`${BASE}/nope`);
    if (!(await page.textContent('h1')).includes('could not be found')) throw new Error('404 text');
    await page.goto(`${BASE}/terms`);
    if (!(await page.locator('.draft-banner').innerText()).includes('under legal review')) throw new Error('no draft banner');
    await page.goto(`${BASE}/privacy`); await page.waitForSelector('h1');
    await page.goto(`${BASE}/licenses`); await page.waitForSelector('table');
    return 'ok';
  });
  await check('login page: no providers configured message + error param', async () => {
    await page.goto(`${BASE}/login?error=state`);
    await page.waitForSelector('.banner-error');
    const t = await page.locator('main').innerText();
    if (!/not configured/i.test(t)) throw new Error('no config notice: ' + t.slice(0, 200));
    return 'ok';
  });
  await check('protected route redirects to /login when signed out', async () => {
    await page.goto(`${BASE}/app/billing`);
    await page.waitForURL(/\/audio\/login$/);
    return 'ok';
  });
}

if (want('auth')) {
  await check('login-link sign-in: hash cleared, lands on /app', async () => {
    admin('login-link', EMAIL); // creates the user
    const { loginLink } = await import('./helpers.mjs');
    const link = loginLink(EMAIL);
    await page.goto(link);
    await page.waitForURL(/\/audio\/app$/, { timeout: 15000 });
    const h = await page.evaluate(() => location.hash + '|' + document.referrer);
    if (page.url().includes('token')) throw new Error('token still in URL');
    await page.waitForSelector('[data-testid=available]');
    return `signed in as ${EMAIL}`;
  });
  await check('reused login link is rejected with friendly error', async () => {
    const { loginLink } = await import('./helpers.mjs');
    const ctx2 = await browser.newContext();
    const p2 = await ctx2.newPage();
    const link = loginLink('e2e-reuse@example.com');
    await p2.goto(link); await p2.waitForURL(/\/app$/);
    await p2.context().clearCookies();
    await p2.goto(link);
    await p2.waitForSelector('.banner-error', { timeout: 10000 });
    const t = await p2.locator('.banner-error').innerText();
    await ctx2.close();
    if (!/invalid or expired/i.test(t)) throw new Error(t);
    return t;
  });
  admin('grant', EMAIL, '5', 'e2e test');
}

if (want('studio')) {
  await check('dashboard: balance, ToS banner', async () => {
    await gotoApp('/app');
    await page.reload(); await page.waitForSelector('[data-testid=available]');
    const bal = await page.getByTestId('available').innerText();
    if (bal !== '$5.00') throw new Error('balance ' + bal);
    const hdr = await page.locator('.balance-chip').innerText();
    if (!/\$5\.00/.test(hdr)) throw new Error('header ' + hdr);
    if (!(await page.getByText('review and accept the Terms').count())) throw new Error('no ToS banner');
    return `balance ${bal}`;
  });

  await check('studio clean (wav) end-to-end incl. ToS modal', async () => {
    await page.setInputFiles('input[type=file]', join(FILES, 'noisy.wav'));
    await page.getByTestId('estimate').waitFor();
    const est = await page.getByTestId('estimate').innerText();
    if (!/\$0\.0[0-9]+/.test(est)) throw new Error('estimate: ' + est);
    await page.getByRole('button', { name: /Start job/ }).click();
    const dlg = page.getByRole('dialog');
    await dlg.waitFor();
    await dlg.getByRole('checkbox').check();
    await dlg.getByRole('button', { name: 'Accept and continue' }).click();
    await page.getByTestId('submitted').waitFor({ timeout: 60000 });
    const card = await waitJob('noisy.'); // uncompressed WAV is uploaded as noisy.flac
    const status = await card.getAttribute('data-status');
    if (status !== 'done') throw new Error('status ' + status + ' ' + (await card.innerText()));
    await page.screenshot({ path: join(SHOTS, 'studio-done.png'), fullPage: true });
    const [dl] = await Promise.all([page.waitForEvent('download'), card.getByTestId('download').first().click()]);
    const p = join(FILES, 'out-studio.wav');
    await dl.saveAs(p);
    return `${est.split('\n')[0]}; downloaded ${dl.suggestedFilename()} ${statSync(p).size} bytes; expiry: ${await card.getByTestId('expiry').innerText()}`;
  });

  await check('balance after job reflects cost', async () => {
    await page.waitForFunction(() => document.querySelector('[data-testid=available]')?.textContent !== '$5.00', null, { timeout: 15000 });
    return await page.getByTestId('available').innerText();
  });

  await check('studio clean of VIDEO: audio-only upload + local remux', async () => {
    await page.setInputFiles('input[type=file]', join(FILES, 'test.mp4'));
    await page.getByTestId('estimate').waitFor();
    await page.getByRole('button', { name: /Start job/ }).click();
    const msg = await page.getByTestId('submitted').innerText({ timeout: 60000 });
    if (!/Only the audio was uploaded/.test(msg)) throw new Error('msg: ' + msg);
    const card = await waitJob('test');
    const jobs = page.locator('[data-testid=job]', { hasText: /test\.m4a|test\.mp4/ }).first();
    const st = await jobs.getAttribute('data-status');
    if (st !== 'done') throw new Error('status ' + st + ' ' + (await jobs.innerText()));
    const names = await jobs.locator('.oname').allInnerTexts();
    const btn = jobs.getByTestId('remux-btn');
    await btn.waitFor();
    const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 60000 }), btn.click()]);
    const p = join(FILES, 'out-studio-video.mp4');
    await dl.saveAs(p);
    return `${msg}; outputs ${names.join(', ')}; remuxed ${dl.suggestedFilename()} ${statSync(p).size} bytes`;
  });

  await check('transcription + denoise (both) job: srt/vtt/txt outputs', async () => {
    await page.setInputFiles('input[type=file]', join(FILES, 'noisy.wav'));
    await page.getByLabel('Both').check().catch(() => page.locator('label:has(input[value=denoise_transcribe])').click());
    await page.getByTestId('estimate').waitFor();
    await page.getByRole('button', { name: /Start job/ }).click();
    await page.getByTestId('submitted').waitFor({ timeout: 60000 });
    await page.waitForTimeout(500);
    const card = page.locator('[data-testid=job]').first();
    await page.waitForFunction(() => ['done', 'failed'].includes(document.querySelector('[data-testid=job]').getAttribute('data-status')), null, { timeout: 300000 });
    const st = await card.getAttribute('data-status');
    if (st !== 'done') throw new Error(st + ' ' + (await card.innerText()));
    const names = await card.locator('.oname').allInnerTexts();
    return names.join(', ');
  });
}

if (want('transcribe')) {
  await check('free transcription page', async () => {
    await gotoApp('/transcribe');
    const intro = await page.locator('main header').innerText();
    if (!/60 free minutes a day/.test(intro) || !/30 minutes per file/.test(intro)) throw new Error(intro);
    await page.setInputFiles('input[type=file]', join(FILES, 'noisy.wav'));
    await page.getByRole('button', { name: 'Start free transcription' }).click();
    await page.getByTestId('submitted').waitFor({ timeout: 60000 });
    const card = await waitJob('noisy.', 300000);
    const st = await card.getAttribute('data-status');
    if (st !== 'done') throw new Error(st + ' ' + (await card.innerText()));
    const names = await card.locator('.oname').allInnerTexts();
    const [dl] = await Promise.all([page.waitForEvent('download'), card.locator('li', { hasText: '.txt' }).getByTestId('download').click()]);
    await dl.saveAs(join(FILES, 'out-transcript.txt'));
    const text = readFileSync(join(FILES, 'out-transcript.txt'), 'utf8').trim();
    return `${names.join(', ')} | "${text.slice(0, 80)}"`;
  });
  await check('balance unchanged by free job', async () => {
    const b = await page.locator('.balance-chip').innerText();
    return b;
  });
}

if (want('billing')) {
  await check('billing: payments off -> Pay, refund and card buttons are disabled with an explanation', async () => {
    await gotoApp('/app/billing');
    await page.waitForSelector('[data-testid=payments-off]');
    const pay = page.getByRole('button', { name: /^Pay / });
    if (!(await pay.isDisabled())) throw new Error('Pay button should be disabled while payments are off');
    const note = await page.getByTestId('topup-off').innerText();
    if (!/not available yet/i.test(note)) throw new Error(note);
    if ((await pay.getAttribute('aria-describedby')) !== 'topup-off') throw new Error('Pay button is not described by the note');
    if (!(await page.getByRole('button', { name: 'Manage card and receipts' }).isDisabled())) throw new Error('card portal button enabled');
    await page.screenshot({ path: join(SHOTS, 'billing.png'), fullPage: true });
    return note;
  });
  await check('billing: custom amount validation + auto-reload text + ledger', async () => {
    await page.locator('label:has(input[value=custom])').click();
    const hint = page.locator('#custom-hint');
    await page.getByLabel('Custom amount (USD)').fill('5');
    if (!/neg/.test(await hint.getAttribute('class'))) throw new Error('should be invalid under $10');
    await page.getByLabel('Custom amount (USD)').fill('600');
    if (!/neg/.test(await hint.getAttribute('class'))) throw new Error('should be invalid over $500');
    await page.getByLabel('Custom amount (USD)').fill('25');
    if (/neg/.test(await hint.getAttribute('class'))) throw new Error('$25 should be valid');
    if (!(await page.getByRole('button', { name: /^Pay/ }).isDisabled())) throw new Error('still disabled while payments are off');
    const text = await page.getByTestId('ar-consent-text').innerText();
    if (!/I authorize Audio Cleaner to charge my saved card \$20\.00 whenever my balance falls below \$5\.00, up to \$100\.00 per calendar month, until I turn this off\./.test(text)) throw new Error(text);
    const rows = await page.locator('[data-testid=ledger] tbody tr').count();
    return `ledger rows ${rows}`;
  });
  await check('billing: ?checkout=success banner polls balance', async () => {
    await page.goto(`${BASE}/app/billing?checkout=success`);
    await page.getByTestId('checkout-banner').waitFor();
    await page.goto(`${BASE}/app/billing?checkout=cancel`);
    const t = await page.getByTestId('checkout-banner').innerText();
    if (!/canceled/i.test(t)) throw new Error(t);
    return t;
  });
}

if (want('keys')) {
  await check('live: API key create (shown once, copy), list, revoke', async () => {
    await gotoApp('/app/live');
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    await page.getByLabel('Name').fill('e2e key');
    await page.getByRole('button', { name: 'Create key' }).click();
    const key = await page.getByTestId('key-value').innerText();
    if (!/^acw_/.test(key)) throw new Error(key);
    await page.getByRole('button', { name: 'Copy' }).click();
    const clip = await page.evaluate(() => navigator.clipboard.readText());
    if (clip !== key) throw new Error('clipboard mismatch');
    await page.reload();
    await page.waitForSelector('[data-testid=keys]');
    if (await page.getByTestId('new-key').count()) throw new Error('key still shown after reload');
    const row = page.locator('[data-testid=keys] li', { hasText: 'e2e key' });
    await row.getByRole('button', { name: /Revoke key/ }).click();
    await row.getByRole('button', { name: 'Revoke', exact: true }).click();
    await page.waitForFunction(() => !document.querySelector('[data-testid=keys]') || !document.querySelector('[data-testid=keys]').textContent.includes('e2e key'));
    return `${key.slice(0, 12)}… created, copied, revoked`;
  });
}

if (want('account')) {
  await check('account: export download + sign out everywhere', async () => {
    await gotoApp('/app/account');
    const [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download my data' }).click()]);
    const p = join(FILES, 'acw-export.json');
    await dl.saveAs(p);
    const j = JSON.parse(readFileSync(p, 'utf8'));
    if (!j.user || j.user.email !== EMAIL) throw new Error('bad export');
    if (dl.suggestedFilename() !== 'acw-account-export.json') throw new Error(dl.suggestedFilename());
    return `${Object.keys(j).join(',')}`;
  });
  await check('account: delete requires typing email; redirects to /', async () => {
    const { loginLink } = await import('./helpers.mjs');
    const ctx2 = await browser.newContext();
    const p2 = await ctx2.newPage();
    const mail = `e2e-del-${Date.now()}@example.com`;
    await p2.goto(loginLink(mail)); await p2.waitForURL(/\/app$/);
    await p2.goto(`${BASE}/app/account`);
    const btn = p2.getByRole('button', { name: 'Delete my account' });
    await btn.waitFor();
    if (!(await btn.isDisabled())) throw new Error('enabled too early');
    await p2.getByLabel(/Type your email/).fill('wrong@example.com');
    if (!(await btn.isDisabled())) throw new Error('enabled with wrong email');
    await p2.getByLabel(/Type your email/).fill(mail);
    await btn.click();
    await p2.waitForURL(/\/audio\/?$/, { timeout: 10000 });
    await p2.getByTestId('account-deleted').waitFor({ timeout: 10000 });
    await p2.getByText('Your account has been deleted.').waitFor();
    await p2.waitForFunction(() => document.querySelector('.site-header a[href$="/login"]'), null, { timeout: 10000 }); // header now shows "Sign in"
    await p2.goto(`${BASE}/app`);
    await p2.waitForURL(/\/login$/);
    await ctx2.close();
    return 'deleted, signed out';
  });
  await check('account: sign out everywhere', async () => {
    await page.getByRole('button', { name: 'Sign out everywhere' }).click();
    await page.waitForURL(/\/login$/, { timeout: 10000 });
    return 'ok';
  });
}

console.log('LOGS:\n' + logs.join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
