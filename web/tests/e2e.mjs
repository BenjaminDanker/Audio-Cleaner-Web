// The main paths against the local backend, one per feature: sign-in (and the API's sign-in answers), the price list against
// the server's own list, a paid file job (Denoising + Leveling, the estimate, the download), a paid subtitles job, the billing
// flow (top-up rules, the checkout return, a refused card, refunds), API keys, and the account (export, deletion, sign-out).
// Usage: node tests/e2e.mjs [section ...]   sections: basic auth studio transcribe billing keys account
import { readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { BASE, FILES, SHOTS, admin, check, dropFile, launch, loginLink, pick, signIn, summary } from './helpers.mjs';

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
  await check('landing: the price table is the server\'s price list (per hour, groups, no status column)', async () => {
    const pricing = await (await fetch(`${BASE}/api/billing/pricing`)).json();
    await page.goto(`${BASE}/`);
    await page.waitForSelector('table tbody tr');
    const txt = await page.locator('table').innerText();
    for (const s of ['Denoising', 'Leveling', 'Voice rebuild', 'Files', 'Live streaming']) if (!txt.includes(s)) throw new Error('missing ' + s);
    const missing = pricing.rates.filter((r) => r.available && r.product !== 'stream_denoise' && !txt.includes(`$${r.usdPerHour.toFixed(2)} / hour`) && !txt.includes(`$${(r.usdPerMinute).toFixed(3)} / min`));
    if (missing.length) throw new Error('prices not on the page: ' + missing.map((r) => `${r.product}/${r.tier}`).join(', '));
    if (/Available|Coming soon/.test(txt)) throw new Error('status column is back');
    return `${pricing.rates.length} server rates checked`;
  });
}

if (want('auth')) {
  // Each sign-in runs in its own browser context, so the shared page below stays the e2e account.
  await check('sign-in: a login link lands on /app with the token gone; a reused link and a cut-off token get a friendly error; signed-out /app/billing goes to /login', async () => {
    const mail = `e2e-auth-${Date.now()}@example.com`;
    const ctx = await browser.newContext();
    const p1 = await ctx.newPage();
    try {
      await p1.goto(`${BASE}/app/billing`);
      await p1.waitForURL(/\/login$/);
      const link = loginLink(mail);
      await p1.goto(link);
      await p1.waitForURL((u) => !u.pathname.startsWith('/login'), { timeout: 15000 });
      if (p1.url().includes('token')) throw new Error('token still in URL');
      await p1.waitForSelector('[data-testid=available]');
      // a reused link is refused
      await p1.context().clearCookies();
      await p1.goto(link);
      await p1.waitForSelector('.banner-error', { timeout: 10000 });
      const reused = await p1.locator('.banner-error').innerText();
      if (!/invalid or expired/i.test(reused)) throw new Error(reused);
      // a token cut off in the middle of a %-code is not a crash
      const p2 = await ctx.newPage();
      await p2.goto(`${BASE}/login/verify#token=%E0%A4%A`);
      await p2.waitForSelector('.banner-error', { timeout: 10000 });
      if (!/cut off/.test(await p2.locator('.banner-error').innerText())) throw new Error('cut-off token not explained');
      return 'ok';
    } finally {
      await ctx.close();
    }
  });
  loginLink(EMAIL); // creates the account (the login link itself is not used)
  admin('grant', EMAIL, '5', 'e2e test');
  await check('API: a 401 on a list sends the visitor to sign in', async () => {
    const ctx = await browser.newContext();
    const p1 = await ctx.newPage();
    try {
      await signIn(p1, EMAIL);
      await p1.route('**/api/jobs', (r) => r.fulfill({ status: 401, json: { detail: 'Sign in required.' } }));
      await p1.goto(`${BASE}/app`);
      await p1.waitForURL(/\/login$/, { timeout: 10000 });
      return 'ok';
    } finally {
      await ctx.close();
    }
  });
}

if (want('studio')) {
  await signIn(page, EMAIL);
  await check('cleanup page: balance and no Terms prompt for an account that agreed (signing in is agreeing)', async () => {
    await gotoApp('/app');
    await page.waitForFunction(() => document.querySelector('[data-testid=available]')?.textContent === '$5.00', null, { timeout: 15000 });
    const bal = await page.getByTestId('available').innerText();
    if (bal !== '$5.00') throw new Error('balance ' + bal);
    if (await page.getByRole('dialog').count()) throw new Error('a Terms prompt is still shown');
    return `balance ${bal}`;
  });

  await check('Terms changed since this session began: a notice (not a gate); OK agrees to the new version, so it stays gone after a reload', async () => {
    if (await page.getByTestId('terms-notice').count()) throw new Error('notice shown to an account that has agreed');
    let agreed = null; // the version the OK button agreed to (the account here has agreed already: /me is made to say otherwise until then)
    await page.route('**/api/me', async (route) => {
      const res = await route.fetch();
      const j = await res.json();
      await route.fulfill({ response: res, json: j.user ? { ...j, user: { ...j.user, tosAccepted: !!j.user.tosVersion && agreed === j.user.tosVersion } } : j });
    });
    await page.route('**/api/account/tos', async (route) => {
      agreed = route.request().postDataJSON()?.version ?? null;
      await route.continue();
    });
    try {
      await page.reload();
      await page.getByTestId('terms-notice').waitFor();
      const t = await page.getByTestId('terms-notice').innerText();
      const [res] = await Promise.all([page.waitForResponse('**/api/account/tos'), page.getByTestId('terms-ok').click()]);
      if (!res.ok()) throw new Error(`account/tos answered ${res.status()}`);
      if (!agreed) throw new Error('OK did not send the current Terms version');
      await page.getByTestId('terms-notice').waitFor({ state: 'detached' });
      await page.reload();
      await page.getByTestId('available').waitFor();
      if (await page.getByTestId('terms-notice').count()) throw new Error('notice back after a reload');
      return `${t.replace(/\s+/g, ' ')} (agreed to ${agreed})`;
    } finally {
      await page.unroute('**/api/me');
      await page.unroute('**/api/account/tos');
    }
  });

  await check('a paid file job, Denoising + Leveling: the estimate on Start, the request that is sent, the finished file downloads, the balance drops', async () => {
    await gotoApp('/app');
    await dropFile(page, join(FILES, 'noisy.wav'));
    await pick(page, 'noise', 'denoise');
    await page.getByTestId('estimate').waitFor();
    const est = await page.getByTestId('estimate').innerText();
    if (!/\$0\.0[0-9]+/.test(est)) throw new Error('estimate: ' + est);
    if (!/^Start – about \$0\.0/.test(await page.getByTestId('start').innerText())) throw new Error('start label: ' + await page.getByTestId('start').innerText());
    if (!(await page.locator('input[type=checkbox][name=leveling]').isChecked())) throw new Error('Leveling is on by default with Denoising');
    const [req] = await Promise.all([page.waitForRequest((r) => r.url().endsWith('/api/jobs') && r.method() === 'POST'), page.getByTestId('start').click()]);
    const body = req.postDataJSON();
    if (body.kind !== 'denoise' || body.denoiseTier !== 'standard' || body.options.leveling !== true || body.options.audio_format !== 'flac') throw new Error('job sent: ' + JSON.stringify(body));
    await page.getByTestId('submitted').waitFor({ timeout: 60000 });
    const card = await waitJob('noisy.'); // uncompressed WAV is uploaded as noisy.flac
    const status = await card.getAttribute('data-status');
    if (status !== 'done') throw new Error('status ' + status + ' ' + (await card.innerText()));
    await page.screenshot({ path: join(SHOTS, 'studio-done.png'), fullPage: true });
    const [dl] = await Promise.all([page.waitForEvent('download'), card.getByTestId('download').first().click()]);
    if (!dl.suggestedFilename().endsWith('.flac')) throw new Error('the audio a server job makes should come as FLAC, got ' + dl.suggestedFilename());
    const p = join(FILES, 'out-studio.flac');
    await dl.saveAs(p);
    await page.waitForFunction(() => document.querySelector('[data-testid=available]')?.textContent !== '$5.00', null, { timeout: 15000 });
    return `${est.split('\n')[0]}; downloaded ${dl.suggestedFilename()} ${statSync(p).size} bytes; balance ${await page.getByTestId('available').innerText()}`;
  });
}

if (want('transcribe')) {
  await check('a paid subtitles job (Standard) on the Cleanup page: the estimate, the finished transcript downloads', async () => {
    await gotoApp('/app');
    await dropFile(page, join(FILES, 'noisy.wav'));
    await pick(page, 'noise', 'off');
    await pick(page, 'subs', 'standard');
    await page.getByTestId('estimate').waitFor();
    await page.getByRole('button', { name: /^Start – about/ }).click();
    await page.getByTestId('submitted').waitFor({ timeout: 60000 });
    const card = await waitJob('noisy.', 300000);
    const st = await card.getAttribute('data-status');
    if (st !== 'done') throw new Error(st + ' ' + (await card.innerText()));
    const names = await card.locator('.oname').allInnerTexts();
    const [dl] = await Promise.all([page.waitForEvent('download'), card.locator('li', { hasText: '.txt' }).getByTestId('download').click()]);
    await dl.saveAs(join(FILES, 'out-transcript.txt'));
    const text = readFileSync(join(FILES, 'out-transcript.txt'), 'utf8').trim();
    if (!text) throw new Error('empty transcript');
    return `${names.join(', ')} | "${text.slice(0, 80)}"`;
  });
}

if (want('billing')) {
  await check('billing: payments off -> Pay and refund are disabled with the reason; the card button is off too', async () => {
    await gotoApp('/app/billing');
    await page.waitForSelector('[data-testid=payments-off]');
    const pay = page.getByRole('button', { name: /^Pay / });
    if (!(await pay.isDisabled())) throw new Error('Pay button should be disabled while payments are off');
    const note = await page.getByTestId('topup-off').innerText();
    if (!/not available yet|switched off/i.test(note)) throw new Error(note);
    if (!(await page.getByRole('button', { name: 'Manage card and receipts' }).isDisabled())) throw new Error('card portal button enabled');
    return note;
  });

  await check('billing: a top-up amount is read strictly and limited ($5 smallest, $50 for a first top-up, $500 after), and the auto-reload consent states the exact charge', async () => {
    await page.locator('label:has(input[value=custom])').click();
    const hint = page.locator('#custom-hint');
    const field = page.getByLabel('Custom amount (USD)');
    const valid = async () => !/neg/.test(await hint.getAttribute('class'));
    for (const [text, ok] of [['4.99', false], ['5', true], ['600', false], ['-5', false], ['12,5', true], ['$25', true], ['1,250.00', false], ['abc', false], ['25', true]]) {
      await field.fill(text);
      if ((await valid()) !== ok) throw new Error(`"${text}" valid=${!ok}; hint: ${await hint.innerText()}`);
    }
    await field.fill('4.99');
    if (!/smallest top-up is \$5\.00/.test(await hint.innerText())) throw new Error(`says why: ${await hint.innerText()}`);
    await field.fill('600');
    // this account has not paid yet: its first top-up can be up to $50, and the hint says so
    if (!/first top-up can be up to \$50\.00\. After that, up to \$500\.00/.test(await hint.innerText())) throw new Error(`says why: ${await hint.innerText()}`);
    await field.fill('25');
    if (!(await page.getByRole('button', { name: /^Pay/ }).isDisabled())) throw new Error('still disabled while payments are off');
    const text = await page.getByTestId('ar-consent-text').innerText();
    if (!/I authorize Audio Cleaner to charge my saved card \$20\.00 whenever my balance falls below \$5\.00, up to \$100\.00 per calendar month, until I turn this off\./.test(text)) throw new Error(text);
    return 'limits and consent text match';
  });

  await check('billing: returning from checkout shows the thank-you at once when the payment is already booked, and "canceled" when it was not', async () => {
    await page.route('**/api/billing/history', (r) => r.fulfill({ json: { payments: [{ id: 'pay_x', kind: 'topup', status: 'succeeded', amount_cents: 2000, refunded_cents: 0, created_at: Math.floor(Date.now() / 1000) - 30 }], ledger: [] } }));
    await page.goto(`${BASE}/app/billing?checkout=success&p=pay_x`);
    await page.waitForFunction(() => /Thank you/.test(document.querySelector('[data-testid=checkout-banner]')?.textContent || ''), null, { timeout: 8000 });
    await page.unroute('**/api/billing/history');
    await page.goto(`${BASE}/app/billing?checkout=cancel`);
    const t = await page.getByTestId('checkout-banner').innerText();
    if (!/canceled/i.test(t)) throw new Error(t);
    return t;
  });

  await check('billing: a card from a country we don\'t serve: "not charged" banner and status, never "Paid"', async () => {
    await page.route('**/api/billing/history', async (route) => {
      const res = await route.fetch();
      const j = await res.json();
      j.payments = [{ id: 'pay_refused', kind: 'checkout', status: 'refused_region', amount_cents: 1000, refunded_cents: 0, created_at: Math.floor(Date.now() / 1000) }, ...j.payments];
      await route.fulfill({ response: res, json: j });
    });
    await page.goto(`${BASE}/app/billing?checkout=success&p=pay_refused`);
    await page.getByTestId('checkout-banner').filter({ hasText: 'wasn’t charged' }).waitFor({ timeout: 15000 });
    const badge = await page.getByTestId('pay-status').first().innerText();
    await page.unroute('**/api/billing/history');
    if (badge !== 'Not charged') throw new Error('status badge: ' + badge);
    return (await page.getByTestId('checkout-banner').innerText()).replace(/\s+/g, ' ');
  });

  await check('refund: only where the server says some of a payment is unused; the confirmation states the card fee; the success shows the refunded amount; no fee on a second refund', async () => {
    const now = Math.floor(Date.now() / 1000);
    // pay1: unused $15 of $20; pay2: refunded before, $7 left; pay3: nothing left to refund; pay4: older than 14 days
    const payments = [
      { id: 'pay1', kind: 'topup', status: 'succeeded', amount_cents: 2000, refunded_cents: 0, refundable_cents: 1500, created_at: now - 3600 },
      { id: 'pay2', kind: 'topup', status: 'succeeded', amount_cents: 1000, refunded_cents: 300, refundable_cents: 700, created_at: now - 7200 },
      { id: 'pay3', kind: 'topup', status: 'succeeded', amount_cents: 1000, refunded_cents: 400, refundable_cents: 0, created_at: now - 9000 },
      { id: 'pay4', kind: 'topup', status: 'succeeded', amount_cents: 1000, refunded_cents: 0, refundable_cents: 0, created_at: now - 20 * 86400 },
    ];
    await page.route('**/api/billing/history', (r) => r.fulfill({ json: { payments, ledger: [] } }));
    await page.route('**/api/billing/pricing', async (r) => { const res = await r.fetch(); await r.fulfill({ response: res, json: { ...(await res.json()), paymentsEnabled: true } }); });
    let posted = null;
    await page.route('**/api/billing/refund', (r) => { posted = r.request().postDataJSON(); return r.fulfill({ json: { refundedCents: 4321 } }); });
    await page.goto(`${BASE}/app/billing`);
    await page.getByRole('button', { name: 'Refund unused' }).first().waitFor();
    const buttons = page.getByRole('button', { name: 'Refund unused' });
    if ((await buttons.count()) !== 2) throw new Error(`${await buttons.count()} refund buttons; want 2 (pay1, pay2)`);
    const rows = page.locator('tr', { has: page.getByTestId('pay-status') });
    for (const [i, want] of [[0, 1], [1, 1], [2, 0], [3, 0]]) {
      const n = await rows.nth(i).getByRole('button', { name: 'Refund unused' }).count();
      if (n !== want) throw new Error(`payment ${payments[i].id}: ${n} refund buttons, want ${want}`);
    }
    await buttons.first().click();
    const est = await page.getByTestId('refund-estimate').first().innerText();
    const fee = Math.ceil(2000 * 0.029 + 30);
    const money = (c) => `$${(c / 100).toFixed(2)}`;
    const wantText = `You’ll get back about ${money(1500 - fee)} (unused ${money(1500)} minus the card fee of ${money(fee)}).`;
    if (est.trim() !== wantText) throw new Error(`"${est}" vs "${wantText}"`);
    await page.getByRole('button', { name: 'Yes, refund' }).click();
    await page.getByText('$43.21 is on its way back to your card').waitFor();
    if (posted?.paymentId !== 'pay1') throw new Error('posted ' + JSON.stringify(posted));
    await page.reload();
    await page.getByRole('button', { name: 'Refund unused' }).nth(1).click();
    const est2 = (await page.getByTestId('refund-estimate').innerText()).trim();
    if (/card fee of/.test(est2) || !est2.includes(money(700))) throw new Error('second refund: ' + est2);
    await page.unroute('**/api/billing/refund');
    await page.unroute('**/api/billing/history');
    await page.unroute('**/api/billing/pricing');
    return `${est.trim()} / ${est2}`;
  });
}

if (want('keys')) {
  await check('API key: the create request is retried once after a 403 (CSRF), the key is shown once and copied, then revoked', async () => {
    await gotoApp('/app/live');
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    let n = 0;
    await page.route('**/api/keys', async (route) => {
      if (route.request().method() === 'POST' && n++ === 0) return route.fulfill({ status: 403, json: { detail: 'Missing or invalid CSRF token.' } });
      return route.continue();
    });
    await page.getByLabel('Name').fill('e2e key');
    await page.getByRole('button', { name: 'Create key' }).click();
    const key = await page.getByTestId('key-value').innerText({ timeout: 10000 });
    await page.unroute('**/api/keys');
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
    return `${key.slice(0, 12)}… created (after one CSRF retry), copied, revoked`;
  });
}

if (want('account')) {
  await check('account: the data export downloads with its name and holds the account', async () => {
    await gotoApp('/app/account');
    const [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download my data' }).click()]);
    const p = join(FILES, 'acw-export.json');
    await dl.saveAs(p);
    const j = JSON.parse(readFileSync(p, 'utf8'));
    if (!j.user || j.user.email !== EMAIL) throw new Error('bad export');
    if (dl.suggestedFilename() !== 'acw-account-export.json') throw new Error(dl.suggestedFilename());
    return `${Object.keys(j).join(',')}`;
  });

  await check('account deletion: a running job blocks it with a reason; typing the email is required; after success the visitor lands on / and /app asks to sign in', async () => {
    const mail = `e2e-del-${Date.now()}@example.com`;
    const ctx2 = await browser.newContext();
    const p2 = await ctx2.newPage();
    try {
      await p2.goto(loginLink(mail)); await p2.waitForURL(/\/app$/);
      admin('grant', mail, '1', 'e2e delete test');
      await p2.goto(`${BASE}/app/account`);
      const btn = p2.getByRole('button', { name: 'Delete my account' });
      await btn.waitFor();
      if (!(await btn.isDisabled())) throw new Error('enabled too early');
      await p2.getByLabel(/Type your email/).fill('wrong@example.com');
      if (!(await btn.isDisabled())) throw new Error('enabled with wrong email');
      // hold a paid job so the deletion is refused
      const made = await p2.evaluate(async () => {
        const me = await (await fetch('/api/me', { credentials: 'same-origin' })).json();
        const res = await fetch('/api/jobs', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': me.csrfToken }, body: JSON.stringify({ kind: 'transcribe', transcribeTier: 'local', filename: 'hold.wav', sizeBytes: 1000, durationSeconds: 5, options: { audio_format: 'wav', caption_formats: ['txt'] } }) });
        return { status: res.status, body: await res.json() };
      });
      if (made.status !== 200) throw new Error('could not create a held job: ' + JSON.stringify(made));
      await p2.getByLabel(/Type your email/).fill(mail);
      await btn.click();
      const alert = p2.getByTestId('delete-error');
      await alert.waitFor({ timeout: 10000 });
      if (!(await alert.innerText()).includes('Wait for your running jobs to finish first.')) throw new Error(await alert.innerText());
      await p2.evaluate(async (id) => {
        const me = await (await fetch('/api/me', { credentials: 'same-origin' })).json();
        await fetch(`/api/jobs/${id}/cancel`, { method: 'POST', credentials: 'same-origin', headers: { 'X-CSRF-Token': me.csrfToken } });
      }, made.body.id);
      await btn.click();
      await p2.waitForURL(/\/$/, { timeout: 15000 });
      await p2.getByTestId('account-deleted').waitFor({ timeout: 10000 });
      await p2.getByText('Your account has been deleted.').waitFor();
      await p2.goto(`${BASE}/app`);
      await p2.waitForURL(/\/login$/);
      return 'refused while a job runs, deleted once it was finished, signed out';
    } finally {
      await ctx2.close();
    }
  });

  await check('account: sign out everywhere ends the session', async () => {
    await gotoApp('/app/account');
    await page.getByRole('button', { name: 'Sign out everywhere' }).click();
    await page.waitForURL(/\/login$/, { timeout: 10000 });
    return 'ok';
  });
}

console.log('LOGS:\n' + logs.join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
