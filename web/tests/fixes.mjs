// Fixes from the website audit: chunk-load recovery and the error screen, malformed links, job wording, ledger wording,
// delete confirmation, "try again in N minutes", stale return targets, odd files, meta tags, and the few pure helpers.
// Needs the local test backend (port 8792, see helpers.mjs) serving `npm run build`; most checks use mocked answers.
// Usage: node tests/fixes.mjs [section ...]   sections: pure pages jobs billing login files
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { BASE, FILES, check, dropFile, launch, pick, signIn, summary } from './helpers.mjs';

const only = process.argv.slice(2);
const want = (s) => only.length === 0 || only.includes(s);
const EMAIL = process.env.E2E_EMAIL || `fixes-${Date.now()}@example.com`;
const now = () => Math.floor(Date.now() / 1000);

/** Page errors that are not part of what a check is looking for. */
const pageErrors = (logs) => logs.filter((l) => l.startsWith('[pageerror]'));

if (want('pure')) {
  await check('pure helpers: WAV size limit, safe return paths, ledger words', async () => {
    const { wavFits } = await import('../src/media/wav.js');
    const { isSafeReturn } = await import('../src/returnTo.js');
    const { ledgerKind, ledgerText } = await import('../src/ledger.js');
    const h = 3600;
    if (!wavFits(6 * h, 2, 16)) throw new Error('6 h stereo 16-bit should fit');
    if (wavFits(6 * h, 2, 24)) throw new Error('6 h stereo 24-bit must not fit (over 4 GB)');
    if (!wavFits(4 * h, 2, 24)) throw new Error('4 h stereo 24-bit fits');
    if (wavFits(4.2 * h, 2, 24)) throw new Error('4.2 h stereo 24-bit must not fit');
    for (const ok of ['/app', '/app/billing', '/terms?x=1']) if (!isSafeReturn(ok)) throw new Error(`${ok} should be accepted`);
    for (const bad of ['//evil.example', '/\\evil.example', '/app\\..\\x', '/login', '/login/verify', 'https://evil.example', '', null, '/app\n/x']) if (isSafeReturn(bad)) throw new Error(`${JSON.stringify(bad)} should be refused`);
    const t = ledgerText({ kind: 'usage', description: 'denoise + transcribe: noisy.flac (0.2 min)' });
    if (t !== 'Denoising and subtitles: noisy.flac (0.2 min)') throw new Error(t);
    if (ledgerText({ kind: 'usage', description: 'transcribe: a.mp3 (12.5 min)' }) !== 'Subtitles & transcript: a.mp3 (12.5 min)') throw new Error('transcribe');
    if (ledgerText({ kind: 'usage', description: 'Live session (3.0 min, cut short by a restart)' }) !== 'Live session (3.0 min, ended early)') throw new Error('live');
    if (ledgerText({ kind: 'topup', description: 'Top-up $20.00' }) !== 'Top-up $20.00') throw new Error('topup');
    if (ledgerKind({ kind: 'usage' }) !== 'Used' || ledgerKind({ kind: 'grant' }) !== 'Credit added' || ledgerKind({ kind: 'weird' }) !== 'weird' || ledgerKind({ kind: 'weird', label: 'Weird' }) !== 'Weird') throw new Error('kinds');
    // newer servers send the pieces as fields
    const f = ledgerText({ kind: 'usage', label: 'Denoising', product: 'clean', minutes: 0.2, displayName: 'noisy.wav', description: 'denoise: noisy.flac (0.2 min)' });
    if (f !== 'Denoising: noisy.wav (0.2 min)') throw new Error(f);
    if (ledgerText({ kind: 'usage', label: 'Live noise removal + Live captions', product: 'live', minutes: 3 }) !== 'Live noise removal + Live captions (3 min)') throw new Error('live fields');
    return 'ok';
  });
}

if (want('pages')) {
  const { browser, page, logs } = await launch();

  await check('a page file that cannot be loaded: one retry, one reload, then a friendly screen with Reload (never a white page)', async () => {
    let served = 0;
    await page.route('**/assets/Terms-*.js', (r) => { served += 1; return r.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>x</title>' }); });
    let loads = 0;
    page.on('load', () => { loads += 1; });
    await page.goto(`${BASE}/`);
    await page.waitForSelector('main h1');
    await page.locator('footer').getByRole('link', { name: 'Terms of Service' }).click();
    await page.getByTestId('error-boundary').waitFor({ timeout: 20000 });
    const text = await page.getByTestId('error-boundary').innerText();
    if (!/Something went wrong/.test(text) || !/Reload/.test(text)) throw new Error(text);
    if (loads !== 2) throw new Error(`expected the first load and exactly one reload, saw ${loads} loads`);
    const flag = await page.evaluate(() => sessionStorage.getItem('acw-chunk-reload'));
    if (flag !== '1') throw new Error('reload guard not set');
    // the files are back: Reload shows the page, and the guard is cleared once a page has loaded
    await page.unroute('**/assets/Terms-*.js');
    await page.getByRole('button', { name: 'Reload' }).click();
    await page.getByRole('heading', { name: 'Terms of Service' }).waitFor({ timeout: 15000 });
    const after = await page.evaluate(() => sessionStorage.getItem('acw-chunk-reload'));
    if (after !== null) throw new Error('guard not cleared after a good load: ' + after);
    return `file requested ${served}x, ${loads} loads, then Terms shown`;
  });

  await check('/login/verify with a cut-off %-code in the token: the "looks cut off" message, no crash', async () => {
    await page.goto(`${BASE}/login/verify#token=%E0%A4%A`);
    await page.waitForSelector('.banner-error', { timeout: 10000 });
    const t = await page.locator('.banner-error').innerText();
    if (!/cut off/.test(t)) throw new Error(t);
    if (page.url().includes('token')) throw new Error('hash kept: ' + page.url());
    return t;
  });

  await check('/#1 and other hashes that are not valid selectors do not throw', async () => {
    logs.length = 0;
    for (const h of ['#1', '#a%20b', '#%E0%A4%A', '#pricing']) {
      await page.goto(`${BASE}/${h}`);
      await page.waitForSelector('main h1');
      await page.waitForTimeout(300);
    }
    const errs = pageErrors(logs);
    if (errs.length) throw new Error(errs.join(' | '));
    return 'no page errors';
  });

  await check('meta: description without "Studio", canonical, social preview tags', async () => {
    await page.goto(`${BASE}/`);
    const m = await page.evaluate(() => ({
      desc: document.querySelector('meta[name=description]')?.content,
      canon: document.querySelector('link[rel=canonical]')?.href,
      ogTitle: document.querySelector('meta[property="og:title"]')?.content,
      ogImage: document.querySelector('meta[property="og:image"]')?.content,
      tw: document.querySelector('meta[name="twitter:card"]')?.content,
    }));
    if (!m.desc || /studio/i.test(m.desc)) throw new Error('description: ' + m.desc);
    if (!/^https?:\/\/.+\/audio\/$/.test(m.canon || '')) throw new Error('canonical: ' + m.canon);
    if (!m.ogTitle || !/^https?:\/\//.test(m.ogImage || '') || !m.tw) throw new Error(JSON.stringify(m));
    await page.goto(`${BASE}/terms`);
    await page.waitForSelector('main h1');
    const c2 = await page.evaluate(() => document.querySelector('link[rel=canonical]').href);
    if (!/\/audio\/terms$/.test(c2)) throw new Error('canonical on /terms: ' + c2);
    const robots = await (await page.request.get(`${BASE}/robots.txt`)).text();
    if (!/Disallow: \/audio\/app/.test(robots)) throw new Error('robots.txt: ' + robots.slice(0, 80));
    return 'ok';
  });

  await check('footer headings are h2 (heading order) and wide tables can be scrolled by keyboard', async () => {
    await page.goto(`${BASE}/licenses`);
    await page.waitForSelector('main h1');
    if (await page.locator('footer h4').count()) throw new Error('footer still has h4');
    if ((await page.locator('footer h2').count()) < 2) throw new Error('footer h2 missing');
    await page.waitForSelector('.table-wrap[tabindex="0"]');
    const label = await page.locator('.table-wrap').first().getAttribute('aria-label');
    if (!label) throw new Error('scroll region has no name');
    return `scroll region "${label}"`;
  });

  await check('a signed-out visitor sent to sign in is taken back only if they went straight to it; wandering off forgets the target', async () => {
    await page.goto(`${BASE}/app/billing`);
    await page.waitForURL(/\/login/);
    const kept = await page.evaluate(() => sessionStorage.getItem('acw-return-to'));
    if (kept !== '/app/billing') throw new Error('not remembered: ' + kept);
    await page.locator('footer').getByRole('link', { name: 'Terms of Service' }).click();
    await page.getByRole('heading', { name: 'Terms of Service' }).waitFor();
    const gone = await page.evaluate(() => sessionStorage.getItem('acw-return-to'));
    if (gone !== null) throw new Error('still remembered after leaving: ' + gone);
    await page.evaluate(() => sessionStorage.setItem('acw-return-to', '/app/billing'));
    await page.goto(`${BASE}/login`);
    await page.waitForSelector('main h1');
    const direct = await page.evaluate(() => sessionStorage.getItem('acw-return-to'));
    if (direct !== null) throw new Error('a direct visit to /login kept an old target: ' + direct);
    return 'ok';
  });

  await check('429 on email sign-in with Retry-After: the message says how long to wait', async () => {
    await page.route('**/api/auth/providers', (r) => r.fulfill({ json: { google: false, email: true, turnstileSiteKey: null } }));
    await page.route('**/api/auth/email', (r) => r.fulfill({ status: 429, headers: { 'Retry-After': '720' }, json: { detail: 'Too many requests, slow down.' } }));
    await page.goto(`${BASE}/login`);
    await page.getByLabel('Email address').fill('someone@example.com');
    await page.getByRole('button', { name: /Email me/ }).click();
    const t = await page.locator('.banner-error').innerText();
    if (!/12 minutes/.test(t)) throw new Error(t);
    return t;
  });

  await browser.close();
}

if (want('jobs') || want('billing') || want('files')) {
  const first = await launch();
  await signIn(first.page, EMAIL);
  const state = await first.context.storageState();
  await first.browser.close();
  const job = (id, extra) => ({
    id, kind: 'denoise', free: false, status: 'running', inputName: `${id}.wav`, createdAt: now(), durationSeconds: 60, progress: 0, stage: '',
    outputs: [], costMicros: 0, expiresAt: null, queuePosition: null, queueWaitSeconds: null, ...extra,
  });
  const openApp = async (page) => {
    await page.goto(`${BASE}/app`);
    await page.getByTestId('available').waitFor();
    await page.evaluate(async () => {
      const me = await (await fetch('/audio/api/me', { credentials: 'same-origin' })).json();
      await fetch('/audio/api/account/tos', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': me.csrfToken }, body: JSON.stringify({ version: '2026-10-16' }) });
    });
  };

  if (want('jobs')) {
    const { browser, page } = await launch([], { storageState: state });
    await check('running jobs: the server\'s step names are shown in plain words ("premium" and "cloud asr" never appear); an unknown step shows only the percentage; no doubled "Working on it"', async () => {
      const list = [
        job('j1', { status: 'cloud_asr', stage: 'cloud asr running', progress: 85 }),
        job('j2', { stage: 'transcribing (premium)', progress: 40 }),
        job('j3', { stage: 'Cleaning', progress: 41 }),
        job('j4', { stage: 'denoising', progress: 12 }),
        job('j5', { stage: '', progress: 0 }),
        job('j6', { stage: 'some brand new step', progress: 7 }),
      ];
      await page.route('**/api/jobs', (r) => (r.request().method() === 'GET' ? r.fulfill({ json: { jobs: list } }) : r.continue()));
      await openApp(page);
      await page.getByTestId('job').first().waitFor();
      const lines = await page.locator('[data-testid=job] p[role=status]').allInnerTexts();
      // the step lines only: a card's own title ("Denoising" is the product's name now) is not a raw step
      const all = lines.join('\n');
      if (/premium|cloud asr|denoising|some brand new/i.test(all)) throw new Error('raw step shown: ' + all.slice(0, 400));
      const want = ['Writing the transcript · 85%', 'Writing the transcript · 40%', 'Cleaning the sound · 41%', 'Cleaning the sound · 12%', 'Just a moment…', '7% done'];
      for (const w of want) if (!lines.some((l) => l.trim() === w)) throw new Error(`missing "${w}" in ${JSON.stringify(lines)}`);
      const doubled = (await page.locator('[data-testid=job]').nth(4).innerText()).match(/Working on it/g)?.length || 0;
      if (doubled > 1) throw new Error('"Working on it" shown ' + doubled + ' times');
      return lines.map((l) => l.trim()).join(' | ');
    });

    await check('job cards name the work with the server\'s label (Voice rebuild + Leveling, Leveling) and fall back to Denoising; the Leveling step is shown in plain words', async () => {
      await page.unroute('**/api/jobs');
      const list = [
        job('k1', { label: 'Voice rebuild + Leveling', status: 'queued' }),
        job('k2', { label: 'Leveling', status: 'queued' }),
        job('k3', { status: 'queued' }),
        job('k4', { label: 'Denoising + Leveling', stage: 'Leveling', progress: 97 }),
      ];
      await page.route('**/api/jobs', (r) => (r.request().method() === 'GET' ? r.fulfill({ json: { jobs: list } }) : r.continue()));
      await openApp(page);
      await page.getByTestId('job').first().waitFor();
      const cards = await page.locator('[data-testid=job]').allInnerTexts();
      const sub = (i) => (cards[i].match(/^(Voice rebuild \+ Leveling|Leveling|Denoising(?: \+ Leveling)?) · /m) || [])[1];
      if (sub(0) !== 'Voice rebuild + Leveling' || sub(1) !== 'Leveling' || sub(2) !== 'Denoising' || sub(3) !== 'Denoising + Leveling') throw new Error(JSON.stringify(cards));
      const lines = await page.locator('[data-testid=job] p[role=status]').allInnerTexts();
      if (!lines.some((l) => l.trim() === 'Leveling the voice · 97%')) throw new Error(JSON.stringify(lines));
      return cards.map((c) => c.split('\n').find((l) => / · /.test(l))).join(' | ');
    });

    await check('the job list shows the file as the visitor chose it (displayName), not the compressed upload name', async () => {
      await page.unroute('**/api/jobs');
      await page.route('**/api/jobs', (r) => (r.request().method() === 'GET' ? r.fulfill({ json: { jobs: [job('jn', { status: 'queued', inputName: 'noisy.flac', displayName: 'noisy.wav' })] } }) : r.continue()));
      await openApp(page);
      await page.getByTestId('job').first().waitFor();
      const t = await page.locator('.job-title').first().innerText();
      if (t !== 'noisy.wav') throw new Error(t);
      return t;
    });

    await check('polling uses the list endpoint once per round (no request per job); a 401 sends the visitor to sign in', async () => {
      await page.unroute('**/api/jobs');
      let lists = 0;
      let singles = 0;
      let expired = false;
      await page.route(/\/api\/jobs(\/[0-9a-z]+)?$/, (r) => {
        const u = new URL(r.request().url());
        if (r.request().method() !== 'GET') return r.continue();
        if (u.pathname.endsWith('/api/jobs')) {
          lists += 1;
          if (expired) return r.fulfill({ status: 401, json: { detail: 'Sign in first.' } });
          return r.fulfill({ json: { jobs: [job('a1', { progress: 10, stage: 'Cleaning' }), job('a2', { progress: 20, stage: 'Cleaning' }), job('a3', { progress: 30, stage: 'Cleaning' })] } });
        }
        singles += 1;
        return r.fulfill({ status: 404, json: { detail: 'gone' } });
      });
      await openApp(page);
      await page.getByTestId('job').first().waitFor();
      const before = lists;
      await page.waitForTimeout(4200);
      const rounds = lists - before;
      if (singles !== 0) throw new Error(`${singles} per-job requests`);
      if (rounds < 2 || rounds > 4) throw new Error(`${rounds} list requests in 4.2 s`);
      expired = true;
      await page.waitForURL(/\/login/, { timeout: 10000 });
      return `${rounds} list requests in 4.2 s for 3 jobs; 401 -> sign-in page`;
    });
    await browser.close();

    const second = await launch([], { storageState: state });
    await check('"Delete now" on a finished job asks first; "Keep them" backs out; "Yes, delete" deletes', async () => {
      const p = second.page;
      const done = job('d1', { status: 'done', outputs: [{ name: 'noisy-cleaned.wav', kind: 'audio', bytes: 1000 }], expiresAt: now() + 3600, costMicros: 2600 });
      let deletes = 0;
      await p.route(/\/api\/jobs(\/d1)?$/, (r) => {
        const m = r.request().method();
        if (m === 'DELETE') { deletes += 1; return r.fulfill({ json: { ok: true } }); }
        if (m === 'GET') return r.fulfill({ json: { jobs: [done] } });
        return r.continue();
      });
      await openApp(p);
      await p.getByRole('button', { name: 'Delete now' }).click();
      await p.getByTestId('delete-confirm').waitFor();
      if (deletes) throw new Error('deleted without asking');
      await p.getByRole('button', { name: 'Keep them' }).click();
      if (!(await p.getByRole('button', { name: 'Delete now' }).count())) throw new Error('did not go back');
      await p.getByRole('button', { name: 'Delete now' }).click();
      await p.getByRole('button', { name: 'Yes, delete' }).click();
      await p.waitForTimeout(500);
      if (deletes !== 1) throw new Error('deletes: ' + deletes);
      return 'ok';
    });
    await second.browser.close();
  }

  if (want('billing')) {
    const { browser, page } = await launch([], { storageState: state });
    await check('Billing: a payment that is already booked when the page opens shows the thank-you at once (not "hasn\'t caught up")', async () => {
      await page.route('**/api/billing/history', (r) => r.fulfill({ json: { payments: [{ id: 'pay_x', kind: 'topup', status: 'succeeded', amount_cents: 2000, refunded_cents: 0, created_at: now() - 30 }], ledger: [] } }));
      await page.goto(`${BASE}/app/billing?checkout=success&p=pay_x`);
      await page.getByTestId('checkout-banner').waitFor();
      await page.waitForFunction(() => /Thank you/.test(document.querySelector('[data-testid=checkout-banner]')?.textContent || ''), null, { timeout: 8000 });
      const t = await page.getByTestId('checkout-banner').innerText();
      if (/caught up/.test(t)) throw new Error(t);
      return t.trim();
    });

    await check('Billing: ledger shows names, not internal words; scroll region named; "100 most recent" hint only when the server says there is more', async () => {
      await page.unroute('**/api/billing/history');
      const ledger = Array.from({ length: 100 }, (_, i) => ({
        id: i + 1, amount_micros: i === 0 ? -2600 : 1000000, kind: i === 0 ? 'usage' : i === 1 ? 'grant' : 'topup', ref_type: 'job', ref_id: 'x',
        description: i === 0 ? 'denoise + transcribe: noisy.flac (0.2 min)' : i === 1 ? 'Welcome gift' : 'Top-up $1.00', balance_after_micros: 5000000, created_at: now() - i * 60,
      }));
      await page.route('**/api/billing/history', (r) => r.fulfill({ json: { payments: [], ledger, ledgerHasMore: false } }));
      await page.goto(`${BASE}/app/billing`);
      await page.getByTestId('ledger').waitFor();
      const first = await page.locator('[data-testid=ledger] tbody tr').first().innerText();
      if (!/Denoising and subtitles: noisy\.flac \(0\.2 min\)/.test(first) || /denoise|usage/i.test(first)) throw new Error(first);
      const second = await page.locator('[data-testid=ledger] tbody tr').nth(1).innerText();
      if (!/Credit added/.test(second) || /grant/.test(second)) throw new Error(second);
      if (await page.getByText('Showing your 100 most recent entries.').count()) throw new Error('hint shown although the server said there is nothing older');
      await page.unroute('**/api/billing/history');
      await page.route('**/api/billing/history', (r) => r.fulfill({ json: { payments: [], ledger, ledgerHasMore: true } }));
      await page.reload();
      await page.getByTestId('ledger').waitFor();
      if (!(await page.getByText('Showing your 100 most recent entries.').count())) throw new Error('no hint');
      const wrap = page.locator('.table-wrap').first();
      if ((await wrap.getAttribute('tabindex')) !== '0' || !(await wrap.getAttribute('aria-label'))) throw new Error('scroll region');
      return first.replace(/\s+/g, ' ');
    });

    await check('Billing: the auto-reload consent never shows <angle> placeholders; a custom amount is read strictly ("-5" is not $5; "12,5" is $12.50)', async () => {
      await page.goto(`${BASE}/app/billing`);
      await page.getByTestId('ar-consent-text').waitFor();
      const t = await page.getByTestId('ar-consent-text').innerText();
      if (/[<>]/.test(t)) throw new Error(t);
      await page.locator('label:has(input[value=custom])').click();
      const hint = page.locator('#custom-hint');
      for (const [text, ok] of [['-5', false], ['12,5', true], ['-12', false], ['1.2.3', false], ['25', true], ['$25', true], ['25.50', true], ['1,250.00', false]]) {
        await page.getByLabel('Custom amount (USD)').fill(text);
        const valid = !/neg/.test(await hint.getAttribute('class'));
        if (valid !== ok) throw new Error(`"${text}" valid=${valid}`);
      }
      return 'ok';
    });
    await browser.close();
  }

  if (want('files')) {
    const { browser, page, logs } = await launch([], { storageState: state });
    await check('an empty file is refused with a clear message instead of offering options with made-up numbers', async () => {
      await openApp(page);
      await page.setInputFiles('input[type=file]', { name: 'empty.wav', mimeType: 'audio/wav', buffer: Buffer.alloc(0) });
      const box = page.getByTestId('job-error');
      await box.waitFor({ timeout: 5000 });
      const t = await box.innerText();
      if (!/empty/.test(t)) throw new Error(t);
      if (await page.getByTestId('start').count()) throw new Error('Start offered for an empty file');
      return t;
    });

    await check('a file name with direction-changing characters is shown with them marked, never reversed', async () => {
      const buffer = readFileSync(join(FILES, 'noisy.wav'));
      await page.reload();
      await page.getByTestId('available').waitFor();
      await page.setInputFiles('input[type=file]', { name: 'evil‮gpj.wav', mimeType: 'audio/wav', buffer });
      await page.getByTestId('start').waitFor({ timeout: 20000 });
      const shown = await page.locator('.file-chip .name').innerText();
      if (/[‪-‮⁦-⁩]/.test(shown) || !shown.includes('□')) throw new Error(JSON.stringify(shown));
      return shown;
    });

    await check('prices that cannot be loaded: Start stays off and a reason with "Try again" is shown', async () => {
      await page.route('**/api/billing/pricing', (r) => r.fulfill({ status: 500, json: { detail: 'broken' } }));
      await page.goto(`${BASE}/app`);
      await page.getByTestId('available').waitFor();
      await dropFile(page, join(FILES, 'noisy.wav'));
      await pick(page, 'noise', 'denoise');
      await page.getByText('couldn’t load the prices').waitFor({ timeout: 5000 });
      if (!(await page.getByRole('button', { name: 'Try again' }).count())) throw new Error('no retry button');
      if (await page.getByTestId('start').isEnabled()) throw new Error('Start is enabled without a price');
      await page.unroute('**/api/billing/pricing');
      return 'ok';
    });

    await check('the header does not offer "Sign in" while the server cannot say who is signed in', async () => {
      const { browser: b2, page: p2 } = await launch([], { storageState: state });
      await p2.route('**/api/me', (r) => r.fulfill({ status: 503, json: { detail: 'maintenance' } }));
      await p2.goto(`${BASE}/`);
      await p2.waitForSelector('main h1');
      await p2.waitForTimeout(800);
      const signIn = await p2.locator('header').getByRole('link', { name: 'Sign in' }).count();
      await b2.close();
      if (signIn) throw new Error('"Sign in" shown during an outage');
      return 'no Sign in button';
    });
    void logs;
    await browser.close();
  }
}

process.exit(summary().some((r) => !r.ok) ? 1 : 0);
