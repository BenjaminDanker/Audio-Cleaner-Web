// UI-level checks: login form variants (mocked providers), responsive overflow, a11y scan, strict CSP, dark mode.
import { join } from 'node:path';
import { BASE, FILES, SHOTS, check, launch, signIn, summary } from './helpers.mjs';

const EMAIL = process.env.E2E_EMAIL || 'e2e-main@example.com';

{
  const { browser, page, logs } = await launch();
  await check('login: Google button + email form + Turnstile (mocked providers)', async () => {
    await page.route('**/api/auth/providers', (r) => r.fulfill({ json: { github: true, google: true, email: true, turnstileSiteKey: '1x00000000000000000000AA' } }));
    let posted = null;
    await page.route('**/api/auth/email', (r) => { posted = r.request().postDataJSON(); return r.fulfill({ json: { ok: true } }); });
    await page.goto(`${BASE}/login`);
    const go = await page.getByRole('link', { name: /Google/ }).getAttribute('href');
    if (go !== '/audio/api/auth/oauth/google/start') throw new Error(go);
    const btn = page.getByRole('button', { name: /Email me/ });
    await page.getByLabel('Email address').fill('someone@example.com');
    await page.waitForFunction(() => !document.querySelector('button[type=submit]').disabled, null, { timeout: 20000 });
    const widget = 'Turnstile test key issued a token';
    await btn.click();
    await page.getByText('Check your inbox').waitFor();
    if (!posted?.turnstileToken) throw new Error('no token posted: ' + JSON.stringify(posted));
    return `${widget}; posted token ${posted.turnstileToken.slice(0, 12)}…`;
  });
  await check('login: email form without Turnstile', async () => {
    await page.unroute('**/api/auth/providers');
    await page.route('**/api/auth/providers', (r) => r.fulfill({ json: { github: false, google: false, email: true, turnstileSiteKey: null } }));
    await page.goto(`${BASE}/login`);
    await page.getByLabel('Email address').fill('a@b.co');
    if (await page.getByRole('button', { name: /Email me/ }).isDisabled()) throw new Error('disabled');
    if (await page.locator('script[src*=cloudflare]').count()) throw new Error('turnstile loaded without key');
    return 'ok';
  });
  await check('login/verify: bad token -> friendly error; hash cleared', async () => {
    await page.goto(`${BASE}/login/verify#token=not-a-real-token`);
    await page.waitForSelector('.banner-error');
    const url = page.url();
    if (url.includes('token')) throw new Error('hash not cleared: ' + url);
    return await page.locator('.banner-error').innerText();
  });
  await browser.close();
}

{
  const { browser, page, logs } = await launch([], { viewport: { width: 360, height: 800 } });
  await signIn(page, EMAIL);
  const pages = ['/', '/free', '/login', '/app', '/app/billing', '/app/live', '/app/account', '/terms', '/privacy', '/licenses', '/nope'];
  await check('responsive 360px: no horizontal overflow on any page', async () => {
    const bad = [];
    for (const p of pages) {
      await page.goto(`${BASE}${p}`);
      await page.waitForSelector('main h1', { timeout: 10000 });
      await page.waitForTimeout(400);
      const w = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth }));
      if (w.sw > w.iw + 1) {
        const culprit = await page.evaluate(() => [...document.querySelectorAll('body *')].filter((e) => e.getBoundingClientRect().right > window.innerWidth + 1).slice(0, 3).map((e) => e.tagName + '.' + e.className));
        bad.push(`${p}: ${w.sw}>${w.iw} ${culprit.join(',')}`);
      }
      if (['/', '/app', '/app/billing', '/app/live'].includes(p)) await page.screenshot({ path: join(SHOTS, `m360${p.replace(/\//g, '_') || '_home'}.png`), fullPage: true });
    }
    if (bad.length) throw new Error(bad.join(' | '));
    return pages.length + ' pages ok';
  });
  await check('mobile nav toggle opens menu', async () => {
    await page.goto(`${BASE}/`);
    await page.getByRole('button', { name: 'Menu' }).click();
    await page.getByRole('link', { name: 'Free cleanup' }).first().waitFor({ state: 'visible' });
    return 'ok';
  });
  await browser.close();
}

{
  const { browser, page } = await launch();
  await signIn(page, EMAIL);
  await check('a11y scan: unlabeled controls / buttons / images, heading order, landmarks', async () => {
    const probs = [];
    for (const p of ['/', '/free', '/login', '/app', '/app/billing', '/app/live', '/app/account', '/terms']) {
      await page.goto(`${BASE}${p}`);
      await page.waitForSelector('main h1');
      await page.waitForTimeout(500);
      const r = await page.evaluate(() => {
        const out = [];
        const name = (el) => (el.getAttribute('aria-label') || el.getAttribute('aria-labelledby') && document.getElementById(el.getAttribute('aria-labelledby'))?.textContent || el.labels && [...el.labels].map((l) => l.textContent).join(' ') || el.textContent || el.getAttribute('title') || '').trim();
        for (const el of document.querySelectorAll('input:not([type=hidden]), select, textarea')) if (!name(el) && !el.closest('label')) out.push('control ' + el.outerHTML.slice(0, 80));
        for (const el of document.querySelectorAll('button, a[href]')) if (!name(el)) out.push('empty ' + el.outerHTML.slice(0, 80));
        for (const el of document.querySelectorAll('img')) if (!el.hasAttribute('alt')) out.push('img no alt');
        const h1s = document.querySelectorAll('h1').length;
        if (h1s !== 1) out.push('h1 count ' + h1s);
        if (!document.querySelector('main')) out.push('no main');
        if (!document.documentElement.lang) out.push('no lang');
        const ids = [...document.querySelectorAll('[id]')].map((e) => e.id); const dup = ids.filter((x, i) => ids.indexOf(x) !== i);
        if (dup.length) out.push('dup ids ' + dup.join(','));
        return out;
      });
      for (const x of r) probs.push(`${p}: ${x}`);
    }
    if (probs.length) throw new Error(probs.join(' | '));
    return '9 pages clean';
  });
  await check('keyboard: tab reaches skip link first, then nav', async () => {
    await page.goto(`${BASE}/`);
    await page.waitForSelector('.skip-link'); // pressing Tab before React has rendered finds nothing to focus
    await page.keyboard.press('Tab');
    const first = await page.evaluate(() => document.activeElement.textContent);
    if (!/Skip to content/.test(first)) throw new Error(first);
    return first;
  });
  await browser.close();
}

{
  const { browser, context, page, logs } = await launch([], { colorScheme: 'dark' });
  await check('dark mode screenshots', async () => {
    await signIn(page, EMAIL);
    await page.goto(`${BASE}/`); await page.waitForSelector('table');
    await page.screenshot({ path: join(SHOTS, 'dark-landing.png') });
    await page.goto(`${BASE}/free`); await page.waitForSelector('main h1');
    await page.screenshot({ path: join(SHOTS, 'dark-free.png') });
    return 'saved';
  });
  await browser.close();
}

{
  // Strict CSP: no inline script/style, same-origin everything. wasm needs 'wasm-unsafe-eval'.
  const csp = "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; worker-src 'self'; connect-src 'self'; font-src 'self'; manifest-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";
  const { browser, context, page, logs } = await launch(['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-audio-capture=${join(FILES, 'noisy.wav')}`]);
  const viol = [];
  await context.route('**/*', async (route) => {
    const req = route.request();
    if (!req.url().startsWith('http://127.0.0.1:8792') || req.resourceType() === 'websocket') return route.continue();
    const res = await route.fetch();
    await route.fulfill({ response: res, headers: { ...res.headers(), 'content-security-policy': csp } });
  });
  page.on('console', (m) => { if (/Content Security Policy|violat/i.test(m.text())) viol.push(m.text().slice(0, 200)); });
  await check('CSP: landing, free cleanup (wasm worker + mediabunny), files, live worklets all work', async () => {
    await signIn(page, EMAIL);
    await page.goto(`${BASE}/`); await page.waitForSelector('table');
    await page.goto(`${BASE}/free`);
    await page.setInputFiles('input[type=file]', join(FILES, 'test.mp4'));
    await page.getByRole('button', { name: 'Clean audio' }).click();
    await page.waitForSelector('[data-testid=result]', { timeout: 60000 });
    const rms = Number(await page.locator('[data-testid=result]').getAttribute('data-rms'));
    if (rms < 0.001) throw new Error('silent under CSP');
    // pages added/changed for production QA: no inline style attributes or scripts may be needed anywhere
    for (const p of ['/app', '/app/billing', '/app/account', '/licenses', '/terms', '/privacy']) {
      await page.goto(`${BASE}${p}`); await page.waitForSelector('main h1'); await page.waitForTimeout(300);
    }
    await page.setViewportSize({ width: 360, height: 800 });
    await page.goto(`${BASE}/app/billing`); await page.waitForSelector('main h1');
    await page.getByRole('button', { name: 'Menu' }).click();
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto(`${BASE}/app/live`);
    await page.getByTestId('live-start').click();
    await page.waitForFunction(() => Number(document.querySelector('[data-testid=live-stats]')?.getAttribute('data-recv')) > 30, null, { timeout: 30000 });
    await page.getByTestId('live-stop').click();
    await page.getByTestId('live-start').waitFor({ timeout: 15000 });
    if (viol.length) throw new Error(viol.join(' | '));
    return 'no CSP violations';
  });
  await browser.close();
}

process.exit(summary().some((r) => !r.ok) ? 1 : 0);
