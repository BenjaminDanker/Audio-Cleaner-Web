// The sign-in form (Google and email, with the bot check's token sent), an accessibility scan of the main pages (labels, names,
// one heading, landmarks, unique ids), and one session under the production Content-Security-Policy (read from deploy/vps/audio.caddy; only the tests' local model host is
// added to connect-src) that does the free cleanup,
// the free subtitles, an MP3 export and a live test: no policy violation may show up anywhere.
// Usage: node tests/ui.mjs
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { BASE, FILES, check, dropFile, launch, mirror, pick, signIn, summary, waitResult } from './helpers.mjs';

const EMAIL = process.env.E2E_EMAIL || 'e2e-main@example.com';

{
  const { browser, page } = await launch();
  await check('sign-in form: Google button, email form with the bot check (a Turnstile test key issues a token that is sent with the request)', async () => {
    await page.route('**/api/auth/providers', (r) => r.fulfill({ json: { github: true, google: true, email: true, turnstileSiteKey: '1x00000000000000000000AA' } }));
    let posted = null;
    await page.route('**/api/auth/email', (r) => { posted = r.request().postDataJSON(); return r.fulfill({ json: { ok: true } }); });
    await page.goto(`${BASE}/login`);
    const go = await page.getByRole('link', { name: /Google/ }).getAttribute('href');
    if (go !== '/audio/api/auth/oauth/google/start') throw new Error(go);
    await page.getByLabel('Email address').fill('someone@example.com');
    await page.waitForFunction(() => !document.querySelector('button[type=submit]').disabled, null, { timeout: 20000 });
    await page.getByRole('button', { name: /Email me/ }).click();
    await page.getByText('Check your inbox').waitFor();
    if (!posted?.turnstileToken) throw new Error('no token posted: ' + JSON.stringify(posted));
    return `posted token ${posted.turnstileToken.slice(0, 12)}…`;
  });
  await browser.close();
}

{
  const { browser, page } = await launch();
  await signIn(page, EMAIL);
  await check('a11y scan: unlabeled controls, buttons and images, heading count, landmarks, language and duplicate ids on the main pages (and the Cleanup page with a file)', async () => {
    const probs = [];
    for (const p of ['/', '/login', '/app', '/app+file', '/app/billing', '/app/live', '/app/account', '/terms']) {
      await page.goto(`${BASE}${p.replace('+file', '')}`);
      await page.waitForSelector('main h1');
      if (p.endsWith('+file')) await dropFile(page, join(FILES, 'noisy.wav'));
      await page.waitForTimeout(300);
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
    return '8 pages clean';
  });
  await browser.close();
}

{
  // The policy the front door sends (deploy/vps/audio.caddy): no inline script or style, same-origin everything, wasm needs
  // 'wasm-unsafe-eval', and the models may come from Hugging Face. The tests' local model host stands in for it.
  const caddy = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'deploy', 'vps', 'audio.caddy'), 'utf8');
  const real = /Content-Security-Policy "([^"]+)"/.exec(caddy)?.[1];
  if (!real || !/connect-src [^;]*https:\/\/huggingface\.co/.test(real)) throw new Error('no usable policy in deploy/vps/audio.caddy');
  const csp = real.replace(/connect-src /, `connect-src ${(await mirror()).origin} `);
  // (A page whose responses are made by route.fulfill below counts as a public page to Chromium, which would then refuse to reach the
  // local model host on loopback: that check is switched off for this one session.)
  const { browser, context, page } = await launch(['--disable-features=LocalNetworkAccessChecks', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-audio-capture=${join(FILES, 'speech.wav')}`], { acceptDownloads: true });
  const viol = [];
  await context.route('**/*', async (route) => {
    const req = route.request();
    if (!req.url().startsWith('http://127.0.0.1:8792') || req.resourceType() === 'websocket') return route.continue();
    const res = await route.fetch();
    await route.fulfill({ response: res, headers: { ...res.headers(), 'content-security-policy': csp } });
  });
  page.on('console', (m) => { if (/Content Security Policy|violat|refused to/i.test(m.text())) viol.push(m.text().slice(0, 200)); });
  page.on('pageerror', (e) => viol.push(e.message));
  await check('production CSP: the free cleanup (wasm worker), the free subtitles (speech model), an MP3 export (encoder worker) and a free live test run with no violation', async () => {
    await signIn(page, EMAIL);
    await page.goto(`${BASE}/app`);
    await dropFile(page, join(FILES, 'speech.wav'));
    await pick(page, 'noise', 'browser');
    await page.getByRole('button', { name: 'Start – free' }).click();
    await waitResult(page, 'result', 120000);
    if (Number(await page.locator('[data-testid=result]').getAttribute('data-rms')) < 0.001) throw new Error('silent under CSP');
    await page.getByTestId('save-as').locator('[data-testid=out-name]').fill('csp.mp3');
    const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 60000 }), page.getByTestId('save-as').locator('[data-testid=save-btn]').click()]);
    await dl.saveAs(join(FILES, 'csp-out.mp3'));
    await page.goto(`${BASE}/app`);
    await dropFile(page, join(FILES, 'speech.wav'));
    await pick(page, 'noise', 'off');
    await pick(page, 'subs', 'browser');
    await page.getByRole('button', { name: 'Start – free' }).click();
    await waitResult(page, 'subs-result');
    await page.goto(`${BASE}/app/live`);
    await page.getByTestId('live-start').click();
    await page.getByTestId('live-stop').waitFor({ timeout: 120000 });
    await page.waitForFunction(() => Number(document.querySelector('[data-testid=live-stats]')?.getAttribute('data-recv')) > 30, null, { timeout: 30000 });
    await page.getByTestId('live-stop').click();
    await page.getByTestId('live-start').waitFor({ timeout: 15000 });
    if (viol.length) throw new Error(viol.join(' | '));
    return 'no CSP violations';
  });
  await browser.close();
}

process.exit(summary().some((r) => !r.ok) ? 1 : 0);
