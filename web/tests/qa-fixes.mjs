// Checks for the production-QA frontend fixes (site config, licences, header at 360 px, billing-off state, price notes,
// account deletion, premium live captions, WAV -> FLAC upload). Needs the same local backend as tests/e2e.mjs
// (port 8792, see helpers.mjs) with `npm run build` output served by it, and the fixtures from `npm run test:fixtures`.
// Usage: node tests/qa-fixes.mjs [section ...]   sections: site licenses header billing account live flac
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { BASE, FILES, SHOTS, admin, check, dropFile, launch, loginLink, pick, signIn, summary } from './helpers.mjs';

const only = process.argv.slice(2);
const want = (s) => only.length === 0 || only.includes(s);
const EMAIL = `qa-${Date.now()}@example.com`;
const FFMPEG = process.env.FFMPEG || join(homedir(), 'audio-cleaner', 'bin', 'ffmpeg');

admin('login-link', EMAIL); // creates the user
admin('grant', EMAIL, '5', 'qa test');

const ctxOpts = (extra = {}) => ({ ...extra });
async function freshPage(extraArgs = [], opts = {}) {
  const l = await launch(extraArgs, ctxOpts(opts));
  return l;
}

/** fetch() with the CSRF token, run inside the page (same origin). */
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

if (want('site')) {
  const { browser, page } = await freshPage();
  await check('site config: no placeholders in footer or legal pages; draft banner while legal fields are empty', async () => {
    const bad = [];
    for (const p of ['/', '/terms', '/privacy', '/licenses']) {
      await page.goto(`${BASE}${p}`);
      await page.waitForSelector('main h1');
      const text = await page.locator('body').innerText();
      for (const re of [/example\.com/i, /\(placeholder\)/i, /\[OPERATOR/, /\[JURISDICTION/, /\[VENUE/, /\[postal address/i, /\[support@/i, /\[privacy@/i, /\[placeholder\]/i, /to be attached before launch/i]) {
        if (re.test(text)) bad.push(`${p}: ${re}`);
      }
      if (p === '/') {
        if (/Contact:/.test(await page.locator('footer').innerText())) bad.push('footer shows a contact while supportEmail is empty');
        if (await page.locator('footer a[href^="mailto:"]').count()) bad.push('footer mailto link');
      }
    }
    await page.goto(`${BASE}/terms`);
    await page.waitForSelector('main h1');
    if (await page.locator('.draft-banner').count() || /legal review|draft/i.test(await page.locator('main').innerText())) bad.push('terms: draft/review wording is back');
    if (!(await page.locator('main').innerText()).includes('operated by the operator of this site')) bad.push('terms: neutral operator wording missing');
    await page.goto(`${BASE}/privacy`);
    await page.waitForSelector('main h1');
    if (/legal review|draft/i.test(await page.locator('main').innerText())) bad.push('privacy: draft/review wording is back');
    if (bad.length) throw new Error(bad.join(' | '));
    return 'clean';
  });
  await browser.close();
}

if (want('licenses')) {
  const { browser, page } = await freshPage();
  await check('licences: every component links to a real licence text', async () => {
    await page.goto(`${BASE}/licenses`);
    await page.waitForSelector('[data-testid=licenses] tbody tr');
    const links = await page.locator('[data-testid=licenses] a[href*="/licenses/"]:not([href^="http"])').evaluateAll((as) => as.map((a) => ({ href: a.getAttribute('href'), text: a.textContent })));
    if (links.length < 12) throw new Error(`only ${links.length} licence links`);
    const rows = await page.locator('[data-testid=licenses] tbody tr').count();
    const seen = new Set();
    const sizes = [];
    for (const l of links) {
      if (seen.has(l.href)) continue;
      seen.add(l.href);
      const res = await page.request.get(new URL(l.href, BASE + '/').href);
      const body = await res.text();
      if (!res.ok() || body.length < 500) throw new Error(`${l.href}: ${res.status()} ${body.length} bytes`);
      if (!/MIT|Apache|Mozilla Public|Lesser General Public|Permission is hereby granted|Creative Commons Attribution/.test(body.slice(0, 4000))) throw new Error(`${l.href}: does not look like a licence`);
      if (/^\s*<!doctype html/i.test(body)) throw new Error(`${l.href}: got the SPA shell`);
      sizes.push(`${l.href.split('/').pop()}=${body.length}`);
    }
    const main = await page.locator('main').innerText();
    for (const need of ['tract', 'mediabunny', 'React', 'Scheduler', 'React Router', 'Vite', 'Rust crates', 'FastConformer', 'sherpa-onnx', 'CC BY 4.0']) if (!main.includes(need)) throw new Error('missing ' + need);
    // the Creative Commons licence text is not shipped, it is linked
    if (!(await page.locator('[data-testid=licenses] a[href^="https://creativecommons.org/licenses/by/4.0/"]').count())) throw new Error('no link to the CC BY 4.0 licence');
    if (/to be attached/i.test(main)) throw new Error('placeholder text still present');
    return `${rows} components, ${seen.size} licence files: ${sizes.join(' ')}`;
  });
  await browser.close();
}

if (want('header')) {
  for (const [label, signed] of [['signed out', false], ['signed in', true]]) {
    for (const width of [360, 320]) {
      const { browser, page } = await freshPage([], { viewport: { width, height: 740 } });
      if (signed) await signIn(page, EMAIL);
      await check(`header ${width}px ${label}: menu toggle stays on the header row; menu opens as a panel`, async () => {
        const probs = [];
        for (const p of ['/', '/app/billing']) {
          if (!signed && p === '/app/billing') continue;
          await page.goto(`${BASE}${p}`);
          await page.waitForSelector('main h1');
          await page.waitForTimeout(300);
          const g = await page.evaluate(() => {
            const r = (sel) => { const e = document.querySelector(sel); if (!e) return null; const b = e.getBoundingClientRect(); return { top: b.top, bottom: b.bottom, left: b.left, right: b.right, h: b.height }; };
            return { header: r('.site-header'), toggle: r('.nav-toggle'), brand: r('.brand'), right: r('.header-right'), chip: r('.balance-chip'), vw: window.innerWidth, sw: document.documentElement.scrollWidth };
          });
          const mid = (b) => (b.top + b.bottom) / 2;
          if (!g.toggle || g.toggle.h === 0) probs.push(`${p}: no toggle`);
          else {
            if (g.header.h > 72) probs.push(`${p}: header ${g.header.h}px tall (wrapped)`);
            if (Math.abs(mid(g.toggle) - mid(g.brand)) > 8) probs.push(`${p}: toggle not on the brand row`);
            if (g.toggle.right > g.vw + 0.5 || g.toggle.left < 0) probs.push(`${p}: toggle off-screen`);
            if (g.right.right > g.toggle.left + 1) probs.push(`${p}: header-right overlaps the toggle`);
            if (g.brand.right > g.right.left + 1) probs.push(`${p}: brand overlaps account controls`);
          }
          if (g.sw > g.vw + 1) probs.push(`${p}: horizontal overflow ${g.sw}>${g.vw}`);
        }
        await page.getByRole('button', { name: 'Menu' }).click();
        await page.getByRole('link', { name: 'Cleanup', exact: true }).first().waitFor({ state: 'visible' });
        const open = await page.evaluate(() => ({ header: document.querySelector('.site-header').getBoundingClientRect().height, nav: document.querySelector('#main-nav').getBoundingClientRect(), vw: window.innerWidth }));
        if (open.header > 72) probs.push(`menu open: header grew to ${open.header}px`);
        if (open.nav.right > open.vw + 1 || open.nav.left < -1) probs.push('menu panel overflows');
        await page.screenshot({ path: join(SHOTS, `header-${width}-${signed ? 'in' : 'out'}-open.png`) });
        if (probs.length) throw new Error(probs.join(' | '));
        return 'ok';
      });
      await browser.close();
    }
  }
}

if (want('billing')) {
  const { browser, page } = await freshPage();
  await signIn(page, EMAIL);
  await check('billing off: Pay, refund and card buttons disabled with notes; billing-rule note shown', async () => {
    const now = Math.floor(Date.now() / 1000);
    await page.route('**/api/billing/history', (r) => r.fulfill({ json: {
      payments: [{ id: 'pay1', kind: 'topup', status: 'succeeded', amount_cents: 2000, refunded_cents: 0, refundable_cents: 2000, created_at: now - 3600 }],
      ledger: [],
    } }));
    await page.goto(`${BASE}/app/billing`);
    await page.waitForSelector('[data-testid=payments-off]');
    const pay = page.getByRole('button', { name: /^Pay / });
    if (!(await pay.isDisabled())) throw new Error('Pay enabled');
    const refund = page.getByRole('button', { name: 'Refund unused' });
    await refund.waitFor();
    if (!(await refund.isDisabled())) throw new Error('Refund enabled');
    const notes = [await page.getByTestId('topup-off').innerText(), await page.getByTestId('refund-off').innerText()];
    // with payments on, nothing is disabled (pricing says so)
    await page.unroute('**/api/billing/history');
    await page.route('**/api/billing/history', (r) => r.fulfill({ json: { payments: [{ id: 'pay1', kind: 'topup', status: 'succeeded', amount_cents: 2000, refunded_cents: 0, refundable_cents: 2000, created_at: now - 3600 }], ledger: [] } }));
    await page.route('**/api/billing/pricing', async (r) => {
      const res = await r.fetch();
      const j = await res.json();
      await r.fulfill({ response: res, json: { ...j, paymentsEnabled: true } });
    });
    await page.goto(`${BASE}/app/billing`);
    await page.getByRole('button', { name: 'Refund unused' }).waitFor();
    if (await page.getByRole('button', { name: /^Pay / }).isDisabled()) throw new Error('Pay disabled although payments are enabled');
    if (await page.getByRole('button', { name: 'Refund unused' }).isDisabled()) throw new Error('Refund disabled although payments are enabled');
    if (await page.getByTestId('topup-off').count()) throw new Error('note shown although payments are enabled');
    return notes.join(' / ').slice(0, 160);
  });
  await check('"Billed per second of audio, no minimum charge." near the price table and the dashboard estimate (no $0.01 floor)', async () => {
    await page.goto(`${BASE}/`);
    await page.waitForSelector('table tbody tr');
    const landing = await page.getByTestId('billing-note').innerText();
    if (landing !== 'Billed per second of audio, no minimum charge.') throw new Error(landing);
    await page.goto(`${BASE}/app`);
    await page.waitForSelector('[data-testid=available]');
    await dropFile(page, join(FILES, 'noisy.wav'));
    await pick(page, 'noise', 'denoise');
    const est = await page.getByTestId('estimate').innerText();
    if (!/Billed per second of audio, no minimum charge\./.test(est)) throw new Error(est);
    const m = /Estimated cost: \$([0-9.]+)/.exec(est);
    if (!m || Number(m[1]) >= 0.01) throw new Error('estimate still has a $0.01 minimum: ' + est);
    return est.split('\n')[0];
  });
  await browser.close();
}

if (want('account')) {
  const mail = `qa-del-${Date.now()}@example.com`;
  const { browser, page } = await freshPage();
  await page.goto(loginLink(mail));
  await page.waitForURL(/\/audio\/app$/);
  await check('account delete: 409 reason is shown in an alert; after success lands on / with a one-time banner (no bounce to /login)', async () => {
    admin('grant', mail, '1', 'qa delete test'); // a running job (jobs here are paid) blocks the delete
    const made = await apiCall(page, 'POST', 'jobs', { kind: 'transcribe', transcribeTier: 'local', filename: 'qa-hold.wav', sizeBytes: 1000, durationSeconds: 5, options: { audio_format: 'wav', caption_formats: ['txt'] } });
    if (made.status !== 200) throw new Error('could not create a held job: ' + JSON.stringify(made));
    await page.goto(`${BASE}/app/account`);
    await page.getByLabel(/Type your email/).fill(mail);
    await page.getByRole('button', { name: 'Delete my account' }).click();
    const alert = page.getByTestId('delete-error');
    await alert.waitFor({ timeout: 10000 });
    const text = await alert.innerText();
    if (!text.includes('Wait for your running jobs to finish first.')) throw new Error(text);
    if ((await alert.getAttribute('role')) !== 'alert') throw new Error('not role=alert');
    if (!(await alert.isVisible())) throw new Error('not visible');
    const inView = await alert.evaluate((el) => { const b = el.getBoundingClientRect(); return b.top >= 0 && b.bottom <= window.innerHeight; });
    if (!inView) throw new Error('alert is outside the viewport');
    await page.screenshot({ path: join(SHOTS, 'account-409.png') });
    const c = await apiCall(page, 'POST', `jobs/${made.body.id}/cancel`);
    if (c.status !== 200) throw new Error('cancel failed ' + JSON.stringify(c));
    return text;
  });
  await check('account delete success: banner on /, URL stays /, header signed out, reload clears banner, /app -> /login', async () => {
    await page.getByLabel(/Type your email/).fill(mail);
    await page.getByRole('button', { name: 'Delete my account' }).click();
    await page.waitForURL(/\/audio\/?$/, { timeout: 15000 });
    const banner = page.getByTestId('account-deleted');
    await banner.waitFor({ timeout: 10000 });
    if (!(await banner.innerText()).includes('Your account has been deleted.')) throw new Error(await banner.innerText());
    await page.waitForSelector('.site-header a[href$="/login"]', { timeout: 10000 });
    await page.waitForTimeout(800);
    if (!/\/audio\/?$/.test(page.url())) throw new Error('bounced to ' + page.url());
    if (!(await banner.isVisible())) throw new Error('banner vanished');
    if (await page.evaluate(() => JSON.stringify(history.state || {}).includes('account-deleted'))) throw new Error('notice left in history state');
    await page.screenshot({ path: join(SHOTS, 'account-deleted.png') });
    await page.reload();
    await page.waitForSelector('main h1');
    if (await page.getByTestId('account-deleted').count()) throw new Error('banner survived a reload');
    await page.goto(`${BASE}/app`);
    await page.waitForURL(/\/login$/, { timeout: 10000 });
    return 'ok';
  });
  await browser.close();
}

if (want('live')) {
  const wav = join(FILES, 'noisy.wav');
  const { browser, page } = await freshPage(['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-audio-capture=${wav}`]);
  await signIn(page, EMAIL);
  await apiCall(page, 'POST', 'account/tos', { version: '2026-10-16' });
  await check('live: price of the chosen combination (denoise light + captions tier) comes from /api/billing/pricing', async () => {
    await page.goto(`${BASE}/app/live`);
    await page.getByTestId('live-mode').locator('label', { hasText: 'Paid, done on our side' }).click();
    await page.getByTestId('live-price').waitFor();
    const price = async () => {
      const el = page.getByTestId('live-price');
      return { usd: Number(await el.getAttribute('data-usd-per-minute')), text: await el.innerText() };
    };
    const pricing = await (await page.request.get(`${BASE}/api/billing/pricing`)).json();
    const r = (p, t) => pricing.rates.find((x) => x.product === p && x.tier === t).usdPerMinute;
    const out = [];
    for (const [cap, den] of [['local', true], ['premium', true], ['premium', false], ['', true]]) {
      await page.locator(`label.choice:has(input[name=live-captions][value="${cap}"])`).click();
      const box = page.getByRole('checkbox', { name: /Remove noise/ });
      if ((await box.isChecked()) !== den) await box.setChecked(den);
      const want = (den ? r('stream_denoise', 'light') : 0) + (cap ? r('stream_captions', cap) : 0);
      const got = await price();
      if (want > 0 && Math.abs(got.usd - want) > 1e-9) throw new Error(`${cap}/${den}: ${got.usd} != ${want}`);
      if (want > 0 && !got.text.includes(`$${want.toFixed(4)}`)) throw new Error(`${cap}/${den}: text lacks $${want.toFixed(4)}: ${got.text}`);
      out.push(`${cap || 'none'}${den ? '+denoise' : ''}=$${want.toFixed(4)}`);
    }
    const opts = await page.locator('label.choice:has(input[name=live-captions])').allInnerTexts();
    if (!opts.some((o) => /Extra accurate[\s\S]*any language/.test(o))) throw new Error('no Extra accurate option that says "any language": ' + opts.join('|'));
    return out.join(', ');
  });
  await check('live: premium captions stream partials by id and finalize them in place (mocked server)', async () => {
    let startMsg = null;
    const send = [];
    await page.routeWebSocket(/\/api\/stream\/ws/, (ws) => {
      ws.onMessage((m) => {
        if (typeof m === 'string') {
          const j = JSON.parse(m);
          if (j.type === 'start') {
            startMsg = j;
            ws.send(JSON.stringify({ type: 'ready', sessionId: 's1', sampleRate: 48000, hopSamples: 480, modelLatencyMs: 30, usdPerMinute: 0.029, maxSessionSeconds: 3600 }));
          } else if (j.type === 'stop') ws.close();
        }
      });
      send.push((o) => ws.send(JSON.stringify({ type: 'caption', ...o })));
    });
    await page.goto(`${BASE}/app/live`);
    await page.getByTestId('live-mode').locator('label', { hasText: 'Paid, done on our side' }).click();
    await page.locator('label.choice:has(input[name=live-captions][value=premium])').click();
    const box = page.getByRole('checkbox', { name: /Remove noise/ });
    if (!(await box.isChecked())) await box.check();
    await page.getByTestId('live-start').click();
    await page.getByTestId('live-stop').waitFor({ timeout: 20000 });
    if (startMsg?.captions !== 'premium' || startMsg?.denoise !== 'light') throw new Error('start message: ' + JSON.stringify(startMsg));
    const emit = send[0];
    const partials = () => page.getByTestId('caption-partial').allInnerTexts();
    const lines = () => page.getByTestId('caption-line').allInnerTexts();
    emit({ id: 'item_a', final: false, text: 'hello' });
    await page.getByTestId('caption-partial').first().waitFor();
    await page.evaluate(() => { document.querySelector('[data-testid=caption-partial]').__mark = 'A'; });
    emit({ id: 'item_a', final: false, text: 'hello wor' });
    await page.waitForFunction(() => document.querySelector('[data-testid=caption-partial]')?.textContent === 'hello wor');
    if ((await partials()).length !== 1) throw new Error('partial update duplicated the line: ' + (await partials()).join('|'));
    emit({ id: 'item_b', final: false, text: 'second' });
    await page.waitForFunction(() => document.querySelectorAll('[data-testid=caption-partial]').length === 2);
    emit({ id: 'item_a', final: true, text: 'Hello world.' });
    await page.waitForFunction(() => document.querySelectorAll('[data-testid=caption-line]').length === 1);
    const mark = await page.evaluate(() => document.querySelector('[data-testid=caption-line]').__mark);
    if (mark !== 'A') throw new Error('final did not replace the partial in place (different DOM node)');
    if ((await lines())[0] !== 'Hello world.') throw new Error((await lines()).join('|'));
    if ((await partials()).join('|') !== 'second') throw new Error('partials: ' + (await partials()).join('|'));
    emit({ id: 'item_a', final: false, text: 'late partial' }); // must not reopen a finished line
    emit({ id: 'item_b', final: true, text: 'Second one.' });
    await page.waitForFunction(() => document.querySelectorAll('[data-testid=caption-line]').length === 2);
    const order = (await lines()).join('|');
    if (order !== 'Hello world.|Second one.') throw new Error('order/content: ' + order);
    if ((await partials()).length) throw new Error('partials left over');
    // standard captions: id-less partial replaced in place, then final
    emit({ final: false, text: 'plain partial' });
    await page.waitForFunction(() => document.querySelector('[data-testid=caption-partial]')?.textContent === 'plain partial');
    emit({ final: false, text: 'plain partial two' });
    await page.waitForFunction(() => document.querySelector('[data-testid=caption-partial]')?.textContent === 'plain partial two');
    emit({ final: true, text: 'Plain final.' });
    await page.waitForFunction(() => document.querySelectorAll('[data-testid=caption-line]').length === 3 && !document.querySelector('[data-testid=caption-partial]'));
    await page.screenshot({ path: join(SHOTS, 'live-premium.png'), fullPage: true });
    await page.getByTestId('live-stop').click();
    await page.getByTestId('live-start').waitFor({ timeout: 15000 });
    return 'partials updated in place, finals kept order';
  });
  await browser.close();
}

if (want('flac')) {
  // 24-bit stereo WAV + AIFF fixtures derived from noisy.wav
  const f24 = join(FILES, 'noisy24st.wav');
  const faiff = join(FILES, 'noisy.aiff');
  execFileSync(FFMPEG, ['-v', 'error', '-y', '-i', join(FILES, 'noisy.wav'), '-ac', '2', '-c:a', 'pcm_s24le', f24]);
  execFileSync(FFMPEG, ['-v', 'error', '-y', '-i', join(FILES, 'noisy.wav'), '-c:a', 'pcm_s16be', faiff]);
  const { browser, page, logs } = await freshPage();
  await signIn(page, EMAIL);
  await apiCall(page, 'POST', 'account/tos', { version: '2026-10-16' });
  // The size of every blob PUT through XMLHttpRequest (Playwright cannot see XHR blob bodies).
  await page.addInitScript(() => {
    window.__puts = [];
    const send = XMLHttpRequest.prototype.send;
    const open = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function (m, u, ...rest) { this.__m = m; this.__u = u; return open.call(this, m, u, ...rest); };
    XMLHttpRequest.prototype.send = function (body) { if (this.__m === 'PUT' && body && body.size != null) window.__puts.push({ url: this.__u, size: body.size }); return send.call(this, body); };
  });
  const reqs = [];
  page.on('request', (r) => {
    const u = new URL(r.url());
    if (r.method() === 'POST' && u.pathname.endsWith('/api/jobs')) reqs.push({ kind: 'create', body: r.postDataJSON() });
    if (r.method() === 'PUT' && /\/input$/.test(u.pathname)) reqs.push({ kind: 'put', req: r });
  });
  const putSize = async () => { const puts = await page.evaluate(() => window.__puts); return { length: puts[puts.length - 1]?.size ?? -1, count: puts.length }; };
  const sha256 = (b) => createHash('sha256').update(b).digest('hex');
  const jobOutputHash = (id) => page.evaluate(async (id) => {
    const j = await (await fetch(`/audio/api/jobs/${id}`)).json();
    const audio = j.outputs.find((o) => o.kind === 'audio');
    const buf = await (await fetch(`/audio/api/jobs/${id}/outputs/${encodeURIComponent(audio.name)}`)).arrayBuffer();
    const h = await crypto.subtle.digest('SHA-256', buf);
    return { name: audio.name, bytes: buf.byteLength, sha: [...new Uint8Array(h)].map((x) => x.toString(16).padStart(2, '0')).join('') };
  }, id);
  const waitDone = async (id) => {
    for (let i = 0; i < 240; i++) {
      const j = await apiCall(page, 'GET', `jobs/${id}`);
      if (['done', 'failed', 'canceled'].includes(j.body.status)) return j.body;
      await page.waitForTimeout(2500);
    }
    throw new Error('job timeout');
  };
  /** Uploads the raw file through the API, bypassing the UI (reference for "same result as before"). */
  const rawJob = async (file) => {
    const bytes = readFileSync(file);
    const name = file.split('/').pop();
    const dur = 10.6;
    const made = await apiCall(page, 'POST', 'jobs', { kind: 'denoise', denoiseTier: 'standard', filename: name, sizeBytes: bytes.length, durationSeconds: dur, options: { audio_format: 'wav', loudness_lufs: -16, atten_lim_db: 100, caption_formats: [] } });
    if (made.status !== 200) throw new Error('raw create: ' + JSON.stringify(made));
    const put = await page.evaluate(async ({ url, b64 }) => {
      const me = await (await fetch('/audio/api/me')).json();
      const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const r = await fetch(`/audio/${url}`, { method: 'PUT', headers: { 'X-CSRF-Token': me.csrfToken, 'Content-Type': 'application/octet-stream' }, body: bin });
      return r.status;
    }, { url: made.body.uploadUrl, b64: bytes.toString('base64') });
    if (put !== 200) throw new Error('raw upload ' + put);
    return made.body.id;
  };

  for (const [file, label] of [['noisy.wav', 'mono 16-bit WAV'], ['noisy24st.wav', 'stereo 24-bit WAV'], ['noisy.aiff', 'mono 16-bit AIFF']]) {
    await check(`flac upload (${label}): compressed in the browser, note shown, declared name/size match, job succeeds, result identical to uploading the original`, async () => {
      reqs.length = 0;
      await page.goto(`${BASE}/app`);
      await page.waitForSelector('[data-testid=available]');
      await page.evaluate(() => { window.__puts.length = 0; });
      await dropFile(page, join(FILES, file));
      await pick(page, 'noise', 'denoise');
      await page.getByTestId('estimate').waitFor();
      await page.getByRole('button', { name: /^Start – about/ }).click();
      const banner = await page.getByTestId('submitted').innerText({ timeout: 90000 });
      const m = /Compressed for upload: ([\d.]+ [KMG]?B) → ([\d.]+ [KMG]?B)/.exec(banner);
      if (!m) throw new Error('no compression note: ' + banner);
      const create = reqs.find((r) => r.kind === 'create');
      const put = await putSize();
      const orig = statSync(join(FILES, file)).size;
      const stem = file.replace(/\.[^.]+$/, '');
      if (create.body.filename !== `${stem}.flac`) throw new Error('declared filename ' + create.body.filename);
      if (create.body.sizeBytes !== put.length) throw new Error(`declared ${create.body.sizeBytes} != uploaded ${put.length}`);
      if (!(put.length < orig)) throw new Error(`not smaller: ${put.length} vs ${orig}`);
      const jobs = await apiCall(page, 'GET', 'jobs');
      const mine = jobs.body.jobs[0];
      const done = await waitDone(mine.id);
      if (done.status !== 'done') throw new Error('job ' + done.status + ' ' + (done.error || ''));
      // the upload is the .flac; the job list shows the file the visitor chose
      if (create.body.displayName !== file) throw new Error('declared display name ' + create.body.displayName);
      if (done.inputName !== file) throw new Error('server input name ' + done.inputName);
      const viaFlac = await jobOutputHash(mine.id);
      const rawId = await rawJob(join(FILES, file));
      const rawDone = await waitDone(rawId);
      if (rawDone.status !== 'done') throw new Error('raw job ' + rawDone.status + ' ' + (rawDone.error || ''));
      const viaRaw = await jobOutputHash(rawId);
      if (viaFlac.sha !== viaRaw.sha) throw new Error(`outputs differ: ${viaFlac.bytes} B vs ${viaRaw.bytes} B`);
      await page.screenshot({ path: join(SHOTS, `flac-${stem}.png`), fullPage: true });
      return `${orig} B -> ${put.length} B (${(100 * put.length / orig).toFixed(0)}%); banner "${m[0]}"; output identical (${viaFlac.bytes} B, ${viaFlac.sha.slice(0, 12)})`;
    });
  }
  await check('flac upload: non-WAV files are untouched (mp3 keeps its name and size)', async () => {
    reqs.length = 0;
    await page.goto(`${BASE}/app`);
    await page.waitForSelector('[data-testid=available]');
    await page.evaluate(() => { window.__puts.length = 0; });
    await dropFile(page, join(FILES, 'noisy.mp3'));
    await pick(page, 'noise', 'denoise');
    await page.getByTestId('estimate').waitFor();
    await page.getByRole('button', { name: /^Start – about/ }).click();
    const banner = await page.getByTestId('submitted').innerText({ timeout: 90000 });
    if (/Compressed for upload/.test(banner)) throw new Error('mp3 was "compressed": ' + banner);
    const create = reqs.find((r) => r.kind === 'create');
    const put = await putSize();
    if (create.body.filename !== 'noisy.mp3' || create.body.sizeBytes !== statSync(join(FILES, 'noisy.mp3')).size || put.length !== create.body.sizeBytes) throw new Error(JSON.stringify([create.body.filename, create.body.sizeBytes, put.length]));
    return banner;
  });
  await check('flac upload: Standard transcription of a WAV uploads FLAC and succeeds', async () => {
    reqs.length = 0;
    await page.goto(`${BASE}/app`);
    await page.waitForSelector('input[type=file]');
    await dropFile(page, join(FILES, 'noisy.wav'));
    await pick(page, 'noise', 'off');
    await pick(page, 'subs', 'standard');
    await page.getByTestId('estimate').waitFor();
    await page.getByRole('button', { name: /^Start – about/ }).click();
    const banner = await page.getByTestId('submitted').innerText({ timeout: 90000 });
    if (!/Compressed for upload/.test(banner)) throw new Error(banner);
    const jobs = await apiCall(page, 'GET', 'jobs');
    const done = await waitDone(jobs.body.jobs[0].id);
    if (done.status !== 'done' || done.inputName !== 'noisy.wav') throw new Error(done.status + ' ' + done.inputName + ' ' + (done.error || ''));
    return banner;
  });
  console.log('LOGS:\n' + logs.join('\n'));
  await browser.close();
}

process.exit(summary().some((r) => !r.ok) ? 1 : 0);
