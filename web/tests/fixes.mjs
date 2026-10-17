// Website fixes that keep their own checks: the pure safety helpers (a return link that must stay on this site, the WAV size
// limit), a page file that cannot be loaded (one retry, then a friendly screen, never a white page), and the delete-now
// confirmation for a finished job. The rest of the old audit checks are covered by the suites next to this one.
// Needs the local test backend (port 8792, see helpers.mjs) serving `npm run build`. Usage: node tests/fixes.mjs [section ...]
// sections: pure pages jobs
import { BASE, check, launch, signIn, summary } from './helpers.mjs';

const only = process.argv.slice(2);
const want = (s) => only.length === 0 || only.includes(s);
const EMAIL = process.env.E2E_EMAIL || `fixes-${Date.now()}@example.com`;
const now = () => Math.floor(Date.now() / 1000);

if (want('pure')) {
  await check('pure helpers: a return link must stay on this site (no other host, no backslash, no line break); a WAV above 4 GB is refused (6 h stereo 24-bit is not a WAV)', async () => {
    const { wavFits } = await import('../src/media/wav.js');
    const { isSafeReturn } = await import('../src/returnTo.js');
    const h = 3600;
    if (!wavFits(6 * h, 2, 16)) throw new Error('6 h stereo 16-bit should fit');
    if (wavFits(6 * h, 2, 24)) throw new Error('6 h stereo 24-bit must not fit (over 4 GB)');
    if (!wavFits(4 * h, 2, 24) || wavFits(4.2 * h, 2, 24)) throw new Error('the 4 h / 4.2 h boundary moved');
    for (const ok of ['/app', '/app/billing', '/terms?x=1']) if (!isSafeReturn(ok)) throw new Error(`${ok} should be accepted`);
    for (const bad of ['//evil.example', '/\\evil.example', '/app\\..\\x', '/login', '/login/verify', 'https://evil.example', '', null, '/app\n/x']) if (isSafeReturn(bad)) throw new Error(`${JSON.stringify(bad)} should be refused`);
    return 'ok';
  });
}

if (want('pages')) {
  const { browser, page } = await launch();
  await check('a page file that cannot be loaded: one retry, one reload, then a friendly screen with Reload (never a white page); Reload shows the page again', async () => {
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
    if ((await page.evaluate(() => sessionStorage.getItem('acw-chunk-reload'))) !== '1') throw new Error('reload guard not set');
    await page.unroute('**/assets/Terms-*.js');
    await page.getByRole('button', { name: 'Reload' }).click();
    await page.getByRole('heading', { name: 'Terms of Service' }).waitFor({ timeout: 15000 });
    if ((await page.evaluate(() => sessionStorage.getItem('acw-chunk-reload'))) !== null) throw new Error('guard not cleared after a good load');
    return `file requested ${served}x, ${loads} loads, then Terms shown`;
  });
  await browser.close();
}

if (want('jobs')) {
  const first = await launch();
  await signIn(first.page, EMAIL);
  const state = await first.context.storageState();
  await first.browser.close();
  const job = (id, extra) => ({
    id, kind: 'denoise', free: false, status: 'done', inputName: `${id}.wav`, createdAt: now(), durationSeconds: 60, progress: 0, stage: '',
    outputs: [], costMicros: 0, expiresAt: null, queuePosition: null, queueWaitSeconds: null, ...extra,
  });
  const { browser, page } = await launch([], { storageState: state });
  await check('"Delete now" on a finished job asks first; "Keep them" backs out; "Yes, delete" deletes it', async () => {
    const done = job('d1', { outputs: [{ name: 'noisy-cleaned.wav', kind: 'audio', bytes: 1000 }], expiresAt: now() + 3600, costMicros: 2600 });
    let deletes = 0;
    await page.route(/\/api\/jobs(\/d1)?$/, (r) => {
      const m = r.request().method();
      if (m === 'DELETE') { deletes += 1; return r.fulfill({ json: { ok: true } }); }
      if (m === 'GET') return r.fulfill({ json: { jobs: [done] } });
      return r.continue();
    });
    await page.goto(`${BASE}/app`);
    await page.getByTestId('available').waitFor();
    await page.getByRole('button', { name: 'Delete now' }).click();
    await page.getByTestId('delete-confirm').waitFor();
    if (deletes) throw new Error('deleted without asking');
    await page.getByRole('button', { name: 'Keep them' }).click();
    if (!(await page.getByRole('button', { name: 'Delete now' }).count())) throw new Error('did not go back');
    await page.getByRole('button', { name: 'Delete now' }).click();
    await page.getByRole('button', { name: 'Yes, delete' }).click();
    for (let i = 0; i < 20 && deletes === 0; i++) await page.waitForTimeout(100);
    if (deletes !== 1) throw new Error('deletes: ' + deletes);
    await page.unroute(/\/api\/jobs(\/d1)?$/);
    return 'asked first, kept on "Keep them", deleted on "Yes, delete"';
  });
  await browser.close();
}

process.exit(summary().some((r) => !r.ok) ? 1 : 0);
