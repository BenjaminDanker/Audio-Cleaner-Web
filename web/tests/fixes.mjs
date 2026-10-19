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
    // the list on /app holds a short row with no Delete button: the job's own page has it
    await page.getByTestId('job-row').waitFor();
    if (await page.getByRole('button', { name: 'Delete now' }).count()) throw new Error('a Delete button on the list');
    await page.getByTestId('job-open').click();
    await page.waitForURL(/\/app\/job\/d1$/);
    await page.getByRole('button', { name: 'Delete now' }).click();
    await page.getByTestId('delete-confirm').waitFor();
    if (deletes) throw new Error('deleted without asking');
    await page.getByRole('button', { name: 'Keep them' }).click();
    if (!(await page.getByRole('button', { name: 'Delete now' }).count())) throw new Error('did not go back');
    await page.getByRole('button', { name: 'Delete now' }).click();
    await page.getByRole('button', { name: 'Yes, delete' }).click();
    for (let i = 0; i < 20 && deletes === 0; i++) await page.waitForTimeout(100);
    if (deletes !== 1) throw new Error('deletes: ' + deletes);
    await page.waitForURL(/\/app$/, { timeout: 10000 }); // after the delete the visitor is back on the Cleanup page
    await page.unroute(/\/api\/jobs(\/d1)?$/);
    return 'asked first, kept on "Keep them", deleted on "Yes, delete", back on /app';
  });
  // A job the server lists after a reload (this browser has no file or sound of it): the card lists what it made, with one
  // Download again, which brings exactly what was asked for; Change and save again can still change the subtitle formats.
  const stubFiles = {
    'noisy-cleaned.flac': 'fLaC-not-really-audio', 'noisy.srt': '1\n00:00:00,000 --> 00:00:01,000\nHello there.\n', 'noisy.vtt': 'WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello there.\n',
    'noisy-transcript.txt': 'Hello there.\n',
  };
  const stubJob = (id, names) => job(id, {
    kind: 'denoise_transcribe', inputName: 'noisy.flac', displayName: 'noisy.wav', expiresAt: now() + 3600, costMicros: 2600,
    outputs: names.map((n) => ({ name: n, kind: n.endsWith('.flac') ? 'audio' : n.endsWith('.txt') ? 'transcript' : 'captions', bytes: stubFiles[n].length, lang: n.endsWith('.flac') ? undefined : 'en' })),
  });
  const downloads = [];
  page.on('download', (d) => downloads.push(d));
  const withJobs = async (list, id) => {
    await page.route(/\/api\/jobs$/, (r) => (r.request().method() === 'GET' ? r.fulfill({ json: { jobs: list } }) : r.continue()));
    await page.route(`**/api/jobs/${id}/outputs/**`, (r) => {
      const name = decodeURIComponent(new URL(r.request().url()).pathname.split('/').pop());
      return r.fulfill({ body: stubFiles[name] ?? '', contentType: 'application/octet-stream' });
    });
  };
  const without = async (id) => { await page.unroute(/\/api\/jobs$/); await page.unroute(`**/api/jobs/${id}/outputs/**`); };
  const sorted = () => downloads.map((d) => d.suggestedFilename()).sort();
  const waitFor = async (n) => { for (let i = 0; i < 60 && downloads.length < n; i++) await page.waitForTimeout(250); await page.waitForTimeout(500); };

  await check('a finished job after a reload: its files are listed with their sizes, one Download again brings all of them, and no other download buttons', async () => {
    const j = stubJob('d2', ['noisy-cleaned.flac', 'noisy.srt', 'noisy.vtt', 'noisy-transcript.txt']);
    await withJobs([j], 'd2');
    try {
      await page.goto(`${BASE}/app/job/d2`);
      const card = page.getByTestId('job').first();
      await card.getByTestId('download-again').waitFor();
      const rows = (await card.getByTestId('made-file').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
      if (rows.length !== 4 || !rows.every((t) => /\d+ B$/.test(t))) throw new Error('listed: ' + JSON.stringify(rows));
      if ((await card.getByRole('button', { name: /download/i }).count()) !== 1 || (await card.getByRole('link', { name: /download/i }).count())) throw new Error('want exactly one download button');
      downloads.length = 0;
      await card.getByTestId('download-again').click();
      await waitFor(4);
      const want = ['noisy-cleaned.flac', 'noisy-transcript.txt', 'noisy.srt', 'noisy.vtt'];
      if (sorted().join() !== want.join()) throw new Error('downloaded: ' + sorted());
      return rows.join(' | ');
    } finally {
      await without('d2');
    }
  });

  await check('what was asked for decides what is downloaded: only the TXT and the sound were ticked (the server also kept a VTT for the video), so only those two come down; Change and save again adds the SRT', async () => {
    const j = stubJob('d3', ['noisy-cleaned.flac', 'noisy.srt', 'noisy.vtt', 'noisy-transcript.txt']);
    await withJobs([j], 'd3');
    await page.goto(`${BASE}/app/job/d3`);
    await page.evaluate(() => localStorage.setItem('acw.wants.d3', JSON.stringify({
      sound: true, format: 'flac', name: 'noisy-cleaned.flac', size: 'same', subs: true, caps: { srt: false, vtt: false, txt: true }, embed: false, soundHere: false, subsHere: false,
    })));
    try {
      await page.reload();
      const card = page.getByTestId('job').first();
      await card.getByTestId('download-again').waitFor();
      const rows = await card.getByTestId('made-file').locator('.oname').allInnerTexts();
      if (rows.sort().join() !== 'noisy-cleaned.flac,noisy-transcript.txt') throw new Error('listed: ' + rows);
      downloads.length = 0;
      await card.getByTestId('download-again').click();
      await waitFor(2);
      if (sorted().join() !== 'noisy-cleaned.flac,noisy-transcript.txt') throw new Error('downloaded: ' + sorted());
      await card.getByTestId('again').locator('summary').click();
      await card.getByTestId('again-cap-srt').check();
      downloads.length = 0;
      await card.getByTestId('again-save').click();
      await waitFor(3);
      if (sorted().join() !== 'noisy-cleaned.flac,noisy-transcript.txt,noisy.srt') throw new Error('after adding the SRT: ' + sorted());
      return 'TXT + sound, then + SRT';
    } finally {
      await without('d3');
    }
  });

  await browser.close();
}

process.exit(summary().some((r) => !r.ok) ? 1 : 0);
