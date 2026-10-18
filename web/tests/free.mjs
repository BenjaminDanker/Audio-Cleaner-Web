// Free noise removal in the browser (Cleanup page): what it makes and how it is downloaded, the file types it reads itself
// (WAV, MP3, WMA and AVI through libav.js, stereo at 44.1 kHz, MP4 video), that nothing about the file goes to our server, where
// the noise remover comes from (df.js from the site; the WebAssembly and the model from the model host, once, kept in the browser,
// a wrong file refused), that data saver mode downloads nothing before Start, and the free + free combination (browser noise
// removal + free subtitles, no job at all).
// Needs the local backend (port 8792, see helpers.mjs) with `ACW_MODEL_HOST=http://127.0.0.1:8798/ npm run build` output (the
// model host is mirror.mjs) and the fixtures (make-fixtures.sh).
// Usage: node tests/free.mjs
import { readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { BASE, FILES, SHOTS, check, dropFile, launch, mirror, pick, settleRecommendation, signIn, summary, waitResult } from './helpers.mjs';

const EMAIL = process.env.E2E_EMAIL || 'e2e-main@example.com'; // the free option needs an account
const SOURCE_FRAMES = 508591; // noisy.wav at 48 kHz; the other fixtures are made from it (tests/make-fixtures.sh)
const close = (frames) => Math.abs(frames - SOURCE_FRAMES) < SOURCE_FRAMES * 0.02; // codec padding is a few ms

const { browser, page, logs } = await launch();
const models = await mirror(); // the local model host: what the browser downloaded from it
const WASM = 'df_bg.wasm';
const MODEL = 'DeepFilterNet3_onnx.tar.gz';
const apiCalls = [];
const libavFetches = [];
page.on('request', (r) => {
  const u = new URL(r.url());
  if (u.pathname.includes('/audio/api/')) apiCalls.push(`${r.method()} ${u.pathname}`);
  if (u.pathname.includes('/dfn/libav/')) libavFetches.push(u.pathname.split('/').pop());
});
await signIn(page, EMAIL);
apiCalls.length = 0;

async function runFree(file, strengthLabel = 'Strong') {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, file)); // "Free, in your browser" is the default whenever the browser can open the file
  await page.getByTestId('strength-lstrength').getByText(strengthLabel, { exact: true }).click();
  const t0 = Date.now();
  await page.getByRole('button', { name: 'Start – free' }).click();
  await page.waitForSelector('[data-testid=result]', { timeout: 180000 });
  const res = page.locator('[data-testid=result]');
  return { ms: Date.now() - t0, rms: Number(await res.getAttribute('data-rms')), peak: Number(await res.getAttribute('data-peak')), frames: Number(await res.getAttribute('data-frames')), channels: Number(await res.getAttribute('data-channels')) };
}

await check('free: a mono 48 kHz WAV is cleaned in the browser (length kept, not silent), and the cleaned WAV downloads with the same length', async () => {
  const r = await runFree('noisy.wav');
  if (!close(r.frames)) throw new Error(`frames ${r.frames}`);
  if (r.rms < 0.001) throw new Error(`silent rms=${r.rms}`);
  if (libavFetches.length) throw new Error('a WAV fetched the extra decoder: ' + libavFetches.join(', '));
  const [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: /Cleaned WAV/ }).click()]);
  const p = join(FILES, 'out-free.wav');
  await dl.saveAs(p);
  const b = readFileSync(p);
  if (b.toString('ascii', 0, 4) !== 'RIFF') throw new Error('not RIFF');
  const got = b.readUInt32LE(40) / 2 / b.readUInt16LE(22);
  if (!close(got)) throw new Error(`downloaded frames ${got}`);
  await page.screenshot({ path: join(SHOTS, 'free-done.png'), fullPage: true });
  return `frames=${r.frames} rms=${r.rms} in ${r.ms} ms; download ${b.length} bytes`;
});

await check('free: nothing about the file is sent to our server (only the account, prices and the job list are read)', async () => {
  const odd = apiCalls.filter((c) => !/api\/me$|billing\/pricing$|^GET .*api\/jobs$/.test(c));
  if (odd.length) throw new Error(odd.join(', '));
  return apiCalls.join(', ');
});

await check('free: other inputs clean in the browser: 44.1 kHz stereo WAV (two channels kept), MP3', async () => {
  const st = await runFree('stereo44.wav');
  if (st.channels !== 2 || !close(st.frames) || st.rms < 0.001) throw new Error(JSON.stringify(st));
  const mp3 = await runFree('noisy.mp3');
  if (mp3.rms < 0.001) throw new Error('mp3 silent');
  return `stereo: frames=${st.frames}; mp3: frames=${mp3.frames}`;
});

await check('free: WMA (read by libav.js) shows its real length and is cleaned; AVI (AC-3 sound) gives the sound back as WAV with a note why', async () => {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'noisy.wma'));
  await page.locator('.file-chip .meta').filter({ hasText: /0:1[01]/ }).waitFor({ timeout: 30000 });
  if (await page.getByTestId('length-unknown').count()) throw new Error('length shown as a guess');
  const w = await runFree('noisy.wma');
  if (!close(w.frames) || w.rms < 0.001) throw new Error(`wma ${JSON.stringify(w)}`);
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'noisy.avi'));
  await page.getByTestId('audio-only-note').waitFor({ timeout: 30000 });
  if (await page.getByRole('checkbox', { name: /Keep the video/ }).count()) throw new Error('keep-video offered');
  const a = await runFree('noisy.avi');
  if (!close(a.frames) || a.rms < 0.001) throw new Error(`avi ${JSON.stringify(a)}`);
  if (await page.getByTestId('dl-video').count()) throw new Error('video download offered for an AVI');
  return `wma frames=${w.frames}; avi frames=${a.frames}; decoder files: ${[...new Set(libavFetches)].join(', ')}`;
});

await check('free: an MP4 is cleaned and the video comes back with the cleaned sound', async () => {
  const r = await runFree('test.mp4');
  if (r.rms < 0.001) throw new Error('silent');
  const btn = page.getByTestId('dl-video');
  await btn.waitFor();
  await page.waitForFunction(() => !document.querySelector('[data-testid=dl-video]').disabled, null, { timeout: 5000 });
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 60000 }), btn.click()]);
  const p = join(FILES, 'out-free-video' + (dl.suggestedFilename().match(/\.[a-z0-9]+$/)?.[0] || '.mp4'));
  await dl.saveAs(p);
  return `${dl.suggestedFilename()} ${statSync(p).size} bytes`;
});

/** A fresh browser profile (empty Cache Storage), signed in; `site` lists what it fetched from the site's /dfn/ folder. */
async function freshProfile(init) {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  if (init) await ctx.addInitScript(init);
  const site = [];
  ctx.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.includes('/dfn/')) site.push(u.pathname.split('/').pop()); });
  const p = await ctx.newPage();
  await signIn(p, EMAIL);
  models.reset();
  return { ctx, p, site };
}
const startFree = async (p, file) => {
  await p.goto(`${BASE}/app`);
  await p.setInputFiles('input[type=file]', join(FILES, file));
  await p.getByTestId('start').waitFor({ timeout: 20000 });
  await settleRecommendation(p);
  await pick(p, 'noise', 'browser');
  await p.getByRole('button', { name: 'Start – free' }).click();
};
const keptFiles = (p) => p.evaluate(async () => (await (await caches.open('acw-models')).keys()).map((r) => r.url.split('/').pop()));

await check('free: the noise remover is downloaded after a file is chosen (df.js from the site; the wasm and the model once each from the model host, never from /dfn/); a stereo file and a second page load use the kept copy', async () => {
  const { ctx, p, site } = await freshProfile();
  try {
    await p.goto(`${BASE}/app`); await p.waitForSelector('main h1');
    await p.waitForTimeout(1500);
    if (site.length || Object.keys(models.hits).length) throw new Error('downloaded before a file was chosen: ' + site.join(', ') + Object.keys(models.hits));
    await startFree(p, 'stereo44.wav');
    await p.waitForSelector('[data-testid=result]', { timeout: 120000 });
    if (JSON.stringify(models.byName()) !== JSON.stringify({ [WASM]: 1, [MODEL]: 1 })) throw new Error('model host downloads: ' + JSON.stringify(models.byName()));
    if (!site.includes('df.js') || site.includes(WASM) || site.includes(MODEL)) throw new Error('from the site: ' + site.join(', '));
    if ((await p.locator('[data-testid=result]').getAttribute('data-channels')) !== '2') throw new Error('expected a stereo run');
    await startFree(p, 'noisy.wav'); // a new page load: the browser's own copy, no second download
    await p.waitForSelector('[data-testid=result]', { timeout: 120000 });
    if (JSON.stringify(models.byName()) !== JSON.stringify({ [WASM]: 1, [MODEL]: 1 })) throw new Error('second page load downloaded again: ' + JSON.stringify(models.byName()));
    const kept = (await keptFiles(p)).sort();
    if (kept.join() !== [WASM, MODEL].sort().join()) throw new Error('kept in the browser: ' + kept);
    return 'wasm x1, model x1 from the model host; stereo and a second page load from Cache Storage';
  } finally {
    await ctx.close();
  }
});

await check('free: a model file that is not the pinned one (changed bytes) is refused with an error and not kept; the real file then works', async () => {
  const { ctx, p } = await freshProfile();
  try {
    models.tamper(MODEL);
    await startFree(p, 'noisy.wav');
    const err = await p.getByTestId('job-error').waitFor({ timeout: 120000 }).then(() => p.getByTestId('job-error').innerText());
    if (!/not the expected file/.test(err)) throw new Error('error text: ' + err);
    if (await p.getByTestId('result').count()) throw new Error('a result was made from the changed file');
    if ((await keptFiles(p)).includes(MODEL)) throw new Error('the changed file was kept');
    models.tamper(null);
    await p.getByRole('button', { name: 'Start – free' }).click();
    await p.waitForSelector('[data-testid=result]', { timeout: 120000 });
    if (!(await keptFiles(p)).includes(MODEL)) throw new Error('the real file was not kept');
    return err.split('\n')[0];
  } finally {
    models.tamper(null);
    await ctx.close();
  }
});

await check('free: when the browser asks to save data, nothing is downloaded before Start', async () => {
  const { ctx, p, site } = await freshProfile(() => Object.defineProperty(navigator, 'connection', { value: { saveData: true } }));
  try {
    await p.goto(`${BASE}/app`);
    await p.setInputFiles('input[type=file]', join(FILES, 'noisy.wav'));
    await p.getByTestId('start').waitFor({ timeout: 20000 });
    await pick(p, 'noise', 'browser');
    await p.waitForTimeout(2000);
    if (site.length || Object.keys(models.hits).length) throw new Error(`downloaded before Start: ${site.join(', ')} ${Object.keys(models.hits)}`);
    await p.getByRole('button', { name: 'Start – free' }).click();
    await p.waitForSelector('[data-testid=result]', { timeout: 120000 });
    if (models.byName()[WASM] !== 1) throw new Error('Start did not download the noise remover');
    return 'nothing before Start, then the noise remover and a result';
  } finally {
    await ctx.close();
  }
});

await check('free + free: browser noise removal AND free subtitles at the same time, no job is sent at all', async () => {
  const posts = [];
  await page.route('**/api/jobs', (r) => { if (r.request().method() === 'POST') posts.push(r.request().postDataJSON()); return r.continue(); });
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'noisy.wav'));
  await pick(page, 'subs', 'browser');
  const label = (await page.getByTestId('start').innerText()).trim();
  if (label !== 'Start – free') throw new Error('label: ' + label);
  await page.getByTestId('start').click();
  await page.waitForSelector('[data-testid=result]', { timeout: 180000 });
  await waitResult(page, 'subs-result');
  await page.unroute('**/api/jobs');
  if (posts.length) throw new Error('a job was sent: ' + JSON.stringify(posts.at(-1)));
  const [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: /Cleaned WAV/ }).click()]);
  await page.screenshot({ path: join(SHOTS, 'free-free-combo.png'), fullPage: true });
  return `download ${dl.suggestedFilename()} + subtitles (${await page.getByTestId('subs-result').getAttribute('data-cues')} cues)`;
});

console.log('LOGS:\n' + logs.join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
