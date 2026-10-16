// Free subtitles made in the browser (Cleanup page, "Free, in your browser"): the model comes from our server once,
// the subtitles are right, the files download, a long file spans several pieces, a video gets them inside (MKV),
// nothing is uploaded, and our side never makes free subtitles any more.
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { BASE, FILES, SHOTS, admin, check, dropFile, launch, pick, signIn, summary } from './helpers.mjs';

const FFMPEG = process.env.FFMPEG || join(homedir(), 'audio-cleaner', 'bin', 'ffmpeg');
const FFPROBE = process.env.FFPROBE || join(homedir(), 'audio-cleaner', 'bin', 'ffprobe');
const EMAIL = `e2e-subs-${Date.now()}@example.com`;
const SAID = /responsible|hearing|damage|exposure/i; // what speech.wav says

const { browser, context, page, logs } = await launch([], { acceptDownloads: true });
const calls = [];
context.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.includes('/audio/api/')) calls.push(`${r.method()} ${u.pathname}`); });
await signIn(page, EMAIL);

async function run(file, { noise = 'off', embed = false } = {}) {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, file));
  await pick(page, 'noise', noise);
  await pick(page, 'subs', 'browser');
  if (embed) await page.getByRole('checkbox', { name: /Put the subtitles into the video/ }).check();
  const t0 = Date.now();
  await page.getByRole('button', { name: 'Start – free' }).click();
  const res = page.getByTestId('subs-result');
  await res.waitFor({ timeout: 300000 });
  return { res, ms: Date.now() - t0 };
}

async function download(locator, name) {
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 120000 }), locator.click()]);
  const p = join(FILES, name);
  await dl.saveAs(p);
  return { path: p, name: dl.suggestedFilename() };
}

await check('subtitles: the choices are Off | Free, in your browser | Standard | Extra accurate; the free one says it is made on the device', async () => {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'speech.wav'));
  const titles = (await page.locator('label.choice:has(input[name=subs]) b').allInnerTexts()).map((x) => x.trim());
  if (JSON.stringify(titles) !== JSON.stringify(['Off', 'Free, in your browser', 'Standard', 'Extra accurate'])) throw new Error(JSON.stringify(titles));
  const hint = await page.locator('label.choice:has(input[name=subs][value=browser]) small').innerText();
  if (!/nothing is uploaded/i.test(hint) || !/80 MB/.test(hint)) throw new Error(hint);
  if (/free minutes|left today/i.test(await page.locator('main').innerText())) throw new Error('the old free minutes are still mentioned');
  return hint;
});

await check('subtitles: a short recording, made in the browser: right words, English found by itself, SRT/VTT/TXT download; nothing uploaded', async () => {
  calls.length = 0;
  const { res, ms } = await run('speech.wav');
  const preview = await res.getByTestId('subs-preview').innerText();
  if (!SAID.test(preview)) throw new Error(`preview: ${preview}`);
  if ((await res.getAttribute('data-language')) !== 'en') throw new Error(`language ${await res.getAttribute('data-language')}`);
  const srt = readFileSync((await download(res.getByTestId('subs-dl-srt'), 'subs.srt')).path, 'utf8');
  if (!/^1\n00:00:\d\d,\d{3} --> 00:00:\d\d,\d{3}\n/.test(srt) || !SAID.test(srt)) throw new Error(`srt: ${srt.slice(0, 200)}`);
  const vtt = readFileSync((await download(res.getByTestId('subs-dl-vtt'), 'subs.vtt')).path, 'utf8');
  if (!vtt.startsWith('WEBVTT\n\n00:00:')) throw new Error(`vtt: ${vtt.slice(0, 80)}`);
  const txt = await download(res.getByTestId('subs-dl-txt'), 'subs.txt');
  if (txt.name !== 'speech-transcript.txt' || !SAID.test(readFileSync(txt.path, 'utf8'))) throw new Error(txt.name);
  const posted = calls.filter((c) => c.startsWith('POST') || c.includes('/jobs/'));
  if (posted.length) throw new Error(`sent to the server: ${posted.join(', ')}`);
  const model = calls.filter((c) => c.includes('/downloads/browser/'));
  if (model.length < 7) throw new Error(`model files fetched: ${model.join(', ')}`);
  await page.screenshot({ path: join(SHOTS, 'subs-free.png'), fullPage: true });
  return `${ms} ms; "${preview.slice(0, 80)}"; ${model.length} model files from our server`;
});

await check('subtitles: after a reload the model comes from the browser\'s cache, not our server', async () => {
  calls.length = 0;
  const { res } = await run('speech.wav');
  if (!SAID.test(await res.getByTestId('subs-preview').innerText())) throw new Error('wrong words');
  const model = calls.filter((c) => c.includes('/downloads/browser/'));
  if (model.length) throw new Error(`fetched again: ${model.join(', ')}`);
  return 'cached';
});

await check('subtitles: a 48 s recording (longer than one 30 s piece): every repeat is there, in order, timed across the whole file', async () => {
  const { res, ms } = await run('speech-long.wav');
  const srt = readFileSync((await download(res.getByTestId('subs-dl-srt'), 'subs-long.srt')).path, 'utf8');
  const times = [...srt.matchAll(/(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})/g)].map((m) => [
    +m[1] * 3600 + +m[2] * 60 + +m[3] + +m[4] / 1000, +m[5] * 3600 + +m[6] * 60 + +m[7] + +m[8] / 1000]);
  for (let i = 1; i < times.length; i++) if (times[i][0] < times[i - 1][1] - 0.001) throw new Error(`cues overlap at ${i}`);
  const last = times.at(-1)[1];
  if (last < 40 || last > 49) throw new Error(`last cue ends at ${last}`);
  const n = (srt.match(/responsible/gi) || []).length;
  if (n < 3) throw new Error(`"responsible" ${n} times: ${srt}`);
  return `${times.length} cues, last ends at ${last.toFixed(1)} s, "responsible" ${n}/4 times, ${ms} ms`;
});

await check('subtitles: a video with "Put the subtitles into the video": an MKV with the picture and sound copied and an SRT track', async () => {
  const { res } = await run('speech.mp4', { embed: true });
  const out = await download(res.getByTestId('subs-video'), 'subs-video.mkv');
  if (out.name !== 'speech-subtitled.mkv') throw new Error(out.name);
  const j = JSON.parse(execFileSync(FFPROBE, ['-v', 'error', '-show_streams', '-of', 'json', out.path], { encoding: 'utf8' }));
  const kinds = j.streams.map((s) => `${s.codec_type}:${s.codec_name}`).sort();
  if (JSON.stringify(kinds) !== JSON.stringify(['audio:aac', 'subtitle:subrip', 'video:h264'])) throw new Error(kinds.join(', '));
  const text = execFileSync(FFMPEG, ['-v', 'error', '-i', out.path, '-map', '0:s:0', '-f', 'srt', '-'], { encoding: 'utf8' });
  if (!SAID.test(text)) throw new Error(text);
  return kinds.join(', ');
});

await check('subtitles: free noise removal and free subtitles together: the video gets the cleaned sound and the subtitles', async () => {
  const { res } = await run('speech.mp4', { noise: 'browser', embed: true });
  await page.getByTestId('result').waitFor({ timeout: 180000 });
  const btn = res.getByTestId('subs-video');
  if (!/cleaned sound/.test(await btn.innerText())) throw new Error(await btn.innerText());
  const out = await download(btn, 'subs-video-clean.mkv');
  if (out.name !== 'speech-cleaned.mkv') throw new Error(out.name);
  const j = JSON.parse(execFileSync(FFPROBE, ['-v', 'error', '-show_streams', '-of', 'json', out.path], { encoding: 'utf8' }));
  if (!j.streams.some((s) => s.codec_type === 'subtitle')) throw new Error('no subtitles');
  return j.streams.map((s) => `${s.codec_type}:${s.codec_name}`).join(', ');
});

await check('subtitles: Clean & polish with free subtitles: the job is only the cleanup (kind denoise, nothing free asked of our side)', async () => {
  admin('grant', EMAIL, '1', 'subtitles test'); // the cleanup is paid
  const posts = [];
  await page.route('**/api/jobs', (r) => {
    if (r.request().method() !== 'POST') return r.continue();
    posts.push(r.request().postDataJSON());
    return r.fulfill({ status: 400, json: { detail: 'Stopped by the test.' } });
  });
  try {
    await page.goto(`${BASE}/app`);
    await dropFile(page, join(FILES, 'speech.wav'));
    await pick(page, 'noise', 'polish');
    await pick(page, 'subs', 'browser');
    await page.getByTestId('polish-local-subs-note').waitFor();
    await page.getByTestId('start').click();
    await page.getByTestId('job-error').waitFor({ timeout: 60000 });
  } finally {
    await page.unroute('**/api/jobs');
  }
  const b = posts.at(-1);
  if (!b || b.kind !== 'denoise' || b.transcribeTier || 'free' in b) throw new Error(JSON.stringify(b));
  return JSON.stringify({ kind: b.kind, denoiseTier: b.denoiseTier });
});

await check('subtitles: our side refuses a free job (free subtitles are only made in the browser now)', async () => {
  const r = await page.evaluate(async (base) => {
    const me = await (await fetch(`${base}/api/me`)).json();
    const res = await fetch(`${base}/api/jobs`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': me.csrfToken },
      body: JSON.stringify({ kind: 'transcribe', transcribeTier: 'local', free: true, filename: 'a.wav', sizeBytes: 1000, durationSeconds: 10 }),
    });
    return { status: res.status, body: await res.json() };
  }, BASE);
  if (r.status !== 400 || !/browser/.test(r.body.detail)) throw new Error(JSON.stringify(r));
  return r.body.detail;
});

await check('subtitles: under the production Content-Security-Policy the subtitle maker loads and works (no blob scripts)', async () => {
  const csp = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; connect-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'";
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const viol = [];
  ctx.on('weberror', (e) => viol.push(e.error().message));
  await ctx.route('**/*', async (route) => {
    if (!route.request().url().startsWith('http://127.0.0.1:8792')) return route.continue();
    const res = await route.fetch();
    await route.fulfill({ response: res, headers: { ...res.headers(), 'content-security-policy': csp } });
  });
  const p2 = await ctx.newPage();
  p2.on('console', (m) => { if (/Content Security Policy|refused to/i.test(m.text())) viol.push(m.text().slice(0, 200)); });
  try {
    await signIn(p2, EMAIL);
    await p2.goto(`${BASE}/app`);
    await dropFile(p2, join(FILES, 'speech.wav'));
    await pick(p2, 'noise', 'off');
    await pick(p2, 'subs', 'browser');
    await p2.getByRole('button', { name: 'Start – free' }).click();
    await Promise.race([
      p2.getByTestId('subs-result').waitFor({ timeout: 300000 }),
      p2.getByTestId('job-error').waitFor({ timeout: 300000 }).then(async () => { throw new Error(await p2.getByTestId('job-error').innerText()); }),
    ]);
    if (!SAID.test(await p2.getByTestId('subs-preview').innerText())) throw new Error('wrong words');
  } finally {
    await ctx.close();
  }
  if (viol.length) throw new Error(viol.join(' | '));
  return 'no violations';
});

const errors = logs.filter((l) => !/Failed to load resource|favicon/i.test(l));
if (errors.length) console.log('console:', errors.slice(0, 10).join('\n'));
await browser.close();
const res = summary();
process.exit(res.every((r) => r.ok) ? 0 : 1);
