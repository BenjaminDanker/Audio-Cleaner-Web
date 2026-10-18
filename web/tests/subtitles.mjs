// Free subtitles made in the browser (Cleanup page, "Free, in your browser"): the right words, the SRT/VTT/TXT downloads, a
// recording longer than one 30 s piece, a video that gets them inside (MKV), nothing about the file is uploaded, and our side
// never accepts a free subtitles job. The speech model comes from the model host (Hugging Face; here the local copy of
// mirror.mjs), each pinned file once, checked and kept by the browser; a changed file is refused.
// Needs the local backend (port 8792, see helpers.mjs) with `ACW_MODEL_HOST=http://127.0.0.1:8798/ npm run build` output and
// the fixtures (make-fixtures.sh).
// Usage: node tests/subtitles.mjs
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { BASE, FILES, SHOTS, check, dropFile, launch, mirror, pick, signIn, summary, waitResult } from './helpers.mjs';

const FFMPEG = process.env.FFMPEG || join(homedir(), 'audio-cleaner', 'bin', 'ffmpeg');
const FFPROBE = process.env.FFPROBE || join(homedir(), 'audio-cleaner', 'bin', 'ffprobe');
const EMAIL = `e2e-subs-${Date.now()}@example.com`;
const SAID = /responsible|hearing|damage|exposure/i; // what speech.wav says

const { browser, context, page, logs } = await launch([], { acceptDownloads: true });
const models = await mirror(); // the local model host
const { SPEECH } = await import('../src/media/models.js'); // the pinned files (mirror.mjs has set the build-time host)
const SPEECH_FILES = Object.keys(SPEECH.files).map((f) => f.split('/').pop());
const calls = [];
context.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.includes('/api/') || u.pathname.includes('/dfn/models/')) calls.push(`${r.method()} ${u.pathname}`); });
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
  await waitResult(page, 'subs-result');
  return { res, ms: Date.now() - t0 };
}

async function download(locator, name) {
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 120000 }), locator.click()]);
  const p = join(FILES, name);
  await dl.saveAs(p);
  return { path: p, name: dl.suggestedFilename() };
}

await check('subtitles, free: a short recording gives the right words; SRT, VTT and TXT download; nothing is uploaded; the speech model comes from the model host, each file once', async () => {
  calls.length = 0;
  models.reset();
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
  if (calls.some((c) => c.includes('/dfn/models/'))) throw new Error('the speech model was fetched from the site: ' + calls.join(', '));
  const got = models.byName(); // every pinned file, one download each
  if (SPEECH_FILES.some((f) => got[f] !== 1)) throw new Error('model host downloads: ' + JSON.stringify(got));
  await page.screenshot({ path: join(SHOTS, 'subs-free.png'), fullPage: true });
  return `${ms} ms; "${preview.slice(0, 80)}"; no upload; ${SPEECH_FILES.length} model files from the model host, once each`;
});

await check('subtitles, free: a 48 s recording (longer than one 30 s piece) has every repeat, in order, timed across the whole file', async () => {
  const before = JSON.stringify(models.byName());
  const { res, ms } = await run('speech-long.wav'); // a new page load: the model comes from the browser's kept copy
  if (JSON.stringify(models.byName()) !== before) throw new Error('second page load downloaded the model again: ' + JSON.stringify(models.byName()));
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

await check('subtitles, free: a model file that is not the pinned one (changed bytes) is refused with an error and no subtitles are made', async () => {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } }); // empty Cache Storage
  const p = await ctx.newPage();
  try {
    await signIn(p, EMAIL);
    models.reset();
    models.tamper('preprocessor_config.json');
    await p.goto(`${BASE}/app`);
    await dropFile(p, join(FILES, 'speech.wav'));
    await pick(p, 'noise', 'off');
    await pick(p, 'subs', 'browser');
    await p.getByRole('button', { name: 'Start – free' }).click();
    await p.getByTestId('job-error').waitFor({ timeout: 120000 });
    const err = await p.getByTestId('job-error').innerText();
    if (!/subtitle maker couldn.t start/.test(err)) throw new Error('error text: ' + err); // the worker's own reason is not passed on
    if (await p.getByTestId('subs-result').count()) throw new Error('subtitles were made with a changed file');
    if (!models.byName()['preprocessor_config.json']) throw new Error('the changed file was never requested, so the refusal is not the one under test');
    return err.split('\n')[0];
  } finally {
    models.tamper(null);
    await ctx.close();
  }
});

await check('subtitles, free, video: "Put the subtitles into the video" gives an MKV with the picture and sound copied and a subtitle track any player reads', async () => {
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

await check('subtitles, our side: a free subtitles job is refused (free subtitles are only made in the browser)', async () => {
  const r = await page.evaluate(async (base) => {
    const me = await (await fetch(`${base}/api/me`)).json();
    const res = await fetch(`${base}/api/jobs`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': me.csrfToken },
      body: JSON.stringify({ kind: 'transcribe', transcribeTier: 'local', free: true, filename: 'a.wav', sizeBytes: 1000, durationSeconds: 10 }),
    });
    return { status: res.status, body: await res.json() };
  }, BASE);
  if (r.status !== 422 || JSON.stringify(r.body.fields) !== '["free"]') throw new Error(JSON.stringify(r));
  return r.body.detail;
});

const errors = logs.filter((l) => !/Failed to load resource|favicon/i.test(l));
if (errors.length) console.log('console:', errors.slice(0, 10).join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
