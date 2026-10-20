// The output picker (src/media/convert.js, src/components/OutputPicker.jsx): the file type and the name of a result, made in the
// browser. Audio -> WAV / FLAC / MP3 / Ogg (and M4A where this browser can encode AAC), video -> sound only, a sound file -> a video
// with a black picture, a video kept as it is (MKV) or made smaller (MP4 at 480p), the name and the type list following each
// other, the name a paid job is saved under; every result comes down by itself, and Save changes brings the new type.
// The choices live in the editor's Output menu (OutputFields). The files that come out are looked at with ffprobe. An encoder the
// headless Chromium lacks is reported as a skip.
// Needs the local backend (port 8792), `npm run build` output and the fixtures. Usage: node tests/convert.mjs
import { execFileSync } from 'node:child_process';
import { existsSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { BASE, FILES, SHOTS, admin, check, dropFile, launch, pick, signIn, summary, waitCardDone, waitResult } from './helpers.mjs';

const EMAIL = process.env.E2E_EMAIL || `e2e-convert-${Date.now()}@example.com`; // a fresh account: its job list starts empty
const FFMPEG = process.env.FFMPEG || join(homedir(), 'audio-cleaner', 'bin', 'ffmpeg');
const FFPROBE = process.env.FFPROBE || join(homedir(), 'audio-cleaner', 'bin', 'ffprobe');

const { page, logs, browser } = await launch();
await signIn(page, EMAIL);

const probe = (p) => {
  const j = JSON.parse(execFileSync(FFPROBE, ['-v', 'error', '-show_streams', '-show_format', '-of', 'json', p], { encoding: 'utf8' }));
  const v = j.streams.find((s) => s.codec_type === 'video');
  const a = j.streams.find((s) => s.codec_type === 'audio');
  return { v, a, dur: Number(j.format.duration), size: statSync(p).size };
};

// WebCodecs needs a secure context: 127.0.0.1 counts as one.
const enc = await page.evaluate(async () => {
  const ok = async (fn) => { try { return !!(await fn()).supported; } catch { return false; } };
  return {
    avc: await ok(() => VideoEncoder.isConfigSupported({ codec: 'avc1.42001f', width: 640, height: 360, bitrate: 1e6 })),
    aac: await ok(() => AudioEncoder.isConfigSupported({ codec: 'mp4a.40.2', numberOfChannels: 2, sampleRate: 48000, bitrate: 128000 })),
    vp9: await ok(() => VideoEncoder.isConfigSupported({ codec: 'vp09.00.10.08', width: 640, height: 360, bitrate: 1e6 })),
  };
}).catch(() => ({}));

const big = join(FILES, 'conv720.mp4');
if (!existsSync(big)) {
  try {
    execFileSync(FFMPEG, ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=1280x720:rate=25', '-i', join(FILES, 'noisy.wav'), '-t', '4', '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', big]);
  } catch (e) {
    console.log('could not make the 720p fixture:', e.message.split('\n')[0]);
  }
}

// Before Start the file type and name are chosen in the Output menu; after the result is made they can change there too
// ("Save changes"), and saving brings down the files with the new choice.
const downloads = [];
page.on('download', (d) => downloads.push(d));
const field = (id) => page.locator(`[data-testid=output-pop] [data-testid=${id}]`);
const formField = field;
const optionValues = (f) => f('out-format').locator('option').evaluateAll((os) => os.map((o) => o.value));
const openOutput = async () => {
  if (!(await page.getByTestId('output-pop').count())) await page.getByTestId('output-button').click();
  await field('out-format').waitFor();
  await page.waitForFunction(() => { const s = document.querySelector('[data-testid=output-pop] [data-testid=out-format]'); return s && !s.disabled; }, null, { timeout: 15000 });
};
const closeOutput = async () => { if (await page.getByTestId('output-pop').count()) await page.getByTestId('output-button').click(); };
/** What Your files says about the last file made (its note, e.g. "copied as it is"). */
const madeNote = async () => {
  await page.getByTestId('bar-files').click();
  const t = (await page.getByTestId('made-note').first().textContent().catch(() => '')) || '';
  await page.getByTestId('bar-files').click();
  return t;
};

/** Drop a file on the Cleanup page (the free cleanup is the starting choice): the editor is ready, nothing started. */
async function prepare(file) {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, file));
  await openOutput();
}

/** Start the free cleanup and wait for the editor to hand the files over (they come down by themselves). */
async function startFree() {
  downloads.length = 0;
  await closeOutput();
  await page.getByTestId('start').click();
  await waitResult(page, 'result', 180000);
}
const runFree = async (file) => { await prepare(file); await startFree(); };

/** On the finished page: set the name (and so the type), Save changes, and return the downloaded file's path and the note shown. */
async function saveAs(name, { format, size } = {}) {
  await openOutput();
  if (format) await field('out-format').selectOption(format);
  if (name != null) await field('out-name').fill(name);
  if (size) await field('out-size').selectOption(size);
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 120000 }), page.getByTestId('save-changes').click()]);
  const p = join(FILES, `conv-${dl.suggestedFilename()}`);
  await dl.saveAs(p);
  await waitResult(page, 'result', 120000);
  await page.getByTestId('download-again').waitFor({ timeout: 120000 });
  return { p, name: dl.suggestedFilename(), note: await madeNote() };
}

await check('audio: the name starts as <name>-cleaned.<type> in the Output menu, the type list is only what this browser can make, and the name and the list follow each other both ways', async () => {
  await prepare('noisy.wav');
  if ((await formField('out-name').inputValue()) !== 'noisy-cleaned.wav') throw new Error('default name');
  const vals = await optionValues(formField);
  for (const need of ['wav', 'flac', 'mp3', 'ogg']) if (!vals.includes(need)) throw new Error(`${need} missing from ${vals}`);
  if (vals.includes('m4a') !== enc.aac) throw new Error(`m4a listed=${vals.includes('m4a')} but AAC encoder=${enc.aac}`);
  await formField('out-name').fill('my talk.mp3');
  if ((await formField('out-format').inputValue()) !== 'mp3') throw new Error('typing .mp3 did not switch the list');
  await formField('out-format').selectOption('flac');
  if ((await formField('out-name').inputValue()) !== 'my talk.flac') throw new Error('list did not change the ending');
  await formField('out-name').fill('my talk.opus');
  if ((await formField('out-format').inputValue()) !== 'ogg') throw new Error('typing .opus did not switch the list to Ogg');
  await formField('out-format').selectOption('wav');
  if ((await formField('out-name').inputValue()) !== 'my talk.wav') throw new Error('wav');
  return `list ${vals.join(', ')}`;
});

await check('a name that cannot be used (unknown ending, no ending, empty, a forbidden character) turns Start off with a short reason', async () => {
  for (const bad of ['song.xyz', 'song', '', 'a/b.wav', 'what?.wav']) {
    await formField('out-name').fill(bad);
    const hint = (await formField('out-hint').innerText()).trim();
    if (!(await page.getByTestId('start').isDisabled()) || (await formField('out-name').getAttribute('aria-invalid')) !== 'true' || !hint || hint.length > 140) throw new Error(`"${bad}": hint="${hint}"`);
  }
  await formField('out-name').fill('good.wav');
  if (await page.getByTestId('start').isDisabled()) throw new Error('a good name is refused');
  return 'refused, then accepted';
});

await check('free cleanup of a sound file: the chosen file comes down by itself as the only file; Your files lists it with its size, the bar has one Download again and no variant buttons', async () => {
  await formField('out-name').fill('chosen name.flac');
  await startFree();
  const names = downloads.map((d) => d.suggestedFilename());
  if (names.join() !== 'chosen name.flac') throw new Error('downloaded by itself: ' + names);
  await page.getByTestId('bar-files').click();
  const files = await page.getByTestId('made-file').allInnerTexts();
  await page.getByTestId('bar-files').click();
  if (files.length !== 1 || !/chosen name\.flac/.test(files[0]) || !/\d(\.\d)? (KB|MB)/.test(files[0])) throw new Error('listed: ' + files);
  if ((await page.getByTestId('download-again').count()) !== 1 || (await page.getByRole('button', { name: /24-bit|16-bit|Video with/ }).count())) throw new Error('buttons');
  return files[0].replace(/\s+/g, ' ');
});

await check('audio to each type: WAV, FLAC, MP3 (its encoder is fetched only now), Ogg (Opus), M4A where AAC can be encoded; each file has the right codec and length', async () => {
  const fetched = [];
  const onReq = (rq) => { const u = new URL(rq.url()); if (/mp3-encoder/.test(u.pathname)) fetched.push(u.pathname.split('/').pop()); };
  page.on('request', onReq);
  const out = [];
  for (const [name, format, codec] of [['chosen name.wav', null, 'pcm_s16le'], [null, 'flac', 'flac'], ['speech.mp3', null, 'mp3'], ['speech.ogg', null, 'opus']]) {
    const r = await saveAs(name, format ? { format } : {});
    const i = probe(r.p);
    if (i.a.codec_name !== codec) throw new Error(`${r.name}: ${i.a.codec_name}`);
    if (Math.abs(i.dur - 10.6) > 0.6) throw new Error(`${r.name}: duration ${i.dur}`);
    out.push(`${r.name} ${codec}`);
  }
  page.off('request', onReq);
  if (!fetched.length) throw new Error('MP3 encoder was not fetched');
  if (enc.aac) {
    const r = await saveAs('speech.m4a');
    if (probe(r.p).a.codec_name !== 'aac') throw new Error('m4a is not AAC');
    out.push(`${r.name} aac`);
  } else if (await openOutput().then(() => optionValues(field)).then((v) => v.includes('m4a'))) {
    throw new Error('m4a offered without an AAC encoder');
  }
  return out.join(', ');
});

await check('sound to video: a WebM with a plain black picture and the cleaned sound', async () => {
  const w = await saveAs('radio.webm', { format: 'webm' });
  const i = probe(w.p);
  if (i.v?.codec_name !== 'vp9' && i.v?.codec_name !== 'vp8') throw new Error(`picture ${i.v?.codec_name}`);
  if (!i.a) throw new Error('no sound');
  if (Math.abs(i.dur - 10.6) > 0.8) throw new Error(`duration ${i.dur}`);
  return `${w.name} ${i.v.codec_name} ${i.v.width}x${i.v.height}, ${i.a.codec_name}, ${i.dur.toFixed(1)} s`;
});

await check('video kept as it is: MKV copies the picture unchanged (H.264 stays H.264) with the cleaned sound; the video to sound only (MP3, FLAC) has no picture', async () => {
  await prepare('test.mp4');
  if ((await formField('out-name').inputValue()) !== 'test-cleaned.mp4') throw new Error('default name ' + (await formField('out-name').inputValue()));
  await startFree();
  if (downloads.length !== 1 || downloads[0].suggestedFilename() !== 'test-cleaned.mp4') throw new Error('the video should come down by itself as the one file: ' + downloads.map((d) => d.suggestedFilename()));
  const before = probe(join(FILES, 'test.mp4'));
  const r = await saveAs(null, { format: 'mkv' });
  const i = probe(r.p);
  if (r.name !== 'test-cleaned.mkv' || i.v.codec_name !== before.v.codec_name || `${i.v.width}x${i.v.height}` !== `${before.v.width}x${before.v.height}`) throw new Error(`picture ${before.v.codec_name} -> ${i.v?.codec_name}`);
  if (!/copied as it is/.test(r.note)) throw new Error(`note: ${r.note}`);
  const m = await saveAs('podcast.mp3');
  if (probe(m.p).v) throw new Error('a picture is in the MP3');
  return `${r.name} ${i.v.codec_name} ${i.v.width}x${i.v.height} + ${i.a.codec_name}; note: ${r.note}`;
});

await check('video made smaller: a 1280x720 video at 480p is encoded again with the shape kept (H.264 in an MP4)', async () => {
  if (!existsSync(big)) return 'SKIP (no 720p fixture: ffmpeg missing)';
  if (!enc.avc && !enc.vp9) return 'SKIP (this Chromium can write neither H.264 nor VP9)';
  await runFree('conv720.mp4');
  const r = await saveAs(null, { format: 'mp4', size: '480' });
  const i = probe(r.p);
  if (i.v.height !== 480) throw new Error(`height ${i.v.height}`);
  if (Math.abs(i.v.width / i.v.height - 16 / 9) > 0.02) throw new Error(`shape ${i.v.width}x${i.v.height}`);
  if (!/encoded again/.test(r.note)) throw new Error(`note: ${r.note}`);
  return `${r.name} ${i.v.codec_name} ${i.v.width}x${i.v.height}, ${i.a.codec_name}; ${r.note}`;
});

await check('paid job: a Denoising job for a sound file with a chosen name and type comes down by itself under that name (and again with Download again); changing the type and saving brings the new type', async () => {
  admin('grant', EMAIL, '5', 'convert test');
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'noisy.wav'));
  await pick(page, 'noise', 'denoise');
  await openOutput();
  await formField('out-name').fill('episode one.mp3');
  const hint = await page.getByTestId('will-download').innerText();
  if (!/episode one\.mp3/.test(hint)) throw new Error('the line before Start: ' + hint);
  await closeOutput();
  downloads.length = 0;
  await page.getByTestId('start').click();
  await page.getByTestId('submitted').waitFor({ timeout: 60000 });
  await waitCardDone(page);
  await waitResult(page, 'result', 90000);
  if (downloads.length !== 1 || downloads[0].suggestedFilename() !== 'episode one.mp3') throw new Error('downloaded by itself: ' + downloads.map((d) => d.suggestedFilename()));
  const p = join(FILES, 'conv-paid.mp3');
  await downloads[0].saveAs(p);
  if (probe(p).a.codec_name !== 'mp3') throw new Error('not MP3');
  const [again] = await Promise.all([page.waitForEvent('download'), page.getByTestId('download-again').click()]);
  if (again.suggestedFilename() !== 'episode one.mp3') throw new Error('again: ' + again.suggestedFilename());
  // change the type and Save changes: the new type comes down (the server's sound is converted here)
  await openOutput();
  await field('out-format').selectOption('flac');
  const [flac] = await Promise.all([page.waitForEvent('download', { timeout: 60000 }), page.getByTestId('save-changes').click()]);
  if (flac.suggestedFilename() !== 'episode one.flac') throw new Error('after changing the type: ' + flac.suggestedFilename());
  await flac.saveAs(join(FILES, 'conv-paid-again.flac'));
  if (probe(join(FILES, 'conv-paid-again.flac')).a.codec_name !== 'flac') throw new Error('not FLAC');
  return `${downloads[0].suggestedFilename()}, again, then ${flac.suggestedFilename()}`;
});

await check('paid job on a video: the video with the cleaned sound comes down by itself; after a reload the card asks for the original video once, then makes it and downloads it again by itself', async () => {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'test.mp4'));
  await pick(page, 'noise', 'denoise');
  await openOutput();
  await formField('out-format').selectOption('mkv');
  await closeOutput();
  downloads.length = 0;
  await page.getByTestId('start').click();
  await page.getByTestId('submitted').waitFor({ timeout: 60000 });
  await waitCardDone(page);
  await waitResult(page, 'result', 90000);
  if (downloads.length !== 1 || downloads[0].suggestedFilename() !== 'test-cleaned.mkv') throw new Error('downloaded by itself: ' + downloads.map((d) => d.suggestedFilename()));
  const p = join(FILES, 'conv-paid-video.mkv');
  await downloads[0].saveAs(p);
  const i = probe(p);
  if (!i.v || !i.a) throw new Error('picture or sound missing');
  await page.screenshot({ path: join(SHOTS, 'convert-paid-video.png'), fullPage: true });
  // after a reload this tab has no copy of the video: it is asked for once
  await page.reload();
  const again = page.getByTestId('download-again');
  await again.waitFor({ timeout: 30000 });
  if (!/Choose the original video/.test(await again.innerText())) throw new Error('should ask for the video: ' + await again.innerText());
  downloads.length = 0;
  await page.getByTestId('need-video-input').setInputFiles(join(FILES, 'test.mp4'));
  await waitResult(page, 'result', 90000);
  if (downloads.length !== 1 || downloads[0].suggestedFilename() !== 'test-cleaned.mkv') throw new Error('after choosing the video: ' + downloads.map((d) => d.suggestedFilename()));
  if (!/Download again/.test(await again.innerText())) throw new Error('then it is a plain Download again');
  return `${downloads[0].suggestedFilename()} ${i.v.codec_name}+${i.a.codec_name}; asked for the video once after the reload`;
});

const bad = logs.filter((l) => !/Failed to load resource|favicon|net::ERR/.test(l));
await check('no console errors on the way', async () => {
  if (bad.length) throw new Error(bad.slice(0, 3).join(' || '));
  return `${logs.length} warnings ignored`;
});

await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
