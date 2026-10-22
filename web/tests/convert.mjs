// The output picker (src/media/convert.js, src/components/OutputPicker.jsx): the file type and the name of a result, made in the
// browser. Audio -> WAV / FLAC / MP3 / Ogg (and M4A where this browser can encode AAC), video -> sound only, a sound file -> a video
// with a black picture, a video kept as it is (MKV) or made smaller (MP4 at 480p), the name and the type list following each
// other, the name a paid job is saved under; every result comes down by itself, and Save changes brings the new type. Another file type or
// video size is a feature of its own: with nothing else chosen it is free, runs in the browser (a card named File conversion) and gives the file
// in the new type or size with its own sound; a video type this browser can't make for a video is offered as made on our side.
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

/** Drop a file with nothing else chosen (no noise removal, no Leveling, no subtitles): the file type and size are the only things left to choose. */
async function prepareConversion(file) {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, file));
  await pick(page, 'noise', 'off');
  await pick(page, 'subs', 'off');
  await openOutput();
}
const BIG = join(FILES, 'big4k.avi'); // 10 s of a plain 4K picture at 30 frames a second (MPEG-4 in an AVI, read through libav) with the sound of noisy.wav: a few MB
if (!existsSync(BIG)) execFileSync(FFMPEG, ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=0x336699:size=3840x2160:rate=30:duration=10', '-i', join(FILES, 'noisy.wav'), '-t', '10', '-map', '0:v', '-map', '1:a',
  '-c:v', 'mpeg4', '-b:v', '1M', '-g', '300', '-threads', '4', '-c:a', 'ac3', '-b:a', '192k', BIG]);
const NOTHING_TO_DO = 'Choose at least one thing to do: noise removal, subtitles, or another file type or video size.';
const startLabel = () => page.locator('.studio .bb .btn-lg').innerText().then((t) => t.trim());
const startReady = () => page.waitForFunction(() => !document.querySelector('[data-testid=notes] .banner-warn') && document.querySelector('[data-testid=start]')?.disabled === false, null, { timeout: 10000 });

await check('file type alone: with the file\'s own type and size Start says what to choose; another type makes Start free, the file comes down in that type with its own picture and sound, and the card is named File conversion', async () => {
  await prepareConversion('test.mp4');
  await page.locator('[data-testid=notes] .banner-warn', { hasText: NOTHING_TO_DO }).waitFor({ timeout: 5000 });
  if (await page.getByTestId('start').isEnabled()) throw new Error('Start is on with nothing chosen');
  await field('out-format').selectOption('mkv');
  await field('out-name').fill('only the type.mkv');
  await closeOutput();
  await startReady();
  if ((await startLabel()) !== 'Start · Free') throw new Error('button: ' + (await startLabel()));
  const hint = await page.getByTestId('will-download').innerText().catch(() => '');
  downloads.length = 0;
  const calls = [];
  const onReq = (rq) => { const u = new URL(rq.url()); if (u.pathname.startsWith('/api/') && rq.method() !== 'GET') calls.push(`${rq.method()} ${u.pathname}`); };
  page.on('request', onReq);
  await page.getByTestId('start').click();
  await waitResult(page, 'result', 120000);
  page.off('request', onReq);
  if (calls.length) throw new Error('sent to our side: ' + calls.join(', '));
  if (downloads.length !== 1 || downloads[0].suggestedFilename() !== 'only the type.mkv') throw new Error('downloads: ' + downloads.map((d) => d.suggestedFilename()));
  const p = join(FILES, 'conv-only-type.mkv');
  await downloads[0].saveAs(p);
  const src = probe(join(FILES, 'test.mp4'));
  const got = probe(p);
  if (got.v.codec_name !== src.v.codec_name || `${got.v.width}x${got.v.height}` !== `${src.v.width}x${src.v.height}`) throw new Error(`picture ${src.v.codec_name} -> ${got.v?.codec_name}`);
  if (got.a.codec_name !== src.a.codec_name || Math.abs(got.dur - src.dur) > 0.8) throw new Error(`sound ${src.a.codec_name} ${src.dur} -> ${got.a?.codec_name} ${got.dur}`);
  // the card: Your files lists the one file; the Cleanup page's list names the job File conversion
  await page.getByTestId('bar-files').click();
  const files = await page.getByTestId('made-file').allInnerTexts();
  await page.getByTestId('bar-files').click();
  if (files.length !== 1 || !/only the type\.mkv/.test(files[0])) throw new Error('listed: ' + files);
  await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Cleanup', exact: true }).click();
  await page.waitForURL(/\/app$/);
  const row = (await page.locator('[data-testid=job-row]').first().innerText()).replace(/\s+/g, ' ');
  if (!/File conversion/.test(row) || !/test\.mp4/.test(row)) throw new Error('card: ' + row);
  return `${got.v.codec_name}+${got.a.codec_name} in MKV; ${hint.replace(/\s+/g, ' ').slice(0, 70)}; card: ${row.slice(0, 70)}`;
});

await check('video size alone: a 720p video at 480p is made again in the browser with the shape kept, for free, with nothing else chosen; a sound file in another type is the same', async () => {
  const out = [];
  if (existsSync(big) && (enc.avc || enc.vp9)) {
    await prepareConversion('conv720.mp4');
    await field('out-size').selectOption('480');
    await closeOutput();
    await startReady();
    if ((await startLabel()) !== 'Start · Free') throw new Error('button: ' + (await startLabel()));
    downloads.length = 0;
    await page.getByTestId('start').click();
    await waitResult(page, 'result', 180000);
    if (downloads.length !== 1 || !/\.mp4$/.test(downloads[0].suggestedFilename())) throw new Error('downloads: ' + downloads.map((d) => d.suggestedFilename()));
    const p = join(FILES, 'conv-only-size.mp4');
    await downloads[0].saveAs(p);
    const i = probe(p);
    if (i.v.height !== 480 || Math.abs(i.v.width / i.v.height - 16 / 9) > 0.02 || !i.a) throw new Error(`${i.v.width}x${i.v.height} sound ${i.a?.codec_name}`);
    out.push(`${i.v.codec_name} ${i.v.width}x${i.v.height}`);
  } else out.push('size skipped (no 720p fixture, or this Chromium can write neither H.264 nor VP9)');
  await prepareConversion('noisy.wav');
  await page.locator('[data-testid=notes] .banner-warn', { hasText: NOTHING_TO_DO }).waitFor({ timeout: 5000 });
  await field('out-format').selectOption('flac');
  await field('out-name').fill('only the type.flac');
  await closeOutput();
  await startReady();
  downloads.length = 0;
  await page.getByTestId('start').click();
  await waitResult(page, 'result', 120000);
  if (downloads.length !== 1 || downloads[0].suggestedFilename() !== 'only the type.flac') throw new Error('downloads: ' + downloads.map((d) => d.suggestedFilename()));
  const f = join(FILES, 'conv-only-type.flac');
  await downloads[0].saveAs(f);
  const s = probe(f);
  if (s.a.codec_name !== 'flac' || Math.abs(s.dur - 10.6) > 0.8) throw new Error(`flac ${s.a.codec_name} ${s.dur}`);
  out.push('flac');
  return out.join('; ');
});

await check('made on our side: a video this browser can only read through libav (an AVI) is offered every video type under "Video, made on our side" and every size says so in the note; a video the browser opens itself is not offered its own type that way', async () => {
  await prepareConversion('noisy.avi');
  const groups = () => field('out-format').evaluate((sel) => [...sel.querySelectorAll('optgroup')].map((g) => ({ label: g.label, values: [...g.querySelectorAll('option')].map((o) => o.value) })));
  let g = await groups();
  const ours = g.find((x) => x.label === 'Video, made on our side');
  if (!ours || ours.values.join() !== 'mp4,mov,mkv,webm') throw new Error('groups: ' + JSON.stringify(g));
  if (g.some((x) => x.label === 'Video')) throw new Error('a video type offered as made here: ' + JSON.stringify(g));
  await field('out-format').selectOption('mp4');
  const sizes = await field('out-size').evaluate((sel) => [...sel.querySelectorAll('option')].map((o) => o.value));
  if (sizes.join() !== 'same,2160,1440,1080,720,480') throw new Error('sizes: ' + sizes);
  // (the note says it is made on our side and uploaded whole, for every size; what it says about the price is the price list's business)
  const note = (await field('out-size-note').innerText()).replace(/\s+/g, ' ');
  if (!/^Made on our side, so the whole video is uploaded\./.test(note)) throw new Error('note: ' + note);
  await field('out-size').selectOption('720');
  const sized = (await field('out-size-note').innerText()).replace(/\s+/g, ' ');
  if (!/^Made on our side, so the whole video is uploaded\./.test(sized)) throw new Error('size note: ' + sized);
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'test.mp4'));
  await openOutput();
  g = await groups();
  const here = g.find((x) => x.label === 'Video');
  const mine = (g.find((x) => x.label === 'Video, made on our side') || { values: [] }).values;
  if (!here || !here.values.includes('mp4') || !here.values.includes('mkv') || mine.includes('mp4') || mine.includes('mkv')) throw new Error('groups for an MP4: ' + JSON.stringify(g));
  return `AVI: ${ours.values.join(', ')}; MP4: here ${here.values.join(', ')}${mine.length ? `, ours ${mine.join(', ')}` : ''}`;
});

/** The Details popup: its lines (the total apart) and the total. */
async function detailsOf() {
  await page.getByTestId('bar-det').click();
  const lines = (await page.locator('[data-testid=details] ul.fl li:not(.tot)').allInnerTexts()).map((l) => l.replace(/\s+/g, ' ').trim());
  const total = (await page.getByTestId('details-total').innerText()).trim();
  await page.getByTestId('bar-det').click();
  return { lines, total };
}
const COUNTS = (n) => new RegExp(`^Video conversion · counts ${n}× \\(bigger or faster than 1080p at 30 fps\\) `);

await check('a video made on our side is priced up front from what the browser read of it: the estimate asked for carries the video\'s codec, size and frame rate, Details says how much the picture counts, and a copy into another type is free and says so', async () => {
  const sent = [];
  const onReq = (rq) => { const u = new URL(rq.url()); if (u.pathname === '/api/jobs/estimate') sent.push(rq.postDataJSON()); };
  page.on('request', onReq);
  try {
    await prepareConversion('big4k.avi');
    await field('out-format').selectOption('webm'); // the MPEG-4 picture does not go into WebM: made again; 4K at 30 frames a second is 4 units
    await closeOutput();
    await page.waitForFunction(() => /^Start · \$0\.004$/.test(document.querySelector('.studio .bb .btn-lg')?.textContent.trim() || ''), null, { timeout: 15000 });
    let d = await detailsOf();
    if (d.lines.length !== 1 || !COUNTS(4).test(d.lines[0]) || !/ \$0\.004$/.test(d.lines[0]) || d.total !== '$0.004') throw new Error('webm: ' + JSON.stringify(d));
    const body = sent.at(-1);
    if (body?.kind !== 'convert' || body.options.video_out.format !== 'webm' || body.options.video_out.size !== null) throw new Error('estimate asked: ' + JSON.stringify(body));
    const source = body.options.video_out.source;
    if (source?.codec !== 'mpeg4' || source.width !== 3840 || source.height !== 2160 || Math.abs(source.fps - 30) > 0.5) throw new Error('source: ' + JSON.stringify(source));
    await openOutput();
    await field('out-size').selectOption('720');
    await closeOutput();
    await page.waitForFunction(() => /^Start · \$0\.002$/.test(document.querySelector('.studio .bb .btn-lg')?.textContent.trim() || ''), null, { timeout: 15000 });
    d = await detailsOf();
    if (d.lines.length !== 1 || !COUNTS(2).test(d.lines[0]) || d.total !== '$0.002') throw new Error('webm 720p: ' + JSON.stringify(d));
    if (sent.at(-1).options.video_out.size !== 720) throw new Error('estimate for 720p: ' + JSON.stringify(sent.at(-1)));
    await openOutput();
    await field('out-format').selectOption('mkv');
    await field('out-size').selectOption('same');
    await closeOutput();
    await page.waitForFunction(() => /^Start · Free$/.test(document.querySelector('.studio .bb .btn-lg')?.textContent.trim() || ''), null, { timeout: 15000 });
    d = await detailsOf();
    if (d.lines.join('|') !== 'Video on our side, the picture copied as it is Free' || d.total !== 'Free') throw new Error('mkv: ' + JSON.stringify(d));
    const copied = sent.at(-1);
    return `webm 4× ($0.004), 720p 2× ($0.002), MKV copied; source ${JSON.stringify(source)}; last estimate format ${copied.options.video_out.format}`;
  } finally {
    page.off('request', onReq);
  }
});

await check('a picture copied on our side, end to end: an AVI into MKV is a free job (kind convert), the video comes down as made there with its own picture and sound, and costs nothing', async () => {
  await prepareConversion('noisy.avi');
  await field('out-format').selectOption('mkv');
  await closeOutput();
  await page.waitForFunction(() => /^Start · Free$/.test(document.querySelector('.studio .bb .btn-lg')?.textContent.trim() || ''), null, { timeout: 15000 });
  downloads.length = 0;
  const posted = [];
  const onReq = (rq) => { const u = new URL(rq.url()); if (rq.method() === 'POST' && u.pathname === '/api/jobs') posted.push(rq.postDataJSON()); };
  page.on('request', onReq);
  await page.getByTestId('start').click();
  await page.getByTestId('submitted').waitFor({ timeout: 60000 });
  await waitCardDone(page);
  await waitResult(page, 'result', 90000);
  page.off('request', onReq);
  const job = posted.at(-1);
  if (job?.kind !== 'convert' || job.options.video_out.format !== 'mkv' || job.options.video_out.source?.codec !== 'mpeg4' || job.denoiseTier) throw new Error('job asked for: ' + JSON.stringify(job));
  if (downloads.length !== 1 || downloads[0].suggestedFilename() !== 'noisy-converted.mkv') throw new Error('downloaded by itself: ' + downloads.map((d) => d.suggestedFilename()));
  const p = join(FILES, 'conv-server-copy.mkv');
  await downloads[0].saveAs(p);
  const src = probe(join(FILES, 'noisy.avi'));
  const got = probe(p);
  if (got.v.codec_name !== src.v.codec_name || `${got.v.width}x${got.v.height}` !== `${src.v.width}x${src.v.height}` || got.a.codec_name !== src.a.codec_name || Math.abs(got.dur - src.dur) > 0.8) {
    throw new Error(`picture ${src.v.codec_name} ${src.v.width}x${src.v.height} + ${src.a.codec_name} -> ${got.v?.codec_name} ${got.v?.width}x${got.v?.height} + ${got.a?.codec_name} ${got.dur}`);
  }
  await page.getByTestId('bar-files').click();
  const paid = (await page.locator('[data-testid=files] .note b').first().innerText()).trim();
  await page.getByTestId('bar-files').click();
  if (paid !== 'Free') throw new Error('cost: ' + paid);
  return `${got.v.codec_name}+${got.a.codec_name} copied into MKV, cost ${paid}`;
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
