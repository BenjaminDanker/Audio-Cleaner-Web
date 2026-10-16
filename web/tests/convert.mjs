import { execFileSync } from 'node:child_process';
import { existsSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { BASE, FILES, admin, check, dropFile, launch, pick, signIn, summary } from './helpers.mjs';

// Choosing the name, the file type and the video size of a result, all in the browser (src/media/convert.js,
// src/components/OutputPicker.jsx): audio -> mp3 / flac / wav / opus, video -> mkv / webm, a re-encoded smaller picture,
// video -> sound only, sound -> video (plain black picture), the name box and the list following each other, a bad name.
// The files that come out are looked at with ffprobe. An encoder the headless Chromium lacks is reported as a skip.
const EMAIL = process.env.E2E_EMAIL || `e2e-convert-${Date.now()}@example.com`; // a fresh account: its job list starts empty
const FFMPEG = process.env.FFMPEG || join(homedir(), 'audio-cleaner', 'bin', 'ffmpeg');
const FFPROBE = process.env.FFPROBE || join(homedir(), 'audio-cleaner', 'bin', 'ffprobe');

const { page, logs, browser } = await launch();
await signIn(page, EMAIL);

const probe = (p) => {
  const j = JSON.parse(execFileSync(FFPROBE, ['-v', 'error', '-show_streams', '-show_format', '-of', 'json', p], { encoding: 'utf8' }));
  const v = j.streams.find((s) => s.codec_type === 'video');
  const a = j.streams.find((s) => s.codec_type === 'audio');
  return { v, a, dur: Number(j.format.duration), size: statSync(p).size, fmt: j.format.format_name };
};

const enc = await page.evaluate(async () => {
  const ok = async (fn) => { try { return !!(await fn()).supported; } catch { return false; } };
  return {
    avc: await ok(() => VideoEncoder.isConfigSupported({ codec: 'avc1.42001f', width: 640, height: 360, bitrate: 1e6 })),
    vp9: await ok(() => VideoEncoder.isConfigSupported({ codec: 'vp09.00.10.08', width: 640, height: 360, bitrate: 1e6 })),
    aac: await ok(() => AudioEncoder.isConfigSupported({ codec: 'mp4a.40.2', numberOfChannels: 2, sampleRate: 48000, bitrate: 128000 })),
    opus: await ok(() => AudioEncoder.isConfigSupported({ codec: 'opus', numberOfChannels: 2, sampleRate: 48000, bitrate: 128000 })),
  };
}).catch(() => ({}));
// WebCodecs needs a secure context: 127.0.0.1 counts as one.

const big = join(FILES, 'conv720.mp4');
if (!existsSync(big)) {
  try {
    execFileSync(FFMPEG, ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=1280x720:rate=25', '-i', join(FILES, 'noisy.wav'), '-t', '4', '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', big]);
  } catch (e) {
    console.log('could not make the 720p fixture:', e.message.split('\n')[0]);
  }
}


async function waitJob(name, timeout = 240000) {
  const card = page.locator('[data-testid=job]', { hasText: name }).first();
  await card.waitFor({ timeout: 30000 });
  await page.waitForFunction((n) => {
    const c = [...document.querySelectorAll('[data-testid=job]')].find((e) => e.textContent.includes(n));
    return c && ['done', 'failed', 'canceled'].includes(c.dataset.status);
  }, name, { timeout });
  return card;
}

const SAVE = '[data-testid=save-as]';
const field = (id) => page.locator(`${SAVE} [data-testid=${id}]`);
const optionValues = () => field('out-format').locator('option').evaluateAll((os) => os.map((o) => o.value));

/** Clean a file in the browser (free) and wait for the result card. */
async function runFree(file) {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, file));
  await page.getByRole('button', { name: 'Start – free' }).click();
  await page.waitForSelector('[data-testid=result]', { timeout: 180000 });
  await page.waitForFunction(() => { const s = document.querySelector('[data-testid=save-as] [data-testid=out-format]'); return s && !s.disabled; }, null, { timeout: 15000 });
}

/** Set the name (and so the type), save, and return the downloaded file's path and the note shown. */
async function saveAs(name, { format, size, out } = {}) {
  if (format) await field('out-format').selectOption(format);
  if (name != null) await field('out-name').fill(name);
  if (size) await field('out-size').selectOption(size);
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 120000 }), field('save-btn').click()]);
  const p = join(FILES, out || `conv-${dl.suggestedFilename()}`);
  await dl.saveAs(p);
  await page.waitForFunction(() => !document.querySelector('[data-testid=save-as] [data-testid=save-btn]').disabled, null, { timeout: 120000 });
  return { p, name: dl.suggestedFilename(), note: (await field('save-note').textContent().catch(() => '')) || '' };
}

await check('audio: the name starts as <name>-cleaned.<ext> with the file type list beside it, only types this browser can make', async () => {
  await runFree('noisy.wav');
  const name = await field('out-name').inputValue();
  if (name !== 'noisy-cleaned.wav') throw new Error(`name ${name}`);
  if ((await field('out-format').inputValue()) !== 'wav') throw new Error('list not on WAV');
  const vals = await optionValues();
  for (const need of ['wav', 'flac', 'mp3', 'ogg']) if (!vals.includes(need)) throw new Error(`${need} missing from ${vals}`);
  if (vals.includes('m4a') !== enc.aac) throw new Error(`m4a listed=${vals.includes('m4a')} but AAC encoder=${enc.aac}`);
  const tip = await field('out-hint').innerText();
  if (!/type another ending/i.test(tip)) throw new Error(`tip: ${tip}`);
  return `list ${vals.join(', ')}`;
});

await check('name and list follow each other (typing .mp3 -> list MP3; list FLAC -> .flac; .opus -> Ogg; kept stem)', async () => {
  await field('out-name').fill('my talk.mp3');
  if ((await field('out-format').inputValue()) !== 'mp3') throw new Error('typing .mp3 did not switch the list');
  await field('out-format').selectOption('flac');
  if ((await field('out-name').inputValue()) !== 'my talk.flac') throw new Error(`name ${await field('out-name').inputValue()}`);
  await field('out-name').fill('my talk.opus');
  if ((await field('out-format').inputValue()) !== 'ogg') throw new Error('typing .opus did not switch the list to Ogg');
  await field('out-format').selectOption('ogg');
  if ((await field('out-name').inputValue()) !== 'my talk.opus') throw new Error('picking Ogg replaced a .opus ending that already fits');
  await field('out-format').selectOption('wav');
  if ((await field('out-name').inputValue()) !== 'my talk.wav') throw new Error('wav');
  return 'in sync both ways';
});

await check('a name that cannot be used shows a short hint and Save stays off (unknown ending, no ending, empty, forbidden character)', async () => {
  const seen = [];
  for (const bad of ['song.xyz', 'song', '', '.mp3', 'a/b.wav', 'what?.wav']) {
    await field('out-name').fill(bad);
    const hint = (await field('out-hint').innerText()).trim();
    const off = await field('save-btn').isDisabled();
    const invalid = (await field('out-name').getAttribute('aria-invalid')) === 'true';
    if (!off || !invalid || !hint || /^Tip/.test(hint)) throw new Error(`"${bad}": disabled=${off} invalid=${invalid} hint="${hint}"`);
    if (hint.length > 140) throw new Error(`hint too long for "${bad}"`);
    seen.push(`"${bad}" -> ${hint}`);
  }
  await field('out-name').fill('good.wav');
  if (await field('save-btn').isDisabled()) throw new Error('a good name is refused');
  return seen[0];
});

await check('audio -> WAV with the typed name', async () => {
  const r = await saveAs('chosen name.wav');
  if (r.name !== 'chosen name.wav') throw new Error(r.name);
  const i = probe(r.p);
  if (i.a.codec_name !== 'pcm_s16le') throw new Error(i.a.codec_name);
  if (Math.abs(i.dur - 10.6) > 0.6) throw new Error(`duration ${i.dur}`);
  return `${r.name} ${i.size} bytes ${i.dur.toFixed(2)} s`;
});

await check('audio -> FLAC', async () => {
  const r = await saveAs(null, { format: 'flac' });
  if (r.name !== 'chosen name.flac') throw new Error(r.name);
  const i = probe(r.p);
  if (i.a.codec_name !== 'flac') throw new Error(i.a.codec_name);
  if (Math.abs(i.dur - 10.6) > 0.6) throw new Error(`duration ${i.dur}`);
  return `${r.name} ${i.size} bytes`;
});

await check('audio -> MP3 (the encoder file is fetched only now)', async () => {
  const fetched = [];
  const onReq = (rq) => { const u = new URL(rq.url()); if (/mp3-encoder/.test(u.pathname)) fetched.push(u.pathname.split('/').pop()); };
  page.on('request', onReq);
  const r = await saveAs('speech.mp3');
  page.off('request', onReq);
  const i = probe(r.p);
  if (i.a.codec_name !== 'mp3') throw new Error(i.a.codec_name);
  if (Math.abs(i.dur - 10.6) > 0.6) throw new Error(`duration ${i.dur}`);
  if (!fetched.length) throw new Error('encoder file was not fetched');
  return `${r.name} ${i.size} bytes ${i.a.sample_rate} Hz; fetched ${fetched[0]}`;
});

await check('audio -> Ogg (Opus)', async () => {
  const r = await saveAs('speech.ogg');
  const i = probe(r.p);
  if (i.a.codec_name !== 'opus') throw new Error(i.a.codec_name);
  if (Math.abs(i.dur - 10.6) > 0.6) throw new Error(`duration ${i.dur}`);
  return `${r.name} ${i.size} bytes`;
});

await check('audio -> M4A (AAC), or not offered when this browser has no AAC encoder', async () => {
  if (!enc.aac) {
    if ((await optionValues()).includes('m4a')) throw new Error('m4a offered without an encoder');
    const note = await field('out-missing').innerText();
    if (!/M4A/.test(note)) throw new Error(`no note: ${note}`);
    return `SKIP (no AAC encoder in this Chromium); list note: ${note}`;
  }
  const r = await saveAs('speech.m4a');
  const i = probe(r.p);
  if (i.a.codec_name !== 'aac') throw new Error(i.a.codec_name);
  return `${r.name} ${i.size} bytes`;
});

await check('sound -> video: a plain black picture and the cleaned sound (WebM; MP4 when H.264 can be written)', async () => {
  const vals = await optionValues();
  if (!vals.includes('webm')) throw new Error(`webm not offered: ${vals}`);
  const w = await saveAs('radio.webm');
  const i = probe(w.p);
  if (i.v?.codec_name !== 'vp9' && i.v?.codec_name !== 'vp8') throw new Error(`picture ${i.v?.codec_name}`);
  if (!i.a) throw new Error('no sound');
  if (Math.abs(i.dur - 10.6) > 0.8) throw new Error(`duration ${i.dur}`);
  const out = [`${w.name} ${i.v.codec_name} ${i.v.width}x${i.v.height}, ${i.a.codec_name}, ${i.dur.toFixed(1)} s, ${i.size} bytes`];
  if (vals.includes('mp4')) {
    const r = await saveAs('radio.mp4');
    const j = probe(r.p);
    if (!j.v) throw new Error('mp4 has no picture');
    out.push(`${r.name} ${j.v.codec_name}, ${j.a.codec_name}`);
  } else if (enc.avc) {
    throw new Error('H.264 can be written but MP4 is not offered');
  }
  return out.join(' | ');
});

await check('video -> MKV: the picture is copied as it is (H.264 stays H.264), the sound is the cleaned one', async () => {
  await runFree('test.mp4');
  const name = await field('out-name').inputValue();
  if (name !== 'test-cleaned.mp4' || (await field('out-format').inputValue()) !== 'mp4') throw new Error(`default ${name}`);
  const before = probe(join(FILES, 'test.mp4'));
  const r = await saveAs(null, { format: 'mkv' });
  if (r.name !== 'test-cleaned.mkv') throw new Error(r.name);
  const i = probe(r.p);
  if (i.v.codec_name !== before.v.codec_name) throw new Error(`picture ${before.v.codec_name} -> ${i.v.codec_name}`);
  if (`${i.v.width}x${i.v.height}` !== `${before.v.width}x${before.v.height}`) throw new Error('size changed');
  if (!/copied as it is/.test(r.note)) throw new Error(`note: ${r.note}`);
  if (Math.abs(i.dur - before.dur) > 0.8) throw new Error(`duration ${i.dur} vs ${before.dur}`);
  return `${r.name} ${i.v.codec_name} ${i.v.width}x${i.v.height} + ${i.a.codec_name}, ${i.dur.toFixed(1)} s; ${r.note}`;
});

await check('video -> WebM: H.264 cannot go in WebM, so the picture is encoded again (VP9)', async () => {
  if (!enc.vp9) return 'SKIP (no VP9 encoder in this Chromium)';
  const vals = await optionValues();
  if (!vals.includes('webm')) throw new Error(`webm not offered: ${vals}`);
  const r = await saveAs(null, { format: 'webm' });
  const i = probe(r.p);
  if (i.v.codec_name !== 'vp9') throw new Error(`picture ${i.v.codec_name}`);
  if (i.a.codec_name !== 'opus') throw new Error(`sound ${i.a.codec_name}`);
  if (!/encoded again/.test(r.note)) throw new Error(`note: ${r.note}`);
  return `${r.name} ${i.v.codec_name} ${i.v.width}x${i.v.height} + ${i.a.codec_name}; ${r.note}`;
});

await check('video -> sound only (MP3 and FLAC): no picture in the result', async () => {
  const m = await saveAs('podcast.mp3');
  const i = probe(m.p);
  if (i.v) throw new Error('a picture is in the MP3');
  if (i.a.codec_name !== 'mp3') throw new Error(i.a.codec_name);
  const f = await saveAs('podcast.flac');
  const j = probe(f.p);
  if (j.v || j.a.codec_name !== 'flac') throw new Error(`flac: ${j.a?.codec_name}`);
  return `${m.name} ${i.size} bytes; ${f.name} ${j.size} bytes`;
});

await check('video size: bigger than the original says so; the sizes are listed, plain resizing keeps the shape', async () => {
  await field('out-format').selectOption('mp4');
  const sizes = await field('out-size').locator('option').evaluateAll((os) => os.map((o) => o.value));
  if (sizes[0] !== 'same') throw new Error(`first size ${sizes[0]}`);
  const want = ['same', '2160', '1440', '1080', '720', '480'];
  const miss = want.filter((s) => !sizes.includes(s));
  if (miss.length && (enc.avc || enc.vp9)) throw new Error(`sizes missing: ${miss}`);
  if (!sizes.includes('1080')) return `SKIP (no video encoder; sizes ${sizes})`;
  await field('out-size').selectOption('1080');
  const note = await field('out-size-note').innerText();
  if (!/stretched/.test(note)) throw new Error(`no upscale note: ${note}`);
  await field('out-size').selectOption('same');
  return `sizes ${sizes.join(', ')}; upscale note: ${note}`;
});

await check('video re-encode with a size change (1280x720 -> 480p, shape kept, H.264 in an MP4)', async () => {
  if (!existsSync(big)) return 'SKIP (no 720p fixture: ffmpeg missing)';
  if (!enc.avc && !enc.vp9) return 'SKIP (this Chromium can write neither H.264 nor VP9)';
  await runFree('conv720.mp4');
  const r = await saveAs(null, { format: 'mp4', size: '480' });
  const i = probe(r.p);
  if (i.v.height !== 480) throw new Error(`height ${i.v.height}`);
  if (Math.abs(i.v.width / i.v.height - 16 / 9) > 0.02) throw new Error(`shape ${i.v.width}x${i.v.height}`);
  if (!/encoded again/.test(r.note)) throw new Error(`note: ${r.note}`);
  return `${r.name} ${i.v.codec_name} ${i.v.width}x${i.v.height}, ${i.a.codec_name}, ${i.dur.toFixed(1)} s; ${r.note}`;
});

await check('a Matroska file at 720p from the same video', async () => {
  if (!existsSync(big) || (!enc.avc && !enc.vp9)) return 'SKIP (no 720p fixture or encoder)';
  const r = await saveAs('talk.mkv', { size: '720' });
  const i = probe(r.p);
  if (i.v.height !== 720) throw new Error(`height ${i.v.height}`);
  return `${r.name} ${i.v.codec_name} ${i.v.width}x${i.v.height}`;
});

await check('cancel: Cancel stops the work, nothing is saved, the form is usable again', async () => {
  if (!existsSync(big) || (!enc.avc && !enc.vp9)) return 'SKIP (no 720p fixture or encoder)';
  let downloads = 0;
  const onDl = () => { downloads += 1; };
  page.on('download', onDl);
  await field('out-name').fill('stop.mkv');
  await field('out-size').selectOption('1080');
  await field('save-btn').click();
  let clicked = true;
  await page.locator(`${SAVE} [data-testid=save-cancel]`).click({ timeout: 5000 }).catch(() => { clicked = false; });
  await page.waitForFunction(() => !document.querySelector('[data-testid=save-as] [data-testid=save-btn]').disabled, null, { timeout: 60000 });
  await page.waitForTimeout(500);
  page.off('download', onDl);
  const err = await page.locator('[data-testid=result] [role=alert]').count();
  if (err) throw new Error(`error shown: ${await page.locator('[data-testid=result] [role=alert]').first().innerText()}`);
  if (clicked && downloads) throw new Error('a file was saved after Cancel');
  if (!clicked && !downloads) throw new Error('neither a Cancel button nor a finished file');
  return clicked ? 'canceled, nothing saved' : 'finished before the click (fast machine); no error';
});

await check('Clean & polish: the same name and type list is on the form, any sound type is listed (our side writes it), the video types depend on this browser', async () => {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'noisy.wav'));
  await pick(page, 'noise', 'polish');
  const f = page.locator('[data-testid=output-fields]').first();
  await f.getByTestId('out-name').waitFor();
  if ((await f.getByTestId('out-name').inputValue()) !== 'noisy-cleaned.wav') throw new Error('default name');
  await page.waitForFunction(() => { const s = document.querySelector('[data-testid=out-format]'); return s && !s.disabled; });
  const vals = await f.getByTestId('out-format').locator('option').evaluateAll((os) => os.map((o) => o.value));
  for (const need of ['wav', 'flac', 'mp3', 'ogg', 'm4a']) if (!vals.includes(need)) throw new Error(`${need} missing: ${vals}`);
  await f.getByTestId('out-name').fill('episode.mp3');
  if ((await f.getByTestId('out-format').inputValue()) !== 'mp3') throw new Error('not in sync');
  await f.getByTestId('out-format').selectOption('flac');
  if ((await f.getByTestId('out-name').inputValue()) !== 'episode.flac') throw new Error('list did not change the ending');
  return `list ${vals.join(', ')}`;
});

await check('Clean & polish of a video: it starts as a video type (the picture is kept); the file type alone decides, no extra box', async () => {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'test.mp4'));
  await pick(page, 'noise', 'polish');
  const f = page.locator('[data-testid=output-fields]').first();
  await page.waitForFunction(() => { const s = document.querySelector('[data-testid=out-format]'); return s && !s.disabled; });
  if (await page.getByRole('checkbox', { name: /Keep the video/ }).count()) throw new Error('the Keep-the-video box is back');
  const kind = await f.getByTestId('out-format').inputValue();
  if (!['mp4', 'mkv', 'webm', 'mov'].includes(kind)) throw new Error(`starts as ${kind}`);
  await pick(page, 'subs', 'standard');
  await page.getByRole('checkbox', { name: /Put the subtitles into the video/ }).waitFor();
  await f.getByTestId('out-format').selectOption('mp3');
  if (await page.getByRole('checkbox', { name: /Put the subtitles into the video/ }).count()) throw new Error('embedding offered for a sound file');
  return `starts as ${kind}`;
});


await check('Clean & polish of a sound file with a chosen name and type: the job file is saved under that name', async () => {
  admin('grant', EMAIL, '5', 'convert test');
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'noisy.wav'));
  await pick(page, 'noise', 'polish');
  const f = page.locator('[data-testid=output-fields]').first();
  await f.getByTestId('out-name').fill('episode one.mp3');
  await page.getByTestId('start').click();
  await page.getByTestId('submitted').waitFor({ timeout: 60000 });
  const card = await waitJob('noisy');
  const link = card.getByTestId('download').first();
  const [dl] = await Promise.all([page.waitForEvent('download'), link.click()]);
  const p = join(FILES, 'conv-paid.mp3');
  await dl.saveAs(p);
  const i = probe(p);
  if (dl.suggestedFilename() !== 'episode one.mp3') throw new Error(dl.suggestedFilename());
  if (i.a.codec_name !== 'mp3') throw new Error(`codec ${i.a.codec_name}`);
  return `${dl.suggestedFilename()} ${i.size} bytes ${i.a.codec_name}`;
});

await check('Clean & polish of a video with a chosen type and size: the cleaned sound is put back in the browser as asked', async () => {
  if (!enc.avc && !enc.vp9) return 'SKIP (no video encoder in this Chromium)';
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'test.mp4'));
  await pick(page, 'noise', 'polish');
  const f = page.locator('[data-testid=output-fields]').first();
  await page.waitForFunction(() => { const s = document.querySelector('[data-testid=out-size]'); return s && !s.disabled; }, null, { timeout: 15000 });
  await f.getByTestId('out-format').selectOption('mkv');
  await f.getByTestId('out-size').selectOption('480');
  await f.getByTestId('out-name').fill('my clip.mkv');
  await page.getByTestId('start').click();
  await page.getByTestId('submitted').waitFor({ timeout: 60000 });
  const card = await waitJob('test');
  const btn = card.getByTestId('remux-btn');
  await btn.waitFor({ timeout: 240000 });
  const txt = await btn.innerText();
  if (!/my clip\.mkv/.test(txt)) throw new Error(`button says: ${txt}`);
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 120000 }), btn.click()]);
  const p = join(FILES, 'conv-paid-video.mkv');
  await dl.saveAs(p);
  const i = probe(p);
  if (dl.suggestedFilename() !== 'my clip.mkv') throw new Error(dl.suggestedFilename());
  if (i.v.height !== 480) throw new Error(`height ${i.v.height}`);
  return `${dl.suggestedFilename()} ${i.v.codec_name} ${i.v.width}x${i.v.height} + ${i.a.codec_name}`;
});

await check('Free subtitles put into the video in the browser: an MKV with the picture and sound copied and the subtitles as a track players read', async () => {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'test.mp4'));
  await pick(page, 'noise', 'off');
  await pick(page, 'subs', 'free');
  await page.getByRole('checkbox', { name: /Put the subtitles into the video/ }).check();
  let sent = null;
  const spy = (r) => { if (r.method() === 'POST' && /\/api\/jobs$/.test(r.url())) sent = r.postDataJSON(); };
  page.on('request', spy);
  await page.getByTestId('start').click();
  await page.getByTestId('submitted').waitFor({ timeout: 60000 });
  page.off('request', spy);
  if (!sent || sent.options.embed_subtitles !== false || sent.filename.endsWith('.mp4')) throw new Error(`the video went up for the subtitles: ${JSON.stringify(sent)}`);
  // whatever the speech model heard in the test clip, the video gets these cues
  const vtt = 'WEBVTT\n\n00:00:00.500 --> 00:00:02.000\nHello there\n\n00:00:02.500 --> 00:00:03.500\nSecond line\n';
  await page.route('**/api/jobs/*/outputs/*.vtt', (r) => r.fulfill({ status: 200, contentType: 'text/vtt', body: vtt }));
  const card = await waitJob('test');
  const btn = card.getByTestId('remux-btn');
  await btn.waitFor({ timeout: 240000 });
  if (!/subtitles/.test(await btn.innerText())) throw new Error(`button says: ${await btn.innerText()}`);
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 120000 }), btn.click()]);
  const p = join(FILES, 'conv-subtitled.mkv');
  await dl.saveAs(p);
  await page.unroute('**/api/jobs/*/outputs/*.vtt');
  const j = JSON.parse(execFileSync(FFPROBE, ['-v', 'error', '-show_streams', '-of', 'json', p], { encoding: 'utf8' }));
  const s = j.streams.find((x) => x.codec_type === 'subtitle');
  const orig = probe(join(FILES, 'test.mp4'));
  const i = probe(p);
  if (dl.suggestedFilename() !== 'test-subtitled.mkv') throw new Error(dl.suggestedFilename());
  if (!s || s.codec_name !== 'subrip') throw new Error(`subtitle track: ${s?.codec_name}`); // SRT-style: what FFmpeg-based players read
  if (i.v.codec_name !== orig.v.codec_name || i.a.codec_name !== orig.a.codec_name) throw new Error('picture or sound was changed');
  const text = execFileSync(FFMPEG, ['-v', 'error', '-i', p, '-map', '0:s:0', '-f', 'webvtt', '-'], { encoding: 'utf8' });
  if (!/Hello there[\s\S]*Second line/.test(text)) throw new Error(`cues: ${text}`);
  // the file was made on disk (the browser's private storage), not in memory, so a video of any size can be done
  const onDisk = await page.evaluate(async () => { const n = []; for await (const k of (await navigator.storage.getDirectory()).keys()) n.push(k); return n.filter((k) => k.startsWith('build-')).length; });
  if (onDisk !== 1) throw new Error(`files in private storage: ${onDisk}`);
  return `${dl.suggestedFilename()}: ${i.v.codec_name} + ${i.a.codec_name} + ${s.codec_name}, made on disk`;
});

await check('under the production Content-Security-Policy: MP3 (blob worker + wasm), Opus and a video type are saved with no violations', async () => {
  const csp = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; connect-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'";
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
  const p2 = await ctx.newPage();
  const viol = [];
  p2.on('console', (m) => { if (/Content Security Policy|violat|refused to/i.test(m.text())) viol.push(m.text().slice(0, 200)); });
  p2.on('pageerror', (e) => viol.push(e.message));
  await ctx.route('**/*', async (route) => {
    if (!route.request().url().startsWith('http://127.0.0.1:8792') || route.request().resourceType() === 'websocket') return route.continue();
    const res = await route.fetch();
    await route.fulfill({ response: res, headers: { ...res.headers(), 'content-security-policy': csp } });
  });
  await signIn(p2, EMAIL);
  await p2.goto(`${BASE}/app`);
  await dropFile(p2, join(FILES, 'noisy.wav'));
  await p2.getByRole('button', { name: 'Start – free' }).click();
  await p2.waitForSelector('[data-testid=result]', { timeout: 120000 });
  const f = (id) => p2.locator(`[data-testid=save-as] [data-testid=${id}]`);
  await p2.waitForFunction(() => { const s = document.querySelector('[data-testid=save-as] [data-testid=out-format]'); return s && !s.disabled; });
  const got = [];
  for (const name of ['csp.mp3', 'csp.ogg', 'csp.webm']) {
    await f('out-name').fill(name);
    const [dl] = await Promise.all([p2.waitForEvent('download', { timeout: 60000 }), f('save-btn').click()]);
    const out = join(FILES, `conv-${dl.suggestedFilename()}`);
    await dl.saveAs(out);
    got.push(`${dl.suggestedFilename()} ${statSync(out).size}`);
    await p2.waitForFunction(() => !document.querySelector('[data-testid=save-as] [data-testid=save-btn]').disabled);
  }
  await ctx.close();
  if (viol.length) throw new Error(viol.join(' | '));
  return got.join(', ');
});

const bad = logs.filter((l) => !/Failed to load resource|favicon|net::ERR/.test(l));
await check('no console errors on the way', async () => {
  if (bad.length) throw new Error(bad.slice(0, 3).join(' || '));
  return `${logs.length} warnings ignored`;
});

await browser.close();
const res = summary();
const failed = res.filter((r) => !r.ok);
console.log(`\n${res.length - failed.length}/${res.length} passed`);
process.exit(failed.length ? 1 : 0);
