// The editor on a recording of any length: a 40 minute mono file (generated here: a moving tone in pink noise) is cleaned by the real worker
// (a job made through the API with no timeline, so it also keeps its two copies), then
//  1. the job keeps both copies (the recording is no longer dropped for a long file) and says how to read them;
//  2. the preview stretches (GET .../preview/{source|noise}?start=&seconds=): the same short FLACs with the headers that say what is in them;
//  3. the editor plays from short stretches of the layers it needs: never a whole copy, a few seconds held at a time, the next stretch fetched
//     ahead, a seek fetches the stretches there, a change of strength asks for nothing;
//  4. playing across stretch borders keeps going (the time moves on without a pause) and the end of the recording stops it;
//  5. the memory of the page stays small: the renderer's resident memory grows by far less than the two whole copies (921 MB) would take;
//  6. saving another strength (no Leveling) is free for this length: "Save changes · Free", an export on the same job, no new job, its file
//     comes down by itself, and it is the blend of the two layers at that strength (checked on a window of the file, decoded to float).
// Needs the test backend (tests/run.sh stretch) and ffmpeg. Takes a few minutes (the cleaning of 40 minutes). Usage: node tests/stretch.mjs
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { BASE, FFMPEG, admin, apiCall, check, launch, signIn, studio, summary, waitCardDone, waitJobDone, waitResult } from './helpers.mjs';

const EMAIL = `stretch-${Date.now()}@example.com`;
const MINUTES = 40;
const SECONDS = MINUTES * 60;
const SR = 48000;
const WORK = join(process.env.ACW_TEST_STATE || tmpdir(), `stretch-${process.pid}`);
mkdirSync(WORK, { recursive: true });

const { browser, context, page, logs } = await launch();
const got = [];
page.on('download', (d) => got.push({ name: d.suggestedFilename(), download: d }));
const asked = []; // every request for a preview stretch: path and query
const requests = [];
context.on('request', (r) => {
  const u = new URL(r.url());
  if (/\/preview\/(source|noise|rebuild|echo)$/.test(u.pathname)) asked.push(u.pathname + u.search);
  if (u.pathname.startsWith('/api/')) requests.push(`${r.method()} ${u.pathname}`);
});
await signIn(page, EMAIL);
admin('grant', EMAIL, '5', 'stretch test');

const sleep = (ms) => page.waitForTimeout(ms);
const near = (a, b, tol, what) => { if (!(Math.abs(a - b) <= tol)) throw new Error(`${what}: ${a} vs ${b}`); };
const stretchQuery = (u) => { const q = new URL(u, BASE).searchParams; return { start: Number(q.get('start')), seconds: Number(q.get('seconds')), has: q.has('start') && q.has('seconds') }; };
const layerOf = (u) => u.split('?')[0].split('/').pop();
const startsOf = (list) => [...new Set(list.map(stretchQuery).map((x) => x.start))].sort((a, b) => a - b);
const attr = (n) => studio(page).getAttribute(n);
const timeNow = async () => Number(await attr('data-time'));
const held = async () => Number(await attr('data-held'));
const countBy = (re) => requests.filter((r) => re.test(r)).length;

/** Resident memory (MB) of the browser's renderer processes, from /proc (Linux): the page's audio buffers live there. */
async function rendererMb() {
  const cdp = await browser.newBrowserCDPSession();
  try {
    const { processInfo } = await cdp.send('SystemInfo.getProcessInfo');
    let kb = 0;
    for (const p of processInfo.filter((x) => x.type === 'renderer')) {
      try { kb += Number(/VmRSS:\s+(\d+) kB/.exec(readFileSync(`/proc/${p.id}/status`, 'utf8'))?.[1] || 0); } catch { /* gone */ }
    }
    return kb / 1024;
  } finally {
    await cdp.detach().catch(() => {});
  }
}

const decode = (path, args = []) => {
  const buf = execFileSync(FFMPEG, ['-v', 'error', ...args, '-i', path, '-ar', '48000', '-ac', '1', '-f', 'f32le', '-'], { maxBuffer: 1 << 29 });
  return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
};

/** Click the waveform at second t (moves the playhead; a 40 minute recording is a few seconds a pixel). */
async function seekTo(t) {
  await page.getByTestId('waveform').scrollIntoViewIfNeeded();
  const box = await page.getByTestId('waveform').boundingBox();
  const T = Number(await page.getByTestId('wave').getAttribute('data-seconds'));
  await page.mouse.click(box.x + (t / T) * box.width, box.y + box.height / 2);
  await sleep(200);
  return timeNow();
}

let JOB;
let preview;
let cost0 = 0;

await check(`setup: a ${MINUTES} minute mono recording is cleaned (Denoising, light remover, no Leveling, through the API with no timeline) and the job keeps both copies`, async () => {
  const flac = join(WORK, 'long.flac');
  execFileSync(FFMPEG, ['-v', 'error', '-f', 'lavfi', '-i', `sine=f=220:r=48000:d=${SECONDS}`, '-f', 'lavfi', '-i', `anoisesrc=c=pink:a=0.08:r=48000:d=${SECONDS}`,
    '-filter_complex', '[0][1]amix=inputs=2:normalize=0,volume=0.8', '-ac', '1', '-c:a', 'flac', '-compression_level', '0', '-y', flac], { timeout: 300000 });
  const size = statSync(flac).size;
  const made = await apiCall(page, 'POST', 'jobs', {
    kind: 'denoise', denoiseTier: 'standard', filename: 'long.flac', sizeBytes: size, durationSeconds: SECONDS,
    options: { noise: 'light', leveling: false, atten_lim_db: 100, output_video: false, audio_format: 'flac' },
  });
  if (made.status !== 200) throw new Error(`create ${made.status} ${JSON.stringify(made.body)}`);
  JOB = made.body.id;
  const csrf = (await (await context.request.get(`${BASE}/api/me`)).json()).csrfToken;
  const put = await context.request.put(`${BASE}/api/jobs/${JOB}/input`, { data: readFileSync(flac), headers: { 'X-CSRF-Token': csrf, 'Content-Type': 'application/octet-stream' }, timeout: 600000 });
  if (!put.ok()) throw new Error(`upload ${put.status()} ${await put.text()}`);
  rmSync(flac, { force: true });
  const done = await waitJobDone(page, JOB, { timeout: 1500000 });
  if (done.status !== 'done') throw new Error(`the job ${done.status}: ${done.error}`);
  const redo = done.redo;
  if (!redo?.available || !redo.noise || !redo.preview) throw new Error('redo: ' + JSON.stringify(redo));
  preview = redo.preview;
  cost0 = done.costMicros;
  near(preview.samples, SECONDS * SR, SR, 'samples');
  if (!(preview.bytes.noise > 1e6) || !(preview.bytes.input > 1e6) || preview.stretchMaxSeconds !== 30) throw new Error('preview: ' + JSON.stringify(preview));
  if (!(preview.energy?.full > 0) || !(preview.energy?.input > 0)) throw new Error('no energy sums: ' + JSON.stringify(preview.energy));
  return `${JOB.slice(0, 8)}… ${preview.samples} samples, copies ${(preview.bytes.noise / 1e6).toFixed(0)} + ${(preview.bytes.input / 1e6).toFixed(0)} MB`;
});

await check('the endpoint: a stretch of each copy is a small FLAC with the headers that say what is in it; a start past the end is refused; asking for more than 30 seconds gives 30', async () => {
  const seen = [];
  for (const which of ['noise', 'input']) {
    const res = await context.request.get(`${BASE}/api/jobs/${JOB}/kept/${which}?start=1507.000000&seconds=10.000000`);
    if (res.status() !== 200 || !/^audio\/flac/.test(res.headers()['content-type'] || '')) throw new Error(`${which}: ${res.status()}`);
    const h = res.headers();
    if (h['x-stretch-start'] !== String(1507 * SR) || h['x-stretch-samples'] !== String(10 * SR) || h['x-total-samples'] !== String(preview.samples)) throw new Error('headers ' + JSON.stringify(h));
    if (!/no-store/.test(h['cache-control'])) throw new Error('cacheable');
    const body = await res.body();
    if (body.subarray(0, 4).toString('latin1') !== 'fLaC' || body.length > 3e6) throw new Error(`${which}: ${body.length} bytes`);
    seen.push(`${which} ${(body.length / 1e6).toFixed(2)} MB`);
  }
  if ((await context.request.get(`${BASE}/api/jobs/${JOB}/kept/noise?start=${SECONDS + 5}&seconds=1`)).status() !== 400) throw new Error('a start past the end');
  const long = await context.request.get(`${BASE}/api/jobs/${JOB}/kept/input?start=100&seconds=900`);
  if (long.headers()['x-stretch-samples'] !== String(30 * SR)) throw new Error('not capped: ' + long.headers()['x-stretch-samples']);
  return seen.join(', ');
});

await check('the preview endpoint the editor plays from: a stretch of the recording and of the noise removal is a small FLAC with the same headers, they line up sample for sample, a start past the end is refused, more than 30 seconds gives 30', async () => {
  const seen = [];
  for (const layer of ['source', 'noise']) {
    const res = await context.request.get(`${BASE}/api/jobs/${JOB}/preview/${layer}?start=1507.000000&seconds=10.000000`);
    if (res.status() !== 200 || !/^audio\/flac/.test(res.headers()['content-type'] || '')) throw new Error(`${layer}: ${res.status()} ${await res.text()}`);
    const h = res.headers();
    if (h['x-stretch-start'] !== String(1507 * SR) || h['x-stretch-samples'] !== String(10 * SR) || h['x-total-samples'] !== String(preview.samples) || !(Number(h['x-stretch-scale']) > 0)) throw new Error('headers ' + JSON.stringify(h));
    const body = await res.body();
    if (body.subarray(0, 4).toString('latin1') !== 'fLaC' || body.length > 4e6) throw new Error(`${layer}: ${body.length} bytes`);
    seen.push(`${layer} ${(body.length / 1e6).toFixed(2)} MB`);
  }
  if ((await context.request.get(`${BASE}/api/jobs/${JOB}/preview/noise?start=${SECONDS + 5}&seconds=1`)).status() !== 400) throw new Error('a start past the end');
  const long = await context.request.get(`${BASE}/api/jobs/${JOB}/preview/source?start=100&seconds=900`);
  if (long.headers()['x-stretch-samples'] !== String(30 * SR)) throw new Error('not capped: ' + long.headers()['x-stretch-samples']);
  return seen.join(', ');
});

let baseline = 0;
let peak = 0; // the most the renderer held while the checks played
await check('the editor: opening the job page asks only for the first short stretches of the layers it plays (no whole copy), the strength starts at what the job was made with, and the page holds a few seconds of sound', async () => {
  asked.length = 0;
  await page.goto(`${BASE}/app/job/${JOB}`);
  await studio(page).waitFor({ timeout: 60000 });
  await page.locator('[data-stage=done][data-testid=studio]').waitFor({ timeout: 60000 });
  await page.getByTestId('hear-voice').waitFor();
  await page.waitForFunction(() => Number(document.querySelector('[data-testid=studio]')?.getAttribute('data-held')) >= 3 && document.querySelector('[data-testid=studio]')?.getAttribute('data-buffering') === 'no', null, { timeout: 60000 });
  baseline = await rendererMb();
  const q = asked.map(stretchQuery);
  if (asked.length < 6 || !q.every((x) => x.has && x.seconds > 0 && x.seconds <= 10.000001)) throw new Error('requests: ' + asked.join(' '));
  if ([...new Set(asked.map(layerOf))].sort().join() !== 'noise,source') throw new Error('not just the two layers: ' + asked.join(' '));
  const starts = startsOf(asked);
  if (starts.join() !== '0,10,20') throw new Error('stretches asked for: ' + starts);
  if ((await page.getByTestId('strength-name').innerText()) !== 'Maximum (100 dB)') throw new Error('strength ' + await page.getByTestId('strength-name').innerText());
  if ((await page.getByTestId('hearing').innerText().catch(() => 'Hearing: everything')) !== 'Hearing: everything') throw new Error('hearing');
  return `${asked.length} requests: stretches at ${starts.join(', ')} s of both layers`;
});

await check('a change of strength asks for nothing: the four names and the slider change the gains of the two signals that are already here', async () => {
  const before = asked.length;
  for (const [id, db] of [['balanced', 24], ['gentle', 12], ['strong', 40], ['max', 100]]) {
    await page.locator(`.studio button[data-act=v-str][data-v="${{ gentle: 1, balanced: 2, strong: 3, max: 4 }[id]}"]`).click();
    await page.waitForFunction((x) => document.querySelector('[data-testid=strength-name]')?.textContent.includes(`(${x} dB)`), db, { timeout: 5000 });
    const a = db >= 100 ? 0 : 10 ** (-db / 20);
    const g = JSON.parse(await attr('data-gains'));
    near(g.source / (g.source + g.noise), a, 2e-3, `${db} dB share of the recording`);
  }
  await page.getByTestId('strength-slider').fill('1.5');
  await sleep(500);
  if (asked.length !== before) throw new Error(`${asked.length - before} requests for a change of strength`);
  return 'no requests';
});

await check('Play: the sound starts, the time moves on across the borders of stretches (a seek to about 25:07) without pausing, the next stretches are asked for ahead, and the old ones are let go', async () => {
  await page.locator('.studio button[data-act=v-str][data-v="2"]').click(); // Balanced: both layers are heard
  asked.length = 0;
  const t0 = await seekTo(1507);
  near(t0, 1507, 8, 'the playhead after a click at 25:07'); // (a pixel is a few seconds of 40 minutes)
  const first = Math.floor(t0 / 10) * 10;
  await sleep(600); // (the stretches there are fetched at the seek, before Play, so that Play starts at once: they are counted from the seek)
  await page.getByTestId('play').click();
  await page.waitForFunction(() => document.querySelector('[data-testid=studio]')?.getAttribute('data-playing') === 'yes', null, { timeout: 8000 });
  await page.waitForFunction(() => Number(document.querySelector('[data-testid=studio]')?.getAttribute('data-held')) >= 3 && document.querySelector('[data-testid=studio]')?.getAttribute('data-buffering') === 'no', null, { timeout: 30000 });
  const starts0 = startsOf(asked);
  if (starts0[0] !== first || starts0.some((s) => s < first || s > first + 20)) throw new Error(`stretches asked for after the seek to ${t0}: ${starts0}`);
  const times = [];
  const t1 = Date.now();
  let stalled = 0;
  while (Date.now() - t1 < 12000) {
    peak = Math.max(peak, await rendererMb());
    times.push(await timeNow());
    if ((await attr('data-buffering')) === 'yes') stalled++;
    await sleep(400);
  }
  if (!(times.at(-1) >= Math.ceil(t0 / 10) * 10 + 2)) throw new Error('the time never passed the next border: ' + times.join(' '));
  for (let i = 1; i < times.length; i++) if (times[i] < times[i - 1] || times[i] - times[i - 1] > 2) throw new Error('the time jumped: ' + times.join(' '));
  if (stalled > 1) throw new Error(`waited for sound ${stalled} times`);
  const now = await held();
  const starts = startsOf(asked);
  const at = Math.floor(times.at(-1) / 10) * 10;
  if (starts.some((s) => s < first) || !starts.includes(at + 20)) throw new Error(`stretches asked for while playing (the time is ${times.at(-1)}): ${starts}`);
  if (now > 4) throw new Error(`${now} stretches held`);
  await page.getByTestId('play').click();
  return `time ${times[0]} → ${times.at(-1)} s without a gap; stretches asked for ${starts.join(', ')}; ${now} held`;
});

await check('a seek far away: move to 10:00 and Play: the stretches around 10:00 are fetched (only those, none of 0:00), and it plays on from there; seeking again and again while it plays asks only for where it ends up', async () => {
  asked.length = 0;
  const t = await seekTo(600);
  near(t, 600, 8, 'seek to 10:00');
  await sleep(400);
  await page.getByTestId('play').click();
  await page.waitForFunction(() => document.querySelector('[data-testid=studio]')?.getAttribute('data-playing') === 'yes', null, { timeout: 8000 });
  await page.waitForFunction((x) => Number(document.querySelector('[data-testid=studio]')?.getAttribute('data-time')) > x + 2, t, { timeout: 30000 });
  const starts = startsOf(asked);
  if (starts.some((s) => s < Math.floor(t / 10) * 10 - 10 || s > Math.floor(t / 10) * 10 + 40)) throw new Error('stretches asked for: ' + starts); // (the one before the playhead's own may be asked for too, near a border)
  // dragging the position while it plays: a few seeks one after the other in one go (the last one is where it plays)
  asked.length = 0;
  const target = await page.evaluate(() => {
    const wf = document.getElementById('wf');
    const T = Number(document.querySelector('[data-testid=wave]').getAttribute('data-seconds'));
    const r = wf.getBoundingClientRect();
    for (const s of [900, 1200, 1800]) wf.dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: r.left + (s / T) * r.width, clientY: r.top + r.height / 2 }));
    return (1800 / T) * 100;
  });
  await page.waitForFunction(() => Number(document.querySelector('[data-testid=studio]')?.getAttribute('data-time')) > 1790, null, { timeout: 30000 });
  await sleep(500);
  await page.waitForFunction(() => document.querySelector('[data-testid=studio]')?.getAttribute('data-buffering') === 'no' && Number(document.querySelector('[data-testid=studio]')?.getAttribute('data-held')) >= 3, null, { timeout: 30000 });
  const after = startsOf(asked);
  const now = await timeNow();
  if (after.some((s) => s < 1790 || s > 1850)) throw new Error(`stretches asked for after dragging (the time is ${now}): ${after}`);
  if (!(now >= 1790 && now < 1850)) throw new Error('playing at ' + now);
  await page.getByTestId('play').click();
  return `10:00 → ${starts.join(', ')}; dragging to 30:00 (${target.toFixed(0)}% of the way) → ${after.join(', ')}`;
});

await check('the end: from the last seconds it plays the last stretch to the end, the button goes back to Play and the time rests at the end', async () => {
  asked.length = 0;
  // (a mouse click a few pixels from the very edge of the picture lands on the end handle of the selection, not on the picture: the click is sent to the picture itself)
  await page.evaluate((s) => {
    const wf = document.getElementById('wf');
    const T = Number(document.querySelector('[data-testid=wave]').getAttribute('data-seconds'));
    const r = wf.getBoundingClientRect();
    wf.dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: r.left + (s / T) * r.width, clientY: r.top + r.height / 2 }));
  }, SECONDS - 4);
  await sleep(200);
  const t = await timeNow();
  near(t, SECONDS - 4, 1, 'the playhead after a click near the end');
  await sleep(400);
  await page.getByTestId('play').click();
  await page.waitForFunction(() => document.querySelector('[data-testid=studio]')?.getAttribute('data-playing') === 'yes', null, { timeout: 8000 });
  await page.waitForFunction(() => document.querySelector('[data-testid=studio]')?.getAttribute('data-playing') === 'no', null, { timeout: 30000 })
    .catch(async (e) => { throw new Error(`still playing 30 s after a start at ${t.toFixed(1)} s: time ${await timeNow()}, buffering ${await attr('data-buffering')}, held ${await attr('data-held')}, asked ${startsOf(asked)} (${e.message.split('\n')[0]})`); });
  if ((await page.getByTestId('play').getAttribute('aria-label')) !== 'Play') throw new Error('the button says ' + await page.getByTestId('play').getAttribute('aria-label'));
  const last = asked.map(stretchQuery).filter((x) => x.start >= SECONDS - 10);
  if (!last.length) throw new Error('no request for the last stretch');
  near(await timeNow(), SECONDS, 1, 'the time rests at the end');
  return `from ${t.toFixed(0)} s; the last stretch: ${last.map((x) => `${x.start}+${x.seconds}`).join(' ')}`;
});

await check('memory: nothing but short stretches was ever fetched (no whole copy), at most 4 stretches are held, and the renderer\'s memory is far below the 921 MB the two whole copies would take', async () => {
  const whole = asked.filter((u) => !u.includes('?'));
  if (whole.length || requests.some((r) => /\/kept\//.test(r))) throw new Error('whole copies fetched: ' + whole.join(' '));
  const now = await held();
  if (now > 4) throw new Error(`${now} held`);
  const rss = await rendererMb();
  const grew = Math.max(rss, peak) - baseline;
  if (!(grew < 250)) throw new Error(`the renderer grew by ${grew.toFixed(0)} MB (${baseline.toFixed(0)} → ${Math.max(rss, peak).toFixed(0)})`);
  return `${now} stretches held at the end; renderer ${baseline.toFixed(0)} MB before, ${peak.toFixed(0)} at the most while playing, ${rss.toFixed(0)} now (+${grew.toFixed(0)} at the most); the two whole copies would be ${(SECONDS * SR * 4 * 2 / 1048576).toFixed(0)} MB`;
});

await check('saving another strength with nothing after the noise removal is free at this length: "Save changes · Free", an export on the same job (no new job, no re-run), its file comes down by itself', async () => {
  await page.locator('.studio button[data-act=v-str][data-v="2"]').click(); // Balanced
  await page.waitForFunction(() => { const b = document.querySelector('[data-testid=save-changes]'); return b && !b.disabled; }, null, { timeout: 30000 });
  const label = (await page.getByTestId('save-changes').innerText()).replace(/\s+/g, ' ').trim();
  if (label !== 'Save changes · Free') throw new Error('label: ' + label);
  const jobs = countBy(/^POST \/api\/jobs$/);
  got.length = 0;
  const [req] = await Promise.all([page.waitForRequest((r) => r.url().endsWith(`/api/jobs/${JOB}/export`) && r.method() === 'POST'), page.getByTestId('save-changes').click()]);
  const body = req.postDataJSON();
  if ('maxCostMicros' in body || body.timeline?.layers?.noise?.length !== 1 || body.timeline.layers.noise[0].strength_db !== 24) throw new Error('sent: ' + JSON.stringify(body).slice(0, 300));
  await page.locator('[data-testid=studio][data-stage=processing]').waitFor({ timeout: 30000 });
  await waitCardDone(page, 900000);
  await waitResult(page, 'result', 600000);
  const t0 = Date.now();
  while (!got.length && Date.now() - t0 < 120000) await sleep(300);
  if (got.length !== 1 || !/\.flac$/.test(got[0].name)) throw new Error('downloaded by itself: ' + got.map((g) => g.name));
  await got[0].download.saveAs(join(WORK, 'balanced.flac'));
  const done = (await apiCall(page, 'GET', `jobs/${JOB}`)).body;
  if (done.status !== 'done' || done.costMicros !== cost0 || !done.edit?.hasTimeline) throw new Error(`the job: ${done.status} cost ${done.costMicros} (was ${cost0}) ${JSON.stringify(done.edit)}`);
  if (countBy(/^POST \/api\/jobs$/) !== jobs || countBy(/rerun$/)) throw new Error('a new job or a re-run was asked for');
  if (asked.some((u) => !u.includes('?')) || requests.some((r) => /\/kept\//.test(r))) throw new Error('a whole copy was fetched to save');
  return `${label}; ${got[0].name} (${(statSync(join(WORK, 'balanced.flac')).size / 1e6).toFixed(1)} MB), cost unchanged`;
});

await check('what comes down is the blend of the two layers at that strength: a window of the file (at 25:07, 10 s) equals (1 - a) * noise removal + a * recording from the two stretches, to the 16-bit file\'s precision; the slider holds that strength after a reload', async () => {
  const a = 10 ** (-24 / 20);
  const windowOf = async (layer) => {
    const res = await context.request.get(`${BASE}/api/jobs/${JOB}/preview/${layer}?start=1507.000000&seconds=10.000000`);
    const scale = Number(res.headers()['x-stretch-scale']) || 1;
    const p = join(WORK, `${layer}.flac`);
    writeFileSync(p, await res.body());
    return decode(p).map((v) => v / scale);
  };
  const full = await windowOf('noise');
  const input = await windowOf('source');
  const file = decode(join(WORK, 'balanced.flac'), ['-ss', '1507', '-t', '10']);
  if (full.length !== 10 * SR || input.length !== 10 * SR || Math.abs(file.length - 10 * SR) > 2) throw new Error(`lengths ${full.length} ${input.length} ${file.length}`);
  let worst = 0;
  let rms = 0;
  let apart = 0;
  for (let i = 0; i < full.length; i++) {
    worst = Math.max(worst, Math.abs(file[i] - ((1 - a) * full[i] + a * input[i])));
    rms += file[i] * file[i];
    apart += (file[i] - full[i]) ** 2;
  }
  rms = Math.sqrt(rms / full.length);
  apart = Math.sqrt(apart / full.length);
  if (!(rms > 1e-3)) throw new Error('a silent file: ' + rms);
  if (!(apart > 1e-3)) throw new Error('the file is the full-strength sound, not the blend: ' + apart);
  if (!(worst < 3e-4)) throw new Error(`the file differs from the blend by ${worst}`);
  await page.goto(`${BASE}/app/job/${JOB}`);
  await page.locator('[data-stage=done][data-testid=studio]').waitFor({ timeout: 60000 });
  await page.getByTestId('strength-name').waitFor();
  if ((await page.getByTestId('strength-name').innerText()) !== 'Balanced (24 dB)') throw new Error('starts at ' + await page.getByTestId('strength-name').innerText());
  return `rms ${rms.toFixed(4)} (${apart.toFixed(4)} away from the full-strength sound), worst difference from the blend ${worst.toExponential(1)}; the slider starts at Balanced`;
});

rmSync(WORK, { recursive: true, force: true });
try { for (const f of readdirSync(WORK)) rmSync(join(WORK, f), { force: true }); } catch { /* removed */ }
console.log('LOGS:\n' + logs.filter((l) => !/Failed to load resource|favicon|net::ERR/.test(l)).join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
