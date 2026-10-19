// The strength slider on a recording of any length: a 40 minute mono file (generated here: a moving tone in pink noise) is cleaned by the
// real worker, then
//  1. the job keeps both copies (the recording is no longer dropped for a long file) and says how to read them;
//  2. the slider plays from short stretches (GET .../kept/{noise|input}?start=&seconds=): never a whole copy, a few seconds held at a time,
//     the next stretch fetched ahead, a seek fetches the stretches there, a change of strength asks for nothing;
//  3. playing across stretch borders keeps going (the time moves on without a pause) and the end of the recording stops it;
//  4. the memory of the page stays small: the renderer's resident memory grows by far less than the two whole copies (921 MB) would take;
//  5. saving a new strength (no Leveling) is a free re-run on our side for this length: "Save this strength · Free", a new job, its file
//     comes down by itself, and it is the blend of the two copies at that strength (checked on a window of the file, decoded to float).
// Needs the test backend (tests/run.sh stretch) and ffmpeg. Takes a few minutes (the cleaning of 40 minutes). Usage: node tests/stretch.mjs
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { BASE, FFMPEG, JOB_PATH, admin, apiCall, check, jobIdOf, launch, signIn, summary, waitJobDone } from './helpers.mjs';

const EMAIL = `stretch-${Date.now()}@example.com`;
const MINUTES = 40;
const SECONDS = MINUTES * 60;
const SR = 48000;
const WORK = join(process.env.ACW_TEST_STATE || tmpdir(), `stretch-${process.pid}`);
mkdirSync(WORK, { recursive: true });

const { browser, context, page, logs } = await launch();
const got = [];
page.on('download', (d) => got.push({ name: d.suggestedFilename(), download: d }));
const kept = []; // every request for a kept copy: path and query
const requests = [];
page.on('request', (r) => {
  const u = new URL(r.url());
  if (/\/kept\/(noise|input)$/.test(u.pathname)) kept.push(u.pathname + u.search);
  if (u.pathname.startsWith('/api/')) requests.push(`${r.method()} ${u.pathname}`);
});
await signIn(page, EMAIL);
admin('grant', EMAIL, '5', 'stretch test');

const sleep = (ms) => page.waitForTimeout(ms);
const card = () => page.getByTestId('job').first();
const blend = () => card().getByTestId('blend-player');
const attr = (n) => blend().getAttribute(n);
const timeText = () => card().locator('[data-testid=blend-player] .time').innerText().then((t) => t.split('/')[0].trim());
const secs = (t) => t.split(':').map(Number).reduce((s, v) => s * 60 + v, 0);
const near = (a, b, tol, what) => { if (!(Math.abs(a - b) <= tol)) throw new Error(`${what}: ${a} vs ${b}`); };
const stretchQuery = (u) => { const q = new URL(u, BASE).searchParams; return { start: Number(q.get('start')), seconds: Number(q.get('seconds')), has: q.has('start') && q.has('seconds') }; };

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

let JOB;
let preview;

await check(`setup: a ${MINUTES} minute mono recording is cleaned (Denoising, light remover, no Leveling) and the job keeps both copies`, async () => {
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

let baseline = 0;
let peak = 0; // the most the renderer held while the checks played
await check('the slider: opening Listen on the job page shows the player at once (no whole copy is fetched, only the first stretches), the strength starts at what the job was made with, and the page holds a few seconds of sound', async () => {
  await page.goto(`${BASE}/app/job/${JOB}`);
  await card().getByTestId('download-again').waitFor({ timeout: 30000 });
  baseline = await rendererMb();
  kept.length = 0;
  await card().getByTestId('listen').locator('summary').click();
  await blend().waitFor({ timeout: 60000 });
  if ((await attr('data-source')) !== 'server' || (await attr('data-db')) !== '100.00') throw new Error(`source ${await attr('data-source')}, ${await attr('data-db')} dB`);
  await page.waitForFunction(() => Number(document.querySelector('[data-testid=blend-player]')?.getAttribute('data-held')) >= 3, null, { timeout: 30000 });
  const q = kept.map(stretchQuery);
  if (kept.length < 6 || !q.every((x) => x.has && x.seconds > 0 && x.seconds <= 10.000001)) throw new Error('requests: ' + kept.join(' '));
  if (new Set(kept.map((u) => u.split('?')[0].split('/').pop())).size !== 2) throw new Error('not both copies: ' + kept.join(' '));
  const starts = [...new Set(q.map((x) => x.start))].sort((a, b) => a - b);
  if (starts.join() !== '0,10,20') throw new Error('stretches asked for: ' + starts);
  return `${kept.length} requests: stretches at ${starts.join(', ')} s of both copies`;
});

await check('a change of strength asks for nothing: the four names and the range change the gains of the two signals that are already here', async () => {
  const before = kept.length;
  for (const [id, db] of [['balanced', 24], ['gentle', 12], ['strong', 40], ['max', 100]]) {
    await card().getByTestId(`strength-set-${id}`).click();
    await page.waitForFunction((x) => document.querySelector('[data-testid=blend-player]')?.getAttribute('data-db') === x, db.toFixed(2), { timeout: 5000 });
    const a = db >= 100 ? 0 : 10 ** (-db / 20);
    const full = Number(await attr('data-gain-full'));
    const input = Number(await attr('data-gain-input'));
    near(input / (full + input), a, 2e-3, `${db} dB share of the recording`);
  }
  await sleep(500);
  if (kept.length !== before) throw new Error(`${kept.length - before} requests for a change of strength`);
  return 'no requests';
});

await check('Play: the sound starts, the time moves on across the borders of stretches (a seek to 25:07, three seconds before a border) without pausing, the next stretches are asked for ahead, and the old ones are let go', async () => {
  await card().getByTestId('strength-set-balanced').click();
  kept.length = 0;
  await card().getByTestId('blend-seek').fill('1507');
  await sleep(600); // (the stretches there are fetched after a short pause, before Play, so that Play starts at once)
  await card().getByTestId('blend-play').click();
  await page.waitForFunction(() => document.querySelector('[data-testid=blend-play]')?.textContent === 'Pause', null, { timeout: 8000 });
  await page.waitForFunction(() => Number(document.querySelector('[data-testid=blend-player]')?.getAttribute('data-held')) >= 3 && document.querySelector('[data-testid=blend-player]')?.getAttribute('data-buffering') === 'no', null, { timeout: 30000 });
  const starts0 = [...new Set(kept.map(stretchQuery).map((x) => x.start))].sort((a, b) => a - b);
  if (starts0[0] !== 1500 || starts0.some((s) => s < 1500 || s > 1520)) throw new Error('stretches asked for after the seek: ' + starts0);
  const times = [];
  const t0 = Date.now();
  let stalled = 0;
  while (Date.now() - t0 < 9000) {
    peak = Math.max(peak, await rendererMb());
    times.push(secs(await timeText()));
    if ((await attr('data-buffering')) === 'yes') stalled++;
    await sleep(400);
  }
  if (!(times.at(-1) >= 1512)) throw new Error('the time never passed the border at 25:10: ' + times.join(' '));
  for (let i = 1; i < times.length; i++) if (times[i] < times[i - 1] || times[i] - times[i - 1] > 2) throw new Error('the time jumped: ' + times.join(' '));
  if (stalled > 1) throw new Error(`waited for sound ${stalled} times`);
  const held = Number(await attr('data-held'));
  const starts = [...new Set(kept.map(stretchQuery).map((x) => x.start))].sort((a, b) => a - b);
  if (starts.some((s) => s < 1500) || !starts.includes(1530)) throw new Error('stretches asked for while playing: ' + starts);
  if (held > 4) throw new Error(`${held} stretches held`);
  await card().getByTestId('blend-play').click();
  return `time ${times[0]} → ${times.at(-1)} s without a gap; stretches asked for ${starts.join(', ')}; ${held} held`;
});

await check('a seek far away: Pause, move the range to 10:00 and Play: the stretches around 10:00 are fetched (only those, none of 0:00), and it plays on from there', async () => {
  kept.length = 0;
  await card().getByTestId('blend-seek').fill('600');
  await sleep(400);
  await card().getByTestId('blend-play').click();
  await page.waitForFunction(() => document.querySelector('[data-testid=blend-play]')?.textContent === 'Pause', null, { timeout: 8000 });
  await page.waitForFunction(() => { const t = document.querySelector('[data-testid=blend-player] .time')?.textContent || ''; return /^10:0[2-9]|^10:[1-5]/.test(t); }, null, { timeout: 30000 });
  const starts = [...new Set(kept.map(stretchQuery).map((x) => x.start))].sort((a, b) => a - b);
  if (!starts.includes(600) || starts.some((s) => s < 590 || s > 640)) throw new Error('stretches asked for: ' + starts);
  // dragging the range while it plays asks only for where it ends up
  kept.length = 0;
  for (const v of ['900', '1200', '1800']) await card().getByTestId('blend-seek').fill(v);
  await page.waitForFunction(() => /^30:/.test(document.querySelector('[data-testid=blend-player] .time')?.textContent || ''), null, { timeout: 30000 });
  await sleep(500);
  await page.waitForFunction(() => document.querySelector('[data-testid=blend-player]')?.getAttribute('data-buffering') === 'no' && Number(document.querySelector('[data-testid=blend-player]')?.getAttribute('data-held')) >= 3, null, { timeout: 30000 });
  const after = [...new Set(kept.map(stretchQuery).map((x) => x.start))].sort((a, b) => a - b);
  if (!after.includes(1800) || after.some((s) => s < 1790 || s > 1830)) throw new Error('stretches asked for after dragging: ' + after);
  const time = secs(await timeText());
  if (!(time >= 1800 && time < 1830)) throw new Error('playing at ' + time);
  await card().getByTestId('blend-play').click();
  return `10:00 → ${starts.join(', ')}; dragging to 30:00 → ${after.join(', ')}`;
});

await check('the end: from 39:55 it plays the last, shorter stretch to the end, the button goes back to Play and the time rests at the start', async () => {
  await card().getByTestId('blend-seek').fill(String(SECONDS - 5));
  await sleep(400);
  await card().getByTestId('blend-play').click();
  await page.waitForFunction(() => document.querySelector('[data-testid=blend-play]')?.textContent === 'Pause', null, { timeout: 8000 });
  await page.waitForFunction(() => document.querySelector('[data-testid=blend-play]')?.textContent === 'Play', null, { timeout: 30000 });
  const last = kept.map(stretchQuery).filter((x) => x.start >= SECONDS - 10);
  if (!last.length) throw new Error('no request for the last stretch');
  return `the last stretch: ${last.map((x) => `${x.start}+${x.seconds}`).join(' ')}`;
});

await check('memory: nothing but stretches was ever fetched (no whole copy), at most 4 stretches are held, and the renderer\'s memory is far below the 921 MB the two whole copies would take', async () => {
  const whole = kept.filter((u) => !u.includes('?'));
  if (whole.length) throw new Error('whole copies fetched: ' + whole.join(' '));
  const held = Number(await attr('data-held'));
  if (held > 4) throw new Error(`${held} held`);
  const now = await rendererMb();
  const grew = Math.max(now, peak) - baseline;
  if (!(grew < 250)) throw new Error(`the renderer grew by ${grew.toFixed(0)} MB (${baseline.toFixed(0)} → ${Math.max(now, peak).toFixed(0)})`);
  return `${held} stretches held at the end; renderer ${baseline.toFixed(0)} MB before, ${peak.toFixed(0)} at the most while playing, ${now.toFixed(0)} now (+${grew.toFixed(0)} at the most); the two whole copies would be ${(SECONDS * SR * 4 * 2 / 1048576).toFixed(0)} MB`;
});

let rerun;
await check('saving another strength with nothing after the noise removal is a free re-run on our side at this length: "Save this strength · Free", a new job opens, its file comes down by itself', async () => {
  await card().getByTestId('strength-set-balanced').click();
  const save = card().getByTestId('blend-save');
  await save.waitFor({ timeout: 10000 });
  await page.waitForFunction(() => /Free$/.test((document.querySelector('[data-testid=blend-save]')?.textContent || '').trim()), null, { timeout: 15000 });
  const label = (await save.innerText()).replace(/\s+/g, ' ').trim();
  if (label !== 'Save this strength · Free') throw new Error('label: ' + label);
  if (!/Put together on our side, free/.test(await card().innerText())) throw new Error('no words about where it is made');
  got.length = 0;
  const [req] = await Promise.all([page.waitForRequest((r) => r.url().endsWith(`/api/jobs/${JOB}/rerun`) && r.method() === 'POST'), save.click()]);
  const body = req.postDataJSON();
  if (JSON.stringify(body.options) !== '{"atten_lim_db":24}' || body.denoiseTier || body.transcribeTier) throw new Error('sent: ' + JSON.stringify(body));
  await page.waitForURL((u) => JOB_PATH.test(u.pathname) && !u.pathname.endsWith(JOB), { timeout: 30000 });
  rerun = jobIdOf(page);
  const done = await waitJobDone(page, rerun, { timeout: 900000 });
  if (done.status !== 'done' || done.costMicros !== 0 || done.options.atten_lim_db !== 24 || done.sourceJobId !== JOB) throw new Error(`re-run: ${done.status} cost ${done.costMicros} atten ${done.options.atten_lim_db}`);
  const t0 = Date.now();
  while (!got.length && Date.now() - t0 < 120000) await sleep(300);
  if (got.length !== 1 || !/\.flac$/.test(got[0].name)) throw new Error('downloaded by itself: ' + got.map((g) => g.name));
  await got[0].download.saveAs(join(WORK, 'balanced.flac'));
  if (kept.some((u) => !u.includes('?'))) throw new Error('a whole copy was fetched to save');
  return `${label}; ${got[0].name} (${(statSync(join(WORK, 'balanced.flac')).size / 1e6).toFixed(1)} MB), cost 0`;
});

await check('what comes down is the blend of the two copies at that strength: a window of the file (at 25:07, 10 s) equals (1 - a) * noise + a * input from the two stretches, to the 16-bit file\'s precision', async () => {
  const a = 10 ** (-24 / 20);
  const windowOf = async (which) => {
    const res = await context.request.get(`${BASE}/api/jobs/${JOB}/kept/${which}?start=1507.000000&seconds=10.000000`);
    const p = join(WORK, `${which}.flac`);
    writeFileSync(p, await res.body());
    return decode(p);
  };
  const full = await windowOf('noise');
  const input = await windowOf('input');
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
  // and the slider is on the new job's page too (the copies were carried over)
  await card().getByTestId('listen').locator('summary').click();
  await blend().waitFor({ timeout: 60000 });
  if ((await attr('data-db')) !== '24.00') throw new Error('starts at ' + await attr('data-db'));
  return `rms ${rms.toFixed(4)} (${apart.toFixed(4)} away from the full-strength sound), worst difference from the blend ${worst.toExponential(1)}; the new job's slider starts at Balanced`;
});

rmSync(WORK, { recursive: true, force: true });
try { for (const f of readdirSync(WORK)) rmSync(join(WORK, f), { force: true }); } catch { /* removed */ }
console.log('LOGS:\n' + logs.filter((l) => !/Failed to load resource|favicon|net::ERR/.test(l)).join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
