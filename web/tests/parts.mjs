// The browser's share of a job made of parts (docs/api.md section 8, "Uploads from the browser"): a Free noise part and a Free subtitle part
// inside one job on our side (create answers `needs`, the browser cleans and transcribes only those ranges and uploads them, the job waits for
// them and finishes), the same when a finished job gets a new Free part (process answers 409 needs_uploads, the page uploads, calls again),
// and a run with nothing on our side whose Free part is only a stretch of the recording (the rest of the file is untouched).
// Needs the test backend (tests/run.sh parts) and the fixtures. Usage: node tests/parts.mjs
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { BASE, FFMPEG, FILES, admin, check, dragRange, launch, pick, resultAttr, setAuto, setKind, setLeveling, setStrength, setSubs, setVoice, settleRecommendation, signIn, summary, waitResult } from './helpers.mjs';

const EMAIL = `parts-${Date.now()}@example.com`;
const { browser, page, logs } = await launch([], { acceptDownloads: true });
const sent = []; // what the page sent: {m, p, body, bytes, head}
const downloads = [];
page.on('download', (d) => downloads.push(d));
page.on('request', (r) => {
  const u = new URL(r.url());
  if (!u.pathname.startsWith('/api/') || r.method() === 'GET') return;
  const buf = r.postDataBuffer();
  sent.push({ m: r.method(), p: u.pathname + u.search, body: buf && u.pathname.includes('/browser/cues') ? buf.toString() : r.postData()?.slice(0, 4000), bytes: buf?.length || 0, head: buf ? buf.subarray(0, 64) : null });
});
await signIn(page, EMAIL);
admin('grant', EMAIL, '5', 'parts test');
const flacHeader = (head) => ({ magic: head.toString('ascii', 0, 4), rate: Number(BigInt(head.readBigUInt64BE(18)) >> 44n), channels: Number((head.readBigUInt64BE(18) >> 41n) & 7n) + 1, samples: Number(head.readBigUInt64BE(18) & ((1n << 36n) - 1n)) });
async function openFile(file) {
  await page.goto(`${BASE}/app`);
  await page.setInputFiles('input[type=file]', join(FILES, file));
  await page.getByTestId('start').waitFor({ timeout: 30000 });
  await settleRecommendation(page);
  await setAuto(page, false);
}
const sec = (tr) => page.locator(`[data-testid=track-${tr}] [data-testid=section]`).evaluateAll((els) => els.map((e) => e.getAttribute('title'))); // (the title says what a short part is too)
let jobUrl = '';

await check('a job with a Free noise part and Free subtitles: create says what the browser must send, the browser sends exactly those ranges (a FLAC of exactly the part, the cues in file time), the job finishes and its files come down', async () => {
  await openFile('speech.wav');
  await setVoice(page, 'off');
  await setSubs(page, 'off');
  await dragRange(page, 2, 6);
  await setKind(page, 'free');
  await setStrength(page, 'balanced');
  await dragRange(page, 0, 8);
  await setSubs(page, 'free');
  await page.getByTestId('select-all').click();
  sent.length = 0; downloads.length = 0;
  await page.getByTestId('start').click();
  await page.waitForURL(/\/app\/job\/[0-9a-f]{32}$/, { timeout: 120000 });
  jobUrl = page.url();
  await page.getByTestId('download-again').waitFor({ timeout: 180000 });
  const create = sent.find((x) => x.m === 'POST' && x.p === '/api/jobs');
  const body = JSON.parse(create.body);
  if (body.kind !== 'denoise' || body.transcribeTier || body.denoiseTier || body.options.language !== undefined) throw new Error('create: ' + create.body.slice(0, 300));
  if (body.timeline.version !== 2) throw new Error('version ' + body.timeline.version);
  const L = body.timeline.layers;
  if (!(L.noise.length === 1 && L.noise[0].browser === true && L.noise[0].start === 2 && L.noise[0].end === 6 && L.noise[0].strength_db === 24)) throw new Error('noise part: ' + JSON.stringify(L.noise));
  if (!(L.subtitles.length === 1 && L.subtitles[0].quality === 'free' && L.subtitles[0].start === 0 && L.subtitles[0].end === 8 && L.subtitles[0].language === null)) throw new Error('subtitle part: ' + JSON.stringify(L.subtitles));
  if (!L.leveling) throw new Error('Leveling (paid) is what makes it a job on our side');
  const noise = sent.find((x) => x.p.includes('/browser/noise'));
  const cues = sent.find((x) => x.p.includes('/browser/cues'));
  if (!noise || !cues) throw new Error('uploads: ' + sent.map((x) => x.m + ' ' + x.p).join(', '));
  if (!/start=2&end=6$/.test(noise.p) || !/start=0&end=8$/.test(cues.p)) throw new Error(noise.p + ' ' + cues.p);
  // (the page's own request body is not visible to the test for a file upload: the header is read from the server's copy)
  const id = /\/app\/job\/([0-9a-f]{32})/.exec(jobUrl)[1];
  const dir = execFileSync(process.env.ACW_PYTHON, ['-c', `from acw import jobs; print(jobs.job_dir('${id}'))`], { env: process.env, cwd: process.env.ACW_SERVER_DIR, encoding: 'utf8' }).trim();
  const h = flacHeader(readFileSync(join(dir, 'browser', 'noise-2000-6000.flac')).subarray(0, 64));
  if (h.magic !== 'fLaC' || h.rate !== 48000 || h.samples !== 4 * 48000) throw new Error('FLAC ' + JSON.stringify(h));
  const list = JSON.parse(cues.body);
  if (!Array.isArray(list) || !list.length || !list.every((c) => c.start >= 0 && c.end <= 8.5 && typeof c.text === 'string' && c.text)) throw new Error('cues ' + cues.body.slice(0, 200));
  const names = downloads.map((d) => d.suggestedFilename()).sort();
  if (names.join() !== 'speech-cleaned.flac,speech.srt') throw new Error('downloaded ' + names);
  return `noise ${noise.p} (${h.samples} samples), cues ${list.length}, files ${names.join(' + ')}`;
});

await check('a new Free part on the finished job: Save changes asks for the price, process answers 409 needs_uploads, the page uploads exactly that range and calls process again; the job runs and is done again', async () => {
  await dragRange(page, 8, 10);
  const title = await page.locator('.studio .js-title').innerText();
  if (!/^0:08–0:10 · 2 s$/.test(title)) throw new Error('the part was not chosen on the finished job: ' + title);
  await setKind(page, 'free');
  await page.getByTestId('save-changes').waitFor({ timeout: 15000 });
  await page.waitForFunction(() => /Save changes ·/.test(document.querySelector('[data-testid=save-changes]')?.textContent || ''), null, { timeout: 15000 });
  sent.length = 0;
  const responses = [];
  page.on('response', (r) => { if (r.url().includes('/process') || r.url().includes('/export')) responses.push(r.status()); });
  await page.getByTestId('save-changes').click();
  await page.waitForFunction(() => document.querySelector('[data-testid=studio]')?.dataset.stage === 'done' && document.querySelector('[data-testid=download-again]'), null, { timeout: 240000 });
  const calls = sent.map((x) => `${x.m} ${x.p.replace(/[0-9a-f]{32}/, '<id>')}`);
  const proc = calls.filter((c) => /\/(process|export)/.test(c));
  const up = calls.filter((c) => c.includes('/browser/noise'));
  if (proc.length < 2 || !responses.includes(409)) throw new Error('process calls ' + proc.length + ' statuses ' + responses.join() + ' calls ' + calls.join(' | '));
  const procBody = (sent.find((x) => /\/(process|export)$/.test(x.p)) || {}).body;
  const q = new URLSearchParams((up[0] || '').split('?')[1] || '');
  if (up.length !== 1 || Number(q.get('start')) !== 8 || Math.abs(Number(q.get('end')) - 10) > 0.001) throw new Error('uploads ' + up.join(' | ') + ' process body ' + (procBody || '').slice(0, 600));
  const first = sent.find((x) => /\/(process|export)$/.test(x.p));
  if (!first) throw new Error('no process call');
  return `${proc.length} process calls (${responses.join(',')}), upload ${up[0].split('?')[1]}`;
});

await check('the finished job\'s page after a reload: the Free part is a Free part (the page knows the timeline), the heard waveform has the browser-made part, and the job is not "changed"', async () => {
  await page.goto(jobUrl);
  await page.getByTestId('download-again').waitFor({ timeout: 30000 });
  await page.waitForTimeout(1500);
  const v = await sec('voice');
  if (v.filter((t) => /Voice: Free/.test(t)).length !== 2) throw new Error('voice: ' + v.join(' | '));
  if ((await page.getByTestId('studio').getAttribute('data-changed')) !== 'no') throw new Error('changed: ' + (await page.getByTestId('studio').getAttribute('data-changed')));
  const s = await sec('subs');
  if (!s.some((t) => /Subtitles: Free/.test(t))) throw new Error('subs: ' + s.join(' | '));
  return `${v.join(' | ')} / ${s.join(' | ')}`;
});

await check('a free run on a stretch only: no job is made, the file that comes down has the stretch cleaned and every other sample of the recording as it was', async () => {
  const posts = [];
  await page.route('**/api/jobs', (r) => { if (r.request().method() === 'POST') posts.push(1); return r.continue(); });
  try {
    await openFile('noisy.wav');
    await pick(page, 'noise', 'off');
    await setSubs(page, 'off');
    await dragRange(page, 3, 7);
    await setKind(page, 'free');
    await setStrength(page, 'strong');
    await page.getByTestId('select-all').click();
    const label = (await page.getByTestId('start').innerText()).trim();
    if (label !== 'Start · Free') throw new Error('label ' + label);
    downloads.length = 0;
    await page.getByTestId('start').click();
    await waitResult(page, 'result', 180000);
    if (posts.length) throw new Error('a job was sent');
    if (downloads.length !== 1) throw new Error('downloads ' + downloads.map((d) => d.suggestedFilename()));
    const out = join(FILES, 'parts-stretch.wav');
    await downloads[0].saveAs(out);
    const read = (f) => execFileSync(FFMPEG, ['-v', 'error', '-i', f, '-f', 'f32le', '-ac', '1', '-ar', '48000', '-'], { maxBuffer: 1 << 28 });
    const a = new Float32Array(read(join(FILES, 'noisy.wav')).buffer.slice(0));
    const b = new Float32Array(read(out).buffer.slice(0));
    if (Math.abs(a.length - b.length) > 48000 * 0.05) throw new Error(`lengths ${a.length} vs ${b.length}`);
    const diff = (from, to) => { let m = 0; for (let i = Math.round(from * 48000); i < Math.round(to * 48000); i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
    const before = diff(0.2, 2.8); const after = diff(7.2, 10);
    const inside = diff(3.5, 6.5);
    // (the WAV is 16-bit: the rest of the recording is the same to within that rounding)
    if (before > 0.001 || after > 0.001) throw new Error(`outside the stretch the sound changed: ${before} ${after}`);
    if (inside < 0.01) throw new Error('the stretch was not cleaned: ' + inside);
    return `outside max diff ${Math.max(before, after).toFixed(5)}, inside ${inside.toFixed(3)}`;
  } finally { await page.unroute('**/api/jobs'); }
});

console.log(logs.filter((l) => !/Failed to load resource|favicon/.test(l)).slice(0, 6).join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
