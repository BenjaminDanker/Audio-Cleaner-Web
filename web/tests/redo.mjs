// "Change and save again" and the strength slider (Result panel of a job's page):
//  1. a re-run on our side: the paid choices of the panel, the price on the line and on Save, the new job's own page, its files by
//     themselves, one line in the billing history; with Leveling off the same save is free, made on this device, no job;
//     a change that runs on our side but costs nothing is free too;
//  2. the file saved on this device equals what our side makes for the same choice (decoded to float, the difference is under -60 dB);
//  3. the slider while listening: the player is there at once and plays from short stretches of the two copies (never a whole copy), the gains
//     of the two copies against 10^(-dB/20), the names, in-between values, Original, Play / Pause, and the note that says whether what you
//     hear is what you save; the panel and the slider share one strength;
//  4. the free noise removal in the browser has the slider at once and saves other strengths with no job (mono and stereo);
//  5. a Voice rebuild job has no slider;
//  6. the kept copies: audio/flac for the owner only, and the redo block of the job says how to read them.
// Needs the test backend (tests/run.sh redo) and the fixtures. Usage: node tests/redo.mjs
import { execFileSync } from 'node:child_process';
import { readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { BASE, FFMPEG, FILES, JOB_PATH, admin, apiCall, check, dropFile, jobIdOf, launch, listJobs, pick, signIn, skip, summary, waitCardDone, waitResult } from './helpers.mjs';

const EMAIL = `redo-${Date.now()}@example.com`;
const TAG = Date.now().toString(36);
const FFPROBE = process.env.FFPROBE || join(process.env.HOME, 'audio-cleaner', 'bin', 'ffprobe');

const { browser, context, page, logs } = await launch();
const got = []; // [{name, download}] in order: what the page downloaded
page.on('download', (d) => got.push({ name: d.suggestedFilename(), download: d }));
const requests = [];
const keptAsked = []; // the kept copies asked for, with their query: [path?start=&seconds=]
page.on('request', (r) => {
  const u = new URL(r.url());
  if (u.pathname.startsWith('/api/')) requests.push(`${r.method()} ${u.pathname}`);
  if (/\/kept\/(noise|input)$/.test(u.pathname)) keptAsked.push(u.pathname + u.search);
});
await signIn(page, EMAIL);
admin('grant', EMAIL, '5', 'redo test');

const sleep = (ms) => page.waitForTimeout(ms);
const card = () => page.getByTestId('job').first();
const gotoApp = async (path = '/app') => { await page.goto(`${BASE}${path}`); await page.getByTestId('available').or(page.getByTestId('job-page')).first().waitFor({ timeout: 20000 }); };
const waitDownloads = async (n, timeout = 60000) => {
  const t0 = Date.now();
  while (got.length < n && Date.now() - t0 < timeout) await sleep(250);
  await sleep(800); // anything extra would show up now
};
const near = (a, b, tol, what) => { if (!(Math.abs(a - b) <= tol)) throw new Error(`${what}: ${a} vs ${b}`); };
const countBy = (re) => requests.filter((r) => re.test(r)).length;
const pricing = await (await fetch(`${BASE}/api/billing/pricing`)).json();

/** Float samples of an audio file, as ffmpeg decodes them (mono mixdown not applied: channels kept, 48 kHz). */
function decode(path, channels) {
  const buf = execFileSync(FFMPEG, ['-v', 'error', '-i', path, '-ar', '48000', '-ac', String(channels), '-f', 'f32le', '-acodec', 'pcm_f32le', '-'], { maxBuffer: 1 << 29 });
  return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
}
const rms = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / Math.max(1, a.length));
/** Signal to difference ratio in dB, over the shorter of the two. */
function snr(a, b) {
  const n = Math.min(a.length, b.length);
  let s = 0;
  let d = 0;
  for (let i = 0; i < n; i++) { s += a[i] * a[i]; const e = a[i] - b[i]; d += e * e; }
  return d === 0 ? 200 : 10 * Math.log10(s / d);
}
const probe = (p) => JSON.parse(execFileSync(FFPROBE, ['-v', 'error', '-show_streams', '-show_format', '-of', 'json', p], { encoding: 'utf8' }));

/** Starts a paid Denoising job from the form (every choice that matters is set), waits for its result and its automatic download. Answers the job's id. */
async function startJob({ file = 'noisy.wav', strength = 'max', leveling = true } = {}) {
  await gotoApp('/app');
  await dropFile(page, join(FILES, file));
  await pick(page, 'noise', 'denoise');
  await page.locator(`label:has(input[name=strength][value=${strength}])`).click();
  await page.locator('input[name=leveling]').setChecked(leveling);
  await page.getByTestId('estimate').waitFor();
  got.length = 0;
  await page.getByTestId('start').click();
  await page.waitForURL(JOB_PATH, { timeout: 90000 });
  const id = jobIdOf(page);
  await waitCardDone(page);
  await card().getByTestId('result').waitFor({ timeout: 90000 });
  await waitDownloads(1);
  const j = (await apiCall(page, 'GET', `jobs/${id}`)).body;
  const want = { atten_lim_db: { max: 100, strong: 40, balanced: 24, gentle: 12 }[strength], leveling };
  for (const [k, v] of Object.entries(want)) if (j.options[k] !== v) throw new Error(`the job was made with ${k}=${j.options[k]}, wanted ${v}`);
  return id;
}
const openAgain = async () => {
  if (!(await card().getByTestId('again').evaluate((e) => e.open))) await card().getByTestId('again').locator('summary').click();
  await card().getByTestId('again-fields').waitFor();
};
const openListen = async () => {
  if (!(await card().getByTestId('listen').evaluate((e) => e.open))) await card().getByTestId('listen').locator('summary').click();
};
const panel = () => card().getByTestId('again-fields');
const price = () => panel().getByTestId('again-price');
const strengthInPanel = (id, name) => card().getByTestId(`strength-again-${id}`).getByText(name, { exact: true }).click();
const setBox = (testid, on) => panel().getByTestId(testid).setChecked(on);
/** The price line has settled (the estimate for a run on our side takes a moment). */
const settled = async () => { await page.waitForFunction(() => { const p = document.querySelector('[data-testid=again-fields] [data-testid=again-price]'); return p && !['loading', 'idle'].includes(p.getAttribute('data-state')); }, null, { timeout: 15000 }); await sleep(150); };

let J1; // made with Leveling, Maximum
let J2; // made without Leveling, Maximum

// ---------------------------------------------------------------------------------------------------- 1. changing a job again

await check('setup: two paid Denoising jobs (J1 with Leveling, J2 without), both at Maximum, each on its own page with its file downloaded by itself', async () => {
  J1 = await startJob({ leveling: true });
  const first = got.map((g) => g.name);
  J2 = await startJob({ leveling: false });
  if (J1 === J2) throw new Error('same job twice');
  const redo = (await apiCall(page, 'GET', `jobs/${J2}`)).body.redo;
  if (!redo?.available || !redo.preview) throw new Error('J2 has no redo preview: ' + JSON.stringify(redo));
  return `${J1.slice(0, 8)}… (${first}), ${J2.slice(0, 8)}… (${got.map((g) => g.name)})`;
});

let rerunBody = null;
let rerunJob = null;
let rerunNote = null; // the one-time note on the new job's page (null: none was shown)
let freeNote = null;
let rerunNoteInHistory = null; // the note the Save handed to the new page, as it still sits in the history entry
await check('re-run on our side: a new strength with Leveling still on is paid (the price line says so, Save shows it), Save opens the NEW job\'s page and sends only what changed', async () => {
  await gotoApp(`/app/job/${J1}`);
  await card().getByTestId('download-again').waitFor({ timeout: 20000 });
  await openAgain();
  for (const id of ['again-leveling', 'again-loudness', 'again-subs']) if (!(await panel().getByTestId(id).count())) throw new Error(`${id} is not in the panel`);
  if ((await panel().getByTestId('again-rebuild').count()) !== (pricing.rebuildAvailable ? 1 : 0)) throw new Error(`the Voice rebuild box ${pricing.rebuildAvailable ? 'is missing' : 'is offered although the server does not offer it'}`);
  if (!(await panel().getByTestId('again-leveling').isChecked())) throw new Error('the panel does not start with what the job was made with (Leveling on)');
  if (await panel().getByTestId('again-highs').count() || /highs/i.test(await panel().innerText())) throw new Error('Restore highs is in the panel (it is not offered any more)');
  if (!/Nothing paid is changed/.test(await price().innerText()) || (await price().getAttribute('data-state')) === 'paid') throw new Error('unchanged panel: ' + await price().innerText());
  await strengthInPanel(J1, 'Strong');
  await settled();
  if ((await price().getAttribute('data-state')) !== 'paid') throw new Error('price state ' + await price().getAttribute('data-state'));
  const line = (await price().innerText()).replace(/\s+/g, ' ');
  if (!/^Costs about \$0\.000\d/.test(line) || !/Leveling/.test(line) || /Denoising/.test(line)) throw new Error('price line (only Leveling runs again): ' + line);
  const label = (await panel().getByTestId('again-save').innerText()).replace(/\s+/g, ' ').trim();
  if (!/^Save · about \$0\.000\d$/.test(label)) throw new Error('Save label: ' + label);
  got.length = 0;
  const [req] = await Promise.all([page.waitForRequest((r) => r.url().endsWith(`/api/jobs/${J1}/rerun`) && r.method() === 'POST'), panel().getByTestId('again-save').click()]);
  rerunBody = req.postDataJSON();
  await page.waitForURL((u) => JOB_PATH.test(u.pathname) && !u.pathname.endsWith(J1), { timeout: 30000 });
  rerunJob = jobIdOf(page);
  rerunNote = await page.getByTestId('submitted').innerText({ timeout: 4000 }).then((t) => t.trim(), () => null);
  rerunNoteInHistory = await page.evaluate(() => history.state?.usr?.submitted || null);
  if (JSON.stringify(rerunBody.options) !== '{"atten_lim_db":40}' || rerunBody.denoiseTier || rerunBody.transcribeTier) throw new Error('sent: ' + JSON.stringify(rerunBody));
  if (!(rerunBody.maxCostMicros > 0)) throw new Error('the quoted price was not sent along: ' + JSON.stringify(rerunBody));
  await card().waitFor();
  if ((await card().getAttribute('data-job-id')) !== rerunJob) throw new Error('the page still shows the first job');
  return `${line}; "${label}"; sent ${JSON.stringify(rerunBody)}`;
});

await check('the new job\'s page says once what was started ("Got it. This costs about $0.0002; your files download by themselves when it\u2019s done.") and not again after a reload', async () => {
  if (rerunNote === null) throw new Error(`no note on the new job's page after Save${rerunNoteInHistory ? ` (the page was opened with the note "${rerunNoteInHistory}" in its history entry, but never showed it, and left it there for a reload to show)` : ''}`);
  if (!/^Got it\. This costs about \$0\.000\d; your files download by themselves when it’s done\.$/.test(rerunNote)) throw new Error('note: ' + rerunNote);
  return rerunNote;
});

await check('the new job runs and ends on its own page; its file comes down by itself, once; the first job is untouched; the billing history has one line "Changed and saved again (Leveling)"', async () => {
  await waitCardDone(page);
  await card().getByTestId('result').waitFor({ timeout: 90000 });
  await waitDownloads(1);
  if (got.length !== 1 || !/\.flac$/.test(got[0].name)) throw new Error('downloaded by itself: ' + got.map((g) => g.name));
  const done = (await apiCall(page, 'GET', `jobs/${rerunJob}`)).body;
  if (done.options.atten_lim_db !== 40 || done.options.leveling !== true || done.sourceJobId !== J1) throw new Error(JSON.stringify({ a: done.options.atten_lim_db, l: done.options.leveling, s: done.sourceJobId }));
  const first = (await apiCall(page, 'GET', `jobs/${J1}`)).body;
  if (first.status !== 'done' || first.options.atten_lim_db !== 100) throw new Error('the first job changed');
  await gotoApp('/app/billing');
  await page.getByTestId('ledger').waitFor();
  const rows = (await page.locator('[data-testid=ledger] tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
  const mine = rows.filter((t) => /Changed and saved again \(Leveling\)/.test(t));
  if (mine.length !== 1) throw new Error(`${mine.length} lines for the change:\n${rows.join('\n')}`);
  if (/Changed and saved again \(Denoising/.test(rows.join('\n'))) throw new Error('the re-run was charged for Denoising too');
  return mine[0];
});

await check('Leveling switched off: the price line says Free and "Made on this device"; Save downloads the new strength at once with no job and no re-run request', async () => {
  await gotoApp(`/app/job/${J1}`);
  await card().getByTestId('download-again').waitFor({ timeout: 20000 });
  await openAgain();
  await strengthInPanel(J1, 'Balanced');
  await setBox('again-leveling', false);
  await settled();
  if ((await price().getAttribute('data-state')) !== 'free') throw new Error('price state ' + await price().getAttribute('data-state'));
  const line = (await price().innerText()).replace(/\s+/g, ' ');
  if (!/^Free /.test(line) || !/Made on this device/.test(line)) throw new Error('price line: ' + line);
  const label = (await panel().getByTestId('again-save').innerText()).replace(/\s+/g, ' ').trim();
  if (label !== 'Save') throw new Error('Save label: ' + label);
  const jobsBefore = (await listJobs(page)).jobs.length;
  const reruns = countBy(/rerun$/);
  const posts = countBy(/^POST \/api\/jobs$/);
  const url = page.url();
  got.length = 0;
  await panel().getByTestId('again-save').click();
  await waitDownloads(1, 90000);
  if (got.length !== 1 || !/^noisy.*\.flac$/.test(got[0].name)) throw new Error('downloaded: ' + got.map((g) => g.name));
  await got[0].download.saveAs(join(FILES, `redo-${TAG}-j1-balanced.flac`));
  if (page.url() !== url) throw new Error('the page moved to ' + page.url());
  if (countBy(/rerun$/) !== reruns || countBy(/^POST \/api\/jobs$/) !== posts) throw new Error('a job was requested');
  if ((await listJobs(page)).jobs.length !== jobsBefore) throw new Error('a job was created');
  if ((await card().getAttribute('data-job-id')) !== J1) throw new Error('another job');
  const kept = countBy(/\/kept\/(noise|input)$/);
  if (kept !== 2) throw new Error(`the two kept copies were fetched ${kept} times`);
  const j = (await apiCall(page, 'GET', `jobs/${J1}`)).body;
  if (j.status !== 'done') throw new Error('the job changed');
  const st = probe(join(FILES, `redo-${TAG}-j1-balanced.flac`)).streams[0];
  if (st.codec_name !== 'flac') throw new Error(`file is ${st.codec_name}`);
  return `${line}; ${got[0].name}, ${st.sample_rate} Hz ${st.channels} ch, no job, kept copies fetched once each`;
});

let freeRerun = null;
await check('a run on our side that costs nothing (Leveling switched off on a page that has no kept copies to blend here): the line says Free, Save shows "Free", a NEW job opens, its file comes down, and the billing history gets no new line', async () => {
  // This page is not given the two kept copies (the job's redo preview is left out of the list), so changes are made on our side
  // even when nothing is paid for. The server is real: it quotes and runs the re-run.
  await page.route('**/api/jobs', async (route) => {
    if (route.request().method() !== 'GET') return route.continue();
    const res = await route.fetch();
    const body = await res.json();
    await route.fulfill({ response: res, json: { ...body, jobs: body.jobs.map((j) => (j.id === J1 ? { ...j, redo: { ...j.redo, preview: undefined } } : j)) } });
  });
  try {
    await gotoApp(`/app/job/${J1}`);
    await card().getByTestId('download-again').waitFor({ timeout: 20000 });
    await openAgain();
    if (await card().getByTestId(`strength-again-${J1}`).count()) throw new Error('the strength is offered without the kept copies');
    await setBox('again-leveling', false);
    await settled();
    if ((await price().getAttribute('data-state')) !== 'free') throw new Error('price state ' + await price().getAttribute('data-state'));
    const line = (await price().innerText()).replace(/\s+/g, ' ');
    if (!/^Free /.test(line) || /Made on this device/.test(line) || !/Only steps that cost nothing run again/.test(line)) throw new Error('price line: ' + line);
    const label = (await panel().getByTestId('again-save').innerText()).replace(/\s+/g, ' ').trim();
    if (label !== 'Save · Free') throw new Error('Save label: ' + label);
    const before = (await listJobs(page)).jobs.length;
    got.length = 0;
    const [req] = await Promise.all([page.waitForRequest((r) => r.url().endsWith(`/api/jobs/${J1}/rerun`) && r.method() === 'POST'), panel().getByTestId('again-save').click()]);
    await page.waitForURL((u) => JOB_PATH.test(u.pathname) && !u.pathname.endsWith(J1), { timeout: 30000 });
    freeRerun = jobIdOf(page);
    // (this page's strength is the one saved here earlier, Balanced: that is what the re-run starts from)
    if (JSON.stringify(req.postDataJSON().options) !== '{"atten_lim_db":24,"leveling":false}') throw new Error('sent: ' + JSON.stringify(req.postDataJSON()));
    freeNote = await page.getByTestId('submitted').innerText({ timeout: 4000 }).then((t) => t.trim(), () => null);
    await waitCardDone(page);
    await card().getByTestId('result').waitFor({ timeout: 90000 });
    await waitDownloads(1);
    if (got.length !== 1) throw new Error('downloaded by itself: ' + got.map((g) => g.name));
    if ((await listJobs(page)).jobs.length !== before + 1) throw new Error('the job list did not grow by one');
    const j = (await apiCall(page, 'GET', `jobs/${freeRerun}`)).body;
    if (j.costMicros !== 0 || j.options.leveling !== false) throw new Error(`cost ${j.costMicros}, leveling ${j.options.leveling}`);
    await page.unroute('**/api/jobs');
    await gotoApp('/app/billing');
    await page.getByTestId('ledger').waitFor();
    const rows = (await page.locator('[data-testid=ledger] tbody tr').allInnerTexts()).filter((t) => /Changed and saved again/.test(t));
    if (rows.length !== 1) throw new Error(`${rows.length} re-run lines (only the paid one belongs there)`);
    return `${line}; "${label}"`;
  } finally {
    await page.unroute('**/api/jobs').catch(() => {});
  }
});

await check('the free re-run\'s page says so once: "Got it. This one is free; your files download by themselves when it\u2019s done."', async () => {
  if (freeNote === null) throw new Error('no note on the new job\'s page after Save');
  if (freeNote !== 'Got it. This one is free; your files download by themselves when it’s done.') throw new Error('note: ' + freeNote);
  return freeNote;
});

let levelingNote = null;
let levelingJob = null;
await check('Leveling turned on in a re-run of a job made without it is a paid step by itself: the line shows "Leveling" (the 3.60 an hour price, per second rounded up), Save shows the price, the new job costs it, and the billing history has a line for it', async () => {
  await gotoApp(`/app/job/${J2}`);
  await card().getByTestId('download-again').waitFor({ timeout: 20000 });
  await openAgain();
  if (await panel().getByTestId('again-leveling').isChecked()) throw new Error('J2 should start with Leveling off');
  const dur = (await apiCall(page, 'GET', `jobs/${J2}`)).body.durationSeconds;
  const micros = Math.ceil((1000 * Math.ceil(dur)) / 60); // Leveling: 0.001 USD a minute, per second rounded up
  const usd = `$${(micros / 1e6).toFixed(4)}`;
  await setBox('again-leveling', true);
  await settled();
  if ((await price().getAttribute('data-state')) !== 'paid') throw new Error('price state ' + await price().getAttribute('data-state'));
  const line = (await price().innerText()).replace(/\s+/g, ' ');
  if (!line.startsWith(`Costs about ${usd}`) || !line.includes(`Leveling ${usd}`) || /Denoising/.test(line)) throw new Error(`price line "${line}", wanted Leveling ${usd} alone`);
  const label = (await panel().getByTestId('again-save').innerText()).replace(/\s+/g, ' ').trim();
  if (label !== `Save · about ${usd}`) throw new Error('Save label: ' + label);
  got.length = 0;
  const [req] = await Promise.all([page.waitForRequest((r) => r.url().endsWith(`/api/jobs/${J2}/rerun`) && r.method() === 'POST'), panel().getByTestId('again-save').click()]);
  await page.waitForURL((u) => JOB_PATH.test(u.pathname) && !u.pathname.endsWith(J2), { timeout: 30000 });
  levelingJob = jobIdOf(page);
  const sent = req.postDataJSON();
  if (JSON.stringify(sent.options) !== '{"leveling":true}' || sent.maxCostMicros !== micros) throw new Error('sent: ' + JSON.stringify(sent));
  levelingNote = await page.getByTestId('submitted').innerText({ timeout: 4000 }).then((t) => t.trim(), () => null);
  await waitCardDone(page);
  await card().getByTestId('result').waitFor({ timeout: 90000 });
  await waitDownloads(1);
  if (got.length !== 1) throw new Error('downloaded by itself: ' + got.map((g) => g.name));
  const j = (await apiCall(page, 'GET', `jobs/${levelingJob}`)).body;
  if (j.costMicros !== micros || j.options.leveling !== true || j.sourceJobId !== J2 || 'restore_highs' in j.options) throw new Error(`cost ${j.costMicros} (wanted ${micros}), leveling ${j.options.leveling}, source ${j.sourceJobId}`);
  await gotoApp('/app/billing');
  await page.getByTestId('ledger').waitFor();
  const rows = (await page.locator('[data-testid=ledger] tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' ')).filter((t) => /Changed and saved again/.test(t));
  const mine = rows.filter((t) => t.includes(`−${usd}`) && /Changed and saved again \(Leveling\)/.test(t));
  if (rows.length !== 2 || mine.length !== 2) throw new Error(`re-run lines (the first re-run, J1's, and this one):\n${rows.join('\n')}`);
  return `${line}; "${label}"; ${mine[0]}`;
});

await check('the note on that new page: "Got it. This costs about $0.0001; your files download by themselves when it’s done."', async () => {
  if (levelingNote === null) throw new Error('no note on the new job\'s page after Save');
  if (!/^Got it\. This costs about \$0\.000\d; your files download by themselves when it’s done\.$/.test(levelingNote)) throw new Error('note: ' + levelingNote);
  return levelingNote;
});

// ---------------------------------------------------------------------------------------------------- 2. the same file from both

await check('what is saved on this device equals what our side makes for the same choice: J2 at Strong, saved here and re-run there, decoded to float, the difference is below -60 dB', async () => {
  await gotoApp(`/app/job/${J2}`);
  await card().getByTestId('download-again').waitFor({ timeout: 20000 });
  await openAgain();
  await strengthInPanel(J2, 'Strong');
  await settled();
  if ((await price().getAttribute('data-state')) !== 'free' || !/Made on this device/.test(await price().innerText())) throw new Error('not a save on this device: ' + await price().innerText());
  got.length = 0;
  await panel().getByTestId('again-save').click();
  await waitDownloads(1, 90000);
  if (got.length !== 1) throw new Error('downloads: ' + got.map((g) => g.name));
  const here = join(FILES, `redo-${TAG}-here.flac`);
  await got[0].download.saveAs(here);
  // our side: the same change as a re-run (free: Leveling is off)
  const r = await apiCall(page, 'POST', `jobs/${J2}/rerun`, { options: { atten_lim_db: 40 } });
  if (r.status !== 200) throw new Error(`re-run ${r.status} ${JSON.stringify(r.body)}`);
  if (!r.body.free) throw new Error('the server wants money for it: ' + JSON.stringify(r.body));
  const done = await (async () => {
    for (let i = 0; i < 160; i++) {
      const j = (await apiCall(page, 'GET', `jobs/${r.body.id}`)).body;
      if (['done', 'failed', 'canceled'].includes(j.status)) return j;
      await sleep(1500);
    }
    throw new Error('the re-run did not finish');
  })();
  if (done.status !== 'done') throw new Error(`re-run ${done.status}: ${done.error}`);
  const audio = done.outputs.find((o) => o.kind === 'audio');
  const res = await context.request.get(`${BASE}/api/jobs/${r.body.id}/outputs/${encodeURIComponent(audio.name)}`);
  if (!res.ok()) throw new Error(`download ${res.status()}`);
  const there = join(FILES, `redo-${TAG}-there.flac`);
  writeFileSync(there, await res.body());
  const a = decode(there, 1);
  const b = decode(here, 1);
  if (Math.abs(a.length - b.length) > 480) throw new Error(`lengths ${a.length} and ${b.length}`);
  if (rms(a) < 0.005) throw new Error('silent file ' + rms(a));
  const ratio = snr(a, b);
  if (!(ratio > 60)) throw new Error(`the two files differ: SNR ${ratio.toFixed(1)} dB (rms ${rms(a).toFixed(4)} vs ${rms(b).toFixed(4)})`);
  return `SNR ${ratio.toFixed(1)} dB over ${Math.min(a.length, b.length)} samples; here ${statSync(here).size} B, there ${statSync(there).size} B`;
});

// ---------------------------------------------------------------------------------------------------- 3. the slider

const A = (db) => (db >= 100 ? 0 : 10 ** (-db / 20));
const blend = () => card().getByTestId('blend-player');
const attr = (n) => blend().getAttribute(n);

await check('slider, a job on our side: opening Listen shows the player at once; it asks for short stretches of the two copies (the first ten seconds of each, and the part after), never a whole copy, and starts at the strength last saved here (else the one the job was made with)', async () => {
  await gotoApp(`/app/job/${J1}`);
  await card().getByTestId('download-again').waitFor({ timeout: 20000 });
  const asked = keptAsked.length;
  await openListen();
  if (await card().getByTestId('blend-load').count()) throw new Error('the page still asks to load the sound');
  const t0 = Date.now();
  await blend().waitFor({ timeout: 120000 });
  await page.waitForFunction(() => Number(document.querySelector('[data-testid=blend-player]')?.getAttribute('data-held')) >= 2, null, { timeout: 30000 });
  const gets = keptAsked.slice(asked);
  if (!gets.length || gets.some((u) => !/\?start=\d+\.\d{6}&seconds=\d+\.\d{6}$/.test(u))) throw new Error('requests: ' + gets);
  for (const which of ['noise', 'input']) if (!gets.includes(`/api/jobs/${J1}/kept/${which}?start=0.000000&seconds=10.000000`)) throw new Error(`no first stretch of ${which}: ${gets}`);
  if ((await attr('data-source')) !== 'server') throw new Error('source ' + await attr('data-source'));
  // the strength of what this browser saved last (Balanced, a few checks above), else the one the job was made with
  if ((await attr('data-db')) !== '24.00' || (await card().getByTestId('strength-name').innerText()) !== 'Balanced') throw new Error(`starts at ${await attr('data-db')} dB`);
  if ((await attr('data-mode')) !== 'cleaned') throw new Error('mode ' + await attr('data-mode'));
  return `ready in ${Date.now() - t0} ms with ${gets.length} stretch requests, at Balanced (the last strength saved here)`;
});

await check('slider, Leveling still on in the panel: data-exact is "no", the note says they are added when you save, the gains keep the ratio a / (1 - a) of the strength and a loudness match g between 0.1 and 3', async () => {
  if ((await attr('data-exact')) !== 'no') throw new Error('exact: ' + await attr('data-exact'));
  const note = (await card().getByTestId('blend-note').innerText()).trim();
  if (note !== 'This is the noise removal only, at an even loudness. Leveling is added when you save.') throw new Error('note: ' + note);
  const seen = [];
  for (const [id, db] of [['balanced', 24], ['gentle', 12], ['strong', 40], ['max', 100]]) {
    await card().getByTestId(`strength-set-${id}`).click();
    await page.waitForFunction((x) => document.querySelector('[data-testid=blend-player]')?.getAttribute('data-db') === x, db.toFixed(2), { timeout: 5000 });
    const a = A(db);
    const full = Number(await attr('data-gain-full'));
    const input = Number(await attr('data-gain-input'));
    const g = full / (1 - a);
    if (!(g >= 0.0999 && g <= 3.0001)) throw new Error(`${db} dB: g ${g}`);
    if (a > 0) near(input / full, a / (1 - a), 2e-3 * (a / (1 - a)) + 1e-4, `${db} dB input/full`);
    near(input, a * g, 2e-5, `${db} dB input gain`);
    near(Number(await attr('data-share')), a, 1e-5, `${db} dB share`);
    seen.push(`${db} dB: g ${g.toFixed(3)}`);
  }
  return seen.join(', ');
});

await check('slider, Leveling off in the panel: data-exact is "yes", the note says what you hear is what you save, and the four names give exactly (1 - a) and a (24 dB: 0.93690 and 0.06310)', async () => {
  await openAgain();
  await setBox('again-leveling', false);
  await page.waitForFunction(() => document.querySelector('[data-testid=blend-player]')?.getAttribute('data-exact') === 'yes', null, { timeout: 5000 });
  const note = (await card().getByTestId('blend-note').innerText()).trim();
  if (note !== 'What you hear is what you save.') throw new Error('note: ' + note);
  const seen = [];
  for (const [id, name, db] of [['max', 'Maximum', 100], ['balanced', 'Balanced', 24], ['gentle', 'Gentle', 12], ['strong', 'Strong', 40]]) {
    await card().getByTestId(`strength-set-${id}`).click();
    await page.waitForFunction((x) => document.querySelector('[data-testid=blend-player]')?.getAttribute('data-db') === x, db.toFixed(2), { timeout: 5000 });
    const a = A(db);
    if ((await attr('data-gain-full')) !== (1 - a).toFixed(5) || (await attr('data-gain-input')) !== a.toFixed(5)) throw new Error(`${db} dB: ${await attr('data-gain-full')} / ${await attr('data-gain-input')} wanted ${(1 - a).toFixed(5)} / ${a.toFixed(5)}`);
    if ((await attr('data-share')) !== a.toFixed(5)) throw new Error(`${db} dB share ${await attr('data-share')}`);
    if ((await card().getByTestId('strength-name').innerText()) !== name) throw new Error(`name ${await card().getByTestId('strength-name').innerText()}`);
    seen.push(`${name} ${(1 - a).toFixed(5)}/${a.toFixed(5)}`);
  }
  if ((await attr('data-gain-full')) !== '0.99000' || (await attr('data-gain-input')) !== '0.01000') throw new Error('Strong is not 0.99 / 0.01');
  return seen.join(', ');
});

await check('slider: dragging the range to a value between two names gives a strength between their dB values, named "Between X and Y", with the gains of that blend; the range runs 0 to 3 over the four names', async () => {
  const slider = card().getByTestId('strength-slider');
  if ((await slider.getAttribute('min')) !== '0' || (await slider.getAttribute('max')) !== '3') throw new Error('range ' + await slider.getAttribute('min') + '..' + await slider.getAttribute('max'));
  const cases = [[0, 12, 'Gentle'], [1, 24, 'Balanced'], [2, 40, 'Strong'], [3, 100, 'Maximum'], [0.5, 18, 'Between Gentle and Balanced'], [1.5, 32, 'Between Balanced and Strong'], [2.5, 70, 'Between Strong and Maximum'], [1.2, 24 + 16 * 0.2, 'Between Balanced and Strong']];
  const seen = [];
  for (const [pos, db, name] of cases) {
    await slider.fill(String(pos));
    await page.waitForFunction((x) => document.querySelector('[data-testid=blend-player]')?.getAttribute('data-db') === x, db.toFixed(2), { timeout: 5000 });
    if ((await card().getByTestId('strength-name').innerText()) !== name) throw new Error(`position ${pos}: name "${await card().getByTestId('strength-name').innerText()}"`);
    const a = A(db);
    if ((await attr('data-gain-full')) !== (1 - a).toFixed(5) || (await attr('data-gain-input')) !== a.toFixed(5)) throw new Error(`position ${pos}: gains ${await attr('data-gain-full')} / ${await attr('data-gain-input')}`);
    seen.push(`${pos}→${db}`);
  }
  await slider.fill('1.5');
  const db = Number(await attr('data-db'));
  if (!(db > 24 && db < 40)) throw new Error('in between: ' + db);
  // the panel follows: none of the four names is ticked, and the panel says where the strength is
  const hint = await card().getByTestId(`again-${J1}-hint`).innerText();
  if (!/Between Balanced and Strong \(32 dB\)/.test(hint)) throw new Error('panel hint: ' + hint);
  if ((await panel().locator(`input[name=again-${J1}]:checked`).count()) !== 0) throw new Error('a name is ticked for an in-between strength');
  return seen.join(' ');
});

await check('slider and panel share the strength: Strong in the panel moves the slider, the slider\'s Gentle ticks Gentle in the panel; the price line follows', async () => {
  await strengthInPanel(J1, 'Strong');
  await page.waitForFunction(() => document.querySelector('[data-testid=blend-player]')?.getAttribute('data-db') === '40.00', null, { timeout: 5000 });
  if (!(await card().getByTestId('strength-slider').inputValue()).startsWith('2')) throw new Error('slider at ' + await card().getByTestId('strength-slider').inputValue());
  await card().getByTestId('strength-set-gentle').click();
  await page.waitForFunction((id) => document.querySelector(`input[name=again-${id}][value=gentle]`)?.checked, J1, { timeout: 5000 });
  await settled();
  if ((await price().getAttribute('data-state')) !== 'free' || !/Made on this device/.test(await price().innerText())) throw new Error('price: ' + await price().innerText());
  // with Leveling back on, the same strength is a paid re-run, and the slider's note and Save say so
  await setBox('again-leveling', true);
  await settled();
  if ((await price().getAttribute('data-state')) !== 'paid') throw new Error('price with Leveling: ' + await price().getAttribute('data-state'));
  if ((await attr('data-exact')) !== 'no') throw new Error('exact stays yes with Leveling on');
  const save = (await card().getByTestId('blend-save').innerText()).replace(/\s+/g, ' ').trim();
  if (!/^Save this strength · about \$0\.000\d$/.test(save)) throw new Error('blend-save: ' + save);
  await setBox('again-leveling', false);
  await settled();
  const free = (await card().getByTestId('blend-save').innerText()).replace(/\s+/g, ' ').trim();
  if (free !== 'Save this strength · Free') throw new Error('blend-save: ' + free);
  return `"${save}" then "${free}"`;
});

await check('slider: Original plays only the recording (gains 0 and 1, exact) and the range is off; Cleaned brings the blend back', async () => {
  await card().getByTestId('strength-set-balanced').click();
  await card().getByTestId('blend-original').locator('xpath=ancestor::label').click();
  await page.waitForFunction(() => document.querySelector('[data-testid=blend-player]')?.getAttribute('data-mode') === 'original', null, { timeout: 5000 });
  if ((await attr('data-gain-full')) !== '0.00000' || (await attr('data-gain-input')) !== '1.00000') throw new Error(`Original gains ${await attr('data-gain-full')} / ${await attr('data-gain-input')}`);
  if (!(await card().getByTestId('strength-slider').isDisabled())) throw new Error('the range is on in Original');
  await card().getByTestId('blend-cleaned').locator('xpath=ancestor::label').click();
  await page.waitForFunction(() => document.querySelector('[data-testid=blend-player]')?.getAttribute('data-mode') === 'cleaned', null, { timeout: 5000 });
  if ((await attr('data-gain-full')) !== (1 - A(24)).toFixed(5)) throw new Error('Cleaned gains ' + await attr('data-gain-full'));
  // choosing a name while on Original goes back to Cleaned
  await card().getByTestId('blend-original').locator('xpath=ancestor::label').click();
  await card().getByTestId('strength-set-strong').click();
  await page.waitForFunction(() => document.querySelector('[data-testid=blend-player]')?.getAttribute('data-mode') === 'cleaned', null, { timeout: 5000 });
  return 'ok';
});

await check('slider: Play starts the sound (the button says Pause and the time moves), Pause stops it (the time stands still), Play goes on from there', async () => {
  const time = () => card().locator('[data-testid=blend-player] .time').innerText().then((t) => t.split('/')[0].trim());
  const secs = (t) => { const [m, s] = t.split(':').map(Number); return m * 60 + s; };
  const play = card().getByTestId('blend-play');
  if ((await play.innerText()) !== 'Play') throw new Error('button ' + await play.innerText());
  await play.click();
  await page.waitForFunction(() => document.querySelector('[data-testid=blend-play]')?.textContent === 'Pause', null, { timeout: 5000 });
  await page.waitForFunction(() => { const t = document.querySelector('[data-testid=blend-player] .time')?.textContent || ''; return !t.startsWith('0:00 '); }, null, { timeout: 8000 });
  const running = secs(await time());
  await play.click();
  if ((await play.innerText()) !== 'Play') throw new Error('button after Pause: ' + await play.innerText());
  await sleep(300);
  const stopped = await time();
  await sleep(1500);
  if ((await time()) !== stopped) throw new Error(`the time moved while paused: ${stopped} -> ${await time()}`);
  await play.click();
  await page.waitForFunction((t) => { const x = document.querySelector('[data-testid=blend-player] .time')?.textContent.split('/')[0].trim(); return x !== t; }, stopped, { timeout: 8000 });
  await play.click();
  if (!(running >= 1)) throw new Error('the time never passed 0:01');
  return `1 s played, paused at ${stopped}, played on`;
});

await check('slider on our side, Save this strength (Leveling off): a file at that strength comes down at once, no job, no re-run request', async () => {
  await card().getByTestId('strength-set-gentle').click();
  await settled();
  const jobsBefore = (await listJobs(page)).jobs.length;
  const reruns = countBy(/rerun$/);
  got.length = 0;
  await card().getByTestId('blend-save').click();
  await waitDownloads(1, 90000);
  if (got.length !== 1) throw new Error('downloads: ' + got.map((g) => g.name));
  const p = join(FILES, `redo-${TAG}-j1-gentle.flac`);
  await got[0].download.saveAs(p);
  if (countBy(/rerun$/) !== reruns || (await listJobs(page)).jobs.length !== jobsBefore) throw new Error('a job was requested');
  // Gentle keeps more of the recording than Balanced did: it is louder than the Balanced file saved earlier
  const balanced = rms(decode(join(FILES, `redo-${TAG}-j1-balanced.flac`), 1));
  const gentle = rms(decode(p, 1));
  if (!(gentle > balanced)) throw new Error(`Gentle ${gentle.toFixed(5)} is not above Balanced ${balanced.toFixed(5)}`);
  return `Gentle rms ${gentle.toFixed(5)} > Balanced ${balanced.toFixed(5)}`;
});

await check('slider on a re-run job: the new job carried the two copies over, so its page has the slider too, at the strength it was made with (Strong)', async () => {
  await gotoApp(`/app/job/${rerunJob}`);
  await card().getByTestId('download-again').waitFor({ timeout: 20000 });
  const redo = (await apiCall(page, 'GET', `jobs/${rerunJob}`)).body.redo;
  if (!redo?.preview) throw new Error('the re-run has no kept copies: ' + JSON.stringify(redo));
  await openListen();
  await blend().waitFor({ timeout: 120000 });
  if ((await attr('data-db')) !== '40.00' || (await card().getByTestId('strength-name').innerText()) !== 'Strong') throw new Error(`starts at ${await attr('data-db')} dB`);
  return 'Strong, from the carried copies';
});

// ---------------------------------------------------------------------------------------------------- 4. the free run in the browser

async function freeRun(file, strengthLabel) {
  await gotoApp('/app');
  await dropFile(page, join(FILES, file));
  await pick(page, 'noise', 'browser');
  await pick(page, 'subs', 'off');
  await page.getByTestId('strength-lstrength').getByText(strengthLabel, { exact: true }).click();
  got.length = 0;
  await page.getByRole('button', { name: 'Start – free' }).click();
  await page.waitForURL(/\/app\/job\/local-[0-9a-z]+$/, { timeout: 180000 });
  await waitResult(page, 'result', 60000);
  await waitDownloads(1);
}
const dataRms = async () => Number(await card().getByTestId('result').getAttribute('data-rms'));

for (const [file, channels, label] of [['noisy.wav', 1, 'mono'], ['noisy24st.wav', 2, 'stereo']]) {
  await check(`free noise removal in the browser (${label}): its page /app/job/local-… has the slider at once; Gentle is saved with no job and is louder than the Maximum one (the page's rms and the files')`, async () => {
    await freeRun(file, 'Maximum');
    if (got.length !== 1) throw new Error('downloaded by itself: ' + got.map((g) => g.name));
    const maxFile = join(FILES, `redo-${TAG}-free-${label}-max.wav`);
    await got[0].download.saveAs(maxFile);
    const maxRms = await dataRms();
    if (!jobIdOf(page).startsWith('local-')) throw new Error('page ' + page.url());
    if ((await card().getByTestId('result').getAttribute('data-channels')) !== String(channels)) throw new Error('channels ' + await card().getByTestId('result').getAttribute('data-channels'));
    await openListen();
    if (await card().getByTestId('blend-load').count()) throw new Error('a free run asks to load the sound');
    await blend().waitFor({ timeout: 30000 });
    if ((await attr('data-exact')) !== 'yes' || (await attr('data-db')) !== '100.00') throw new Error(`exact ${await attr('data-exact')}, db ${await attr('data-db')}`);
    if ((await card().getByTestId('blend-note').innerText()).trim() !== 'What you hear is what you save.') throw new Error('note');
    const jobsBefore = (await listJobs(page)).jobs.length;
    const reqBefore = requests.length;
    await card().getByTestId('strength-set-gentle').click();
    await page.waitForFunction(() => document.querySelector('[data-testid=blend-player]')?.getAttribute('data-db') === '12.00', null, { timeout: 5000 });
    if ((await attr('data-gain-full')) !== (1 - A(12)).toFixed(5) || (await attr('data-gain-input')) !== A(12).toFixed(5)) throw new Error('Gentle gains ' + await attr('data-gain-full'));
    const save = (await card().getByTestId('blend-save').innerText()).replace(/\s+/g, ' ').trim();
    if (save !== 'Save this strength · Free') throw new Error('blend-save: ' + save);
    got.length = 0;
    await card().getByTestId('blend-save').click();
    await waitDownloads(1, 90000);
    if (got.length !== 1) throw new Error('second download: ' + got.map((g) => g.name));
    const gentleFile = join(FILES, `redo-${TAG}-free-${label}-gentle.wav`);
    await got[0].download.saveAs(gentleFile);
    await page.waitForFunction((x) => Number(document.querySelector('[data-testid=result]')?.getAttribute('data-rms')) > x, maxRms, { timeout: 15000 });
    const gentleRms = await dataRms();
    const sent = requests.slice(reqBefore).filter((r) => /rerun|^POST \/api\/jobs/.test(r));
    if (sent.length) throw new Error('requests: ' + sent.join(', '));
    if ((await listJobs(page)).jobs.length !== jobsBefore) throw new Error('a job was created');
    const fileMax = rms(decode(maxFile, channels));
    const fileGentle = rms(decode(gentleFile, channels));
    if (!(fileGentle > fileMax)) throw new Error(`files: Gentle ${fileGentle} is not above Maximum ${fileMax}`);
    near(fileGentle, gentleRms, gentleRms * 0.03, 'the page and the file agree on the loudness');
    const st = probe(gentleFile).streams[0];
    if (st.channels !== channels) throw new Error(`the saved file has ${st.channels} channels`);
    return `rms ${maxRms.toFixed(4)} (Maximum) -> ${gentleRms.toFixed(4)} (Gentle); files ${fileMax.toFixed(4)} -> ${fileGentle.toFixed(4)}, ${st.channels} ch; no job, no request`;
  });
}

// ---------------------------------------------------------------------------------------------------- 5. Voice rebuild

await check('a Voice rebuild job gets no slider (and no strength choice in the panel): its page, as the server describes it with the rebuild tier, offers neither', async () => {
  const real = (await apiCall(page, 'GET', `jobs/${J1}`)).body;
  const mock = { ...real, denoiseTier: 'rebuild', label: 'Voice rebuild', options: { ...real.options, atten_lim_db: undefined } };
  await page.route('**/api/jobs', (r) => (r.request().method() === 'GET' ? r.fulfill({ json: { jobs: [mock], limits: { maxActiveJobs: 3, retentionSeconds: 7200 } } }) : r.continue()));
  try {
    await gotoApp(`/app/job/${J1}`);
    await card().getByTestId('download-again').waitFor({ timeout: 20000 });
    if (!/Voice rebuild/.test(await card().innerText())) throw new Error('the card does not say Voice rebuild');
    if (await card().getByTestId('listen').count()) {
      await openListen();
      await sleep(300);
      if (await card().getByTestId('blend-player').count()) throw new Error('a slider for a Voice rebuild job');
    }
    await openAgain();
    if (await card().locator(`[data-testid=strength-again-${J1}]`).count()) throw new Error('a strength choice for a Voice rebuild job');
    return 'no slider, no strength choice';
  } finally {
    await page.unroute('**/api/jobs').catch(() => {});
  }
});

if (pricing.rebuildAvailable) {
  skip('a real Voice rebuild job', 'the price list offers Voice rebuild, but the test backend has no GPU worker to run it');
} else {
  console.log('SKIP a real Voice rebuild job -- the price list does not offer Voice rebuild here (no GPU worker in the test backend); the rebuild page was checked with the server\'s description above');
}

await check('Change and save again with Voice rebuild on offer (the price list is made to offer it): Rebuild the voice comes right below the strength, with its words and price, and there is no Restore highs; ticking the rebuild disables the strength, unticking brings it back', async () => {
  await page.route('**/api/billing/pricing', async (route) => {
    const res = await route.fetch();
    await route.fulfill({ response: res, json: { ...(await res.json()), rebuildAvailable: true } });
  });
  try {
    await gotoApp(`/app/job/${J2}`);
    await card().getByTestId('download-again').waitFor({ timeout: 20000 });
    await openAgain();
    await panel().getByTestId('again-rebuild').waitFor();
    const order = await panel().evaluate((el) => {
      const strength = el.querySelector('[data-testid^=strength-again-]');
      const rebuild = el.querySelector('[data-testid=again-rebuild]');
      return { strengthThenRebuild: !!strength && !!rebuild && (strength.compareDocumentPosition(rebuild) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0 };
    });
    if (!order.strengthThenRebuild) throw new Error('order in the panel: ' + JSON.stringify(order));
    if (await panel().getByTestId('again-highs').count() || /highs/i.test(await panel().innerText())) throw new Error('Restore highs is in the panel');
    const rate = (p, t) => pricing.rates.find((r) => r.product === p && r.tier === t)?.usdPerMinute;
    const fmt = (usd) => `$${usd.toFixed(4).replace(/(\.\d\d\d*?)0+$/, '$1')}`;
    const rebuildText = (await panel().getByTestId('again-rebuild').locator('xpath=ancestor::label').innerText()).replace(/\s+/g, ' ').trim();
    const wantRebuild = `Rebuild the voice For an echoey room, a muffled voice, phone-quality or damaged sound, or distortion from recording too loud: the whole voice is rebuilt, which also brings back the high sounds that are missing. Slower, ${fmt(rate('denoise', 'rebuild'))} a minute, and on very echoey recordings it can change a word, so listen before you publish.`;
    if (rebuildText !== wantRebuild) throw new Error(`Rebuild the voice in the panel: "${rebuildText}"`);
    await setBox('again-rebuild', true);
    if (!(await card().locator(`input[name=again-${J2}]`).first().isDisabled())) throw new Error('the strength can still be changed with the voice rebuilt');
    await setBox('again-rebuild', false);
    await page.waitForFunction((id) => !document.querySelector(`input[name=again-${id}]`)?.disabled, J2, { timeout: 5000 });
    return 'strength, Rebuild the voice; no Restore highs';
  } finally {
    await page.unroute('**/api/billing/pricing').catch(() => {});
  }
});

await check('noise Off, Leveling on its own: a paid job whose price is the Leveling price alone; its page offers Leveling in the panel and no strength and no Restore highs', async () => {
  await gotoApp('/app');
  await dropFile(page, join(FILES, 'noisy.wav'));
  await pick(page, 'noise', 'off');
  await page.getByTestId('start').waitFor();
  if (await page.getByTestId('start').isEnabled()) throw new Error('Start is on with nothing chosen');
  await page.locator('input[name=leveling-alone]').check();
  await page.getByTestId('estimate').waitFor();
  const bodyReq = page.waitForRequest((r) => r.url().endsWith('/api/jobs') && r.method() === 'POST');
  got.length = 0;
  await page.getByTestId('start').click();
  const body = (await bodyReq).postDataJSON();
  await page.waitForURL(JOB_PATH, { timeout: 90000 });
  const id = jobIdOf(page);
  if (body.kind !== 'denoise' || body.denoiseTier !== 'off' || body.options.leveling !== true || 'restore_highs' in body.options || body.transcribeTier) throw new Error('sent: ' + JSON.stringify(body));
  const micros = Math.ceil((1000 * Math.ceil(body.durationSeconds)) / 60);
  await waitCardDone(page);
  await card().getByTestId('result').waitFor({ timeout: 90000 });
  await waitDownloads(1);
  const j = (await apiCall(page, 'GET', `jobs/${id}`)).body;
  if (j.denoiseTier !== 'off' || j.options.leveling !== true || j.estCostMicros !== micros || j.costMicros !== micros) throw new Error(`tier ${j.denoiseTier}, leveling ${j.options.leveling}, est ${j.estCostMicros}, cost ${j.costMicros}, wanted ${micros}`);
  if (got.length !== 1) throw new Error('downloaded by itself: ' + got.map((g) => g.name));
  await openAgain();
  if (!(await panel().getByTestId('again-leveling').isChecked())) throw new Error('the panel does not show what the job was made with');
  if (await card().locator('[data-testid^=strength-again-]').count()) throw new Error('a strength choice for a job with no noise removal');
  if (await panel().getByTestId('again-highs').count()) throw new Error('Restore highs in the panel');
  await gotoApp('/app/billing');
  await page.getByTestId('ledger').waitFor();
  const rows = (await page.locator('[data-testid=ledger] tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
  const mine = rows.find((t) => /Leveling/.test(t) && !/Changed and saved again/.test(t) && t.includes(`−$${(micros / 1e6).toFixed(4)}`));
  if (!mine) throw new Error(`no billing line for ${micros} micros:\n${rows.join('\n')}`);
  return `${j.estCostMicros} micros = Leveling alone; ${mine}`;
});

// ---------------------------------------------------------------------------------------------------- 6. the kept copies

await check('kept copies: GET /api/jobs/<id>/kept/noise and /input are audio/flac for the owner (the size the job says), 404 for another account, 404 for anything else, 401 signed out; the job\'s redo block has the preview', async () => {
  const j = (await apiCall(page, 'GET', `jobs/${J1}`)).body;
  const redo = j.redo;
  if (!redo?.available || !redo.noise || !redo.preview) throw new Error('redo: ' + JSON.stringify(redo));
  const p = redo.preview;
  if (!(p.samples > 400000) || !Array.isArray(p.channels) || !Array.isArray(p.plan) || !(p.scale?.noise > 0) || !(p.scale?.input > 0) || !(p.bytes?.noise > 1000) || !(p.bytes?.input > 1000)) throw new Error('preview: ' + JSON.stringify(p));
  const seen = [];
  for (const which of ['noise', 'input']) {
    const res = await context.request.get(`${BASE}/api/jobs/${J1}/kept/${which}`);
    if (res.status() !== 200 || !/^audio\/flac/.test(res.headers()['content-type'] || '')) throw new Error(`${which}: ${res.status()} ${res.headers()['content-type']}`);
    const body = await res.body();
    if (body.length !== p.bytes[which] || body.subarray(0, 4).toString('latin1') !== 'fLaC') throw new Error(`${which}: ${body.length} bytes (the job says ${p.bytes[which]}), starts ${body.subarray(0, 4).toString('latin1')}`);
    if (!/no-store/.test(res.headers()['cache-control'] || '')) throw new Error('cacheable: ' + res.headers()['cache-control']);
    seen.push(`${which} ${body.length} B`);
  }
  if ((await context.request.get(`${BASE}/api/jobs/${J1}/kept/other`)).status() !== 404) throw new Error('an unknown copy name is not a 404');
  // another account
  const other = await browser.newContext();
  const op = await other.newPage();
  try {
    await signIn(op, `redo-other-${Date.now()}@example.com`);
    for (const which of ['noise', 'input']) {
      const res = await other.request.get(`${BASE}/api/jobs/${J1}/kept/${which}`);
      if (res.status() !== 404) throw new Error(`another account got ${res.status()} for ${which}`);
    }
    const info = await other.request.get(`${BASE}/api/jobs/${J1}`);
    if (info.status() !== 404) throw new Error('another account can read the job: ' + info.status());
  } finally {
    await other.close();
  }
  const anon = await browser.newContext();
  try {
    const res = await anon.request.get(`${BASE}/api/jobs/${J1}/kept/noise`);
    if (res.status() !== 401) throw new Error('signed out: ' + res.status());
  } finally {
    await anon.close();
  }
  // and not once the job is deleted
  const del = await apiCall(page, 'DELETE', `jobs/${freeRerun}`);
  if (del.status !== 200) throw new Error('delete ' + del.status);
  if ((await context.request.get(`${BASE}/api/jobs/${freeRerun}/kept/noise`)).status() !== 404) throw new Error('a deleted job still serves its copies');
  return `${seen.join(', ')}; ${p.samples} samples, ${p.channels.length} ch, scale ${p.scale.noise.toFixed(3)}/${p.scale.input.toFixed(3)}; 404 after Delete`;
});

for (const f of readdirSync(FILES)) if (f.startsWith(`redo-${TAG}-`)) rmSync(join(FILES, f), { force: true }); // the files this run saved
console.log('LOGS:\n' + logs.filter((l) => !/Failed to load resource|favicon|net::ERR/.test(l)).join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
