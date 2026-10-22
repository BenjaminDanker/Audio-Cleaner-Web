// Saving a finished job again, and the strength slider (the editor's inspector on a job's page):
//  1. the same job saved again on our side through Save changes: the paid choices (a new strength with Leveling on, Leveling alone), the quote
//     and the price on the button, `process` (paid, on the SAME job, with the shown price) or `export` (free), the new files come down by
//     themselves, one line "Edited (…)" in the billing history and none for a free save; no new job;
//  2. what a free save makes equals what the editor plays for the same choice: the exported file against (1 - a) * noise removal + a * recording,
//     both read from the preview stretches the editor mixes (below -60 dB of difference);
//  3. the continuous slider in the inspector: the names and dB it shows ("Between X and Y (N dB)", the range runs over the four names), the
//     gains the mix plays (`data-gains`), short stretches of the layers asked for (never a whole one), the speaker button for the original, Play /
//     Pause, a free save at the strength it holds, and the strength a saved job shows after a reload;
//  4. a Free cleanup made in the browser (a card of this tab): its strength slider is there at once and Save changes makes the new strength in
//     the browser, free, with no job (mono and stereo);
//  5. Voice rebuild: a job whose Voice is rebuilt has no strength slider; with Voice rebuild on offer (the price list is made to offer it) it is a
//     row below noise removal with its words and price; Voice Off with Leveling on its own is a paid job priced as Leveling alone;
//  6. a job made through the API with whole-file options is made with a timeline: the editor opens it on that timeline, another account and a
//     signed-out call get nothing, Delete removes the job.
// Needs the test backend (tests/run.sh redo) and the fixtures. Usage: node tests/redo.mjs
import { execFileSync } from 'node:child_process';
import { readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  BASE, FFMPEG, FILES, JOB_PATH, admin, apiCall, check, dropFile, jobIdOf, launch, listJobs, mainButton, pick, resultAttr, setKind, setLeveling, setStrength,
  setVoice, signIn, skip, studio, summary, waitCardDone, waitJobDone, waitResult,
} from './helpers.mjs';

const EMAIL = `redo-${Date.now()}@example.com`;
const TAG = Date.now().toString(36);
const FFPROBE = process.env.FFPROBE || join(process.env.HOME, 'audio-cleaner', 'bin', 'ffprobe');

const { browser, context, page, logs } = await launch();
const got = []; // [{name, download}] in order: what the page downloaded
page.on('download', (d) => got.push({ name: d.suggestedFilename(), download: d }));
const requests = [];
const reqTimes = []; // when each of them was asked (the server answers 429 to a client past a burst of 120 calls, refilled at 2 a second)
const previewAsked = []; // the preview stretches asked for, with their query: [path?start=&seconds=]
context.on('request', (r) => {
  const u = new URL(r.url());
  if (u.pathname.startsWith('/api/')) { requests.push(`${r.method()} ${u.pathname}`); reqTimes.push(Date.now()); }
  if (/\/preview\/(source|noise|rebuild|echo)$/.test(u.pathname)) previewAsked.push(u.pathname + u.search);
});
const quotes = [];
const quoteLost = []; // the quotes whose answer could not be read here
context.on('response', async (res) => {
  const u = new URL(res.url());
  if (res.request().method() !== 'POST' || !/\/api\/jobs\/[0-9a-f]+\/quote$/.test(u.pathname)) return;
  try { quotes.push({ path: u.pathname, body: JSON.parse(res.request().postData()), answer: await res.json() }); } catch (e) { quoteLost.push(String(e?.message || e).slice(0, 120)); /* not an answer */ }
});
await signIn(page, EMAIL);
admin('grant', EMAIL, '5', 'redo test');

const sleep = (ms) => page.waitForTimeout(ms);
const gotoApp = async (path = '/app') => { await spare(); await page.goto(`${BASE}${path}`); await page.getByTestId('available').or(page.getByTestId('job-page')).first().waitFor({ timeout: 20000 }); };
const waitDownloads = async (n, timeout = 60000) => {
  const t0 = Date.now();
  while (got.length < n && Date.now() - t0 < timeout) await sleep(250);
  await sleep(800); // anything extra would show up now
};
const near = (a, b, tol, what) => { if (!(Math.abs(a - b) <= tol)) throw new Error(`${what}: ${a} vs ${b}`); };
const countBy = (re) => requests.filter((r) => re.test(r)).length;
const pricing = await (await fetch(`${BASE}/api/billing/pricing`)).json();
/** The prototype's price words (the price on the main button). Micros of a dollar. */
const money = (m) => (m <= 0 ? 'Free' : m < 1000 ? '<$0.001' : `$${m >= 100000 || m % 10000 === 0 ? (m / 1e6).toFixed(2) : (m / 1e6).toFixed(3)}`);
const clock = (t) => { t = Math.round(t); return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`; };
const sections = (tr) => page.locator(`[data-testid=track-${tr}] [data-testid=section]`).evaluateAll((els) => els.map((e) => ({ s: Number(e.dataset.s), e: Number(e.dataset.e), text: e.textContent.trim() })));
const say = async (tr) => (await sections(tr)).map((x) => `${x.text} ${clock(x.s)}–${clock(x.e)}`).join(' | ');
const gains = async () => JSON.parse(await studio(page).getAttribute('data-gains'));
const timeNow = async () => Number(await studio(page).getAttribute('data-time'));
const lastQuote = () => quotes.at(-1);
const jobOf = async (id) => (await apiCall(page, 'GET', `jobs/${id}`)).body;
/** The rows of the billing page's credit history (the page is open already); says what the page shows when there is no table. */
async function ledgerRows() {
  try { await page.getByTestId('ledger').waitFor({ timeout: 15000 }); } catch { throw new Error('no credit history on the billing page: ' + (await page.locator('main').innerText().catch(() => '')).replace(/\s+/g, ' ').replace(/^.*Credit history/, 'Credit history').slice(0, 300)); }
  return (await page.locator('[data-testid=ledger] tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
}
const timelineOf = async (id) => (await apiCall(page, 'GET', `jobs/${id}/timeline`)).body;

/** A finished job's editor is on the page, with its tracks drawn. */
async function waitEditor() {
  await page.locator('[data-testid=studio][data-stage=done]').waitFor({ timeout: 40000 });
  await page.getByTestId('hear-voice').waitFor();
  await page.locator('[data-testid=track-voice] [data-testid=section]').first().waitFor();
  await page.waitForFunction(() => document.querySelector('[data-testid=waveform]')?.dataset.reading === 'no', null, { timeout: 40000 });
}
const openJob = async (id) => { await spare(); await page.goto(`${BASE}/app/job/${id}`); await waitEditor(); };
/** Waits until `need` calls are spare in the server's per-client allowance (a burst of 120, refilled at 2 a second), so the pace of a long run does not meet 429. A load of the editor makes only about 14 calls (a finished job's page; a file chosen on /app, 6), but what a test does after it (saving, its quotes, the stretches it plays) comes in bursts: 30 spare was not enough (a job's page answered 429 in the middle of a check), 70 is. */
async function spare(need = 70) {
  for (;;) {
    let tokens = 120; let last = reqTimes[0] ?? Date.now();
    for (const t of reqTimes) { tokens = Math.min(120, tokens + (t - last) / 500) - 1; last = t; }
    tokens = Math.min(120, tokens + (Date.now() - last) / 500);
    if (tokens >= need) return;
    await new Promise((r) => setTimeout(r, 500));
  }
}
/** The price on Save changes has settled: the quote is in and the button is on. Answers the button's text. */
async function savePrice() {
  await page.getByTestId('save-changes').waitFor({ timeout: 15000 });
  await page.waitForFunction(() => { const x = document.querySelector('[data-testid=save-changes]'); return x && !x.disabled; }, null, { timeout: 15000 });
  // every quote the page asked for has been answered and noted here (the price on the button is the last one's)
  for (let i = 0; i < 50 && quotes.length + quoteLost.length < countBy(/\/quote$/); i++) await sleep(100);
  await sleep(150);
  return (await page.getByTestId('save-changes').innerText()).replace(/\s+/g, ' ').trim();
}
/** After Save changes was pressed: the job runs again on its own page, ends, and its file comes down by itself. */
async function waitSaved(timeout = 90000) {
  await page.locator('[data-testid=studio][data-stage=processing]').waitFor({ timeout: 15000 });
  await waitCardDone(page);
  await waitResult(page, 'result', 120000);
  await waitDownloads(1, timeout);
}
/** Click the waveform at second t (moves the playhead). */
async function seekTo(t) {
  await page.getByTestId('waveform').scrollIntoViewIfNeeded();
  const box = await page.getByTestId('waveform').boundingBox();
  const T = Number(await page.getByTestId('wave').getAttribute('data-seconds'));
  await page.mouse.click(box.x + (t / T) * box.width, box.y + box.height / 2);
  await sleep(200);
}

/** Float samples of an audio file, as ffmpeg decodes them (channels kept, 48 kHz). */
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

const STARTED = {}; // job id -> what Start said
/** Starts a paid job from the start screen (every choice that matters is set), waits for its editor to say the files are made and for the file to come down. Answers the job's id. */
async function startJob({ file = 'noisy.wav', strength = 'max', leveling = true } = {}) {
  await gotoApp('/app');
  await dropFile(page, join(FILES, file));
  await setKind(page, 'paid');
  await setStrength(page, strength);
  await setLeveling(page, leveling ? 'youtube' : 'off');
  await sleep(400);
  const text = await mainButton(page);
  got.length = 0;
  await page.getByTestId('start').click();
  await page.waitForURL(JOB_PATH, { timeout: 90000 });
  const id = jobIdOf(page);
  STARTED[id] = text;
  await waitCardDone(page);
  await waitResult(page, 'result', 120000);
  await waitDownloads(1);
  // what the job holds: the timeline says what was asked (the strength and Leveling)
  const t = (await timelineOf(id)).timeline.layers;
  const want = { max: 100, strong: 40, balanced: 24, gentle: 12 }[strength];
  if (t.noise?.length !== 1 || t.noise[0].strength_db !== want || !!t.leveling !== leveling) throw new Error(`the job was made with ${JSON.stringify(t)}, wanted strength ${want}, leveling ${leveling}`);
  return id;
}

let J1; // made with Leveling, Maximum
let J2; // made without Leveling, Maximum

// ---------------------------------------------------------------------------------------------------- 1. saving a job again

await check('setup: two paid Standard noise removal jobs (J1 with Leveling, J2 without), both at Max, each on its own page with its file downloaded by itself', async () => {
  J1 = await startJob({ leveling: true });
  const first = got.map((g) => g.name);
  J2 = await startJob({ leveling: false });
  if (J1 === J2) throw new Error('same job twice');
  const j = await jobOf(J2);
  if (!j.edit?.available || !j.edit.hasTimeline) throw new Error('a job made by the website: ' + JSON.stringify(j.edit));
  return `${J1.slice(0, 8)}… (${first}, Start said ${STARTED[J1]}), ${J2.slice(0, 8)}… (${got.map((g) => g.name)}, Start said ${STARTED[J2]})`;
});

let processBody = null;
let jobsBefore = 0;
await check('a new strength with Leveling still on is paid (the button shows its price, only Leveling runs again): Save changes is a process on the SAME job, with the shown price, sending the timeline only', async () => {
  await openJob(J1);
  if ((await mainButton(page)) !== 'Download again') throw new Error('main button: ' + await mainButton(page));
  await setStrength(page, 'strong'); // (nothing is chosen: the whole file)
  const label = await savePrice();
  const q = lastQuote();
  if (q.answer.free || !q.answer.lines.some((l) => l.item === 'leveling') || q.answer.lines.some((l) => l.item === 'denoise')) throw new Error('price lines (only Leveling runs again): ' + JSON.stringify(q.answer.lines));
  if (label !== `Save changes · ${money(q.answer.estCostMicros)}` || !/^Save changes · \$0\.00\d$/.test(label)) throw new Error('Save label: ' + label);
  jobsBefore = (await listJobs(page)).jobs.length;
  got.length = 0;
  const [req] = await Promise.all([page.waitForRequest((r) => r.url().endsWith(`/api/jobs/${J1}/process`) && r.method() === 'POST'), page.getByTestId('save-changes').click()]);
  processBody = req.postDataJSON();
  if (Object.keys(processBody).some((k) => !['audioFormat', 'maxCostMicros', 'timeline'].includes(k)) || !processBody.timeline || processBody.free) throw new Error('sent: ' + JSON.stringify(processBody).slice(0, 300));
  if (processBody.maxCostMicros !== q.answer.estCostMicros || !(processBody.maxCostMicros > 0)) throw new Error('the quoted price was not sent along: ' + JSON.stringify(processBody).slice(0, 300));
  const noise = processBody.timeline.layers.noise;
  if (noise.length !== 1 || noise[0].strength_db !== 40 || noise[0].start !== 0 || processBody.timeline.layers.leveling?.length !== 1) throw new Error('timeline: ' + JSON.stringify(processBody.timeline));
  if (!page.url().endsWith(`/app/job/${J1}`)) throw new Error('the page moved to ' + page.url());
  return `${label}; sent ${JSON.stringify(processBody).slice(0, 160)}…`;
});

await check('the page says what is going on while it saves (the line and the progress, then "Saved"), shows no "Got it" note for a job that is not new, and the same job ends on its own page; its file comes down by itself, once; the billing history has one line "Edited (Leveling)"', async () => {
  const line = (await page.getByTestId('job-line').innerText()).trim();
  if (!line) throw new Error('no line about the run');
  if ((await page.getByTestId('job-progress').count()) !== 1) throw new Error('no progress bar');
  if (await page.getByTestId('submitted').count()) throw new Error('the welcome note of a new job on a saved one: ' + await page.getByTestId('submitted').innerText());
  await page.getByTestId('toast').filter({ hasText: 'Saved' }).waitFor({ timeout: 20000 });
  await waitCardDone(page);
  await waitResult(page, 'result', 120000);
  await waitDownloads(1);
  if (got.length !== 1 || !/\.flac$/.test(got[0].name)) throw new Error('downloaded by itself: ' + got.map((g) => g.name));
  const done = await jobOf(J1);
  if (done.status !== 'done' || !done.edit?.hasTimeline) throw new Error(JSON.stringify({ s: done.status, e: done.edit }));
  if ((await listJobs(page)).jobs.length !== jobsBefore) throw new Error('a new job appeared');
  if (countBy(/^POST \/api\/jobs$/) !== 2) throw new Error('a new job was asked for');
  await openJob(J1);
  if (await page.getByTestId('submitted').count()) throw new Error('the note comes back after a reload');
  if ((await studio(page).getAttribute('data-changed')) !== 'no' || (await page.getByTestId('strength-name').innerText()) !== 'Strong (40 dB)') throw new Error(`after a reload: changed ${await studio(page).getAttribute('data-changed')}, strength ${await page.getByTestId('strength-name').innerText()}`);
  await gotoApp('/app/billing');
  const rows = await ledgerRows();
  const mine = rows.filter((t) => /Edited \(Leveling\)/.test(t));
  if (mine.length !== 1) throw new Error(`${mine.length} lines for the change:\n${rows.join('\n')}`);
  if (/Edited \(Denoising|Edited \(Noise removal|Changed and saved again/.test(rows.join('\n'))) throw new Error('the save was charged for noise removal too');
  return `${line}; ${mine[0]}`;
});

await check('Leveling switched off with another strength is free on our side: the button says "Save changes · Free", Save is an export on the same job (no process, no new job), the new file comes down at once, and the billing history gets no new line', async () => {
  await openJob(J1);
  await setStrength(page, 'balanced');
  await setLeveling(page, 'off');
  const label = await savePrice();
  if (label !== 'Save changes · Free') throw new Error('label: ' + label); // (that the quote says free is checked by the request below: the page saves with an export only when it does)
  const processes = countBy(/process$/);
  const exports = countBy(/export$/);
  got.length = 0;
  const url = page.url();
  await page.getByTestId('save-changes').click();
  await waitSaved();
  if (got.length !== 1 || !/^noisy.*\.flac$/.test(got[0].name)) throw new Error('downloaded: ' + got.map((g) => g.name));
  await got[0].download.saveAs(join(FILES, `redo-${TAG}-j1-balanced.flac`));
  if (page.url() !== url) throw new Error('the page moved to ' + page.url());
  if (countBy(/process$/) !== processes || countBy(/export$/) !== exports + 1) throw new Error('requests: a process was asked for');
  if ((await listJobs(page)).jobs.length !== jobsBefore) throw new Error('a job was created');
  const j = await jobOf(J1);
  if (j.status !== 'done') throw new Error('the job changed');
  const st = probe(join(FILES, `redo-${TAG}-j1-balanced.flac`)).streams[0];
  if (st.codec_name !== 'flac') throw new Error(`file is ${st.codec_name}`);
  await gotoApp('/app/billing');
  const rows = (await ledgerRows()).filter((t) => /Edited \(/.test(t));
  if (rows.length !== 1) throw new Error(`${rows.length} lines for edits (only the paid one belongs there)`);
  return `${label}; ${got[0].name}, ${st.sample_rate} Hz ${st.channels} ch, no job`;
});

// ---------------------------------------------------------------------------------------------------- 2. the same file from what is played

await check('what a free save makes equals what the editor plays for the same choice: J2 at Strong, exported on our side, against (1 - a) * noise removal + a * recording from the preview stretches, decoded to float, the difference is below -60 dB', async () => {
  await openJob(J2);
  await setStrength(page, 'strong');
  const label = await savePrice();
  if (label !== 'Save changes · Free') throw new Error('not a free save: ' + label);
  got.length = 0;
  await page.getByTestId('save-changes').click();
  await waitSaved();
  if (got.length !== 1) throw new Error('downloads: ' + got.map((g) => g.name));
  const here = join(FILES, `redo-${TAG}-j2-strong.flac`);
  await got[0].download.saveAs(here);
  // the two layers the mix plays
  const stretch = async (layer) => {
    const res = await context.request.get(`${BASE}/api/jobs/${J2}/preview/${layer}?start=0.000000&seconds=30.000000`);
    if (res.status() !== 200) throw new Error(`${layer}: ${res.status()}`);
    const scale = Number(res.headers()['x-stretch-scale']) || 1;
    const p = join(FILES, `redo-${TAG}-${layer}.flac`);
    writeFileSync(p, await res.body());
    const a = decode(p, 1);
    return a.map((v) => v / scale);
  };
  const noise = await stretch('noise');
  const source = await stretch('source');
  const a = 10 ** (-40 / 20);
  const played = noise.map((v, i) => (1 - a) * v + a * (source[i] ?? 0));
  const file = decode(here, 1);
  if (Math.abs(played.length - file.length) > 480) throw new Error(`lengths ${played.length} and ${file.length}`);
  if (rms(file) < 0.005) throw new Error('silent file ' + rms(file));
  const ratio = snr(played, file);
  if (!(ratio > 60)) throw new Error(`the file and what is played differ: SNR ${ratio.toFixed(1)} dB (rms ${rms(played).toFixed(4)} vs ${rms(file).toFixed(4)})`);
  return `SNR ${ratio.toFixed(1)} dB over ${Math.min(played.length, file.length)} samples; file ${statSync(here).size} B`;
});

await check('Leveling turned on for a job made without it is a paid step by itself: the quote lists Leveling alone (its rate, whole seconds rounded up, plus the fixed amount of a run), the button shows the price, a process asks for it, the job is charged it, and the billing history has a line for it', async () => {
  await openJob(J2);
  if ((await say('level')).startsWith('YouTube')) throw new Error('J2 should start with Leveling off');
  const cost0 = (await jobOf(J2)).costMicros;
  await setLeveling(page, 'youtube');
  const label = await savePrice();
  const q = lastQuote().answer;
  const lv = q.lines;
  if (lv.length !== 1 || lv[0].item !== 'leveling') throw new Error('lines: ' + JSON.stringify(lv));
  const micros = Math.ceil(((lv[0].usdPerMinute * 1e6) / 60) * (Math.ceil(lv[0].seconds) + lv[0].fixedSeconds));
  near(lv[0].micros, micros, 1, 'Leveling micros (rate x (whole seconds + the fixed amount of a run))');
  if (q.estCostMicros !== lv[0].micros) throw new Error('the price is not the Leveling line: ' + q.estCostMicros);
  if (label !== `Save changes · ${money(q.estCostMicros)}`) throw new Error('Save label: ' + label);
  got.length = 0;
  const [req] = await Promise.all([page.waitForRequest((r) => r.url().endsWith(`/api/jobs/${J2}/process`) && r.method() === 'POST'), page.getByTestId('save-changes').click()]);
  const sent = req.postDataJSON();
  if (sent.maxCostMicros !== q.estCostMicros || sent.timeline.layers.leveling?.length !== 1) throw new Error('sent: ' + JSON.stringify(sent).slice(0, 300));
  await waitSaved();
  if (got.length !== 1) throw new Error('downloaded by itself: ' + got.map((g) => g.name));
  const j = await jobOf(J2);
  if (j.costMicros - cost0 !== q.estCostMicros) throw new Error(`cost ${cost0} -> ${j.costMicros}, wanted +${q.estCostMicros}${j.costMicros === undefined ? ` (the job's page answered: ${JSON.stringify(j).slice(0, 160)})` : ''}`);
  await gotoApp('/app/billing');
  const rows = (await ledgerRows()).filter((t) => /Edited \(/.test(t));
  const usd = `−$${(q.estCostMicros / 1e6).toFixed(4)}`;
  const mine = rows.filter((t) => t.includes(usd) && /Edited \(Leveling\)/.test(t));
  if (rows.length !== 2 || mine.length !== 2) throw new Error(`edit lines (J1's, and this one, both Leveling alone for ${usd}):\n${rows.join('\n')}`); // (J1's line from before costs the same now: only the count says this job added one)
  return `${label}; ${mine[0]}`;
});

// ---------------------------------------------------------------------------------------------------- 3. the slider

const A = (db) => (db >= 100 ? 0 : 10 ** (-db / 20));
const NAMES = { 12: 'Gentle', 24: 'Balanced', 40: 'Strong', 100: 'Maximum' };
const sliderName = () => page.getByTestId('strength-name').innerText();
const slider = () => page.getByTestId('strength-slider');
/** The gains at the playhead are the blend of a strength (3 decimals, as the page rounds them). */
async function expectBlend(db, what) {
  const g = await gains();
  near(g.noise, 1 - A(db), 6e-4, `${what}: noise removal`);
  near(g.source, A(db), 6e-4, `${what}: recording`);
  return g;
}

await check('slider, a job on our side: the page asks for short stretches (10 seconds) of the layers it needs and never a whole one, the first ones at 0 and 10, the slider starts at the strength the job holds (Balanced) and the range runs 0 to 3 over the four names', async () => {
  const asked = previewAsked.length;
  await spare();
  await page.goto(`${BASE}/app/job/${J1}`);
  await waitEditor();
  await page.waitForFunction((n) => (window.performance.getEntriesByType('resource').filter((e) => /\/preview\//.test(e.name)).length >= n), 4, { timeout: 30000 }).catch(() => {});
  await sleep(800);
  const gets = previewAsked.slice(asked);
  if (!gets.length || gets.some((u) => !/\?start=\d+\.\d{6}&seconds=\d+\.\d{6}$/.test(u) || Number(/seconds=([\d.]+)/.exec(u)[1]) > 10.000001)) throw new Error('requests: ' + gets);
  for (const layer of ['source', 'noise']) if (!gets.includes(`/api/jobs/${J1}/preview/${layer}?start=0.000000&seconds=10.000000`)) throw new Error(`no first stretch of ${layer}: ${gets}`);
  if (!gets.some((u) => /\/preview\/noise\?start=10\.000000&seconds=0\.\d+$/.test(u))) throw new Error('no stretch after the first ten seconds: ' + gets);
  // the strength of what this job holds (Balanced, saved above)
  if ((await sliderName()) !== 'Balanced (24 dB)') throw new Error('starts at ' + await sliderName());
  if ((await slider().getAttribute('min')) !== '0' || (await slider().getAttribute('max')) !== '3') throw new Error('range ' + await slider().getAttribute('min') + '..' + await slider().getAttribute('max'));
  if (Number(await slider().inputValue()) !== 1) throw new Error('the range is at ' + await slider().inputValue());
  if ((await page.locator('.studio button[data-act=v-str][aria-pressed=true]').innerText()) !== 'Balanced') throw new Error('Balanced is not pressed');
  await seekTo(5);
  await expectBlend(24, 'Balanced');
  return `${gets.length} stretch requests (${gets.slice(0, 2).map((u) => u.replace(/^.*preview\//, '')).join(', ')} …), at Balanced`;
});

await check('slider, Leveling off: the four names give exactly the gains of the blend, noise removal (1 - a) and the recording a (24 dB: 0.937 and 0.063), the name and dB are shown, and a change of strength asks for nothing', async () => {
  const asked = previewAsked.length;
  const seen = [];
  for (const [id, db] of [['max', 100], ['balanced', 24], ['gentle', 12], ['strong', 40]]) {
    await setStrength(page, id);
    await page.waitForFunction((t) => document.querySelector('[data-testid=strength-name]')?.textContent === t, `${NAMES[db]} (${db} dB)`, { timeout: 5000 });
    const g = await expectBlend(db, NAMES[db]);
    near(Number(await slider().inputValue()), [12, 24, 40, 100].indexOf(db), 0.001, `${db} dB range position`);
    seen.push(`${NAMES[db]} ${g.noise}/${g.source}`);
  }
  const g = await gains();
  if (g.noise !== 0.99 || g.source !== 0.01) throw new Error('Strong is not 0.99 / 0.01: ' + JSON.stringify(g));
  await sleep(500);
  if (previewAsked.length !== asked) throw new Error(`${previewAsked.length - asked} requests for changes of strength`);
  return seen.join(', ');
});

await check('slider: dragging the range to a value between two names gives a strength between their dB values, named "Between X and Y (N dB)" with the gains of that blend, and none of the four names is pressed', async () => {
  const cases = [[0, 12, 'Gentle'], [1, 24, 'Balanced'], [2, 40, 'Strong'], [3, 100, 'Maximum'], [0.5, 18, 'Between Gentle and Balanced'], [1.5, 32, 'Between Balanced and Strong'], [2.5, 70, 'Between Strong and Maximum'], [1.2, 24 + 16 * 0.2, 'Between Balanced and Strong']];
  const seen = [];
  for (const [pos, db, name] of cases) {
    await slider().fill(String(pos));
    const want = `${name} (${Math.round(db)} dB)`;
    await page.waitForFunction((t) => document.querySelector('[data-testid=strength-name]')?.textContent === t, want, { timeout: 5000 }).catch(async () => { throw new Error(`position ${pos}: name "${await sliderName()}", wanted "${want}"`); });
    await expectBlend(db, `position ${pos}`);
    if ((await slider().getAttribute('aria-valuetext')) !== name) throw new Error(`position ${pos}: aria-valuetext ${await slider().getAttribute('aria-valuetext')}`);
    const pressed = await page.locator('.studio button[data-act=v-str][aria-pressed=true]').count();
    if (pressed !== (Number.isInteger(pos) ? 1 : 0)) throw new Error(`position ${pos}: ${pressed} names pressed`);
    seen.push(`${pos}→${Math.round(db)}`);
  }
  await slider().fill('1.5');
  if ((await sliderName()) !== 'Between Balanced and Strong (32 dB)') throw new Error('in between: ' + await sliderName());
  // choosing a name moves the slider, and the slider's own Gentle presses Gentle
  await setStrength(page, 'strong');
  if (Number(await slider().inputValue()) !== 2) throw new Error('slider at ' + await slider().inputValue());
  await slider().fill('0');
  if ((await page.locator('.studio button[data-act=v-str][aria-pressed=true]').innerText()) !== 'Gentle') throw new Error('Gentle is not pressed after the slider went there');
  // the price follows: a different strength is a change to save (and free here: no Leveling)
  if ((await savePrice()) !== 'Save changes · Free') throw new Error('price: ' + await mainButton(page));
  return seen.join(' ');
});

await check('slider: the Voice speaker off plays the recording only (gains 1 and 0), on brings the blend back at the strength the slider holds', async () => {
  await setStrength(page, 'balanced');
  await seekTo(5);
  await page.getByTestId('hear-voice').click();
  await sleep(150);
  let g = await gains();
  if (!(g.source === 1 && g.noise === 0)) throw new Error(`the original: ${JSON.stringify(g)}`);
  if ((await page.getByTestId('hearing').innerText()) !== 'Hearing: without Voice') throw new Error('hearing: ' + await page.getByTestId('hearing').innerText());
  await page.getByTestId('hear-voice').click();
  await sleep(150);
  g = await expectBlend(24, 'back to the blend');
  return JSON.stringify(g);
});

await check('slider: Play starts the sound (the button says Pause and the time moves), Pause stops it (the time stands still), Play goes on from there', async () => {
  await seekTo(1);
  const play = page.getByTestId('play');
  if ((await play.getAttribute('aria-label')) !== 'Play') throw new Error('button ' + await play.getAttribute('aria-label'));
  await play.click();
  await page.waitForFunction(() => document.querySelector('[data-testid=studio]')?.dataset.playing === 'yes' && document.querySelector('[data-testid=play]')?.getAttribute('aria-label') === 'Pause', null, { timeout: 15000 });
  await page.waitForFunction(() => Number(document.querySelector('[data-testid=studio]')?.dataset.time) > 2.2, null, { timeout: 15000 });
  const running = await timeNow();
  await play.click();
  if ((await play.getAttribute('aria-label')) !== 'Play') throw new Error('button after Pause: ' + await play.getAttribute('aria-label'));
  await sleep(300);
  const stopped = await timeNow();
  await sleep(1500);
  if ((await timeNow()) !== stopped) throw new Error(`the time moved while paused: ${stopped} -> ${await timeNow()}`);
  await play.click();
  await page.waitForFunction((t) => Number(document.querySelector('[data-testid=studio]')?.dataset.time) > t + 0.5, stopped, { timeout: 15000 });
  await play.click();
  if (!(running >= 2)) throw new Error('the time never passed 0:02');
  return `played to ${running.toFixed(1)} s, paused at ${stopped.toFixed(1)}, played on`;
});

await check('slider on our side, Save changes (Leveling off): a file at that strength comes down at once, an export on the same job, no job and no re-run; Gentle keeps more of the recording than Balanced did', async () => {
  await setStrength(page, 'gentle');
  const label = await savePrice();
  if (label !== 'Save changes · Free') throw new Error('label: ' + label);
  const exports = countBy(/export$/);
  got.length = 0;
  await page.getByTestId('save-changes').click();
  await waitSaved();
  if (got.length !== 1) throw new Error('downloads: ' + got.map((g) => g.name));
  const p = join(FILES, `redo-${TAG}-j1-gentle.flac`);
  await got[0].download.saveAs(p);
  if (countBy(/export$/) !== exports + 1 || (await listJobs(page)).jobs.length !== jobsBefore) throw new Error('a job was requested');
  const balanced = rms(decode(join(FILES, `redo-${TAG}-j1-balanced.flac`), 1));
  const gentle = rms(decode(p, 1));
  if (!(gentle > balanced)) throw new Error(`Gentle ${gentle.toFixed(5)} is not above Balanced ${balanced.toFixed(5)}`);
  await openJob(J1);
  if ((await sliderName()) !== 'Gentle (12 dB)') throw new Error('after a reload the slider says ' + await sliderName());
  return `Gentle rms ${gentle.toFixed(5)} > Balanced ${balanced.toFixed(5)}; the saved strength is what the page shows`;
});

// ---------------------------------------------------------------------------------------------------- 4. the free run in the browser

async function freeRun(file, strength) {
  await gotoApp('/app');
  await dropFile(page, join(FILES, file));
  await pick(page, 'noise', 'browser');
  await pick(page, 'subs', 'off');
  await setStrength(page, strength);
  got.length = 0;
  await page.getByTestId('start').click();
  await page.waitForURL(/\/app\/job\/local-[0-9a-z]+$/, { timeout: 180000 });
  await waitResult(page, 'result', 60000);
  await waitDownloads(1);
}
const dataRms = async () => Number(await resultAttr(page, 'rms'));

for (const [file, channels, label] of [['noisy.wav', 1, 'mono'], ['noisy24st.wav', 2, 'stereo']]) {
  await check(`free noise removal in the browser (${label}): its page /app/job/local-… has the strength slider at once; Gentle is saved with no job and is louder than the Strong one (the page's rms and the files')`, async () => {
    await freeRun(file, 'strong');
    if (got.length !== 1) throw new Error('downloaded by itself: ' + got.map((g) => g.name));
    const maxFile = join(FILES, `redo-${TAG}-free-${label}-max.wav`);
    await got[0].download.saveAs(maxFile);
    const maxRms = await dataRms();
    if (!jobIdOf(page).startsWith('local-')) throw new Error('page ' + page.url());
    if ((await resultAttr(page, 'channels')) !== String(channels)) throw new Error('channels ' + await resultAttr(page, 'channels'));
    // the slider is there at once (no sound to fetch), at the strength it was made with; Free goes up to Strong
    if ((await sliderName()) !== 'Strong (40 dB)' || (await slider().getAttribute('max')) !== '2') throw new Error(`slider: ${await sliderName()}, max ${await slider().getAttribute('max')}`);
    if ((await resultAttr(page, 'strength-db')) !== '40.00') throw new Error('data-strength-db ' + await resultAttr(page, 'strength-db'));
    const jobs0 = (await listJobs(page)).jobs.length;
    const req0 = requests.length;
    await setStrength(page, 'gentle');
    await page.waitForFunction(() => document.querySelector('[data-testid=strength-name]')?.textContent === 'Gentle (12 dB)', null, { timeout: 5000 });
    const save = (await page.getByTestId('save-changes').innerText()).replace(/\s+/g, ' ').trim();
    if (save !== 'Save changes · Free') throw new Error('save: ' + save);
    got.length = 0;
    await page.getByTestId('save-changes').click();
    await waitDownloads(1, 90000);
    if (got.length !== 1) throw new Error('second download: ' + got.map((g) => g.name));
    const gentleFile = join(FILES, `redo-${TAG}-free-${label}-gentle.wav`);
    await got[0].download.saveAs(gentleFile);
    await page.waitForFunction((x) => Number(document.querySelector('[data-testid=studio]')?.getAttribute('data-rms')) > x, maxRms, { timeout: 15000 });
    const gentleRms = await dataRms();
    if ((await resultAttr(page, 'strength-db')) !== '12.00') throw new Error('data-strength-db after Gentle ' + await resultAttr(page, 'strength-db'));
    const sent = requests.slice(req0).filter((r) => /^POST \/api\/jobs|process$|export$|quote$/.test(r));
    if (sent.length) throw new Error('requests: ' + sent.join(', '));
    if ((await listJobs(page)).jobs.length !== jobs0) throw new Error('a job was created');
    const fileMax = rms(decode(maxFile, channels));
    const fileGentle = rms(decode(gentleFile, channels));
    if (!(fileGentle > fileMax)) throw new Error(`files: Gentle ${fileGentle} is not above Strong ${fileMax}`);
    near(fileGentle, gentleRms, gentleRms * 0.03, 'the page and the file agree on the loudness');
    const st = probe(gentleFile).streams[0];
    if (st.channels !== channels) throw new Error(`the saved file has ${st.channels} channels`);
    return `rms ${maxRms.toFixed(4)} (Strong) -> ${gentleRms.toFixed(4)} (Gentle); files ${fileMax.toFixed(4)} -> ${fileGentle.toFixed(4)}, ${st.channels} ch; no job, no request`;
  });
}

// ---------------------------------------------------------------------------------------------------- 5. Voice rebuild

await check('a job whose Voice is rebuilt gets no strength slider (and no strength to choose): its page, as the server describes it with a rebuild part over the whole file, offers neither', async () => {
  await page.route(`**/api/jobs/${J1}/timeline`, async (route) => {
    const res = await route.fetch();
    const body = await res.json();
    const tl = { version: 1, seconds: body.seconds, layers: { rebuild: [{ start: 0, end: body.seconds }] } };
    await route.fulfill({ response: res, json: { ...body, timeline: tl, rendered: tl, custom: true } });
  });
  try {
    await openJob(J1);
    if ((await say('voice')).split(' ')[0] !== 'Rebuild') throw new Error('the editor does not say Voice rebuild: ' + await say('voice'));
    if ((await page.getByTestId('voice-rebuild').getAttribute('aria-checked')) !== 'true') throw new Error('Voice rebuild is not the Voice choice');
    if (await page.getByTestId('strength-slider').count() || await page.locator('.studio button[data-act=v-str]').count()) throw new Error('a strength for a rebuilt voice');
    return 'no slider, no strength choice';
  } finally {
    await page.unroute(`**/api/jobs/${J1}/timeline`).catch(() => {});
  }
});

if (pricing.rebuildAvailable) {
  skip('a real Voice rebuild job', 'the price list offers Voice rebuild, but the test backend has no GPU worker to run it');
} else {
  console.log('SKIP a real Voice rebuild job -- the price list does not offer Voice rebuild here (no GPU worker in the test backend); the rebuild page was checked with the server\'s description above');
}

await check('Voice rebuild on offer (the price list is made to offer it): Voice rebuild is the row below noise removal with its words and price, there is no Restore highs; choosing it takes the strength away, choosing noise removal brings it back', async () => {
  await page.route('**/api/billing/pricing', async (route) => {
    const res = await route.fetch();
    await route.fulfill({ response: res, json: { ...(await res.json()), rebuildAvailable: true } });
  });
  try {
    await openJob(J2);
    const row = page.getByTestId('voice-rebuild');
    if ((await row.getAttribute('aria-disabled')) === 'true' || /paused/i.test(await row.innerText())) throw new Error('Voice rebuild is paused although it is offered: ' + await row.innerText());
    const order = await page.evaluate(() => {
      const strength = document.querySelector('.studio button[data-act=v-str]');
      const rebuild = document.querySelector('[data-testid=voice-rebuild]');
      const off = document.querySelector('[data-testid=voice-off]');
      const noise = document.querySelector('[data-testid=voice-noise]');
      const after = (a, b) => !!a && !!b && (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0;
      return { rows: after(off, noise) && after(noise, rebuild), strengthThenRebuild: after(strength, rebuild) };
    });
    if (!order.rows || !order.strengthThenRebuild) throw new Error('order in the inspector: ' + JSON.stringify(order));
    const rate = pricing.rates.find((r) => r.product === 'denoise' && r.tier === 'rebuild')?.usdPerMinute;
    const fmt = (usd) => `$${usd.toFixed(4).replace(/(\.\d\d\d*?)0+$/, '$1')}`;
    const tip = await row.getAttribute('data-tip');
    if (!tip.includes(`Slower, ${fmt(rate)} a minute, files up to 90 minutes.`) || !/On very echoey recordings it can change a word, so listen before you publish\./.test(tip) || !/Keeps only speech/.test(tip)) throw new Error(`Voice rebuild's words: "${tip}"`);
    if (/restore highs/i.test(await page.getByTestId('inspector').innerText()) || /restore highs/i.test(tip)) throw new Error('Restore highs is in the inspector');
    await row.click();
    if (await page.getByTestId('strength-slider').count() || (await row.getAttribute('aria-checked')) !== 'true') throw new Error('the strength can still be changed with the voice rebuilt');
    await page.getByTestId('voice-noise').click();
    await page.getByTestId('strength-slider').waitFor({ timeout: 5000 });
    return 'noise removal, then Voice rebuild below it; no Restore highs';
  } finally {
    await page.unroute('**/api/billing/pricing').catch(() => {});
  }
});

await check('Voice Off, Leveling on its own: a paid job whose timeline is Leveling alone; the start screen shows no noise removal in the price and the job page has Leveling and no strength', async () => {
  await gotoApp('/app');
  await dropFile(page, join(FILES, 'noisy.wav'));
  await setVoice(page, 'off');
  await sleep(300);
  if (await page.getByTestId('start').isEnabled()) throw new Error('Start is on with nothing chosen');
  await setLeveling(page, 'youtube');
  await page.waitForFunction(() => /^Start · /.test(document.querySelector('[data-testid=start]')?.textContent || '') && !document.querySelector('[data-testid=start]').disabled, null, { timeout: 15000 });
  await page.getByTestId('bar-det').click();
  const det = (await page.getByTestId('details').innerText()).replace(/\s+/g, ' ');
  await page.getByTestId('bar-det').click();
  if (!/Leveling · 0:11/.test(det) || /Noise removal/.test(det)) throw new Error('details: ' + det);
  const start = await mainButton(page);
  const bodyReq = page.waitForRequest((r) => r.url().endsWith('/api/jobs') && r.method() === 'POST');
  got.length = 0;
  await page.getByTestId('start').click();
  const body = (await bodyReq).postDataJSON();
  await page.waitForURL(JOB_PATH, { timeout: 90000 });
  const id = jobIdOf(page);
  if (body.kind !== 'denoise' || 'denoiseTier' in body || 'transcribeTier' in body || Object.keys(body.timeline.layers).join() !== 'leveling' || 'leveling' in body.options) throw new Error('sent: ' + JSON.stringify(body).slice(0, 400));
  await waitCardDone(page);
  await waitResult(page, 'result', 120000);
  await waitDownloads(1);
  const j = await jobOf(id);
  const lv = Math.ceil(((0.001 * 1e6) / 60) * (Math.ceil(body.durationSeconds) + 98.462));
  near(j.costMicros, lv, 2, 'the job costs Leveling alone');
  if (j.estCostMicros !== j.costMicros) throw new Error(`est ${j.estCostMicros}, cost ${j.costMicros}`);
  if ((await say('voice')).split(' ')[0] !== 'Off' || (await say('level')).startsWith('Off')) throw new Error(`${await say('voice')} / ${await say('level')}`);
  if (await page.getByTestId('strength-slider').count()) throw new Error('a strength for a job with no noise removal');
  await gotoApp('/app/billing');
  const rows = await ledgerRows();
  const mine = rows.find((t) => !/Edited \(/.test(t) && t.includes(`−$${(j.costMicros / 1e6).toFixed(4)}`)); // the line carries the job's own name for it (Denoising: a job made from parts is not named after its parts), no longer "Leveling"
  if (!mine) throw new Error(`no billing line for ${j.costMicros} micros:\n${rows.join('\n')}`);
  return `${start}; ${j.costMicros} micros = Leveling alone (${lv}); ${mine}`;
});

// ---------------------------------------------------------------------------------------------------- 6. a job made through the API

await check('a job made through the API with whole-file options (Denoising at Max, Leveling off) is made with a timeline: the editor opens it on that timeline; another account gets 404, signed out 401; Delete removes it', async () => {
  const file = join(FILES, 'noisy.wav');
  const made = await apiCall(page, 'POST', 'jobs', {
    kind: 'denoise', denoiseTier: 'standard', filename: 'api.wav', sizeBytes: statSync(file).size, durationSeconds: 10.6,
    options: { noise: 'light', leveling: false, atten_lim_db: 100, output_video: false, audio_format: 'flac' },
  });
  if (made.status !== 200) throw new Error(`create ${made.status} ${JSON.stringify(made.body)}`);
  const id = made.body.id;
  const csrf = (await (await context.request.get(`${BASE}/api/me`)).json()).csrfToken;
  const put = await context.request.put(`${BASE}/api/jobs/${id}/input`, { data: readFileSync(file), headers: { 'X-CSRF-Token': csrf, 'Content-Type': 'application/octet-stream' }, timeout: 120000 });
  if (!put.ok()) throw new Error(`upload ${put.status()} ${await put.text()}`);
  const done = await waitJobDone(page, id, { timeout: 240000 });
  if (done.status !== 'done' || !done.edit?.available || !done.edit.hasTimeline || 'redo' in done || 'sourceJobId' in done) throw new Error('the job: ' + JSON.stringify({ s: done.status, edit: done.edit, redo: done.redo, src: done.sourceJobId }));
  // the website opens such a job in the editor: its timeline is what the options stand for
  await openJob(id);
  if ((await say('voice')) !== 'Max 0:00–0:11' || (await say('level')) !== 'Off 0:00–0:11') throw new Error(`${await say('voice')} / ${await say('level')}`);
  // the kept copies of the old way are gone
  for (const which of ['noise', 'input']) if ((await context.request.get(`${BASE}/api/jobs/${id}/kept/${which}`)).status() !== 404) throw new Error(`/kept/${which} is still served`);
  const other = await browser.newContext();
  const op = await other.newPage();
  try {
    await signIn(op, `redo-other-${Date.now()}@example.com`);
    const info = await other.request.get(`${BASE}/api/jobs/${id}`);
    if (info.status() !== 404) throw new Error('another account can read the job: ' + info.status());
    const del = await other.request.delete(`${BASE}/api/jobs/${id}`, { headers: { 'X-CSRF-Token': (await (await other.request.get(`${BASE}/api/me`)).json()).csrfToken } });
    if (del.status() !== 404) throw new Error('another account can delete the job: ' + del.status());
  } finally {
    await other.close();
  }
  const anon = await browser.newContext();
  try {
    const res = await anon.request.get(`${BASE}/api/jobs/${id}`);
    if (res.status() !== 401) throw new Error('signed out: ' + res.status());
  } finally {
    await anon.close();
  }
  const del = await apiCall(page, 'DELETE', `jobs/${id}`);
  if (del.status !== 200) throw new Error('delete ' + del.status);
  const gone = await apiCall(page, 'GET', `jobs/${id}`); // (the job stays in the list as expired; its files are gone)
  if (gone.status !== 200 || gone.body.status !== 'expired' || (gone.body.outputs || []).length) throw new Error('a deleted job: ' + JSON.stringify({ s: gone.status, st: gone.body?.status, o: gone.body?.outputs }));
  if ((await context.request.get(`${BASE}/api/jobs/${id}/outputs/api-cleaned.flac`)).status() !== 404) throw new Error('a deleted job still serves its result');
  return `${id.slice(0, 8)}… opens on its timeline (Max, Leveling off); 404 for another account, 401 signed out, expired with no files after Delete`;
});

for (const f of readdirSync(FILES)) if (f.startsWith(`redo-${TAG}-`)) rmSync(join(FILES, f), { force: true }); // the files this run saved
console.log('LOGS:\n' + logs.filter((l) => !/Failed to load resource|favicon|net::ERR/.test(l)).join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
