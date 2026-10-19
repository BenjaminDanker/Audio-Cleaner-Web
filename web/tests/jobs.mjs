// Every file job has a page of its own (/app/job/<id>) and the Cleanup page lists the jobs as short rows:
//  1. Start opens the job's page (the "Got it" note once), its progress and the time left, its result, the file that comes down by itself,
//     the same page after a reload, and back / forward;
//  2. the list on /app: a row per job with a progress bar and the time left while it works, the 2-hour countdown when it is done,
//     Open goes to the job's page, a job that ends while the list is on screen downloads by itself;
//  3. "Start another file", Delete now (asks first) and the address of a deleted job;
//  4. at most 3 file jobs at once: the form says so before Start, the server says so (429, too_many_active_jobs) when asked anyway;
//  5. a job past its 2 hours is not listed and its page says it is gone.
// Needs the test backend (tests/run.sh jobs) and the fixtures. Usage: node tests/jobs.mjs
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { BASE, FILES, JOB_PATH, SHOTS, admin, apiCall, check, dropFile, jobIdOf, launch, listJobs, pick, pyRun, signIn, summary } from './helpers.mjs';

const EMAIL = `jobs-${Date.now()}@example.com`;
const TOO_MANY = 'You have 3 files being processed. Start this one when one of them is done.';
const ACTIVE = ['uploading', 'queued', 'running', 'cloud_asr'];

const { browser, context, page, logs } = await launch();
const got = []; // the files the page downloaded (they start by themselves when a job is done)
page.on('download', (d) => got.push(d.suggestedFilename()));
await signIn(page, EMAIL);
admin('grant', EMAIL, '5', 'jobs test');
await page.goto(`${BASE}/app`); // loaded again: the credit is on the page
await page.getByTestId('available').waitFor();

const gotoApp = async (path = '/app') => { await page.goto(`${BASE}${path}`); await page.getByTestId('available').or(page.getByTestId('job-page')).first().waitFor({ timeout: 20000 }); };
const card = () => page.getByTestId('job').first();
const status = (loc) => loc.getAttribute('data-status', { timeout: 3000 }).catch(() => null);
const sleep = (ms) => page.waitForTimeout(ms);

/** On /app: drop a file, ask for paid Denoising, press Start and wait for the job's own page. Answers the job's id. */
async function startPaidJob(file = 'noisy.wav') {
  if (!/\/app$/.test(page.url())) await gotoApp('/app');
  await page.getByTestId('available').waitFor();
  await dropFile(page, join(FILES, file));
  await pick(page, 'noise', 'denoise');
  await page.getByTestId('estimate').waitFor();
  await page.getByTestId('start').click();
  await page.waitForURL(JOB_PATH, { timeout: 90000 });
  return jobIdOf(page);
}

/** Waits until the job card on the page has ended; throws with the card's text unless it is done. */
async function waitCardDone(timeout = 240000) {
  const t0 = Date.now();
  for (;;) {
    const st = await status(card());
    if (st === 'done') return;
    if (st === 'failed' || st === 'canceled') throw new Error(`job ${st}: ${(await card().innerText()).slice(0, 300)}`);
    if (Date.now() - t0 > timeout) throw new Error(`still ${st} after ${timeout / 1000} s`);
    await sleep(500);
  }
}
const waitDownloads = async (n, timeout = 40000) => {
  const t0 = Date.now();
  while (got.length < n && Date.now() - t0 < timeout) await sleep(250);
  await sleep(600); // anything extra would show up now
};

// ---------------------------------------------------------------------------------------------------- 1. the job's page

let A; // the first job: its id
await check('Start on /app opens the job\'s own page /app/job/<id> (the page for this job, titled with the file) and says "Got it" once at the top; going back is the form, forward is the job again', async () => {
  got.length = 0;
  A = await startPaidJob();
  if (!/\/app\/job\/[0-9a-f]{32}$/.test(page.url())) throw new Error(page.url());
  const banner = (await page.getByTestId('submitted').innerText()).trim();
  if (!/^Got it, we’re on it\./.test(banner)) throw new Error(`banner: ${banner}`);
  await card().waitFor(); // the page shows "Job" until the job is loaded
  if (!/noisy\.wav/.test(await page.locator('h1').innerText())) throw new Error('title: ' + await page.locator('h1').innerText());
  if ((await card().getAttribute('data-job-id')) !== A) throw new Error('the card is for another job');
  if ((await page.getByTestId('all-jobs').count()) !== 1 || (await page.getByTestId('start-another').count()) !== 1) throw new Error('"Your jobs" and "Start another file" are not both on the page');
  // the page replaced its own history entry, so back is the form (not another job page) and forward brings the job without the note
  await page.goBack();
  await page.waitForURL(/\/app$/);
  await page.getByTestId('start').or(page.locator('input[type=file]')).first().waitFor({ timeout: 10000 });
  await page.goForward();
  await page.waitForURL(JOB_PATH);
  await card().waitFor();
  if (await page.getByTestId('submitted').count()) throw new Error('the note came back with forward');
  return `${banner.slice(0, 70)} (${A.slice(0, 8)}…)`;
});

await check('while it runs: a progress bar that moves and a line with the time left ("about N min left" / "less than a minute left"), then the result', async () => {
  const lines = new Set();
  let top = 0;
  let sawQueue = false;
  const t0 = Date.now();
  for (;;) {
    const st = await status(card());
    if (st === 'done') break;
    if (st === 'failed' || st === 'canceled') throw new Error(`job ${st}: ${(await card().innerText()).slice(0, 300)}`);
    if (Date.now() - t0 > 240000) throw new Error('did not finish; last seen ' + [...lines].join(' | '));
    if (st === 'queued') sawQueue = true;
    const line = page.getByTestId('job-line');
    if (await line.count()) {
      const text = await line.innerText({ timeout: 2000 }).catch(() => '');
      if (text.trim()) lines.add(text.trim());
      const v = await page.getByTestId('job-progress').evaluate((p) => p.value, null, { timeout: 2000 }).catch(() => 0);
      top = Math.max(top, v);
    }
    await sleep(300);
  }
  const all = [...lines];
  if (!all.length) throw new Error('the running job never showed its progress line');
  if (!all.some((l) => /(about \d+ min|less than a minute|about \d+ h \d\d min) left$/.test(l))) throw new Error('no time left in any line: ' + all.join(' | '));
  if (!(top > 0)) throw new Error('the bar never moved');
  if (all.some((l) => /premium|tier|DeepFilter|undefined|NaN/i.test(l))) throw new Error('odd words: ' + all.join(' | '));
  return `${all.length} different lines, e.g. "${all[Math.floor(all.length / 2)]}"; bar up to ${Math.round(top)}%${sawQueue ? '; was in line first' : ''}`;
});

await check('done: the result panel, the file comes down by itself (once, FLAC), the card says the cost and when the files are cleared away, Download again works', async () => {
  await waitCardDone();
  await card().getByTestId('result').waitFor({ timeout: 90000 });
  await waitDownloads(1);
  if (got.length !== 1 || !/^noisy.*\.flac$/.test(got[0])) throw new Error('downloaded by itself: ' + got.join(', '));
  const text = await card().innerText();
  if (!/Cost \$0\.0\d+/.test(text)) throw new Error('no cost: ' + text.slice(-200));
  const expiry = (await card().getByTestId('expiry').innerText()).trim();
  if (!/^Files are cleared away in (2 h 00|1 h 5\d) min$/.test(expiry)) throw new Error('expiry: ' + expiry);
  got.length = 0;
  await card().getByTestId('download-again').click();
  await waitDownloads(1);
  if (got.length !== 1 || !/\.flac$/.test(got[0])) throw new Error('Download again: ' + got.join(', '));
  await page.screenshot({ path: join(SHOTS, 'jobs-page-done.png'), fullPage: true });
  return `${expiry}; ${got[0]}`;
});

await check('reload keeps the page: the same address and job, the result panel and Download again (no new note, no download by itself); Listen and Change and save again are there', async () => {
  const url = page.url();
  got.length = 0;
  await page.reload();
  await card().waitFor({ timeout: 20000 });
  if (page.url() !== url) throw new Error(`address ${page.url()}`);
  if ((await card().getAttribute('data-job-id')) !== A || (await status(card())) !== 'done') throw new Error('not the same finished job');
  if (await page.getByTestId('submitted').count()) throw new Error('the note came back after a reload');
  await page.locator('[data-testid=result], [data-testid=result-pending]').first().waitFor();
  await sleep(1500);
  if (got.length) throw new Error('a reload downloaded by itself: ' + got.join(', '));
  for (const id of ['download-again', 'listen', 'again']) if (!(await card().getByTestId(id).count())) throw new Error(`${id} is missing after a reload`);
  await card().getByTestId('download-again').click();
  await waitDownloads(1);
  if (got.length !== 1 || !/\.flac$/.test(got[0])) throw new Error('Download again after a reload: ' + got.join(', '));
  return `${got[0]} again after the reload`;
});

await check('back and forward between /app and the job page land on the same job, with its result; "Your jobs" and the list\'s Open link both lead back to it', async () => {
  await page.getByTestId('all-jobs').click();
  await page.waitForURL(/\/app$/);
  await page.getByTestId('job-row').first().waitFor({ timeout: 15000 });
  await page.goBack();
  await page.waitForURL(JOB_PATH);
  await card().waitFor();
  if ((await card().getAttribute('data-job-id')) !== A || (await status(card())) !== 'done') throw new Error('back: not the same finished job');
  await card().getByTestId('download-again').waitFor();
  await page.goForward();
  await page.waitForURL(/\/app$/);
  await page.locator(`[data-testid=job-row][data-job-id="${A}"]`).waitFor({ timeout: 15000 });
  await page.locator(`[data-testid=job-row][data-job-id="${A}"]`).getByTestId('job-open').click();
  await page.waitForURL(JOB_PATH);
  if (jobIdOf(page) !== A) throw new Error('Open went to ' + jobIdOf(page));
  await card().getByTestId('download-again').waitFor();
  return 'same job both ways';
});

// ---------------------------------------------------------------------------------------------------- 2. the list on /app

let B1;
let B2;
await check('/app lists a short row for each job (no full cards): an active job has a progress bar and a line with its step and the time left, a queued one says "In line"', async () => {
  got.length = 0;
  await page.getByTestId('start-another').click();
  await page.waitForURL(/\/app$/);
  B1 = await startPaidJob();
  await page.getByTestId('start-another').click();
  await page.waitForURL(/\/app$/);
  B2 = await startPaidJob();
  await page.getByTestId('all-jobs').click();
  await page.waitForURL(/\/app$/);
  const rowOf = (id) => page.locator(`[data-testid=job-row][data-job-id="${id}"]`);
  for (const id of [A, B1, B2]) await rowOf(id).waitFor({ timeout: 15000 });
  if (await page.getByTestId('job').count()) throw new Error('a full job card is still on the list');
  const seen = new Map(); // row -> lines
  let barSeen = 0;
  let inLine = '';
  const t0 = Date.now();
  while (Date.now() - t0 < 60000) {
    let active = 0;
    for (const id of [B1, B2]) {
      const st = await status(rowOf(id));
      if (!ACTIVE.includes(st)) continue;
      active++;
      const bar = rowOf(id).getByTestId('row-progress');
      const line = rowOf(id).getByTestId('row-line');
      if (!(await bar.count()) || !(await line.count())) throw new Error(`${st} row without its bar or line`);
      barSeen++;
      const text = (await line.innerText({ timeout: 2000 }).catch(() => '')).trim();
      if (text) seen.set(id, [...(seen.get(id) || []), text]);
      if (st === 'queued' && text) inLine = text;
    }
    const withTime = [...seen.values()].flat().some((l) => /left$/.test(l));
    if (!active || (withTime && barSeen > 4)) break;
    await sleep(400);
  }
  const lines = [...seen.values()].flat();
  if (!lines.length) throw new Error('no active row showed a line (both jobs ended before the list was looked at?)');
  if (!lines.some((l) => /(about \d+ min|less than a minute|about \d+ h \d\d min) left$/.test(l))) throw new Error('no time left in any row line: ' + [...new Set(lines)].join(' | '));
  if (inLine && !/^In line/.test(inLine)) throw new Error('a queued row says: ' + inLine);
  const row = await rowOf(A).innerText();
  if (/Download again|Delete now|Change and save again/.test(row)) throw new Error('a row has the card\'s buttons: ' + row);
  await page.screenshot({ path: join(SHOTS, 'jobs-list-active.png'), fullPage: true });
  return `"${[...new Set(lines)].slice(-2).join('" / "')}"${inLine ? `; queued row: "${inLine}"` : ''}`;
});

await check('a job that ends while the list is on screen downloads by itself; its row turns Done with "Files are cleared away in 2 h 00 min", and Open (and the title) go to the job\'s page', async () => {
  const rowOf = (id) => page.locator(`[data-testid=job-row][data-job-id="${id}"]`);
  for (const id of [B1, B2]) await page.waitForFunction((x) => document.querySelector(`[data-testid=job-row][data-job-id="${x}"]`)?.getAttribute('data-status') === 'done', id, { timeout: 240000 });
  await waitDownloads(2, 60000);
  if (got.length !== 2 || !got.every((n) => /^noisy.*\.flac$/.test(n))) throw new Error('downloaded by itself while on the list: ' + got.join(', '));
  const done = rowOf(B2);
  const expiry = (await done.getByTestId('row-expiry').innerText()).trim();
  if (!/^Files are cleared away in (2 h 00|1 h 59) min$/.test(expiry)) throw new Error('countdown: ' + expiry);
  if (await done.getByTestId('row-progress').count()) throw new Error('a done row still has a progress bar');
  const old = (await rowOf(A).getByTestId('row-expiry').innerText()).trim();
  if (!/^Files are cleared away in (2 h 00|1 h \d\d) min$/.test(old)) throw new Error('the older job\'s countdown: ' + old);
  const href = await done.getByTestId('job-open-title').getAttribute('href');
  if (href !== `/app/job/${B2}`) throw new Error('title link ' + href);
  await done.getByTestId('job-open').click();
  await page.waitForURL(JOB_PATH);
  if (jobIdOf(page) !== B2) throw new Error('Open went to ' + jobIdOf(page));
  await card().getByTestId('download-again').waitFor({ timeout: 20000 });
  return `${expiry}; the older job: ${old}; 2 files by themselves`;
});

// ---------------------------------------------------------------------------------------------------- 3. start another, delete now

await check('"Start another file" goes back to the form with nothing chosen; "Your jobs" goes to the list', async () => {
  await page.getByTestId('start-another').click();
  await page.waitForURL(/\/app$/);
  await page.locator('input[type=file]').waitFor({ state: 'attached', timeout: 10000 });
  if (await page.getByTestId('start').count()) throw new Error('a file is still chosen');
  await page.goBack();
  await page.waitForURL(JOB_PATH);
  await page.getByTestId('all-jobs').click();
  await page.waitForURL(/\/app$/);
  await page.locator('[data-testid=job-row]').first().waitFor();
  return 'form, then list';
});

await check('Delete now asks first ("Keep them" backs out), "Yes, delete" returns to /app, the row is gone, the old address says the job is gone, and its files are really deleted', async () => {
  await page.goto(`${BASE}/app/job/${B2}`);
  await card().getByTestId('download-again').waitFor({ timeout: 20000 });
  const outName = await card().getByTestId('made-file').first().locator('.oname').innerText();
  await card().getByRole('button', { name: 'Delete now' }).click();
  await card().getByTestId('delete-confirm').waitFor();
  const ask = await card().getByTestId('delete-confirm').innerText();
  if (!/Delete everything of this file now/.test(ask) || !/can’t be brought back/.test(ask)) throw new Error('confirmation: ' + ask);
  await card().getByRole('button', { name: 'Keep them' }).click();
  if (!(await card().getByRole('button', { name: 'Delete now' }).count())) throw new Error('"Keep them" did not go back');
  if ((await apiCall(page, 'GET', `jobs/${B2}`)).body.status !== 'done') throw new Error('deleted without a Yes');
  await card().getByRole('button', { name: 'Delete now' }).click();
  await card().getByRole('button', { name: 'Yes, delete' }).click();
  await page.waitForURL(/\/app$/, { timeout: 15000 });
  await page.getByTestId('jobs-note').waitFor();
  await page.locator(`[data-testid=job-row][data-job-id="${B1}"]`).waitFor({ timeout: 15000 }); // the other job is still there
  if (await page.locator(`[data-testid=job-row][data-job-id="${B2}"]`).count()) throw new Error('the deleted job is still listed');
  await page.goto(`${BASE}/app/job/${B2}`);
  await page.getByTestId('job-gone').waitFor({ timeout: 15000 });
  const gone = await page.getByTestId('job-gone').innerText();
  if (!/isn’t here any more/.test(gone) || !/2 hours/.test(gone)) throw new Error('gone page: ' + gone);
  if (await page.getByTestId('job').count()) throw new Error('the job card is shown for a deleted job');
  const file = await apiCall(page, 'GET', `jobs/${B2}/outputs/${encodeURIComponent(outName)}`);
  if (file.status !== 404) throw new Error(`the result is still downloadable (${file.status})`);
  const dataDir = process.env.ACW_DATA_DIR;
  if (dataDir && existsSync(join(dataDir, 'jobs', B2))) throw new Error('the job\'s folder is still on disk');
  await page.getByTestId('job-gone').getByRole('link', { name: 'Start another file' }).click();
  await page.waitForURL(/\/app$/);
  return `deleted ${outName}; gone page: ${gone.split('\n')[0]}`;
});

// ---------------------------------------------------------------------------------------------------- 4. at most 3 at once

const hold = (n) => ({ kind: 'transcribe', transcribeTier: 'local', filename: `hold${n}.wav`, sizeBytes: 1000, durationSeconds: 5, options: { audio_format: 'wav', caption_formats: ['txt'] } });
const held = [];

await check('the list answers {jobs, limits}: 3 jobs at once, files kept 2 hours; three jobs that wait for their upload fill the account', async () => {
  const l = await listJobs(page);
  if (!Array.isArray(l.jobs) || l.limits?.maxActiveJobs !== 3 || l.limits?.retentionSeconds !== 7200) throw new Error(JSON.stringify(l.limits));
  const active = l.jobs.filter((j) => ACTIVE.includes(j.status)).length;
  if (active) throw new Error(`${active} jobs are still active before the test`);
  for (let i = 1; i <= 3; i++) {
    const r = await apiCall(page, 'POST', 'jobs', hold(i));
    if (r.status !== 200) throw new Error(`job ${i}: ${r.status} ${JSON.stringify(r.body)}`);
    held.push(r.body.id);
  }
  return `limits ${JSON.stringify(l.limits)}; 3 held`;
});

await check('the form warns before Start when the list already shows 3 active jobs: exactly the server\'s message, and Start stays off', async () => {
  await gotoApp('/app');
  await page.waitForFunction(() => document.querySelectorAll('[data-testid=job-row][data-status=uploading]').length === 3, null, { timeout: 15000 });
  const first = page.locator('[data-testid=job-row][data-status=uploading]').first();
  if (!(await first.getByTestId('row-progress').count())) throw new Error('an uploading row has no bar');
  if (!/upload/i.test(await first.getByTestId('row-line').innerText())) throw new Error('row line: ' + await first.getByTestId('row-line').innerText());
  await dropFile(page, join(FILES, 'noisy.wav'));
  await pick(page, 'noise', 'denoise');
  await page.getByTestId('estimate').waitFor();
  const warn = page.locator('.banner-warn', { hasText: 'files being processed' });
  await warn.waitFor({ timeout: 10000 });
  const text = (await warn.innerText()).trim();
  if (text !== TOO_MANY) throw new Error(`"${text}"`);
  if (await page.getByTestId('start').isEnabled()) throw new Error('Start is on with 3 files being processed');
  // the free choices are not held back
  await pick(page, 'noise', 'browser');
  if (await page.locator('.banner-warn', { hasText: 'files being processed' }).count()) throw new Error('the warning shows for the free cleanup');
  if (!(await page.getByTestId('start').isEnabled())) throw new Error('free Start is off');
  return text;
});

await check('the server says the same when the page did not know (a list that came back empty): Start gives exactly that message, nothing is created', async () => {
  await page.route('**/api/jobs', (r) => (r.request().method() === 'GET' ? r.fulfill({ json: { jobs: [] } }) : r.continue()));
  try {
    await gotoApp('/app');
    await dropFile(page, join(FILES, 'noisy.wav'));
    await pick(page, 'noise', 'denoise');
    await page.getByTestId('estimate').waitFor();
    if (await page.locator('.banner-warn', { hasText: 'files being processed' }).count()) throw new Error('warned although the list said nothing');
    // counted by the server itself (the page's own list is the one that is being made empty)
    const serverCount = async () => (await (await context.request.get(`${BASE}/api/jobs`)).json()).jobs.length;
    const before = await serverCount();
    await page.getByTestId('start').click();
    const err = page.getByTestId('job-error');
    await err.waitFor({ timeout: 60000 });
    const text = (await err.innerText()).trim();
    if (text !== TOO_MANY) throw new Error(`"${text}"`);
    if (page.url().includes('/app/job/')) throw new Error('moved to a job page although nothing was started');
    if ((await serverCount()) !== before) throw new Error('a job was created');
    return text;
  } finally {
    await page.unroute('**/api/jobs').catch(() => {});
  }
});

await check('the API: a 4th file job is refused with 429, the code too_many_active_jobs (in the answer and in X-Error-Code) and the same message; freeing one place lets the next one in', async () => {
  const r = await apiCall(page, 'POST', 'jobs', hold(4));
  if (r.status !== 429) throw new Error(`${r.status} ${JSON.stringify(r.body)}`);
  if (r.body.code !== 'too_many_active_jobs' || r.headers['x-error-code'] !== 'too_many_active_jobs') throw new Error(`code ${r.body.code} / header ${r.headers['x-error-code']}`);
  if (r.body.detail !== TOO_MANY) throw new Error(`detail "${r.body.detail}"`);
  if (!r.headers['retry-after']) throw new Error('no Retry-After');
  const cancel = await apiCall(page, 'POST', `jobs/${held[0]}/cancel`);
  if (cancel.status !== 200) throw new Error('cancel ' + cancel.status);
  const ok = await apiCall(page, 'POST', 'jobs', hold(4));
  if (ok.status !== 200) throw new Error(`after freeing a place: ${ok.status} ${JSON.stringify(ok.body)}`);
  held.push(ok.body.id);
  const again = await apiCall(page, 'POST', 'jobs', hold(5));
  if (again.status !== 429) throw new Error('a 4th is allowed again: ' + again.status);
  // a canceled job no longer counts: the form is open again (2 are active, the list is told)
  const rest = [held[1], held[2], held[3]];
  for (const id of rest) await apiCall(page, 'POST', `jobs/${id}/cancel`);
  const l = await listJobs(page);
  if (l.jobs.some((j) => ACTIVE.includes(j.status))) throw new Error('jobs are still active after all were canceled');
  await gotoApp('/app');
  await dropFile(page, join(FILES, 'noisy.wav'));
  await pick(page, 'noise', 'denoise');
  await page.getByTestId('estimate').waitFor();
  if (await page.locator('.banner-warn', { hasText: 'files being processed' }).count()) throw new Error('the warning stays after the jobs ended');
  if (!(await page.getByTestId('start').isEnabled())) throw new Error('Start stays off after the jobs ended');
  return `${r.status} ${r.body.code}, "${r.body.detail}"`;
});

// ---------------------------------------------------------------------------------------------------- 5. past its 2 hours

await check('the countdown on a row and on the job\'s page follows the job\'s own end: with 20 minutes left both say so', async () => {
  pyRun(`
import time
from acw import db
t = int(time.time())
db.execute("UPDATE jobs SET finished_at = ?, expires_at = ? WHERE id = ?", (t - 6000, t + 1200, ${JSON.stringify(A)}))
`);
  await gotoApp('/app');
  const row = page.locator(`[data-testid=job-row][data-job-id="${A}"]`);
  await row.waitFor({ timeout: 15000 });
  const inRow = (await row.getByTestId('row-expiry').innerText()).trim();
  if (!/^Files are cleared away in (19|20) min 00 s$/.test(inRow)) throw new Error('row: ' + inRow);
  await page.goto(`${BASE}/app/job/${A}`);
  await card().getByTestId('expiry').waitFor({ timeout: 15000 });
  const onPage = (await card().getByTestId('expiry').innerText()).trim();
  if (!/^Files are cleared away in (19|20) min 00 s$/.test(onPage)) throw new Error('job page: ' + onPage);
  return `${inRow} / ${onPage}`;
});

await check('a job past its 2 hours is not listed and its page says it is gone (the sweep marks it expired and clears its files)', async () => {
  const before = (await listJobs(page)).jobs.map((j) => j.id);
  if (!before.includes(A)) throw new Error('the first job is not listed to begin with');
  pyRun(`
import time
from acw import db
t = int(time.time())
db.execute("UPDATE jobs SET finished_at = ?, expires_at = ? WHERE id = ?", (t - 7300, t - 100, ${JSON.stringify(A)}))
`);
  if ((await listJobs(page)).jobs.some((j) => j.id === A)) throw new Error('listed after its 2 hours');
  pyRun('from acw import jobs\njobs.sweep()\n');
  const st = (await apiCall(page, 'GET', `jobs/${A}`)).body.status;
  if (st !== 'expired') throw new Error('status after the sweep: ' + st);
  await gotoApp('/app');
  await page.getByTestId('jobs-note').waitFor();
  await page.locator('[data-testid=job-row]').first().waitFor({ timeout: 10000 }).catch(() => {});
  if (await page.locator(`[data-testid=job-row][data-job-id="${A}"]`).count()) throw new Error('an expired job is on the list');
  await page.goto(`${BASE}/app/job/${A}`);
  await page.getByTestId('job-gone').waitFor({ timeout: 15000 });
  if (await page.getByTestId('job').count()) throw new Error('a card for an expired job');
  const dataDir = process.env.ACW_DATA_DIR;
  if (dataDir && existsSync(join(dataDir, 'jobs', A))) throw new Error('the expired job\'s folder is still on disk');
  const r = await apiCall(page, 'GET', `jobs/${A}`);
  return `status ${r.body.status}; page: ${(await page.getByTestId('job-gone').innerText()).split('\n')[0]}`;
});

console.log('LOGS:\n' + logs.filter((l) => !/Failed to load resource|favicon|net::ERR/.test(l)).join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
