// The per-part editor (the one editor screen: before a file runs, while it runs, and on a finished job's page).
//  1. a finished job's page IS the editor: its file came down by itself, Download again, play and the speaker buttons, the inspector;
//     its waveform, its tracks, and leaving the page and coming back keeps the part that was chosen;
//  2. choosing a part: dragging across the waveform, the handles (mouse and keyboard), the range box (steppers, Use playhead, Play part),
//     Select all, Escape; the merge rules on a file that has not run yet (a change cuts at the part's edges, neighbours that are the same
//     join by themselves, a split stays until a change touches it, Select more, Mixed, undo / redo) and Zoom;
//  3. painting parts (a new one replaces what it covers) and the rows (a click on a section chooses exactly it, a drag across a row chooses a part);
//  4. hearing what a part gets after processing: the speaker buttons, `data-gains` (strength as a blend, a switched-off track drops out,
//     the original), "Hearing: …" and the heard line;
//  5. the price after a change: Save changes · $ from `POST …/quote` a moment after the last change, a paid save is a `process` with the shown
//     price on the SAME job, the ledger has one line, a free save is an `export`; the plain messages when it can't be paid or is refused;
//     a job made through the API with no timeline opens in the editor;
//  6. suggestions made by the server: the chip, the pins, the card (Apply / Dismiss), undo, dismissed ones stay dismissed after a reload;
//  7. the start screen with parts: the estimate (`POST /api/jobs/estimate` with the timeline), Details, the job it makes; Voice Off for the
//     whole file with noise removal on one part is only that part; suggestions found in the browser before the upload (a worker, progress,
//     Cancel, "Suggestions appear after the first run." for a file over 30 minutes);
//  8. phone (390x844, touch, light and dark): no sideways scroll, the inspector is a sheet above the bar, tap / long-press / Select more;
//  9. the words: no avoided words anywhere in the editor, its popovers and the "All the options" window (which holds "Which one should I
//     pick?"); the automatic settings say which one they picked and why; echo removal has its own price and its own row.
// Needs the test backend (tests/run.sh editor) and the fixtures. EDITOR_SHOTS=<dir> also saves screenshots. Usage: node tests/editor.mjs
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import {
  BASE, FFMPEG, FILES, JOB_PATH, admin, apiCall, check, dragRange, dropFile, jobIdOf, launch, mainButton, pick, pyRun, selectionTitle, setAuto, setKind, setLeveling,
  setStrength, setSubs, setVoice, signIn, studio, summary, waitCardDone, waitJobDone, waitResult,
} from './helpers.mjs';

const EMAIL = `editor-${Date.now()}@example.com`;
const SHOTS = process.env.EDITOR_SHOTS || '';
if (SHOTS) mkdirSync(SHOTS, { recursive: true });
const BANNED = [/DeepFilterNet/i, /\bDFN/i, /\bWhisper\b/, /Parakeet/i, /Sidon/i, /OpenAI/i, /WebAssembly/i, /\bwasm\b/i, /\bour servers?\b/i, /premium/i, /\bpolish(ed|es|ing)?\b|\bPolish (your|the|it|up|this|that|my)\b/, /\bmodel\b/i, /\btiers?\b/i, /undefined|NaN/];
const AUTO_KEY = 'acw.autoSettings'; // this browser's choice for "Automatic" (the editor keeps it)

const { browser, context, page, logs } = await launch();
const got = [];
page.on('download', (d) => got.push(d.suggestedFilename()));
const requests = [];
const onRequest = (r) => { const u = new URL(r.url()); if (u.pathname.startsWith('/api/')) requests.push({ m: r.method(), p: u.pathname + u.search, body: r.postData() }); };
context.on('request', onRequest);
const quoteAnswers = [];
context.on('response', async (res) => {
  const u = new URL(res.url());
  if (res.request().method() !== 'POST' || !/\/api\/jobs\/[0-9a-f]+\/quote$/.test(u.pathname)) return;
  try { quoteAnswers.push({ path: u.pathname, body: JSON.parse(res.request().postData()), answer: await res.json() }); } catch { /* not an answer */ }
});
const lastQuote = () => quoteAnswers.at(-1);
await signIn(page, EMAIL);
admin('grant', EMAIL, '5', 'editor test');
const sleep = (ms) => page.waitForTimeout(ms);
const near = (a, b, tol, what) => { if (!(Math.abs(a - b) <= tol)) throw new Error(`${what}: ${a} vs ${b}`); };
const asked = (m, re) => requests.filter((r) => r.m === m && re.test(r.p));
const shot = async (name, p = page) => { if (SHOTS) await p.screenshot({ path: join(SHOTS, `${name}.png`) }); };
const gotoApp = async (path = '/app', p = page) => { await p.goto(`${BASE}${path}`); await p.getByTestId('available').or(p.getByTestId('job-page')).first().waitFor({ timeout: 20000 }); };
const waitDownloads = async (n, timeout = 60000) => { const t0 = Date.now(); while (got.length < n && Date.now() - t0 < timeout) await sleep(250); await sleep(600); };
const balance = async () => (await apiCall(page, 'GET', 'me')).body.balance?.available_micros ?? (await apiCall(page, 'GET', 'billing/balance')).body?.available_micros;
const autoOn = () => page.evaluate((k) => localStorage.setItem(k, 'on'), AUTO_KEY);
/** The prototype's price words (the price on the main button): "Free", "<$0.001", "$0.024", "$1.50". Micros of a dollar. */
const money = (m) => (m <= 0 ? 'Free' : m < 1000 ? '<$0.001' : `$${m >= 100000 || m % 10000 === 0 ? (m / 1e6).toFixed(2) : (m / 1e6).toFixed(3)}`);
const clock = (t) => { t = Math.round(t); return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`; };

/** The sections of a track, left to right: [{s, e, text, hard}]. */
const sections = (tr, p = page) => p.locator(`[data-testid=track-${tr}] [data-testid=section]`).evaluateAll((els) => els.map((e) => ({ s: Number(e.dataset.s), e: Number(e.dataset.e), text: e.textContent.trim(), hard: e.classList.contains('hard') })));
/** "Max 0:00–0:10 | Strong 0:10–0:15": what a track says. */
const say = async (tr, p = page) => (await sections(tr, p)).map((x) => `${x.text} ${clock(x.s)}–${clock(x.e)}`).join(' | ');
const waveSeconds = async (p = page) => Number(await p.getByTestId('wave').getAttribute('data-seconds'));
/** Drag across the waveform from second s to second e (the waveform is brought into view first: the page may have scrolled to a button far down the inspector). */
const drag = async (s, e, p = page) => { await p.getByTestId('waveform').scrollIntoViewIfNeeded(); await dragRange(p, s, e); };
const gains = async () => JSON.parse(await studio(page).getAttribute('data-gains'));
const timeNow = async () => Number(await studio(page).getAttribute('data-time'));
/** Click the waveform at second t (moves the playhead). */
async function seekTo(t, p = page) {
  await p.getByTestId('waveform').scrollIntoViewIfNeeded();
  const box = await p.getByTestId('waveform').boundingBox();
  await p.mouse.click(box.x + (t / (await waveSeconds(p))) * box.width, box.y + box.height / 2);
  await p.waitForTimeout(200);
}
/** Drag across a track's row from second s to second e (the row's own sections start the drag). */
async function dragOnRow(tr, s, e) {
  await page.locator(`[data-testid=track-${tr}]`).scrollIntoViewIfNeeded();
  const wave = await page.getByTestId('wave').boundingBox();
  const row = await page.locator(`[data-testid=track-${tr}] .trk`).boundingBox();
  const T = await waveSeconds();
  const x = (t) => wave.x + (t / T) * wave.width;
  const y = row.y + row.height / 2;
  await page.mouse.move(x(s), y);
  await page.mouse.down();
  await page.mouse.move(x((s + e) / 2), y, { steps: 4 });
  await page.mouse.move(x(e), y, { steps: 4 });
  await page.mouse.up();
}
/** A finished job's editor is on the page, with its tracks drawn. */
async function waitEditor(p = page) {
  await p.locator('[data-testid=studio][data-stage=done]').waitFor({ timeout: 40000 });
  await p.getByTestId('hear-voice').waitFor();
  await p.locator('[data-testid=track-voice] [data-testid=section]').first().waitFor();
  await p.waitForFunction(() => document.querySelector('[data-testid=waveform]')?.dataset.reading === 'no', null, { timeout: 40000 });
}
const openJob = async (id, p = page) => { await p.goto(`${BASE}/app/job/${id}`); await waitEditor(p); };
/** The price on Save changes has settled: the quote is in and the button is on. Answers the button's text. */
async function savePrice(p = page) {
  const b = p.getByTestId('save-changes');
  await b.waitFor({ timeout: 15000 });
  await p.waitForFunction(() => { const x = document.querySelector('[data-testid=save-changes]'); return x && !x.disabled; }, null, { timeout: 15000 });
  await p.waitForTimeout(150);
  return (await b.innerText()).replace(/\s+/g, ' ').trim();
}

/** Makes a job from the start screen: Standard noise removal at Max, Leveling (YouTube) on or off, echo removal on or off. Answers its id once the editor says the files are made and the file has come down. */
async function makeJob({ file = 'speech-long.wav', leveling = true, echo = false } = {}) {
  await gotoApp('/app');
  await dropFile(page, join(FILES, file));
  await setKind(page, 'paid');
  await setStrength(page, 'max');
  await setLeveling(page, leveling ? 'youtube' : 'off');
  if (echo) {
    const sw = page.getByTestId('echo-switch');
    if ((await sw.getAttribute('aria-checked')) !== 'true') await sw.click();
  }
  await sleep(300);
  const before = got.length;
  await page.getByTestId('start').click();
  await page.waitForURL(JOB_PATH, { timeout: 90000 });
  const id = jobIdOf(page);
  await waitCardDone(page);
  await waitResult(page, 'result', 120000);
  await waitDownloads(before + 1);
  return id;
}
const J1 = { id: '' };

/* ------------------------------------------------------------------------------------------------ 1. the page of a finished job */

await check('the page of a finished job is the editor: the file came down by itself, Download again, play and the speaker buttons (Listen), the inspector (change and save again), the output and Your files are all there; nothing is priced for nothing changed', async () => {
  J1.id = await makeJob({ file: 'speech-long.wav', leveling: true });
  if (got.length !== 1 || !/\.flac$/.test(got[0])) throw new Error('downloaded: ' + got.join(', '));
  const st = studio(page);
  for (const [k, v] of [['stage', 'done'], ['phase', 'ready'], ['changed', 'no']]) if ((await st.getAttribute(`data-${k}`)) !== v) throw new Error(`data-${k} is ${await st.getAttribute(`data-${k}`)}`);
  for (const id of ['download-again', 'play', 'time', 'hear-voice', 'hear-level', 'inspector', 'waveform', 'output-button', 'bar-files', 'track-voice', 'track-level', 'track-subs']) if (!(await page.getByTestId(id).count())) throw new Error(`${id} is gone`);
  if (await page.getByTestId('hear-echo').count() || await page.getByTestId('track-echo').count()) throw new Error('an Echo track for a file with one audio track');
  if (await page.getByTestId('save-changes').count()) throw new Error('Save changes with nothing changed');
  if ((await page.getByTestId('download-again').innerText()).trim() !== 'Download again') throw new Error('main button: ' + await mainButton(page));
  if (asked('POST', /\/quote$/).length) throw new Error('a quote was asked for with nothing changed');
  // the same job in a second tab (a fresh editor with Automatic on): no suggestions, so no chip
  await autoOn();
  const p2 = await context.newPage();
  try {
    await openJob(J1.id, p2);
    if (await p2.getByTestId('sg-chip').count() || await p2.getByTestId('pin').count()) throw new Error('a suggestion chip for a job without suggestions');
  } finally {
    await p2.close();
  }
  await shot('job-done');
  return 'editor, flac came down, no quote';
});

await check('the editor of a finished job: the real waveform and the heard line, a row for Voice, Leveling and Subtitles (no Echo for a file with one audio track), "Hearing: everything"; leaving the page and coming back keeps the part that was chosen', async () => {
  const wf = page.getByTestId('waveform');
  if ((await wf.getAttribute('data-known')) !== 'yes' || (await wf.getAttribute('data-reading')) !== 'no') throw new Error('no waveform');
  if (!(await page.getByTestId('heard-line').locator('path').getAttribute('d'))) throw new Error('no heard line');
  const names = (await page.locator('[data-testid^=track-]').evaluateAll((els) => els.map((e) => e.dataset.testid))).join('|');
  if (names !== 'track-voice|track-level|track-subs') throw new Error('tracks: ' + names);
  const T = await waveSeconds();
  near(T, 48.4, 0.3, 'length');
  if ((await say('voice')) !== 'Max 0:00–0:48' || (await say('level')) !== 'YouTube 0:00–0:48' || (await say('subs')) !== 'Off 0:00–0:48') throw new Error(`${await say('voice')} / ${await say('level')} / ${await say('subs')}`);
  if ((await page.getByTestId('hearing').innerText()) !== 'Hearing: everything') throw new Error('hearing: ' + await page.getByTestId('hearing').innerText());
  if ((await selectionTitle(page)) !== 'Whole file · 0:48') throw new Error('title ' + await selectionTitle(page));
  await drag(10, 20);
  await page.getByRole('link', { name: /Available balance/ }).click();
  await page.waitForURL(/\/app\/billing$/);
  await page.goBack();
  await waitEditor();
  if ((await selectionTitle(page)) !== '0:10–0:20 · 10 s') throw new Error('the chosen part was lost: ' + await selectionTitle(page));
  await page.keyboard.press('Escape');
  if ((await selectionTitle(page)) !== 'Whole file · 0:48') throw new Error('Escape did not clear the part: ' + await selectionTitle(page));
  await shot('editor-open');
  return 'waveform, three tracks, part kept across a visit to Billing';
});

/* ------------------------------------------------------------------------------------------------ 2. choosing a part */

await check('a part is chosen by dragging across the waveform, by its handles (mouse and arrow keys), by the range box (steppers, Use playhead), Play part plays it and stops at its end, Select all and Escape give the whole file back', async () => {
  const T = await waveSeconds();
  await drag(20, 30);
  if ((await selectionTitle(page)) !== '0:20–0:30 · 10 s') throw new Error('drag: ' + await selectionTitle(page));
  if ((await page.getByTestId('time-s').innerText()) !== '0:20' || (await page.getByTestId('time-e').innerText()) !== '0:30') throw new Error('range box');
  // the handles, with the mouse
  const box = await page.getByTestId('waveform').boundingBox();
  const x = (t) => box.x + (t / T) * box.width;
  const end = await page.getByTestId('handle-end').boundingBox();
  await page.mouse.move(end.x + end.width / 2, end.y + end.height / 2);
  await page.mouse.down();
  await page.mouse.move(x(36), end.y + end.height / 2, { steps: 6 });
  await page.mouse.up();
  if ((await page.getByTestId('time-e').innerText()) !== '0:36') throw new Error('handle end: ' + await page.getByTestId('time-e').innerText());
  const start = await page.getByTestId('handle-start').boundingBox();
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
  await page.mouse.down();
  await page.mouse.move(x(14), start.y + start.height / 2, { steps: 6 });
  await page.mouse.up();
  if ((await page.getByTestId('time-s').innerText()) !== '0:14') throw new Error('handle start: ' + await page.getByTestId('time-s').innerText());
  // ... and with the arrow keys (a second at a time)
  await page.getByTestId('handle-end').focus();
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowLeft');
  if ((await page.getByTestId('time-e').innerText()) !== '0:37') throw new Error('handle end by keys: ' + await page.getByTestId('time-e').innerText());
  await page.getByTestId('handle-start').focus();
  await page.keyboard.press('ArrowRight');
  if ((await page.getByTestId('time-s').innerText()) !== '0:15') throw new Error('handle start by keys: ' + await page.getByTestId('time-s').innerText());
  // the steppers (a second at a time)
  await page.getByTestId('nudge-s-plus').click();
  await page.getByTestId('nudge-s-plus').click();
  await page.getByTestId('nudge-e-minus').click();
  if ((await selectionTitle(page)) !== '0:17–0:36 · 19 s') throw new Error('steppers: ' + await selectionTitle(page));
  await page.getByTestId('nudge-s-minus').click();
  await page.getByTestId('nudge-e-plus').click();
  if ((await selectionTitle(page)) !== '0:16–0:37 · 21 s') throw new Error('steppers back: ' + await selectionTitle(page));
  // Use playhead: click the waveform (no drag) to move it, then take it as the start, then as the end (the part stays a second long at least)
  await seekTo(8);
  near(await timeNow(), 8, 0.7, 'playhead after a click');
  await page.getByTestId('use-playhead-s').click();
  if ((await page.getByTestId('time-s').innerText()) !== '0:08') throw new Error('start at the playhead: ' + await page.getByTestId('time-s').innerText());
  await page.getByTestId('use-playhead-e').click();
  if ((await selectionTitle(page)) !== '0:08–0:09 · 1 s') throw new Error('an end not after the start: ' + await selectionTitle(page));
  // Play part: it plays the part and stops at its end
  await page.getByTestId('nudge-e-plus').click();
  await page.getByTestId('nudge-e-plus').click();
  if ((await selectionTitle(page)) !== '0:08–0:11 · 3 s') throw new Error('part for Play part: ' + await selectionTitle(page));
  await page.getByTestId('play-part').click();
  await page.waitForFunction(() => document.querySelector('[data-testid=studio]')?.dataset.playing === 'yes', null, { timeout: 20000 });
  await page.waitForFunction(() => document.querySelector('[data-testid=studio]')?.dataset.playing === 'no', null, { timeout: 20000 });
  near(await timeNow(), 11, 0.5, 'Play part stops at the end of the part');
  // Select all, and Escape
  await page.getByTestId('select-all').click();
  if ((await selectionTitle(page)) !== 'Whole file · 0:48' || (await page.getByTestId('select-all').getAttribute('aria-pressed')) !== 'true') throw new Error('select all: ' + await selectionTitle(page));
  if (await page.getByTestId('range-box').count() === 0) throw new Error('the whole file has its range box too (handles at the ends)');
  await drag(20, 30);
  await page.keyboard.press('Escape');
  if ((await selectionTitle(page)) !== 'Whole file · 0:48') throw new Error('Escape: ' + await selectionTitle(page));
  return 'drag, handles (mouse, keys), steppers, Use playhead, Play part, Select all, Escape';
});


await check('the merge rules on a file that has not run yet: a change cuts at the part\'s edges, neighbours that end up the same join by themselves, a split stays until a change touches it, Shift adds a range, Mixed, undo and redo, Zoom', async () => {
  await gotoApp('/app');
  await dropFile(page, join(FILES, 'speech-long.wav'));
  await setKind(page, 'paid');
  await setStrength(page, 'max');
  await setLeveling(page, 'youtube');
  const only = async (tr) => (await sections(tr)).length;
  if ((await say('voice')) !== 'Max 0:00–0:48' || (await say('level')) !== 'YouTube 0:00–0:48') throw new Error(`${await say('voice')} / ${await say('level')}`);
  // a change applies to the part and cuts at its edges; the other tracks stay as they are
  await drag(15, 25);
  if ((await selectionTitle(page)) !== '0:15–0:25 · 10 s') throw new Error('the part was not chosen: ' + await selectionTitle(page));
  await setStrength(page, 'strong');
  if ((await say('voice')) !== 'Max 0:00–0:15 | Strong 0:15–0:25 | Max 0:25–0:48') throw new Error('cut: ' + await say('voice'));
  if ((await only('level')) !== 1 || (await only('subs')) !== 1) throw new Error('another track was cut too');
  // the same as its neighbours again: one section
  await setStrength(page, 'max');
  if ((await say('voice')) !== 'Max 0:00–0:48') throw new Error('merge: ' + await say('voice'));
  await setLeveling(page, 'podcast');
  if ((await say('level')) !== 'YouTube 0:00–0:15 | Podcast 0:15–0:25 | YouTube 0:25–0:48' || (await only('voice')) !== 1) throw new Error('level cut: ' + await say('level'));
  await setLeveling(page, 'youtube');
  if ((await only('level')) !== 1) throw new Error('level merge: ' + await say('level'));
  // a split cuts every track and stays until a change touches it
  await page.getByTestId('toast').waitFor({ state: 'hidden', timeout: 5000 }).catch(() => {});
  await page.keyboard.press('Escape');
  await page.getByTestId('split').click(); // (the playhead is at 0:00: nothing to split)
  if (!/inside the file/.test(await page.getByTestId('toast').innerText())) throw new Error('no word about splitting at the very start');
  await seekTo(30);
  await page.getByTestId('split').click();
  const v = await sections('voice');
  if (v.length !== 2 || !v[1].hard || (await only('level')) !== 2 || (await only('subs')) !== 2) throw new Error('split: ' + JSON.stringify(v));
  if ((await selectionTitle(page)) !== '0:30–0:48 · 18 s') throw new Error('the piece after the split is chosen: ' + await selectionTitle(page));
  await setStrength(page, 'strong');
  const after = await sections('voice');
  if ((await say('voice')) !== 'Max 0:00–0:30 | Strong 0:30–0:48' || after.some((x) => x.hard) || after.length !== 2) throw new Error('change at a split: ' + await say('voice'));
  await setStrength(page, 'max');
  if ((await only('voice')) !== 1 || (await only('level')) !== 1) throw new Error('the split joined again: ' + await say('voice') + ' / ' + await say('level'));
  // Shift adds a range; a change applies to both; Shift-click on a section takes its range out and puts it back
  await drag(5, 10);
  await page.keyboard.down('Shift');
  await drag(20, 25);
  await page.keyboard.up('Shift');
  if ((await selectionTitle(page)) !== '2 ranges · 0:10 total') throw new Error('Shift-drag: ' + await selectionTitle(page));
  await setStrength(page, 'gentle');
  if ((await say('voice')) !== 'Max 0:00–0:05 | Gentle 0:05–0:10 | Max 0:10–0:20 | Gentle 0:20–0:25 | Max 0:25–0:48') throw new Error('two ranges: ' + await say('voice'));
  await page.keyboard.down('Shift');
  await page.locator('[data-testid=track-voice] [data-testid=section]').nth(1).click();
  await page.keyboard.up('Shift');
  if ((await selectionTitle(page)) !== '0:20–0:25 · 5 s') throw new Error('Shift-click took the range out: ' + await selectionTitle(page));
  await page.keyboard.down('Shift');
  await page.locator('[data-testid=track-voice] [data-testid=section]').nth(1).click();
  await page.keyboard.up('Shift');
  if ((await selectionTitle(page)) !== '2 ranges · 0:10 total') throw new Error('Shift-click put the range back: ' + await selectionTitle(page));
  // Mixed: a selection over parts that differ
  await page.getByTestId('select-all').click();
  if (!(await page.locator('.studio .mx').filter({ hasText: 'Mixed' }).count())) throw new Error('no "Mixed" for parts that differ');
  if (await page.locator('.studio button[data-act=v-str][aria-pressed=true]').count()) throw new Error('a strength is pressed for a selection that has two');
  // undo and redo (buttons and keys)
  await page.getByTestId('undo').click();
  if ((await only('voice')) !== 1) throw new Error('undo: ' + await say('voice'));
  await page.getByTestId('redo').click();
  if ((await only('voice')) !== 5) throw new Error('redo: ' + await say('voice'));
  await page.keyboard.press('Control+z');
  if ((await only('voice')) !== 1) throw new Error('Ctrl+Z: ' + await say('voice'));
  await page.keyboard.press('Control+Shift+z');
  if ((await only('voice')) !== 5) throw new Error('Ctrl+Shift+Z: ' + await say('voice'));
  // Zoom: the rows get wider than the screen and scroll together; at 1 there is nothing to zoom out
  const tlw = page.locator('#tlw');
  if (await page.getByTestId('zoom-out').isEnabled()) throw new Error('Zoom out is on at the normal size');
  const w1 = await page.getByTestId('wave').evaluate((e) => e.offsetWidth);
  await page.getByTestId('zoom-in').click();
  await page.getByTestId('zoom-in').click();
  if (!(await tlw.evaluate((e) => e.classList.contains('zoomed'))) || (await tlw.evaluate((e) => getComputedStyle(e).getPropertyValue('--zoom').trim())) !== '4') throw new Error('zoom class');
  const w4 = await page.getByTestId('wave').evaluate((e) => e.offsetWidth);
  near(w4 / w1, 4, 0.2, 'the rows are four times wider');
  await page.getByTestId('zoom-out').click();
  await page.getByTestId('zoom-out').click();
  if (await tlw.evaluate((e) => e.classList.contains('zoomed')) || await page.getByTestId('zoom-out').isEnabled()) throw new Error('zoom did not go back');
  return 'cut, merge, split, Shift, Mixed, undo / redo, zoom';
});

/* ------------------------------------------------------------------------------------------------ 3. painting and the rows */

await check('painting parts on a finished job: a new part replaces what it covers (Max, Strong 0:10–0:15, Gentle 0:15–0:25, Max), one section for the base; a click on a section chooses exactly it and a drag across a row chooses a part', async () => {
  await openJob(J1.id);
  const base = await sections('voice');
  if (base.length !== 1 || base[0].text !== 'Max' || base[0].s !== 0) throw new Error('base: ' + await say('voice'));
  await drag(10, 20);
  await setStrength(page, 'strong');
  await drag(15, 25);
  await setStrength(page, 'gentle');
  const texts = await say('voice');
  if (texts !== 'Max 0:00–0:10 | Strong 0:10–0:15 | Gentle 0:15–0:25 | Max 0:25–0:48') throw new Error('parts: ' + texts);
  const s = await sections('voice');
  if (s[1].e !== s[2].s || s[0].e !== s[1].s) throw new Error('the parts do not touch: ' + JSON.stringify(s));
  if ((await studio(page).getAttribute('data-changed')) !== 'tracks') throw new Error('data-changed ' + await studio(page).getAttribute('data-changed'));
  await shot('editor-parts');
  // the rows
  await page.locator('[data-testid=track-voice] [data-testid=section]').nth(2).click();
  if ((await selectionTitle(page)) !== '0:15–0:25 · 10 s') throw new Error('a click on a section: ' + await selectionTitle(page));
  await dragOnRow('level', 30, 39);
  const m = /^0:(\d\d)–0:(\d\d) · (\d+) s$/.exec(await selectionTitle(page));
  if (!m) throw new Error('drag across a row: ' + await selectionTitle(page));
  near(+m[1], 30, 1, 'drag across a row: start');
  near(+m[2], 39, 1, 'drag across a row: end');
  await page.locator('[data-testid=track-subs] [data-testid=section]').first().click();
  if ((await selectionTitle(page)) !== 'Whole file · 0:48') throw new Error('the whole-file section of another track: ' + await selectionTitle(page));
  return texts;
});

/* ------------------------------------------------------------------------------------------------ 4. hearing */

await check('what is heard follows the timeline: strength is a blend (Gentle: 0.749 noise removal + 0.251 recording), Max has nothing of the recording, a speaker button switches a track off, with every speaker off it is the original, and the waveform says so', async () => {
  const heardBefore = await page.getByTestId('heard-line').locator('path').getAttribute('d');
  await seekTo(20); // inside the Gentle part
  let g = await gains();
  near(g.noise, 1 - 10 ** (-12 / 20), 0.002, 'Gentle: noise removal');
  near(g.source, 10 ** (-12 / 20), 0.002, 'Gentle: recording');
  near(g.levelCover, 1, 0.001, 'Leveling covers the file');
  await seekTo(12); // Strong: 0.01 of the recording
  g = await gains();
  near(g.noise, 0.99, 0.002, 'Strong: noise removal');
  near(g.source, 0.01, 0.002, 'Strong: recording');
  await seekTo(5); // in the Max part: nothing of the recording
  g = await gains();
  near(g.noise, 1, 0.001, 'Max: noise removal');
  near(g.source, 0, 0.001, 'Max: recording');
  // the Voice speaker off: the recording alone (noise removal drops out)
  await seekTo(20);
  await page.getByTestId('hear-voice').click();
  await sleep(150);
  g = await gains();
  if (!(g.source === 1 && g.noise === 0 && g.rebuild === 0)) throw new Error('Voice off: ' + JSON.stringify(g));
  if ((await page.getByTestId('hear-voice').getAttribute('aria-pressed')) !== 'false' || (await page.getByTestId('hearing').innerText()) !== 'Hearing: without Voice') throw new Error('hearing text: ' + await page.getByTestId('hearing').innerText());
  const heardVoiceOff = await page.getByTestId('heard-line').locator('path').getAttribute('d');
  if (heardVoiceOff === heardBefore) throw new Error('the waveform draws the same line without the Voice');
  // every speaker off: the original
  await page.getByTestId('hear-level').click();
  await sleep(150);
  g = await gains();
  if (!(g.source === 1 && g.noise === 0 && g.levelCover === 0)) throw new Error('the original: ' + JSON.stringify(g));
  if ((await page.getByTestId('hearing').innerText()) !== 'Hearing: the original') throw new Error('hearing text: ' + await page.getByTestId('hearing').innerText());
  // the speakers back: everything
  await page.getByTestId('hear-voice').click();
  await page.getByTestId('hear-level').click();
  await sleep(150);
  g = await gains();
  if (!(g.noise > 0.7 && g.levelCover === 1)) throw new Error('everything: ' + JSON.stringify(g));
  if ((await page.getByTestId('hearing').innerText()) !== 'Hearing: everything') throw new Error('hearing text: ' + await page.getByTestId('hearing').innerText());
  if ((await page.getByTestId('heard-line').locator('path').getAttribute('d')) !== heardBefore) throw new Error('the heard line did not come back');
  // changing a part changes what is heard there at once (before it is saved): Gentle -> Max at 20 s
  await drag(15, 25);
  await setStrength(page, 'max');
  await seekTo(20);
  g = await gains();
  near(g.source, 0, 0.001, 'Gentle -> Max: recording');
  await setStrength(page, 'gentle');
  await seekTo(20);
  near((await gains()).source, 10 ** (-12 / 20), 0.002, 'Max -> Gentle: recording');
  return JSON.stringify(g);
});

/* ------------------------------------------------------------------------------------------------ 5. the price, saving */


await check('the price comes from our side a moment after the last change (not on every click): Save changes · $ is the quote, and it follows the next change', async () => {
  const first = await savePrice();
  const n1 = asked('POST', /\/quote$/).length;
  if (n1 < 1 || n1 > 4) throw new Error(`${n1} quotes for four changes (they are asked for a moment after the last one, not for each)`);
  const q1 = lastQuote();
  if (q1.body.timeline?.version !== 1 || q1.body.timeline.layers.noise.length !== 4 || q1.body.timeline.layers.leveling.length !== 1) throw new Error('the quote has the wrong timeline: ' + JSON.stringify(q1.body).slice(0, 300));
  if (q1.answer.free || !(q1.answer.estCostMicros >= 1000)) throw new Error('quote: ' + JSON.stringify(q1.answer).slice(0, 300));
  if (first !== `Save changes · ${money(q1.answer.estCostMicros)}`) throw new Error(`the button says "${first}", the quote ${q1.answer.estCostMicros} micros (${money(q1.answer.estCostMicros)})`);
  if (!q1.answer.lines.some((l) => l.item === 'leveling') || q1.answer.lines.some((l) => l.item === 'denoise')) throw new Error('what runs (only Leveling on the changed parts): ' + JSON.stringify(q1.answer.lines));
  // Details belong to the start screen; here the bar has Your files and the price is the button
  if (await page.getByTestId('bar-det').count() || !(await page.getByTestId('bar-files').count())) throw new Error('bar: Details after the job is done');
  await shot('editor-price');
  // the next change: Leveling off for the Gentle part makes the price smaller, and a new quote is asked for
  const answered = quoteAnswers.length;
  await drag(15, 25);
  if ((await selectionTitle(page)) !== '0:15–0:25 · 10 s') throw new Error('the part was not chosen: ' + await selectionTitle(page));
  await setLeveling(page, 'off');
  if ((await say('level')) !== 'YouTube 0:00–0:15 | Off 0:15–0:25 | YouTube 0:25–0:48') throw new Error('Leveling off for the part: ' + await say('level'));
  const t0 = Date.now();
  const forThis = () => quoteAnswers.slice(answered).filter((q) => q.body.timeline.layers.leveling?.length === 2);
  while (!forThis().length && Date.now() - t0 < 15000) await sleep(200); // (a quote for the half-way state, the part chosen but Leveling not off yet, may come first)
  const second = await savePrice();
  const q2 = forThis().at(-1);
  if (!q2) throw new Error(`no quote for the next change: ${JSON.stringify(quoteAnswers.slice(answered).map((q) => [q.body.timeline.layers.leveling?.length, q.answer.estCostMicros, q.answer.free]))}; the editor says ${await say('level')}; the button says ${await mainButton(page)}`);
  if (q2 !== lastQuote()) throw new Error('a later quote came: ' + JSON.stringify(quoteAnswers.slice(answered).map((q) => [q.body.timeline.layers.leveling?.length, q.answer.estCostMicros])));
  if (!(q2.answer.estCostMicros < q1.answer.estCostMicros)) throw new Error(`${q2.answer.estCostMicros} is not less than ${q1.answer.estCostMicros}: ${JSON.stringify(q2.answer.lines)} ${JSON.stringify(q2.answer.steps)}`);
  if (second !== `Save changes · ${money(q2.answer.estCostMicros)}`) throw new Error(`the button says "${second}", the quote ${q2.answer.estCostMicros}`);
  return `${first} -> ${second} (${q1.answer.estCostMicros} -> ${q2.answer.estCostMicros} micros)`;
});

await check('a paid save is a process on the same job: it asks for exactly the price it was shown (maxCostMicros), the job runs again, the new file comes down by itself, the ledger has one "Edited" line, and the files now hold this timeline', async () => {
  const bal0 = await balance();
  const quoteMicros = lastQuote().answer.estCostMicros;
  const before = got.length;
  const jobsBefore = asked('POST', /^\/api\/jobs$/).length;
  await page.getByTestId('save-changes').click();
  await page.locator('[data-testid=studio][data-stage=processing]').waitFor({ timeout: 15000 });
  if (!/^(Saving|Processing)/.test((await page.getByTestId('processing').innerText()).trim())) throw new Error('the main button while it runs: ' + await page.getByTestId('processing').innerText());
  if (!(await page.getByTestId('job-line').innerText()).trim()) throw new Error('no line about the run');
  await waitCardDone(page);
  await waitResult(page, 'result', 120000);
  await waitDownloads(before + 1);
  if (got.length !== before + 1 || !/\.flac$/.test(got[got.length - 1])) throw new Error('the new file did not come down by itself: ' + got.join(', '));
  const proc = asked('POST', new RegExp(`/jobs/${J1.id}/process$`));
  if (proc.length !== 1 || asked('POST', /\/rerun$/).length || asked('POST', /^\/api\/jobs$/).length !== jobsBefore) throw new Error(`${proc.length} process calls, a new job or a re-run was asked for`);
  const sent = JSON.parse(proc[0].body);
  if (sent.maxCostMicros !== quoteMicros || !sent.timeline?.layers?.noise || sent.timeline.layers.leveling.length !== 2 || (sent.audioFormat ?? 'flac') !== 'flac') throw new Error('process body: ' + proc[0].body.slice(0, 300));
  if (page.url().endsWith(J1.id) === false) throw new Error('the page moved to ' + page.url());
  const bal1 = await balance();
  if (!(bal0 - bal1 > 0 && bal0 - bal1 <= quoteMicros)) throw new Error(`balance ${bal0} -> ${bal1} for a quote of ${quoteMicros}`);
  const j = (await apiCall(page, 'GET', `jobs/${J1.id}`)).body;
  if (!j.edit?.hasTimeline || j.status !== 'done') throw new Error('the job has no timeline of its own now: ' + JSON.stringify(j.edit));
  if ((await studio(page).getAttribute('data-changed')) !== 'no' || (await page.getByTestId('download-again').count()) !== 1) throw new Error('the editor still says it has changes');
  await gotoApp('/app/billing');
  await page.getByTestId('ledger').waitFor();
  const rows = (await page.locator('[data-testid=ledger] tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' ')).filter((t) => /Edited \(/.test(t));
  if (rows.length !== 1 || !/Edited \(Leveling\)/.test(rows[0])) throw new Error('ledger: ' + rows.join(' | '));
  await openJob(J1.id);
  if ((await say('voice')) !== 'Max 0:00–0:10 | Strong 0:10–0:15 | Gentle 0:15–0:25 | Max 0:25–0:48') throw new Error('the saved parts are not in the editor: ' + await say('voice'));
  if ((await say('level')) !== 'YouTube 0:00–0:15 | Off 0:15–0:25 | YouTube 0:25–0:48') throw new Error('the saved Leveling: ' + await say('level'));
  if ((await studio(page).getAttribute('data-changed')) !== 'no' || await page.getByTestId('save-changes').count()) throw new Error('changes after a reload of what was saved');
  return `${quoteMicros} micros, ${rows[0]}`;
});

await check('a job made through the API with no timeline (the old way) opens in the editor with the choices it was made with (Strong, Leveling), plays from the preview stretches, and nothing is priced until something changes; a job made by the website has a timeline and keeps no strength-slider copies', async () => {
  const file = join(FILES, 'noisy.wav');
  const made = await apiCall(page, 'POST', 'jobs', {
    kind: 'denoise', denoiseTier: 'standard', filename: 'api-old.wav', sizeBytes: statSync(file).size, durationSeconds: 10.6,
    options: { noise: 'light', leveling: true, atten_lim_db: 40, output_video: false, audio_format: 'flac' },
  });
  if (made.status !== 200) throw new Error(`create ${made.status} ${JSON.stringify(made.body)}`);
  const id = made.body.id;
  const csrf = (await (await context.request.get(`${BASE}/api/me`)).json()).csrfToken;
  const put = await context.request.put(`${BASE}/api/jobs/${id}/input`, { data: readFileSync(file), headers: { 'X-CSRF-Token': csrf, 'Content-Type': 'application/octet-stream' }, timeout: 120000 });
  if (!put.ok()) throw new Error(`upload ${put.status()} ${await put.text()}`);
  const done = await waitJobDone(page, id, { timeout: 240000 });
  if (done.status !== 'done') throw new Error(`the job ${done.status}: ${done.error}`);
  if (!done.edit?.available || done.edit.hasTimeline || !done.redo?.preview) throw new Error('an old-way job: ' + JSON.stringify({ edit: done.edit, redo: !!done.redo }));
  const mine = (await apiCall(page, 'GET', `jobs/${J1.id}`)).body;
  if (!mine.edit?.hasTimeline || mine.redo?.preview || mine.redo?.available) throw new Error('a job made by the website has strength-slider copies: ' + JSON.stringify({ edit: mine.edit, redo: mine.redo }));
  const before = requests.length;
  await openJob(id);
  if ((await say('voice')) !== 'Strong 0:00–0:11') throw new Error('Voice: ' + await say('voice'));
  if (!/^(YouTube|Podcast|Broadcast|Keep|\-?\d+ LUFS) 0:00–0:11$/.test(await say('level'))) throw new Error('Leveling: ' + await say('level'));
  if (await page.getByTestId('save-changes').count() || (await studio(page).getAttribute('data-changed')) !== 'no') throw new Error('changes on a job nobody changed');
  await seekTo(5);
  near((await gains()).noise, 0.99, 0.002, 'Strong: noise removal');
  await page.getByTestId('play').click();
  await page.waitForFunction(() => document.querySelector('[data-testid=studio]')?.dataset.playing === 'yes', null, { timeout: 20000 });
  await page.waitForFunction(() => Number(document.querySelector('[data-testid=studio]')?.dataset.time) > 6, null, { timeout: 20000 });
  await page.getByTestId('play').click();
  const pv = requests.slice(before).filter((r) => r.m === 'GET' && new RegExp(`/jobs/${id}/preview/`).test(r.p));
  if (!pv.length || pv.some((r) => !/\?start=\d+\.\d{6}&seconds=\d+\.\d{6}$/.test(r.p))) throw new Error('preview requests: ' + pv.map((r) => r.p));
  if (asked('POST', /\/quote$/).filter((r) => r.p.includes(id)).length) throw new Error('a quote for a job nobody changed');
  return `Strong + Leveling, ${pv.length} preview stretches while it played`;
});

/* ------------------------------------------------------------------------------------------------ 6. suggestions (the server's) and the free save */

const J2 = { id: '' };
await check('suggestions made by the server: the chip says how many, the pins sit on the ruler, the card says what was heard and what it suggests; Apply paints the part, undo takes it back, Dismiss removes the pin (and stays dismissed after a reload)', async () => {
  J2.id = await makeJob({ file: 'noisy.wav', leveling: false });
  const made = (await apiCall(page, 'GET', `jobs/${J2.id}/timeline`)).body.cached.heavy?.length ? 'heavy' : 'light';
  const other = made === 'heavy' ? 'light' : 'heavy';
  pyRun(`
import json
from acw import editing, jobs
sug = {"seconds": 10.6, "whole": {"rebuild": False, "remover": "${made}", "strength": "max"},
       "suggestions": [{"start": 2.0, "end": 5.0, "kind": "part", "rebuild": False, "noise": {"strength_db": 40, "remover": "${made}"}, "reasons": ["is noisier"]},
                       {"start": 5.5, "end": 7.0, "kind": "part", "rebuild": False, "noise": {"strength_db": 40, "remover": "${other}"}, "reasons": ["has noise that needs a stronger setting", "needs the other noise remover"]},
                       {"start": 7.5, "end": 9.0, "kind": "clipping", "rebuild": True, "noise": None, "reasons": ["sounds distorted here (clipped)"]}]}
editing.write_suggestions(${JSON.stringify(J2.id)}, sug)
p = jobs.update_params(${JSON.stringify(J2.id)}, edit={**json.loads(__import__("acw.db", fromlist=["one"]).one("SELECT params_json FROM jobs WHERE id = ?", (${JSON.stringify(J2.id)},))["params_json"]).get("edit", {}), "suggestions": True})
`);
  await autoOn();
  await openJob(J2.id);
  const chip = page.getByTestId('sg-chip');
  await chip.waitFor();
  if ((await chip.innerText()).trim() !== '3' || !/3 suggestions/.test(await chip.getAttribute('aria-label'))) throw new Error('chip: ' + await chip.innerText());
  if ((await page.getByTestId('pin').count()) !== 3) throw new Error('pins: ' + await page.getByTestId('pin').count());
  if (await page.getByTestId('sg-card').count()) throw new Error('a suggestion card with no part chosen');
  const labels = await page.getByTestId('pin').evaluateAll((els) => els.map((e) => e.getAttribute('aria-label')));
  if (labels.join('|') !== 'Is noisier, 0:02 to 0:05|Has noise that needs a stronger setting, needs the other noise remover, 0:05 to 0:07|Sounds distorted here (clipped), 0:07 to 0:09') throw new Error('pins: ' + labels.join('|'));
  // the chip goes to the next suggestion and chooses its part; the card is in the inspector
  await chip.click();
  if ((await selectionTitle(page)) !== '0:02–0:05 · 3 s') throw new Error('the chip did not choose the first part: ' + await selectionTitle(page));
  const card = page.getByTestId('sg-card');
  const t0 = (await card.innerText()).replace(/\s+/g, ' ');
  if (!/Is noisier · Strong suggested/.test(t0)) throw new Error('first card: ' + t0);
  await page.getByTestId('pin').nth(2).click();
  const t2 = (await card.innerText()).replace(/\s+/g, ' ');
  if (!/Sounds distorted here \(clipped\) · Voice rebuild suggested/.test(t2) || (await selectionTitle(page)) !== '0:07–0:09 · 2 s') throw new Error('third card: ' + t2);
  await shot('editor-suggestions');
  // Apply paints the part (and the pin is done); the chip counts what is left
  await page.getByTestId('pin').nth(0).click();
  await card.getByTestId('sg-apply').click();
  if (!/^Max 0:00–0:02 \| Strong 0:02–0:05 \| Max 0:05–0:11$/.test(await say('voice'))) throw new Error('the part was not painted: ' + await say('voice'));
  if (await card.count() || (await chip.innerText()).trim() !== '2') throw new Error('after Apply: chip ' + await chip.innerText());
  if ((await studio(page).getAttribute('data-changed')) !== 'tracks') throw new Error('Apply is a change to save');
  // undo takes it back, the suggestion is open again
  await page.getByTestId('undo').click();
  if ((await say('voice')) !== 'Max 0:00–0:11' || (await chip.innerText()).trim() !== '3' || (await studio(page).getAttribute('data-changed')) !== 'no') throw new Error('undo: ' + await say('voice') + ' chip ' + await chip.innerText());
  // Dismiss: gone from the pins and the chip
  await page.getByTestId('pin').nth(2).click();
  await page.getByTestId('sg-card').getByTestId('sg-dismiss').click();
  if ((await page.getByTestId('pin').count()) !== 2 || (await chip.innerText()).trim() !== '2') throw new Error('dismiss: pins ' + await page.getByTestId('pin').count());
  await openJob(J2.id);
  await page.getByTestId('sg-chip').waitFor();
  if ((await page.getByTestId('sg-chip').innerText()).trim() !== '2' || (await page.getByTestId('pin').count()) !== 2) throw new Error('the dismissed one is back after a reload: ' + await page.getByTestId('sg-chip').innerText());
  return 'chip, pins, card, Apply, undo, Dismiss, reload';
});

await check('a change that needs nothing new is free: the button says "Save changes · Free", a save is an export (no process, no charge), the new file comes down by itself', async () => {
  const bal0 = await balance();
  // (a part made with Auto as the remover is decided again when it runs, so a quote counts it as the light remover: the free choice is the remover the job holds)
  const held = (await apiCall(page, 'GET', `jobs/${J2.id}/timeline`)).body.cached;
  const remover = held.heavy?.length ? 2 : 1;
  await page.getByTestId('select-all').click();
  await page.locator(`.studio button[data-act=v-rem][data-v="${remover}"]`).click(); // (the whole file: every part names the remover it was cleaned with)
  await drag(2, 6);
  await setStrength(page, 'balanced');
  const label = await savePrice();
  if (label !== 'Save changes · Free') throw new Error(`label: ${label}; the quote: ${JSON.stringify(lastQuote().answer.lines)} ${JSON.stringify(lastQuote().answer.steps)} ${JSON.stringify(lastQuote().body.timeline.layers.noise)}`);
  const q = lastQuote();
  if (!q.answer.free || q.answer.estCostMicros !== 0) throw new Error('quote: ' + JSON.stringify(q.answer).slice(0, 200));
  await shot('editor-free');
  const before = got.length;
  await page.getByTestId('save-changes').click();
  await waitCardDone(page);
  await waitResult(page, 'result', 120000);
  await waitDownloads(before + 1);
  if (got.length !== before + 1) throw new Error('no new file by itself');
  const ex = asked('POST', new RegExp(`/jobs/${J2.id}/export$`));
  if (ex.length !== 1 || asked('POST', new RegExp(`/jobs/${J2.id}/process$`)).length) throw new Error('export / process calls');
  const body = JSON.parse(ex[0].body);
  if (!body.timeline?.layers?.noise || 'maxCostMicros' in body) throw new Error('export body: ' + ex[0].body.slice(0, 200));
  const bal1 = await balance();
  if (bal1 !== bal0) throw new Error(`a free save cost ${bal0 - bal1} micros`);
  const j = (await apiCall(page, 'GET', `jobs/${J2.id}`)).body;
  if (!j.edit?.hasTimeline) throw new Error('no timeline after the save');
  return 'export, free';
});

await check('errors are plain: a price the person cannot pay says so with a link to add credit and Save changes is off; a timeline the server refuses shows the server\'s words', async () => {
  const id = J1.id;
  await openJob(id);
  // a quote that is too dear for the balance
  await page.route(`**/api/jobs/${id}/quote`, (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
    estCostMicros: 9_000_000, estCostUsd: 9, free: false, unchanged: false, needsProcess: true, approximate: false, lines: [{ item: 'leveling', label: 'Leveling', quantity: 1, seconds: 48, micros: 9_000_000 }], steps: { leveling: { blocks: 1, seconds: 48 } }, asked: {},
    timeline: {}, expiresAt: 1, availableMicros: 1_000_000, affordable: false,
  }) }));
  try {
    await drag(5, 9);
    await setStrength(page, 'gentle');
    const note = page.locator('[data-testid=notes] .banner-warn', { hasText: /This costs about/ });
    await note.waitFor({ timeout: 15000 });
    const t = (await note.innerText()).replace(/\s+/g, ' ');
    if (!/This costs about \$9\.00, but your balance is \$1\.00\./.test(t) || !/Add credit/.test(t)) throw new Error('poor: ' + t);
    if ((await note.getByRole('link', { name: 'Add credit' }).getAttribute('href')) !== '/app/billing') throw new Error('the link');
    if (!(await page.getByTestId('save-changes').isDisabled())) throw new Error('Save changes is on although the price is too high');
    await page.unroute(`**/api/jobs/${id}/quote`);
    await page.route(`**/api/jobs/${id}/quote`, (route) => route.fulfill({ status: 400, contentType: 'application/json', body: JSON.stringify({ detail: 'Voice rebuilding is paused right now. Try again later, or choose Noise removal.' }) }));
    await setStrength(page, 'balanced');
    await page.getByTestId('quote-error').waitFor({ timeout: 15000 });
    if (!/paused right now/.test(await page.getByTestId('quote-error').innerText())) throw new Error('the server\'s words are not shown');
    if (!(await page.getByTestId('save-changes').isDisabled())) throw new Error('Save changes is on without a price');
  } finally {
    await page.unroute(`**/api/jobs/${id}/quote`).catch(() => {});
  }
  return 'insufficient balance and a refused timeline are shown plainly';
});

/* ------------------------------------------------------------------------------------------------ 7. the start screen with parts */

await check('the start screen with parts: the whole file is priced here from the price list (no request), a part chosen makes Start wait for the price of the parts from our side (POST /api/jobs/estimate with the timeline, no tier and no sound options), Details lists what it is made of, and the job is made with that timeline', async () => {
  await gotoApp('/app');
  await dropFile(page, join(FILES, 'noisy.wav'));
  await setKind(page, 'paid');
  await setStrength(page, 'max');
  await setLeveling(page, 'youtube');
  await sleep(1200);
  const est0 = asked('POST', /\/jobs\/estimate$/).length;
  const startWhole = await mainButton(page);
  if (!/^Start · \$0\.0\d+$/.test(startWhole)) throw new Error('Start for the whole file: ' + startWhole);
  if (asked('POST', /\/jobs\/estimate$/).length !== est0) throw new Error('the whole file asked our side for its price');
  await drag(2, 6);
  await setStrength(page, 'gentle');
  await sleep(1000);
  const est = asked('POST', /\/jobs\/estimate$/).slice(est0);
  if (!est.length) throw new Error('no estimate was asked of our side');
  const body = JSON.parse(est.at(-1).body);
  if (body.kind !== 'denoise' || !body.timeline?.layers?.noise?.length || 'denoiseTier' in body || 'transcribeTier' in body || ['atten_lim_db', 'noise', 'leveling', 'loudness_lufs', 'echo_reference_track'].some((k) => k in (body.options || {}))) throw new Error('estimate body: ' + est.at(-1).body.slice(0, 400));
  const startNow = await mainButton(page);
  if (!/^Start · \$0\.0\d+$/.test(startNow)) throw new Error('Start: ' + startNow);
  await page.getByTestId('bar-det').click();
  const det = (await page.getByTestId('details').innerText()).replace(/\s+/g, ' ');
  if (!/Noise removal · 0:11 \$0\.\d+/.test(det) || !/Leveling · 0:11/.test(det) || !/Includes a small amount for each run\. Parts are rounded up to whole seconds\./.test(det)) throw new Error('details: ' + det);
  if ((await page.getByTestId('details-total').innerText()) !== startNow.replace('Start · ', '')) throw new Error(`Details total ${await page.getByTestId('details-total').innerText()} vs ${startNow}`);
  await shot('start-parts');
  await page.getByTestId('bar-det').click();
  const before = got.length;
  await page.getByTestId('start').click();
  await page.waitForURL(JOB_PATH, { timeout: 90000 });
  const id = jobIdOf(page);
  const created = asked('POST', /^\/api\/jobs$/).pop();
  const sent = JSON.parse(created.body);
  if (!sent.timeline || 'denoiseTier' in sent || 'transcribeTier' in sent || ['atten_lim_db', 'noise', 'leveling', 'echo_reference_track'].some((k) => k in sent.options)) throw new Error('create body: ' + created.body.slice(0, 400));
  if (sent.options.audio_format !== 'flac' || sent.kind !== 'denoise') throw new Error('create options: ' + JSON.stringify(sent.options));
  if (!/^Got it, we’re on it\./.test(await page.getByTestId('submitted').innerText())) throw new Error('no welcome note');
  await waitCardDone(page);
  await waitResult(page, 'result', 120000);
  await waitDownloads(before + 1);
  const t = (await apiCall(page, 'GET', `jobs/${id}/timeline`)).body;
  const strengths = t.timeline.layers.noise.map((p) => p.strength_db).join(',');
  if (strengths !== '100,12,100') throw new Error('the parts: ' + JSON.stringify(t.timeline.layers.noise));
  return `${startWhole} -> ${startNow}; made with parts ${strengths}`;
});

await check('Voice Off for the whole file and noise removal on one part is only that part: nothing chosen stops Start with the plain note, the timeline holds only the part, and the price is smaller than the whole file\'s', async () => {
  await gotoApp('/app');
  await dropFile(page, join(FILES, 'speech-long.wav'));
  await setKind(page, 'paid');
  await setStrength(page, 'max');
  await sleep(500);
  const whole = await mainButton(page);
  const wholeMicros = Number(/\$(\d+\.\d+)/.exec(whole)[1]);
  await setVoice(page, 'off');
  await sleep(300);
  if (!(await page.getByTestId('start').isDisabled())) throw new Error('Start is on with nothing to process');
  await page.locator('[data-testid=notes] .banner-warn', { hasText: 'Choose at least one thing to do: noise removal, subtitles, or another file type or video size.' }).waitFor();
  await drag(10, 20);
  await setVoice(page, 'noise'); // (noise removal for the part only)
  await setStrength(page, 'strong');
  await page.waitForFunction(() => /^Start · \$/.test(document.querySelector('[data-testid=start]')?.textContent || ''), null, { timeout: 15000 });
  await sleep(800);
  const only = await mainButton(page);
  const onlyMicros = Number(/\$(\d+\.\d+)/.exec(only)[1]);
  if (!(onlyMicros < wholeMicros)) throw new Error(`only a part ${only} is not less than the whole file ${whole}`);
  if (await page.getByTestId('start').isDisabled()) throw new Error('Start is off with a part chosen');
  if ((await say('voice')) !== 'Off 0:00–0:10 | Strong 0:10–0:20 | Off 0:20–0:48') throw new Error('voice: ' + await say('voice'));
  const body = JSON.parse(asked('POST', /\/jobs\/estimate$/).pop().body);
  const total = body.timeline.layers.noise.reduce((s, p) => s + p.end - p.start, 0);
  if (!(total < 12 && total >= 9) || Object.keys(body.timeline.layers).join() !== 'noise' || body.timeline.layers.noise.length !== 1) throw new Error('the timeline covers ' + total + ' s: ' + JSON.stringify(body.timeline.layers));
  // back to the whole file
  await page.getByTestId('select-all').click();
  await setVoice(page, 'noise');
  await sleep(500);
  if (await page.getByTestId('start').isDisabled()) throw new Error('back to the whole file: Start is off');
  if ((await say('voice')) !== 'Max 0:00–0:10 | Strong 0:10–0:20 | Max 0:20–0:48') throw new Error('the whole file with noise removal (each part keeps its strength): ' + await say('voice'));
  return `whole file ${whole}, only the part ${only}`;
});

/** A file with two kinds of sound: 24 s of speech, then 24 s of the same speech as heard on a phone. */
const PARTS = join(FILES, 'suggest-two-parts.wav');
if (!existsSync(PARTS)) {
  execFileSync(FFMPEG, ['-v', 'error', '-y', '-i', join(FILES, 'speech-long.wav'), '-filter_complex',
    '[0:a]atrim=0:24,asetpts=PTS-STARTPTS[a];[0:a]atrim=24:48,asetpts=PTS-STARTPTS,highpass=f=300,lowpass=f=3300,volume=2[b];[a][b]concat=n=2:v=0:a=1', '-ar', '48000', '-ac', '1', PARTS]);
}
const LONG = join(FILES, 'suggest-over-30-min.flac');
if (!existsSync(LONG)) execFileSync(FFMPEG, ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'anoisesrc=d=1900:c=pink:a=0.01:r=8000', '-ar', '8000', '-ac', '1', LONG]);

await check('suggestions found in the browser (in a worker, with progress) for a file with two kinds of sound: the phone-like half is suggested for Voice rebuild, with the reason, before anything is uploaded; Apply paints it', async () => {
  await gotoApp('/app');
  await autoOn();
  await page.setInputFiles('input[type=file]', PARTS);
  await page.getByTestId('start').waitFor({ timeout: 30000 });
  await page.getByTestId('sg-progress').waitFor({ timeout: 90000 });
  if (!/\d+%/.test(await page.getByTestId('sg-progress').innerText())) throw new Error('no percentage');
  await page.getByTestId('sg-chip').waitFor({ timeout: 150000 });
  await page.getByTestId('sg-progress').waitFor({ state: 'detached', timeout: 20000 });
  const labels = await page.getByTestId('pin').evaluateAll((els) => els.map((e) => e.getAttribute('aria-label')));
  const phone = labels.find((t) => /(phone call|muffled|echoey)/.test(t));
  if (!phone) throw new Error('no suggestion for the narrow, dull half: ' + labels.join(' | '));
  const start = /(\d+):(\d\d) to (\d+):(\d\d)$/.exec(phone);
  near(+start[1] * 60 + +start[2], 24, 3, 'the narrow half starts at 24 s');
  if ((await page.getByTestId('waveform').getAttribute('data-known')) !== 'yes') throw new Error('no waveform');
  const i = labels.indexOf(phone);
  await page.getByTestId('pin').nth(i).click();
  const card = (await page.getByTestId('sg-card').innerText()).replace(/\s+/g, ' ');
  if (!/Voice rebuild suggested/.test(card)) throw new Error('card: ' + card);
  await page.getByTestId('sg-card').getByTestId('sg-apply').click();
  const rebuilt = (await sections('voice')).filter((x) => x.text === 'Rebuild');
  if (!rebuilt.length || rebuilt[0].s < 20) throw new Error('rebuild part not painted: ' + await say('voice'));
  await shot('start-suggestions');
  return labels.join(' | ').slice(0, 300);
});

await check('looking can be canceled, and a file over 30 minutes says "Suggestions appear after the first run." (nothing is measured) while choosing parts yourself still works', async () => {
  await gotoApp('/app');
  await autoOn();
  await page.setInputFiles('input[type=file]', join(FILES, 'long10min.mp3'));
  await page.getByTestId('start').waitFor({ timeout: 30000 });
  await page.getByTestId('sg-progress').waitFor({ timeout: 120000 });
  await page.getByTestId('sg-cancel').click();
  await page.getByTestId('sg-canceled').waitFor({ timeout: 30000 });
  if (await page.getByTestId('sg-progress').count()) throw new Error('still looking after Cancel');
  await gotoApp('/app');
  await autoOn();
  await page.setInputFiles('input[type=file]', LONG);
  await page.getByTestId('start').waitFor({ timeout: 30000 });
  await page.getByTestId('sg-unavailable').waitFor({ timeout: 120000 });
  if ((await page.getByTestId('sg-unavailable').innerText()).trim() !== 'Suggestions appear after the first run.') throw new Error('words: ' + await page.getByTestId('sg-unavailable').innerText());
  if (await page.getByTestId('sg-progress').count()) throw new Error('measuring a long file');
  await page.waitForFunction(() => document.querySelector('[data-testid=waveform]')?.dataset.reading === 'no', null, { timeout: 120000 });
  await drag(600, 700);
  if (!(await page.getByTestId('range-box').count()) || !/^10:00–11:40 · 1 min 40 s$/.test(await selectionTitle(page))) throw new Error('choosing a part by hand: ' + await selectionTitle(page));
  return 'canceled; over 30 minutes: after the first run';
});

/* ------------------------------------------------------------------------------------------------ 8. phone */

for (const scheme of ['light', 'dark']) {
  await check(`phone 390x844, ${scheme}: nothing scrolls sideways (the job page, a part chosen, the sheet open, the start screen), the inspector is a sheet above the bar, a tap on a section chooses it, a long press and Select more add ranges, a finger moves a handle`, async () => {
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true, deviceScaleFactor: 2, colorScheme: scheme });
    const p = await ctx.newPage();
    try {
      await signIn(p, EMAIL);
      const wide = async (what) => {
        const w = await p.evaluate(() => ({ doc: document.documentElement.scrollWidth, body: document.body.scrollWidth, inner: window.innerWidth }));
        if (w.doc > w.inner + 1 || w.body > w.inner + 1) throw new Error(`${what}: the page is ${Math.max(w.doc, w.body)} px wide in ${w.inner}`);
      };
      const cdp = await ctx.newCDPSession(p);
      const touch = (type, x, y) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: type === 'touchEnd' ? [] : [{ x, y }] });
      const centre = async (loc) => { await loc.scrollIntoViewIfNeeded(); const b = await loc.boundingBox(); return { x: b.x + b.width / 2, y: b.y + b.height / 2 }; };
      const tap = async (loc) => { const c = await centre(loc); await touch('touchStart', c.x, c.y); await touch('touchEnd'); await p.waitForTimeout(150); };
      const title = () => p.locator('.studio .js-title').innerText();
      await p.goto(`${BASE}/app/job/${J1.id}`);
      await waitEditor(p);
      await wide('job page');
      const insp = p.getByTestId('inspector');
      if ((await insp.getAttribute('data-sheet')) !== 'min') throw new Error('the inspector starts open: ' + await insp.getAttribute('data-sheet'));
      // a tap on the waveform moves the playhead
      await p.getByTestId('waveform').scrollIntoViewIfNeeded();
      const wf = await p.getByTestId('waveform').boundingBox();
      await touch('touchStart', wf.x + wf.width * 0.5, wf.y + wf.height / 2);
      await touch('touchEnd');
      await p.waitForTimeout(250);
      near(Number(await p.getByTestId('studio').getAttribute('data-time')), 24, 2, 'a tap moves the playhead');
      // a tap on a section chooses exactly it and opens the sheet
      const gentle = p.locator('[data-testid=track-voice] [data-testid=section]').nth(2);
      await tap(gentle);
      if ((await title()) !== '0:15–0:25 · 10 s' || (await insp.getAttribute('data-sheet')) !== 'open') throw new Error(`tap on a section: "${await title()}", sheet ${await insp.getAttribute('data-sheet')}`);
      await wide('with a part chosen');
      const geo = await p.evaluate(() => {
        const h = document.querySelector('[data-testid=inspector]'); const c = document.querySelector('[data-testid=bar]');
        const hs = getComputedStyle(h); const cs = getComputedStyle(c); const hr = h.getBoundingClientRect(); const cr = c.getBoundingClientRect();
        return { hostPos: hs.position, hostBottom: hr.bottom, hostTop: hr.top, barPos: cs.position, barTop: cr.top, barBottom: cr.bottom, h: window.innerHeight };
      });
      if (geo.hostPos !== 'fixed' || !['fixed', 'sticky'].includes(geo.barPos)) throw new Error('not a sheet and a bar: ' + JSON.stringify(geo));
      if (!(geo.hostBottom <= geo.barTop + 1) || !(geo.barBottom >= geo.h - 1) || !(geo.hostTop > geo.h * 0.2)) throw new Error('the sheet is not between the page and the bar: ' + JSON.stringify(geo));
      // the sheet folds with a tap on its title (the sections are behind it while it is open), the part stays
      await tap(p.locator('.studio .ih-main'));
      if ((await insp.getAttribute('data-sheet')) !== 'min' || (await title()) !== '0:15–0:25 · 10 s') throw new Error('the sheet did not fold: ' + await insp.getAttribute('data-sheet'));
      // a long press adds a range (Select more switches on); with Select more on a tap adds a range and a tap on a chosen range takes it out
      const sec = (i) => p.locator('[data-testid=track-voice] [data-testid=section]').nth(i);
      const c1 = await centre(sec(0));
      await touch('touchStart', c1.x, c1.y);
      await p.waitForTimeout(700);
      await touch('touchEnd');
      await p.waitForTimeout(250);
      if ((await p.getByTestId('select-more').getAttribute('aria-pressed')) !== 'true' || (await title()) !== '2 ranges · 0:20 total') throw new Error(`long press: "${await title()}", Select more ${await p.getByTestId('select-more').getAttribute('aria-pressed')}`);
      await tap(sec(1));
      if ((await title()) !== '0:00–0:25 · 25 s') throw new Error('Select more, a tap adds a range (it joins its neighbours): ' + await title());
      await tap(sec(1));
      if ((await title()) !== '2 ranges · 0:20 total') throw new Error('Select more, a tap on a chosen range takes it out: ' + await title());
      await tap(p.getByTestId('select-more'));
      if ((await p.getByTestId('select-more').getAttribute('aria-pressed')) === 'true') throw new Error('Select more stays on');
      await tap(p.getByTestId('select-all'));
      if ((await title()) !== 'Whole file · 0:48') throw new Error('Select all: ' + await title());
      await wide('after Select all');
      // a change in the sheet, with a finger: the price follows
      await tap(sec(2));
      if ((await title()) !== '0:15–0:25 · 10 s') throw new Error('a tap on a section after Select more: ' + await title());
      await tap(p.locator('.studio button[data-act=v-str][data-v="3"]'));
      if ((await say('voice', p)) !== 'Max 0:00–0:10 | Strong 0:10–0:25 | Max 0:25–0:48') throw new Error('voice after a tap on Strong (it joins its neighbour): ' + await say('voice', p));
      await p.waitForFunction(() => /Save changes · /.test(document.querySelector('[data-testid=save-changes]')?.textContent || ''), null, { timeout: 15000 });
      await wide('with the sheet open');
      await shot(`phone-${scheme}-sheet`, p);
      // the sheet folds with a tap on its title, the part stays
      await tap(p.locator('.studio .ih-main'));
      if ((await insp.getAttribute('data-sheet')) !== 'min' || (await title()) !== '0:15–0:25 · 10 s') throw new Error('the sheet did not fold: ' + await insp.getAttribute('data-sheet'));
      await wide('with the sheet folded');
      // a finger moves a handle of the part
      await tap(sec(1));
      if ((await title()) !== '0:10–0:25 · 15 s') throw new Error('a tap on the joined section: ' + await title());
      // the sheet is open and covers the lower half: scroll the page until the handle is above it
      await p.evaluate(() => window.scrollBy(0, Math.max(0, document.querySelector('[data-testid=handle-end]').getBoundingClientRect().top - 120)));
      await p.waitForTimeout(150);
      const hb = await p.getByTestId('handle-end').boundingBox();
      const T = Number(await p.getByTestId('wave').getAttribute('data-seconds'));
      const wb = await p.getByTestId('waveform').boundingBox();
      // the sheet is open again after the tap: aim at the part of the handle that is above it
      const sheetTop = (await insp.boundingBox()).y;
      const hx = hb.x + hb.width / 2; const hy = hb.y + Math.min(hb.height, Math.max(8, sheetTop - 4 - hb.y)) / 2;
      const hit = await p.evaluate(([x, y]) => { const el = document.elementFromPoint(x, y); return el ? (el.closest('[data-testid]')?.getAttribute('data-testid') || el.className) : 'nothing'; }, [hx, hy]);
      if (hit !== 'handle-end') throw new Error(`the end handle is not under the finger at ${Math.round(hx)},${Math.round(hy)}: ${hit} (handle ${JSON.stringify(hb)}, sheet top ${sheetTop})`);
      await touch('touchStart', hx, hy);
      for (let i = 1; i <= 8; i++) await touch('touchMove', hx + ((wb.x + (32 / T) * wb.width - hx) * i) / 8, hy);
      await touch('touchEnd');
      await p.waitForTimeout(250);
      const e = await p.getByTestId('time-e').innerText();
      near(Number(e.split(':')[1]), 32, 1.5, 'a finger moves the end handle');
      await shot(`phone-${scheme}-handle`, p);
      // the start screen
      await p.goto(`${BASE}/app`);
      await p.getByTestId('available').waitFor();
      await dropFile(p, join(FILES, 'noisy.wav'), { recommended: true }); // the choices stay as recommended: the sheet is folded here, so the old-defaults helper would find no buttons
      await p.waitForTimeout(500);
      await wide('start screen');
      await tap(p.getByTestId('split'));
      await p.getByTestId('waveform').scrollIntoViewIfNeeded();
      const wf2 = await p.getByTestId('waveform').boundingBox();
      await touch('touchStart', wf2.x + wf2.width * 0.5, wf2.y + wf2.height / 2);
      await touch('touchEnd');
      await p.waitForTimeout(250);
      await tap(p.getByTestId('split'));
      if (!/^0:0[5-6]–0:11 · \d s$/.test(await title()) || (await p.getByTestId('inspector').getAttribute('data-sheet')) !== 'open') throw new Error('split on the phone: ' + await title());
      await wide('start screen with a part');
      await shot(`phone-${scheme}-start`, p);
      return 'no sideways scroll, sheet and bar placed, taps, long press, Select more, handle by finger';
    } finally {
      await ctx.close();
    }
  });
}

/* ------------------------------------------------------------------------------------------------ 9. words and prices */

const scan = async (what, p = page) => {
  const text = await p.evaluate(() => {
    const parts = [document.body.innerText];
    for (const d of document.querySelectorAll('details, dialog')) parts.push(d.textContent);
    for (const e of document.querySelectorAll('[aria-label],[title],[data-tip]')) parts.push(e.getAttribute('aria-label') || '', e.getAttribute('title') || '', e.getAttribute('data-tip') || '');
    return parts.join('\n');
  });
  const bad = BANNED.map((re) => re.exec(text)).filter(Boolean).map((m) => `${what}: "${m[0]}" in …${text.slice(Math.max(0, m.index - 40), m.index + 50).replace(/\s+/g, ' ')}…`);
  if (bad.length) throw new Error(bad.join(' | '));
  return text.length;
};

await check('words: the editor, its popovers (Output, Your files, Details), its tooltips and the "All the options" window use plain words (no model names, no "premium", never "polish", no "our servers")', async () => {
  let chars = 0;
  await openJob(J2.id);
  await drag(2, 6);
  await setStrength(page, 'strong');
  await sleep(1200);
  chars += await scan('the editor of a finished job');
  await page.getByTestId('output-button').click();
  await page.getByTestId('output-pop').waitFor();
  chars += await scan('the Output menu');
  await page.getByTestId('output-button').click();
  await page.getByTestId('bar-files').click();
  await page.getByTestId('files').waitFor();
  chars += await scan('Your files');
  await page.getByTestId('bar-files').click();
  await page.getByTestId('info-open').click();
  await page.getByTestId('info-modal').waitFor();
  chars += await scan('All the options');
  await page.getByTestId('info-close').click();
  await gotoApp('/app');
  await dropFile(page, join(FILES, 'noisy.wav'));
  await setKind(page, 'paid');
  await drag(2, 6);
  await setStrength(page, 'gentle');
  await setSubs(page, 'standard');
  await sleep(1000);
  chars += await scan('the start screen with parts');
  await page.getByTestId('bar-det').click();
  await page.getByTestId('details').waitFor();
  chars += await scan('Details');
  return `${chars} characters scanned in six places`;
});

await check('"Which one should I pick?" is in the "All the options" window (the (i) next to the part\'s title): Noise removal and Voice rebuild side by side, the Free / Standard and subtitle comparisons and the prices of the live price list; the Voice rows carry their one-line hint', async () => {
  await openJob(J2.id);
  const closed = await page.getByTestId('info-modal').evaluate((d) => d.open);
  if (closed) throw new Error('the window is open before the (i) was pressed');
  await page.getByTestId('info-open').click();
  await page.getByTestId('info-modal').waitFor();
  const t = (await page.getByTestId('info-modal').innerText()).replace(/\s+/g, ' ');
  for (const need of [/All the options/, /Voice: Noise removal or Voice rebuild\?/, /Noise removal/, /Voice rebuild/, /exactly as recorded and never changes words/, /Laughs, shouts, whispers and singing stay as they were/, /game sounds/, /from scratch/, /slightly too clean/, /come out oddly/, /can change a word, so listen before you publish/, /90 minutes/, /Costs more/, /many languages/,
    /Noise removal: Free or Standard/, /Subtitles: Free, Standard or Extra accurate/, /Echo removal/, /Leveling/, /Prices/]) {
    if (!need.test(t)) throw new Error(`the comparison lacks ${need}`);
  }
  const pricing = (await (await fetch(`${BASE}/api/billing/pricing`)).json()).rates;
  const usd = (r) => `$${r.usdPerMinute.toFixed(4).replace(/(\.\d\d\d*?)0+$/, '$1')} a minute`;
  for (const [row, product, tier] of [['Noise removal, Standard', 'denoise', 'standard'], ['Voice rebuild', 'denoise', 'rebuild'], ['Echo removal', 'echo', 'standard'], ['Leveling', 'leveling', 'standard'], ['Subtitles, Standard', 'transcribe', 'local'], ['Subtitles, Extra accurate', 'transcribe', 'premium']]) {
    const rate = pricing.find((r) => r.product === product && r.tier === tier);
    const cells = (await page.locator('table.prices tr', { has: page.locator('th', { hasText: new RegExp(`^${row}$`) }) }).first().innerText()).replace(/\s+/g, ' ');
    if (!rate || !cells.includes(usd(rate))) throw new Error(`price row "${row}": "${cells}" vs ${rate && usd(rate)}`);
  }
  await page.keyboard.press('Escape');
  await page.getByTestId('info-modal').waitFor({ state: 'hidden' });
  // the one-line hints under the Voice rows
  const rebuild = (await page.getByTestId('voice-rebuild').innerText()).replace(/\s+/g, ' ');
  if (!/fixes echo, muffled or distorted voices/.test(rebuild) && !/paused right now/.test(rebuild)) throw new Error('the hint of Voice rebuild is gone: ' + rebuild);
  if (!/keeps your voice as recorded/.test(await page.getByTestId('voice-noise').innerText())) throw new Error('the hint of noise removal is gone');
  return `${t.length} characters of comparison`;
});

await check('the automatic settings say which one they picked and why, in the terms of the comparison ("Voice rebuild: the voice sounds echoey" / "Denoising: the voice is already clear ...")', async () => {
  await gotoApp('/app');
  await autoOn();
  await page.setInputFiles('input[type=file]', join(FILES, 'speech-long.wav'));
  await page.getByTestId('start').waitFor({ timeout: 20000 });
  await page.getByTestId('reco-status').waitFor({ timeout: 60000 });
  await page.getByTestId('reco-details').locator('summary').click();
  const pickText = (await page.getByTestId('reco-pick').innerText()).trim();
  if (!/^(Voice rebuild: .+\.|Denoising: the voice is already clear, so it is kept exactly as recorded and only the background sound is taken out\.)$/.test(pickText)) throw new Error('pick line: ' + pickText);
  const why = await page.getByTestId('reco-why').innerText();
  if (!why.includes(pickText)) throw new Error('the reasons do not start with the pick line');
  if (!(await page.getByTestId('reco-line').innerText()).trim()) throw new Error('no summary line');
  return pickText;
});

await check('echo removal has its own price: the price table row and its line, the editor says it next to the option and adds it to the price, and an estimate lists it as a line of its own (Echo removal)', async () => {
  const pricing = (await (await fetch(`${BASE}/api/billing/pricing`)).json()).rates.find((r) => r.product === 'echo');
  if (!pricing || pricing.usdPerMinute !== 0.0026 || pricing.label !== 'Echo removal') throw new Error('price list: ' + JSON.stringify(pricing));
  await page.goto(`${BASE}/#pricing`);
  await page.waitForSelector('table tbody tr');
  const row = (await page.locator('table.price-table tr', { hasText: 'Echo removal' }).first().innerText()).replace(/\s+/g, ' ');
  if (!/Echo removal.*Speaker echo taken out.*\$0\.0026 \/ min/.test(row)) throw new Error('price table row: ' + row);
  await gotoApp('/app');
  await dropFile(page, join(FILES, 'echo2.mp4'));
  await setKind(page, 'paid');
  const sw = page.getByTestId('echo-switch');
  const hint = await sw.getAttribute('data-tip');
  if (!/\$0\.0026 a minute/.test(hint)) throw new Error('hint: ' + hint);
  if ((await sw.getAttribute('aria-checked')) === 'true') await sw.click();
  await page.getByTestId('bar-det').click();
  const off = (await page.getByTestId('details').innerText()).replace(/\s+/g, ' ');
  await page.getByTestId('bar-det').click();
  if (/Echo removal/.test(off)) throw new Error('echo removal is on the bill without being on: ' + off);
  await sw.click();
  if ((await sw.getAttribute('aria-checked')) !== 'true') throw new Error('the switch did not turn on');
  await page.getByTestId('bar-det').click();
  const on = (await page.getByTestId('details').innerText()).replace(/\s+/g, ' ');
  await page.getByTestId('bar-det').click();
  if (!/Echo removal · 0:\d\d (<\$0\.001|\$0\.\d+)/.test(on)) throw new Error('Details with echo removal: ' + on);
  // our side: an estimate lists it as a line of its own, with a timeline and the old way
  const e = (await apiCall(page, 'POST', 'jobs/estimate', { kind: 'denoise', durationSeconds: 600, options: { audio_format: 'flac' }, timeline: { version: 1, seconds: 600, layers: { echo: [{ start: 0, end: 600 }] } } })).body;
  const line = e.lines.find((l) => l.item === 'echo');
  // (a timeline is priced like a run: the rate for the seconds, plus the fixed amount of starting the step, docs/api.md section 8)
  if (!line || line.label !== 'Echo removal' || line.micros !== Math.ceil(((0.0026 * 1e6) / 60) * (600 + line.fixedSeconds)) || e.estCostMicros !== e.lines.reduce((s, l) => s + l.micros, 0)) throw new Error('estimate lines: ' + JSON.stringify(e.lines));
  const old = (await apiCall(page, 'POST', 'jobs/estimate', { kind: 'denoise', denoiseTier: 'standard', durationSeconds: 600, options: { echo_reference_track: true } })).body;
  const oldLine = old.lines.find((l) => l.item === 'echo');
  if (!oldLine || oldLine.micros !== 26000) throw new Error('estimate lines (the old way): ' + JSON.stringify(old.lines));
  await page.goto(`${BASE}/developers`);
  await page.waitForSelector('pre.code');
  const dev = (await page.locator('main').innerText()).replace(/\s+/g, ' ');
  if (!/Echo removal.*\$0\.0026/.test(dev) || !/own price, per second, in its own line/.test(dev)) throw new Error('Developers page does not give the echo price');
  return `${line.micros} micros for ten minutes (${oldLine.micros} the old way); Details: ${on.slice(0, 80)}`;
});

await check('the editor of a job made with echo removal has the Echo removal track (a row, a speaker button, the switch), the job holds an echo part, and it was charged', async () => {
  const id = await makeJob({ file: 'echo2.mp4', leveling: true, echo: true });
  const t = (await apiCall(page, 'GET', `jobs/${id}/timeline`)).body;
  if (!t.secondTrack || !t.timeline.layers.echo?.length) throw new Error('no echo part on the job: ' + JSON.stringify(t.timeline).slice(0, 300));
  const names = (await page.locator('[data-testid^=track-]').evaluateAll((els) => els.map((e) => e.dataset.testid))).join('|');
  if (names !== 'track-voice|track-echo|track-level|track-subs') throw new Error('tracks: ' + names);
  if ((await say('echo')) !== `Echo 0:00–${clock(await waveSeconds())}`) throw new Error('echo track: ' + await say('echo'));
  if (!(await page.getByTestId('hear-echo').count())) throw new Error('no speaker button for Echo removal');
  if ((await page.getByTestId('echo-switch').getAttribute('aria-checked')) !== 'true') throw new Error('the Echo switch for the whole file: ' + await page.getByTestId('echo-switch').getAttribute('aria-checked'));
  await page.getByTestId('hear-echo').click();
  if ((await page.getByTestId('hearing').innerText()) !== 'Hearing: without Echo removal') throw new Error('hearing: ' + await page.getByTestId('hearing').innerText());
  await page.getByTestId('hear-echo').click();
  const bill = (await apiCall(page, 'GET', `jobs/${id}`)).body;
  if (!(bill.costMicros > 0)) throw new Error('cost ' + bill.costMicros);
  await gotoApp('/app/billing');
  await page.getByTestId('ledger').waitFor();
  const rows = (await page.locator('[data-testid=ledger] tbody tr').allInnerTexts()).map((x) => x.replace(/\s+/g, ' ')).filter((x) => /echo2/.test(x));
  if (rows.length !== 1 || !rows[0].includes(`−$${(bill.costMicros / 1e6).toFixed(4)}`)) throw new Error(`ledger (the job cost ${bill.costMicros} micros): ${rows.join(' | ')}`);
  // ... and what it cost includes Echo removal: the estimate of the same timeline lists it as a line of its own, and the job cost more than the rest of the lines
  const est = (await apiCall(page, 'POST', 'jobs/estimate', { kind: 'denoise', durationSeconds: bill.durationSeconds, options: { audio_format: 'flac' }, timeline: t.timeline })).body;
  const echoLine = est.lines.find((l) => l.item === 'echo');
  if (!echoLine || !(bill.costMicros > est.estCostMicros - echoLine.micros)) throw new Error(`the job cost ${bill.costMicros}, the estimate of its timeline ${est.estCostMicros}: ${JSON.stringify(est.lines.map((l) => [l.item, l.micros]))}`);
  return `${names}; the job cost ${bill.costMicros} micros: ${rows[0]}`;
});

context.off('request', onRequest);
await browser.close();
const bad = logs.filter((l) => !/Failed to load resource|favicon|DevTools|net::ERR/.test(l));
if (bad.length) console.log('console:', bad.slice(0, 8).join(' | ').slice(0, 800));
const failed = summary().filter((r) => !r.ok);
console.log(`${summary().length - failed.length} passed, ${failed.length} failed`);
process.exit(failed.length ? 1 : 0);
