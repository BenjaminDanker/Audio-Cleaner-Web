// "Fix a moment": the per-part editor on a finished job's page and in the upload form.
//  1. the row under the result (collapsed, the simple flow unchanged), opening and closing, the badge with the number of suggestions;
//  2. choosing a part: drag, handles, steppers, "Use playhead", "Around playhead", whole file, the three steps;
//  3. painting parts (a new one replaces what it covers), the layers as chips, "Show all layers" rows, the Hear switches and the gains they
//     give (strength as a blend, a switched-off layer, Original / Edited);
//  4. the price: asked of the server a moment after each change, "No changes yet" / Free / Process changes, Details; a free save is an
//     export, a paid one is a process; the new files download by themselves; the ledger has one line; an old job still works the old way;
//  5. suggestions made by the server: cards, Apply, Show me, Dismiss, Undo, the pins;
//  6. the upload form: "Choose parts (optional)", Only parts, the price on Start from the server, the job it makes; suggestions found in
//     the browser (a worker) on a file with two kinds of sound, cancel, "Suggestions appear after the first run" for a long file;
//  7. phone (390x844, touch, light and dark): no sideways scroll, the part settings in a sheet, touch selection;
//  8. the words: no avoided words anywhere in the editor, the "Which one should I pick?" comparison, the echo price.
// Needs the test backend (tests/run.sh editor) and the fixtures. EDITOR_SHOTS=<dir> also saves screenshots. Usage: node tests/editor.mjs
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { BASE, FFMPEG, FILES, JOB_PATH, admin, apiCall, check, dropFile, jobIdOf, launch, listJobs, pick, pyRun, signIn, waitCardDone } from './helpers.mjs';

const EMAIL = `editor-${Date.now()}@example.com`;
const SHOTS = process.env.EDITOR_SHOTS || '';
if (SHOTS) mkdirSync(SHOTS, { recursive: true });
const BANNED = [/DeepFilterNet/i, /\bDFN/i, /\bWhisper\b/, /Parakeet/i, /Sidon/i, /OpenAI/i, /WebAssembly/i, /\bwasm\b/i, /\bour servers?\b/i, /premium/i, /\bpolish(ed|es|ing)?\b|\bPolish (your|the|it|up|this|that|my)\b/, /\bmodel\b/i, /\btiers?\b/i, /undefined|NaN/];

const { browser, context, page, logs } = await launch();
const got = [];
page.on('download', (d) => got.push(d.suggestedFilename()));
const requests = [];
page.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.startsWith('/api/')) requests.push({ m: r.method(), p: u.pathname + u.search, body: r.postData() }); });
await signIn(page, EMAIL);
admin('grant', EMAIL, '5', 'editor test');
const sleep = (ms) => page.waitForTimeout(ms);
const near = (a, b, tol, what) => { if (!(Math.abs(a - b) <= tol)) throw new Error(`${what}: ${a} vs ${b}`); };
const asked = (m, re) => requests.filter((r) => r.m === m && re.test(r.p));
const shot = async (name, p = page) => { if (SHOTS) await p.screenshot({ path: join(SHOTS, `${name}.png`) }); };
const gotoApp = async (path = '/app') => { await page.goto(`${BASE}${path}`); await page.getByTestId('available').or(page.getByTestId('job-page')).first().waitFor({ timeout: 20000 }); };
const waitDownloads = async (n, timeout = 60000) => { const t0 = Date.now(); while (got.length < n && Date.now() - t0 < timeout) await sleep(250); await sleep(600); };
const balance = async () => (await apiCall(page, 'GET', 'me')).body.balance?.available_micros ?? (await apiCall(page, 'GET', 'billing/balance')).body?.available_micros;

/** Makes a job from the form: Denoising at Maximum, Leveling on or off. Answers its id once its card is done and the file has come down. */
async function makeJob({ file = 'speech-long.wav', leveling = true, echo = false } = {}) {
  await gotoApp('/app');
  await dropFile(page, join(FILES, file));
  await pick(page, 'noise', 'denoise');
  await page.locator('label:has(input[name=strength][value=max])').click();
  await page.locator('input[name=leveling]').setChecked(leveling);
  if (echo) await page.locator('input[name=echo]').check();
  await page.getByTestId('estimate').waitFor();
  const before = got.length;
  await page.getByTestId('start').click();
  await page.waitForURL(JOB_PATH, { timeout: 90000 });
  const id = jobIdOf(page);
  await waitCardDone(page);
  await page.getByTestId('result').waitFor({ timeout: 90000 });
  await waitDownloads(before + 1);
  return id;
}
const openEditor = async () => {
  await page.getByTestId('fix-entry').click();
  await page.getByTestId('editor').waitFor();
  await page.getByTestId('editor').scrollIntoViewIfNeeded();
};
const scrubBox = async () => { await page.getByTestId('scrub').scrollIntoViewIfNeeded(); await sleep(100); return page.getByTestId('scrub').boundingBox(); };
const rangeNow = async () => (await page.getByTestId('range-text').innerText()).trim();
const chips = (layer) => page.locator(`[data-testid=layer-chip][data-layer=${layer}]`).evaluateAll((els) => els.map((e) => ({ text: e.textContent.replace(/\s*\(needs processing\)$/, '').trim(), range: e.dataset.range, pending: e.dataset.pending })));
const gains = async () => JSON.parse(await page.getByTestId('editor').getAttribute('data-gains'));
/** Chooses [a, b] seconds on the scrub bar by dragging with the mouse (the recording is `total` seconds long). */
async function dragSelect(a, b, total) {
  const box = await scrubBox();
  const x = (t) => box.x + (t / total) * box.width;
  await page.mouse.move(x(a), box.y + 30);
  await page.mouse.down();
  await page.mouse.move(x((a + b) / 2), box.y + 30, { steps: 5 });
  await page.mouse.move(x(b), box.y + 30, { steps: 5 });
  await page.mouse.up();
}
const J1 = { id: '', total: 48.4 };
const hearChip = (l) => page.locator(`label:has([data-testid=hear-${l}])`);

/* ------------------------------------------------------------------------------------------------ 1. the row, opening and closing */

await check('the job page: "Fix a moment" sits under the result and is collapsed; the simple flow is unchanged (the file came down by itself, Download again, Listen, Change and save again are all there)', async () => {
  J1.id = await makeJob({ file: 'speech-long.wav', leveling: true });
  if (got.length !== 1 || !/\.flac$/.test(got[0])) throw new Error('downloaded: ' + got.join(', '));
  const entry = page.getByTestId('fix-entry');
  await entry.waitFor();
  if ((await entry.getAttribute('aria-expanded')) !== 'false') throw new Error('not collapsed');
  if (await page.getByTestId('fix-panel').count()) throw new Error('the editor is built before it is opened');
  for (const id of ['download-again', 'listen', 'again']) if (!(await page.getByTestId(id).count())) throw new Error(`${id} is gone`);
  const order = await page.evaluate(() => { const a = document.querySelector('[data-testid=again]'); const b = document.querySelector('[data-testid=fix-entry]'); return !!(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING); });
  if (!order) throw new Error('the row is not below the other choices');
  if (await page.getByTestId('fix-badge').count()) throw new Error('a badge without any suggestions');
  await shot('job-collapsed');
  return 'collapsed under the result';
});

await check('opening shows the editor (the player, the scrub bar with a waveform, the Hear chips for the layers, the three steps, the layer chips) and Close hides it again; the editor keeps its choices while it is closed', async () => {
  await openEditor();
  const panel = page.getByTestId('fix-panel');
  if (!(await panel.isVisible())) throw new Error('not visible');
  for (const id of ['play', 'scrub', 'hear-chips', 'step-home', 'step-range', 'step-change', 'layer-chips', 'pick-part', 'all-layers']) if (!(await page.getByTestId(id).count())) throw new Error(`${id} missing`);
  const names = await page.locator('[data-testid=hear-chips] label').allInnerTexts();
  if (names.join('|') !== 'Noise removal|Voice rebuild|Leveling') throw new Error('hear chips: ' + names.join('|')); // (no second track: no Echo removal)
  await page.waitForFunction(() => document.querySelector('[data-testid=waveform]')?.dataset.known === 'yes', null, { timeout: 20000 });
  const text = await page.getByTestId('cost-title').innerText();
  if (text !== 'No changes yet') throw new Error('cost bar: ' + text);
  if (asked('POST', /\/quote$/).length) throw new Error('a quote was asked for with nothing changed');
  await shot('editor-open');
  await dragSelect(10, 20, J1.total);
  await page.getByTestId('fix-close').click();
  await panel.waitFor({ state: 'hidden' });
  if ((await page.getByTestId('fix-entry').getAttribute('aria-expanded')) !== 'false') throw new Error('aria-expanded after close');
  await page.getByTestId('fix-entry').click();
  await panel.waitFor({ state: 'visible' });
  if (!/0:10–0:20/.test(await rangeNow())) throw new Error('the chosen part was lost: ' + (await rangeNow()));
  await page.getByTestId('pp-done').or(page.getByTestId('step-home')).first().click().catch(() => {});
  await page.getByTestId('step-home').click();
  return 'open, closed, open again with the part kept';
});

/* ------------------------------------------------------------------------------------------------ 2. choosing a part */

await check('a part is chosen by dragging with the mouse, by its handles, by the steppers, with Use playhead, Around playhead and Whole file', async () => {
  await page.getByTestId('step-home').click();
  await dragSelect(20, 30, J1.total);
  let r = await rangeNow();
  const m = /^(\d+):(\d\d)–(\d+):(\d\d)$/.exec(r);
  if (!m) throw new Error('range text ' + r);
  near(+m[2], 20, 1, 'drag start');
  near(+m[4], 30, 1, 'drag end');
  if ((await page.getByTestId('host').getAttribute('data-view')) !== 'range') throw new Error('the Part step did not open');
  // the handles
  const box = await scrubBox();
  const end = await page.getByTestId('handle-end').boundingBox();
  await page.mouse.move(end.x + end.width / 2, end.y + end.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + (36 / J1.total) * box.width, end.y + end.height / 2, { steps: 6 });
  await page.mouse.up();
  r = await rangeNow();
  near(+/–\d+:(\d\d)$/.exec(r)[1], 36, 1, 'handle end');
  const start = await page.getByTestId('handle-start').boundingBox();
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + (14 / J1.total) * box.width, start.y + start.height / 2, { steps: 6 });
  await page.mouse.up();
  near(+/^\d+:(\d\d)–/.exec(await rangeNow())[1], 14, 1, 'handle start');
  // the steppers (a second at a time) and the label of the handle
  const t0 = await page.getByTestId('time-s').innerText();
  await page.getByTestId('nudge-s-plus').click();
  await page.getByTestId('nudge-s-plus').click();
  await page.getByTestId('nudge-e-minus').click();
  const t1 = await page.getByTestId('time-s').innerText();
  near(+t1.split(':')[1] - +t0.split(':')[1], 2, 0.01, 'start moved two seconds');
  // Use playhead: click the bar (no drag) to move it, then take it as the start
  await page.mouse.click(box.x + (8 / J1.total) * box.width, box.y + 30); // (away from the handles)
  const ph = +(await page.getByTestId('editor').getAttribute('data-time'));
  near(ph, 8, 0.7, 'playhead after a click');
  await page.getByTestId('nudge-e-plus').click();
  await page.getByTestId('use-playhead-s').click();
  near(+/^\d+:(\d\d)–/.exec(await rangeNow())[1], 8, 1, 'start at the playhead');
  await page.getByTestId('use-playhead-e').click(); // (the playhead is not after the start: the part stays a second long at least)
  const [a, b] = (await rangeNow()).split('–').map((x) => +x.split(':')[1]);
  if (!(b - a >= 1)) throw new Error('shorter than a second: ' + (await rangeNow()));
  await page.getByTestId('around').click();
  const [a2, b2] = (await rangeNow()).split('–').map((x) => +x.split(':')[1]);
  near(b2 - a2, 10, 1, 'around the playhead is ten seconds');
  await page.getByTestId('whole-file').click();
  if (!/^0:00–0:48$/.test(await rangeNow())) throw new Error('whole file: ' + (await rangeNow()));
  return 'all seven ways give the right part';
});

/* ------------------------------------------------------------------------------------------------ 3. painting, chips, rows, hearing */

await check('painting parts: a new part replaces what it covers; each layer lists its parts as chips ("Maximum · 0:00–0:10", "whole file" for the base)', async () => {
  await page.getByTestId('step-home').click();
  const base = await chips('noise');
  if (base.length !== 1 || !/whole file$/.test(base[0].text) || base[0].range.split('-')[0] !== '0') throw new Error('base: ' + JSON.stringify(base));
  await dragSelect(10, 20, J1.total);
  await page.getByTestId('next-change').click();
  await page.getByTestId('noise-strong').click();
  await page.getByTestId('step-range').click();
  await dragSelect(15, 25, J1.total);
  await page.getByTestId('next-change').click();
  await page.getByTestId('noise-gentle').click();
  await page.getByTestId('step-home').click();
  const c = await chips('noise');
  const texts = c.map((x) => x.text).join(' | ');
  if (!/Maximum · 0:00–0:\d\d \| Strong · 0:\d\d–0:\d\d \| Gentle · 0:\d\d–0:2\d \| Maximum · 0:2\d–0:48/.test(texts)) throw new Error('chips: ' + texts);
  if (c.length !== 4) throw new Error('parts: ' + texts);
  const [, strong, gentle] = c.map((x) => x.range.split('-').map(Number));
  near(strong[1], gentle[0], 0.001, 'Strong ends where Gentle starts (the new part replaced the overlap)');
  await shot('editor-parts');
  return texts;
});

await check('the layers as rows ("Show all layers"): one row for each layer with a clip per part, dragging across a row chooses a part, clicking a clip chooses exactly it, zoom works, and back to the simple view', async () => {
  await page.getByTestId('all-layers').click();
  await page.getByTestId('rows').waitFor();
  for (const l of ['noise', 'rebuild', 'leveling']) if (!(await page.getByTestId(`row-${l}`).count())) throw new Error(`no row for ${l}`);
  if ((await page.locator('[data-testid=clip][data-layer=noise]').count()) !== 4) throw new Error('clips of noise removal');
  const clip = page.locator('[data-testid=clip][data-layer=noise]').nth(2);
  const want = (await clip.getAttribute('data-range')).split('-').map(Number);
  await clip.click();
  await page.getByTestId('step-range').click();
  const [a, b] = (await rangeNow()).split('–').map((x) => { const [mm, ss] = x.split(':').map(Number); return mm * 60 + ss; });
  near(a, want[0], 1, 'clip start');
  near(b, want[1], 1, 'clip end');
  await page.getByTestId('row-rebuild').scrollIntoViewIfNeeded();
  const row = await page.getByTestId('row-rebuild').boundingBox();
  await page.mouse.move(row.x + row.width * 0.62, row.y + 20);
  await page.mouse.down();
  await page.mouse.move(row.x + row.width * 0.8, row.y + 20, { steps: 6 });
  await page.mouse.up();
  const r = (await rangeNow()).split('–').map((x) => { const [mm, ss] = x.split(':').map(Number); return mm * 60 + ss; });
  near(r[0], 30, 2, 'drag across a row: start');
  near(r[1], 39, 2, 'drag across a row: end');
  await page.getByTestId('zoom-in').click();
  if (!/visible/.test(await page.getByTestId('zoom-label').innerText())) throw new Error('zoom label');
  await shot('editor-rows');
  await page.getByTestId('all-layers').click();
  await page.getByTestId('scrub').waitFor();
  return 'rows, clips, drag, zoom';
});

await check('what is heard follows the timeline: strength is a blend (Gentle: 0.749 noise removal + 0.251 recording), a layer switched off drops out, Original is the recording alone, Edited brings everything back', async () => {
  await page.getByTestId('step-home').click();
  const box = await scrubBox();
  await page.mouse.click(box.x + (20 / J1.total) * box.width, box.y + 30); // inside the Gentle part
  await sleep(200);
  let g = await gains();
  near(g.noise, 1 - 10 ** (-12 / 20), 0.002, 'Gentle: noise removal');
  near(g.source, 10 ** (-12 / 20), 0.002, 'Gentle: recording');
  await page.mouse.click(box.x + (5 / J1.total) * box.width, box.y + 30); // in the Maximum part: nothing of the recording
  await sleep(200);
  g = await gains();
  near(g.noise, 1, 0.001, 'Maximum: noise removal');
  near(g.source, 0, 0.001, 'Maximum: recording');
  await page.mouse.click(box.x + (20 / J1.total) * box.width, box.y + 30);
  await hearChip('noise').click();
  await sleep(150);
  g = await gains();
  if (!(g.source === 1 && g.noise === 0)) throw new Error('noise removal off: ' + JSON.stringify(g));
  await hearChip('noise').click();
  await page.getByTestId('orig').click();
  await sleep(150);
  g = await gains();
  if (!(g.source === 1 && g.noise === 0 && g.level === 1)) throw new Error('Original: ' + JSON.stringify(g));
  if (!/the original/.test(await page.getByTestId('hearing').innerText())) throw new Error('hearing text');
  await page.getByTestId('edited').click();
  await sleep(150);
  g = await gains();
  if (!(g.noise > 0.7)) throw new Error('Edited: ' + JSON.stringify(g));
  if (!/applied when you save/.test(await page.getByTestId('level-note').innerText())) throw new Error('no note about Leveling');
  return JSON.stringify(g);
});

/* ------------------------------------------------------------------------------------------------ 4. the price, saving, the old way */

await check('the price comes from the server a moment after the last change (not on every click), says "Process changes · $", lists what runs under Details, and follows the next change', async () => {
  const title = page.getByTestId('cost-title');
  await page.waitForFunction(() => /Process changes · \$\d/.test(document.querySelector('[data-testid=cost-title]')?.textContent || ''), null, { timeout: 15000 });
  const first = await title.innerText();
  if (!/^Process changes · \$0\.\d+$/.test(first)) throw new Error('title: ' + first);
  const n1 = asked('POST', /\/quote$/).length;
  if (n1 < 1 || n1 > 5) throw new Error(`${n1} quotes for six changes (they are asked for a moment after the last one, not for each)`);
  const body = JSON.parse(asked('POST', /\/quote$/).pop().body);
  if (body.timeline?.version !== 1 || !body.timeline.layers.noise.length) throw new Error('the quote has no timeline');
  await page.getByTestId('cost-details').click();
  const list = (await page.getByTestId('cost-detail-list').innerText()).replace(/\s+/g, ' ');
  if (!/Leveling \$0\.\d+: processes .* to do .* well/.test(list) || !/Includes a small amount for each run\. Parts are rounded up to whole seconds\./.test(list)) throw new Error('details: ' + list);
  await shot('editor-cost');
  // the next change: a Leveling off for the Gentle part makes the price smaller, and a new quote is asked for
  await page.getByTestId('step-range').click();
  const sub = await page.getByTestId('cost-sub').innerText().catch(() => '');
  await dragSelect(15, 25, J1.total);
  await page.getByTestId('next-change').click();
  await page.getByTestId('toggle-leveling').uncheck();
  await page.waitForFunction((was) => { const t = document.querySelector('[data-testid=cost-title]')?.textContent; return /Process changes/.test(t) && t !== was; }, first, { timeout: 15000 });
  const second = await title.innerText();
  return `${first} -> ${second}${sub ? '' : ''}`;
});

await check('a paid save is a process: it asks for exactly the price it was shown (maxCostMicros), the job runs again, the new file comes down by itself, the ledger has one "Edited" line, and the files now hold this timeline ("No changes")', async () => {
  const bal0 = await balance();
  const quoted = asked('POST', /\/quote$/).pop();
  const quoteMicros = (await apiCall(page, 'POST', `jobs/${J1.id}/quote`, JSON.parse(quoted.body))).body.estCostMicros;
  const before = got.length;
  await page.getByTestId('cost-save').click();
  await page.getByTestId('cost-bar').getAttribute('data-state');
  await waitCardDone(page);
  await page.getByTestId('result').waitFor({ timeout: 90000 });
  await waitDownloads(before + 1);
  if (got.length !== before + 1 || !/\.flac$/.test(got[got.length - 1])) throw new Error('the new file did not come down by itself: ' + got.join(', '));
  const proc = asked('POST', /\/process$/);
  if (proc.length !== 1) throw new Error(`${proc.length} process calls`);
  const sent = JSON.parse(proc[0].body);
  if (sent.maxCostMicros !== quoteMicros || !sent.timeline?.layers?.noise) throw new Error('process body: ' + proc[0].body.slice(0, 300));
  const bal1 = await balance();
  if (!(bal0 - bal1 > 0 && bal0 - bal1 <= quoteMicros)) throw new Error(`balance ${bal0} -> ${bal1} for a quote of ${quoteMicros}`);
  const j = (await apiCall(page, 'GET', `jobs/${J1.id}`)).body;
  if (!j.edit?.hasTimeline) throw new Error('the job has no timeline of its own now');
  await gotoApp('/app/billing');
  await page.getByTestId('ledger').waitFor();
  const rows = (await page.locator('[data-testid=ledger] tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' ')).filter((t) => /Edited \(/.test(t));
  if (rows.length !== 1 || !/Edited \(Leveling\)/.test(rows[0])) throw new Error('ledger: ' + rows.join(' | '));
  await page.goto(`${BASE}/app/job/${J1.id}`);
  await openEditor();
  const c = (await chips('noise')).map((x) => x.text).join(' | ');
  if (!/Strong/.test(c) || !/Gentle/.test(c)) throw new Error('the saved parts are not in the editor: ' + c);
  if (await page.locator('[data-testid=layer-chip][data-pending=yes]').count()) throw new Error('parts marked as needing processing after they were processed');
  if ((await page.getByTestId('cost-title').innerText()) !== 'No changes yet') throw new Error('cost bar after saving');
  return `${quoteMicros} micros, ${rows[0]}`;
});

await check('a job made the old way still works: Change and save again keeps its choices (file type, name, subtitle files), links to Fix a moment, and an edited job says its sound is changed there now', async () => {
  const old = await makeJob({ file: 'noisy.wav', leveling: true });
  await page.getByTestId('again').locator('summary').click();
  await page.getByTestId('again-fields').waitFor();
  for (const id of ['again-paid', 'again-save', 'again-editor-link', 'again-leveling']) if (!(await page.getByTestId(id).count())) throw new Error(`${id} missing for a job made the old way`);
  await page.getByTestId('again-open-editor').click();
  await page.getByTestId('editor').waitFor({ state: 'visible' });
  await page.goto(`${BASE}/app/job/${J1.id}`);
  await page.getByTestId('again').locator('summary').click();
  await page.getByTestId('again-fields').waitFor();
  if (!(await page.getByTestId('again-edited').count())) throw new Error('an edited job does not say where its sound is changed');
  if (await page.getByTestId('again-paid').count()) throw new Error('an edited job still offers the old re-run choices');
  const t = await page.getByTestId('again-fields').innerText();
  if (!/Fix a moment/.test(t)) throw new Error('no link to the editor: ' + t.slice(0, 300));
  return `old job ${old.slice(0, 8)} keeps its choices; edited job points to the editor`;
});

/* ------------------------------------------------------------------------------------------------ 5. suggestions (the server's) and the free save */

const J2 = { id: '' };
await check('suggestions made by the server: the row says how many, the pins and cards show them with what was heard; Apply paints the part (and says if it needs processing), Undo takes it back, Show me selects it, Dismiss removes the card (and stays dismissed after a reload)', async () => {
  J2.id = await makeJob({ file: 'noisy.wav', leveling: false });
  const cachedNow = (await apiCall(page, 'GET', `jobs/${J2.id}/timeline`)).body.cached;
  const made = cachedNow.heavy.length ? 'heavy' : 'light';
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
  await page.goto(`${BASE}/app/job/${J2.id}`);
  const badge = page.getByTestId('fix-badge');
  await badge.waitFor();
  if ((await badge.innerText()) !== '3 suggestions') throw new Error('badge: ' + (await badge.innerText()));
  await openEditor();
  if ((await page.getByTestId('pin').count()) !== 3) throw new Error('pins');
  const cards = page.getByTestId('sg-card');
  if ((await cards.count()) !== 3) throw new Error('cards');
  const t0 = (await cards.nth(0).innerText()).replace(/\s+/g, ' ');
  if (!/0:02–0:05 is noisier\. Strong here\?/.test(t0) || !/You hear it right away/.test(t0)) throw new Error('first card: ' + t0);
  const t1 = (await cards.nth(1).innerText()).replace(/\s+/g, ' ');
  if (!/Strong here\?/.test(t1) || !/Needs processing/.test(t1) || /You hear it right away/.test(t1)) throw new Error('a different remover than the one made is not heard right away: ' + t1);
  const t2 = (await cards.nth(2).innerText()).replace(/\s+/g, ' ');
  if (!/sounds distorted here \(clipped\)\. Rebuild just this part\?/.test(t2) || !/Needs processing/.test(t2) || /You hear it right away/.test(t2)) throw new Error('third card: ' + t2);
  await shot('editor-suggestions');
  // Show me
  await cards.nth(0).getByTestId('sg-show').click();
  if (!/0:02–0:05/.test(await page.getByTestId('pp-range').innerText())) throw new Error('Show me did not choose the part');
  // Apply, then Undo
  await cards.nth(0).getByTestId('sg-apply').click().catch(() => {});
  await page.getByTestId('step-home').click();
  const applied = page.locator('[data-testid=sg-card][data-state=applied]');
  await applied.waitFor();
  if (!/Strong · 0:02–0:05/.test((await chips('noise')).map((x) => x.text).join(' | '))) throw new Error('the part was not painted: ' + JSON.stringify(await chips('noise')));
  await applied.getByTestId('sg-undo').click();
  if (await applied.count()) throw new Error('undo left the card applied');
  if (/Strong · 0:02/.test((await chips('noise')).map((x) => x.text).join(' | '))) throw new Error('undo did not take the part back');
  if ((await page.getByTestId('cost-title').innerText()) !== 'No changes yet') throw new Error('cost after undo');
  // Dismiss (and after a reload it is still dismissed; the row says one suggestion)
  await page.getByTestId('sg-card').nth(2).getByTestId('sg-dismiss').click();
  if ((await page.getByTestId('sg-card').count()) !== 2 || (await page.getByTestId('pin').count()) !== 2) throw new Error('dismiss');
  await page.goto(`${BASE}/app/job/${J2.id}`);
  await page.getByTestId('fix-badge').waitFor();
  if ((await page.getByTestId('fix-badge').innerText()) !== '2 suggestions') throw new Error('dismissed one is back: ' + (await page.getByTestId('fix-badge').innerText()));
  return 'badge, pins, cards, Show me, Apply, Undo, Dismiss';
});

await check('a change that needs nothing new is free: the bar says "Changes: Free", Save is an export (no process, no charge), the new file comes down by itself', async () => {
  await openEditor();
  const bal0 = await balance();
  await dragSelect(2, 6, 10.6);
  await page.getByTestId('next-change').click();
  await page.getByTestId('noise-balanced').click();
  await page.waitForFunction(() => /Changes:\s*Free/.test(document.querySelector('[data-testid=cost-title]')?.textContent || ''), null, { timeout: 15000 });
  await shot('editor-free');
  const before = got.length;
  await page.getByTestId('cost-save').click();
  await waitCardDone(page);
  await page.getByTestId('result').waitFor({ timeout: 90000 });
  await waitDownloads(before + 1);
  if (got.length !== before + 1) throw new Error('no new file by itself');
  if (asked('POST', new RegExp(`/jobs/${J2.id}/export$`)).length !== 1 || asked('POST', new RegExp(`/jobs/${J2.id}/process$`)).length) throw new Error('export / process calls');
  const bal1 = await balance();
  if (bal1 !== bal0) throw new Error(`a free save cost ${bal0 - bal1} micros`);
  const j = (await apiCall(page, 'GET', `jobs/${J2.id}`)).body;
  if (!j.edit?.hasTimeline) throw new Error('no timeline after the save');
  return 'export, free';
});

await check('errors are plain: a price the person cannot pay says so with a link to add credit and Save is off; a timeline the server refuses shows the server\'s words', async () => {
  const id = J1.id;
  await page.goto(`${BASE}/app/job/${id}`);
  await openEditor();
  // empty the balance for this account by a failing price: fake a quote answer that is too dear
  await page.route(`**/api/jobs/${id}/quote`, (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
    estCostMicros: 9_000_000, estCostUsd: 9, free: false, unchanged: false, needsProcess: true, approximate: false, lines: [{ item: 'leveling', label: 'Leveling', quantity: 1, seconds: 48, micros: 9_000_000 }], steps: { leveling: { blocks: 1, seconds: 48 } }, asked: {},
    timeline: {}, expiresAt: 1, availableMicros: 1_000_000, affordable: false,
  }) }));
  await dragSelect(5, 9, J1.total);
  await page.getByTestId('next-change').click();
  await page.getByTestId('noise-gentle').click();
  await page.getByTestId('cost-poor').waitFor({ timeout: 15000 });
  const t = await page.getByTestId('cost-poor').innerText();
  if (!/costs \$9\.00, but your balance is \$1\.00/.test(t) || !/Add credit/.test(t)) throw new Error('poor: ' + t);
  if (!(await page.getByTestId('cost-save').isDisabled())) throw new Error('Save is on although the price is too high');
  await page.unroute(`**/api/jobs/${id}/quote`);
  await page.route(`**/api/jobs/${id}/quote`, (route) => route.fulfill({ status: 400, contentType: 'application/json', body: JSON.stringify({ detail: 'Voice rebuilding is paused right now. Try again later, or choose Denoising.' }) }));
  await page.getByTestId('noise-balanced').click();
  await page.getByTestId('cost-sub').waitFor({ timeout: 15000 });
  if (!/paused right now/.test(await page.getByTestId('cost-sub').innerText())) throw new Error('the server\'s words are not shown');
  await page.unroute(`**/api/jobs/${id}/quote`);
  return 'insufficient balance and a refused timeline are shown plainly';
});

/* ------------------------------------------------------------------------------------------------ 6. the upload form */

await check('the upload form: "Choose parts (optional)" is collapsed under the options; the same editor, whole file by default; a part chosen makes Start show the server\'s price for the parts (POST /api/jobs/estimate with the timeline) and the job is made with that timeline', async () => {
  await gotoApp('/app');
  await dropFile(page, join(FILES, 'noisy.wav'));
  await pick(page, 'noise', 'denoise');
  await page.locator('label:has(input[name=strength][value=max])').click();
  await page.locator('input[name=leveling]').setChecked(true);
  const box = page.getByTestId('parts-box');
  await box.waitFor();
  if (await box.evaluate((e) => e.open)) throw new Error('not collapsed');
  if (!/Now: the whole file/.test(await page.getByTestId('parts-now').innerText())) throw new Error('parts-now');
  const startWhole = await page.getByTestId('start').innerText();
  await page.getByTestId('parts-summary').click();
  await page.getByTestId('editor').waitFor();
  if ((await page.getByTestId('editor').getAttribute('data-mode')) !== 'plan') throw new Error('mode');
  if (await page.getByTestId('hear-chips').count()) throw new Error('listening choices in the form (nothing is made yet)');
  const est0 = asked('POST', /\/jobs\/estimate$/).length;
  await page.getByTestId('pick-part').click();
  await page.getByTestId('next-change').click();
  await page.getByTestId('noise-gentle').click(); // (the base is Maximum)
  await page.waitForFunction((was) => { const t = document.querySelector('[data-testid=start]')?.textContent; return t !== was && /about \$/.test(t); }, startWhole, { timeout: 15000 }).catch(() => {});
  await sleep(800);
  const est = asked('POST', /\/jobs\/estimate$/).slice(est0);
  if (!est.length) throw new Error('no estimate was asked of the server');
  const body = JSON.parse(est[est.length - 1].body);
  if (!body.timeline?.layers?.noise?.length || body.denoiseTier != null || 'atten_lim_db' in (body.options || {})) throw new Error('estimate body: ' + est[est.length - 1].body.slice(0, 300));
  const startNow = await page.getByTestId('start').innerText();
  if (!/Start – about \$0\.0\d+/.test(startNow)) throw new Error('Start: ' + startNow);
  if (!/Now: the whole file, with some parts done differently/.test(await page.getByTestId('parts-now').innerText())) throw new Error('parts-now after');
  await shot('form-parts');
  const before = got.length;
  await page.getByTestId('start').click();
  await page.waitForURL(JOB_PATH, { timeout: 90000 });
  const id = jobIdOf(page);
  const created = asked('POST', /\/api\/jobs$/).pop();
  const sent = JSON.parse(created.body);
  if (!sent.timeline || sent.denoiseTier != null || 'noise' in sent.options || 'leveling' in sent.options) throw new Error('create body: ' + created.body.slice(0, 400));
  await waitCardDone(page);
  await waitDownloads(before + 1);
  const t = (await apiCall(page, 'GET', `jobs/${id}/timeline`)).body;
  if (!t.custom && !t.timeline.layers.noise.some((p) => p.strength_db === 12)) throw new Error('the job holds ' + JSON.stringify(t.timeline));
  if (!t.timeline.layers.noise.some((p) => p.strength_db === 12) || !t.timeline.layers.noise.some((p) => p.strength_db === 100)) throw new Error('the parts: ' + JSON.stringify(t.timeline.layers.noise));
  return `${startNow}; made with ${t.timeline.layers.noise.length} noise parts`;
});

await check('"Only parts" processes only the chosen parts (a smaller price than the whole file) and with no part chosen Start says to choose one', async () => {
  await gotoApp('/app');
  await dropFile(page, join(FILES, 'speech-long.wav'));
  await pick(page, 'noise', 'denoise');
  await page.locator('label:has(input[name=strength][value=max])').click();
  await page.getByTestId('estimate').waitFor();
  const whole = +(/\$(\d+\.\d+)/.exec(await page.getByTestId('start').innerText())[1]);
  await page.getByTestId('parts-summary').click();
  await page.getByTestId('editor').waitFor();
  await page.getByTestId('plan-only').click();
  await page.getByText('Choose at least one part to process').waitFor();
  if (!(await page.getByTestId('start').isDisabled())) throw new Error('Start is on with nothing to process');
  await dragSelect(10, 20, 48.4);
  await page.getByTestId('next-change').click();
  await page.getByTestId('noise-strong').click();
  await page.waitForFunction(() => /Start – about \$/.test(document.querySelector('[data-testid=start]')?.textContent || ''), null, { timeout: 15000 });
  await sleep(600);
  const only = +(/\$(\d+\.\d+)/.exec(await page.getByTestId('start').innerText())[1]);
  if (!(only < whole)) throw new Error(`only parts $${only} is not less than the whole file $${whole}`);
  const body = JSON.parse(asked('POST', /\/jobs\/estimate$/).pop().body);
  const total = body.timeline.layers.noise.reduce((s, p) => s + p.end - p.start, 0);
  if (!(total < 12 && total >= 9)) throw new Error('the timeline covers ' + total + ' s');
  await page.getByTestId('plan-whole').click();
  if (await page.getByTestId('start').isDisabled()) throw new Error('back to the whole file: Start is off');
  return `whole file $${whole}, only parts $${only}`;
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
  await dropFile(page, PARTS);
  await pick(page, 'noise', 'denoise');
  await page.getByTestId('parts-summary').click();
  await page.getByTestId('editor').waitFor();
  await page.getByTestId('sg-progress').waitFor({ timeout: 20000 });
  if (!/\d+%/.test(await page.getByTestId('sg-progress').innerText())) throw new Error('no percentage');
  await page.getByTestId('sg-card').first().waitFor({ timeout: 150000 });
  await page.getByTestId('sg-progress').waitFor({ state: 'detached', timeout: 20000 });
  const cards = (await page.getByTestId('sg-card').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
  const phone = cards.find((t) => /(phone call|muffled|echoey)/.test(t) && /Rebuild/.test(t));
  if (!phone) throw new Error('no suggestion to rebuild the narrow, dull half: ' + cards.join(' | '));
  const start = /^(\d+):(\d\d)–(\d+):(\d\d)/.exec(phone);
  near(+start[1] * 60 + +start[2], 24, 3, 'the narrow half starts at 24 s');
  await page.getByTestId('waveform').evaluate((c) => c.dataset.known);
  if ((await page.getByTestId('waveform').getAttribute('data-known')) !== 'yes') throw new Error('no waveform');
  await page.getByTestId('sg-card').first().getByTestId('sg-apply').click();
  await page.getByTestId('step-home').click();
  if (!(await chips('rebuild')).some((c) => c.range.split('-')[0] > 20 || +c.range.split('-')[0] >= 20)) throw new Error('rebuild part not painted: ' + JSON.stringify(await chips('rebuild')));
  await shot('form-suggestions');
  return cards.join(' | ').slice(0, 300);
});

await check('looking can be canceled, and a file over 30 minutes says "Suggestions appear after the first run" (nothing is measured)', async () => {
  await gotoApp('/app');
  await dropFile(page, join(FILES, 'long10min.mp3'));
  await pick(page, 'noise', 'denoise');
  await page.getByTestId('parts-summary').click();
  await page.getByTestId('sg-progress').waitFor({ timeout: 30000 });
  await page.getByTestId('sg-cancel').click();
  await page.getByTestId('sg-canceled').waitFor({ timeout: 30000 });
  await gotoApp('/app');
  await dropFile(page, LONG);
  await pick(page, 'noise', 'denoise');
  await page.getByTestId('parts-summary').click();
  await page.getByTestId('sg-unavailable').waitFor();
  if (!/Suggestions appear after the first run/.test(await page.getByTestId('sg-unavailable').innerText())) throw new Error('words');
  if (await page.getByTestId('sg-progress').count()) throw new Error('measuring a long file');
  await page.getByTestId('pick-part').click(); // (choosing parts yourself still works)
  await page.getByTestId('range-step').waitFor();
  return 'canceled; over 30 minutes: after the first run';
});

/* ------------------------------------------------------------------------------------------------ 7. phone */

for (const scheme of ['light', 'dark']) {
  await check(`phone 390x844, ${scheme}: nothing scrolls sideways (the job page, the editor with a part chosen, the rows, the form with parts), the part settings are a sheet at the bottom above the price bar, a finger drags a part`, async () => {
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true, deviceScaleFactor: 2, colorScheme: scheme });
    const p = await ctx.newPage();
    await signIn(p, EMAIL);
    const wide = async (what) => {
      const w = await p.evaluate(() => ({ doc: document.documentElement.scrollWidth, body: document.body.scrollWidth, inner: window.innerWidth }));
      if (w.doc > w.inner + 1 || w.body > w.inner + 1) throw new Error(`${what}: the page is ${Math.max(w.doc, w.body)} px wide in ${w.inner}`);
    };
    await p.goto(`${BASE}/app/job/${J1.id}`);
    await p.getByTestId('fix-entry').waitFor();
    await wide('job page');
    await p.getByTestId('fix-entry').click();
    await p.getByTestId('editor').waitFor();
    await p.waitForTimeout(800);
    await p.getByTestId('scrub').scrollIntoViewIfNeeded();
    await wide('editor');
    // a finger: a tap moves the playhead, a drag across the bar chooses a part
    const box = await p.getByTestId('scrub').boundingBox();
    const cdp = await ctx.newCDPSession(p);
    const touch = (type, x, y) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: type === 'touchEnd' ? [] : [{ x, y }] });
    const y = box.y + box.height / 2;
    await touch('touchStart', box.x + box.width * 0.5, y);
    await touch('touchEnd');
    await p.waitForTimeout(150);
    const ph = +(await p.getByTestId('editor').getAttribute('data-time'));
    near(ph, 24, 2, 'a tap moves the playhead');
    await touch('touchStart', box.x + box.width * 0.25, y);
    for (let i = 1; i <= 8; i++) await touch('touchMove', box.x + box.width * (0.25 + i * 0.04), y);
    await touch('touchEnd');
    await p.waitForTimeout(250);
    const text = await p.getByTestId('range-text').innerText();
    const [a, b] = text.split('–').map((x) => { const [mm, ss] = x.split(':').map(Number); return mm * 60 + ss; });
    near(a, 12, 2.5, 'touch drag start');
    near(b, 28, 2.5, 'touch drag end');
    await wide('with a part chosen');
    await p.getByTestId('next-change').click();
    await p.getByTestId('noise-gentle').click();
    await p.waitForFunction(() => /Process changes|Changes: Free/.test(document.querySelector('[data-testid=cost-title]')?.textContent || ''), null, { timeout: 15000 });
    await p.waitForTimeout(300);
    const geo = await p.evaluate(() => {
      const h = document.querySelector('[data-testid=host]'); const c = document.querySelector('[data-testid=cost-bar]');
      const hs = getComputedStyle(h); const cs = getComputedStyle(c); const hr = h.getBoundingClientRect(); const cr = c.getBoundingClientRect();
      return { hostPos: hs.position, hostBottom: hr.bottom, hostTop: hr.top, barPos: cs.position, barTop: cr.top, barBottom: cr.bottom, h: window.innerHeight };
    });
    if (geo.hostPos !== 'fixed' || geo.barPos !== 'fixed') throw new Error('not a sheet and a bar: ' + JSON.stringify(geo));
    if (!(geo.hostBottom <= geo.barTop + 1) || !(geo.barBottom >= geo.h - 1) || !(geo.hostTop > geo.h * 0.4)) throw new Error('the sheet is not between the page and the bar: ' + JSON.stringify(geo));
    await wide('with the sheet open');
    await shot(`phone-${scheme}-sheet`, p);
    await p.getByTestId('pp-done').click();
    await p.getByTestId('all-layers').scrollIntoViewIfNeeded();
    await p.getByTestId('all-layers').click();
    await p.getByTestId('rows').waitFor();
    await wide('rows');
    await shot(`phone-${scheme}-rows`, p);
    await p.getByTestId('fix-close').click();
    // the form
    await p.goto(`${BASE}/app`);
    await p.getByTestId('available').waitFor();
    await p.setInputFiles('input[type=file]', join(FILES, 'noisy.wav'));
    await p.getByTestId('start').waitFor({ timeout: 20000 });
    await p.waitForFunction(() => !document.querySelector('[data-testid=reco-listening]'), null, { timeout: 30000 });
    await p.locator('label.choice:has(input[name=noise][value=denoise])').click();
    await p.getByTestId('parts-summary').click();
    await p.getByTestId('editor').waitFor();
    await p.waitForTimeout(500);
    await wide('form with parts');
    await p.getByTestId('pick-part').click();
    await wide('form with a part');
    await shot(`phone-${scheme}-form`, p);
    await ctx.close();
    return 'no sideways scroll, sheet and bar placed, touch works';
  });
}

/* ------------------------------------------------------------------------------------------------ 8. words and prices */

await check('words: the editor, its panels, the cost bar and the comparison use plain words (no model names, no "premium", never "polish", no "our servers"); "Which one should I pick?" is in the form, in Change and save again and in the part panel with the owner\'s comparison', async () => {
  const scan = async (what) => {
    const text = await page.evaluate(() => { const parts = [document.body.innerText]; for (const d of document.querySelectorAll('details')) parts.push(d.textContent); for (const e of document.querySelectorAll('[aria-label],[title]')) parts.push(e.getAttribute('aria-label') || '', e.getAttribute('title') || ''); return parts.join('\n'); });
    const bad = BANNED.map((re) => re.exec(text)).filter(Boolean).map((m) => `${what}: "${m[0]}" in …${text.slice(Math.max(0, m.index - 40), m.index + 50).replace(/\s+/g, ' ')}…`);
    if (bad.length) throw new Error(bad.join(' | '));
  };
  await page.goto(`${BASE}/app/job/${J2.id}`);
  await openEditor();
  await dragSelect(2, 6, 10.6);
  await page.getByTestId('next-change').click();
  await page.getByTestId('noise-strong').click();
  await page.getByTestId('cost-title').waitFor();
  await sleep(1200);
  await scan('the editor');
  const which = page.getByTestId('which-one-part');
  const t = (await which.textContent()).replace(/\s+/g, ' ');
  for (const need of [/Denoising/, /Voice rebuild/, /exactly as recorded and never changes words/, /Laughs, shouts, whispers and singing stay as they were/, /game sounds/, /from scratch/, /slightly too clean/, /come out oddly/, /can change a word, so listen before you publish/, /90 minutes/, /Costs more/, /many languages/]) {
    if (!need.test(t)) throw new Error(`the comparison lacks ${need}`);
  }
  if (await which.evaluate((e) => e.open)) throw new Error('the comparison is not collapsed');
  const hints = (await page.getByTestId('pl-rebuild').innerText()).replace(/\s+/g, ' ');
  if (!/Rebuilds the voice in this part only/.test(hints) && !/Paused right now/.test(hints)) throw new Error('the one-line hint of Voice rebuild is gone: ' + hints);
  if (!/Nearly all of the noise goes/.test(await page.getByTestId('pl-noise').innerText())) throw new Error('the one-line hint of noise removal is gone');
  // the form
  await gotoApp('/app');
  await dropFile(page, join(FILES, 'noisy.wav'));
  await pick(page, 'noise', 'denoise');
  if (!(await page.getByTestId('which-one-form').count())) throw new Error('no comparison in the form');
  await page.getByTestId('parts-summary').click();
  await page.getByTestId('editor').waitFor();
  await scan('the form with parts');
  // Change and save again
  await page.goto(`${BASE}/app/job/${J1.id}`);
  const old = await makeJob({ file: 'noisy.wav', leveling: true });
  await page.getByTestId('again').locator('summary').click();
  await page.getByTestId('again-fields').waitFor();
  if (!(await page.getByTestId('which-one-again').count())) throw new Error('no comparison in Change and save again');
  await scan('change and save again');
  return `${old.slice(0, 8)}: three places, ${t.length} characters of comparison`;
});

await check('the automatic settings say which one they picked and why, in the terms of the comparison ("Voice rebuild: the voice sounds echoey" / "Denoising: the voice is already clear ...")', async () => {
  await gotoApp('/app');
  await page.setInputFiles('input[type=file]', join(FILES, 'speech-long.wav'));
  await page.getByTestId('start').waitFor({ timeout: 20000 });
  await page.waitForFunction(() => !document.querySelector('[data-testid=reco-listening]'), null, { timeout: 30000 });
  const pickText = (await page.getByTestId('reco-pick').innerText()).trim();
  if (!/^(Voice rebuild: .+\.|Denoising: the voice is already clear, so it is kept exactly as recorded and only the background sound is taken out\.)$/.test(pickText)) throw new Error('pick line: ' + pickText);
  await page.locator('[data-testid=reco-why]').evaluate((e) => e.closest('details').setAttribute('open', ''));
  const why = await page.getByTestId('reco-why').innerText();
  if (!why.includes(pickText)) throw new Error('the reasons do not start with the pick line');
  return pickText;
});

await check('echo removal has its own price: the price table row and its line, the form says it next to the option and adds it to the estimate, and a job with echo removal lists it as a line of its own (Echo removal)', async () => {
  const pricing = (await (await fetch(`${BASE}/api/billing/pricing`)).json()).rates.find((r) => r.product === 'echo');
  if (!pricing || pricing.usdPerMinute !== 0.0026 || pricing.label !== 'Echo removal') throw new Error('price list: ' + JSON.stringify(pricing));
  await page.goto(`${BASE}/#pricing`);
  await page.waitForSelector('table tbody tr');
  const row = (await page.locator('table.price-table tr', { hasText: 'Echo removal' }).first().innerText()).replace(/\s+/g, ' ');
  if (!/Echo removal.*Speaker echo taken out.*\$0\.0026 \/ min/.test(row)) throw new Error('price table row: ' + row);
  await gotoApp('/app');
  await dropFile(page, join(FILES, 'echo2.mp4'));
  await pick(page, 'noise', 'denoise');
  const hint = (await page.getByTestId('echo-hint').innerText()).replace(/\s+/g, ' ');
  if (!/Echo removal costs \$0\.0026 a minute on top/.test(hint)) throw new Error('hint: ' + hint);
  const est0 = (await page.getByTestId('estimate').innerText()).match(/\$(\d+\.\d+)/)[1];
  await page.locator('input[name=echo]').check();
  await sleep(300);
  const est1 = (await page.getByTestId('estimate').innerText()).match(/\$(\d+\.\d+)/)[1];
  if (!(+est1 > +est0)) throw new Error(`the estimate did not grow with echo removal: ${est0} -> ${est1}`);
  const e = (await apiCall(page, 'POST', 'jobs/estimate', { kind: 'denoise', denoiseTier: 'standard', durationSeconds: 600, options: { echo_reference_track: true } })).body;
  const line = e.lines.find((l) => l.item === 'echo');
  if (!line || line.label !== 'Echo removal' || line.micros !== 26000 || e.estCostMicros !== e.lines.reduce((s, l) => s + l.micros, 0)) throw new Error('estimate lines: ' + JSON.stringify(e.lines));
  await page.goto(`${BASE}/developers`);
  await page.waitForSelector('pre.code');
  const dev = (await page.locator('main').innerText()).replace(/\s+/g, ' ');
  if (!/Echo removal.*\$0\.0026/.test(dev) || !/own price, per second, in its own line/.test(dev)) throw new Error('Developers page does not give the echo price');
  return `$${est0} -> $${est1} with echo removal; ${line.micros} micros for ten minutes`;
});

await check('the editor of a job made with echo removal has the Echo removal layer (a Hear chip, a row, a chip row)', async () => {
  const id = await makeJob({ file: 'echo2.mp4', leveling: true, echo: true });
  const j = (await apiCall(page, 'GET', `jobs/${id}`)).body;
  if (!j.options.echo_reference_track) throw new Error('the job was made without echo removal');
  const t = (await apiCall(page, 'GET', `jobs/${id}/timeline`)).body;
  if (!t.secondTrack) throw new Error('no second track on the job: ' + JSON.stringify(t).slice(0, 200));
  await openEditor();
  const names = (await page.locator('[data-testid=hear-chips] label').allInnerTexts()).join('|');
  if (names !== 'Noise removal|Voice rebuild|Echo removal|Leveling') throw new Error('hear chips: ' + names);
  if (!(await page.getByTestId('crow-echo').count())) throw new Error('no chip row for echo removal');
  const bill = (await apiCall(page, 'GET', `jobs/${id}`)).body;
  if (!(bill.costMicros > 0)) throw new Error('cost ' + bill.costMicros);
  await gotoApp('/app/billing');
  await page.getByTestId('ledger').waitFor();
  return `${names}; the job cost ${bill.costMicros} micros`;
});

await browser.close();
const bad = logs.filter((l) => !/Failed to load resource|favicon|DevTools|net::ERR/.test(l));
if (bad.length) console.log('console:', bad.slice(0, 8).join(' | ').slice(0, 800));
const { summary } = await import('./helpers.mjs');
const failed = summary().filter((r) => !r.ok);
console.log(`${summary().length - failed.length} passed, ${failed.length} failed`);
process.exit(failed.length ? 1 : 0);
