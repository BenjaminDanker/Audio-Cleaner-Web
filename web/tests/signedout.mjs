// The free tools without an account: /app (the Cleanup page and the editor) and the Live page open signed out, the Free noise removal and
// Free subtitles run to the end with nothing about the file sent (no job, no upload, no estimate), a plan that needs our side says
// "Sign in to start" with its price worked out from the public price list, signing in brings the file and the editor's state back (and
// the kept copy is deleted, or is not used after a day), the Live page's free test starts signed out and a paid session asks to sign in,
// the pages that belong to an account still ask for sign-in, and signed in nothing changes. Automatic works without an account as for
// everybody (in the browser; nothing about the file is sent) but picks only among the free options: Free noise removal at the strength it
// finds (the strongest Free one where it would rebuild the voice or use the heavy remover), Free subtitles where there is speech, no Leveling,
// and the suggestions for moments offer Free strengths; with Automatic off every free option is on; the switch shows and saves this
// browser's choice; and the kept work comes back after signing in with its tracks and its Automatic state as they were left.
// Needs the local backend (port 8792, see helpers.mjs) with `ACW_MODEL_HOST=http://127.0.0.1:8798/ npm run build` output and the fixtures.
// Usage: node tests/signedout.mjs
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { noisy, wavBytes } from './reco-audio.mjs';
import {
  BASE, FILES, check, dragRange, dropFile, launch, loginLink, mainButton, mirror, pick, resultAttr, selectionTitle, setLeveling, signIn, summary, waitResult,
} from './helpers.mjs';

const EMAIL = `signedout-${Date.now()}@example.com`;
const NOISY = join(FILES, 'noisy.wav');
const KEPT = 'acw.draft.saved';
const AUTO = 'acw.autoSettings'; // this browser's choice for the editor's Automatic switch (absent: on)
const MODELS = 'http://127.0.0.1:8798'; // the model host the free tools (the quick listen and the search for moments too) download the noise remover from
// Generated recordings whose automatic choices are known (tests/reco-ui.mjs): where the signed-in editor sets Standard noise removal at
// Gentle with Standard subtitles (clean), at Strong with the heavy remover and Extra accurate subtitles (loud), Voice rebuild (hot), and
// Balanced with no subtitles because no one speaks (hum).
const GEN = mkdtempSync(join(tmpdir(), 'signedout-'));
const gen = (name, samples, rate = 48000) => { const f = join(GEN, name); writeFileSync(f, wavBytes(samples, rate)); return f; };
const hotOf = (x) => x.map((a) => Math.max(-0.9995, Math.min(0.9995, a * 8)));
const GENERATED = {
  clean: gen('clean.wav', noisy(10, 60, 1)),
  loud: gen('loud.wav', noisy(10, 2, 3, 'pink')),
  hot: gen('too-loud.wav', hotOf(noisy(10, 55, 1))),
  hum: gen('hum.wav', Float32Array.from({ length: 12 * 44100 }, (_, i) => 0.05 * Math.sin((2 * Math.PI * 50 * i) / 44100)), 44100),
};

const { browser, context, page, logs } = await launch([], { acceptDownloads: true });
await mirror(); // the local model host the free tools download from
const calls = []; // every request to /api/ this context makes: {method, path, body}
context.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.startsWith('/api/')) calls.push({ method: r.method(), path: u.pathname, body: (r.postDataBuffer() || []).length }); });
const downloads = [];
page.on('download', (d) => downloads.push(d));
const sent = () => calls.map((c) => `${c.method} ${c.path}`);
/** Signed out, the page may read who is signed in and the public price list, and nothing else. */
const onlyPublic = (what) => {
  const odd = calls.filter((c) => c.method !== 'GET' || !/^\/api\/(me|billing\/pricing)$/.test(c.path));
  if (odd.length) throw new Error(`${what}: ${sent().join(', ')}`);
};

/** The kept copy in this browser: {name, size, savedAt} or null. (Looks first: opening a database that is not there would make one.) */
const kept = (p) => p.evaluate(async () => {
  if (!(await indexedDB.databases()).some((d) => d.name === 'acw-draft')) return null;
  return new Promise((resolve) => {
    const r = indexedDB.open('acw-draft');
    r.onsuccess = () => {
      const db = r.result;
      if (!db.objectStoreNames.contains('draft')) { db.close(); resolve(null); return; }
      const g = db.transaction('draft').objectStore('draft').get('current');
      g.onsuccess = () => { db.close(); resolve(g.result ? { name: g.result.name, size: g.result.blob.size, savedAt: g.result.savedAt } : null); };
    };
    r.onerror = () => resolve(null);
  });
});
/** Make the kept copy `hours` older (its record, or just the note in localStorage that says when it was kept). */
const age = (p, hours, { record = true, note = true } = {}) => p.evaluate(async ({ hours, record, note, KEPT }) => {
  const ms = hours * 3600 * 1000;
  if (note) localStorage.setItem(KEPT, String(Number(localStorage.getItem(KEPT)) - ms));
  if (!record) return;
  await new Promise((resolve) => {
    const r = indexedDB.open('acw-draft');
    r.onsuccess = () => {
      const db = r.result;
      const tx = db.transaction('draft', 'readwrite');
      const st = tx.objectStore('draft');
      const g = st.get('current');
      g.onsuccess = () => { st.put({ ...g.result, savedAt: g.result.savedAt - ms }, 'current'); };
      tx.oncomplete = () => { db.close(); resolve(); };
    };
  });
}, { hours, record, note, KEPT });

/** A fresh browser context (and its first page) that records what it asks of us (/api/...) and of the model host. `saved`: the Automatic choice this browser starts with (none: it never chose). */
async function watched(saved = null) {
  const storageState = saved ? { cookies: [], origins: [{ origin: BASE, localStorage: [{ name: AUTO, value: saved }] }] } : undefined;
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, ...(storageState ? { storageState } : {}) });
  const seen = { api: [], models: [] };
  ctx.on('request', (r) => {
    const u = new URL(r.url());
    if (u.pathname.startsWith('/api/')) seen.api.push(`${r.method()} ${u.pathname}`);
    else if (u.origin === MODELS) seen.models.push(u.pathname);
  });
  return { ctx, p: await ctx.newPage(), seen };
}
/** What was asked of our side other than the account check and the public prices (nothing about the file is ever sent). */
const sentToUs = (seen) => seen.api.filter((c) => !/^GET \/api\/(me|billing\/pricing)$/.test(c));
/** This browser's saved choice for Automatic ('on', 'off', or null when it never chose). */
const savedAuto = (p) => p.evaluate((k) => localStorage.getItem(k), AUTO);
/** The editor as it stands: the Automatic switch, the choices of the tracks (the sections the whole file is in) and what shows of the quick listen and the moments. */
const editorState = (p) => p.evaluate(() => {
  const sw = document.querySelector('.studio button[data-act=auto]');
  const pressed = (act) => [...document.querySelectorAll(`.studio button[data-act=${act}]`)].find((b) => b.getAttribute('aria-pressed') === 'true')?.dataset.v;
  const id = (t) => document.querySelector(`[data-testid=${t}]`);
  return {
    checked: sw?.getAttribute('aria-checked'), ariaDisabled: sw?.getAttribute('aria-disabled'), tip: sw?.dataset.tip || '',
    voice: ['off', 'noise', 'rebuild'].find((m) => id(`voice-${m}`)?.getAttribute('aria-checked') === 'true'),
    kind: pressed('v-kind'), strength: pressed('v-str'), subs: pressed('s-q'), level: pressed('l-lvl'),
    language: document.querySelector('.studio select[data-act=s-lang]')?.value,
    analysis: ['reco-listening', 'reco-status', 'reco-line', 'sg-progress', 'sg-chip', 'sg-card', 'sg-unavailable', 'sg-canceled', 'sg-error'].filter((t) => id(t)),
  };
});
/** Opens a file on a signed-out /app. With Automatic on it waits for what it set ("Set automatically for this file"); with it off, long enough for a quick listen to have begun. */
async function openSignedOut(p, file, { automatic }) {
  await p.goto(`${BASE}/app`);
  await p.getByTestId('free-no-account').waitFor({ timeout: 15000 });
  await p.setInputFiles('input[type=file]', file);
  await p.getByTestId('start').waitFor({ timeout: 30000 });
  if (automatic) await p.getByTestId('reco-status').waitFor({ timeout: 90000 });
  else { await p.waitForFunction(() => document.querySelector('[data-testid=waveform]')?.dataset.reading === 'no', null, { timeout: 60000 }); await p.waitForTimeout(3000); }
  await p.waitForTimeout(300);
}
const FREE_LINE = /^(Noise removal, free|Subtitles, free) · \d+:\d\d Free$/;
/** Nothing paid anywhere: Free noise removal (strength `str`: 1 Gentle, 2 Balanced, 3 Strong) and Free subtitles (`subs` 1) or none (0) for the whole file, no Leveling, and only free lines in the details. */
async function freeOnly(p, { str, subs }, what) {
  const s = await editorState(p);
  if (s.voice !== 'noise' || s.kind !== 'free' || s.strength !== str) throw new Error(`${what}: voice ${s.voice} / ${s.kind} / strength ${s.strength}, wanted free / ${str}`);
  if (s.subs !== subs || (subs === '1' && s.language !== 'auto')) throw new Error(`${what}: subtitles ${s.subs} / ${s.language}, wanted ${subs}`);
  if (s.level !== '0') throw new Error(`${what}: Leveling ${s.level}`);
  const title = await selectionTitle(p);
  const lines = await detailLines(p);
  if (!/^Whole file/.test(title) || lines.length !== (subs === '1' ? 2 : 1) || lines.some((l) => !FREE_LINE.test(l))) throw new Error(`${what}: ${title}; ${lines.join(' | ')}`);
  if ((await mainButton(p)).trim() !== 'Start · Free') throw new Error(`${what}: button ${await mainButton(p)}`);
  return `${title}; ${lines.join(' | ')}`;
}

/** On a fresh signed-out page: a file, a plan that needs our side (Leveling on 2 to 6 s, Standard noise removal, Standard subtitles), then Sign in to start. */
async function planAndLeave(p, { keepAuto = false } = {}) {
  await p.goto(`${BASE}/app`);
  await dropFile(p, NOISY, { recommended: keepAuto }); // (keepAuto: Automatic stays on and sets what it finds first)
  await pick(p, 'noise', 'denoise');
  await pick(p, 'subs', 'standard');
  await p.getByTestId('waveform').scrollIntoViewIfNeeded();
  await dragRange(p, 2, 6);
  await setLeveling(p, 'podcast');
  await p.getByTestId('output-button').click();
  await p.getByTestId('out-name').fill('my-take.flac');
  await p.getByTestId('output-button').click();
}
const detailLines = async (p) => {
  await p.getByTestId('bar-det').click();
  const lines = await p.locator('[data-testid=details] ul.fl li:not(.tot)').allInnerTexts();
  await p.getByTestId('bar-det').click();
  return lines.map((l) => l.replace(/\s+/g, ' ').trim());
};

await check('signed out: /app opens (no redirect), the header says Sign in, the Cleanup page says the free tools work without an account and links to Sign in, and only the account check and the public prices are asked for', async () => {
  await page.goto(`${BASE}/app`);
  await page.getByTestId('free-no-account').waitFor({ timeout: 15000 });
  if (new URL(page.url()).pathname !== '/app') throw new Error(page.url());
  if (!(await page.locator('.site-header').getByRole('link', { name: 'Sign in' }).count())) throw new Error('no Sign in in the header');
  const text = await page.getByTestId('free-no-account').innerText();
  if (!/free tools work without an account/.test(text) || !(await page.getByTestId('free-no-account').getByRole('link', { name: 'Sign in' }).count())) throw new Error(text);
  if (!(await page.locator('input[type=file]').count())) throw new Error('no file picker');
  if (await page.getByTestId('jobs-note').count() || await page.getByTestId('available').count()) throw new Error('the jobs list or the balance is shown');
  await page.waitForTimeout(800);
  onlyPublic('while the empty page was open');
  return text.replace(/\s+/g, ' ').slice(0, 90);
});

await check('signed out, Automatic on (the default): it listens in the browser and sets only free options for the whole file (Gentle / Strong with the heavy remover / Strong for a voice it would rebuild / Balanced with no subtitles when no one speaks), no Leveling, nothing paid and nothing about the file sent; the switch is on with a tooltip that says it picks among the free options', async () => {
  const w = await watched();
  try {
    const got = [];
    for (const [name, want] of [['clean', { str: '1', subs: '1' }], ['loud', { str: '3', subs: '1' }], ['hot', { str: '3', subs: '1' }], ['hum', { str: '2', subs: '0' }]]) {
      await openSignedOut(w.p, GENERATED[name], { automatic: true });
      const s = await editorState(w.p);
      if (s.checked !== 'true' || s.ariaDisabled) throw new Error(`${name}: switch checked ${s.checked}, aria-disabled ${s.ariaDisabled}`);
      if (s.voice === 'rebuild') throw new Error(`${name}: Voice rebuild was chosen`);
      if (!/Set automatically/.test(await w.p.getByTestId('reco-status').innerText())) throw new Error(`${name}: status ${await w.p.getByTestId('reco-status').innerText()}`);
      got.push(`${name}: ${(await freeOnly(w.p, want, name)).split('; ')[1]}`);
    }
    if (!w.seen.models.length) throw new Error('Automatic did not run: the model host was never asked for the noise remover');
    if (sentToUs(w.seen).length) throw new Error('asked of our side: ' + sentToUs(w.seen).join(', '));
    if (await savedAuto(w.p) !== null) throw new Error('the saved choice was written: ' + await savedAuto(w.p));
    const sw = w.p.locator('.studio button[data-act=auto]');
    if (!/picks among the free options/.test((await editorState(w.p)).tip)) throw new Error('tip: ' + (await editorState(w.p)).tip);
    await sw.hover();
    await w.p.waitForTimeout(1300);
    const shown = await w.p.locator('[role=tooltip]:not([hidden])').allInnerTexts();
    if (!shown.some((x) => /Without an account, Automatic picks among the free options/.test(x))) throw new Error('the tooltip did not show: ' + JSON.stringify(shown));
    return got.join(' || ');
  } finally {
    await w.ctx.close();
  }
});

await check('signed out, Automatic off (saved in this browser): nothing is listened to or sent, and every free option the file can have is on for the whole file (Free noise removal at Balanced, Free subtitles with the language detected, no Leveling); switching Automatic on works as for everybody (saved, free options only), and off again', async () => {
  const w = await watched('off');
  try {
    await openSignedOut(w.p, NOISY, { automatic: false });
    const s = await editorState(w.p);
    if (s.checked !== 'false' || s.ariaDisabled) throw new Error(`switch: checked ${s.checked}, aria-disabled ${s.ariaDisabled}`);
    if (s.analysis.length) throw new Error('the quick listen or the moments showed: ' + s.analysis.join(', '));
    if (w.seen.models.length) throw new Error('the model host was asked for: ' + w.seen.models.join(', '));
    if (sentToUs(w.seen).length) throw new Error('asked of our side: ' + sentToUs(w.seen).join(', '));
    const start = await freeOnly(w.p, { str: '2', subs: '1' }, 'start');
    if (await savedAuto(w.p) !== 'off') throw new Error('the saved choice became ' + await savedAuto(w.p));
    const sw = w.p.locator('.studio button[data-act=auto]');
    await sw.click(); // (an ordinary switch: it runs Automatic in the browser and saves the choice)
    await w.p.waitForFunction(() => document.querySelector('.studio button[data-act=auto]')?.getAttribute('aria-checked') === 'true', null, { timeout: 5000 });
    if (await savedAuto(w.p) !== 'on') throw new Error('switching it on saved ' + await savedAuto(w.p));
    await w.p.getByTestId('reco-status').waitFor({ timeout: 90000 });
    if (!w.seen.models.length) throw new Error('switched on, but it did not listen');
    // (whatever it set, nothing is paid; the strength it finds for this file is the reco-ui suite's business)
    const lines = await detailLines(w.p);
    const on = await editorState(w.p);
    if (on.voice !== 'noise' || on.kind !== 'free' || on.level !== '0' || lines.some((l) => !FREE_LINE.test(l)) || (await mainButton(w.p)).trim() !== 'Start · Free') throw new Error(`switched on: ${JSON.stringify(on)}; ${lines.join(' | ')}`);
    if (sentToUs(w.seen).length) throw new Error('asked of our side: ' + sentToUs(w.seen).join(', '));
    await sw.click();
    await w.p.waitForFunction(() => document.querySelector('.studio button[data-act=auto]')?.getAttribute('aria-checked') === 'false', null, { timeout: 5000 });
    if (await savedAuto(w.p) !== 'off') throw new Error('switching it off saved ' + await savedAuto(w.p));
    return `${start}; on: ${lines.join(' | ')}`;
  } finally {
    await w.ctx.close();
  }
});

await check('signed out, Automatic on: the suggestions for moments offer Free strengths (never Voice rebuild or Standard) and applying them leaves everything free', async () => {
  const w = await watched();
  try {
    await openSignedOut(w.p, join(FILES, 'suggest-two-parts.wav'), { automatic: true });
    await w.p.getByTestId('sg-chip').waitFor({ timeout: 120000 });
    const cards = [];
    for (let i = 0; i < 8; i++) {
      if (!(await w.p.getByTestId('sg-card').count())) {
        if (!(await w.p.getByTestId('sg-chip').count())) break;
        await w.p.getByTestId('sg-chip').click();
      }
      await w.p.getByTestId('sg-card').waitFor({ timeout: 5000 });
      const text = (await w.p.getByTestId('sg-card').innerText()).replace(/\s+/g, ' ').trim();
      cards.push(text);
      if (/rebuild|Standard|Max|Extra/i.test(text.split(' · ').slice(1).join(' · '))) throw new Error('a paid option is suggested: ' + text);
      await w.p.getByTestId('sg-apply').click();
      await w.p.waitForTimeout(400);
    }
    if (!cards.length) throw new Error('no suggestion was offered');
    const lines = await detailLines(w.p);
    const s = await editorState(w.p);
    if (lines.some((l) => !FREE_LINE.test(l)) || s.level !== '0' || (await mainButton(w.p)).trim() !== 'Start · Free') throw new Error(`after applying: ${lines.join(' | ')}; leveling ${s.level}; ${await mainButton(w.p)}`);
    if ((await w.p.getByTestId('track-voice').getByTestId('section').allInnerTexts()).some((t) => /rebuild/i.test(t))) throw new Error('a Voice rebuild section after applying');
    if (sentToUs(w.seen).length) throw new Error('asked of our side: ' + sentToUs(w.seen).join(', '));
    return `${cards.length} applied (${cards.join(' | ')}); ${lines.join(' | ')}`;
  } finally {
    await w.ctx.close();
  }
});

await check('signed out: a Free noise removal run completes (the cleaned file downloads) and nothing about the file is sent: no job, no upload, no price asked for from our side, no request that is not a plain GET', async () => {
  calls.length = 0; downloads.length = 0;
  await dropFile(page, NOISY);
  const label = (await mainButton(page)).trim();
  if (label !== 'Start · Free') throw new Error('button: ' + label);
  await page.getByTestId('start').click();
  await waitResult(page, 'result', 180000);
  const rms = Number(await resultAttr(page, 'rms'));
  if (!(rms > 0.001)) throw new Error('silent ' + rms);
  if (!/\/app\/job\/local-/.test(new URL(page.url()).pathname)) throw new Error('page: ' + page.url());
  if (downloads.length !== 1 || downloads[0].suggestedFilename() !== 'noisy-cleaned.wav') throw new Error('downloads: ' + downloads.map((d) => d.suggestedFilename()));
  onlyPublic('during the run');
  return `rms ${rms.toFixed(4)}; API calls: ${[...new Set(sent())].join(', ')}`;
});

await check('signed out: a Free subtitles run completes (the words are right, the files come down) and nothing about the file is sent', async () => {
  calls.length = 0; downloads.length = 0;
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'speech.wav'));
  await pick(page, 'noise', 'off');
  await pick(page, 'subs', 'browser');
  if ((await mainButton(page)).trim() !== 'Start · Free') throw new Error('button: ' + (await mainButton(page)));
  await page.getByTestId('start').click();
  await waitResult(page, 'result', 240000);
  const cues = Number(await resultAttr(page, 'cues'));
  if (!(cues > 0)) throw new Error('cues ' + cues);
  await page.getByTestId('bar-files').click();
  const preview = await page.getByTestId('subs-preview').innerText();
  if (!/responsible|hearing|damage|exposure/i.test(preview)) throw new Error('words: ' + preview);
  if (!downloads.some((d) => d.suggestedFilename() === 'speech.srt')) throw new Error('downloads: ' + downloads.map((d) => d.suggestedFilename()));
  onlyPublic('during the run');
  return `${cues} cues`;
});

let before; // what the editor held before leaving, to compare with what comes back after signing in
await check('signed out: a plan with a paid part says "Sign in to start" with the price worked out from the public list (no estimate is asked for); Sign in to start keeps the file and goes to sign-in', async () => {
  calls.length = 0;
  await planAndLeave(page);
  const label = (await mainButton(page)).trim();
  if (label !== 'Sign in to start') throw new Error('button: ' + label);
  if (await page.getByTestId('start').isDisabled()) throw new Error('the button is disabled');
  const lines = await detailLines(page);
  if (!lines.some((l) => /^Noise removal · /.test(l) && /\$0\.0\d/.test(l)) || !lines.some((l) => /^Leveling · 0:04/.test(l)) || !lines.some((l) => /^Subtitles, Standard · /.test(l))) throw new Error('details: ' + lines.join(' | '));
  await page.getByTestId('bar-det').click();
  if (!(await page.getByTestId('details-local').count())) throw new Error('no note that the exact price comes with signing in');
  await page.getByTestId('bar-det').click();
  onlyPublic('while a paid plan was chosen');
  before = { title: await selectionTitle(page), lines, out: (await page.getByTestId('output-button').innerText()).replace(/\s+/g, ' ').trim() };
  await page.getByTestId('start').click();
  await page.waitForURL(/\/login$/, { timeout: 15000 });
  const copy = await kept(page);
  if (!copy || copy.name !== 'noisy.wav' || copy.size < 1000) throw new Error('kept: ' + JSON.stringify(copy));
  onlyPublic('while leaving to sign in');
  return `kept ${copy.name} (${copy.size} bytes); details: ${lines.join(' | ')}`;
});

await check('signing in (the link opened in another tab) brings the file and the editor state back as they were, then the kept copy is gone; the price now comes from our side', async () => {
  // (the first tab stays on the sign-in page; a second tab has no memory of where the visitor came from, so it must not depend on that)
  const p2 = await context.newPage();
  const calls2 = [];
  p2.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.startsWith('/api/')) calls2.push(`${r.method()} ${u.pathname}`); });
  await p2.goto(loginLink(EMAIL));
  await p2.getByTestId('draft-restored').waitFor({ timeout: 30000 });
  await p2.getByTestId('start').waitFor();
  await p2.waitForFunction(() => /^Start ·/.test(document.querySelector('[data-testid=start]')?.textContent || ''), null, { timeout: 30000 });
  if (new URL(p2.url()).pathname !== '/app') throw new Error(p2.url());
  const name = await p2.getByTestId('file-name').innerText();
  const title = await selectionTitle(p2);
  const out = (await p2.getByTestId('output-button').innerText()).replace(/\s+/g, ' ').trim();
  const lines = await detailLines(p2);
  const note = await p2.getByTestId('draft-restored').innerText();
  if (name !== 'noisy.wav') throw new Error('file: ' + name);
  if (title !== before.title) throw new Error(`selection: ${before.title} -> ${title}`);
  if (out !== before.out) throw new Error(`output: ${before.out} -> ${out}`);
  if (lines.join('|') !== before.lines.join('|')) throw new Error(`choices: ${before.lines.join(' | ')} -> ${lines.join(' | ')}`);
  if (!/signed in/.test(note)) throw new Error('note: ' + note);
  if (await kept(p2)) throw new Error('the kept copy is still there');
  if (await p2.evaluate((k) => localStorage.getItem(k), KEPT)) throw new Error('the note about the kept copy is still there');
  if (!calls2.includes('POST /api/jobs/estimate')) throw new Error('signed in, the price should be asked of our side: ' + calls2.join(', '));
  // Automatic is as it was left (dropFile switched it off): the restored tracks are not set again
  const auto = await editorState(p2);
  if (auto.checked !== 'false' || auto.ariaDisabled) throw new Error(`the switch after signing in: checked ${auto.checked}, aria-disabled ${auto.ariaDisabled}`);
  if (await savedAuto(p2) !== 'off') throw new Error('the saved choice after signing in: ' + await savedAuto(p2));
  // a second visit does not bring it back
  await p2.goto(`${BASE}/app`);
  await p2.getByTestId('jobs-note').waitFor();
  if (await p2.getByTestId('file-name').count()) throw new Error('it came back a second time');
  await p2.close();
  return `${name}, ${title}, ${out}`;
});

await check('Automatic on when leaving: after signing in the kept editor has the switch on, the tracks as left (not set again), and nothing listened to again', async () => {
  const w = await watched();
  try {
    await planAndLeave(w.p, { keepAuto: true });
    if ((await editorState(w.p)).checked !== 'true') throw new Error('Automatic was not on when leaving');
    const left = await detailLines(w.p);
    await w.p.getByTestId('start').click();
    await w.p.waitForURL(/\/login$/, { timeout: 15000 });
    if (await savedAuto(w.p) !== null) throw new Error('signed out, the saved choice became ' + await savedAuto(w.p));
    const asked = w.seen.models.length; // (what the listen on the signed-out page needed)
    const p2 = await w.ctx.newPage();
    await p2.goto(loginLink(`signedout-on-${Date.now()}@example.com`));
    await p2.getByTestId('draft-restored').waitFor({ timeout: 30000 });
    await p2.waitForFunction(() => /^Start ·/.test(document.querySelector('[data-testid=start]')?.textContent || ''), null, { timeout: 30000 });
    await p2.waitForTimeout(3000);
    const s = await editorState(p2);
    if (s.checked !== 'true' || s.ariaDisabled) throw new Error(`the switch after signing in: checked ${s.checked}, aria-disabled ${s.ariaDisabled}`);
    if (s.analysis.includes('reco-listening')) throw new Error('it listened again');
    const lines = await detailLines(p2);
    if (lines.join('|') !== left.join('|')) throw new Error(`choices: ${left.join(' | ')} -> ${lines.join(' | ')}`);
    if (!lines.some((l) => /^Noise removal · /.test(l) && !/free/.test(l))) throw new Error('the paid choices were lost: ' + lines.join(' | '));
    if (await savedAuto(p2) === 'off') throw new Error('the saved choice became off');
    return `on before, on after; ${lines.join(' | ')}; (${asked} model files needed signed out)`;
  } finally {
    await w.ctx.close();
  }
});

await check('a kept copy older than a day is not used and is deleted: when the page loads (the note says so), and when only the record is old', async () => {
  for (const [what, opts] of [['the note is old', { record: true, note: true }], ['the record is old', { record: true, note: false }]]) {
    const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const p = await ctx.newPage();
    try {
      await planAndLeave(p);
      await p.getByTestId('start').click();
      await p.waitForURL(/\/login$/, { timeout: 15000 });
      if (!(await kept(p))) throw new Error(`${what}: nothing was kept`);
      await age(p, 25, opts);
      await p.goto(`${BASE}/app`);
      await p.getByTestId('free-no-account').waitFor({ timeout: 15000 });
      await p.waitForTimeout(500);
      if (await p.getByTestId('file-name').count()) throw new Error(`${what}: the old copy was brought back`);
      if (await kept(p)) throw new Error(`${what}: the old copy is still kept`);
    } finally {
      await ctx.close();
    }
  }
  return 'both gone';
});

await check('signed out: the Live page opens, API keys ask to sign in, the free test starts and runs with nothing sent to us, and a paid session says "Sign in to start" and goes to sign-in', async () => {
  const wav = join(FILES, 'speech.wav');
  const f = await launch(['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-audio-capture=${wav}`]);
  const seen = [];
  let sockets = 0;
  f.page.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.startsWith('/api/')) seen.push(`${r.method()} ${u.pathname}`); });
  f.page.on('websocket', () => { sockets++; });
  try {
    await f.page.goto(`${BASE}/app/live`);
    await f.page.getByTestId('live-free-banner').waitFor({ timeout: 15000 });
    if (new URL(f.page.url()).pathname !== '/app/live') throw new Error(f.page.url());
    await f.page.getByTestId('keys-signin').waitFor();
    if ((await f.page.getByTestId('live-start').innerText()).trim() !== 'Start test') throw new Error('free button: ' + await f.page.getByTestId('live-start').innerText());
    await f.page.getByTestId('live-start').click();
    await f.page.getByTestId('live-stop').waitFor({ timeout: 120000 });
    await f.page.waitForFunction(() => Number(document.querySelector('[data-testid=live-stats]')?.getAttribute('data-recv')) > 100, null, { timeout: 30000 });
    await f.page.getByTestId('live-stop').click();
    await f.page.getByTestId('live-start').waitFor({ timeout: 15000 });
    if (await f.page.getByTestId('live-error').count()) throw new Error(await f.page.getByTestId('live-error').innerText());
    // (the OBS plugin's download list is a public page of files, the same for everyone)
    const odd = seen.filter((c) => !/^GET \/api\/(me|billing\/pricing|downloads\/obs-plugin\/manifest\.json)$/.test(c));
    if (odd.length || sockets) throw new Error(`sent to us: ${odd.join(', ')}; sockets ${sockets}`);
    // paid: priced from the public list, started only when signed in
    await f.page.getByTestId('live-mode').locator('label', { hasText: 'Paid, done on our side' }).click();
    await f.page.getByTestId('live-free-hint').waitFor();
    const price = await f.page.getByTestId('live-price').innerText();
    if (!/Cost: \$/.test(price)) throw new Error('price: ' + price);
    if ((await f.page.getByTestId('live-start').innerText()).trim() !== 'Sign in to start') throw new Error('paid button: ' + await f.page.getByTestId('live-start').innerText());
    await f.page.getByTestId('live-start').click();
    await f.page.waitForURL(/\/login$/, { timeout: 15000 });
    return `free test ran; paid: ${price.replace(/\s+/g, ' ').slice(0, 60)}`;
  } finally {
    await f.browser.close();
  }
});

await check('the pages that belong to an account still ask for sign-in (billing, account, a job on our side); a free run made in the tab has a page without one', async () => {
  const ctx = await browser.newContext();
  const p = await ctx.newPage();
  try {
    for (const path of ['/app/billing', '/app/account', '/app/job/0123456789abcdef0123456789abcdef']) {
      await p.goto(`${BASE}${path}`);
      await p.waitForURL(/\/login$/, { timeout: 15000 });
    }
    await p.goto(`${BASE}/app/job/local-abc`);
    await p.getByTestId('job-gone').waitFor({ timeout: 15000 });
    if (new URL(p.url()).pathname !== '/app/job/local-abc') throw new Error(p.url());
    return 'login for three, none for the free run';
  } finally {
    await ctx.close();
  }
});

await check('signed in nothing changes: the Cleanup page shows the jobs list and the balance (no line about the free tools without an account), the same paid plan shows "Start · price" from the price asked of our side', async () => {
  const p = await context.newPage();
  const seen = [];
  p.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.startsWith('/api/')) seen.push(`${r.method()} ${u.pathname}`); });
  try {
    await signIn(p, EMAIL);
    await p.goto(`${BASE}/app`);
    await p.getByTestId('jobs-note').waitFor({ timeout: 15000 });
    await p.getByTestId('available').waitFor();
    if (await p.getByTestId('free-no-account').count()) throw new Error('the signed-out line is shown');
    if (!seen.includes('GET /api/jobs')) throw new Error('no jobs list: ' + seen.join(', '));
    await dropFile(p, NOISY);
    if ((await editorState(p)).tip) throw new Error('the signed-out tooltip is on the switch while signed in: ' + (await editorState(p)).tip);
    await pick(p, 'noise', 'denoise');
    await p.waitForFunction(() => /^Start · \$/.test(document.querySelector('[data-testid=start]')?.textContent || ''), null, { timeout: 30000 });
    if (!seen.includes('POST /api/jobs/estimate')) throw new Error('no estimate: ' + seen.join(', '));
    await p.goto(`${BASE}/app/live`);
    await p.getByTestId('live-free-banner').waitFor();
    if (await p.getByTestId('keys-signin').count()) throw new Error('keys ask to sign in while signed in');
    return 'jobs list, balance, estimate and API keys as before';
  } finally {
    await p.close();
  }
});

await check('a file chosen before the account check has been answered is still chosen when the answer arrives (the page does not start over)', async () => {
  const p = await context.newPage(); // (signed in by now)
  try {
    await p.route('**/api/me', async (r) => { await new Promise((done) => setTimeout(done, 2500)); await r.continue(); });
    await p.goto(`${BASE}/app`);
    await p.setInputFiles('input[type=file]', NOISY);
    await p.getByTestId('file-name').waitFor({ timeout: 20000 });
    if ((await p.locator('.site-header .avatar').count()) !== 0) throw new Error('the answer had come already: the check proves nothing');
    await p.locator('.site-header .avatar').waitFor({ timeout: 15000 }); // the answer: signed in
    await p.waitForTimeout(700);
    if (!(await p.getByTestId('file-name').count()) || !(await p.getByTestId('start').count())) throw new Error('the page started over when the account check was answered');
    return 'the editor stayed';
  } finally {
    await p.close();
  }
});

console.log('LOGS:\n' + logs.filter((l) => !/Failed to load resource/.test(l)).join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
