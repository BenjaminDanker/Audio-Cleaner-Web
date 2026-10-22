// The free tools without an account: /app (the Cleanup page and the editor) and the Live page open signed out, the Free noise removal and
// Free subtitles run to the end with nothing about the file sent (no job, no upload, no estimate), a plan that needs our side says
// "Sign in to start" with its price worked out from the public price list, signing in brings the file and the editor's state back (and
// the kept copy is deleted, or is not used after a day), the Live page's free test starts signed out and a paid session asks to sign in,
// the pages that belong to an account still ask for sign-in, and signed in nothing changes. Automatic works without an account as for
// everybody (in the browser; nothing about the file is sent) but picks only among the free options: Free noise removal at the strength it
// finds (the strongest Free one where it would rebuild the voice or use the heavy remover), Free subtitles where there is speech, no Leveling,
// and the suggestions for moments offer Free strengths; with Automatic off every free option is on; the switch shows and saves this
// browser's choice; switching it on when it started off sets the choices from what the quick listen finds; another file type or video size
// with nothing else chosen is free and runs in the browser, and a video type this browser can't make is offered as made on our side (it needs
// an account); and the kept work comes back after signing in with its tracks and its Automatic state as they were left.
// Needs the local backend (port 8792, see helpers.mjs) with `ACW_MODEL_HOST=http://127.0.0.1:8798/ npm run build` output and the fixtures.
// Usage: node tests/signedout.mjs
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { noisy, wavBytes } from './reco-audio.mjs';
import {
  BASE, FILES, check, dragRange, dropFile, launch, loginLink, mainButton, mirror, pick, resultAttr, selectionTitle, setLeveling, signIn, summary, waitResult,
} from './helpers.mjs';

const EMAIL = `signedout-${Date.now()}@example.com`;
const FFMPEG = process.env.FFMPEG || join(homedir(), 'audio-cleaner', 'bin', 'ffmpeg');
const FFPROBE = process.env.FFPROBE || join(homedir(), 'audio-cleaner', 'bin', 'ffprobe');
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

await check('signed out, Automatic started off: switching it on later sets the choices from what the quick listen finds, so they change from the starting ones (Strong for a loud recording, no subtitles where no one speaks), and are still free options only', async () => {
  const got = [];
  for (const [name, want, from] of [['loud', { str: '3', subs: '1' }, { str: '2', subs: '1' }], ['hum', { str: '2', subs: '0' }, { str: '2', subs: '1' }]]) {
    const w = await watched('off'); // (a fresh browser that saved Automatic off: it starts with every free option on)
    try {
      await openSignedOut(w.p, GENERATED[name], { automatic: false });
      await freeOnly(w.p, from, `${name} before`);
      const before = await editorState(w.p);
      if (before.checked !== 'false' || before.analysis.length || w.seen.models.length) throw new Error(`${name}: it listened before the switch: ${JSON.stringify(before)} ${w.seen.models.join(', ')}`);
      await w.p.locator('.studio button[data-act=auto]').click();
      await w.p.getByTestId('reco-status').waitFor({ timeout: 90000 });
      const after = await editorState(w.p);
      if (after.checked !== 'true') throw new Error(`${name}: the switch is ${after.checked}`);
      if (!w.seen.models.length) throw new Error(`${name}: switched on, but it did not listen`);
      if (after.strength === before.strength && after.subs === before.subs && after.kind === before.kind) throw new Error(`${name}: the choices did not change: ${JSON.stringify(after)}`);
      got.push(`${name}: ${(await freeOnly(w.p, want, `${name} after`)).split('; ')[1]}`); // (what it found, and only free options)
      if (sentToUs(w.seen).length) throw new Error(`${name}: asked of our side: ` + sentToUs(w.seen).join(', '));
    } finally {
      await w.ctx.close();
    }
  }
  return got.join(' || ');
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

const probeOf = (p) => {
  const j = JSON.parse(execFileSync(FFPROBE, ['-v', 'error', '-show_streams', '-show_format', '-of', 'json', p], { encoding: 'utf8' }));
  return { v: j.streams.find((x) => x.codec_type === 'video'), a: j.streams.find((x) => x.codec_type === 'audio'), dur: Number(j.format.duration) };
};
const BIG = join(FILES, 'big4k.avi'); // 10 s of a plain 4K picture at 30 frames a second (MPEG-4 in an AVI, read through libav) with the sound of noisy.wav: a few MB
if (!existsSync(BIG)) execFileSync(FFMPEG, ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=0x336699:size=3840x2160:rate=30:duration=10', '-i', NOISY, '-t', '10', '-map', '0:v', '-map', '1:a',
  '-c:v', 'mpeg4', '-b:v', '1M', '-g', '300', '-threads', '4', '-c:a', 'ac3', '-b:a', '192k', BIG]);
const NOTHING_TO_DO = 'Choose at least one thing to do: noise removal, subtitles, or another file type or video size.';
const outField = (p, id) => p.locator(`[data-testid=output-pop] [data-testid=${id}]`);
/** The Output menu open (and left open): the file type and size lists are ready. */
async function openOutput(p) {
  if (!(await p.getByTestId('output-pop').count())) await p.getByTestId('output-button').click();
  await outField(p, 'out-format').waitFor();
  await p.waitForFunction(() => { const s = document.querySelector('[data-testid=output-pop] [data-testid=out-format]'); return s && !s.disabled; }, null, { timeout: 15000 });
}
const closeOutput = async (p) => { if (await p.getByTestId('output-pop').count()) await p.getByTestId('output-button').click(); };

await check('signed out: another file type or video size with nothing else chosen is free: with the file\'s own type and size Start says what to choose, with another type it is "Start · Free", the file comes down in the new type with its own sound, the card is "File conversion", and nothing about the file is sent', async () => {
  calls.length = 0; downloads.length = 0;
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'test.mp4'));
  await pick(page, 'noise', 'off');
  await pick(page, 'subs', 'off');
  await page.locator('[data-testid=notes] .banner-warn', { hasText: NOTHING_TO_DO }).waitFor({ timeout: 5000 });
  if (await page.getByTestId('start').isEnabled()) throw new Error('Start is on with nothing chosen');
  await openOutput(page);
  await outField(page, 'out-format').selectOption('mkv');
  await outField(page, 'out-name').fill('converted copy.mkv');
  await closeOutput(page);
  await page.waitForFunction(() => !document.querySelector('[data-testid=notes] .banner-warn') && document.querySelector('[data-testid=start]')?.disabled === false, null, { timeout: 5000 });
  const label = (await mainButton(page)).trim();
  if (label !== 'Start · Free') throw new Error('button: ' + label);
  await page.getByTestId('start').click();
  await waitResult(page, 'result', 120000);
  if (!/\/app\/job\/local-/.test(new URL(page.url()).pathname)) throw new Error('page: ' + page.url());
  if (downloads.length !== 1 || downloads[0].suggestedFilename() !== 'converted copy.mkv') throw new Error('downloads: ' + downloads.map((d) => d.suggestedFilename()));
  const out = join(GEN, 'converted-copy.mkv');
  await downloads[0].saveAs(out);
  const got = probeOf(out);
  const src = probeOf(join(FILES, 'test.mp4'));
  if (got.v?.codec_name !== src.v.codec_name || `${got.v.width}x${got.v.height}` !== `${src.v.width}x${src.v.height}`) throw new Error(`picture ${src.v.codec_name} ${src.v.width}x${src.v.height} -> ${got.v?.codec_name} ${got.v?.width}x${got.v?.height}`);
  if (got.a?.codec_name !== src.a.codec_name || Math.abs(got.dur - src.dur) > 0.8) throw new Error(`sound ${src.a.codec_name} ${src.dur} -> ${got.a?.codec_name} ${got.dur}`);
  onlyPublic('during the conversion');
  // its card: made in this tab, named File conversion (the Cleanup page lists what was made here)
  await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Cleanup', exact: true }).click();
  await page.waitForURL(/\/app$/);
  const row = (await page.locator('[data-testid=job-row]').first().innerText()).replace(/\s+/g, ' ');
  if (!/File conversion/.test(row) || !/test\.mp4/.test(row)) throw new Error('card: ' + row);
  return `${got.v.codec_name}+${got.a.codec_name} in MKV, ${got.dur.toFixed(1)} s; card: ${row.slice(0, 80)}`;
});

await check('signed out: a new picture size alone converts in the browser too (a 720p video at 480p, the shape kept), and a sound file in another type is the same free conversion', async () => {
  const enc = await page.evaluate(async () => {
    const ok = async (c) => { try { return !!(await VideoEncoder.isConfigSupported(c)).supported; } catch { return false; } };
    return (await ok({ codec: 'avc1.42001f', width: 640, height: 360, bitrate: 1e6 })) || (await ok({ codec: 'vp09.00.10.08', width: 640, height: 360, bitrate: 1e6 }));
  }).catch(() => false);
  const out = [];
  if (enc) {
    const big = join(FILES, 'conv720.mp4');
    if (!existsSync(big)) execFileSync(FFMPEG, ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=1280x720:rate=25', '-i', NOISY, '-t', '4', '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', big]);
    calls.length = 0; downloads.length = 0;
    await page.goto(`${BASE}/app`);
    await dropFile(page, big);
    await pick(page, 'noise', 'off');
    await pick(page, 'subs', 'off');
    await openOutput(page);
    await outField(page, 'out-size').selectOption('480');
    await closeOutput(page);
    await page.waitForFunction(() => document.querySelector('[data-testid=start]')?.disabled === false, null, { timeout: 5000 });
    if ((await mainButton(page)).trim() !== 'Start · Free') throw new Error('button: ' + (await mainButton(page)));
    await page.getByTestId('start').click();
    await waitResult(page, 'result', 180000);
    if (downloads.length !== 1 || !/\.mp4$/.test(downloads[0].suggestedFilename())) throw new Error('downloads: ' + downloads.map((d) => d.suggestedFilename()));
    const p = join(GEN, 'small.mp4');
    await downloads[0].saveAs(p);
    const got = probeOf(p);
    if (got.v.height !== 480 || Math.abs(got.v.width / got.v.height - 16 / 9) > 0.02 || !got.a) throw new Error(`${got.v.width}x${got.v.height} ${got.a?.codec_name}`);
    onlyPublic('during the resize');
    out.push(`${got.v.width}x${got.v.height}`);
  } else out.push('size skipped (this Chromium can write neither H.264 nor VP9)');
  calls.length = 0; downloads.length = 0;
  await page.goto(`${BASE}/app`);
  await dropFile(page, NOISY);
  await pick(page, 'noise', 'off');
  await pick(page, 'subs', 'off');
  await page.locator('[data-testid=notes] .banner-warn', { hasText: NOTHING_TO_DO }).waitFor({ timeout: 5000 });
  await openOutput(page);
  await outField(page, 'out-format').selectOption('flac');
  await closeOutput(page);
  await page.waitForFunction(() => document.querySelector('[data-testid=start]')?.disabled === false, null, { timeout: 5000 });
  if ((await mainButton(page)).trim() !== 'Start · Free') throw new Error('button: ' + (await mainButton(page)));
  await page.getByTestId('start').click();
  await waitResult(page, 'result', 120000);
  if (downloads.length !== 1 || !/\.flac$/.test(downloads[0].suggestedFilename())) throw new Error('downloads: ' + downloads.map((d) => d.suggestedFilename()));
  const f = join(GEN, 'copy.flac');
  await downloads[0].saveAs(f);
  const s = probeOf(f);
  if (s.a?.codec_name !== 'flac' || Math.abs(s.dur - 10.6) > 0.8) throw new Error(`flac: ${s.a?.codec_name} ${s.dur}`);
  onlyPublic('during the sound conversion');
  out.push(`${downloads[0].suggestedFilename()} ${s.a.codec_name}`);
  return out.join('; ');
});

await check('signed out: a video type this browser can\'t make for a video (an AVI is read through libav) is offered as "made on our side", and choosing it needs an account: Sign in to start', async () => {
  calls.length = 0;
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'noisy.avi'));
  await pick(page, 'noise', 'off');
  await pick(page, 'subs', 'off');
  await openOutput(page);
  const groups = await outField(page, 'out-format').evaluate((sel) => [...sel.querySelectorAll('optgroup')].map((g) => ({ label: g.label, values: [...g.querySelectorAll('option')].map((o) => o.value) })));
  const ours = groups.find((g) => g.label === 'Video, made on our side');
  if (!ours || ours.values.join() !== 'mp4,mov,mkv,webm') throw new Error('groups: ' + JSON.stringify(groups));
  if (groups.some((g) => g.label === 'Video')) throw new Error('video types offered as made here: ' + JSON.stringify(groups));
  await outField(page, 'out-format').selectOption('mkv');
  const note = (await outField(page, 'out-size-note').innerText()).replace(/\s+/g, ' ');
  if (!/^Made on our side, so the whole video is uploaded\./.test(note)) throw new Error('size note: ' + note);
  await closeOutput(page);
  await page.waitForFunction(() => /Sign in to start/.test(document.querySelector('.studio .bb .btn-lg')?.textContent || ''), null, { timeout: 5000 });
  if (await page.getByTestId('start').isDisabled()) throw new Error('the button is disabled');
  if ((await page.locator('[data-testid=notes] .banner-warn', { hasText: NOTHING_TO_DO }).count())) throw new Error('it says nothing is chosen');
  onlyPublic('while choosing a type made on our side');
  return `${ours.label}: ${ours.values.join(', ')}; ${note.slice(0, 60)}…`;
});

/** The Details popup: its lines (the total apart) and the total. */
async function detailsOf(p) {
  await p.getByTestId('bar-det').click();
  const lines = (await p.locator('[data-testid=details] ul.fl li:not(.tot)').allInnerTexts()).map((l) => l.replace(/\s+/g, ' ').trim());
  const total = (await p.getByTestId('details-total').innerText()).trim();
  await p.getByTestId('bar-det').click();
  return { lines, total };
}
const COUNTS = (n) => new RegExp(`^Video conversion · counts ${n}× \\(bigger or faster than 1080p at 30 fps\\) `);

await check('signed out: a video made on our side shows its price in Details worked out here from the public list and what the browser read of it: a 4K picture made again counts 4 times, a new size 2 times, the picture copied into MKV is free and says so; Start asks to sign in and nothing about the file is sent', async () => {
  calls.length = 0;
  await page.goto(`${BASE}/app`);
  await dropFile(page, BIG);
  await pick(page, 'noise', 'off');
  await pick(page, 'subs', 'off');
  await openOutput(page);
  await outField(page, 'out-format').selectOption('webm'); // (the MPEG-4 picture does not go into WebM: made again, 4K at 30 frames a second is 4 units)
  await closeOutput(page);
  await page.waitForFunction(() => /Sign in to start/.test(document.querySelector('.studio .bb .btn-lg')?.textContent || ''), null, { timeout: 5000 });
  let d = await detailsOf(page);
  if (d.lines.length !== 1 || !COUNTS(4).test(d.lines[0]) || !/ \$0\.004$/.test(d.lines[0]) || d.total !== '$0.004') throw new Error('webm: ' + JSON.stringify(d));
  await openOutput(page);
  await outField(page, 'out-size').selectOption('720'); // 4K made 720p: half the original's pixels a second count, 2 units
  await closeOutput(page);
  d = await detailsOf(page);
  if (d.lines.length !== 1 || !COUNTS(2).test(d.lines[0]) || !/ \$0\.002$/.test(d.lines[0]) || d.total !== '$0.002') throw new Error('webm 720p: ' + JSON.stringify(d));
  await openOutput(page);
  await outField(page, 'out-format').selectOption('mkv');
  await outField(page, 'out-size').selectOption('same');
  await closeOutput(page);
  d = await detailsOf(page);
  if (d.lines.join('|') !== 'Video on our side, the picture copied as it is Free' || d.total !== 'Free') throw new Error('mkv: ' + JSON.stringify(d));
  if ((await mainButton(page)).trim() !== 'Sign in to start') throw new Error('button: ' + (await mainButton(page)));
  await openOutput(page);
  await outField(page, 'out-format').selectOption('mp4'); // (MPEG-4 goes into an MP4 as it is: the same free copy)
  await closeOutput(page);
  d = await detailsOf(page);
  if (d.lines.join('|') !== 'Video on our side, the picture copied as it is Free') throw new Error('mp4: ' + JSON.stringify(d));
  onlyPublic('while pricing a conversion on our side');
  return 'webm 4×, 720p 2×, MKV and MP4 copied';
});

await check('signed out: a sound asked for as a video type this browser can\'t make gets a plain black picture made on our side, free: the types are offered under "Video (plain black picture), made on our side", Details says so, Start asks to sign in, and where the browser can make them itself they are its own', async () => {
  const groupsOf = (pg) => outField(pg, 'out-format').evaluate((sel) => [...sel.querySelectorAll('optgroup')].map((g) => ({ label: g.label, values: [...g.querySelectorAll('option')].map((o) => o.value) })));
  // here: the browser makes them (no group of the other kind)
  await page.goto(`${BASE}/app`);
  await dropFile(page, NOISY);
  await openOutput(page);
  const here = await groupsOf(page);
  if (!here.some((g) => g.label === 'Video (plain black picture)' && g.values.join() === 'mp4,mov,mkv,webm') || here.some((g) => /made on our side/.test(g.label))) throw new Error('groups where the browser can make video: ' + JSON.stringify(here));
  // a browser that cannot encode a picture: every video type for a sound is made on our side
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  await ctx.addInitScript(() => { Object.defineProperty(window, 'VideoEncoder', { value: undefined, configurable: true }); });
  const p = await ctx.newPage();
  const asked = [];
  p.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.startsWith('/api/')) asked.push(`${r.method()} ${u.pathname}`); });
  try {
    await p.goto(`${BASE}/app`);
    await dropFile(p, NOISY);
    await pick(p, 'noise', 'off');
    await pick(p, 'subs', 'off');
    await openOutput(p);
    const groups = await groupsOf(p);
    const ours = groups.find((g) => g.label === 'Video (plain black picture), made on our side');
    if (!ours || ours.values.join() !== 'mp4,mov,mkv,webm' || groups.some((g) => g.label === 'Video (plain black picture)')) throw new Error('groups: ' + JSON.stringify(groups));
    await outField(p, 'out-format').selectOption('mkv');
    await closeOutput(p);
    await p.waitForFunction(() => /Sign in to start/.test(document.querySelector('.studio .bb .btn-lg')?.textContent || ''), null, { timeout: 5000 });
    const d = await detailsOf(p);
    if (d.lines.join('|') !== 'Video with a plain black picture, on our side Free' || d.total !== 'Free') throw new Error('details: ' + JSON.stringify(d));
    if (await p.locator('[data-testid=notes] .banner-warn', { hasText: NOTHING_TO_DO }).count()) throw new Error('it says nothing is chosen');
    const odd = asked.filter((c) => !/^GET \/api\/(me|billing\/pricing)$/.test(c));
    if (odd.length) throw new Error('sent: ' + odd.join(', '));
    return `${ours.label}: ${ours.values.join(', ')}; ${d.lines[0]}`;
  } finally {
    await ctx.close();
  }
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
