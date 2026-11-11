// Website fixes that keep their own checks: the pure safety helpers (a return link that must stay on this site, the WAV size
// limit), a page file that cannot be loaded (one retry, then a friendly screen, never a white page), the delete-now
// confirmation for a finished job, and what a finished job hands over after a reload. The rest of the old audit checks are covered by the suites next to this one.
// Also: the sign-in link's confirm step (a link opened in another browser than the one that asked for it asks "Sign in as <address>?") and the
// "Different file" question (asked only when it would throw the person's own changes away).
// Needs the local test backend (port 8792, see helpers.mjs) serving `npm run build`. Usage: node tests/fixes.mjs [section ...]
// sections: pure pages jobs jobpage confirm different
import { join } from 'node:path';
import { BASE, FILES, check, dropFile, launch, pyRun, setLeveling, signIn, summary } from './helpers.mjs';

const only = process.argv.slice(2);
const want = (s) => only.length === 0 || only.includes(s);
const EMAIL = process.env.E2E_EMAIL || `fixes-${Date.now()}@example.com`;
const now = () => Math.floor(Date.now() / 1000);

if (want('pure')) {
  await check('pure helpers: a return link must stay on this site (no other host, no backslash, no line break); a WAV above 4 GB is refused (6 h stereo 24-bit is not a WAV)', async () => {
    const { wavFits } = await import('../src/media/wav.js');
    const { isSafeReturn } = await import('../src/returnTo.js');
    const h = 3600;
    if (!wavFits(6 * h, 2, 16)) throw new Error('6 h stereo 16-bit should fit');
    if (wavFits(6 * h, 2, 24)) throw new Error('6 h stereo 24-bit must not fit (over 4 GB)');
    if (!wavFits(4 * h, 2, 24) || wavFits(4.2 * h, 2, 24)) throw new Error('the 4 h / 4.2 h boundary moved');
    for (const ok of ['/app', '/app/billing', '/terms?x=1']) if (!isSafeReturn(ok)) throw new Error(`${ok} should be accepted`);
    for (const bad of ['//evil.example', '/\\evil.example', '/app\\..\\x', '/login', '/login/verify', 'https://evil.example', '', null, '/app\n/x']) if (isSafeReturn(bad)) throw new Error(`${JSON.stringify(bad)} should be refused`);
    return 'ok';
  });
}

if (want('pages')) {
  const { browser, page } = await launch();
  await check('a page file that cannot be loaded: one retry, one reload, then a friendly screen with Reload (never a white page); Reload shows the page again', async () => {
    let served = 0;
    await page.route('**/assets/Terms-*.js', (r) => { served += 1; return r.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>x</title>' }); });
    let loads = 0;
    page.on('load', () => { loads += 1; });
    await page.goto(`${BASE}/`);
    await page.waitForSelector('main h1');
    await page.locator('footer').getByRole('link', { name: 'Terms of Service' }).click();
    await page.getByTestId('error-boundary').waitFor({ timeout: 20000 });
    const text = await page.getByTestId('error-boundary').innerText();
    if (!/Something went wrong/.test(text) || !/Reload/.test(text)) throw new Error(text);
    if (loads !== 2) throw new Error(`expected the first load and exactly one reload, saw ${loads} loads`);
    if ((await page.evaluate(() => sessionStorage.getItem('acw-chunk-reload'))) !== '1') throw new Error('reload guard not set');
    await page.unroute('**/assets/Terms-*.js');
    await page.getByRole('button', { name: 'Reload' }).click();
    await page.getByRole('heading', { name: 'Terms of Service' }).waitFor({ timeout: 15000 });
    if ((await page.evaluate(() => sessionStorage.getItem('acw-chunk-reload'))) !== null) throw new Error('guard not cleared after a good load');
    return `file requested ${served}x, ${loads} loads, then Terms shown`;
  });
  await browser.close();
}

if (want('jobs')) {
  const first = await launch();
  await signIn(first.page, EMAIL);
  const state = await first.context.storageState();
  await first.browser.close();
  const job = (id, extra) => ({
    id, kind: 'denoise', free: false, status: 'done', inputName: `${id}.wav`, createdAt: now(), durationSeconds: 60, progress: 0, stage: '',
    outputs: [], costMicros: 0, expiresAt: null, queuePosition: null, queueWaitSeconds: null, ...extra,
  });
  const { browser, page } = await launch([], { storageState: state });
  await check('"Delete now" on a finished job asks first; "Keep them" backs out; "Yes, delete" deletes it', async () => {
    const done = job('d1', { outputs: [{ name: 'noisy-cleaned.wav', kind: 'audio', bytes: 1000 }], expiresAt: now() + 3600, costMicros: 2600 });
    let deletes = 0;
    await page.route(/\/api\/jobs(\/d1)?$/, (r) => {
      const m = r.request().method();
      if (m === 'DELETE') { deletes += 1; return r.fulfill({ json: { ok: true } }); }
      if (m === 'GET') return r.fulfill({ json: { jobs: [done] } });
      return r.continue();
    });
    await page.goto(`${BASE}/app`);
    await page.getByTestId('bar-balance').waitFor();
    // the list on /app holds a short row with no Delete button: the job's own page has it
    await page.getByTestId('job-row').waitFor();
    if (await page.getByRole('button', { name: 'Delete now' }).count()) throw new Error('a Delete button on the list');
    await page.getByTestId('job-open').click();
    await page.waitForURL(/\/app\/job\/d1$/);
    await page.getByRole('button', { name: 'Delete now' }).click();
    await page.getByTestId('delete-confirm').waitFor();
    if (deletes) throw new Error('deleted without asking');
    await page.getByRole('button', { name: 'Keep them' }).click();
    if (!(await page.getByRole('button', { name: 'Delete now' }).count())) throw new Error('did not go back');
    await page.getByRole('button', { name: 'Delete now' }).click();
    await page.getByRole('button', { name: 'Yes, delete' }).click();
    for (let i = 0; i < 20 && deletes === 0; i++) await page.waitForTimeout(100);
    if (deletes !== 1) throw new Error('deletes: ' + deletes);
    await page.waitForURL(/\/app$/, { timeout: 10000 }); // after the delete the visitor is back on the Cleanup page
    await page.unroute(/\/api\/jobs(\/d1)?$/);
    return 'asked first, kept on "Keep them", deleted on "Yes, delete", back on /app';
  });
  // A job the server lists after a reload (this browser has no file or sound of it): the card lists what it made, with one
  // Download again, which brings exactly what was asked for; Change and save again can still change the subtitle formats.
  const stubFiles = {
    'noisy-cleaned.flac': 'fLaC-not-really-audio', 'noisy.srt': '1\n00:00:00,000 --> 00:00:01,000\nHello there.\n', 'noisy.vtt': 'WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello there.\n',
    'noisy-transcript.txt': 'Hello there.\n',
  };
  const stubJob = (id, names) => job(id, {
    kind: 'denoise_transcribe', inputName: 'noisy.flac', displayName: 'noisy.wav', expiresAt: now() + 3600, costMicros: 2600,
    outputs: names.map((n) => ({ name: n, kind: n.endsWith('.flac') ? 'audio' : n.endsWith('.txt') ? 'transcript' : 'captions', bytes: stubFiles[n].length, lang: n.endsWith('.flac') ? undefined : 'en' })),
  });
  const downloads = [];
  page.on('download', (d) => downloads.push(d));
  const withJobs = async (list, id) => {
    await page.route(/\/api\/jobs$/, (r) => (r.request().method() === 'GET' ? r.fulfill({ json: { jobs: list } }) : r.continue()));
    await page.route(`**/api/jobs/${id}/outputs/**`, (r) => {
      const name = decodeURIComponent(new URL(r.request().url()).pathname.split('/').pop());
      return r.fulfill({ body: stubFiles[name] ?? '', contentType: 'application/octet-stream' });
    });
  };
  const without = async (id) => { await page.unroute(/\/api\/jobs$/); await page.unroute(`**/api/jobs/${id}/outputs/**`); };
  const sorted = () => downloads.map((d) => d.suggestedFilename()).sort();
  const waitFor = async (n) => { for (let i = 0; i < 60 && downloads.length < n; i++) await page.waitForTimeout(250); await page.waitForTimeout(500); };

  await check('a finished job after a reload: its files are listed with their sizes, one Download again brings all of them, and no other download buttons', async () => {
    const j = stubJob('d2', ['noisy-cleaned.flac', 'noisy.srt', 'noisy.vtt', 'noisy-transcript.txt']);
    await withJobs([j], 'd2');
    try {
      await page.goto(`${BASE}/app/job/d2`);
      const card = page.getByTestId('job').first();
      await card.getByTestId('download-again').waitFor();
      const rows = (await card.getByTestId('made-file').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
      if (rows.length !== 4 || !rows.every((t) => /\d+ B$/.test(t))) throw new Error('listed: ' + JSON.stringify(rows));
      if ((await card.getByRole('button', { name: /download/i }).count()) !== 1 || (await card.getByRole('link', { name: /download/i }).count())) throw new Error('want exactly one download button');
      downloads.length = 0;
      await card.getByTestId('download-again').click();
      await waitFor(4);
      const want = ['noisy-cleaned.flac', 'noisy-transcript.txt', 'noisy.srt', 'noisy.vtt'];
      if (sorted().join() !== want.join()) throw new Error('downloaded: ' + sorted());
      return rows.join(' | ');
    } finally {
      await without('d2');
    }
  });

  await check('what was asked for decides what is downloaded: only the TXT and the sound were ticked (the server also kept a VTT for the video), so only those two come down; Save changes adds the SRT', async () => {
    // (this job is one the editor opens: it has audio steps, so the page asks for its timeline, which is stubbed here)
    const j = { ...stubJob('d3', ['noisy-cleaned.flac', 'noisy.srt', 'noisy.vtt', 'noisy-transcript.txt']), transcribeTier: 'local', edit: { available: true, hasTimeline: false, run: null, lastRun: null, suggestions: false } };
    await withJobs([j], 'd3');
    await page.route('**/api/jobs/d3/timeline', (r) => r.fulfill({ json: {
      timeline: { version: 1, seconds: 60, layers: { noise: [{ start: 0, end: 60, strength_db: 100, remover: 'auto' }] } }, rendered: null, seconds: 60, channels: 1,
      secondTrack: false, cached: {}, suggestions: null, run: null, lastRun: null, preview: null,
    } }));
    await page.route('**/api/jobs/d3/peaks/**', (r) => r.fulfill({ status: 404, json: { detail: 'Nothing is cached for this job yet.', code: 'not_cached' } }));
    await page.goto(`${BASE}/app/job/d3`);
    await page.evaluate(() => localStorage.setItem('acw.wants.d3', JSON.stringify({
      sound: true, format: 'flac', name: 'noisy-cleaned.flac', size: 'same', subs: true, caps: { srt: false, vtt: false, txt: true }, embed: false, soundHere: false, subsHere: false,
    })));
    try {
      await page.reload();
      await page.getByTestId('download-again').waitFor();
      await page.getByTestId('bar-files').click();
      const rows = await page.getByTestId('made-file').locator('span').first().innerText().then(async () => (await page.getByTestId('made-file').allInnerTexts()).map((t) => t.split(/\s+/)[0]));
      if (rows.sort().join() !== 'noisy-cleaned.flac,noisy-transcript.txt') throw new Error('listed: ' + rows);
      await page.getByTestId('bar-files').click();
      downloads.length = 0;
      await page.getByTestId('download-again').click();
      await waitFor(2);
      if (sorted().join() !== 'noisy-cleaned.flac,noisy-transcript.txt') throw new Error('downloaded: ' + sorted());
      await page.getByTestId('cap-srt').check();
      downloads.length = 0;
      await page.getByTestId('save-changes').click();
      await waitFor(3);
      if (sorted().join() !== 'noisy-cleaned.flac,noisy-transcript.txt,noisy.srt') throw new Error('after adding the SRT: ' + sorted());
      return 'TXT + sound, then + SRT';
    } finally {
      await without('d3');
      await page.unroute('**/api/jobs/d3/timeline');
      await page.unroute('**/api/jobs/d3/peaks/**');
    }
  });

  await browser.close();
}

if (want('jobpage')) {
  const first = await launch();
  await signIn(first.page, EMAIL);
  const state = await first.context.storageState();
  await first.browser.close();
  const { browser, page } = await launch([], { storageState: state });
  await check('a job page whose lookup says "no such job" (404) says it is gone; any other failed lookup (server error, asked to wait) says it could not be loaded and "Try again" asks again, down to the job', async () => {
    const id = 'e'.repeat(32);
    let answer = 404; // an error status, or 'ok' ('no connection' is not tried here: the whole site then says it is taking a break)
    let lookups = 0;
    await page.route(new RegExp(`/api/jobs/${id}$`), (r) => {
      lookups += 1;
      if (answer === 'ok') {
        return r.fulfill({ json: { id, kind: 'denoise', free: false, status: 'failed', error: 'It did not work.', inputName: 'x.wav', createdAt: now(), durationSeconds: 60, progress: 0, stage: '',
          outputs: [], costMicros: 0, expiresAt: null, queuePosition: null, queueWaitSeconds: null } });
      }
      return r.fulfill({ status: answer, json: { detail: answer === 404 ? 'Not found.' : 'Something broke.' } });
    });
    const text = (t) => page.getByTestId(t).innerText();
    await page.goto(`${BASE}/app/job/${id}`);
    await page.getByTestId('job-gone').waitFor({ timeout: 15000 });
    if (!/isn’t here any more/.test(await text('job-gone')) || (await page.getByTestId('job-load-failed').count())) throw new Error('404: ' + await page.getByTestId('job-page').innerText());
    for (const how of [500, 429]) {
      answer = how;
      await page.goto(`${BASE}/app/job/${id}`);
      await page.getByTestId('job-load-failed').waitFor({ timeout: 15000 });
      if (!/couldn’t load this job/.test(await text('job-load-failed'))) throw new Error(`${how}: ` + await text('job-load-failed'));
      if ((await page.getByTestId('job-gone').count()) || (await page.getByTestId('job-retry').count()) !== 1) throw new Error(`${how}: "gone" shown, or no single Try again`);
    }
    const before = lookups;
    await page.getByTestId('job-retry').click(); // still failing: it asks again and says so again
    for (let i = 0; i < 50 && lookups === before; i++) await page.waitForTimeout(100);
    if (lookups !== before + 1) throw new Error(`lookups ${before} -> ${lookups}`);
    await page.getByTestId('job-load-failed').waitFor();
    answer = 'ok';
    await page.getByTestId('job-retry').click();
    await page.locator(`[data-testid=job][data-job-id="${id}"]`).waitFor({ timeout: 15000 });
    if ((await page.getByTestId('job-load-failed').count()) || (await page.getByTestId('job-gone').count())) throw new Error('the failure notice stayed after the job loaded');
    return `404 -> gone; 500 and 429 -> "couldn’t load"; Try again asked again (${lookups} lookups), then the job`;
  });
  await browser.close();
}

if (want('confirm')) {
  // The test backend sends no mail, so the links are made the way the sign-in mail makes them: a token that knows the browser that asked (the SHA-256
  // of the acw_login cookie's value), written straight into the test backend's database. The answers are the real server's (no page.route).
  const stamp = Date.now();
  const py = (code) => pyRun(code).trim().split('\n').pop();
  const bound = (email, browser) => py(`from acw import auth; print(auth.create_login_token(${JSON.stringify(email)}, 'email_login', 'test', ${JSON.stringify(browser)}))`);
  const used = (token) => py(`from acw import auth, db; print(db.one('SELECT used_at FROM login_tokens WHERE token_hash = ?', (auth.sha256_hex(${JSON.stringify(token)}),))['used_at'])`) !== 'None';
  const expire = (token) => py(`from acw import auth, db; db.execute('UPDATE login_tokens SET expires_at = ? WHERE token_hash = ?', (db.now() - 1, auth.sha256_hex(${JSON.stringify(token)}))); print('ok')`);
  const { browser } = await launch();
  const open = async (cookies = []) => {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    if (cookies.length) await context.addCookies(cookies);
    const page = await context.newPage();
    const bodies = []; // what the page sent to POST /api/auth/email/verify
    page.on('request', (r) => { if (r.method() === 'POST' && /\/api\/auth\/email\/verify$/.test(r.url())) bodies.push(r.postDataJSON()); });
    return { context, page, bodies };
  };
  const cookie = (value) => ({ name: 'acw_login', value, domain: new URL(BASE).hostname, path: '/api/auth/email', httpOnly: true, sameSite: 'Lax' });
  const me = (page) => page.evaluate(() => fetch('/api/me', { credentials: 'same-origin' }).then((r) => r.json()));
  const card = (page) => page.locator('.login-card');
  const heading = (email) => ({ name: `Sign in as ${email}?` });

  await check('sign-in link, another browser: a link that was asked for elsewhere asks "Sign in as <address>?" (a Sign in button and "That’s not my address") and is not used up; "That’s not my address" goes back to sign-in', async () => {
    const email = `ask-${stamp}@example.com`;
    const token = bound(email, 'a'.repeat(24));
    const { context, page, bodies } = await open(); // (no acw_login cookie)
    try {
      await page.goto(`${BASE}/login/verify#token=${token}`);
      await card(page).getByRole('heading', heading(email)).waitFor({ timeout: 15000 });
      if ((await card(page).getByRole('button', { name: 'Sign in', exact: true }).count()) !== 1) throw new Error('no single Sign in button: ' + await card(page).innerText());
      if ((await card(page).getByRole('link', { name: 'That’s not my address' }).count()) !== 1) throw new Error('no "That’s not my address"');
      if (!/different browser or device/.test(await card(page).innerText())) throw new Error('the page does not say why: ' + await card(page).innerText());
      if (await page.getByRole('alert').count()) throw new Error('an error is shown beside the question: ' + await page.getByRole('alert').first().innerText());
      if (new URL(page.url()).hash !== '') throw new Error('the token is still in the address bar');
      if (JSON.stringify(bodies) !== JSON.stringify([{ token }])) throw new Error('first request: ' + JSON.stringify(bodies)); // (no confirm: the page asked first)
      if ((await me(page)).signedIn) throw new Error('signed in without the yes');
      if (used(token)) throw new Error('the link was used up by asking');
      await card(page).getByRole('link', { name: 'That’s not my address' }).click();
      await page.waitForURL(/\/login$/, { timeout: 10000 });
      if ((await me(page)).signedIn || used(token)) throw new Error('"not my address" signed in or used the link');
      return 'asked, nothing used, back to /login';
    } finally {
      await context.close();
    }
  });

  await check('sign-in link, another browser: pressing "Sign in" sends {token, confirm: true} and lands in the app, signed in as that address', async () => {
    const email = `yes-${stamp}@example.com`;
    const token = bound(email, 'b'.repeat(24));
    const { context, page, bodies } = await open([cookie('c'.repeat(24))]); // (a cookie, but not the one the link was asked with)
    try {
      await page.goto(`${BASE}/login/verify#token=${token}`);
      await card(page).getByRole('heading', heading(email)).waitFor({ timeout: 15000 });
      const sent = page.waitForRequest((r) => r.method() === 'POST' && /\/api\/auth\/email\/verify$/.test(r.url()));
      await card(page).getByRole('button', { name: 'Sign in', exact: true }).click();
      const body = (await sent).postDataJSON();
      if (JSON.stringify(body) !== JSON.stringify({ token, confirm: true })) throw new Error('confirm request: ' + JSON.stringify(body));
      await page.waitForURL((u) => !u.pathname.startsWith('/login'), { timeout: 15000 });
      await page.getByTestId('bar-balance').waitFor({ timeout: 15000 });
      const who = await me(page);
      if (!who.signedIn || who.user.email !== email) throw new Error('signed in as ' + JSON.stringify(who.user));
      if (!used(token)) throw new Error('the link is still unused after the yes');
      if (bodies.length !== 2 || 'confirm' in bodies[0]) throw new Error('requests: ' + JSON.stringify(bodies));
      return `${new URL(page.url()).pathname}, 2 requests (ask, then confirm)`;
    } finally {
      await context.close();
    }
  });

  await check('sign-in link, same browser: the browser that holds the acw_login cookie the link was asked with is signed in at once (one request, no question)', async () => {
    const email = `same-${stamp}@example.com`;
    const mine = 'd'.repeat(24);
    const token = bound(email, mine);
    const { context, page, bodies } = await open([cookie(mine)]);
    try {
      await page.goto(`${BASE}/login/verify#token=${token}`);
      await page.waitForURL((u) => !u.pathname.startsWith('/login'), { timeout: 15000 });
      await page.getByTestId('bar-balance').waitFor({ timeout: 15000 });
      if (await page.getByRole('heading', heading(email)).count()) throw new Error('asked in the browser that asked');
      if (JSON.stringify(bodies) !== JSON.stringify([{ token }])) throw new Error('requests: ' + JSON.stringify(bodies));
      if ((await me(page)).user.email !== email) throw new Error('wrong account');
      return 'signed in with one request';
    } finally {
      await context.close();
    }
  });

  await check('sign-in link: when the link ran out while the question was shown, "Sign in" shows the ordinary expired-link message (no loop of questions)', async () => {
    const email = `late-${stamp}@example.com`;
    const token = bound(email, 'e'.repeat(24));
    const { context, page, bodies } = await open();
    try {
      await page.goto(`${BASE}/login/verify#token=${token}`);
      await card(page).getByRole('heading', heading(email)).waitFor({ timeout: 15000 });
      expire(token);
      await card(page).getByRole('button', { name: 'Sign in', exact: true }).click();
      await card(page).getByRole('alert').waitFor({ timeout: 10000 });
      const text = await card(page).innerText();
      if (!/invalid or expired/.test(text) || !/Back to sign in/.test(text) || (await card(page).getByRole('heading', { name: /^Sign in as / }).count())) throw new Error(text);
      if (bodies.length !== 2) throw new Error('requests: ' + JSON.stringify(bodies));
      if ((await me(page)).signedIn) throw new Error('signed in with an expired link');
      return 'expired message after the yes';
    } finally {
      await context.close();
    }
  });

  await check('sign-in link: a "Sign in" that did not get through (503, the link is not the problem) keeps the question with the error beside it, and pressing it again signs in', async () => {
    const email = `retry-${stamp}@example.com`;
    const token = bound(email, 'g'.repeat(24));
    const { context, page, bodies } = await open();
    try {
      await page.goto(`${BASE}/login/verify#token=${token}`);
      await card(page).getByRole('heading', heading(email)).waitFor({ timeout: 15000 });
      let forced = 0;
      await page.route('**/api/auth/email/verify', (r) => (forced++ === 0 ? r.fulfill({ status: 503, json: { detail: 'Try again in a moment.' } }) : r.continue())); // (the first answer to the yes only)
      const signIn = card(page).getByRole('button', { name: 'Sign in', exact: true });
      await signIn.click();
      await card(page).getByRole('alert').waitFor({ timeout: 10000 });
      const text = await card(page).innerText();
      if (!/Try again in a moment\./.test(await card(page).getByRole('alert').innerText())) throw new Error('banner: ' + await card(page).getByRole('alert').innerText());
      if ((await card(page).getByRole('heading', heading(email)).count()) !== 1) throw new Error('the question went: ' + text);
      if ((await signIn.count()) !== 1 || !(await signIn.isEnabled())) throw new Error('no enabled Sign in button: ' + text);
      if ((await card(page).getByRole('link', { name: 'That’s not my address' }).count()) !== 1) throw new Error('"That’s not my address" went: ' + text);
      if ((await card(page).getByText('Back to sign in').count()) || (await page.getByRole('link', { name: 'Back to sign in' }).count())) throw new Error('"Back to sign in" is shown: ' + text);
      if (!/\/login\/verify$/.test(page.url())) throw new Error('left the page: ' + page.url());
      if ((await me(page)).signedIn || used(token)) throw new Error('signed in, or the link used, by a 503');
      // press again: the real server answers now
      await signIn.click();
      await page.waitForURL((u) => !u.pathname.startsWith('/login'), { timeout: 15000 });
      await page.getByTestId('bar-balance').waitFor({ timeout: 15000 });
      const who = await me(page);
      if (!who.signedIn || who.user.email !== email) throw new Error('signed in as ' + JSON.stringify(who.user));
      if (!used(token)) throw new Error('the link is still unused after the second yes');
      if (forced !== 2 || bodies.length !== 3 || 'confirm' in bodies[0] || !bodies[1].confirm || !bodies[2].confirm) throw new Error('requests: ' + JSON.stringify(bodies));
      return 'question, banner and Sign in stayed after the 503; the second press signed in (3 requests: ask, failed yes, yes)';
    } finally {
      await context.close();
    }
  });
  await browser.close();
}

if (want('different')) {
  const first = await launch();
  await signIn(first.page, EMAIL);
  const state = await first.context.storageState();
  await first.browser.close();
  const { browser, page } = await launch([], { storageState: state });
  const noisy = join(FILES, 'noisy.wav');
  const chooser = async () => {
    await page.locator('input[type=file]').waitFor({ state: 'attached', timeout: 10000 });
    if (await page.getByTestId('start').count()) throw new Error('a file is still chosen');
  };
  const askText = async () => (await page.getByTestId('different-file-ask').innerText()).replace(/\s+/g, ' ');

  await check('"Different file" on a file nobody has changed goes straight to the file chooser; after one edit it asks (Yes, different file / Keep this one): Keep keeps the file and the edit, Yes goes to the chooser', async () => {
    await page.goto(`${BASE}/app`);
    await page.getByTestId('bar-balance').waitFor();
    await dropFile(page, noisy, { recommended: true }); // (the recommendation is not an edit of the person's)
    await page.getByTestId('different-file').click();
    await chooser();
    if (await page.getByTestId('different-file-ask').count()) throw new Error('asked although nothing was changed');
    // the same file again, and one edit of the person's
    await dropFile(page, noisy, { recommended: true });
    await setLeveling(page, 'youtube');
    const level = page.locator('.studio button[data-act=l-lvl][data-v="2"]');
    if ((await level.getAttribute('aria-pressed')) !== 'true') throw new Error('the edit did not take');
    await page.getByTestId('different-file').click();
    await page.getByTestId('different-file-ask').waitFor({ timeout: 5000 });
    const ask = await askText();
    if (!/Choose a different file\? What you set up for this one will be lost\./.test(ask) || !/Yes, different file/.test(ask) || !/Keep this one/.test(ask)) throw new Error('ask: ' + ask);
    if (await page.getByTestId('different-file').count()) throw new Error('the plain link is still there beside the question');
    if ((await page.getByTestId('different-file-yes').innerText()).trim() !== 'Yes, different file' || (await page.getByTestId('different-file-no').innerText()).trim() !== 'Keep this one') throw new Error('button words');
    if (!(await page.getByTestId('start').count()) || !/noisy\.wav/.test(await page.getByTestId('file-name').innerText())) throw new Error('the file went while the question was open');
    // Keep this one: the file and the edit stay, the link is back
    await page.getByTestId('different-file-no').click();
    if (await page.getByTestId('different-file-ask').count() || (await page.getByTestId('different-file').count()) !== 1) throw new Error('the question stayed, or the link did not come back');
    if (!/noisy\.wav/.test(await page.getByTestId('file-name').innerText()) || !(await page.getByTestId('start').count())) throw new Error('the file is gone after Keep this one');
    if ((await level.getAttribute('aria-pressed')) !== 'true') throw new Error('the edit is gone after Keep this one');
    // Yes, different file: the chooser
    await page.getByTestId('different-file').click();
    await page.getByTestId('different-file-ask').waitFor({ timeout: 5000 });
    await page.getByTestId('different-file-yes').click();
    await chooser();
    if (await page.getByTestId('different-file-ask').count()) throw new Error('the question is still shown on the chooser');
    // the next file starts clean: asked about nothing
    await dropFile(page, noisy, { recommended: true });
    await page.getByTestId('different-file').click();
    await chooser();
    return 'straight / ask / keep / yes / straight again';
  });

  // A finished job (stubbed here: this is about the page, not the server): nothing is lost by leaving it unless its tracks were changed and not
  // saved; then the words are the job's own.
  const now = () => Math.floor(Date.now() / 1000);
  const done = {
    id: 'd4', kind: 'denoise', free: false, status: 'done', inputName: 'noisy.wav', displayName: 'noisy.wav', createdAt: now(), durationSeconds: 60, progress: 0, stage: '',
    outputs: [{ name: 'noisy-cleaned.flac', kind: 'audio', bytes: 1000 }], costMicros: 2600, expiresAt: now() + 3600, queuePosition: null, queueWaitSeconds: null,
    edit: { available: true, hasTimeline: false, run: null, lastRun: null, suggestions: false },
  };
  await check('"Different file" on a finished job: straight on while its changes are saved; with unsaved changes it asks in the job\'s own words, and "Keep this one" keeps them', async () => {
    await page.route(/\/api\/jobs(\/d4)?$/, (r) => (r.request().method() === 'GET' ? r.fulfill({ json: r.request().url().endsWith('/d4') ? done : { jobs: [done] } }) : r.continue()));
    await page.route('**/api/jobs/d4/timeline', (r) => r.fulfill({ json: {
      timeline: { version: 1, seconds: 60, layers: { noise: [{ start: 0, end: 60, strength_db: 100, remover: 'auto' }] } }, rendered: null, seconds: 60, channels: 1,
      secondTrack: false, cached: {}, suggestions: null, run: null, lastRun: null, preview: null,
    } }));
    await page.route('**/api/jobs/d4/peaks/**', (r) => r.fulfill({ status: 404, json: { detail: 'Nothing is cached for this job yet.', code: 'not_cached' } }));
    try {
      await page.goto(`${BASE}/app/job/d4`);
      await page.getByTestId('download-again').waitFor({ timeout: 20000 });
      await page.getByTestId('different-file').click();
      await page.waitForURL(/\/app$/, { timeout: 10000 });
      if (await page.getByTestId('different-file-ask').count()) throw new Error('asked about a job with nothing changed');
      await page.goto(`${BASE}/app/job/d4`);
      await page.getByTestId('download-again').waitFor({ timeout: 20000 });
      await setLeveling(page, 'youtube');
      await page.locator('[data-testid=studio][data-changed=tracks]').waitFor({ timeout: 5000 });
      await page.getByTestId('different-file').click();
      await page.getByTestId('different-file-ask').waitFor({ timeout: 5000 });
      const ask = await askText();
      if (!/Choose a different file\? The changes you haven.t saved will be lost\./.test(ask)) throw new Error('ask: ' + ask);
      await page.getByTestId('different-file-no').click();
      if (!/\/app\/job\/d4$/.test(page.url()) || await page.getByTestId('different-file-ask').count()) throw new Error('left the job, or the question stayed, after Keep this one');
      await page.locator('[data-testid=studio][data-changed=tracks]').waitFor({ timeout: 5000 }); // (the change is still there)
      await page.getByTestId('different-file').click();
      await page.getByTestId('different-file-yes').click();
      await page.waitForURL(/\/app$/, { timeout: 10000 });
      return 'straight with nothing changed; asked with a change; Keep stays; Yes leaves';
    } finally {
      await page.unroute(/\/api\/jobs(\/d4)?$/);
      await page.unroute('**/api/jobs/d4/timeline');
      await page.unroute('**/api/jobs/d4/peaks/**');
    }
  });
  await browser.close();
}

process.exit(summary().some((r) => !r.ok) ? 1 : 0);
