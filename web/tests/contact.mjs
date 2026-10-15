// The contact form (/contact): the footer link and the pointers on the legal pages only when the server can send mail,
// the form itself, its success and error states (including a Retry-After wait), the hidden trap field, a signed-in
// visitor, and the failed-job link. The test backend has no mail account, so a "mail is set up" server is played by
// answering /api/me and /api/contact in the browser; one check also runs against the real backend, where the form must be off.
// Needs the local test backend (port 8792, see helpers.mjs) serving `npm run build`.
// Usage: node tests/contact.mjs
import { BASE, check, launch, signIn, summary } from './helpers.mjs';

const ME_OUT = { signedIn: false, contactForm: true };
const GOOD = { name: 'Sam', email: 'sam@example.com', message: 'How long do you keep my results for?' };

/** Pretend the server can send mail: /api/me says so (signed out), and /api/contact is answered by `handler`. */
async function mailOn(page, handler = () => ({ status: 200, json: { ok: true } })) {
  const posts = [];
  await page.route('**/api/me', (r) => r.fulfill({ status: 200, json: ME_OUT }));
  await page.route('**/api/contact', async (r) => {
    const body = r.request().postDataJSON();
    posts.push(body);
    await r.fulfill(handler(body, posts.length));
  });
  return posts;
}

async function fill(page, { name = GOOD.name, email = GOOD.email, message = GOOD.message } = {}) {
  if (name !== null) await page.getByLabel('Your name').fill(name);
  await page.getByLabel('Your email address').fill(email);
  await page.getByLabel('Your message').fill(message);
}

const send = (page) => page.getByRole('button', { name: /Send message|Sending/ });

const { browser, page, logs } = await launch();

await check('contact: the real backend has no mail set up, so there is no Contact link and the page says so', async () => {
  await page.goto(`${BASE}/`);
  await page.waitForSelector('footer');
  await page.waitForTimeout(400);
  if (await page.locator('footer a[href$="/contact"]').count()) throw new Error('footer links to /contact while mail is off');
  const r = await page.evaluate(async (url) => (await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email: 'a@example.com', topic: 'other', message: 'Hello, is anyone there?' }) })).status, `${BASE}/api/contact`);
  if (r !== 404) throw new Error(`POST /api/contact answered ${r}, expected 404`);
  await page.goto(`${BASE}/contact`);
  await page.getByRole('heading', { name: 'Contact' }).waitFor();
  const t = await page.locator('main').innerText();
  if (!/isn’t switched on/.test(t) || (await page.locator('form').count())) throw new Error(t);
  return '404 and a calm page';
});

await check('contact: with mail on, the footer has a Contact link and the form opens from it', async () => {
  const posts = await mailOn(page);
  await page.goto(`${BASE}/`);
  await page.locator('footer').getByRole('link', { name: 'Contact' }).click();
  await page.waitForURL(/\/contact$/);
  await page.getByRole('heading', { name: 'Contact' }).waitFor();
  for (const l of ['What is it about?', 'Your name', 'Your email address', 'Your message']) await page.getByLabel(l).first().waitFor();
  const topics = await page.locator('#c-topic option').allInnerTexts();
  if (topics.join('|') !== 'A question|Billing or a refund|A problem with a job|Privacy or my data|Something else') throw new Error(topics.join('|'));
  if (posts.length) throw new Error('posted without being asked');
  return 'ok';
});

await check('contact: Send stays off until there is an address and a real message; a good one is sent as typed and thanked', async () => {
  const posts = await mailOn(page);
  await page.goto(`${BASE}/contact`);
  await page.getByLabel('Your message').waitFor();
  if (!(await send(page).isDisabled())) throw new Error('enabled on an empty form');
  await fill(page, { email: 'not-an-address', message: 'short' });
  if (!(await send(page).isDisabled())) throw new Error('enabled with a bad address and a short message');
  await page.getByLabel('Your email address').fill(GOOD.email);
  if (!(await send(page).isDisabled())) throw new Error('enabled with a message under 10 characters');
  await page.getByLabel('Your message').fill(GOOD.message);
  if (await send(page).isDisabled()) throw new Error('still disabled with a good form');
  await page.getByLabel('What is it about?').selectOption('billing');
  await send(page).click();
  await page.getByTestId('contact-done').waitFor({ timeout: 10000 });
  const t = await page.getByTestId('contact-done').innerText();
  if (!/Thanks, we got your message and will reply by email/.test(t) || !t.includes('sam@example.com')) throw new Error(t);
  if (posts.length !== 1) throw new Error(`${posts.length} posts`);
  const p = posts[0];
  if (p.email !== GOOD.email || p.topic !== 'billing' || p.message !== GOOD.message || p.name !== 'Sam' || p.website !== '' || 'job' in p) throw new Error(JSON.stringify(p));
  await page.getByRole('button', { name: 'Send another message' }).click();
  await page.getByLabel('Your message').waitFor();
  if ((await page.getByLabel('Your message').inputValue()) !== '') throw new Error('the old message is still there');
  return 'ok';
});

await check('contact: the trap field is out of sight and out of reach, and a human leaves it empty', async () => {
  await mailOn(page);
  await page.goto(`${BASE}/contact`);
  await page.getByLabel('Your message').waitFor();
  const trap = page.locator('#c-extra');
  const box = await trap.boundingBox();
  if (box && box.x > -1000 && box.x + box.width > 0) throw new Error(`the trap field is on screen: ${JSON.stringify(box)}`);
  if ((await trap.getAttribute('tabindex')) !== '-1' || (await page.locator('.hp').getAttribute('aria-hidden')) !== 'true') throw new Error('trap field is reachable');
  const seen = [];
  for (let i = 0; i < 8; i += 1) { await page.keyboard.press('Tab'); seen.push(await page.evaluate(() => document.activeElement?.id)); }
  if (seen.includes('c-extra')) throw new Error(`Tab reached the trap field: ${seen}`);
  return 'ok';
});

await check('contact: errors keep what was typed and say what to do (bad request, mail down, daily cap, a wait)', async () => {
  const answers = [
    { status: 400, json: { detail: 'Enter a valid email address so we can reply.' } },
    { status: 503, json: { detail: 'We couldn’t send your message just now. Please try again in a little while.' } },
    { status: 422, json: { detail: 'Invalid request.', fields: ['message'] } },
    { status: 404, json: { detail: 'The contact form isn’t available right now.' } },
    { status: 429, json: { detail: 'We’ve had a lot of messages today. Please try again tomorrow.' }, headers: { 'Retry-After': '40000' } },
    { status: 429, json: { detail: 'Too many requests, try again in 5 minutes.' }, headers: { 'Retry-After': '300' } },
  ];
  const posts = await mailOn(page, (_b, n) => answers[n - 1]);
  await page.goto(`${BASE}/contact`);
  await page.getByLabel('Your message').waitFor();
  await fill(page);
  const expect = [/valid email address/, /couldn’t send your message/, /doesn’t look right/, /isn’t available/, /try again tomorrow/];
  for (let i = 0; i < expect.length; i += 1) {
    if (i === 4) { // the cap: a long wait, so Send is off and the text says how long
      await send(page).click();
      await page.getByTestId('contact-error').waitFor();
      const t = await page.getByTestId('contact-error').innerText();
      if (!expect[i].test(t)) throw new Error(`answer ${i}: ${t}`);
      await page.getByTestId('contact-wait').waitFor();
      if (!(await send(page).isDisabled())) throw new Error('Send is on during a wait');
      if (!/You can send again in about \d+ hours/.test(await page.getByTestId('contact-wait').innerText())) throw new Error(await page.getByTestId('contact-wait').innerText());
      break;
    }
    await send(page).click();
    await page.getByTestId('contact-error').waitFor();
    const t = await page.getByTestId('contact-error').innerText();
    if (!expect[i].test(t)) throw new Error(`answer ${i}: ${t}`);
    if ((await page.getByLabel('Your message').inputValue()) !== GOOD.message || (await page.getByLabel('Your email address').inputValue()) !== GOOD.email) throw new Error('the form lost what was typed');
    if (await send(page).isDisabled()) throw new Error(`Send stayed off after answer ${i}`);
    await page.getByLabel('Your message').fill(`${GOOD.message} ${i}`);
    await page.getByLabel('Your message').fill(GOOD.message);
  }
  if (posts.length !== 5) throw new Error(`${posts.length} posts`);
  // a short wait: the message gives the minutes, Send comes back by itself
  await page.reload();
  await page.getByLabel('Your message').waitFor();
  await fill(page);
  const posts2 = [];
  await page.unroute('**/api/contact');
  await page.route('**/api/contact', async (r) => {
    posts2.push(1);
    await r.fulfill(posts2.length === 1
      ? { status: 429, json: { detail: 'Too many requests, try again in 3 seconds.' }, headers: { 'Retry-After': '3' } }
      : { status: 200, json: { ok: true } });
  });
  await send(page).click();
  await page.getByTestId('contact-wait').waitFor();
  if (!/[2-4] seconds/.test(await page.getByTestId('contact-wait').innerText())) throw new Error(await page.getByTestId('contact-wait').innerText());
  await page.waitForFunction(() => !document.querySelector('[data-testid="contact-wait"]'), null, { timeout: 8000 });
  await send(page).click();
  await page.getByTestId('contact-done').waitFor({ timeout: 8000 });
  return 'ok';
});

await check('contact: the server dropping out while the form is open keeps the form and what was typed', async () => {
  await page.unroute('**/api/contact');
  await page.unroute('**/api/me');
  await mailOn(page);
  await page.goto(`${BASE}/contact`);
  await page.getByLabel('Your message').waitFor();
  await fill(page);
  await page.unroute('**/api/contact');
  await page.route('**/api/contact', (r) => r.abort());
  await send(page).click();
  await page.getByTestId('contact-error').waitFor();
  const t = await page.getByTestId('contact-error').innerText();
  if (!/Could not reach/.test(t)) throw new Error(t);
  if ((await page.getByLabel('Your message').inputValue()) !== GOOD.message) throw new Error('the message was thrown away');
  return 'ok';
});

await check('contact: opened while the server is away, /contact shows the "right back" notice, not a blank page or a wrong "switched off"', async () => {
  await page.unroute('**/api/me').catch(() => {});
  await page.unroute('**/api/contact').catch(() => {});
  await page.route('**/api/me', (r) => r.abort());
  await page.route('**/api/health', (r) => r.abort());
  await page.goto(`${BASE}/contact`);
  await page.getByTestId('maintenance').waitFor({ timeout: 15000 });
  if (/isn’t switched on/.test(await page.locator('main').innerText())) throw new Error('said the form is off');
  await page.unroute('**/api/me');
  await page.unroute('**/api/health');
  return 'ok';
});

await check('contact: Terms section 13, Privacy and the licences page point to the form (and only when it exists)', async () => {
  for (const mail of [true, false]) {
    await page.unroute('**/api/me').catch(() => {});
    await page.route('**/api/me', (r) => r.fulfill({ status: 200, json: { signedIn: false, contactForm: mail } }));
    const found = {};
    for (const path of ['/terms', '/privacy', '/licenses']) {
      await page.goto(`${BASE}${path}`);
      await page.waitForSelector('main h1');
      await page.waitForTimeout(300);
      found[path] = await page.locator('main a[href$="/contact"]').count();
    }
    if (mail && (!found['/terms'] || found['/privacy'] < 2 || !found['/licenses'])) throw new Error(`missing links: ${JSON.stringify(found)}`);
    if (!mail && Object.values(found).some(Boolean)) throw new Error(`links to /contact while mail is off: ${JSON.stringify(found)}`);
    await page.goto(`${BASE}/terms`);
    await page.waitForSelector('#law');
    if (mail && !/Contact: use the contact form/.test(await page.locator('main').innerText())) throw new Error('Terms 13 does not say to use the form');
  }
  // the retention text matches what the server does
  await page.goto(`${BASE}/privacy`);
  const t = await page.locator('#retention ~ ul').first().innerText();
  for (const need of [/7 years/, /12 months/, /90 days/, /30 days/, /60 days/, /contact form/]) if (!need.test(t)) throw new Error(`Privacy retention lacks ${need}`);
  return 'ok';
});

await check('contact: the page reads like the rest of the site (no internals, no avoided words) and fits a phone', async () => {
  await page.unroute('**/api/me').catch(() => {});
  await mailOn(page);
  await page.goto(`${BASE}/contact`);
  await page.getByLabel('Your message').waitFor();
  const text = await page.evaluate(() => [document.title, document.body.innerText, ...[...document.querySelectorAll('option,[placeholder],[aria-label]')].map((e) => e.textContent + (e.getAttribute('placeholder') || '') + (e.getAttribute('aria-label') || ''))].join('\n'));
  for (const re of [/premium/i, /\btiers?\b/i, /\bplans?\b/i, /\bour servers?\b/i, /database/i, /\bupgrade/i, /\bsolutions?\b/i, /\bundefined\b/, /\bnull\b/, /'/]) {
    const m = re.exec(text);
    if (m) throw new Error(`"${m[0]}" in: …${text.slice(Math.max(0, m.index - 30), m.index + 40).replace(/\s+/g, ' ')}…`);
  }
  await page.setViewportSize({ width: 360, height: 740 });
  await page.reload();
  await page.getByLabel('Your message').waitFor();
  const wide = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  await page.setViewportSize({ width: 1280, height: 900 });
  if (wide) throw new Error('horizontal scroll on a 360 px phone');
  return 'ok';
});

await check('contact: signed in, the account address is filled in; /contact?topic=job&job=ID opens the job form with the number', async () => {
  await page.unroute('**/api/me').catch(() => {});
  await page.unroute('**/api/contact').catch(() => {});
  await signIn(page, `e2e-contact-${Date.now()}@example.com`);
  const mine = await page.evaluate(async (url) => (await (await fetch(url)).json()).user.email, `${BASE}/api/me`);
  // the same answer as the real server's, with the form switched on
  await page.route('**/api/me', async (r) => { const real = await (await r.fetch()).json(); await r.fulfill({ status: 200, json: { ...real, contactForm: true } }); });
  const posts = [];
  await page.route('**/api/contact', async (r) => { posts.push({ body: r.request().postDataJSON(), csrf: r.request().headers()['x-csrf-token'] }); await r.fulfill({ status: 200, json: { ok: true } }); });
  const jid = 'a1b2c3d4e5f60718293a4b5c6d7e8f90';
  await page.goto(`${BASE}/contact?topic=job&job=${jid}`);
  await page.getByLabel('Your message').waitFor();
  if ((await page.getByLabel('Your email address').inputValue()) !== mine) throw new Error('the account address is not filled in');
  if ((await page.getByLabel('What is it about?').inputValue()) !== 'job' || (await page.getByLabel(/Job number/).inputValue()) !== jid) throw new Error('topic or job number not taken from the link');
  await page.getByLabel('Your message').fill('The cleaned file has a gap near the start.');
  await send(page).click();
  await page.getByTestId('contact-done').waitFor();
  if (posts.length !== 1 || posts[0].body.job !== jid || posts[0].body.topic !== 'job' || !posts[0].csrf) throw new Error(JSON.stringify(posts));
  // a job number that is not one is not taken over
  await page.goto(`${BASE}/contact?job=nonsense`);
  await page.getByLabel('Your message').waitFor();
  if ((await page.getByLabel('What is it about?').inputValue()) === 'job') throw new Error('a bad job number selected the job topic');
  return 'ok';
});

await check('contact: a failed job offers "Tell us about this job", with the job number in the link', async () => {
  const jid = 'f0e1d2c3b4a5968778695a4b3c2d1e0f';
  const t0 = Math.floor(Date.now() / 1000);
  await page.unroute('**/api/jobs').catch(() => {});
  await page.route('**/api/jobs', (r) => (r.request().method() === 'GET'
    ? r.fulfill({ status: 200, json: { jobs: [{ id: jid, kind: 'denoise', tier: 'standard', free: false, status: 'failed', stage: null, progress: 0, error: 'We couldn’t read that file. You were not charged.', inputName: 'talk.wav', label: 'Clean & polish', durationSeconds: 60, estCostMicros: 0, costMicros: 0, options: {}, notice: null, outputs: [], createdAt: t0, finishedAt: t0, expiresAt: null }] } })
    : r.continue()));
  await page.goto(`${BASE}/app`);
  const link = page.getByTestId('job-contact');
  await link.waitFor({ timeout: 15000 });
  const href = await link.getAttribute('href');
  if (!href.endsWith(`/contact?topic=job&job=${jid}`)) throw new Error(href);
  await link.click();
  await page.waitForURL(/\/contact\?topic=job/);
  if ((await page.getByLabel(/Job number/).inputValue()) !== jid) throw new Error('job number missing');
  await page.unroute('**/api/jobs');
  return 'ok';
});

console.log('LOGS:\n' + logs.filter((l) => !/Failed to load resource|net::ERR_FAILED/.test(l)).join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
