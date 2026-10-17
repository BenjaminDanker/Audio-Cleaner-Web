// The contact form (/contact): with no mail set up on the test backend it is switched off (404 and a calm page); with mail
// "on" (answered by the test, as the backend has no mail account) the form sends what was typed and nothing more, a hidden trap
// field stays empty and out of reach, a rate-limit wait is said in hours, and a failed job's link opens the form with its number.
// Needs the local backend (port 8792, see helpers.mjs) with `npm run build` output. Usage: node tests/contact.mjs
import { BASE, check, launch, signIn, summary } from './helpers.mjs';

const ME_OUT = { signedIn: false, contactForm: true };
const GOOD = { name: 'Sam', email: 'sam@example.com', message: 'How long do you keep my results for?' };

/** Pretend the server can send mail: /api/me says so (signed out), and /api/contact is answered by `handler`. */
async function mailOn(page, handler = () => ({ status: 200, json: { ok: true } })) {
  const posts = [];
  await page.route('**/api/me', (r) => r.fulfill({ status: 200, json: ME_OUT }));
  await page.route('**/api/contact', async (r) => {
    posts.push(r.request().postDataJSON());
    await r.fulfill(handler(r.request().postDataJSON(), posts.length));
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

await check('contact, no mail set up on this server: /api/contact answers 404 and /contact says it is not switched on, with no form', async () => {
  await page.goto(`${BASE}/contact`);
  const r = await page.evaluate(async (url) => (await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email: 'a@example.com', topic: 'other', message: 'Hello, is anyone there?' }) })).status, `${BASE}/api/contact`);
  if (r !== 404) throw new Error(`POST /api/contact answered ${r}, expected 404`);
  await page.goto(`${BASE}/contact`);
  await page.getByRole('heading', { name: 'Contact' }).waitFor();
  const t = await page.locator('main').innerText();
  if (!/isn’t switched on/.test(t) || (await page.locator('form').count())) throw new Error(t);
  return '404 and a calm page';
});

await check('contact, mail on: Send stays off until the address and a real message are in; the message is sent as typed with the topic and nothing else (no job, no hidden value); the visitor is thanked', async () => {
  const posts = await mailOn(page);
  await page.goto(`${BASE}/contact`);
  await page.getByLabel('Your message').waitFor();
  if (!(await send(page).isDisabled())) throw new Error('enabled on an empty form');
  await fill(page, { email: 'not-an-address', message: 'short' });
  if (!(await send(page).isDisabled())) throw new Error('enabled with a bad address and a short message');
  await fill(page, { email: GOOD.email, message: 'short' });
  if (!(await send(page).isDisabled())) throw new Error('enabled with a message under 10 characters');
  await fill(page, { email: GOOD.email, message: GOOD.message });
  await page.getByLabel('What is it about?').selectOption('billing');
  await send(page).click();
  await page.getByTestId('contact-done').waitFor({ timeout: 10000 });
  const t = await page.getByTestId('contact-done').innerText();
  if (!/Thanks, we got your message and will reply by email/.test(t) || !t.includes(GOOD.email)) throw new Error(t);
  if (posts.length !== 1) throw new Error(`${posts.length} posts`);
  const p = posts[0];
  if (p.email !== GOOD.email || p.topic !== 'billing' || p.message !== GOOD.message || p.name !== 'Sam' || p.website !== '' || 'job' in p) throw new Error(JSON.stringify(p));
  return 'sent as typed';
});

await check('contact: the trap field is out of sight, out of the tab order and empty when a person sends', async () => {
  await mailOn(page);
  await page.goto(`${BASE}/contact`);
  await page.getByLabel('Your message').waitFor();
  const box = await page.locator('#c-extra').boundingBox();
  if (box && box.x > -1000 && box.x + box.width > 0) throw new Error(`the trap field is on screen: ${JSON.stringify(box)}`);
  if ((await page.locator('#c-extra').getAttribute('tabindex')) !== '-1' || (await page.locator('.hp').getAttribute('aria-hidden')) !== 'true') throw new Error('trap field is reachable');
  return 'hidden and skipped by Tab';
});

await check('contact: a rate limit with a long wait keeps the form, says how long in hours, and Send stays off until then', async () => {
  const posts = await mailOn(page, () => ({ status: 429, json: { detail: 'Too many requests, try again in 5 minutes.' }, headers: { 'Retry-After': '40000' } }));
  await page.goto(`${BASE}/contact`);
  await page.getByLabel('Your message').waitFor();
  await fill(page);
  await send(page).click();
  await page.getByTestId('contact-wait').waitFor();
  if (!(await send(page).isDisabled())) throw new Error('Send is on during a wait');
  if (!/You can send again in about \d+ hours/.test(await page.getByTestId('contact-wait').innerText())) throw new Error(await page.getByTestId('contact-wait').innerText());
  if ((await page.getByLabel('Your message').inputValue()) !== GOOD.message) throw new Error('the form lost what was typed');
  if (posts.length !== 1) throw new Error(`${posts.length} posts`);
  return (await page.getByTestId('contact-wait').innerText()).trim();
});

await check('contact, signed in and from a failed job: the account address is filled in; the job link opens the job topic with its number, which is sent', async () => {
  await page.unroute('**/api/me').catch(() => {});
  await page.unroute('**/api/contact').catch(() => {});
  const mail = `e2e-contact-${Date.now()}@example.com`;
  await signIn(page, mail);
  await page.route('**/api/me', async (r) => { const real = await (await r.fetch()).json(); await r.fulfill({ status: 200, json: { ...real, contactForm: true } }); });
  const posts = [];
  await page.route('**/api/contact', async (r) => { posts.push(r.request().postDataJSON()); await r.fulfill({ status: 200, json: { ok: true } }); });
  const jid = 'f0e1d2c3b4a5968778695a4b3c2d1e0f';
  await page.route('**/api/jobs', (r) => (r.request().method() === 'GET'
    ? r.fulfill({ status: 200, json: { jobs: [{ id: jid, kind: 'denoise', tier: 'standard', free: false, status: 'failed', stage: null, progress: 0, error: 'We couldn’t read that file. You were not charged.', inputName: 'talk.wav', label: 'Denoising', durationSeconds: 60, estCostMicros: 0, costMicros: 0, options: {}, notice: null, outputs: [], createdAt: Math.floor(Date.now() / 1000), finishedAt: Math.floor(Date.now() / 1000), expiresAt: null }] } })
    : r.continue()));
  await page.goto(`${BASE}/app`);
  const link = page.getByTestId('job-contact');
  await link.waitFor({ timeout: 15000 });
  if (!(await link.getAttribute('href')).endsWith(`/contact?topic=job&job=${jid}`)) throw new Error(await link.getAttribute('href'));
  await page.unroute('**/api/jobs');
  await link.click();
  await page.waitForURL(/\/contact\?topic=job/);
  await page.getByLabel('Your message').waitFor();
  if ((await page.getByLabel('Your email address').inputValue()) !== mail) throw new Error('the account address is not filled in');
  if ((await page.getByLabel(/Job number/).inputValue()) !== jid) throw new Error('job number missing');
  await page.getByLabel('Your message').fill('The cleaned file has a gap near the start.');
  await send(page).click();
  await page.getByTestId('contact-done').waitFor();
  if (posts.length !== 1 || posts[0].job !== jid || posts[0].topic !== 'job') throw new Error(JSON.stringify(posts));
  return 'ok';
});

console.log('LOGS:\n' + logs.filter((l) => !/Failed to load resource|net::ERR_FAILED/.test(l)).join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
