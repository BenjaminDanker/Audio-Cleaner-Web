import { BASE, check, launch, signIn, summary } from './helpers.mjs';
const { browser, page, logs } = await launch();
await signIn(page, `e2e-api-${Date.now()}@example.com`);
await check('api: 403 CSRF -> refresh /api/me once and retry', async () => {
  let n = 0; const seen = [];
  page.on('request', (r) => { if (r.url().includes('/api/me') || r.url().includes('/api/keys')) seen.push(`${r.method()} ${new URL(r.url()).pathname}`); });
  await page.route('**/api/keys', async (route) => {
    if (route.request().method() === 'POST' && n++ === 0) return route.fulfill({ status: 403, json: { detail: 'Missing or invalid CSRF token.' } });
    return route.continue();
  });
  await page.goto(`${BASE}/app/live`);
  await page.getByLabel('Name').fill('retry key');
  seen.length = 0;
  await page.getByRole('button', { name: 'Create key' }).click();
  await page.getByTestId('key-value').waitFor({ timeout: 10000 });
  return seen.join(' -> ');
});
await check('api: 401 on an API call -> redirected to /login', async () => {
  await page.unroute('**/api/keys');
  await page.route('**/api/jobs', (r) => r.fulfill({ status: 401, json: { detail: 'Sign in required.' } }));
  await page.goto(`${BASE}/app`);
  await page.waitForURL(/\/login$/, { timeout: 10000 });
  return 'ok';
});
await check('api: server down -> friendly error not a crash', async () => {
  await page.unroute('**/api/jobs');
  await page.route('**/api/billing/history', (r) => r.abort());
  await signIn(page, `e2e-api2-${Date.now()}@example.com`).catch(() => {});
  await page.goto(`${BASE}/app/billing`);
  await page.getByText(/Could not reach the server/).waitFor({ timeout: 10000 });
  return 'ok';
});
console.log('LOGS:\n' + logs.join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
