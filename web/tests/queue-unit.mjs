// The line under a queued file job (src/jobs.js queueText): its place in the queue, or the maintenance pause the server reports.
// The file is bundled with esbuild so that `import.meta.env` in src/api.js is defined.  Usage: node tests/queue-unit.mjs
import assert from 'node:assert/strict';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const HERE = dirname(fileURLToPath(import.meta.url));
const bundle = await build({
  stdin: { contents: "export { queueText } from '../src/jobs.js';", resolveDir: HERE, loader: 'js' },
  bundle: true,
  format: 'esm',
  platform: 'node',
  write: false,
  logLevel: 'silent',
  define: { 'import.meta.env.BASE_URL': '"/"', __LIBAV_DIR__: '"/"' },
});
const { queueText } = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);

let failed = 0;
async function check(name, fn) {
  try {
    await fn();
    console.log(`PASS ${name}`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${name} -- ${String(e.message).split('\n').slice(0, 4).join(' / ')}`);
  }
}

const MSG = "Paused for a short maintenance: it starts as soon as we're back";

await check('a queued job shows its place and the rough wait', () => {
  assert.equal(queueText({ status: 'queued', queuePosition: 1 }), 'You’re next');
  assert.equal(queueText({ status: 'queued', queuePosition: 1, queueWaitSeconds: 20 }), 'You’re next, about under a minute');
  assert.equal(queueText({ status: 'queued', queuePosition: 3, queueWaitSeconds: 300 }), 'In line: 2 ahead of you, about 5 min');
  assert.equal(queueText({ status: 'queued', queuePosition: 3, queueWaitSeconds: null }), 'In line: 2 ahead of you');
});

await check('a paused job shows the server text instead of a place and a wait', () => {
  assert.equal(queueText({ status: 'queued', queuePosition: 4, queueWaitSeconds: null, paused: MSG }), MSG);
  assert.equal(queueText({ status: 'queued', queuePosition: 1, queueWaitSeconds: 600, paused: MSG }), MSG);
});

await check('a paused flag without text still says it is paused', () => {
  assert.equal(queueText({ status: 'queued', queuePosition: 2, paused: true }), 'Paused for a short maintenance');
});

await check('a falsy paused value changes nothing', () => {
  for (const paused of [undefined, null, false, '']) {
    assert.equal(queueText({ status: 'queued', queuePosition: 2, queueWaitSeconds: 120, paused }), 'In line: 1 ahead of you, about 2 min');
  }
});

await check('nothing is shown unless the job is queued with a position, paused or not', () => {
  assert.equal(queueText({ status: 'running', queuePosition: 1, paused: MSG }), '');
  assert.equal(queueText({ status: 'done', paused: MSG }), '');
  assert.equal(queueText({ status: 'queued', queuePosition: null, paused: MSG }), '');
  assert.equal(queueText({ status: 'queued', queuePosition: 0, paused: MSG }), '');
});

console.log(failed ? `${failed} FAILED` : 'all passed');
process.exit(failed ? 1 : 0);
