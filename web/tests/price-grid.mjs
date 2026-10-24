// What src/editor/tracks.js says about a list of track sets, as JSON on stdout, for the server's tests that check the website's price agrees
// with the server's (server/tests/test_web_agreement.py). Reads {pricing, cases: [{T, tracks}], converts: [{factor, seconds}]} as JSON on stdin:
//   cases:    -> {price: priceOf(tracks, pricing, T), timeline: toTimeline(tracks, T).toJSON()}
//   converts: -> convertPrice(pricing, factor, seconds)
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const HERE = dirname(fileURLToPath(import.meta.url));
let input = '';
for await (const chunk of process.stdin) input += chunk;
const { pricing, cases, converts } = JSON.parse(input);
const bundle = await build({
  stdin: { contents: "export { priceOf, convertPrice, toTimeline } from '../src/editor/tracks.js';", resolveDir: HERE, loader: 'js' },
  bundle: true,
  format: 'esm',
  platform: 'node',
  write: false,
  logLevel: 'silent',
});
const m = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);
process.stdout.write(JSON.stringify({
  cases: cases.map((c) => ({ price: m.priceOf(c.tracks, pricing, c.T), timeline: m.toTimeline(c.tracks, c.T).toJSON() })),
  converts: converts.map((c) => m.convertPrice(pricing, c.factor, c.seconds)),
}));
