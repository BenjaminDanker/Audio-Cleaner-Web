// What src/editor/quote.js (quoteLocal) says about a list of timelines on a job, as JSON on stdout, for the server's tests that check it agrees with
// editing.make_quote (server/tests/test_web_quote_agreement.py). Reads {cases: [{timeline, seconds, quote, rendered, stepFixed}]} as JSON on stdin:
// `timeline` is the timeline asked about, `quote` and `rendered` come from the job's GET /timeline answer, `stepFixed` is pricing.stepFixedSeconds.
// Writes one {micros, lines, free, unchanged, approximate} or {error} per case.
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const HERE = dirname(fileURLToPath(import.meta.url));
let input = '';
for await (const chunk of process.stdin) input += chunk;
const { cases } = JSON.parse(input);
const bundle = await build({
  stdin: { contents: "export { quoteLocal } from '../src/editor/quote.js'; export { Timeline } from '../src/editor/timeline.js';", resolveDir: HERE, loader: 'js' },
  bundle: true,
  format: 'esm',
  platform: 'node',
  write: false,
  logLevel: 'silent',
});
const m = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);
process.stdout.write(JSON.stringify(cases.map((c) => {
  try {
    const tl = m.Timeline.fromJSON(c.timeline, c.seconds);
    const rendered = c.rendered ? m.Timeline.fromJSON(c.rendered, c.seconds) : null;
    const { micros, lines, free, unchanged, approximate } = m.quoteLocal(tl, c.quote, rendered, c.stepFixed);
    return { micros, lines, free, unchanged, approximate };
  } catch (e) {
    return { error: String(e.message || e) };
  }
})));
