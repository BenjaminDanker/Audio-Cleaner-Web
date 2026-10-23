// What src/media/convert.js says about a list of cases, as JSON on stdout, for the server's tests that check it agrees with
// common/acw_common/ffmpeg.py (server/tests/test_web_agreement.py). Reads {factors, works} as JSON on stdin:
//   factors: [{w, h, fps, out: [w, h] | null}]                        -> convertFactor
//   works:   [{w, h, fps, codec, source, format, size}]               -> serverPictureWork (size: the shorter side in pixels, null: the same)
import { loadConvert } from './convert-bundle.mjs';

let input = '';
for await (const chunk of process.stdin) input += chunk;
const cases = JSON.parse(input);
const m = await loadConvert({});
const out = {
  factors: cases.factors.map((c) => m.convertFactor(c.w, c.h, c.fps ?? undefined, c.out ? { width: c.out[0], height: c.out[1] } : null)),
  works: cases.works.map((c) => {
    const info = { hasVideo: true, width: c.w, height: c.h, fps: c.fps ?? undefined, ...(c.codec ? { ffVideoCodec: c.codec } : {}) };
    return m.serverPictureWork(info, `talk.${c.source}`, c.format, c.size == null ? 'same' : String(c.size));
  }),
};
process.stdout.write(JSON.stringify(out));
