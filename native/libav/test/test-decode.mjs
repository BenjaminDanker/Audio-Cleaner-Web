#!/usr/bin/env node
// Decode every fixture with the custom libav.js build (non-threaded wasm, run in Node),
// compare the PCM against native ffmpeg's decode of the same stream, and report decode
// speed as a multiple of real time (single-threaded; one JS thread, wasm only).
//
//   node test-decode.mjs [--dist DIR] [--fixtures DIR] [--ffmpeg PATH] [--runs N] [--only a,b]
//
// Exit code is non-zero if any fixture fails to decode or does not match the reference.
import { createRequire } from "node:module";
import { readdirSync, readFileSync, existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { homedir } from "node:os";
import path from "node:path";

const require = createRequire(import.meta.url);
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const BUILD = path.join(homedir(), "audio-cleaner/src/libav-build");
const DIST = arg("--dist", path.join(BUILD, "dist"));
const FIX = arg("--fixtures", path.join(BUILD, "fixtures"));
const FFMPEG = arg("--ffmpeg", path.join(homedir(), "audio-cleaner/bin/ffmpeg"));
const RUNS = Number(arg("--runs", "3"));
const ONLY = arg("--only", "") ? arg("--only", "").split(",") : null;
const VARIANT = "acw-audio";

const LibAV = require(path.join(DIST, `libav-${VARIANT}.js`));

function toF32(frame, libav) {
  // frame.data: typed array (interleaved) or array of typed arrays (planar)
  const F = libav;
  const planar = Array.isArray(frame.data);
  const n = frame.nb_samples;
  const ch = frame.channels || (planar ? frame.data.length : 1);
  const scale = (fmt) => {
    switch (fmt) {
      case F.AV_SAMPLE_FMT_U8: case F.AV_SAMPLE_FMT_U8P: return [128, 1 / 128];
      case F.AV_SAMPLE_FMT_S16: case F.AV_SAMPLE_FMT_S16P: return [0, 1 / 32768];
      case F.AV_SAMPLE_FMT_S32: case F.AV_SAMPLE_FMT_S32P: return [0, 1 / 2147483648];
      default: return [0, 1];
    }
  };
  const [off, mul] = scale(frame.format);
  const out = new Float32Array(n * ch);
  if (planar) {
    for (let c = 0; c < ch; c++) { const p = frame.data[c]; for (let i = 0; i < n; i++) out[i * ch + c] = (p[i] - off) * mul; }
  } else {
    const p = frame.data; for (let i = 0; i < n * ch; i++) out[i] = (p[i] - off) * mul;
  }
  return { pcm: out, ch };
}

async function decodeFile(libav, name, bytes) {
  const vname = "in_" + name.replace(/[^A-Za-z0-9._-]/g, "_");
  await libav.writeFile(vname, bytes);
  const [fmt, streams] = await libav.ff_init_demuxer_file(vname);
  const audio = streams.filter((s) => s.codec_type === libav.AVMEDIA_TYPE_AUDIO);
  if (!audio.length) throw new Error("no audio stream");
  const st = audio[0];
  const [, c, pkt, frame] = await libav.ff_init_decoder(st.codec_id, {
    codecpar: st.codecpar, time_base: [st.time_base_num, st.time_base_den],
  });
  const chunks = [];
  let total = 0, ch = 0, sampleRate = 0, codecName = "";
  try { codecName = await libav.avcodec_get_name?.(st.codec_id); } catch (e) { /* optional */ }
  const t0 = performance.now();
  for (;;) {
    const [res, packets] = await libav.ff_read_frame_multi(fmt, pkt, { limit: 256 * 1024 });
    const eof = res === libav.AVERROR_EOF;
    if (res !== 0 && !eof && res !== -libav.EAGAIN) throw new Error("read error " + res);
    const frames = await libav.ff_decode_multi(c, pkt, frame, packets[st.index] || [], { fin: eof, ignoreErrors: true });
    for (const f of frames) {
      const { pcm, ch: fch } = toF32(f, libav);
      ch = fch; sampleRate = f.sample_rate || sampleRate;
      chunks.push(pcm); total += pcm.length;
    }
    if (eof) break;
  }
  const ms = performance.now() - t0;
  await libav.ff_free_decoder(c, pkt, frame);
  await libav.avformat_close_input_js(fmt);
  await libav.unlink(vname);
  const pcm = new Float32Array(total);
  let o = 0; for (const x of chunks) { pcm.set(x, o); o += x.length; }
  return { pcm, ch, sampleRate, ms, nStreams: streams.length, codecId: st.codec_id, codecName };
}

function reference(file) {
  // native ffmpeg decode of the first audio stream at its native rate / layout, as f32le
  const probe = spawnSync(FFMPEG, ["-hide_banner", "-i", file], { encoding: "utf8" });
  const m = /Audio:.*?, (\d+) Hz, ([^,]+),/.exec(probe.stderr);
  const r = spawnSync(FFMPEG, ["-v", "error", "-i", file, "-map", "0:a:0", "-f", "f32le", "-"], { maxBuffer: 1 << 30 });
  const buf = r.stdout;
  const pcm = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length - (buf.length % 4)));
  return { pcm, rate: m ? Number(m[1]) : 0, layout: m ? m[2] : "?" };
}

function snr(a, b) {
  const n = Math.min(a.length, b.length);
  let sig = 0, err = 0;
  for (let i = 0; i < n; i++) { const d = a[i] - b[i]; sig += b[i] * b[i]; err += d * d; }
  if (err === 0) return Infinity;
  return 10 * Math.log10((sig + 1e-20) / err);
}

const tLoad0 = performance.now();
const libav = await LibAV.LibAV({ noworker: true, nothreads: true, base: DIST, variant: VARIANT });
const loadMs = performance.now() - tLoad0;
console.log(`libav.js mode=${libav.libavjsMode}, wasm module load+instantiate ${loadMs.toFixed(0)} ms`);
if (typeof SharedArrayBuffer !== "undefined") { /* node has it; the glue must not need it */ }

const files = readdirSync(FIX).filter((f) => !f.startsWith("_") && (!ONLY || ONLY.some((o) => f.startsWith(o)))).sort();
const rows = [];
let failed = 0;
for (const f of files) {
  const file = path.join(FIX, f);
  const bytes = new Uint8Array(readFileSync(file));
  let row = { f };
  try {
    const ref = reference(file);
    let best = Infinity, res;
    for (let i = 0; i < RUNS; i++) { res = await decodeFile(libav, f, bytes); best = Math.min(best, res.ms); }
    const dur = res.pcm.length / res.ch / res.sampleRate;
    const refDur = ref.pcm.length / res.ch / (ref.rate || res.sampleRate);
    const q = snr(res.pcm, ref.pcm);
    const lenOk = Math.abs(res.pcm.length - ref.pcm.length) <= Math.max(res.ch * 4096, ref.pcm.length * 0.01);
    const ok = res.pcm.length > 0 && lenOk && q > 60 && res.sampleRate === ref.rate;
    row = { f, ok, ch: res.ch, sr: res.sampleRate, dur: dur.toFixed(2), refDur: refDur.toFixed(2), snr: q === Infinity ? "exact" : q.toFixed(1) + " dB", ms: best.toFixed(0), xrt: (dur * 1000 / best).toFixed(0) + "x", streams: res.nStreams };
  } catch (e) {
    row = { f, ok: false, err: String(e && e.message || e) };
  }
  if (!row.ok) failed++;
  rows.push(row);
  console.log(row.ok ? `PASS ${f.padEnd(16)} ${row.ch}ch ${row.sr}Hz dur ${row.dur}s (ref ${row.refDur}s) snr ${row.snr} decode ${row.ms} ms = ${row.xrt} real time (${row.streams} streams)`
                    : `FAIL ${f.padEnd(16)} ${row.err || JSON.stringify(row)}`);
}

// Resample / channel-convert path (swresample via the aresample filter)
try {
  const f = files.find((x) => x.startsWith("wma")) || files[0];
  const bytes = new Uint8Array(readFileSync(path.join(FIX, f)));
  await libav.writeFile("rs_in", bytes);
  const [fmt, streams] = await libav.ff_init_demuxer_file("rs_in");
  const st = streams.find((s) => s.codec_type === libav.AVMEDIA_TYPE_AUDIO);
  const [, c, pkt, frame] = await libav.ff_init_decoder(st.codec_id, { codecpar: st.codecpar, time_base: [st.time_base_num, st.time_base_den] });
  let graph, src, sink, nOut = 0, srcRate = 0;
  const t0 = performance.now();
  for (;;) {
    const [res, packets] = await libav.ff_read_frame_multi(fmt, pkt, { limit: 256 * 1024 });
    const eof = res === libav.AVERROR_EOF;
    const frames = await libav.ff_decode_multi(c, pkt, frame, packets[st.index] || [], { fin: eof, ignoreErrors: true });
    // Some containers (e.g. ASF) leave the channel layout unspecified; libav.js then reports
    // channel_layout 0 and the filter graph would be built as mono. Give such frames the
    // default layout for their channel count (1 = mono, 2 = stereo) before filtering.
    for (const fr of frames) if (!fr.channel_layout) fr.channel_layout = fr.channels === 1 ? 4 : 3;
    if (frames.length && !graph) {
      const f0 = frames[0]; srcRate = f0.sample_rate;
      [graph, src, sink] = await libav.ff_init_filter_graph("aresample=16000,aformat=sample_fmts=flt:channel_layouts=mono",
        { sample_rate: f0.sample_rate, sample_fmt: f0.format, channel_layout: f0.channel_layout },
        { sample_rate: 16000, sample_fmt: libav.AV_SAMPLE_FMT_FLT, channel_layout: 4 });
    }
    if (graph) {
      const out = await libav.ff_filter_multi(src, sink, frame, frames, eof);
      for (const o of out) nOut += o.nb_samples;
    }
    if (eof) break;
  }
  const ms = performance.now() - t0;
  console.log(`PASS resample ${f}: ${srcRate} Hz stereo -> 16000 Hz mono via aresample, ${(nOut / 16000).toFixed(2)} s out, ${ms.toFixed(0)} ms (decode+resample ${((nOut / 16000) * 1000 / ms).toFixed(0)}x real time)`);
  if (nOut === 0) { failed++; console.log("FAIL resample produced no samples"); }
} catch (e) {
  failed++; console.log("FAIL resample:", e && e.message || e);
}

console.log(`\n${rows.length - failed}/${rows.length} fixtures ok` + (failed ? `, ${failed} problem(s)` : ""));
process.exit(failed ? 1 : 0);
