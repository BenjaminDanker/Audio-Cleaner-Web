// Lazy libav.js decoder: FFmpeg's audio decoders compiled to WebAssembly (LGPL build, decode only,
// native/libav/build.sh). Used only for files the browser cannot read itself (WMA, AVI, FLV, MPEG,
// AC-3, DTS, AMR, WavPack, APE, ...), so most people never download it (~1.2 MB compressed).
// It runs in its own worker and reads the file in pieces, so large files never sit in memory.

const DIR = __LIBAV_DIR__;

let libavPromise = null;

export const libavAvailable = () => Boolean(DIR);

function loadLibav() {
  if (!libavPromise) {
    const url = new URL(`${import.meta.env.BASE_URL}dfn/libav/${DIR}/libav-acw-audio.mjs`, location.href).href;
    libavPromise = import(/* @vite-ignore */ url)
      .then((mod) => mod.default.LibAV())
      .catch((e) => {
        libavPromise = null; // let a later file try again (e.g. after a network hiccup)
        throw e;
      });
  }
  return libavPromise;
}

function toPlanes(libav, frame) {
  const n = frame.nb_samples;
  const planar = Array.isArray(frame.data);
  const ch = frame.channels || (planar ? frame.data.length : 1);
  let off = 0;
  let mul = 1;
  switch (frame.format) {
    case libav.AV_SAMPLE_FMT_U8: case libav.AV_SAMPLE_FMT_U8P: off = 128; mul = 1 / 128; break;
    case libav.AV_SAMPLE_FMT_S16: case libav.AV_SAMPLE_FMT_S16P: mul = 1 / 32768; break;
    case libav.AV_SAMPLE_FMT_S32: case libav.AV_SAMPLE_FMT_S32P: mul = 1 / 2147483648; break;
    default: break; // float / double
  }
  const planes = [];
  for (let c = 0; c < ch; c++) {
    const out = new Float32Array(n);
    if (planar) {
      const p = frame.data[c];
      for (let i = 0; i < n; i++) out[i] = (p[i] - off) * mul;
    } else {
      const p = frame.data;
      for (let i = 0; i < n; i++) out[i] = (p[i * ch + c] - off) * mul;
    }
    planes.push(out);
  }
  return { planes, rate: frame.sample_rate };
}

/**
 * Async generator: first {type:'info', sourceRate, srcChannels}, then {type:'frame', planes, rate}
 * (one Float32Array per source channel, in order). Timestamps are not used: samples come out back to
 * back, which is right for audio files (these formats rarely carry gaps).
 */
export async function* libavFrames(file, { signal } = {}) {
  const libav = await loadLibav();
  const name = `in-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  await libav.mkreadaheadfile(name, file);
  let fmt = 0;
  let dec = null;
  try {
    let streams;
    try {
      [fmt, streams] = await libav.ff_init_demuxer_file(name);
    } catch {
      throw new Error('unreadable');
    }
    const st = streams.find((s) => s.codec_type === libav.AVMEDIA_TYPE_AUDIO);
    if (!st) throw new Error('no audio');
    dec = await libav.ff_init_decoder(st.codec_id, { codecpar: st.codecpar, time_base: [st.time_base_num, st.time_base_den] });
    const [, c, pkt, frame] = dec;
    let info = false;
    for (;;) {
      if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
      const [res, packets] = await libav.ff_read_frame_multi(fmt, pkt, { limit: 256 * 1024 });
      const eof = res === libav.AVERROR_EOF;
      if (res !== 0 && !eof && res !== -libav.EAGAIN) throw new Error(`read error ${res}`);
      const frames = await libav.ff_decode_multi(c, pkt, frame, packets[st.index] || [], { fin: eof, ignoreErrors: true });
      for (const f of frames) {
        const { planes, rate } = toPlanes(libav, f);
        if (!planes.length || !rate) continue;
        if (!info) {
          info = true;
          yield { type: 'info', sourceRate: rate, srcChannels: planes.length };
        }
        yield { type: 'frame', planes, rate };
      }
      if (eof) break;
    }
    if (!info) throw new Error('no audio decoded');
  } finally {
    if (dec) await libav.ff_free_decoder(dec[1], dec[2], dec[3]).catch(() => {});
    if (fmt) await libav.avformat_close_input_js(fmt).catch(() => {});
    await libav.unlinkreadaheadfile(name).catch(() => {});
  }
}

/** What libav.js can tell about a file the browser can't read: {duration, hasAudio, hasVideo, audioTracks, channels, sampleRate}, or null. */
export async function libavProbe(file) {
  if (!DIR) return null;
  let libav;
  try {
    libav = await loadLibav();
  } catch {
    return null;
  }
  const name = `probe-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  let fmt = 0;
  try {
    await libav.mkreadaheadfile(name, file);
    let streams;
    [fmt, streams] = await libav.ff_init_demuxer_file(name);
    const a = streams.find((s) => s.codec_type === libav.AVMEDIA_TYPE_AUDIO);
    const v = streams.some((s) => s.codec_type === libav.AVMEDIA_TYPE_VIDEO);
    const lo = await libav.AVFormatContext_duration(fmt);
    const hi = await libav.AVFormatContext_durationhi(fmt);
    const us = hi * 4294967296 + (lo >>> 0);
    return {
      duration: us > 0 && Number.isFinite(us) ? us / 1e6 : null, // AV_TIME_BASE units; unknown is a huge negative number
      hasAudio: !!a,
      hasVideo: v,
      audioTracks: streams.filter((s) => s.codec_type === libav.AVMEDIA_TYPE_AUDIO).length,
      channels: a ? await libav.AVCodecParameters_ch_layout_nb_channels(a.codecpar) : null,
      sampleRate: a ? await libav.AVCodecParameters_sample_rate(a.codecpar) : null,
    };
  } catch {
    return null;
  } finally {
    if (fmt) await libav.avformat_close_input_js(fmt).catch(() => {});
    await libav.unlinkreadaheadfile(name).catch(() => {});
  }
}
