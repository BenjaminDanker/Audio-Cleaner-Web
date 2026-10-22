// Choose what the result is called and what it is: audio formats (WAV, FLAC, MP3, Ogg Opus, M4A) and video files
// (MP4, MOV, MKV, WebM), at the original picture size or another one. Everything happens in the visitor's browser
// (their processor and graphics chip, nothing uploaded, no charge) with mediabunny and WebCodecs.
//
// - what this browser can make is asked of the browser (WebCodecs), never assumed: outputCaps() / listOutputs()
// - the picture is copied as it is whenever the new container can hold it and the size stays the same; otherwise it is
//   re-encoded (H.264, or VP9 for WebM; VP9 stands in when the browser cannot write H.264)
// - MP3 has no encoder in browsers: a small LAME build (@mediabunny/mp3-encoder, its own file, downloaded only when
//   somebody saves an MP3) does it; FLAC is written by our own encoder (flac.js)
// - frames are streamed through mediabunny, so a long video is never held decoded in memory; only the finished (small)
//   output file is kept until it is saved

import { encodeFlac, sniffPcm } from './flac.js';
import { loadMediabunny, openInput } from './mb.js';
import { buildWav } from './wav.js';

export class RemuxUnsupported extends Error {}

/** Without a place on disk for it (see diskTarget), a video this big is not rebuilt: the finished file would be
 *  assembled in memory. Browsers that have one (Chrome, Edge, Firefox) build any size the disk has room for. */
export const MAX_BUILD_BYTES = 1.5 * 1024 ** 3;

/** Can this browser write a finished file straight to disk (its private storage), so its size doesn't matter? */
export const canBuildOnDisk = () => typeof navigator !== 'undefined' && typeof navigator.storage?.getDirectory === 'function'
  && typeof globalThis.FileSystemFileHandle?.prototype?.createWritable === 'function';

/** The most this browser can rebuild for a video of `bytes` (null: no limit but the disk's free room). */
export const buildLimit = () => (canBuildOnDisk() ? null : MAX_BUILD_BYTES);

/* ------------------------------------------------------------------------------------------------ the formats */

export const FORMATS = [
  { id: 'wav', kind: 'audio', label: 'WAV', ext: 'wav', exts: ['wav', 'wave'], mime: 'audio/wav' },
  { id: 'flac', kind: 'audio', label: 'FLAC', ext: 'flac', exts: ['flac'], mime: 'audio/flac' },
  { id: 'mp3', kind: 'audio', label: 'MP3', ext: 'mp3', exts: ['mp3'], mime: 'audio/mpeg' },
  { id: 'ogg', kind: 'audio', label: 'Ogg (Opus)', ext: 'ogg', exts: ['ogg', 'opus', 'oga'], mime: 'audio/ogg' },
  { id: 'm4a', kind: 'audio', label: 'M4A (AAC)', ext: 'm4a', exts: ['m4a'], mime: 'audio/mp4' },
  { id: 'mp4', kind: 'video', label: 'MP4', ext: 'mp4', exts: ['mp4', 'm4v'], mime: 'video/mp4' },
  { id: 'mov', kind: 'video', label: 'MOV', ext: 'mov', exts: ['mov'], mime: 'video/quicktime' },
  { id: 'mkv', kind: 'video', label: 'MKV', ext: 'mkv', exts: ['mkv'], mime: 'video/x-matroska' },
  { id: 'webm', kind: 'video', label: 'WebM', ext: 'webm', exts: ['webm'], mime: 'video/webm' },
];
export const FORMAT = Object.fromEntries(FORMATS.map((f) => [f.id, f]));

// Endings people may type that mean "a media file we do not write": named in the hint, never silently changed.
const OTHER_EXTS = ['aac', 'aif', 'aiff', 'wma', 'amr', 'avi', 'flv', 'wmv', 'mpg', 'mpeg', 'ts', '3gp', 'mka', 'ac3', 'dts'];
const EXT_TO_ID = new Map(FORMATS.flatMap((f) => f.exts.map((e) => [e, f.id])));

/** The ending of a file name, lower case, no dot ('' when there is none). */
export function endingOf(name) {
  const base = String(name || '').split(/[\\/]/).pop();
  const i = base.lastIndexOf('.');
  return i > 0 && i < base.length - 1 ? base.slice(i + 1).toLowerCase() : '';
}

/** The format a typed name asks for ('' when its ending is not one of ours). */
export function formatOfName(name) {
  return EXT_TO_ID.get(endingOf(name)) || '';
}

/** The name with its ending changed to `id`'s (a name already ending the way that format is written stays as typed). */
export function nameForFormat(name, id) {
  const f = FORMAT[id];
  if (!f) return name;
  const end = endingOf(name);
  if (end && f.exts.includes(end)) return name;
  const known = EXT_TO_ID.has(end) || OTHER_EXTS.includes(end);
  const stem = known ? String(name).slice(0, String(name).length - end.length - 1) : String(name);
  return `${stem.replace(/\.+$/, '')}.${f.ext}`;
}

const BAD_CHARS = /[\\/:*?"<>|\u0000-\u001f]/;

/**
 * Is this a usable result name? '' when it is, else a short hint in plain words.
 * `allowed` is the list of format ids this browser can make for the file.
 */
export function checkName(name, allowed) {
  const n = String(name ?? '').trim();
  const end = endingOf(n);
  const stem = end ? n.slice(0, n.length - end.length - 1) : n;
  if (!stem.trim()) return 'Give the file a name first.';
  if (n.length > 180) return 'That name is a bit long. Please shorten it.';
  if (BAD_CHARS.test(n)) return 'A file name can’t have any of these: \\ / : * ? “ < > |';
  const id = EXT_TO_ID.get(end);
  if (!id) {
    const list = (allowed || []).map((a) => `.${FORMAT[a].ext}`).join(', ');
    return end ? `We can’t make .${end} files. Try ${list}.` : `End the name with the kind of file you want, like ${list}.`;
  }
  if (allowed && !allowed.includes(id)) return `This browser can’t make .${end} files for this recording. Pick one from the list.`;
  return '';
}

/** A file name safe to hand to the download (spaces trimmed; the checks above already refused the rest). */
export const cleanName = (name) => String(name ?? '').trim();

/* ------------------------------------------------------------------------------------------------ the sizes */

export const SIZES = [
  { id: 'same', label: 'Same as the original', px: 0 },
  { id: '2160', label: '2160p (4K)', px: 2160 },
  { id: '1440', label: '1440p', px: 1440 },
  { id: '1080', label: '1080p (Full HD)', px: 1080 },
  { id: '720', label: '720p (HD)', px: 720 },
  { id: '480', label: '480p', px: 480 },
];

/** Width and height (both even) for a picture whose shorter side is `px` and whose shape stays the same. */
export function scaledDims(width, height, px) {
  const k = px / Math.min(width, height);
  const even = (v) => Math.max(2, Math.round(v / 2) * 2);
  return { width: even(width * k), height: even(height * k) };
}

/* ------------------------------------------------------------------------------------------------ what this browser can do */

// Which audio codec each video file type gets for the cleaned sound (first one the browser can write), and which
// picture codecs are used when the picture has to be encoded again (first one the browser can write).
const AUDIO_PREFS = { mp4: ['aac', 'opus'], mov: ['aac', 'pcm-s16'], mkv: ['aac', 'opus', 'mp3', 'pcm-s16'], webm: ['opus'] };
const VIDEO_PREFS = { mp4: ['avc', 'vp9'], mov: ['avc'], mkv: ['avc', 'vp9', 'vp8'], webm: ['vp9', 'vp8'] };

const wasmOk = () => typeof WebAssembly !== 'undefined' && typeof Worker !== 'undefined';

let mp3Ready = null;
/** Makes MP3 writing available (downloads the small encoder file the first time). Resolves true when it can be used. */
export function ensureMp3() {
  if (!mp3Ready) {
    mp3Ready = (async () => {
      const mb = await loadMediabunny();
      if (await mb.canEncodeAudio('mp3')) return true;
      if (!wasmOk()) return false;
      const { registerMp3Encoder } = await import('@mediabunny/mp3-encoder');
      registerMp3Encoder();
      return true;
    })().catch(() => {
      mp3Ready = null;
      return false;
    });
  }
  return mp3Ready;
}

let capsPromise = null;
/** What the browser can encode: {audio: {opus, aac, mp3}, video: {avc, vp9, vp8}}. Asked once. */
export function outputCaps() {
  if (!capsPromise) {
    capsPromise = (async () => {
      const mb = await loadMediabunny();
      const safe = async (fn) => {
        try {
          return !!(await fn());
        } catch {
          return false;
        }
      };
      const a = { numberOfChannels: 2, sampleRate: 48000 };
      const v = { width: 1280, height: 720 };
      const [opus, aac, avc, vp9, vp8] = await Promise.all([
        safe(() => mb.canEncodeAudio('opus', a)), safe(() => mb.canEncodeAudio('aac', a)),
        safe(() => mb.canEncodeVideo('avc', v)), safe(() => mb.canEncodeVideo('vp9', v)), safe(() => mb.canEncodeVideo('vp8', v)),
      ]);
      return { audio: { opus, aac, mp3: wasmOk() }, video: { avc, vp9, vp8 } };
    })();
  }
  return capsPromise;
}

function makeFormat(mb, id) {
  switch (id) {
    case 'wav': return new mb.WavOutputFormat();
    case 'mp3': return new mb.Mp3OutputFormat();
    case 'ogg': return new mb.OggOutputFormat();
    case 'm4a': return new mb.Mp4OutputFormat();
    case 'mp4': return new mb.Mp4OutputFormat();
    case 'mov': return new mb.MovOutputFormat();
    case 'mkv': return new mb.MkvOutputFormat();
    case 'webm': return new mb.WebMOutputFormat();
    default: throw new Error('Unknown format');
  }
}

/** Audio codecs for a video file type that this browser can write, best first. */
function audioCodecsFor(caps, container) {
  return AUDIO_PREFS[container].filter((c) => (c === 'pcm-s16' ? true : caps.audio[c === 'aac' ? 'aac' : c === 'opus' ? 'opus' : 'mp3']));
}

/**
 * How the picture of a video file would be made. `info` describes the source ({hasVideo, videoCodec, videoCanDecode}).
 * @returns {Promise<{ok:boolean, reencode:boolean, codec:string|null, why:string}>}
 */
export async function videoPlan(info, container, px = 0, dims = null) {
  const mb = await loadMediabunny();
  const caps = await outputCaps();
  const supported = makeFormat(mb, container).getSupportedVideoCodecs();
  const sourceIsVideo = !!info?.hasVideo && !info?.viaLibav;
  if (sourceIsVideo && !px && info.videoCodec && supported.includes(info.videoCodec)) {
    return { ok: true, reencode: false, codec: null, why: '' };
  }
  if (sourceIsVideo && info.videoCanDecode === false) {
    return { ok: false, reencode: true, codec: null, why: 'This browser can’t open the picture of this video, so it can only be copied as it is.' };
  }
  const want = dims || { width: 1280, height: 720 };
  for (const c of VIDEO_PREFS[container]) {
    if (!caps.video[c === 'avc' ? 'avc' : c === 'vp9' ? 'vp9' : 'vp8']) continue;
    if (!supported.includes(c)) continue;
    let ok = false;
    try {
      ok = await mb.canEncodeVideo(c, want);
    } catch { /* not this one */ }
    if (ok) return { ok: true, reencode: true, codec: c, why: '' };
  }
  return { ok: false, reencode: true, codec: null, why: 'This browser can’t make the picture for this kind of file.' };
}

/**
 * Everything the "save as" list may hold, with whether this browser can make each for this file.
 * `serverAudio`: the sound itself comes from our server (paid cleaning), which writes any audio format, so only the
 * video files depend on this browser. A video type this browser can't make for a video is made on our side (`server`: the whole
 * video is uploaded; docs/api.md options.video_out).
 * @returns {Promise<{id:string, kind:'audio'|'video', label:string, ok:boolean, why:string, server?:boolean}[]>}
 */
export async function listOutputs({ info, serverAudio = false }) {
  const caps = await outputCaps();
  const out = [];
  const sec = info?.duration || 0;
  const flacFits = !sec || sec * 48000 * (info?.channels >= 2 ? 2 : 1) * 2 <= 1024 ** 3;
  const audio = {
    wav: [true, ''],
    flac: [serverAudio || flacFits, 'This recording is too long to write as FLAC in a browser.'],
    mp3: [serverAudio || caps.audio.mp3, 'This browser can’t write MP3.'],
    ogg: [serverAudio || caps.audio.opus, 'This browser can’t write Opus.'],
    m4a: [serverAudio || caps.audio.aac, 'This browser can’t write AAC.'],
  };
  for (const f of FORMATS.filter((x) => x.kind === 'audio')) out.push({ id: f.id, kind: 'audio', label: f.label, ok: audio[f.id][0], why: audio[f.id][1] });
  const videoSource = !!info?.hasVideo && !info?.viaLibav;
  for (const f of FORMATS.filter((x) => x.kind === 'video')) {
    const plan = info?.hasVideo && !videoSource ? { ok: false, why: '' } : await videoPlan(info, f.id, 0);
    const audioOk = audioCodecsFor(caps, f.id).length > 0;
    if (plan.ok && audioOk) out.push({ id: f.id, kind: 'video', label: f.label, ok: true, why: '' });
    else if (info?.hasVideo) out.push({ id: f.id, kind: 'video', label: f.label, ok: true, why: '', server: true }); // made on our side
    else out.push({ id: f.id, kind: 'video', label: f.label, ok: false, why: plan.ok ? 'This browser can’t write the sound for this kind of file.' : plan.why });
  }
  return out;
}

/** Which picture sizes can be made for this container (the first, 'same', always can when the format is offered). A size this browser
 * can't make for a video is made on our side (`server`); with `server` (a file type made on our side) every size is. */
export async function listSizes(info, container, server = false) {
  const out = [];
  for (const s of SIZES) {
    if (!s.px) { out.push({ ...s, ok: true, ...(server ? { server: true } : {}) }); continue; }
    if (server) { out.push({ ...s, ok: true, server: true }); continue; }
    const d = info?.width && info?.height ? scaledDims(info.width, info.height, s.px) : null;
    const plan = await videoPlan(info, container, s.px, d);
    out.push(plan.ok ? { ...s, ok: true } : info?.hasVideo ? { ...s, ok: true, server: true } : { ...s, ok: false, why: plan.why });
  }
  return out;
}

/** The picture size our side is asked for: the shorter side in pixels, or null for the original size (docs/api.md options.video_out). */
export const serverSize = (id) => (SIZES.find((s) => s.id === id)?.px || null);

/** The format a result starts out as: the video's own kind of file when it is kept, WAV otherwise. */
export function defaultFormatId(file, info, keepVideo = true, serverAudio = false) {
  if (info?.hasVideo && keepVideo && !info?.viaLibav) {
    const id = EXT_TO_ID.get(endingOf(file?.name));
    return id && FORMAT[id].kind === 'video' ? id : 'mp4';
  }
  // Sound our server makes comes as FLAC (lossless, about half the download of WAV); sound made in this browser as WAV
  // (no download, and written at once).
  return serverAudio ? 'flac' : 'wav';
}

/** The audio format our server is asked for. For a video it is the small one that suits the file (the picture is put back here). */
export function serverAudioFormat(id, file, info) {
  if (FORMAT[id]?.kind === 'audio') return id === 'ogg' ? 'opus' : id;
  return endingOf(file?.name) === 'webm' ? 'opus' : 'm4a';
}

/** Three-letter language codes (ISO 639-2/T) for subtitle tracks, from the site's two-letter ones. */
export const ISO639_2 = {
  en: 'eng', es: 'spa', fr: 'fra', de: 'deu', it: 'ita', pt: 'por', nl: 'nld', pl: 'pol', ru: 'rus', uk: 'ukr', tr: 'tur',
  ar: 'ara', hi: 'hin', ja: 'jpn', ko: 'kor', zh: 'zho', sv: 'swe', da: 'dan', fi: 'fin', no: 'nor', cs: 'ces', el: 'ell',
  he: 'heb', id: 'ind', vi: 'vie', hu: 'hun', ro: 'ron', th: 'tha',
};

/** Can a video file of this type made here carry subtitles that players show? Only MKV: there they are stored the way
 *  nearly every player reads them (SRT-style text). An MP4 made here would hold them in a form most players skip. */
export const canHoldSubtitles = (id) => id === 'mkv';

/** WebVTT cues with only their times and text (no cue names, settings, notes or styles): plain SRT-style cues. */
function plainCues(vtt) {
  const out = ['WEBVTT'];
  for (const block of String(vtt).replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n').split(/\n{2,}/)) {
    const lines = block.split('\n');
    const at = lines.findIndex((l) => l.includes('-->'));
    const times = at >= 0 && lines[at].match(/^\s*(\S+)\s+-->\s+(\S+)/);
    if (!times) continue; // the header, NOTE, STYLE and REGION blocks
    const text = lines.slice(at + 1).join('\n').trim();
    if (text) out.push(`${times[1]} --> ${times[2]}\n${text}`);
  }
  return `${out.join('\n\n')}\n`;
}

/**
 * mediabunny marks an MKV subtitle track as S_TEXT/WEBVTT, which FFmpeg-based players (mpv, Plex, Jellyfin...) don't
 * know. With plain cues the blocks are the same as SRT's, so the track is relabelled S_TEXT/UTF8 (SRT), which they all
 * read: same-length CodecID (EBML strings may end in zero bytes) and its "WEBVTT" CodecPrivate turned into a Void
 * element. Only the exact bytes mediabunny writes are touched; if they are not found the file stays as it was.
 */
function relabelAsSrt(buf, whole = true) {
  const b = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const enc = new TextEncoder();
  const find = (pat, from, to) => {
    for (let i = b.indexOf(pat[0], from); i >= 0 && i <= to - pat.length; i = b.indexOf(pat[0], i + 1)) {
      let k = 1;
      while (k < pat.length && b[i + k] === pat[k]) k++;
      if (k === pat.length) return i;
    }
    return -1;
  };
  const end = whole ? Math.min(b.length, 1 << 20) : b.length; // in a whole file the track list is near the start
  // every subtitle track (one per language)
  for (let id = find([0x86, 0x8d, ...enc.encode('S_TEXT/WEBVTT')], 0, end); id >= 0; id = find([0x86, 0x8d, ...enc.encode('S_TEXT/WEBVTT')], id + 15, end)) {
    b.set([...enc.encode('S_TEXT/UTF8'), 0, 0], id + 2);
    const priv = id + 15;
    if (find([0x63, 0xa2, 0x86, ...enc.encode('WEBVTT')], priv, priv + 9) === priv) b.set([0xec, 0x87, 0, 0, 0, 0, 0, 0, 0], priv);
  }
  return buf;
}

/** SRT text as WebVTT (the same cues; WebVTT writes the milliseconds after a dot). */
export function srtToVtt(srt) {
  const body = String(srt).replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n')
    .replace(/(\d{2}:\d{2}:\d{2}),(\d{3})/g, '$1.$2');
  return `WEBVTT\n\n${body.trim()}\n`;
}

/** WebVTT text as SRT (numbered cues, the milliseconds after a comma; styles, notes and cue settings are left out). */
export function vttToSrt(vtt) {
  const stamp = (t) => {
    const [h, m, rest] = t.split(':').length === 3 ? t.split(':') : ['0', ...t.split(':')];
    return `${h.padStart(2, '0')}:${m.padStart(2, '0')}:${rest.replace('.', ',').padEnd(6, '0')}`;
  };
  const out = [];
  for (const block of String(vtt).replace(/^﻿/, '').replace(/\r\n?/g, '\n').split(/\n{2,}/)) {
    const lines = block.split('\n');
    const at = lines.findIndex((l) => l.includes('-->'));
    const times = at >= 0 && lines[at].match(/^\s*(\S+)\s+-->\s+(\S+)/);
    if (!times) continue;
    const text = lines.slice(at + 1).join('\n').trim();
    if (text) out.push(`${out.length + 1}\n${stamp(times[1])} --> ${stamp(times[2])}\n${text}\n`);
  }
  return out.join('\n');
}

/* ------------------------------------------------------------------------------------------------ making the file */

// Finished files written to the browser's private storage. The newest one stays until the next build (its download
// may still be reading it); older ones, and any left from an earlier visit, are removed.
const BUILD_PREFIX = 'build-';
async function clearOldBuilds(root) {
  try {
    for await (const name of root.keys()) {
      if (name.startsWith(BUILD_PREFIX)) await root.removeEntry(name).catch(() => {});
    }
  } catch { /* nothing to clear */ }
}

/**
 * Where a video file is written while it is made: a file on disk in the browser's private storage when the browser
 * has one (memory use stays small whatever the size), else memory. `needBytes` is a rough size of the result.
 * `relabel`: rewrite the subtitle track's label as the bytes go by (relabelAsSrt).
 * @returns {Promise<{target, onDisk:boolean, result:()=>Promise<Blob>, drop:()=>Promise<void>}>}
 */
async function diskTarget(mb, needBytes, { relabel = false } = {}) {
  if (canBuildOnDisk()) {
    let root = null;
    try {
      root = await navigator.storage.getDirectory();
    } catch { /* private storage is switched off (some private windows): use memory */ }
    if (root) {
      const name = `${BUILD_PREFIX}${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      await clearOldBuilds(root);
      const est = await navigator.storage.estimate?.().catch(() => null);
      if (est?.quota && est.quota - (est.usage || 0) < needBytes * 1.1) {
        throw new RemuxUnsupported(`There isn’t enough free space on this device to make the file (it needs about ${Math.ceil(needBytes / 1024 ** 3)} GB). Free some space and try again.`);
      }
      const handle = await root.getFileHandle(name, { create: true });
      const file = await handle.createWritable();
      const sink = new WritableStream({
        write: (chunk) => file.write({ type: 'write', position: chunk.position, data: relabel ? relabelAsSrt(chunk.data, false) : chunk.data }),
        close: () => file.close(),
        abort: (why) => file.abort(why),
      });
      return {
        target: new mb.StreamTarget(sink, { chunked: true }),
        onDisk: true,
        result: () => handle.getFile(),
        drop: async () => {
          await file.abort().catch(() => {});
          await root.removeEntry(name).catch(() => {});
        },
      };
    }
  }
  const target = new mb.BufferTarget();
  return {
    target,
    onDisk: false,
    result: async () => new Blob([relabel ? relabelAsSrt(target.buffer) : target.buffer]),
    drop: async () => {},
  };
}

const abortError = () => new DOMException('Canceled', 'AbortError');

async function durationOf(audioFile) {
  const { input } = await openInput(audioFile);
  try {
    return (await input.computeDuration()) || 0;
  } finally {
    input.dispose();
  }
}

/** The cleaned sound as a WAV file (16-bit) from whatever we were given. */
async function asWav(mb, source, { signal, onProgress }) {
  if (source.wav) return source.wav;
  if (source.pcm) return buildWav(source.pcm, 16, { onProgress });
  const { input } = await openInput(source.audioFile);
  try {
    const output = new mb.Output({ format: new mb.WavOutputFormat(), target: new mb.BufferTarget() });
    const conv = await mb.Conversion.init({ input, output, tracks: 'primary', video: { discard: true }, audio: { codec: 'pcm-s16' }, showWarnings: false });
    if (!conv.isValid) throw new RemuxUnsupported('This browser can’t open the cleaned sound to change its format.');
    conv.onProgress = (p) => onProgress?.(p);
    signal?.addEventListener('abort', () => conv.cancel(), { once: true });
    await conv.execute();
    if (signal?.aborted) throw abortError();
    return new Blob([output.target.buffer], { type: 'audio/wav' });
  } finally {
    input.dispose();
  }
}

/**
 * Put the cleaned sound into `output` as a track. Resolves {codec, run} once the track exists; call `run()` after
 * output.start() (it resolves when all sound has been written). `codecs`: what may be encoded (this browser can);
 * `copyOk`: what a sound file's own codec may be to be copied as it is (copying needs no encoder).
 */
async function addCleanSound(mb, output, source, codecs, { signal, onProgress, copyOk = codecs }) {
  const format = output.format;
  const allowed = codecs.filter((c) => format.getSupportedAudioCodecs().includes(c));
  if (source.pcm) {
    const quality = new mb.Quality({ bitrate: 192000 });
    const codec = await mb.getFirstEncodableAudioCodec(allowed, { numberOfChannels: source.pcm.channels, sampleRate: source.pcm.sampleRate, quality });
    if (!codec) throw new RemuxUnsupported('This browser can’t write the sound for this kind of file. Try another format.');
    const src = new mb.AudioSampleSource({ codec, quality });
    output.addAudioTrack(src);
    const pcm = source.pcm;
    return {
      codec,
      cancel: () => {},
      run: async () => {
        let t = 0;
        for (let i = 0; i < pcm.parts.length; i++) {
          if (signal?.aborted) throw abortError();
          const data = new Float32Array(await pcm.parts[i].arrayBuffer());
          const sample = new mb.AudioSample({ data, format: 'f32', numberOfChannels: pcm.channels, sampleRate: pcm.sampleRate, timestamp: t });
          t += data.length / pcm.channels / pcm.sampleRate;
          try {
            await src.add(sample);
          } finally {
            sample.close();
          }
          onProgress?.((i + 1) / pcm.parts.length);
        }
        src.close();
      },
    };
  }
  // sound from a file (our server's cleaned audio): copied when the container can hold it, else encoded again
  const { input } = await openInput(source.audioFile);
  const track = await input.getPrimaryAudioTrack();
  if (!track) {
    input.dispose();
    throw new RemuxUnsupported('The cleaned sound could not be read.');
  }
  const own = await track.getCodec();
  let pick;
  if (own && copyOk.includes(own) && format.getSupportedAudioCodecs().includes(own)) {
    pick = undefined;
  } else {
    const rate = await track.getSampleRate();
    const ch = await track.getNumberOfChannels();
    pick = await mb.getFirstEncodableAudioCodec(allowed, { numberOfChannels: ch, sampleRate: rate, quality: new mb.Quality('high') });
    if (!pick) {
      input.dispose();
      throw new RemuxUnsupported('This browser can’t write the sound for this kind of file. Try another format.');
    }
  }
  const init = (opts) => mb.Conversion.init({ input, output, composable: true, tracks: 'primary', video: { discard: true }, showWarnings: false, ...opts });
  let conv = null;
  if (!pick) {
    // Copy it as it is. 'forced' also copies AAC whose first (priming) packet starts a little before zero, which
    // the default mode would encode again.
    conv = await init({ copy: { mode: 'forced' } }).catch(() => null);
    if (conv && (!conv.isValid || conv.utilizedTracks.length === 0)) conv = null;
  }
  if (!conv) conv = await init({ audio: pick ? { codec: pick } : undefined });
  if (!conv.isValid || conv.utilizedTracks.length === 0) {
    input.dispose();
    throw new RemuxUnsupported('This browser can’t put the cleaned sound into this kind of file. Try another format.');
  }
  conv.onProgress = (p) => onProgress?.(p);
  return {
    codec: pick || own,
    cancel: () => conv.cancel(),
    run: async () => {
      try {
        await conv.execute();
      } finally {
        input.dispose();
      }
    },
  };
}

/**
 * Make the finished file.
 * @param {object} p
 * @param {string} p.format one of FORMATS' ids
 * @param {string} [p.size] 'same' | '2160' | ... (video files, with a video source only)
 * @param {File} [p.videoFile] the original video (when the result keeps its picture)
 * @param {object} [p.info] probeMedia() of the original
 * @param {{parts:Blob[], channels:number, sampleRate:number, seconds?:number, frames?:number}} [p.pcm] cleaned sound, free path
 * @param {Blob} [p.wav] the cleaned sound as a 16-bit WAV file, free path
 * @param {Blob} [p.audioFile] the cleaned sound as a file, paid path
 * @param {string} [p.audioName] name of that file (so an identical format is handed back untouched)
 * @param {string|{text:string, lang?:string, name?:string}[]} [p.subtitles] WebVTT text to put into a video file as
 *   a subtitle track (MKV), or several tracks, one per language (players let viewers pick)
 * @param {(frac:number)=>void} [p.onProgress]
 * @param {AbortSignal} [p.signal]
 * @returns {Promise<{blob:Blob, ext:string, mime:string, note:string|null}>}
 */
export async function buildMedia({ format, size = 'same', videoFile = null, info = null, pcm = null, wav = null, audioFile = null, audioName = '', subtitles = '', onProgress, signal }) {
  const def = FORMAT[format];
  if (!def) throw new Error('Unknown format');
  const mb = await loadMediabunny();
  const source = { pcm, wav, audioFile };
  if (!pcm && !wav && !audioFile) throw new Error('No cleaned sound to save.');
  const report = (p) => onProgress?.(Math.max(0, Math.min(1, p)));
  const done = (blob, note = null) => {
    report(1);
    return { blob, ext: `.${def.ext}`, mime: def.mime, note };
  };

  /* ---- audio files ---- */
  if (def.kind === 'audio') {
    // handed back untouched when the server's file already is what was asked for
    if (audioFile && def.exts.includes(endingOf(audioName))) return done(audioFile);
    if (format === 'wav') {
      const w = await asWav(mb, source, { signal, onProgress: (p) => report(p * 0.9) });
      return done(w);
    }
    if (format === 'flac') {
      const w = await asWav(mb, source, { signal, onProgress: (p) => report(p * 0.4) });
      const header = await sniffPcm(w);
      if (!header) throw new RemuxUnsupported('This recording can’t be written as FLAC here. Try WAV.');
      const blob = await encodeFlac(w, header, { signal, onProgress: (p) => report(0.4 + p * 0.6) });
      return done(blob);
    }
    if (format === 'mp3' && !(await ensureMp3())) throw new RemuxUnsupported('This browser can’t write MP3.');
    const output = new mb.Output({ format: makeFormat(mb, format), target: new mb.BufferTarget() });
    const codec = { mp3: 'mp3', ogg: 'opus', m4a: 'aac' }[format];
    const sound = await addCleanSound(mb, output, source, [codec], { signal, onProgress: report });
    signal?.addEventListener('abort', () => sound.cancel(), { once: true });
    try {
      await output.start();
      await sound.run();
      if (signal?.aborted) throw abortError();
      await output.finalize();
    } catch (e) {
      await output.cancel().catch(() => {});
      if (signal?.aborted) throw abortError();
      throw e;
    }
    return done(new Blob([output.target.buffer], { type: def.mime }));
  }

  /* ---- video files ---- */
  const caps = await outputCaps();
  const px = SIZES.find((s) => s.id === size)?.px || 0;
  const withPicture = !!videoFile && !!info?.hasVideo;
  const store = await diskTarget(mb, (withPicture ? videoFile.size : 0) + (audioFile?.size || 0) + 64 * 1024 ** 2, { relabel: !!subtitles && format === 'mkv' });
  if (!store.onDisk && withPicture && videoFile.size > MAX_BUILD_BYTES) {
    throw new RemuxUnsupported('This video is too large for this browser to rebuild (over 1.5 GB). Chrome, Edge or Firefox can do it; or save the cleaned audio and combine it in your video editor.');
  }
  const output = new mb.Output({ format: makeFormat(mb, format), target: store.target });
  const cancels = [];
  let vin = null;
  try {
    const sound = await addCleanSound(mb, output, source, audioCodecsFor(caps, format), { signal, copyOk: AUDIO_PREFS[format] });
    cancels.push(sound.cancel);
    let pictureRun;
    let pictureNote = null;

    if (withPicture) {
      ({ input: vin } = await openInput(videoFile));
      const vTrack = await vin.getPrimaryVideoTrack();
      if (!vTrack) throw new RemuxUnsupported('No video track found in the original file.');
      const srcW = await vTrack.getDisplayWidth();
      const srcH = await vTrack.getDisplayHeight();
      const dims = px ? scaledDims(srcW, srcH, px) : null;
      const plan = await videoPlan({ ...info, videoCodec: await vTrack.getCodec() }, format, px, dims);
      if (!plan.ok) throw new RemuxUnsupported(plan.why || 'This browser can’t build this kind of video.');
      const video = { };
      if (plan.reencode) video.codec = plan.codec;
      if (dims) { video.width = dims.width; video.height = dims.height; video.fit = 'fill'; }
      const vconv = await mb.Conversion.init({
        input: vin, output, composable: true, tracks: 'primary', audio: { discard: true }, video, showWarnings: false,
      });
      if (!vconv.isValid || vconv.utilizedTracks.length === 0) {
        throw new RemuxUnsupported('The picture of this video can’t be put into this kind of file by this browser.');
      }
      vconv.onProgress = (p) => report(0.2 + p * 0.78);
      cancels.push(() => vconv.cancel());
      pictureRun = () => vconv.execute();
      if (plan.reencode) {
        const name = { avc: 'H.264', vp9: 'VP9', vp8: 'VP8' }[plan.codec] || plan.codec;
        pictureNote = `The picture was encoded again (${name}${dims ? `, ${dims.width}×${dims.height}` : ''}).`;
        if (plan.codec === 'vp9' && format !== 'webm') pictureNote += ' Your browser can’t write H.264, so VP9 was used; most newer players handle it.';
      } else {
        pictureNote = 'The picture was copied as it is, without re-encoding.';
      }
    } else {
      // no picture to keep (an audio file): a small black one, a couple of frames a second
      const plan = await videoPlan({ hasVideo: false }, format, 0, { width: 640, height: 360 });
      if (!plan.ok) throw new RemuxUnsupported(plan.why || 'This browser can’t build this kind of video.');
      const seconds = pcm?.seconds || (pcm?.frames && pcm?.sampleRate ? pcm.frames / pcm.sampleRate : 0) || (await durationOf(audioFile || wav));
      const canvas = new OffscreenCanvas(640, 360);
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, 640, 360);
      const src = new mb.CanvasSource(canvas, { codec: plan.codec, quality: new mb.Quality('very-low'), keyFrameInterval: 2 });
      output.addVideoTrack(src, { frameRate: 2 });
      pictureRun = async () => {
        const fps = 2;
        const n = Math.max(1, Math.ceil(seconds * fps));
        for (let i = 0; i < n; i++) {
          if (signal?.aborted) throw abortError();
          await src.add(i / fps, 1 / fps);
          if (i % 200 === 0) report(0.2 + (i / n) * 0.78);
        }
        src.close();
      };
      pictureNote = 'The picture is plain black.';
    }

    let subtitlesRun = async () => {};
    const tracks = !subtitles ? [] : typeof subtitles === 'string' ? [{ text: subtitles }] : subtitles.filter((t) => t?.text);
    if (tracks.length) {
      if (!canHoldSubtitles(format)) throw new RemuxUnsupported(`Subtitles put into a video here need an MKV file (most players can’t show them from a ${def.label} made here). Pick MKV.`);
      const sources = tracks.map((t, i) => {
        const src = new mb.TextSubtitleSource('webvtt');
        const code = ISO639_2[t.lang];
        output.addSubtitleTrack(src, { name: t.name || (tracks.length > 1 ? `Subtitles ${i + 1}` : 'Subtitles'), ...(code ? { languageCode: code } : {}) });
        return [src, t.text];
      });
      subtitlesRun = async () => {
        for (const [src, text] of sources) {
          await src.add(plainCues(text));
          src.close();
        }
      };
    }

    signal?.addEventListener('abort', () => cancels.forEach((c) => c()), { once: true });
    await output.start();
    await Promise.all([pictureRun(), sound.run(), subtitlesRun()]);
    if (signal?.aborted) throw abortError();
    await output.finalize();
    const mkNote = [];
    if (sound.codec === 'opus' && (format === 'mp4' || format === 'mov')) mkNote.push('The sound is stored as Opus in the MP4 file. Modern browsers and VLC play it; some older video editors do not.');
    if (sound.codec === 'pcm-s16') mkNote.push('The sound is stored uncompressed (PCM), so the file is a little bigger.');
    if (pictureNote) mkNote.unshift(pictureNote);
    const made = await store.result();
    return done(made.type === def.mime ? made : new Blob([made], { type: def.mime }), mkNote.join(' ') || null);
  } catch (e) {
    await output.cancel().catch(() => {});
    await store.drop();
    if (signal?.aborted) throw abortError();
    throw e;
  } finally {
    vin?.dispose();
  }
}
