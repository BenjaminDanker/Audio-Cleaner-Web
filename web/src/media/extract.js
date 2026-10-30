// Pull the audio track out of a video so only the audio has to be uploaded.
// The encoded audio packets are copied into a small container without re-encoding whenever possible.

import { loadMediabunny, openInput } from './mb.js';

function planFor(mb, codec) {
  switch (codec) {
    case 'aac': return { format: () => new mb.Mp4OutputFormat(), ext: '.m4a', mime: 'audio/mp4' };
    case 'opus':
    case 'vorbis': return { format: () => new mb.OggOutputFormat(), ext: '.ogg', mime: 'audio/ogg' };
    case 'mp3': return { format: () => new mb.Mp3OutputFormat(), ext: '.mp3', mime: 'audio/mpeg' };
    case 'flac': return { format: () => new mb.FlacOutputFormat(), ext: '.flac', mime: 'audio/flac' };
    default:
      if (codec && codec.startsWith('pcm-')) return { format: () => new mb.WavOutputFormat(), ext: '.wav', mime: 'audio/wav' };
      return null;
  }
}

// Codecs a Matroska audio file can hold as they are; any other audio track is decoded to 16-bit PCM.
const MKA_CODECS = new Set(['aac', 'opus', 'vorbis', 'mp3', 'flac']);
const needsDecode = (codec) => !MKA_CODECS.has(codec) && !(codec && codec.startsWith('pcm-'));

/**
 * `tracks: 2` keeps the first two audio tracks (mic and desktop sound, for echo removal) in one Matroska audio file.
 * Runs in a worker (extract.worker.js): mediabunny copies an hour of packets in one go without giving the page a turn, which held the
 * whole page still for seconds. Where the worker can't start, it runs here instead.
 * @returns {Promise<{blob: Blob, filename: string, codec: string, reencoded: boolean} | null>}
 *   null when extraction is not possible (caller uploads the original file instead).
 */
export function extractAudio(file, { tracks = 1, onProgress, signal } = {}) {
  let worker;
  try {
    worker = new Worker(new URL('./extract.worker.js', import.meta.url), { type: 'module' });
  } catch {
    return extractAudioHere(file, { tracks, onProgress, signal });
  }
  return new Promise((resolve, reject) => {
    const end = () => { signal?.removeEventListener('abort', stop); worker.terminate(); };
    function stop() { end(); reject(new DOMException('Canceled', 'AbortError')); }
    if (signal?.aborted) { stop(); return; }
    signal?.addEventListener('abort', stop, { once: true });
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') { onProgress?.(m.frac); return; }
      end();
      if (m.type === 'done') resolve(m.result);
      else reject(new Error(m.message || 'Could not pull the sound out of the video.'));
    };
    // the worker itself could not load (an old browser): do it here
    worker.onerror = (e) => { e.preventDefault?.(); end(); resolve(extractAudioHere(file, { tracks, onProgress, signal })); };
    worker.postMessage({ file, tracks });
  });
}

/** extractAudio's work itself, wherever it runs (the worker, or the page as a fallback). */
export async function extractAudioHere(file, { tracks = 1, onProgress, signal } = {}) {
  const mb = await loadMediabunny();
  const { input } = await openInput(file);
  try {
    const track = await input.getPrimaryAudioTrack();
    if (!track) return null;
    const codec = await track.getCodec();
    const multi = tracks > 1;
    if (multi && (await input.getAudioTracks()).length < tracks) return null;
    let plan = multi ? { format: () => new mb.MkvOutputFormat(), ext: '.mka', mime: 'audio/x-matroska' } : planFor(mb, codec);
    let reencode = false;
    if (!plan) {
      // exotic codec (AC-3, DTS, ...): decode to 16-bit PCM if this browser can
      if (!(await track.canDecode())) return null;
      plan = { format: () => new mb.WavOutputFormat(), ext: '.wav', mime: 'audio/wav' };
      reencode = true;
    }
    const output = new mb.Output({ format: plan.format(), target: new mb.BufferTarget() });
    const conv = await mb.Conversion.init({
      input,
      output,
      tracks: multi ? 'all' : 'primary',
      video: { discard: true },
      audio: multi
        ? async (t, n) => (n > tracks ? { discard: true } : needsDecode(await t.getCodec()) ? { codec: 'pcm-s16' } : undefined)
        : reencode ? { codec: 'pcm-s16' } : undefined,
      showWarnings: false,
    });
    if (!conv.isValid || conv.utilizedTracks.length < tracks) return null; // both tracks or nothing: one would be no use
    conv.onProgress = (p) => onProgress?.(p);
    if (signal) signal.addEventListener('abort', () => conv.cancel(), { once: true });
    await conv.execute();
    const blob = new Blob([output.target.buffer], { type: plan.mime });
    const base = file.name.replace(/\.[^./\\]+$/, '') || 'audio';
    return { blob, filename: `${base}${plan.ext}`, codec, reencoded: reencode };
  } finally {
    input.dispose();
  }
}
