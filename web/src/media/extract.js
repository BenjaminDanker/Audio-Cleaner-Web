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

/**
 * @returns {Promise<{blob: Blob, filename: string, codec: string, reencoded: boolean} | null>}
 *   null when extraction is not possible (caller uploads the original file instead).
 */
export async function extractAudio(file, { onProgress, signal } = {}) {
  const mb = await loadMediabunny();
  const { input } = await openInput(file);
  try {
    const track = await input.getPrimaryAudioTrack();
    if (!track) return null;
    const codec = await track.getCodec();
    let plan = planFor(mb, codec);
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
      tracks: 'primary',
      video: { discard: true },
      audio: reencode ? { codec: 'pcm-s16' } : undefined,
      showWarnings: false,
    });
    if (!conv.isValid || conv.utilizedTracks.length === 0) return null;
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
