// Put cleaned audio back into the original video (the picture is copied, never re-encoded, unless a different size or
// file type is asked for: see convert.js, which does the work). Everything happens in the browser; the video itself
// never leaves the device.

import { buildMedia, defaultFormatId, outputCaps, RemuxUnsupported } from './convert.js';

export { RemuxUnsupported };

/**
 * @param {object} p
 * @param {File} p.videoFile original video
 * @param {{parts:Blob[], channels:number, sampleRate:number}} [p.pcm] cleaned audio as interleaved float32 parts
 * @param {Blob} [p.audioFile] cleaned audio as a file (m4a/ogg/wav/flac/mp3...)
 * @param {string} [p.format] file type of the result (convert.js FORMATS id); the original's own type when left out
 * @param {string} [p.size] picture size ('same', '1080', ...)
 * @returns {Promise<{blob:Blob, ext:string, mime:string, audioCodec:string|null, note:string|null}>}
 */
export async function remuxWithCleanAudio({ videoFile, pcm, audioFile, audioName, info, format, size, onProgress, signal }) {
  const out = await buildMedia({
    format: format || defaultFormatId(videoFile, { hasVideo: true }, true),
    size: size || 'same',
    videoFile,
    info: info || { hasVideo: true },
    pcm,
    audioFile,
    audioName,
    onProgress,
    signal,
  });
  return { ...out, audioCodec: null };
}

/** Which audio codec this browser can encode for a video container (null: none, only WAV downloads are possible). */
export async function videoAudioEncoder() {
  try {
    const caps = await outputCaps();
    return caps.audio.aac ? 'aac' : caps.audio.opus ? 'opus' : null;
  } catch {
    return null;
  }
}

