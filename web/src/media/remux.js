// Put cleaned audio back into the original video (video packets are copied, never re-encoded).
// Everything happens in the browser; the video itself never leaves the device.

import { loadMediabunny, openInput } from './mb.js';

export class RemuxUnsupported extends Error {}

const MAX_REMUX_BYTES = 1.5 * 1024 ** 3; // output is assembled in memory

function pickFormat(mb, file, videoCodec) {
  const name = file.name.toLowerCase();
  const prefersWebm = /\.webm$/.test(name);
  const prefersMkv = /\.mkv$/.test(name);
  const candidates = prefersWebm
    ? [() => new mb.WebMOutputFormat(), () => new mb.MkvOutputFormat()]
    : prefersMkv
      ? [() => new mb.MkvOutputFormat()]
      : [() => new mb.Mp4OutputFormat(), () => new mb.MkvOutputFormat()];
  for (const make of candidates) {
    const fmt = make();
    if (!videoCodec || fmt.getSupportedVideoCodecs().includes(videoCodec)) return fmt;
  }
  return new mb.MkvOutputFormat();
}

function describe(fmt, mb) {
  if (fmt instanceof mb.WebMOutputFormat) return { ext: '.webm', mime: 'video/webm' };
  if (fmt instanceof mb.MkvOutputFormat) return { ext: '.mkv', mime: 'video/x-matroska' };
  return { ext: '.mp4', mime: 'video/mp4' };
}

/**
 * @param {object} p
 * @param {File} p.videoFile original video
 * @param {{parts:Blob[], channels:number, sampleRate:number}} [p.pcm] cleaned audio as interleaved float32 parts
 * @param {Blob} [p.audioFile] cleaned audio as a file (m4a/ogg/wav/flac/mp3...)
 * @returns {Promise<{blob:Blob, ext:string, mime:string, audioCodec:string|null, note:string|null}>}
 */
export async function remuxWithCleanAudio({ videoFile, pcm, audioFile, onProgress, signal }) {
  if (videoFile.size > MAX_REMUX_BYTES) {
    throw new RemuxUnsupported('This video is too large to rebuild in the browser. Download the cleaned audio and combine it in your video editor.');
  }
  const mb = await loadMediabunny();
  const { input: vin } = await openInput(videoFile);
  let ain = null;
  const cancels = [];
  try {
    const vTrack = await vin.getPrimaryVideoTrack();
    if (!vTrack) throw new RemuxUnsupported('No video track found in the original file.');
    const videoCodec = await vTrack.getCodec();
    const format = pickFormat(mb, videoFile, videoCodec);
    const { ext, mime } = describe(format, mb);
    const output = new mb.Output({ format, target: new mb.BufferTarget() });

    const vconv = await mb.Conversion.init({
      input: vin, output, composable: true, tracks: 'primary', audio: { discard: true }, showWarnings: false,
    });
    if (!vconv.isValid || vconv.utilizedTracks.length === 0) {
      throw new RemuxUnsupported('The video track of this file cannot be copied into a new file by this browser.');
    }
    vconv.onProgress = (p) => onProgress?.(p * 0.9);
    cancels.push(() => vconv.cancel());

    let audioCodec = null;
    let note = null;
    let feed;
    let aconv = null;
    if (pcm) {
      const wanted = ['aac', 'opus', 'flac', 'vorbis', 'mp3'].filter((c) => format.getSupportedAudioCodecs().includes(c));
      const quality = new mb.Quality({ bitrate: 192000 });
      audioCodec = await mb.getFirstEncodableAudioCodec(wanted, { numberOfChannels: pcm.channels, sampleRate: pcm.sampleRate, quality });
      if (!audioCodec) {
        throw new RemuxUnsupported('This browser cannot encode audio for video files. Download the cleaned WAV instead.');
      }
      if (audioCodec !== 'aac' && format instanceof mb.Mp4OutputFormat) {
        note = 'Audio is stored as Opus in the MP4 file. Modern browsers and VLC play it; some older video editors do not.';
      }
      const source = new mb.AudioSampleSource({ codec: audioCodec, quality });
      output.addAudioTrack(source);
      await output.start();
      feed = (async () => {
        let t = 0;
        for (const part of pcm.parts) {
          if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
          const data = new Float32Array(await part.arrayBuffer());
          const frames = data.length / pcm.channels;
          const sample = new mb.AudioSample({ data, format: 'f32', numberOfChannels: pcm.channels, sampleRate: pcm.sampleRate, timestamp: t });
          t += frames / pcm.sampleRate;
          await source.add(sample);
          sample.close();
        }
        source.close();
      })();
    } else {
      ({ input: ain } = await openInput(audioFile));
      aconv = await mb.Conversion.init({
        input: ain, output, composable: true, tracks: 'primary', video: { discard: true }, showWarnings: false,
      });
      if (!aconv.isValid || aconv.utilizedTracks.length === 0) {
        throw new RemuxUnsupported('This browser cannot put the cleaned audio into the video. Use the audio download instead.');
      }
      cancels.push(() => aconv.cancel());
      await output.start();
      feed = Promise.resolve();
      audioCodec = (await (await ain.getPrimaryAudioTrack())?.getCodec()) || null;
    }

    if (signal) signal.addEventListener('abort', () => cancels.forEach((c) => c()), { once: true });
    await Promise.all([vconv.execute(), aconv ? aconv.execute() : null, feed]);
    await output.finalize();
    onProgress?.(1);
    return { blob: new Blob([output.target.buffer], { type: mime }), ext, mime, audioCodec, note };
  } catch (e) {
    if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
    throw e;
  } finally {
    vin.dispose();
    ain?.dispose();
  }
}

/** Which audio codec this browser can encode for a video container (null: none, only WAV downloads are possible). */
export async function videoAudioEncoder() {
  try {
    const mb = await loadMediabunny();
    return await mb.getFirstEncodableAudioCodec(['aac', 'opus'], { numberOfChannels: 2, sampleRate: 48000, quality: new mb.Quality({ bitrate: 192000 }) });
  } catch {
    return null;
  }
}
