// Shared helpers for file jobs (Clean & polish, subtitles & transcripts): create -> upload -> poll.

import { api, uploadBlob } from './api.js';
import { extractAudio } from './media/extract.js';
import { compressForUpload } from './media/flac.js';

export const ACTIVE = new Set(['uploading', 'queued', 'running', 'cloud_asr', 'canceling']);

export const LANGUAGES = [
  ['', 'Detect automatically'],
  ['en', 'English'], ['es', 'Spanish'], ['fr', 'French'], ['de', 'German'], ['it', 'Italian'], ['pt', 'Portuguese'],
  ['nl', 'Dutch'], ['pl', 'Polish'], ['ru', 'Russian'], ['uk', 'Ukrainian'], ['tr', 'Turkish'], ['ar', 'Arabic'],
  ['hi', 'Hindi'], ['ja', 'Japanese'], ['ko', 'Korean'], ['zh', 'Chinese'], ['sv', 'Swedish'], ['da', 'Danish'],
  ['fi', 'Finnish'], ['no', 'Norwegian'], ['cs', 'Czech'], ['el', 'Greek'], ['he', 'Hebrew'], ['id', 'Indonesian'],
  ['vi', 'Vietnamese'],
];

export const LOUDNESS = [
  { id: '-16', label: 'Podcast (−16 LUFS)', value: -16 },
  { id: '-14', label: 'YouTube (−14 LUFS)', value: -14 },
  { id: '-23', label: 'Broadcast (−23 LUFS)', value: -23 },
  { id: 'keep', label: 'Keep original loudness', value: null },
];

// Background noise in the recording: "auto" measures it and picks the noise remover that suits it.
export const NOISE = [
  { id: 'auto', label: 'Automatic', hint: 'We measure how noisy the recording is and pick the best way to clean it.' },
  { id: 'light', label: 'Light noise', hint: 'Hiss, hum, a quiet room or a fan. Keeps the voice the most natural.' },
  { id: 'heavy', label: 'Heavy noise', hint: 'A café, traffic, a crowd, wind: noise almost as loud as the voice.' },
];

export const AUDIO_FORMATS = [
  { id: 'wav', label: 'WAV (lossless)' },
  { id: 'flac', label: 'FLAC (lossless, smaller)' },
  { id: 'm4a', label: 'M4A / AAC' },
  { id: 'mp3', label: 'MP3' },
  { id: 'opus', label: 'Opus' },
];

export function statusLabel(status) {
  return {
    uploading: 'Uploading', queued: 'In line', running: 'Working on it', cloud_asr: 'Transcribing (extra accurate)', canceling: 'Canceling',
    done: 'Done', failed: 'Failed', canceled: 'Canceled', expired: 'Expired',
  }[status] || status;
}

/** "5 min", "under a minute", "1 h 20 min" for a rough wait. */
function roughWait(seconds) {
  const m = Math.round(seconds / 60);
  if (m < 1) return 'under a minute';
  if (m < 90) return `${m} min`;
  const h = Math.floor(m / 60);
  return `${h} h ${String(m % 60).padStart(2, '0')} min`;
}

/**
 * Where a queued file job stands: "You're next", "In line: 2 ahead of you, about 5 min". Nothing is ever refused for being
 * busy; jobs simply wait in line. `queuePosition` is 1-based and null unless the job is queued, `queueWaitSeconds` a rough estimate.
 */
export function queueText(job) {
  if (job.status !== 'queued' || !Number.isFinite(job.queuePosition) || job.queuePosition < 1) return '';
  const wait = Number.isFinite(job.queueWaitSeconds) && job.queueWaitSeconds > 0 ? `, about ${roughWait(job.queueWaitSeconds)}` : '';
  const ahead = job.queuePosition - 1;
  if (ahead === 0) return `You’re next${wait}`;
  return `In line: ${ahead} ahead of you${wait}`;
}

export function kindLabel(job) {
  const base = { denoise: 'Clean & polish', transcribe: 'Subtitles & transcript', denoise_transcribe: 'Clean & polish + subtitles & transcript' }[job.kind] || job.kind;
  return base;
}

/**
 * Create a job and upload its input. For video inputs the audio track is extracted in the browser first so only
 * the audio is uploaded (falls back to uploading the whole file). With options.echo_reference_track the first two
 * audio tracks are kept (a file with only one has nothing to extract and goes up whole). Uncompressed WAV/AIFF audio is losslessly encoded to
 * FLAC first (falls back to the original). The declared filename and sizeBytes always describe what is uploaded.
 * @returns {Promise<{id:string, estCostMicros:number, extracted:boolean, compressed:{from:number,to:number}|null,
 *   uploadBytes:number, originalBytes:number, uploadName:string}>}
 */
export async function submitJob({ file, info, body, onStage, onProgress, onCompressed, signal }) {
  let blob = file;
  let name = file.name;
  let extracted = false;
  const options = { ...body.options };
  const wantsEmbed = !!(options.embed_subtitles && info?.hasVideo);
  if (wantsEmbed) options.output_video = true; // the whole video goes up so the server can put the subtitles in it
  if (!wantsEmbed && info?.hasVideo && info.viaMediabunny && info.hasAudio !== false) {
    onStage?.('extract');
    try {
      // Echo removal needs the desktop sound too: keep the first two audio tracks, not just the first.
      const out = await extractAudio(file, { tracks: options.echo_reference_track ? 2 : 1, onProgress: (p) => onProgress?.({ phase: 'extract', frac: p }), signal });
      if (out && out.blob.size < file.size * 0.9) {
        blob = out.blob;
        name = out.filename;
        extracted = true;
        options.output_video = false;
      }
    } catch (e) {
      if (e?.name === 'AbortError' || signal?.aborted) throw e;
      // fall back to uploading the whole file
    }
  }
  let compressed = null;
  if (!info?.hasVideo || extracted) {
    try {
      const out = await compressForUpload(blob, {
        name, signal,
        onStart: () => onStage?.('compress'),
        onProgress: (p) => onProgress?.({ phase: 'compress', frac: p }),
      });
      if (out) {
        compressed = { from: blob.size, to: out.bytesOut };
        onCompressed?.(compressed);
        blob = out.blob;
        name = out.filename;
      }
    } catch (e) {
      if (e?.name === 'AbortError' || signal?.aborted) throw e;
      // keep the uncompressed file
    }
  }
  if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
  onStage?.('create');
  const created = await api('jobs', {
    method: 'POST',
    body: { ...body, filename: name, sizeBytes: blob.size, durationSeconds: Math.max(0.5, body.durationSeconds), options },
  });
  onStage?.('upload');
  try {
    await uploadBlob(created.uploadUrl, blob, { signal, onProgress: (loaded, total) => onProgress?.({ phase: 'upload', loaded, total, frac: loaded / total }) });
  } catch (e) {
    api(`jobs/${created.id}/cancel`, { method: 'POST', redirectOn401: false }).catch(() => {});
    throw e;
  }
  return { id: created.id, estCostMicros: created.estCostMicros, extracted, compressed, uploadBytes: blob.size, originalBytes: file.size, uploadName: name };
}
