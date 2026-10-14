// Shared helpers for file jobs (Studio clean, transcription): create -> upload -> poll.

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

export const AUDIO_FORMATS = [
  { id: 'wav', label: 'WAV (lossless)' },
  { id: 'flac', label: 'FLAC (lossless, smaller)' },
  { id: 'm4a', label: 'M4A / AAC' },
  { id: 'mp3', label: 'MP3' },
  { id: 'opus', label: 'Opus' },
];

export function statusLabel(status) {
  return {
    uploading: 'Uploading', queued: 'Queued', running: 'Processing', cloud_asr: 'Transcribing (premium)', canceling: 'Canceling',
    done: 'Done', failed: 'Failed', canceled: 'Canceled', expired: 'Expired',
  }[status] || status;
}

export function kindLabel(job) {
  const base = { denoise: 'Studio clean', transcribe: 'Transcription', denoise_transcribe: 'Studio clean + transcription' }[job.kind] || job.kind;
  return job.free ? `${base} (free)` : base;
}

/**
 * Create a job and upload its input. For video inputs the audio track is extracted in the browser first so only
 * the audio is uploaded (falls back to uploading the whole file). Uncompressed WAV/AIFF audio is losslessly encoded to
 * FLAC first (falls back to the original). The declared filename and sizeBytes always describe what is uploaded.
 * @returns {Promise<{id:string, estCostMicros:number, extracted:boolean, compressed:{from:number,to:number}|null,
 *   uploadBytes:number, originalBytes:number, uploadName:string}>}
 */
export async function submitJob({ file, info, body, onStage, onProgress, onCompressed, signal }) {
  let blob = file;
  let name = file.name;
  let extracted = false;
  const options = { ...body.options };
  if (info?.hasVideo && info.viaMediabunny && info.hasAudio !== false) {
    onStage?.('extract');
    try {
      const out = await extractAudio(file, { onProgress: (p) => onProgress?.({ phase: 'extract', frac: p }), signal });
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
