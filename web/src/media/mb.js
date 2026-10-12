// Lazy mediabunny loader + probing helpers. mediabunny (MPL-2.0) is code-split: it is only downloaded
// when someone actually picks a file.

let mbPromise = null;

export function loadMediabunny() {
  if (!mbPromise) mbPromise = import('mediabunny');
  return mbPromise;
}

export async function openInput(file) {
  const mb = await loadMediabunny();
  const input = new mb.Input({
    source: new mb.BlobSource(file),
    formats: [mb.MP4, mb.QTFF, mb.MATROSKA, mb.WEBM, mb.MP3, mb.WAVE, mb.OGG, mb.FLAC, mb.ADTS, mb.MPEG_TS],
  });
  return { mb, input };
}

/** Duration (seconds) via a media element's metadata. Works for anything the browser can play. */
export function elementDuration(file, timeoutMs = 8000) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const isVideo = /^video\//.test(file.type) || /\.(mp4|m4v|mov|webm|mkv)$/i.test(file.name);
    const el = document.createElement(isVideo ? 'video' : 'audio');
    el.preload = 'metadata';
    const done = (v) => {
      clearTimeout(timer);
      el.removeAttribute('src');
      el.load();
      URL.revokeObjectURL(url);
      resolve(v);
    };
    const timer = setTimeout(() => done(null), timeoutMs);
    el.onloadedmetadata = () => done(Number.isFinite(el.duration) ? el.duration : null);
    el.onerror = () => done(null);
    el.src = url;
  });
}

/**
 * Inspect a file. Never throws; unknown fields are null/false.
 * @returns {Promise<{duration:number|null, hasVideo:boolean, hasAudio:boolean|null, audioCodec:string|null,
 *   videoCodec:string|null, channels:number|null, sampleRate:number|null, canDecode:boolean|null, viaMediabunny:boolean}>}
 */
export async function probeMedia(file) {
  const info = {
    duration: null, hasVideo: false, hasAudio: null, audioCodec: null, videoCodec: null,
    channels: null, sampleRate: null, canDecode: null, viaMediabunny: false,
  };
  try {
    const { input } = await openInput(file);
    const [vTrack, aTrack] = await Promise.all([input.getPrimaryVideoTrack(), input.getPrimaryAudioTrack()]);
    info.viaMediabunny = true;
    info.hasVideo = !!vTrack;
    info.hasAudio = !!aTrack;
    if (vTrack) info.videoCodec = await vTrack.getCodec();
    if (aTrack) {
      info.audioCodec = await aTrack.getCodec();
      info.channels = await aTrack.getNumberOfChannels();
      info.sampleRate = await aTrack.getSampleRate();
      info.canDecode = await aTrack.canDecode();
    }
    let d = null;
    try {
      d = await input.getDurationFromMetadata();
    } catch { /* fall through */ }
    if (!d) {
      try {
        d = await input.computeDuration();
      } catch { /* fall through */ }
    }
    info.duration = d && Number.isFinite(d) ? d : null;
    input.dispose();
  } catch {
    /* not a container mediabunny understands; fall back to the browser */
  }
  if (!info.duration) info.duration = await elementDuration(file);
  if (info.duration == null && !info.viaMediabunny) info.hasAudio = null;
  return info;
}
