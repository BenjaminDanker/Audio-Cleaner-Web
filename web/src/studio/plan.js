// What the editor's choices mean for the files: which files the result has, whether the subtitles can go into the video, and what stops a
// start (the messages are the Cleanup form's, word for word). Pure functions of the tracks and the output choice.

import { balanceMoney, money as siteMoney, stem } from '../format.js';
import { tooManyText } from '../jobs.js';
import { canHoldSubtitles, defaultFormatId } from '../media/convert.js';
import { anyFreeNoise, anyFreeSubs, anyPaidSubs, anySubs, changesSound, hasWork, needsServer, serverSound } from '../editor/tracks.js';

export const MAX_SECONDS = 6 * 3600; // the most the browser will take (Free noise removal and subtitles)
export const LONG_SECONDS = 2 * 3600; // in the browser, a recording this long takes a while
/** The most a single upload can be, as the server's default allows it (the price list may say otherwise: maxUploadBytes). */
export const DEFAULT_MAX_UPLOAD_BYTES = 4 * 1024 ** 3;
/** A file that cannot hold any sound: nothing to read, nothing to clean. */
export const TINY_FILE_BYTES = 100;

export const gigabytes = (n) => `${Math.round((n / 1024 ** 3) * 10) / 10} GB`;
/** "Files must be under 4096 MB." -> "Files must be under 4 GB." */
export const inGigabytes = (text) => text.replace(/\b(\d{4,})\s*MB\b/g, (_, n) => gigabytes(Number(n) * 1024 ** 2));

export const supported = typeof Worker !== 'undefined' && typeof WebAssembly !== 'undefined';

/** Can this browser do the Free subtitles for this file itself? `why`: the plain words when it can't. */
export function subsVerdict(info, ok = supported) {
  const paid = 'Standard can (paid).';
  if (!ok) return { ok: false, why: `This browser can’t run the subtitle maker. ${paid}` };
  if (info.hasAudio === false) return { ok: false, why: 'This file has no audio track.' };
  const readable = info.canDecode !== false && (info.viaMediabunny || info.viaLibav || info.duration != null);
  if (!readable) return { ok: false, why: `Your browser can’t open this file type. ${paid}` };
  if (info.duration && info.duration > MAX_SECONDS) return { ok: false, why: `This is longer than ${MAX_SECONDS / 3600} hours, which is too long for your browser. ${paid}` };
  return { ok: true, why: '' };
}

/** Can this browser do the Free noise removal for this file? */
export function browserVerdict(info, { ok = supported, wavFits = null } = {}) {
  const paid = 'Paid Denoising can do it.';
  if (!ok) return { ok: false, why: `This browser can’t run the noise remover. ${paid}` };
  if (info.hasAudio === false) return { ok: false, why: 'This file has no audio track.' };
  const readable = info.canDecode !== false && (info.viaMediabunny || info.viaLibav || info.duration != null);
  if (!readable) return { ok: false, why: `Your browser can’t open this file type. ${paid}` };
  if (info.duration && info.duration > MAX_SECONDS) return { ok: false, why: `This is longer than ${MAX_SECONDS / 3600} hours, which is too long for your browser. ${paid}` };
  // The cleaned sound is offered as a WAV file, and a WAV file can't hold more than about 4 GB: say so now, not after hours of work.
  if (info.duration && wavFits && !wavFits(info.duration, info.channels >= 2 ? 2 : 1, 16)) return { ok: false, why: `The cleaned sound would be too big for one WAV file (4 GB). ${paid}` };
  return { ok: true, why: '' };
}

/**
 * The result files as the choices say: a cleaned sound (as `choice`'s type and name) when the sound is changed, the subtitle files, and the
 * subtitles inside the video. `limit`: the biggest video this browser can put together (media/convert.js buildLimit; null: any).
 */
export function outputPlan({ tracks, info, file, choice, out, limit = null }) {
  const cleaning = changesSound(tracks);
  const wantsText = anySubs(tracks);
  const serverText = anyPaidSubs(tracks);
  const isVideo = !!info?.hasVideo;
  // Subtitles in the video: put in here, in the browser, once the job is done (nothing extra is uploaded). Only a video the browser can't
  // open goes up whole, and then our side puts them in. A recording with no picture can get them in the plain black video made here when
  // a video file type is picked.
  const videoResult = isVideo ? !cleaning || !!choice?.isVideoFile : cleaning && !!choice?.isVideoFile;
  const embedOnServer = serverText && isVideo && !info?.viaMediabunny;
  const tooBigHere = isVideo && !embedOnServer && !!file && limit != null && file.size > limit;
  const canEmbed = wantsText && videoResult && !tooBigHere && (serverText || !isVideo || !!info?.viaMediabunny);
  const embedNow = canEmbed && !!out?.burn;
  const embedHere = embedNow && !embedOnServer;
  // A subtitles-only video keeps its own file type when that can carry them (WebM), else it becomes MKV.
  const embedFormat = cleaning ? choice?.format : isVideo ? defaultFormatId(file, info, true) : null;
  return { cleaning, wantsText, serverText, isVideo, videoResult, embedOnServer, tooBigHere, canEmbed, embedNow, embedHere, embedFormat };
}

/** What kind of job our side runs for these tracks (docs/api.md section 8: a job made with parts is `denoise`, its subtitles are parts too), or null when nothing needs it. */
export function jobKind(tracks) {
  return needsServer(tracks) ? 'denoise' : null;
}

/** The choices as the Finish step wants them (finish.js `wants`). `local`: no job on our side; the sound and subtitles are made here. */
export function wantsOf({ tracks, plan, choice, out, file, fileName = '', local }) {
  const w = {
    sound: plan.cleaning, format: choice.format, name: choice.name.trim(), size: choice.size,
    subs: plan.wantsText, caps: { srt: out.sfiles.includes('srt'), vtt: out.sfiles.includes('vtt'), txt: out.sfiles.includes('txt') },
    embed: plan.embedNow, soundHere: local && anyFreeNoise(tracks), subsHere: local && anyFreeSubs(tracks),
  };
  // subtitles into the video with no cleaning: the video keeps its own sound, as an MKV
  if (!plan.cleaning && plan.embedNow) Object.assign(w, { format: 'mkv', size: 'same', name: `${stem(fileName || file?.name)}-subtitled.mkv` });
  return w;
}

/** The messages that stop a start, in order (each {text, kind?}: 'credit' adds the link to billing, 'prices' the Try again button). */
export function problemsFor({
  info, file, tracks, est, estReady = true, duration, maxJob, pricing, pricingError, balance, active, maxActive, plan, choice, out,
  partsError = '', noParts = false,
}) {
  const problems = [];
  const server = needsServer(tracks);
  if (info?.hasAudio === false) problems.push({ text: 'This file has no audio track.' });
  if (server && duration && maxJob && duration > maxJob) problems.push({ text: `Files can be at most ${Math.floor(maxJob / 3600)} hours long.` });
  const maxUpload = pricing?.maxUploadBytes || DEFAULT_MAX_UPLOAD_BYTES;
  // A video's picture is not uploaded and uncompressed WAV/AIFF is packed smaller first; anything else goes up as it is.
  const canShrink = (info?.hasVideo && info?.viaMediabunny) || /^pcm/.test(info?.audioCodec || '');
  if (server && file && file.size > maxUpload && !canShrink) {
    problems.push({ text: `This file is ${gigabytes(file.size)}. The most we can take for upload is ${gigabytes(maxUpload)}. Cleaning in your browser doesn’t upload the file, so this limit doesn’t apply there.` });
  }
  if (server && maxActive && active >= maxActive) problems.push({ text: tooManyText(active) });
  if (partsError) problems.push({ text: partsError });
  if (server && estReady && est == null && duration && pricingError) problems.push({ text: 'We couldn’t load the prices just now, so this can’t start yet.', kind: 'prices' });
  if (server && est != null && balance && est > balance.available_micros) {
    problems.push({ text: `This costs about ${siteMoney(est)}, but your balance is ${balanceMoney(balance.available_micros)}. You can also choose the free options.`, kind: 'credit' });
  }
  if (info && !hasWork(tracks)) problems.push({ text: 'Choose at least one thing to do: noise removal, subtitles, or both.' });
  if (plan.wantsText && !out.sfiles.length) problems.push({ text: 'Choose at least one caption or transcript format.' });
  if (plan.embedHere && plan.cleaning && !canHoldSubtitles(choice.format)) {
    problems.push({ text: `Subtitles put into the video need an MKV file (most players can’t show them from ${choice.format.toUpperCase()}). Pick MKV, or untick “Put the subtitles into the video”.` });
  }
  if (noParts) problems.push({ text: 'Choose at least one part to process, or switch back to the whole file.' });
  return problems;
}

export { anyFreeNoise, anyFreeSubs, anyPaidSubs, anySubs, changesSound, hasWork, needsServer, serverSound };
