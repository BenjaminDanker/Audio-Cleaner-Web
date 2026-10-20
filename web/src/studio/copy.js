// The words of the editor: tooltips (the website's own hints, with the prices of the live price list) and the "All the options" table.
// Prices are never written here: they come from the price list (usePricing) and are left out while it is not here.

import { ALTERNATIVES } from '../alternatives.jsx';
import { rate } from '../format.js';
import { findRate } from '../hooks.js';

/** "$0.009 a minute" for a row of the price list, '' when the list doesn't have it. */
export function perMin(pricing, product, tier) {
  const r = findRate(pricing, product, tier);
  return r ? `${rate(r.usdPerMinute)} a minute` : '';
}

const sentence = (t) => (t ? `${t}.` : '');
const words = (...parts) => parts.filter(Boolean).join(' ');
const good = (k) => `Good to know: ${ALTERNATIVES[k]}`;

/** Every tooltip of the editor, by key. */
export function tipsFor(pricing) {
  const noise = perMin(pricing, 'denoise', 'standard');
  const rebuild = perMin(pricing, 'denoise', 'rebuild');
  const echo = perMin(pricing, 'echo', 'standard');
  const level = perMin(pricing, 'leveling', 'standard');
  const std = perMin(pricing, 'transcribe', 'local');
  const ext = perMin(pricing, 'transcribe', 'premium');
  const tr = perMin(pricing, 'transcribe_translate', 'standard');
  return {
    'v.off': 'Leave the voice as it is in this part.',
    'v.noise': 'Takes the background sound out and keeps your voice exactly as recorded; never changes words. Removes most background sound, including game sounds; the gentler strengths keep some room sound. Works on any length of recording.',
    'v.rebuild': `For an echoey room, a muffled voice, phone-quality or damaged sound, or distortion from recording too loud: the whole voice is rebuilt, which also brings back the high sounds that are missing. Keeps only speech, so game sounds, music and room sound are gone. Slower, ${rebuild ? `${rebuild}, ` : ''}files up to 90 minutes. On very echoey recordings it can change a word, so listen before you publish.`,
    'k.free': `Done in your browser, on your own device: nothing is uploaded. Gentle to Strong (no Max), and no choice of remover. The first time, it downloads the noise remover (about 22 MB); how fast it goes depends on your device, and the tab has to stay open. ${good('browserDenoise')}`,
    'k.paid': words('Done on our side. Only the sound is uploaded; a video’s picture stays on your device. All four strengths and a choice of remover.', sentence(noise), good('denoise')),
    's.1': 'Gentle: takes the edge off. Keeps some room sound and music, so it sounds the most natural.',
    's.2': 'Balanced: most of the noise goes; the room still sounds like a room.',
    's.3': 'Strong: nearly all of the noise goes.',
    's.4': 'Max: removes as much as it can. Best for noisy places, and good for most voices.',
    's.4free': 'Max is Standard only.',
    'r.0': 'Auto: we measure how noisy the recording is and pick the best way to clean it.',
    'r.1': 'Light noise: hiss, hum, a quiet room or a fan. Keeps the voice the most natural.',
    'r.2': 'Heavy noise: a café, traffic, a crowd, wind: noise almost as loud as the voice.',
    'r.free': 'The choice of remover is Standard only.',
    echo: `Removes your speaker echo using the file’s second audio track (your desktop sound). OBS can record your mic and your desktop audio on separate tracks: put the mic on track 1 and the desktop sound on track 2. Done on our side${echo ? `, ${echo}` : ''}.`,
    'echo.free': 'This file has a second audio track (your desktop sound). Removing speaker echo with it needs Denoising; the free cleanup in your browser can’t do it.',
    level: words('Done on our side: evens out the volume and sets the loudness, with or without noise removal.', sentence(level), 'Turn it off to keep the voice as it was recorded.'),
    'l.0': 'Leave the volume as it is.',
    'l.1': 'Podcast: evens out the volume at −16 LUFS, the usual loudness for podcasts.',
    'l.2': 'YouTube: evens out the volume at −14 LUFS, the loudness YouTube plays videos at.',
    'l.3': 'Broadcast: evens out the volume at −23 LUFS, the TV and radio standard.',
    'l.4': 'Keep loudness: evens out the volume but keeps the overall loudness as recorded.',
    subs: 'Writes down what is said, in the language you choose: subtitle files (SRT, VTT) and a transcript (TXT).',
    'q.0': 'No subtitles or transcript for this part.',
    'q.1': `Made in your browser, on your own device: nothing is uploaded. Fine for clear speech. The first time, it downloads about 80 MB, and it is slower on long files. No translations. ${good('browserSubs')}`,
    'q.2': words('Done on our side. Good for most recordings, in many languages.', sentence(std), good('standardSubs')),
    'q.3': words('Done on our side, with a speech partner we work with: your audio is sent to them, only for this option. Best for accents, noisy rooms and specialist words. Works in any language.', sentence(ext), good('accurateSubs')),
    lang: 'The language spoken. Detect automatically works it out from the recording.',
    tr: words('Up to 3 more languages, made from the transcript: one set of subtitle files per language.', tr ? `${tr} each.` : '', 'Comes with Standard or Extra accurate.', good('translations')),
    'tr.no': 'Translations come with Standard or Extra accurate subtitles.',
    info: 'Compare all the options',
    res: 'The picture is copied as it is when you keep the same size. Another size is made again on your device, so it takes a little longer; a bigger size than the original only stretches the picture.',
    'sf.srt': 'SRT: subtitles that most video players and editors take.',
    'sf.vtt': 'VTT: subtitles for web players.',
    'sf.txt': 'TXT: the transcript as plain text, without times.',
    burn: 'Put into the video on your device, so nothing extra is uploaded. Players can turn them on and off. They go into an MKV file, the kind players show subtitles from. Video editors usually take the .srt file instead.',
    'burn.server': 'Your browser can’t open this video type, so it’s uploaded whole and we put them in on our side.',
    hear: (name) => `Hear the file with ${name} or without it. The waveform shows the difference.`,
  };
}

/** The prices of the "All the options" table, from the price list ('' where the list has no row). */
export function priceWords(pricing) {
  return {
    noise: perMin(pricing, 'denoise', 'standard'), rebuild: perMin(pricing, 'denoise', 'rebuild'), echo: perMin(pricing, 'echo', 'standard'),
    level: perMin(pricing, 'leveling', 'standard'), std: perMin(pricing, 'transcribe', 'local'), ext: perMin(pricing, 'transcribe', 'premium'),
    tr: perMin(pricing, 'transcribe_translate', 'standard'),
  };
}
