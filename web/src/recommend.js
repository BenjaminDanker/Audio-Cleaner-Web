// "Recommended for this file" and "Recommended for your room": turns the measurements of src/media/analysis.js into choices
// and the reasons for them, in plain words (the copy rules of tests/copy.mjs apply: no model names, no jargon).
//
// The numbers below were calibrated against what the server's noise removers actually achieve (docs/research/recommend-2026-10.md):
// the noise figure is the noise remover's own (voice against what it takes out: pipeline.noise_figure_db), "much noise" is only
// ever said where pipeline.choose_noise_profile would pick the stronger remover, and the strength is the gentlest one that removes
// about as much noise as the strongest. The OBS plugin (obs-plugin/core/src/input_advice.cpp) uses the same numbers:
// tests/reco-unit.mjs checks that they agree.
import { HEAVY_BELOW_DB } from './media/analysis.js';
import { STRENGTHS } from './strengths.js';

export const T = {
  /** noise figure (dB between the voice and what the noise remover takes out) from which the sound counts as clean */
  CLEAN_AT_LEAST: 24.5,
  /** only call it "much noise" when the server's own figure would pick the stronger remover, with this much to spare */
  HEAVY_MARGIN_DB: 1,
  /** strength by noise figure: Maximum below the first number, then Strong, Balanced, and Gentle from the last (see strengthFor) */
  MAX_BELOW: 15,
  STRONG_BELOW: 20,
  BALANCED_BELOW: 24,
  /** below this noise figure the standard subtitles make clearly more mistakes */
  SUBS_NOISY_BELOW: 4,
  /** the speech stops at this frequency or lower: a phone call or a very small file (bwe.pick_model would use its 8k or 12k generator) */
  PHONE_HZ: 6600,
  /** the level can't fall faster than this within 60 ms (dB): the voice rings on, as in a large or bare room (analysis.js: dropOf) */
  ECHO_DROP_DB: 10.5,
  /** the top of the speech band sits this far below its middle (dB) or more: a muffled voice without a sharp cut-off */
  DULL_TILT_DB: -19,
  /** quiet or loud enough that the loudness setting is worth a note (LUFS) */
  QUIET_LUFS: -32,
  LOUD_LUFS: -10,
  CLIP_FRACTION: 0.002,
};

const STANDARD_LANGUAGES = new Set('en bg hr cs da nl et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk'.split(' '));

/** The gentlest strength id that still does the job at this noise figure. */
export function strengthFor(noiseDb) {
  if (noiseDb < T.MAX_BELOW) return 'max';
  if (noiseDb < T.STRONG_BELOW) return 'strong';
  if (noiseDb < T.BALANCED_BELOW) return 'balanced';
  return 'gentle';
}

const strengthName = (id) => (STRENGTHS.find((s) => s.id === id) || STRENGTHS[3]).label;

/** The noise-related verdict shared by the file and live recommendations. */
function noiseVerdict(stats) {
  const nd = stats.noiseDb;
  const clean = nd >= T.CLEAN_AT_LEAST;
  const heavy = nd < HEAVY_BELOW_DB - T.HEAVY_MARGIN_DB;
  return { clean, heavy, strength: clean ? null : strengthFor(nd), nd };
}

/**
 * @param {object} stats Analyser.result()
 * @param {{language?: string}} ctx language: what is picked for the subtitles ('' = detect automatically)
 * @returns {null | {
 *   noise: 'off'|'browser'|'denoise', strength: string, heavy: boolean, restoreHighs: boolean, rebuild: boolean, leveling: boolean,
 *   subs: 'standard'|'accurate'|null, nothing: boolean, noSpeech: boolean, reasons: string[], line: string }}
 */
export function recommendForFile(stats, ctx = {}) {
  if (!stats || !(stats.seconds >= 2) || stats.noiseDb == null) return null;
  if (!stats.hasSpeech) {
    return {
      noise: 'off', strength: 'balanced', heavy: false, restoreHighs: false, rebuild: false, leveling: false, subs: null, nothing: true, noSpeech: true,
      reasons: ['We could hardly hear anyone speaking in this file, so there is nothing to clean up or write down.'],
      line: 'nothing needed, we could hardly hear anyone speaking',
    };
  }
  const reasons = [];
  const v = noiseVerdict(stats);
  const phone = stats.bandwidthHz <= T.PHONE_HZ;
  const echoey = stats.dropDb != null && stats.dropDb < T.ECHO_DROP_DB;
  const dull = !phone && stats.tiltDb != null && stats.tiltDb <= T.DULL_TILT_DB;
  const rebuild = echoey || dull;
  const quiet = stats.loudnessLufs != null && stats.loudnessLufs < T.QUIET_LUFS;
  const loud = stats.loudnessLufs != null && stats.loudnessLufs > T.LOUD_LUFS;
  let noise = 'off';
  if (!v.clean) noise = v.heavy || phone || rebuild ? 'denoise' : 'browser';
  else if (phone || rebuild) noise = 'denoise';
  const strength = v.strength || 'balanced';

  if (v.clean) reasons.push('It’s clean and quiet, so noise removal would change very little.');
  else if (v.heavy) reasons.push('There is a lot of background noise, nearly as loud as the voice. Denoising uses a stronger noise remover for this, and it does clearly better than the free one.');
  else reasons.push(`There is some background noise. ${noise === 'browser' ? 'The free cleanup in your browser handles it well.' : 'Denoising will take it out.'}`);
  if (!v.clean) reasons.push(`${strengthName(strength)} is enough to take it out${strength === 'max' || strength === 'strong' ? '' : ' while keeping the voice natural'}.`);
  if (phone) reasons.push('It sounds like a phone call or a very small file: the high sounds are missing. Restoring them makes the voice clearer.');
  if (echoey) reasons.push('The voice sounds echoey, as if it was recorded in a big or bare room. Rebuilding the voice can take that out. Listen to the result before you publish.');
  else if (dull) reasons.push('The voice sounds muffled. Rebuilding the voice can bring the clarity back. Listen to the result before you publish.');
  if (stats.clipFraction > T.CLIP_FRACTION) reasons.push('The sound distorts in places because it was recorded too loud. Cleaning can’t fix that, so turn the microphone level down next time.');
  if (quiet) reasons.push('It is very quiet. Denoising would also bring it up to a normal level.');
  if (loud) reasons.push('It is already very loud.');
  if (stats.layout === 'one-side') reasons.push('Only one side of this stereo file has sound; the other side stays silent.');

  // subtitles: which kind to use if you want them
  const lang = ctx.language || '';
  const notStandard = !!lang && !STANDARD_LANGUAGES.has(lang);
  const busy = stats.noiseDb < T.SUBS_NOISY_BELOW;
  const subs = busy || notStandard ? 'accurate' : 'standard';
  if (notStandard) reasons.push('For subtitles in this language, Extra accurate gets more words right.');
  else if (busy) reasons.push('For subtitles, Extra accurate is the safer choice: the noisy background would make Standard miss words.');
  else reasons.push('For subtitles, Standard is plenty: the speech is clear.');

  const nothing = noise === 'off' && !rebuild && !phone && !quiet;
  // Leveling is done on our side, next to Denoising or on its own: it is set where the sound needs it (and with Denoising, where it is cheap).
  const leveling = noise === 'denoise' || (noise === 'off' && (quiet || loud));
  const bits = [];
  if (nothing) bits.push('nothing needed');
  else {
    if (noise === 'off') bits.push('no noise removal');
    else bits.push(`${noise === 'browser' ? 'free cleanup in your browser' : 'Denoising'} (${strengthName(strength)})`);
    if (rebuild) bits.push('rebuild the voice');
    else if (phone) bits.push('restore the highs');
  }
  if (leveling) bits.push('leveling');
  bits.push(`${subs === 'accurate' ? 'Extra accurate' : 'Standard'} subtitles`);
  return { noise, strength, heavy: v.heavy && !v.clean, restoreHighs: phone, rebuild, leveling, subs, nothing, noSpeech: false, reasons, line: bits.join(', ') };
}

/**
 * The live settings that fit a room ("Test my room": twenty seconds of the microphone). Noise removal stays on however quiet the
 * room sounded: twenty seconds don't show the keyboard, the game, the dog or the door of a whole stream, so a quiet room gets the
 * gentlest strength instead of none. The OBS plugin decides the same way (obs-plugin/core/src/input_advice.cpp).
 * @param {object} stats Analyser.result() of the microphone
 * @param {string} language the spoken language ('' = not chosen)
 * @returns {{strength: string, captions: 'local'|'premium'}}
 */
export function liveSettingsFor(stats, language = '') {
  const v = noiseVerdict(stats);
  return {
    strength: v.clean ? 'gentle' : v.strength,
    captions: stats.noiseDb < T.SUBS_NOISY_BELOW || (language && !STANDARD_LANGUAGES.has(language)) ? 'premium' : 'local',
  };
}

const CAPTION_NAME = { local: 'Standard', premium: 'Extra accurate' };

/**
 * "Test my room": the settings that fit what the microphone heard, as changes to `now`.
 * @param {object} stats Analyser.result() of the room test
 * @param {{denoise: boolean, strength: string, captions: ''|'local'|'premium', language?: string}} now what is set (captions '' = off, or
 *   the free test, where the captions aren't a choice of level)
 * @returns {{heardSpeech: boolean, settings: object, changes: {key: string, text: string}[], notes: string[]}}
 */
export function roomChanges(stats, now) {
  const r = { heardSpeech: !!stats?.hasSpeech, settings: { ...now, denoise: true }, changes: [], notes: [] };
  if (!r.heardSpeech) return { ...r, settings: { ...now } };
  const fit = liveSettingsFor(stats, now.language || '');
  r.settings.strength = fit.strength;
  if (!now.denoise) r.changes.push({ key: 'denoise', text: `Noise removal: off → on, ${strengthName(fit.strength)}` });
  else if (now.strength !== fit.strength) {
    r.changes.push({ key: 'strength', text: `Noise removal strength: ${strengthName(now.strength)} → ${strengthName(fit.strength)}` });
  }
  if (now.captions && fit.captions !== now.captions) {
    r.settings.captions = fit.captions;
    r.changes.push({ key: 'captions', text: `Captions: ${CAPTION_NAME[now.captions]} → ${CAPTION_NAME[fit.captions]}` });
  }
  if (stats.clipFraction > T.CLIP_FRACTION) r.notes.push('Your microphone is too loud and distorts: turn it down.');
  return r;
}
