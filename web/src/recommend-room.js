// "Recommended for your room" (Live, "Test my room") and the numbers both recommendations share: turns the measurements of
// src/media/analysis.js into settings and the reasons for them, in plain words (the copy rules of tests/copy.mjs apply).
//
// The numbers below were calibrated against what the server's noise removers actually achieve (docs/research/recommend-2026-10.md):
// the noise figure is the noise remover's own (voice against what it takes out: pipeline.noise_figure_db), "much noise" is only
// ever said where pipeline.choose_noise_profile would pick the stronger remover, and the strength is the gentlest one that removes
// about as much noise as the strongest. The OBS plugin (obs-plugin/core/src/input_advice.cpp) uses the same numbers for a room:
// tests/reco-unit.mjs checks that they agree. Files are decided by the trained model instead (src/recommend.js).
// Kept apart from recommend.js so the Live page doesn't load the model.
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
  /** the speech stops at this frequency or lower: a phone call or a very small file */
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

export const STANDARD_LANGUAGES = new Set('en bg hr cs da nl et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk'.split(' '));

/** The gentlest strength id that still does the job at this noise figure. */
export function strengthFor(noiseDb) {
  if (noiseDb < T.MAX_BELOW) return 'max';
  if (noiseDb < T.STRONG_BELOW) return 'strong';
  if (noiseDb < T.BALANCED_BELOW) return 'balanced';
  return 'gentle';
}

export const strengthName = (id) => (STRENGTHS.find((s) => s.id === id) || STRENGTHS[3]).label;

/** The noise-related verdict shared by the file and live recommendations. */
export function noiseVerdict(stats) {
  const nd = stats.noiseDb;
  const clean = nd >= T.CLEAN_AT_LEAST;
  const heavy = nd < HEAVY_BELOW_DB - T.HEAVY_MARGIN_DB;
  return { clean, heavy, strength: clean ? null : strengthFor(nd), nd };
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
