// "Recommended for this file" and "Recommended for your room": turns the measurements of src/media/analysis.js into choices
// and the reasons for them, in plain words (the copy rules of tests/copy.mjs apply: no model names, no jargon).
//
// The numbers below were calibrated against what the server actually does (docs/research/recommend-2026-10.md): the noise
// figure is the one pipeline.noise_estimate_db gives, "much noise" is only ever said where pipeline.choose_noise_profile would
// pick the stronger remover, and the strength is the gentlest one that removes about as much noise as the strongest.
// The OBS plugin (obs-plugin/core/src/input_advice.cpp) uses the same numbers: tests/reco-unit.mjs checks that they agree.
import { HEAVY_BELOW_DB } from './media/analysis.js';
import { STRENGTHS } from './strengths.js';

export const T = {
  /** noise figure (dB between the voice and the background) from which the sound counts as clean: nothing to remove */
  CLEAN_AT_LEAST: 29,
  /** only call it "much noise" when the server's own estimate would pick the stronger remover, with this much to spare */
  HEAVY_MARGIN_DB: 1.5,
  /** strength by noise figure: Maximum below the first number, then Strong, Balanced, and Gentle from the last (see strengthFor) */
  MAX_BELOW: 18,
  STRONG_BELOW: 21,
  BALANCED_BELOW: 27,
  /** below this noise figure the standard subtitles make clearly more mistakes */
  SUBS_NOISY_BELOW: 18,
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
  /** the microphone is mostly the speakers' sound: share of its energy that the playback explains */
  ECHO_SHARE: 0.03,
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
 *   noise: 'off'|'browser'|'polish', strength: string, heavy: boolean, restoreHighs: boolean, rebuild: boolean,
 *   subs: 'standard'|'accurate'|null, nothing: boolean, noSpeech: boolean, reasons: string[], line: string }}
 */
export function recommendForFile(stats, ctx = {}) {
  if (!stats || !(stats.seconds >= 2)) return null;
  if (!stats.hasSpeech) {
    return {
      noise: 'off', strength: 'balanced', heavy: false, restoreHighs: false, rebuild: false, subs: null, nothing: true, noSpeech: true,
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
  if (!v.clean) noise = v.heavy || phone || rebuild ? 'polish' : 'browser';
  else if (phone || rebuild) noise = 'polish';
  const strength = v.strength || 'balanced';

  if (v.clean) reasons.push('It’s clean and quiet, so noise removal would change very little.');
  else if (v.heavy) reasons.push('There is a lot of background noise, nearly as loud as the voice. Clean & polish uses a stronger noise remover for this, and it does clearly better than the free one.');
  else reasons.push(`There is some steady background noise. ${noise === 'browser' ? 'The free cleanup in your browser handles it well.' : 'Clean & polish will take it out.'}`);
  if (!v.clean) reasons.push(`${strengthName(strength)} is enough to take it out${strength === 'max' || strength === 'strong' ? '' : ' while keeping the voice natural'}.`);
  if (phone) reasons.push('It sounds like a phone call or a very small file: the high sounds are missing. Restoring them makes the voice clearer.');
  if (echoey) reasons.push('The voice sounds echoey, as if it was recorded in a big or bare room. Rebuilding the voice can take that out. Listen to the result before you publish.');
  else if (dull) reasons.push('The voice sounds muffled. Rebuilding the voice can bring the clarity back. Listen to the result before you publish.');
  if (stats.clipFraction > T.CLIP_FRACTION) reasons.push('The sound distorts in places because it was recorded too loud. Cleaning can’t fix that, so turn the microphone level down next time.');
  if (quiet) reasons.push('It is very quiet. Clean & polish would also bring it up to a normal level.');
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
  const bits = [];
  if (nothing) bits.push('nothing needed');
  else {
    if (noise === 'off') bits.push('no noise removal');
    else bits.push(`${noise === 'browser' ? 'free cleanup in your browser' : 'Clean & polish'}, ${strengthName(strength)}`);
    if (rebuild) bits.push('rebuild the voice');
    else if (phone) bits.push('restore the highs');
  }
  bits.push(`${subs === 'accurate' ? 'Extra accurate' : 'Standard'} subtitles if you want them`);
  return { noise, strength, heavy: v.heavy && !v.clean, restoreHighs: phone, rebuild, subs, nothing, noSpeech: false, reasons, line: bits.join('; ') };
}

/**
 * The same for a live microphone, about ten seconds in: noise removal (or none, in a quiet room), whether the speakers are heard in
 * the microphone, and the captions level, in one short line.
 * @param {object} stats Analyser.result() of the microphone
 * @param {{explained: number, known?: boolean}|null} echo EchoProbe.result(): the speakers' sound compared with the microphone
 * @param {{mode?: 'free'|'paid', denoise?: boolean, strength?: string, captions?: string, language?: string, final?: boolean}} ctx what is set now
 *   (final: nothing more will be heard, so there is no point asking for more speech)
 * @returns {null | {pending: boolean, denoise: boolean, strength: string, echo: boolean, captions: 'local'|'premium', notes: string[], line: string}}
 */
export function recommendForLive(stats, echo = null, ctx = {}) {
  if (!stats || !(stats.seconds >= 5)) return null;
  if (!stats.hasSpeech) {
    if (ctx.final) return null;
    const line = 'Say a few words and we’ll suggest the best settings for your room.';
    return { pending: true, denoise: true, strength: 'balanced', echo: false, captions: 'local', notes: [line], line };
  }
  const v = noiseVerdict(stats);
  const notes = [];
  if (v.clean) {
    notes.push(ctx.denoise === false ? 'Your room is quiet, so you don’t need noise removal.' : 'Your room is quiet: you can leave noise removal off.');
  } else if (ctx.denoise === false) {
    notes.push(`Your room is noisy: turn on noise removal (${strengthName(v.strength)} suits it).`);
  } else if (ctx.strength && ctx.strength !== v.strength) {
    notes.push(`For your room, ${strengthName(v.strength)} noise removal suits it best.`);
  }
  const heardSpeakers = !!echo && echo.known !== false && echo.explained >= T.ECHO_SHARE;
  if (heardSpeakers) notes.push('Your speakers can be heard in your microphone: wear headphones, or set “You hear” to Nothing.');
  if (stats.clipFraction > T.CLIP_FRACTION) notes.push('Your microphone is too loud and distorts: turn it down.');
  const lang = ctx.language || '';
  const captions = stats.noiseDb < T.SUBS_NOISY_BELOW || (lang && !STANDARD_LANGUAGES.has(lang)) ? 'premium' : 'local';
  if (ctx.mode === 'paid' && ctx.captions) {
    if (ctx.captions === 'local' && captions === 'premium') notes.push('Extra accurate captions will catch more words here.');
    else if (ctx.captions === 'premium' && captions === 'local') notes.push('Standard captions are enough for your clear voice.');
  }
  if (!notes.length) notes.push('The settings fit your room.');
  return { pending: false, denoise: !v.clean, strength: v.strength || 'balanced', echo: heardSpeakers, captions, notes, line: notes.join(' ') };
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
  const r = { heardSpeech: !!stats?.hasSpeech, settings: { ...now }, changes: [], notes: [] };
  if (!r.heardSpeech) return r;
  const v = noiseVerdict(stats);
  if (v.clean) {
    r.settings.denoise = false;
    if (now.denoise) r.changes.push({ key: 'denoise', text: 'Noise removal: on → off' });
  } else {
    r.settings.denoise = true;
    r.settings.strength = v.strength;
    if (!now.denoise) r.changes.push({ key: 'denoise', text: `Noise removal: off → on, ${strengthName(v.strength)}` });
    else if (now.strength !== v.strength) {
      r.changes.push({ key: 'strength', text: `Noise removal strength: ${strengthName(now.strength)} → ${strengthName(v.strength)}` });
    }
  }
  if (now.captions) {
    const lang = now.language || '';
    const want = stats.noiseDb < T.SUBS_NOISY_BELOW || (lang && !STANDARD_LANGUAGES.has(lang)) ? 'premium' : 'local';
    if (want !== now.captions) {
      r.settings.captions = want;
      r.changes.push({ key: 'captions', text: `Captions: ${CAPTION_NAME[now.captions]} → ${CAPTION_NAME[want]}` });
    }
  }
  if (stats.clipFraction > T.CLIP_FRACTION) r.notes.push('Your microphone is too loud and distorts: turn it down.');
  return r;
}
