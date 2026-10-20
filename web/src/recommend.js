// "Recommended for this file": turns the measurements of src/media/analysis.js into choices and the reasons for them, in plain
// words (the copy rules of tests/copy.mjs apply: no model names, no jargon).
//
// The noise strength and the voice rebuild come from the trained model (src/reco-model.js), which learned them from
// what actually came out of the noise remover and the voice rebuild. Voice rebuild is also chosen by the measurements alone
// where the voice sounds echoey or muffled, like a phone call, or distorts from being recorded too loud (more often than the
// model would): it re-creates the whole voice, which fixes all of these and sounds fuller and clearer nearly every time.
// The heavy noise remover stays the server's own rule. The live recommendations ("Test my room") are in src/recommend-room.js.
import { modelDecision } from './reco-model.js';
import { noiseVerdict, STANDARD_LANGUAGES, strengthName, T } from './recommend-room.js';

export { liveSettingsFor, roomChanges, strengthFor, T } from './recommend-room.js';

/**
 * @param {object} stats Analyser.result()
 * @param {{language?: string}} ctx language: what is picked for the subtitles ('' = detect automatically)
 * @returns {null | {
 *   noise: 'off'|'browser'|'denoise', strength: string, heavy: boolean, rebuild: boolean, leveling: boolean,
 *   subs: 'standard'|'accurate'|null, nothing: boolean, noSpeech: boolean, reasons: string[], pick: string, line: string }}
 */
export function recommendForFile(stats, ctx = {}) {
  if (!stats || !(stats.seconds >= 2) || stats.noiseDb == null) return null;
  if (!stats.hasSpeech) {
    return {
      noise: 'denoise', strength: 'balanced', heavy: false, rebuild: false, leveling: true, subs: null, nothing: false, noSpeech: true,
      reasons: ['We could hardly hear anyone speaking in this file, so there is nothing to write down: no subtitles.'],
      line: 'Denoising (Balanced), leveling, no subtitles (we could hardly hear anyone speaking)',
    };
  }
  const reasons = [];
  const v = noiseVerdict(stats);
  const m = modelDecision(stats);
  const phone = stats.bandwidthHz <= T.PHONE_HZ;
  const echoey = stats.dropDb != null && stats.dropDb < T.ECHO_DROP_DB;
  const dull = !phone && stats.tiltDb != null && stats.tiltDb <= T.DULL_TILT_DB;
  const clipped = stats.clipFraction > T.CLIP_FRACTION;
  const rebuild = echoey || dull || phone || clipped || m.rebuild;
  const quiet = stats.loudnessLufs != null && stats.loudnessLufs < T.QUIET_LUFS;
  const loud = stats.loudnessLufs != null && stats.loudnessLufs > T.LOUD_LUFS;
  // Automatic best settings choose among the paid options, the best one for every feature: Denoising (or Voice rebuild
  // when the voice needs rebuilding) at the strength the noise needs, Leveling, the subtitle kind that suits the speech.
  // The loudness target stays the default (YouTube).
  const noise = 'denoise';
  const strength = m.strength;
  const quietAndGentle = v.clean && strength === 'gentle';

  // which of the two ways to clean the voice, and why, in the terms of "Which one should I pick?" (components/WhichOne.jsx)
  const heard = [];
  if (echoey) heard.push('the voice sounds echoey');
  if (dull) heard.push('the voice sounds muffled');
  if (phone) heard.push('it sounds like a phone call or a very small file');
  if (clipped) heard.push('the sound distorts from being recorded too loud');
  if (rebuild && !heard.length) heard.push('the voice has lost some of its clarity');
  const pick = rebuild
    ? `Voice rebuild: ${heard.length > 1 ? `${heard.slice(0, -1).join(', ')} and ${heard[heard.length - 1]}` : heard[0]}.`
    : 'Denoising: the voice is already clear, so it is kept exactly as recorded and only the background sound is taken out.';
  reasons.push(pick);
  if (quietAndGentle) reasons.push('It’s clean and quiet: Denoising takes out what little there is, gently.');
  else if (v.heavy) reasons.push('There is a lot of background noise, nearly as loud as the voice: Denoising uses its stronger noise remover for this.');
  else reasons.push('There is some background noise: Denoising takes it out.');
  const soft = strength === 'gentle' || strength === 'balanced';
  if (quietAndGentle) { /* said above */ } else if (v.heavy) reasons.push(`With it, ${strengthName(strength)} is enough${soft ? ' and keeps the voice natural' : ''}.`);
  else reasons.push(`${strengthName(strength)} is enough to take it out${soft ? ' while keeping the voice natural' : ''}.`);
  if (rebuild) {
    // every problem heard, then the one fix: Voice rebuild re-creates the whole voice
    if (echoey) reasons.push('The voice sounds echoey, as if it was recorded in a big or bare room.');
    if (dull) reasons.push('The voice sounds muffled.');
    if (phone) reasons.push('It sounds like a phone call or a very small file: the high sounds are missing.');
    if (clipped) reasons.push('The sound distorts in places because it was recorded too loud.');
    if (!echoey && !dull && !phone && !clipped) reasons.push('The voice has lost some of its clarity, from echo or from being saved as a very small file.');
    reasons.push('Rebuilding the voice fixes this. Listen to the result before you publish.');
    if (clipped) reasons.push('Next time, turn the microphone level down.');
  }
  if (quiet) reasons.push('It is very quiet: Leveling brings it up to a normal level.');
  if (loud) reasons.push('It is very loud: Leveling brings it down to a normal level.');
  if (stats.layout === 'one-side') reasons.push('Only one side of this stereo file has sound; the other side stays silent.');

  // subtitles: the paid kind that suits the speech
  const lang = ctx.language || '';
  const notStandard = !!lang && !STANDARD_LANGUAGES.has(lang);
  const busy = stats.noiseDb < T.SUBS_NOISY_BELOW;
  const subs = busy || notStandard ? 'accurate' : 'standard';
  if (notStandard) reasons.push('For subtitles in this language, Extra accurate gets more words right.');
  else if (busy) reasons.push('For subtitles, Extra accurate: the noisy background would make Standard miss words.');
  else reasons.push('For subtitles, Standard: the speech is clear.');

  const leveling = true;
  const nothing = false;
  const bits = [rebuild ? 'Voice rebuild' : `Denoising (${strengthName(strength)})`];
  bits.push('leveling', `${subs === 'accurate' ? 'Extra accurate' : 'Standard'} subtitles`);
  return { noise, strength, heavy: v.heavy && !v.clean, rebuild, leveling, subs, nothing, noSpeech: false, reasons, pick, line: bits.join(', ') };
}

