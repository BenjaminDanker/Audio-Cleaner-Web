// "Good to know" notes: when something free you already have does the job just as well, we say so. A small
// community tool should send you to the right thing, even when that isn't us. Checked October 2026 (YouTube and
// Twitch help pages, OBS's own docs, Apple/Microsoft/Google help on built-in live captions); keep them honest and short, and update them when the platforms change.

export const ALTERNATIVES = {
  browserDenoise: 'Just a little hum? Your editor’s or OBS’s own noise filter may be enough.',
  deviceCaptions: 'Only reading along yourself? Windows, Mac, Chrome, Android and iPhone have free Live Captions on your screen.',
  denoise: 'Light, steady noise from a phone recording? YouTube Create’s free Audio cleanup may be enough.',
  rebuild: 'Only a bit noisy? Plain Denoising is enough; rebuilding is for muffled, thin or echoey voices.',
  browserSubs: 'English video for YouTube? Its free automatic captions are fine for a first pass.',
  standardSubs: 'Rough English transcript of a YouTube video? YouTube’s free automatic captions may be enough.',
  accurateSubs: 'Clear speech? Standard is plenty; Extra accurate is for accents, noise and names that must be right.',
  translations: 'On YouTube, viewers can turn on its free auto-translate.',
  liveDenoise: 'NVIDIA RTX card or a Mac? NVIDIA Broadcast or the Mac’s Voice Isolation are free alternatives.',
  liveCaptions: 'Streaming on YouTube in English? Its own live captions are free (1,000+ subscribers).',
  liveAccurate: 'Clear English? Standard captions are usually enough.',
  liveTranslations: 'Twitch and YouTube don’t translate live captions themselves. Translations are made by software and can be wrong.',
};

/** A small "good to know" line under a choice. */
export function GoodToKnow({ k, testId }) {
  const text = ALTERNATIVES[k];
  if (!text) return null;
  return <span className="hint good-to-know" data-testid={testId || `gtk-${k}`}><b>Good to know:</b> {text}</span>;
}
