// "Good to know" notes: when something free you already have does the job just as well, we say so. A small
// community tool should send you to the right thing, even when that isn't us. Checked October 2026 (YouTube and
// Twitch help pages, OBS's own docs, Apple/Microsoft/Google help on built-in live captions); keep them honest and short, and update them when the platforms change.

export const ALTERNATIVES = {
  browserDenoise: 'Only a little fan or room hum? Your video editor’s or OBS’s own noise filter may be enough. This is for when those still leave hiss or noise behind.',
  deviceCaptions: 'Just want to read along yourself? Live Captions are built into Windows 11, Macs with Apple silicon, Chrome, Android and iPhone: free, on your device, and often about as accurate for clear English. Chrome can even translate them. They’re only on your screen, though, not in your stream, video or files for everyone.',
  polish: 'Steady background noise and recording on your phone? YouTube Create’s free Audio cleanup can handle that. Clean & polish is for stubborn noise, uneven voices and loudness that’s ready to publish.',
  rebuild: 'Only a bit noisy? Try plain Clean & polish (or the free cleanup) first. Rebuilding is for voices that are muffled, thin or echoey.',
  browserSubs: 'Posting to YouTube in English? Its automatic captions are free and fine for a first pass. These are for a file you can fix up and keep, or for other sites and apps.',
  standardSubs: 'Only need a rough transcript of an English YouTube video? YouTube’s automatic captions will do it for free. These make fewer mistakes, get names and punctuation right more often, and you keep the files.',
  accurateSubs: 'Clear speech that nobody will read closely? Standard is plenty. Extra accurate is for interviews, strong accents, noisy rooms, mixed languages, or anything that has to be right.',
  translations: 'For YouTube uploads, viewers can switch on YouTube’s free auto-translate, and YouTube can dub English videos into about 20 languages. These are for subtitles you can check yourself, or sites that don’t translate for you.',
  liveDenoise: 'Have an NVIDIA RTX card? NVIDIA Broadcast is free and good. On a Mac, the Voice Isolation mic mode (Control Center, while OBS uses the mic) is free too, though it can swallow music and sound effects. OBS’s own Noise Suppression is fine for mild noise. This one is for other hardware, or when those still leave noise, without using your GPU.',
  liveCaptions: 'Streaming on YouTube in English with 1,000+ subscribers? YouTube’s own live captions are free. Want free and on your own PC? The LocalVocal OBS plugin does that. These work on Twitch and everywhere else, with no setup and no load on your PC.',
  liveAccurate: 'Speaking clear English? Standard captions are usually enough. Extra accurate is for fast talk, noisy rooms, other languages, or when names have to be right.',
  liveTranslations: 'Twitch and YouTube don’t translate live captions themselves (as far as we could find), so free routes are a plugin on your own PC or your viewers’ browser tools. These are ready to go without setup. Translations are made by software and can be wrong.',
};

/** A small "good to know" line under a choice. */
export function GoodToKnow({ k, testId }) {
  const text = ALTERNATIVES[k];
  if (!text) return null;
  return <span className="hint good-to-know" data-testid={testId || `gtk-${k}`}><b>Good to know:</b> {text}</span>;
}
