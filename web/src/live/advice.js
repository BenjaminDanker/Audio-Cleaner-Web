// "Recommended for your room" for the live test: listens to the first seconds of the microphone (the frames the capture
// worklet already delivers) and, once, about ten seconds in, reports what it heard: onAdvice({stats, echo}). Reports again
// when more has been heard (at 20 s), then stops costing anything.
import { Analyser, EchoProbe } from '../media/analysis.js';

const FIRST_AT = 10; // seconds
const LAST_AT = 20;

export class LiveAdvisor {
  /** @param {{onAdvice: (found: {stats: object, echo: object|null, final: boolean}) => void, withEcho?: boolean}} opts */
  constructor({ onAdvice, withEcho = true }) {
    this.onAdvice = onAdvice;
    this.a = new Analyser({ channels: 1 });
    // The sound that went to the speakers is our own cleaned version of the microphone, so the first 80 ms are skipped: the echo comes
    // back later than that (processing, playback buffer, speakers), the voice itself does not.
    this.echo = withEcho ? new EchoProbe({ minLagMs: 80, seconds: LAST_AT }) : null;
    this.seconds = 0;
    this.stage = 0;
  }

  /** A frame from the microphone (48 kHz, any length). Call for every frame while live. */
  pushMic(f32) {
    if (this.stage >= 2) return;
    this.a.push([f32]);
    this.echo?.pushMic(f32);
    this.seconds += f32.length / 48000;
    if (this.stage === 0 && this.seconds >= FIRST_AT) this._report(1);
    else if (this.stage === 1 && this.seconds >= LAST_AT) this._report(2);
  }

  /** What went to the speakers for the frame pushed last: the cleaned sound, or `null` plus its length when that is not known apart
   *  from the microphone (nothing played, or the microphone itself). */
  pushPlayed(f32, silentLength = 0) {
    if (this.stage < 2) this.echo?.pushRef(f32, silentLength);
  }

  _report(stage) {
    this.stage = stage;
    try {
      this.onAdvice({ stats: this.a.result(), echo: this.echo ? this.echo.result() : null, final: stage === 2 });
    } catch { /* advice is a nicety: never let it disturb the audio */ }
  }
}
