// What plays in the editor, all with one small face: play(), pause(), seek(t), time, playing, close(). Before anything is made (and for the
// original) the file's own media element plays; for a job on our side it is the mix of the layers the speaker buttons leave on
// (editor/mix.js MixPlayer); for a run made here the two signals it kept (media/blend.js BlendPlayer), or the cleaned file next to the
// original. The picture of a video is a muted element that follows the sound.

/** A media element (<video> or <audio>) as an engine. */
export class ElementEngine {
  constructor(el, { onEnd } = {}) {
    this.el = el;
    this.onEnd = onEnd;
    this.ended = () => { this.onEnd?.(); };
    el.addEventListener('ended', this.ended);
  }

  get time() { return this.el.currentTime || 0; }

  get playing() { return !this.el.paused && !this.el.ended; }

  get duration() { return this.el.duration || 0; }

  async play() { this.el.muted = false; await this.el.play(); }

  pause() { this.el.pause(); }

  seek(t) { this.el.currentTime = Math.max(0, t); return Promise.resolve(); }

  close() { this.el.removeEventListener('ended', this.ended); this.el.pause(); }
}

/** The cleaned file next to the original: `hearCleaned` picks which one is heard; the original element also shows the picture. */
export class AbEngine {
  constructor(original, cleaned, { onEnd } = {}) {
    this.a = original;
    this.b = cleaned;
    this.hearCleaned = true;
    this.onEnd = onEnd;
    this.ended = () => { this.onEnd?.(); };
    cleaned.addEventListener('ended', this.ended);
    this.apply();
  }

  apply() {
    this.a.muted = this.hearCleaned;
    this.b.muted = !this.hearCleaned;
  }

  setHearCleaned(on) { this.hearCleaned = on; this.apply(); }

  get time() { return this.b.currentTime || 0; }

  get playing() { return !this.b.paused && !this.b.ended; }

  get duration() { return this.b.duration || 0; }

  async play() {
    this.apply();
    this.a.currentTime = this.b.currentTime;
    await Promise.all([this.b.play(), this.a.play()]);
  }

  pause() { this.a.pause(); this.b.pause(); }

  seek(t) { this.a.currentTime = t; this.b.currentTime = t; return Promise.resolve(); }

  close() { this.cleaned?.removeEventListener('ended', this.ended); this.b.removeEventListener('ended', this.ended); this.pause(); }
}

/** Keep a muted element (the picture) at the engine's time: started, stopped and sought with it, and pulled back when it drifts. */
export function syncPicture(el, engine, playing) {
  if (!el || !engine) return;
  const t = engine.time;
  if (playing && el.paused) el.play().catch(() => {});
  if (!playing && !el.paused) el.pause();
  if (Math.abs(el.currentTime - t) > 0.6) el.currentTime = t;
}
