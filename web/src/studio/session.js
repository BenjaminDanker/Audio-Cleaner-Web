// One recording in the editor, from the moment its file is chosen until its files are gone. It holds what the person is making (the tracks,
// the suggestions, undo), what the browser knows about the file (its waveform, what the quick listen found), and does the work that is not
// drawing: the quick listen and the suggestions, starting a run (our side and/or this browser), saving changes, the browser's share of a job.
// The page (components/studio/Studio.jsx) draws from it and calls it; it survives moving between /app and /app/job/<id> in one tab.

import { api } from '../api.js';
import { addLocalCard, beginFinish } from '../finish.js';
import { estimateTimeline, loadEditor, loadLayerPeaks, needsOf, quoteTimeline, saveTimeline, uploadBrowserCues, uploadBrowserNoise } from '../editor/api.js';
import { filePeaks, layerPeaks, overlayPeaks, rangePeaks } from '../editor/peaks.js';
import { loadSuggestions, suggestionState } from '../editor/suggest.js';
import {
  applyReco, autoDone, freeNoiseRanges, freeSubRanges, fromTimeline, makeState, needsServer, pushUndo,
  recoPatches, redo as redoState, sameTracks, setSel, snapJ, splitAll, toSuggestion, toTimeline, undo as undoState, vkey, freeOnlySuggestion } from '../editor/tracks.js';
import { bytes } from '../format.js';
import { submitJob } from '../jobs.js';
import { buildLimit, serverAudioFormat, serverSize } from '../media/convert.js';
import { analyseFile } from '../media/analyse.js';
import { preloadDenoiser } from '../media/denoise.js';
import { probeMedia } from '../media/mb.js';
import { denoiseFile } from '../media/pipeline.js';
import { previewFits } from '../media/blend.js';
import { preloadSpeech, subtitleFiles } from '../media/speech.js';
import { startSuggestions, suggestVerdict } from '../media/suggest-run.js';
import { buildWav } from '../media/wav.js';
import { recommendForFile } from '../recommend.js';
import { newLocalJobId } from '../localid.js';
import { browserVerdict, inGigabytes, jobKind, outputPlan, subsVerdict, supported, TINY_FILE_BYTES, videoOutOf, wantsOf } from './plan.js';
import { cleanRange, fitPcm, joinSegments, pcmToFlac, samplesOf, splicePcm, transcribeRange } from './runs.js';

const AUTO_KEY = 'acw.autoSettings'; // this browser's choice for "Automatic" (on unless switched off)
export const storedAuto = () => { try { return localStorage.getItem(AUTO_KEY) !== 'off'; } catch { return true; } };
const saveAuto = (on) => { try { localStorage.setItem(AUTO_KEY, on ? 'on' : 'off'); } catch { /* not remembered: private window or storage blocked */ } };

const sessions = new Map(); // job id -> Session
export const sessionFor = (id) => sessions.get(id) || null;
export const forgetSession = (id) => { const s = sessions.get(id); if (s) { s.dispose(); sessions.delete(id); } };

/** Start fetching the noise remover (about 22 MB, once, low priority) once the free in-browser cleanup is on the table for a file. */
export function warmUpDenoiser() {
  if (!supported || navigator.connection?.saveData) return;
  preloadDenoiser().catch(() => {});
}
/** ... and the subtitle maker (about 80 MB, once). */
export function warmUpSpeech() {
  if (!supported || navigator.connection?.saveData) return;
  preloadSpeech().catch(() => {});
}

const compressedNote = (c) => `Compressed for upload: ${bytes(c.from)} → ${bytes(c.to)}`;
/** "Got it, we're on it." plus what was done to the file before it went up (shown once, at the top of the job's page). */
export function startedNote(out) {
  if (!out) return 'Got it.';
  return `Got it, we’re on it.${out.extracted ? ` Only the sound was uploaded (${bytes(out.uploadBytes)} instead of ${bytes(out.originalBytes)}).` : ''}${out.compressed ? ` ${compressedNote(out.compressed)}.` : ''}`;
}

const clone = (o) => JSON.parse(JSON.stringify(o));
let draftN = 0;

export class Session {
  constructor() {
    this.id = `draft-${++draftN}`;
    this.v = 0;
    this.listeners = new Set();
    this.file = null;
    this.info = null;
    this.fileName = '';
    this.T = 0;
    this.S = null;
    this.lengthUnknown = false;
    this.out = { sfiles: ['srt'], burn: false }; // (the file type, name and picture size are the Output popover's own state)
    this.wave = { state: 'reading', data: null }; // 'reading' | 'ok' | 'none'
    this.layers = null; // the server's peaks per layer, once the job is done: {source, echo, noise, rebuild}
    this.auto = storedAuto();
    this.stats = null;
    this.listening = false;
    this.applied = false;
    this.undoTo = null;
    this.sg = { phase: 'idle', progress: 0, error: '', why: '' }; // finding moments worth a look in the browser
    this.job = null; // the job as the server lists it
    this.fin = null;
    this.snap = null; // what the files hold now: {tracks, choice, sfiles, burn}
    this.server = null; // loadEditor(): {base, rendered, cached, preview, secondTrack, ...} once the job is done
    this.serverError = '';
    this.loading = false; // a job's tracks are being fetched
    this.stage = 'ready'; // 'ready' | 'processing' (before there is a job) ; a job decides it after that
    this.run = null; // {tasks: {name: {text, frac, eta}}, cancel: bool, saving: bool}
    this.error = '';
    this.errorKind = '';
    this.welcome = '';
    this.env = {}; // set by the page on every draw: pricing, rates, balance, ...
    this.abort = null;
    this.listenCtl = null;
    this.sgTask = null;
    this.sgN = 0;
    this.paid = 0;
    this.pairKept = false;
    this.browserPeaks = []; // the peaks of the ranges this browser cleaned for the job (the server's cache has none of those)
  }

  /* ---- store ---- */
  subscribe = (fn) => { this.listeners.add(fn); return () => this.listeners.delete(fn); };
  getVersion = () => this.v;
  touch() { this.v++; for (const fn of this.listeners) fn(); }
  dispose() {
    this.abort?.abort();
    this.listenCtl?.abort();
    this.sgTask?.cancel();
    this.listeners.clear();
  }

  get kind() { return this.info?.hasVideo ? 'video' : 'audio'; }
  get hasAudio() { return this.info?.hasAudio !== false; }
  get second() { return (this.info?.audioTracks || 0) >= 2; }
  get busy() { return this.stage === 'processing' && !this.job; }
  get local() { return !!this.job?.local; }
  get language() { return this.S?.tracks.subs[0]?.lang === 'auto' ? '' : this.S?.tracks.subs[0]?.lang || ''; }
  get tinyError() { return ''; }

  /* ================================================================================================ opening */

  /**
   * A file has been chosen: look at it. Rejects with a plain sentence when it can't be worked on. `draft`: the work that was kept across a
   * sign-in (studio/draft.js); the editor then starts where the person left it.
   */
  static async open(file, { draft = null, signedIn = true } = {}) {
    if (file.size < TINY_FILE_BYTES) {
      throw new Error(file.size === 0 ? 'That file is empty (0 bytes). Please pick another one.' : 'That file is too small to hold any sound. Please pick another one.');
    }
    const info = await probeMedia(file);
    const s = new Session();
    s.file = file;
    s.info = info;
    s.fileName = file.name;
    // Some formats (WMA, AMR, AVI, MPEG...) the browser can't read, but the server can. Then the length is a rough guess from the file
    // size (128 kbit/s) until the server measures it; the price is corrected to the real length.
    s.lengthUnknown = !!(!info.duration && info.hasAudio !== false);
    const guess = Math.min(Math.max(30, (file.size * 8) / 128000), 4 * 3600);
    s.T = info.duration || (s.lengthUnknown ? guess : 1);
    s.S = makeState(s.T, { second: s.second, auto: s.auto });
    // Without an account only the free options can be chosen: every free option this file can have is on from the start, and
    // Automatic (which runs in this browser, nothing is sent) picks among them only.
    if (!signedIn && info.hasAudio !== false) { s.signedOutStart = true; s.freeStart(); }
    if (info.hasAudio === false) {
      // a file with no sound has nothing to do
      s.S.tracks.voice[0].mode = 0; s.S.tracks.level[0].lvl = 0; s.S.tracks.echo[0].on = 0;
    }
    s.wave = { state: 'reading', data: null };
    if (draft) s.restore(draft);
    s.startWave();
    return s;
  }

  /** Every free option this file can have, on for the whole file, and nothing paid (the start without an account). */
  freeStart() {
    const t = this.S.tracks;
    const free = browserVerdict(this.info).ok;
    const auto = !!this.auto; // (marked automatic while Automatic is on, so what it finds can still set them)
    t.voice = [{ ...t.voice[0], mode: free ? 1 : 0, kind: 'free', str: 2, rem: 0, db: null, auto }];
    t.echo = [{ ...t.echo[0], on: 0, auto: false }];
    t.level = [{ ...t.level[0], lvl: 0, auto }];
    t.subs = [{ ...t.subs[0], q: subsVerdict(this.info).ok ? 1 : 0, lang: 'auto', tr: [], auto }];
  }

  /** Put back what a draft kept: the tracks, the selection, the suggestions, the file choices, and what the quick listen found. */
  restore(rec) {
    this.T = rec.seconds;
    this.lengthUnknown = !!rec.lengthUnknown;
    this.auto = !!rec.auto;
    this.signedOutStart = false;
    this.S = makeState(this.T, { second: this.second, auto: this.auto });
    this.S.tracks = rec.tracks;
    this.S.sel = rec.sel || null;
    this.S.sugg = rec.sugg || [];
    if (rec.out) this.out = rec.out;
    this.applied = !!rec.applied;
    this.undoTo = rec.undoTo || null;
    this.stats = rec.stats || null;
    this.draftChoice = rec.choice || null; // the output choice (file type, name, picture size): the page starts from it
    this.restored = { moments: rec.moments };
    this.restoredNote = true;
  }

  /** A job from the list (no file in this tab, or its own session): the editor for it. */
  static forJob(job, fin) {
    const s = sessions.get(job.id) || new Session();
    if (sessions.get(job.id) === s) { s.attach(job, fin); return s; }
    s.id = job.id;
    s.fileName = job.displayName || job.inputName || '';
    s.file = fin.file || null;
    s.info = fin.info || null;
    s.T = job.durationSeconds || 0;
    sessions.set(job.id, s);
    s.attach(job, fin);
    return s;
  }

  attach(job, fin) {
    this.job = job;
    this.fin = fin;
    if (!this.file && fin?.file) { this.file = fin.file; this.info = fin.info; }
    if (job.id && this.id !== job.id) { sessions.delete(this.id); this.id = job.id; sessions.set(job.id, this); }
  }

  /** Remember this session under the job it has now (so the job's own page finds it). */
  register() { sessions.set(this.id, this); }

  /* ================================================================================================ the waveform and the quick listen */

  async startWave() {
    const ctl = new AbortController();
    this.waveCtl?.abort();
    this.waveCtl = ctl;
    if (!this.file || this.info?.hasAudio === false) { this.wave = { state: 'none', data: null }; this.touch(); this.afterWave(); return; }
    try {
      const r = await filePeaks(this.file, { signal: ctl.signal, info: this.info });
      if (ctl.signal.aborted) return;
      this.wave = { state: 'ok', data: r };
      // the length the file really has (the browser could not tell it before, or told it a little differently); work that was kept already
      // knows it, unless it was kept before the length was found
      if (this.lengthUnknown || (!this.info.duration && !this.restored)) this.setLength(r.dur);
    } catch (e) {
      if (e?.name === 'AbortError' || ctl.signal.aborted) return;
      this.wave = { state: 'none', data: null };
    }
    this.touch();
    this.afterWave();
  }

  /** The recording turned out to be this long: the tracks start again at this length (nothing was changed yet). */
  setLength(seconds) {
    const t = Math.max(1, seconds);
    this.T = t;
    this.lengthUnknown = false;
    this.S = makeState(t, { second: this.second, auto: this.auto });
    if (this.signedOutStart) this.freeStart();
  }

  afterWave() {
    if (this.waveDone) return;
    this.waveDone = true;
    // a browser that asks to save data downloads nothing before Start (the quick listen and the search for moments both need the noise remover)
    if (navigator.connection?.saveData) return;
    // work that was kept: the choices are as they were left (not set again), and the moments are looked for only if that was not done yet
    if (this.restored) { if (this.restored.moments !== 'done') this.findMoments(); return; }
    if (this.hasAudio && this.auto) this.listen();
    else this.findMoments();
  }

  /** A quick listen to the file (about a minute of it, spread over the whole), so the settings can start out right. `again`: what it finds
   * goes over every choice (Automatic was switched on by hand), not only those still marked automatic. */
  async listen({ again = false } = {}) {
    if (!this.file || !this.info || !this.hasAudio) return;
    this.listenCtl?.abort();
    const ctl = new AbortController();
    this.listenCtl = ctl;
    this.listening = true;
    this.touch();
    try {
      const found = await analyseFile(this.file, this.info, { signal: ctl.signal });
      if (ctl.signal.aborted) return;
      this.stats = found;
      const r = found ? recommendForFile(found, { language: this.language }) : null;
      if (r && this.auto) this.setRecommendation(r, { again });
    } catch {
      /* canceled, or unreadable: the tracks keep their usual starting choices */
    } finally {
      if (this.listenCtl === ctl) { this.listening = false; this.touch(); }
    }
    if (!ctl.signal.aborted) this.findMoments();
  }

  get reco() { return this.auto && this.stats ? recommendForFile(this.stats, { language: this.language }) : null; }

  setRecommendation(r, { again = false } = {}) {
    if (!this.undoTo) this.undoTo = snapJ(this.S);
    const p = recoPatches(r, { rebuildOk: this.env.rebuildOk !== false, freeOnly: !!this.signedOutStart });
    if (this.signedOutStart) { // (only what this browser can do for this file)
      if (!browserVerdict(this.info).ok) delete p.voice;
      if (!subsVerdict(this.info).ok) delete p.subs;
    }
    applyReco(this.S, p, { all: again });
    this.applied = true;
    this.touch();
  }

  /** Have any of the automatic choices been changed by hand since they were set? */
  get custom() {
    return this.applied && ['voice', 'level', 'subs'].some((tr) => this.S.tracks[tr].some((x) => !x.auto));
  }

  setAuto(on) {
    this.auto = on;
    saveAuto(on);
    if (!on) {
      this.listenCtl?.abort();
      this.listening = false;
      this.sgTask?.cancel();
      this.touch();
      return;
    }
    if (!this.hasAudio) { this.touch(); return; }
    if (this.reco) this.setRecommendation(this.reco, { again: true });
    else if (!this.listening && this.file && this.stage === 'ready' && !this.job) this.listen({ again: true });
    else this.touch();
    this.findMoments();
  }

  /** "Undo": the tracks go back to what they were before they were set automatically, and Automatic is switched off. */
  undoAuto() {
    if (this.undoTo) {
      const o = JSON.parse(this.undoTo);
      this.S.tracks = o.tracks;
    }
    this.undoTo = null;
    this.applied = false;
    this.setAuto(false);
  }

  /** "Use them again": the automatic choices over everything, even the parts changed by hand. */
  useAgain() {
    if (this.reco) this.setRecommendation(this.reco, { again: true });
  }

  /* ================================================================================================ moments worth a look */

  /** Look for suggestions in this browser (files up to 30 minutes it can open itself); a job's own come from our side after the first run. */
  findMoments() {
    if (this.sgStarted || !this.auto || !this.file || this.job) return;
    const verdict = suggestVerdict(this.info, this.T);
    if (!verdict.ok) { this.sg = { phase: 'unavailable', progress: 0, error: '', why: verdict.why }; this.touch(); return; }
    this.sgStarted = true;
    this.sg = { phase: 'running', progress: 0, error: '', why: '' };
    this.touch();
    const task = startSuggestions(this.file, this.info, {
      seconds: this.T,
      onProgress: (f) => { this.sg = { ...this.sg, progress: f }; this.touch(); },
    });
    this.sgTask = task;
    task.done.then(
      (r) => {
        this.sg = { phase: 'done', progress: 1, error: '', why: '' };
        this.setSuggestions(r.suggestions);
      },
      (e) => {
        this.sg = { phase: e?.name === 'AbortError' ? 'canceled' : 'error', progress: 0, error: e?.message || '', why: '' };
        if (e?.name === 'AbortError') this.sgStarted = false; // (switching Automatic off and on looks again)
        this.touch();
      },
    );
  }

  cancelMoments() { this.sgTask?.cancel(); }

  /** Make the prototype's suggestions from a list (editor/suggest.js shape), keeping what was dismissed or applied. */
  setSuggestions(list, dismissed = []) {
    const old = new Map((this.S.sugg || []).map((g) => [g.key, g]));
    this.S.sugg = list
      .map((x) => ({ ...x, key: x.key || `${x.kind}:${Math.round(x.start * 10)}-${Math.round(x.end * 10)}` }))
      .filter((x) => !dismissed.includes(x.key))
      .map((x) => {
        const g0 = toSuggestion(x, this.T, ++this.sgN);
        const g = this.signedOutStart ? freeOnlySuggestion(g0) : g0;
        const was = old.get(g.key);
        return was ? { ...g, st: was.st } : g;
      });
    autoDone(this.S);
    this.touch();
  }

  /** The job's own suggestions (after the first run). */
  async loadJobSuggestions() {
    if (!this.job?.id || this.job.local) return;
    await loadSuggestions(this.job.id);
    const st = suggestionState(this.job.id);
    this.setSuggestions(st.list, st.dismissed);
  }

  /* ================================================================================================ the tracks */

  /** Change the tracks: undo is kept, a suggestion that is done is marked, and the page is told. */
  commit(fn) {
    this.restoredNote = false;
    pushUndo(this.S);
    fn(this.S);
    autoDone(this.S);
    this.touch();
  }

  select(list) { setSel(this.S, list); this.touch(); }

  undo() { if (undoState(this.S)) this.touch(); }
  redo() { if (redoState(this.S)) this.touch(); }

  split(t) {
    pushUndo(this.S);
    if (!splitAll(this.S, t)) { this.S.past.pop(); return false; }
    this.touch();
    return true;
  }

  /* ================================================================================================ a job that is loaded */

  /** The tracks of a job on our side: what its timeline holds now (and what the files hold), the way to listen, the second track. */
  async loadJob({ force = false } = {}) {
    const job = this.job;
    if (!job || job.local || this.loading) return;
    if (this.server && !force) return;
    const finishedAt = job.finishedAt;
    this.loading = true;
    this.touch();
    try {
      const data = await loadEditor(job.id);
      this.server = data;
      this.serverError = '';
      const T = data.seconds || job.durationSeconds || this.T;
      const subs = job.transcribeTier ? { quality: job.transcribeTier === 'premium' ? 'accurate' : 'standard', language: job.options?.language || null, translate: job.options?.translate || [] } : null;
      // a job whose subtitles were set for the whole file (made without a timeline of its own) can't have subtitle parts: they stay as they are
      this.subsWhole = !!subs && !(data.rendered || data.base).parts('subtitles').length;
      const made = fromTimeline(data.rendered || data.base, T, { subs });
      const working = data.base === data.rendered || !data.rendered ? made : fromTimeline(data.base, T, { subs });
      // what the files hold, as the person made it: the sound timeline the files hold has no subtitles, a remover that was left to us ("Auto")
      // is decided when it runs, and a part the browser cleaned may not say its strength: none of that is a change
      if (data.rendered && !data.rendered.parts('subtitles').length && data.base.parts('subtitles').length) made.subs = clone(working.subs);
      for (const x of made.voice) {
        const w = working.voice.find((y) => y.s <= x.s + 1e-6 && y.e >= x.e - 1e-6);
        if (!w || w.mode !== 1 || x.mode !== 1) continue;
        if (w.rem === 0) x.rem = 0;
        if (w.kind === 'free' && x.kind === 'free') { x.str = w.str; x.db = w.db; }
      }
      const rebuilt = !this.S || !(this.S.T === T && this.S.past.length > 0) || force;
      if (rebuilt) {
        this.T = T;
        this.S = makeState(T, { second: data.secondTrack, auto: false });
        this.S.tracks = clone(working);
      }
      this.S.second = !!data.secondTrack;
      const keep = this.snap?.choice;
      if (!keep) {
        // a job this tab did not start: the subtitle files and "into the video" it was asked for (kept with it in this browser, else what it made)
        const w = this.fin?.wants || {};
        this.out = { sfiles: ['srt', 'vtt', 'txt'].filter((k) => w.caps?.[k]), burn: !!w.embed };
      }
      this.snap = { tracks: rebuilt || !this.snap ? clone(made) : this.snap.tracks, choice: keep || null, sfiles: this.out.sfiles.slice(), burn: this.out.burn };
      this.loadedAt = finishedAt ?? null;
      this.paid = job.costMicros || 0;
      this.applied = false;
      if (data.suggestions) this.loadJobSuggestions();
      else this.S.sugg = [];
      this.loadPeaks();
    } catch (e) {
      this.serverError = e.detail || e.message || 'Couldn’t open the editor for this file.';
    } finally {
      this.loading = false;
      this.touch();
    }
  }

  /** The server's peaks of each layer (after the job is done): what the waveform draws what is heard from. */
  async loadPeaks() {
    const job = this.job;
    if (!job || job.local) return;
    const names = ['source', 'noise', 'rebuild', ...(this.S?.second ? ['echo'] : [])];
    try {
      let answers = await Promise.all(names.map((n) => loadLayerPeaks(job.id, n).catch(() => null)));
      // the values of blocks that were not outlined yet come with the next ask
      for (let again = 0; again < 4 && answers.some((a) => a?.pending); again++) {
        await new Promise((r) => setTimeout(r, 1500));
        answers = await Promise.all(names.map((n, i) => (answers[i]?.pending ? loadLayerPeaks(job.id, n).catch(() => answers[i]) : answers[i])));
      }
      const layers = {};
      names.forEach((n, i) => { layers[n] = layerPeaks(answers[i]); });
      layers.noise = overlayPeaks(layers.noise, this.browserPeaks, this.T);
      this.layers = layers;
      if (!this.wave.data && layers.source) {
        // no file in this tab (a reload, another device): the outline is the recording's own peaks from our side
        const a = layers.source.a;
        const top = Math.max(1e-6, ...a.filter((v) => Number.isFinite(v)));
        const o = Array.from(a, (v) => (Number.isFinite(v) ? v / top : 0));
        this.wave = { state: 'ok', data: { dur: this.T, N: o.length, top, o, m: o, d: o.map(() => 0) } };
      }
    } catch {
      /* the waveform keeps showing the recording */
    }
    this.touch();
  }

  /** The choices the finished files were made with, as the page's output controls hold them. */
  initialOutput() {
    const w = this.fin?.effective?.() || this.fin?.wants;
    if (!w) return null;
    return { name: w.name, format: w.format, size: w.size };
  }

  /* ================================================================================================ starting */

  /** What the person asked for, as the numbers to start with (for tests and the page). */
  plan({ choice, limit = buildLimit() }) {
    return outputPlan({ tracks: this.S.tracks, info: this.info, file: this.file, choice, out: this.out, limit });
  }

  setTask(name, task) {
    this.run = { ...(this.run || { tasks: {} }), tasks: { ...(this.run?.tasks || {}), [name]: task } };
    this.touch();
  }

  clearTask(name) {
    const tasks = { ...(this.run?.tasks || {}) };
    delete tasks[name];
    this.run = { ...(this.run || {}), tasks };
    this.touch();
  }

  cancelRun() { this.abort?.abort(); }

  /**
   * Start: the Free parts run in this browser, everything else on our side as one job (docs/api.md section 8: a timeline of parts; the Free
   * noise parts are cleaned here and uploaded, the Free subtitle parts transcribed here and their words uploaded). With nothing on our side
   * there is no job: the result is a card of this tab only. Resolves {id} once the job (or the card) exists and the browser's share is done,
   * or null when it was canceled or failed (then `error` says why).
   */
  async start({ choice, plan, onJobCreated, refresh, estimate }) {
    const file = this.file;
    const info = this.info;
    const ctl = new AbortController();
    this.abort = ctl;
    this.stage = 'processing';
    this.run = { tasks: {}, saving: false, cancel: true };
    this.error = '';
    this.errorKind = '';
    this.welcome = '';
    this.touch();
    const tracks = clone(this.S.tracks);
    const T = this.T;
    const server = needsServer(tracks) || !!plan.serverConvert; // (a video this browser can't make goes up whole, to be made on our side)
    const wants = wantsOf({ tracks, plan, choice, out: this.out, file, local: !server });
    this.restoredNote = false;
    const freeNoise = freeNoiseRanges(tracks);
    const freeSubs = freeSubRanges(tracks);
    let failed = false;
    try {
      // (the page asks to sign in instead of getting here; this keeps a job from being tried with no account all the same)
      if (server && !this.env.signedIn) throw new Error('Sign in to start this. Your file and choices are kept while you do.');
      if (server) return await this.startJob({ file, info, tracks, T, wants, choice, plan, freeNoise, freeSubs, ctl, onJobCreated, refresh, estimate });
      return await this.startLocal({ file, info, tracks, T, wants, choice, freeNoise, freeSubs, ctl });
    } catch (e) {
      failed = true;
      if (e?.name === 'AbortError' || ctl.signal.aborted) {
        this.error = '';
        this.canceled = true;
      } else {
        this.error = inGigabytes(e.detail || e.message || 'Something went wrong. Please try again.');
        this.errorKind = e?.status === 507 ? 'storage' : '';
      }
      return null;
    } finally {
      if (failed || !this.job) {
        this.stage = 'ready';
        this.run = null;
      }
      this.touch();
    }
  }

  /** The browser's share of a job, once the job says what it needs (`needs`): clean each range and send it, transcribe each range and send the words. */
  async browserShare({ id, needs, file, tracks, T, freeNoise, freeSubs, ctl }) {
    const noiseNeeds = needs.filter((n) => n.kind === 'noise');
    const cueNeeds = needs.filter((n) => n.kind === 'cues');
    const rangeOf = (n) => ({ start: n.start, end: n.end });
    const noiseTasks = noiseNeeds.map(async (n, i) => {
      const name = `clean${i}`;
      this.setTask(name, { text: 'Getting the noise remover ready…', frac: null });
      const pcm = await cleanRange(file, rangeOf(n), freeNoise, {
        signal: ctl.signal, T,
        onStage: (text) => this.setTask(name, { ...(this.run.tasks[name] || {}), text }),
        onProgress: (p) => this.setTask(name, { text: 'Cleaning…', frac: p.total ? p.done / p.total : null, eta: p.eta, rate: p.rate }),
      });
      this.setTask(name, { text: 'Getting it ready to upload…', frac: 1 });
      rangePeaks(pcm, n.start).then((p) => { this.browserPeaks = [...this.browserPeaks.filter((q) => q.from !== p.from), p]; }).catch(() => {});
      // the server takes exactly the samples of the part: a length of the recording that differs a little (the end of the file) is made right
      let want = samplesOf(n.start, n.end);
      let flac = await pcmToFlac(fitPcm(pcm, want), { signal: ctl.signal });
      try {
        await uploadBrowserNoise(id, rangeOf(n), flac, { signal: ctl.signal });
      } catch (e) {
        const m = /exactly the part: (\d+) samples/.exec(e?.detail || '');
        if (!m || Number(m[1]) === want) throw e;
        want = Number(m[1]);
        flac = await pcmToFlac(fitPcm(pcm, want), { signal: ctl.signal });
        await uploadBrowserNoise(id, rangeOf(n), flac, { signal: ctl.signal });
      }
      this.clearTask(name);
    });
    const cueTasks = cueNeeds.map(async (n, i) => {
      const name = `words${i}`;
      this.setTask(name, { text: 'Getting the subtitle maker ready…', frac: null });
      const sub = freeSubs.find((r) => r.start <= n.start + 0.001 && r.end >= n.end - 0.001) || freeSubs[0];
      const r = await transcribeRange(file, rangeOf(n), sub?.language || '', {
        signal: ctl.signal, T,
        onStage: (text) => this.setTask(name, { ...(this.run.tasks[name] || {}), text }),
        onProgress: (p) => this.setTask(name, { text: 'Writing the subtitles…', frac: p.total ? p.done / p.total : null, eta: p.eta }),
      });
      await uploadBrowserCues(id, rangeOf(n), r.cues, ctl.signal);
      this.clearTask(name);
    });
    await Promise.all([...noiseTasks, ...cueTasks]);
  }

  async startJob({ file, info, tracks, T, wants, choice, plan, freeNoise, freeSubs, ctl, onJobCreated, refresh, estimate }) {
    const kind = jobKind(tracks, plan);
    const videoOut = videoOutOf(plan, choice, serverSize(choice.size));
    // a job made with parts says what is done in its timeline (docs/api.md section 8): the subtitles' quality, language and translations are the
    // subtitle parts' own, so the job carries no tier of its own; a conversion alone (kind convert) is only its video_out
    const body = kind === 'convert' ? { kind, durationSeconds: T || 1, options: { video_out: videoOut } } : {
      kind,
      durationSeconds: T || 1,
      options: {
        audio_format: serverAudioFormat(choice.format, file, info),
        // putting subtitles into the video here needs a subtitle file, even when only the transcript was ticked
        caption_formats: plan.wantsText ? ['srt', 'vtt', 'txt'].filter((c) => this.out.sfiles.includes(c) || (c === 'vtt' && plan.embedHere && !this.out.sfiles.includes('srt'))) : [],
        embed_subtitles: !!(plan.embedNow && plan.embedOnServer),
        ...(videoOut ? { video_out: videoOut } : {}),
      },
      timeline: toTimeline(tracks, T).toJSON(),
    };
    let share = null;
    const out = await submitJob({
      file, info, body, signal: ctl.signal,
      onStage: (phase) => this.setTask('upload', { text: { extract: 'Pulling the sound out of your video…', compress: 'Making it smaller for upload…', create: 'Getting things ready…', upload: 'Uploading…' }[phase] || 'Uploading…', frac: null }),
      onProgress: (p) => this.setTask('upload', { text: { extract: 'Pulling the sound out of your video…', compress: 'Making it smaller for upload…', create: 'Getting things ready…', upload: 'Uploading…' }[p.phase] || 'Uploading…', frac: p.frac, loaded: p.loaded, total: p.total }),
      onCompressed: (c) => { this.compressed = c; },
      onCreated: (created) => {
        if (created.needs?.length) share = this.browserShare({ id: created.id, needs: created.needs, file, tracks, T, freeNoise, freeSubs, ctl });
        share?.catch(() => {});
      },
    });
    this.clearTask('upload');
    // the job exists: from here on its page finishes it by itself (finish.js)
    const wantsNow = { ...wants, soundHere: false, subsHere: false };
    this.id = out.id;
    const fin = beginFinish({ ...out, displayName: file.name, outputs: [] }, { file, info, wants: wantsNow, browser: share ? share.catch(() => {}) : null });
    this.fin = fin;
    this.takeSnapshot(tracks, choice);
    this.job = { id: out.id, status: 'uploading', kind, displayName: file.name, inputName: file.name, durationSeconds: T, outputs: [], progress: 0 };
    this.register();
    this.welcome = startedNote(out);
    try {
      if (share) await share;
    } catch (e) {
      // the job is waiting for what the browser could not make: stop it (nothing is charged)
      api(`jobs/${out.id}/cancel`, { method: 'POST', redirectOn401: false }).catch(() => {});
      this.job = null;
      throw e;
    }
    await refresh?.();
    onJobCreated?.(out);
    return { id: out.id, job: out };
  }

  /** What the files hold now: the tracks, the file type and name (as the finishing step has them: a video with only subtitles inside is an MKV), the subtitle files. */
  takeSnapshot(tracks, choice, fin = this.fin) {
    const w = fin?.effective?.();
    const made = w && (w.sound || w.embed) ? { format: w.format, name: w.name, size: w.size } : choice ? { format: choice.format, name: choice.name, size: choice.size } : null;
    this.snap = { tracks: clone(tracks), choice: made, sfiles: this.out.sfiles.slice(), burn: this.out.burn };
  }

  /** A run with nothing on our side: the Free noise removal and subtitles, made here. */
  async startLocal({ file, info, tracks, T, wants, choice, freeNoise, freeSubs, ctl }) {
    const results = await Promise.allSettled([
      freeNoise.length ? this.localNoise({ file, info, T, freeNoise, ctl }) : Promise.resolve(null),
      freeSubs.length ? this.localSubs({ file, info, T, freeSubs, ctl }) : Promise.resolve(null),
    ]);
    const [noise, subs] = results;
    const canceled = ctl.signal.aborted || results.some((r) => r.status === 'rejected' && r.reason?.name === 'AbortError');
    const bad = results.find((r) => r.status === 'rejected' && r.reason?.name !== 'AbortError');
    const cleaned = noise.status === 'fulfilled' ? noise.value : null;
    const subsFiles = subs.status === 'fulfilled' && subs.value ? subs.value : null;
    if (canceled) throw new DOMException('Canceled', 'AbortError');
    if (bad && !cleaned && !subsFiles) throw new Error(bad.reason?.message || (freeNoise.length ? 'Something went wrong while cleaning this file. Please try again.' : 'Something went wrong while making the subtitles. Please try again.'));
    const failedParts = { sound: freeNoise.length > 0 && !cleaned, subs: freeSubs.length > 0 && !subsFiles };
    if (bad) this.error = bad.reason?.message || '';
    const made = { sound: !!cleaned, subs: !!subsFiles };
    const label = made.sound && made.subs ? 'Free noise removal and subtitles' : made.sound ? 'Free noise removal' : made.subs ? 'Free subtitles' : 'File conversion';
    const id = newLocalJobId();
    const fin = beginFinish({
      id, local: true, free: true, status: 'done', kind: 'denoise', label, displayName: file.name, inputName: file.name,
      createdAt: Math.floor(Date.now() / 1000), durationSeconds: info?.duration || T, outputs: [], costMicros: 0,
    }, { file, info, wants: { ...wants, sound: made.sound, subs: made.subs }, browser: null });
    const wav = cleaned ? await buildWav(cleaned.pcm, 16) : null;
    Object.assign(fin, {
      pcm: cleaned?.pcm || null, wav, wavUrl: wav ? URL.createObjectURL(wav) : '', subs: subsFiles, failed: failedParts,
      ...(cleaned?.pcm.pair ? { pairParts: cleaned.pcm.pair, pairChannels: cleaned.pcm.channels, strengthDb: cleaned.attenDb } : {}),
    });
    if (cleaned?.pcm.pair) fin.set({ strengthDb: cleaned.attenDb }); // the page shows the strength it was made with
    this.setTask('finish', { text: 'Putting it together…', frac: 0 });
    const off = fin.subscribe(() => this.setTask('finish', { text: 'Putting it together…', frac: fin.state.progress }));
    await fin.run({ save: true, signal: ctl.signal });
    off();
    this.clearTask('finish');
    addLocalCard(fin.job);
    this.id = id;
    this.fin = fin;
    this.job = fin.job;
    this.takeSnapshot(tracks, choice);
    this.register();
    return { id, job: fin.job };
  }

  /** Free noise removal for every range, with the rest of the recording as it is. */
  async localNoise({ file, info, T, freeNoise, ctl }) {
    const channelsOf = info?.channels >= 2 ? 2 : 1;
    const setProgress = (p) => this.setTask('clean', { text: 'Cleaning…', frac: p.total ? p.done / p.total : null, eta: p.eta, rate: p.rate });
    this.setTask('clean', { text: 'Getting the noise remover ready…', frac: null });
    const common = { signal: ctl.signal, onProgress: setProgress, onStage: (text) => this.setTask('clean', { ...(this.run.tasks.clean || {}), text }) };
    const whole = freeNoise.length === 1 && freeNoise[0].start <= 0.001 && freeNoise[0].end >= T - 0.6;
    let pcm;
    let attenDb = freeNoise[0].db;
    if (whole) {
      // the whole recording: the full-strength sound and the recording are kept (in memory) while they fit, so the strength can change afterwards
      pcm = await denoiseFile(file, { attenDb, durationHint: info?.duration || T, ...common, keepPair: !!info?.duration && previewFits(info.duration, channelsOf) });
    } else {
      const pieces = [];
      for (const r of freeNoise) {
        const p = await cleanRange(file, { start: r.start, end: r.end }, [r], { ...common, T });
        pieces.push({ start: r.start, end: r.end, pcm: p });
      }
      this.setTask('clean', { text: 'Putting the cleaned parts back…', frac: null });
      pcm = await splicePcm(file, pieces, { signal: ctl.signal });
      attenDb = freeNoise[0].db;
    }
    this.clearTask('clean');
    return { pcm, attenDb };
  }

  /** Free subtitles for every range. */
  async localSubs({ file, T, freeSubs, ctl }) {
    const segs = [];
    let language = '';
    this.setTask('words', { text: 'Getting the subtitle maker ready…', frac: null });
    for (const r of freeSubs) {
      const res = await transcribeRange(file, { start: r.start, end: r.end }, r.language, {
        signal: ctl.signal, T,
        onStage: (text) => this.setTask('words', { ...(this.run.tasks.words || {}), text }),
        onProgress: (p) => this.setTask('words', { text: 'Writing the subtitles…', frac: p.total ? p.done / p.total : null, eta: p.eta }),
      });
      segs.push(res.segments);
      language ||= res.language;
    }
    this.clearTask('words');
    return subtitleFiles({ segments: joinSegments(segs), language, seconds: T });
  }

  /* ================================================================================================ saving changes */

  /** The timeline to send for these tracks: a job's whole-file subtitles are not parts of it (and can't be changed here). */
  timelineOf(tracks) {
    const tl = toTimeline(tracks, this.T);
    const key = (t) => JSON.stringify(t.subs.map((x) => [x.s, x.e, vkey('subs', x)]));
    if (this.subsWhole && this.snap && key(tracks) === key(this.snap.tracks)) tl.layers.subtitles = [];
    return tl;
  }

  /** The tracks are not what the files hold. */
  get tracksChanged() { return !!this.snap && !!this.S && !sameTracks(this.S.tracks, this.snap.tracks, this.T); }

  /**
   * Save the changes of a finished recording. Only the choices of the files changed: they are made again in the browser (free). The tracks
   * changed: a run on our side of what is new (quote and process, or export when it is free), after the Free parts that are new have been
   * cleaned or transcribed here and sent; for a card of this tab, the run again from the file.
   */
  async saveChanges({ choice, plan, wants, quote, onStarted, refresh }) {
    const fin = this.fin;
    if (!this.tracksChanged) {
      await fin.run({ save: true, wants });
      this.takeSnapshot(this.S.tracks, choice);
      this.touch();
      return { id: this.job.id };
    }
    const ctl = new AbortController();
    this.abort = ctl;
    this.error = '';
    this.run = { tasks: {}, saving: true, cancel: false };
    this.stage = 'processing';
    this.touch();
    try {
      const tracks = clone(this.S.tracks);
      const T = this.T;
      const tl = this.timelineOf(tracks);
      const freeNoise = freeNoiseRanges(tracks);
      const freeSubs = freeSubRanges(tracks);
      const body = { free: !!quote?.free, maxCostMicros: quote?.estCostMicros, audioFormat: serverAudioFormat(choice.format, this.file, this.info) };
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          await saveTimeline(this.job.id, tl, body);
          break;
        } catch (e) {
          const needs = needsOf(e);
          if (!needs || !needs.length || attempt === 2) throw e;
          if (!this.file) throw new Error('The original file isn’t in this tab. Choose it again to clean or transcribe this part in your browser.');
          await this.browserShare({ id: this.job.id, needs, file: this.file, tracks, T, freeNoise, freeSubs, ctl });
        }
      }
      fin.rearm(); // the new files come down by themselves when the job is done again
      fin.setWants({ ...wants, soundHere: false, subsHere: false });
      this.takeSnapshot(tracks, choice);
      this.run = { tasks: {}, saving: true, cancel: false };
      await refresh?.();
      onStarted?.();
      return { id: this.job.id };
    } catch (e) {
      this.stage = 'ready';
      this.run = null;
      if (e?.name === 'AbortError') return null;
      this.error = e.code === 'price_changed' ? 'The price changed while you were deciding. The new price is shown; save again if it is fine.' : e.detail || e.message || 'Couldn’t save this. Please try again.';
      this.touch();
      return null;
    }
  }

  /* ================================================================================================ the price, asked of our side */

  /** The price of these tracks before the job exists: the server's estimate for a timeline of parts (it counts what each run costs besides). */
  estimateBody({ choice, plan }) {
    const tracks = this.S.tracks;
    const kind = jobKind(tracks, plan);
    if (!kind) return null;
    const videoOut = videoOutOf(plan, choice, serverSize(choice.size));
    if (kind === 'convert') return { kind, durationSeconds: this.T, options: { video_out: videoOut } };
    return {
      kind, durationSeconds: this.T,
      options: { audio_format: serverAudioFormat(choice.format, this.file, this.info), caption_formats: ['srt', 'vtt', 'txt'], ...(videoOut ? { video_out: videoOut } : {}) },
      timeline: toTimeline(tracks, this.T).toJSON(),
    };
  }

  /** The price of saving the changes of a job (nothing is held by asking). */
  quote(signal) { return quoteTimeline(this.job.id, this.timelineOf(this.S.tracks), signal); }

  async estimate(body, signal) { return estimateTimeline(body, signal); }
}

