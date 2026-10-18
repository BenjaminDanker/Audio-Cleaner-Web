// The last step of a job, done here in the browser, and the files the person gets. Every job (a paid one from our side, or
// a free one made in this tab) ends the same way: what was asked for is put together (the cleaned sound back into the video,
// the subtitles into the video, the chosen file type), downloaded by itself, and kept ready for "Download again".
// One Finish object per job lives here for as long as the tab does, so it keeps going when the person moves to another page.

import { useSyncExternalStore } from 'react';
import { outputUrl } from './api.js';
import { saveBlob } from './download.js';
import { stem } from './format.js';
import { LANGUAGES } from './jobs.js';
import { buildMedia, cleanName, endingOf, FORMAT, formatOfName, srtToVtt, vttToSrt } from './media/convert.js';
import { probeMedia } from './media/mb.js';

const LANGUAGE_NAME = Object.fromEntries(LANGUAGES.filter(([c]) => c));
export const langName = (code) => LANGUAGE_NAME[code] || 'Subtitles';

const VIDEO_NAME = /\.(mp4|mov|m4v|mkv|webm)$/i;
const CAPTION_TYPES = { srt: 'application/x-subrip', vtt: 'text/vtt', txt: 'text/plain' };
const GAP_MS = 700; // between two downloads: browsers that ask "allow several downloads?" get one question, not a burst

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const abortError = () => new DOMException('Canceled', 'AbortError');

/* ------------------------------------------------------------------------------------------------ what was asked for */

/**
 * What the person asked for, saved with the job in this browser so that "Download again" and "Change and save again" know it
 * after a reload (the server only keeps its own files):
 *   {sound, format, name, size, subs, caps:{srt,vtt,txt}, embed, soundHere, subsHere}
 * sound: the cleaned sound is part of the result, as `name` (a file type `format`, picture `size` for a video).
 * subs/caps: subtitle files by format. embed: the subtitles go inside the video. soundHere/subsHere: made in this
 * browser, so they exist only in this tab.
 */
const KEY = 'acw.wants.';
export function saveWants(id, wants) {
  try { localStorage.setItem(KEY + id, JSON.stringify(wants)); } catch { /* storage blocked: the card then goes by the files the server has */ }
}
export function loadWants(id) {
  try {
    const t = localStorage.getItem(KEY + id);
    return t ? JSON.parse(t) : null;
  } catch {
    return null;
  }
}
/** Forget the choices of jobs the server no longer lists (they are kept for 24 hours). */
export function pruneWants(ids) {
  try {
    const keep = new Set([...ids, ...finishes.keys()]); // a job this tab just made may not be in the list yet
    for (const k of Object.keys(localStorage)) if (k.startsWith(KEY) && !keep.has(k.slice(KEY.length))) localStorage.removeItem(k);
  } catch { /* nothing to clean */ }
}

/** What a job asked for when this browser never heard of it (another device): the files the server made, as they are. */
function derivedWants(job) {
  const outs = job.outputs || [];
  const audio = outs.find((o) => o.kind === 'audio');
  const video = outs.find((o) => o.kind === 'video');
  const media = video || audio;
  const has = (re) => outs.some((o) => (o.kind === 'captions' || o.kind === 'transcript') && re.test(o.name));
  const caps = { srt: has(/\.srt$/i), vtt: has(/\.vtt$/i), txt: has(/\.txt$/i) };
  const named = media ? formatOfName(media.name) : '';
  return {
    sound: !!media, format: named || 'wav', name: media?.name || '', size: 'same',
    subs: caps.srt || caps.vtt || caps.txt, caps, embed: !!video && (caps.srt || caps.vtt || caps.txt) && !audio, soundHere: false, subsHere: false,
  };
}

/** The rough time the finishing step takes, in seconds (0: nothing to build, the file is handed over as it is). */
export function finishSeconds(w, { info, file, serverAudio = false } = {}) {
  if (!w?.sound && !w?.embed) return 0;
  const def = FORMAT[w.format];
  if (!def) return 0;
  const seconds = info?.duration || 0;
  if (def.kind === 'audio') {
    if (serverAudio) return 0;
    return w.format === 'wav' ? 1 : 2 + seconds * 0.03;
  }
  if (w.size && w.size !== 'same') return 3 + seconds * 0.5;
  return 2 + seconds * 0.02 + (file?.size || 0) / 1e8;
}

/* ------------------------------------------------------------------------------------------------ one job's finishing */

const EMPTY = { phase: 'idle', progress: 0, files: [], note: '', error: '', pending: null };

export class Finish {
  constructor(id) {
    this.id = id;
    this.job = null; // the job as the server lists it (a free run in this tab has a made-up one)
    this.file = null; // the original recording, when this tab has it
    this.info = null;
    this.wants = null;
    this.pcm = null; // sound cleaned in this browser
    this.wav = null;
    this.wavUrl = '';
    this.subs = null; // subtitles made in this browser: {srt, vtt, txt, language, empty}
    this.failed = {}; // the browser parts that did not work out: {sound, subs}
    this.browser = null; // resolves when the browser parts are over
    this.thisTab = false; // started from this tab: finishes by itself
    this.cache = new Map();
    this.made = null; // the finished files, kept for "Download again"
    this.ctl = null;
    this.chain = Promise.resolve();
    this.state = EMPTY;
    this.listeners = new Set();
    this.autoStarted = false;
    this.subscribe = (fn) => { this.listeners.add(fn); return () => this.listeners.delete(fn); };
    this.getSnapshot = () => this.state;
  }

  set(patch) {
    this.state = { ...this.state, ...patch };
    for (const fn of this.listeners) fn();
  }

  /** The job as the server lists it now. */
  update(job) {
    this.job = job;
    if (!this.wants) this.wants = loadWants(job.id) || derivedWants(job);
  }

  get outputs() { return this.job?.outputs || []; }

  get inputName() { return this.file?.name || this.job?.displayName || this.job?.inputName || ''; }

  get inputInfo() { return this.info || { hasVideo: VIDEO_NAME.test(this.inputName), duration: this.job?.durationSeconds || null }; }

  get serverVideo() { return this.outputs.find((o) => o.kind === 'video') || null; }

  get serverAudio() { return this.outputs.find((o) => o.kind === 'audio') || null; }

  /** Our side made the finished video (a video type this browser can't open), and this browser has no better way. */
  get useServerVideo() { return !!this.serverVideo && !(this.file && this.info?.viaMediabunny); }

  /** The choices after taking away what did not work out. */
  effective() {
    const w = { ...(this.wants || derivedWants(this.job || {})), caps: { ...(this.wants?.caps || {}) } };
    if (w.soundHere && (this.failed.sound || !this.pcm)) w.sound = false;
    if (w.subsHere && (this.failed.subs || !this.subs)) { w.subs = false; w.embed = false; }
    if (this.failed.sound && !w.soundHere) w.sound = false;
    if (this.failed.subs && !w.subsHere) { w.subs = false; w.embed = false; }
    if (!w.sound && w.embed) {
      w.format = 'mkv'; w.size = 'same';
      if (!w.name || FORMAT[formatOfName(w.name)]?.kind !== 'video') w.name = `${stem(this.inputName)}-subtitled.mkv`;
    }
    return w;
  }

  /** Does this need the original video, which this tab doesn't have? */
  needsVideo(w = this.effective()) {
    if (this.file || this.useServerVideo) return false;
    if (!w.sound && !w.embed) return false;
    if (!this.inputInfo.hasVideo) return false;
    return !w.sound || FORMAT[w.format]?.kind === 'video';
  }

  /** The finished video made here: is it part of what will be built? (it takes longer than a sound file) */
  willBuild(w = this.effective()) {
    return (w.sound || w.embed) && !(FORMAT[w.format]?.kind === 'audio' && !w.soundHere && this.serverAudio && FORMAT[w.format].exts.includes(endingOf(this.serverAudio.name)));
  }

  /* ---- subtitles ---- */

  async text(name) {
    if (!this.cache.has(name)) {
      const res = await fetch(outputUrl(this.id, name), { credentials: 'same-origin' });
      if (!res.ok) throw new Error('Couldn’t download the subtitles. Please try again.');
      this.cache.set(name, await res.text());
    }
    return this.cache.get(name);
  }

  /** The subtitle files by language: [{lang, label, items:{srt, vtt, txt}}], each item {name, bytes, text()}. No network. */
  tracks() {
    if (this.subs) {
      if (this.subs.empty) return [];
      const base = stem(this.file?.name);
      const item = (ext, name, text) => ({ name, bytes: new Blob([text]).size, text: async () => text, ext });
      return [{
        lang: this.subs.language || '', label: langName(this.subs.language),
        items: { srt: item('srt', `${base}.srt`, this.subs.srt), vtt: item('vtt', `${base}.vtt`, this.subs.vtt), txt: item('txt', `${base}-transcript.txt`, this.subs.txt) },
      }];
    }
    const caps = this.outputs.filter((o) => o.kind === 'captions' || o.kind === 'transcript');
    return [...new Set(caps.map((o) => o.lang || ''))].map((lang) => {
      const mine = caps.filter((o) => (o.lang || '') === lang);
      const find = (re) => mine.find((o) => re.test(o.name));
      const srt = find(/\.srt$/i);
      const vtt = find(/\.vtt$/i);
      const txt = find(/\.txt$/i);
      const real = (o, ext) => ({ name: o.name, bytes: o.bytes, text: () => this.text(o.name), ext });
      const items = {};
      if (srt) items.srt = real(srt, 'srt');
      if (vtt) items.vtt = real(vtt, 'vtt');
      if (txt) items.txt = real(txt, 'txt');
      // the two kinds of subtitle file hold the same words: either one makes the other
      if (!srt && vtt) items.srt = { name: vtt.name.replace(/\.vtt$/i, '.srt'), bytes: null, ext: 'srt', text: async () => vttToSrt(await this.text(vtt.name)) };
      if (!vtt && srt) items.vtt = { name: srt.name.replace(/\.srt$/i, '.vtt'), bytes: null, ext: 'vtt', text: async () => srtToVtt(await this.text(srt.name)) };
      return { lang, label: langName(lang), items };
    });
  }

  /** Which subtitle formats this job can still give. */
  availableCaps() {
    const have = { srt: false, vtt: false, txt: false };
    for (const t of this.tracks()) for (const k of Object.keys(t.items)) have[k] = true;
    return have;
  }

  /* ---- what the files are ---- */

  /** The files that will be (or were) made, with no work done: [{name, bytes, kind, label}]. */
  planned(w = this.effective()) {
    const out = [];
    const media = this.mediaPlan(w);
    if (media) out.push(media);
    if (w.subs) {
      for (const t of this.tracks()) {
        for (const id of ['srt', 'vtt', 'txt']) {
          const it = w.caps[id] && t.items[id];
          if (it) out.push({ name: it.name, bytes: it.bytes ?? null, kind: id === 'txt' ? 'transcript' : 'captions', label: captionLabel(id, t, this.tracks().length > 1) });
        }
      }
    }
    return out;
  }

  mediaPlan(w) {
    if (!w.sound && !w.embed) return null;
    if (this.useServerVideo && (w.embed || FORMAT[w.format]?.kind === 'video' || !this.serverAudio)) {
      const v = this.serverVideo;
      return { name: v.name, bytes: v.bytes, kind: 'media', label: 'Video', url: outputUrl(this.id, v.name) };
    }
    const def = FORMAT[w.format];
    const video = def?.kind === 'video';
    const direct = !video && !w.soundHere && this.serverAudio && def.exts.includes(endingOf(this.serverAudio.name));
    return {
      name: cleanName(w.name) || this.serverAudio?.name || 'result', bytes: direct ? this.serverAudio.bytes : null, kind: 'media',
      label: video ? (w.embed ? 'Video with subtitles' : 'Video') : 'Cleaned audio',
    };
  }

  /* ---- making them ---- */

  async make(w, { signal, onProgress }) {
    const report = (p) => onProgress?.(Math.max(0, Math.min(1, p)));
    const files = [];
    const tracks = w.subs ? this.tracks() : [];
    const media = this.mediaPlan(w);
    let note = '';
    if (media?.url) {
      files.push({ name: media.name, url: media.url, bytes: media.bytes, kind: 'media', label: media.label });
    } else if (media) {
      const def = FORMAT[w.format];
      const info = this.inputInfo;
      const video = def.kind === 'video';
      const picture = video && info.hasVideo;
      const sound = {};
      if (w.sound) {
        if (this.pcm && w.soundHere) {
          sound.pcm = this.pcm; sound.wav = this.wav;
        } else {
          const o = this.serverAudio;
          if (!o) throw new Error('The cleaned sound isn’t here any more. Please run it again.');
          const res = await fetch(outputUrl(this.id, o.name), { credentials: 'same-origin' });
          if (!res.ok) throw new Error('Couldn’t download the cleaned audio. Please try again.');
          sound.audioFile = await res.blob();
          sound.audioName = o.name;
        }
      } else {
        sound.audioFile = this.file; // the subtitles go into the video with its own sound
        sound.audioName = this.file.name;
      }
      report(0.05);
      let subtitles = null;
      if (w.embed && video && tracks.length) {
        subtitles = [];
        for (const t of tracks) subtitles.push({ text: t.items.vtt ? await t.items.vtt.text() : srtToVtt(await t.items.srt.text()), lang: t.lang || undefined, name: t.label });
      }
      const out = await buildMedia({
        format: w.format, size: w.size || 'same', videoFile: picture ? this.file : null, info, ...sound, subtitles,
        onProgress: (p) => report(0.05 + p * 0.95), signal,
      });
      note = out.note || '';
      files.push({ name: cleanName(w.name), blob: out.blob, bytes: out.blob.size, kind: 'media', label: media.label });
    }
    for (const t of tracks) {
      for (const id of ['srt', 'vtt', 'txt']) {
        const it = w.caps[id] && t.items[id];
        if (!it) continue;
        const blob = new Blob([await it.text()], { type: CAPTION_TYPES[id] });
        files.push({ name: it.name, blob, bytes: blob.size, kind: id === 'txt' ? 'transcript' : 'captions', label: captionLabel(id, t, tracks.length > 1) });
      }
    }
    report(1);
    if (w.subs && !tracks.length && this.subs?.empty) note = 'We didn’t hear any speech in this recording, so there are no subtitles.';
    else if (w.subsHere && !this.subs && !this.failed.subs) note = 'The subtitles made in your browser were only kept in this tab.';
    return { files, note };
  }

  /* ---- running ---- */

  /** Make the files and (by default) download them. Runs one at a time. `wants`: new choices to save and use. */
  run(opts = {}) {
    const next = () => this.runNow(opts);
    this.chain = this.chain.then(next, next);
    return this.chain;
  }

  async runNow({ save = true, wants = null, signal = null } = {}) {
    if (wants) {
      this.wants = wants;
      saveWants(this.id, wants);
    }
    if (this.browser) {
      const waiting = this.browser;
      this.set({ phase: 'waiting', error: '' });
      await waiting;
      if (this.browser === waiting) this.browser = null;
    }
    const w = this.effective();
    if (this.needsVideo(w)) {
      this.set({ phase: 'needvideo', pending: { save, wants: null }, error: '' });
      return;
    }
    const ctl = new AbortController();
    this.ctl = ctl;
    signal?.addEventListener('abort', () => ctl.abort(), { once: true });
    this.set({ phase: 'building', progress: 0, error: '', note: '', pending: null });
    try {
      const { files, note } = await this.make(w, { signal: ctl.signal, onProgress: (p) => this.set({ progress: p }) });
      if (ctl.signal.aborted) throw abortError();
      this.made = files;
      this.set({ files: files.map(meta), note, progress: 1 });
      if (save && files.length) {
        this.set({ phase: 'saving' });
        await downloadAll(files, ctl.signal);
      }
      this.set({ phase: 'ready' });
    } catch (e) {
      if (e?.name === 'AbortError' || ctl.signal.aborted) {
        this.set({ phase: this.made ? 'ready' : 'idle', error: '' });
      } else {
        this.set({ phase: 'error', error: e?.message || 'Couldn’t put your files together. Please try again.' });
      }
    } finally {
      if (this.ctl === ctl) this.ctl = null;
    }
  }

  /** The job just finished while this tab was open: finish it by itself, once. */
  auto() {
    if (this.autoStarted || !this.thisTab) return;
    this.autoStarted = true;
    this.run({ save: true });
  }

  /** "Download again": the files that were made, or (after a reload) made now. */
  again() {
    if (this.made?.length && this.state.phase !== 'building') {
      this.set({ phase: 'saving' });
      downloadAll(this.made).catch(() => {}).then(() => this.set({ phase: 'ready' }));
      return Promise.resolve();
    }
    return this.run({ save: true });
  }

  cancel() { this.ctl?.abort(); }

  /** The person chose the original video (needed once, after a reload): carry on where it stopped. */
  async setVideo(file) {
    this.set({ error: '' });
    const info = await probeMedia(file);
    const want = this.job?.durationSeconds;
    if (!info.hasVideo || info.viaLibav) {
      this.set({ phase: 'needvideo', error: info.hasVideo ? 'This browser can’t put the sound back into that kind of video.' : 'That file has no picture. Please choose the original video.' });
      return;
    }
    if (want && info.duration && Math.abs(info.duration - want) > Math.max(3, want * 0.05)) {
      this.set({ phase: 'needvideo', error: 'That doesn’t look like the same video (it’s a different length). Please choose the original.' });
      return;
    }
    this.file = file;
    this.info = info;
    const pending = this.state.pending || { save: true };
    this.run(pending);
  }
}

function captionLabel(id, track, several) {
  if (id === 'txt') return `Transcript${several ? ` · ${track.label}` : ''}`;
  return `Subtitles${several ? ` · ${track.label}` : ''}`;
}

const meta = (f) => ({ name: f.name, bytes: f.bytes, kind: f.kind, label: f.label });

function saveItem(f) {
  if (f.blob) {
    saveBlob(f.blob, f.name);
    return;
  }
  const a = document.createElement('a');
  a.href = f.url;
  a.download = f.name;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/** Download the files one after another (the browser may ask once to allow several downloads). */
export async function downloadAll(files, signal) {
  for (let i = 0; i < files.length; i++) {
    if (i) await sleep(GAP_MS);
    if (signal?.aborted) return;
    saveItem(files[i]);
  }
}

/* ------------------------------------------------------------------------------------------------ the jobs of this tab */

const finishes = new Map();

/** The Finish of a job (made when first asked for: a job listed after a reload has no file or sound of this tab). */
export function finishOf(job) {
  let f = finishes.get(job.id);
  if (!f) {
    f = new Finish(job.id);
    finishes.set(job.id, f);
  }
  f.update(job);
  return f;
}

/** A job just started in this tab: remember its file and choices so it can finish by itself. */
export function beginFinish(job, { file, info, wants, browser }) {
  const f = new Finish(job.id);
  Object.assign(f, { file, info, wants, browser, thisTab: true });
  if (job.local) f.autoStarted = true; // a free run is finished by the form that made it, before its card shows
  finishes.set(job.id, f);
  f.update(job);
  if (!job.local) saveWants(job.id, wants);
  return f;
}

export const getFinish = (id) => finishes.get(id) || null;

export function forgetFinish(id) {
  finishes.delete(id);
  try { localStorage.removeItem(KEY + id); } catch { /* nothing kept */ }
}

export function useFinish(fin) {
  return useSyncExternalStore(fin.subscribe, fin.getSnapshot);
}

/* ---- the cards of free runs (no job on our side): shown in the list like the others, kept for as long as the tab is ---- */

let localCards = [];
const cardListeners = new Set();
const setCards = (next) => { localCards = next; for (const fn of cardListeners) fn(); };

export function addLocalCard(job) { setCards([job, ...localCards]); }
export function removeLocalCard(id) { forgetFinish(id); setCards(localCards.filter((j) => j.id !== id)); }
export function useLocalCards() {
  return useSyncExternalStore((fn) => { cardListeners.add(fn); return () => cardListeners.delete(fn); }, () => localCards);
}
