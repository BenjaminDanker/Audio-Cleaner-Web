/* What the editor screen says, worked out from where it stands: the progress line while it runs, the bar's middle and its button, the
   files a start will download and the notes at the top. No React in here (components/studio/Studio.jsx shows them). */
import { etaSeconds, etaText } from '../eta.js';
import { balanceMoney, bytes, human } from '../format.js';
import { queueText, stageText } from '../jobs.js';
import { money } from '../editor/tracks.js';
import { PAID_OFFLINE } from '../components/Maintenance.jsx';

/** While it runs: { text, value (percent or null), cancel (the cancel button's words, or '') }, or null when nothing runs. */
export function progressLine({ processing, job, isLocal, saving, session, denoiserLoad, speechLoad }) {
    if (!processing) return null;
    if (job && !isLocal) {
      const pct = typeof job.progress === 'number' ? job.progress : 0;
      const step = stageText(job);
      const eta = etaText(etaSeconds(job, Date.now()));
      let text;
      if (saving && (job.status === 'done' || job.status === 'uploading')) text = 'Making the new parts';
      else if (job.status === 'uploading') text = 'Waiting for the upload to finish';
      else if (job.status === 'queued') text = [queueText(job) || 'In line, you’re up soon', eta].filter(Boolean).join(' · ');
      else text = [step ? `${step}${pct ? ` · ${Math.round(pct)}%` : ''}` : pct ? `${Math.round(pct)}% done` : 'Just a moment…', eta].filter(Boolean).join(' · ');
      const value = job.status === 'running' || job.status === 'cloud_asr' ? pct : null;
      const canCancel = job.status === 'uploading' || job.status === 'queued';
      return { text, value, cancel: canCancel ? 'Cancel job' : '' };
    }
    const tasks = Object.entries(session.run?.tasks || {});
    if (!tasks.length) return { text: 'Getting things ready…', value: null, cancel: 'Cancel job' };
    const [, t] = tasks.sort((a, b) => (a[1].frac ?? 0) - (b[1].frac ?? 0))[0];
    const pct = t.frac != null ? Math.floor(t.frac * 100) : null;
    const loading = /ready…$/.test(t.text || '');
    let text = t.text;
    if (/noise remover ready/.test(t.text) && denoiserLoad.total) text = `${t.text} ${(denoiserLoad.loaded / 1048576).toFixed(1)} of ${(denoiserLoad.total / 1048576).toFixed(1)} MB`;
    if (/subtitle maker ready/.test(t.text) && speechLoad.total) text = `${t.text} ${bytes(speechLoad.loaded)} of ${bytes(speechLoad.total)} (only the first time)`;
    return {
      text: [text, !loading && pct != null ? `${pct}%` : '', t.eta ? `about ${human(t.eta)} left` : '', t.rate > 0 ? `${t.rate.toFixed(1)}× real time` : ''].filter(Boolean).join(' · '),
      value: pct,
      cancel: session.run?.cancel === false ? '' : 'Cancel job',
    };
}

/** The files a start downloads, in words ("talk-cleaned.mp4 and the subtitles (SRT, VTT)"). */
export function downloadText({ plan, choice, session }) {
    const parts = [];
    if (plan.cleaning || plan.converting) parts.push(`${choice.name.trim()}${plan.embedNow ? ' (with the subtitles inside)' : ''}`);
    else if (plan.embedNow) parts.push(`${(session.fileName || 'audio').replace(/\.[^.]+$/, '')}-subtitled.mkv (the video with the subtitles inside)`);
    if (plan.wantsText) {
      const f = [['srt', 'SRT'], ['vtt', 'VTT'], ['txt', 'TXT']].filter(([k]) => session.out.sfiles.includes(k)).map(([, l]) => l);
      if (f.length) parts.push(`the subtitles (${f.join(', ')})`);
    }
    return parts.join(' and ');
}

/** The bar: its middle (`mid`), whether a result is being put together (`building`) and its button. */
export function barOf({ stage, account, estMicros, server, processing, saving, procLine, building, finState, changed, deltaMicros, needVideo }) {
    if (stage === 'ready') {
      return {
        mid: { kind: 'details' }, building: false,
        button: account
          ? { label: 'Sign in to start', testid: 'start', state: 'signin' }
          : { label: estMicros != null || !server ? `Start · ${money(estMicros ?? 0)}` : 'Start', testid: 'start', state: 'start' },
      };
    }
    if (processing) {
      const lbl = (saving ? 'Saving' : 'Processing') + (procLine?.value != null ? ` · ${Math.round(procLine.value)}%` : '');
      return { mid: { kind: 'progress', ...procLine }, building: false, button: { label: lbl, testid: 'processing', state: 'processing' } };
    }
    if (building) {
      const p = Math.round((finState.progress || 0) * 100);
      return {
        mid: { kind: 'progress', text: finState.phase === 'waiting' ? 'Waiting for your browser to finish…' : 'Putting it together…', value: finState.phase === 'waiting' ? null : p, cancel: finState.phase === 'building' ? 'Cancel' : '' },
        building: true, button: { label: finState.phase === 'saving' ? 'Downloading…' : `Processing · ${p}%`, testid: 'processing', state: 'building' },
      };
    }
    const label = account ? 'Sign in to start' : changed ? `Save changes · ${money(deltaMicros)}` : finState.phase === 'error' ? 'Try again' : needVideo ? 'Choose the original video' : 'Download again';
    return { mid: { kind: 'files' }, building: false, button: { label, testid: changed ? 'save-changes' : 'download-again', state: changed ? 'save' : 'download' } };
}

/** The notes at the top of the screen: { key, cls, text, testid?, kind? } ('credit': with a link to add credit, 'prices': with Try again). */
export function notesOf({
  offline, session, stage, signedIn, problems, keepOpen, welcome, job, isLocal, done, finState, needVideo, fin, playInfo, changedTracks, quote, affordableSave,
}) {
  const notes = [];
  if (offline) notes.push({ key: 'offline', cls: 'banner banner-warn', text: PAID_OFFLINE, testid: 'paid-offline' });
  if (session.restoredNote && stage === 'ready') notes.push({ key: 'back', cls: 'banner banner-success', text: `${signedIn ? 'You’re signed in. ' : ''}Your file and choices are back as you left them.`, testid: 'draft-restored' });
  if (session.error) notes.push({ key: 'error', cls: session.errorKind === 'storage' ? 'banner banner-warn' : 'banner banner-error', text: session.error, testid: 'job-error' });
  if (stage === 'ready') problems.forEach((p) => notes.push({ key: p.text, cls: 'banner banner-warn', text: p.text, kind: p.kind }));
  if (stage === 'ready' && session.lengthUnknown) notes.push({ key: 'len', cls: 'hint', text: 'Your browser can’t read the length of this file type, so the numbers here are a rough guess from its size. We measure it after the upload and you pay for the real length.', testid: 'length-unknown' });
  if (keepOpen) notes.push({ key: 'open', cls: 'hint', text: 'Keep this tab open while it works.', testid: 'keep-open' });
  if (welcome && job && !isLocal) notes.push({ key: 'welcome', cls: 'banner banner-success', text: welcome, testid: 'submitted' });
  if (job?.notice) notes.push({ key: 'notice', cls: 'banner banner-info small', text: job.notice, testid: 'job-notice' });
  if (done && finState.error) notes.push({ key: 'fin', cls: 'banner banner-error small', text: finState.error, testid: 'made-error' });
  if (done && needVideo && !finState.error) notes.push({ key: 'nv', cls: 'hint', text: 'To put the picture back, choose your original video once. It stays on your device.', testid: 'need-video' });
  if (done && fin?.pcm && fin.pcm.rms < 1e-5) notes.push({ key: 'silent', cls: 'banner banner-warn', text: 'The cleaned audio is nearly silent. The original may only contain noise, or try a gentler setting.' });
  if (session.serverError) notes.push({ key: 'serr', cls: 'banner banner-error', text: session.serverError, testid: 'editor-error' });
  if (playInfo.error) notes.push({ key: 'perr', cls: 'banner banner-error small', text: playInfo.error, testid: 'play-error' });
  if (done && changedTracks && quote.state === 'error') notes.push({ key: 'qerr', cls: 'banner banner-warn small', text: quote.error, testid: 'quote-error' });
  if (done && changedTracks && quote.state === 'ok' && !affordableSave) notes.push({ key: 'poor', cls: 'banner banner-warn small', text: `This costs about ${money(quote.q.estCostMicros)}, but your balance is ${balanceMoney(quote.q.availableMicros)}.`, kind: 'credit' });
  return notes;
}
