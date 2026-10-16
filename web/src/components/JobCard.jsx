import { useRef, useState } from 'react';
import { Link } from 'react-router';
import { api, outputUrl } from '../api.js';
import { useAuth } from '../auth.jsx';
import { saveBlob } from '../download.js';
import { bytes, clock, dateTime, human, money, shownName, stem } from '../format.js';
import { useNow } from '../hooks.js';
import { ACTIVE, jobTitle, kindLabel, queueText, stageText, statusLabel } from '../jobs.js';
import { buildMedia, canHoldSubtitles, defaultFormatId, FORMAT, srtToVtt } from '../media/convert.js';
import { IconDownload } from './Icons.jsx';

const OUT_LABEL = { audio: 'Cleaned audio', video: 'Video', captions: 'Captions', transcript: 'Transcript', segments: 'Segments' };

function badgeClass(status) {
  if (status === 'done') return 'badge badge-ok';
  if (status === 'failed' || status === 'canceled') return 'badge badge-bad';
  if (ACTIVE.has(status)) return 'badge badge-run';
  return 'badge';
}

function Countdown({ expiresAt }) {
  const now = useNow(30_000);
  if (!expiresAt) return null;
  const left = expiresAt - now / 1000;
  if (left <= 0) return <span className="muted small">Your files are being cleared away.</span>;
  return <span className="muted small" data-testid="expiry">Files are cleared away in {human(Math.max(60, Math.round(left / 60) * 60))}</span>;
}

/** The job's subtitles as WebVTT text (from its .vtt file, or its .srt one), for putting into a video here. */
async function jobSubtitles(job) {
  const caps = job.outputs.filter((o) => o.kind === 'captions');
  const pick = caps.find((o) => /\.vtt$/i.test(o.name)) || caps.find((o) => /\.srt$/i.test(o.name));
  if (!pick) throw new Error('This job has no subtitles to put into the video.');
  const res = await fetch(outputUrl(job.id, pick.name), { credentials: 'same-origin' });
  if (!res.ok) throw new Error('Couldn’t download the subtitles. Please try again.');
  const text = await res.text();
  return /\.vtt$/i.test(pick.name) ? text : srtToVtt(text);
}

/** The file type a video with subtitles is made as when nothing was chosen: a WebM stays WebM, anything else becomes MKV. */
function subtitledFormat(file, info, inputName) {
  const id = defaultFormatId(file || { name: inputName }, info || { hasVideo: true }, true);
  return canHoldSubtitles(id) ? id : 'mkv';
}

/**
 * Add the cleaned audio back into the original video, locally, as chosen in the form (file type, picture size, name).
 * For a recording with no picture and a video type chosen, it makes the video (plain black picture) the same way.
 * With `local.embed` the job's subtitles go into the video too. A subtitles-only job (no cleaned audio from us) keeps the
 * video's own sound, or the sound cleaned in this browser when that was chosen as well.
 */
function RemuxPanel({ job, local, noPicture = false }) {
  const [file, setFile] = useState(local?.file || null);
  const [busy, setBusy] = useState(false);
  const [prog, setProg] = useState(0);
  const [err, setErr] = useState('');
  const [note, setNote] = useState('');
  const picker = useRef(null);
  const ctl = useRef(null);
  const audioOut = job.outputs.find((o) => o.kind === 'audio');
  const embed = !!local?.embed;
  if (!audioOut && !embed) return null;
  const choice = local?.output && FORMAT[local.output.format]?.kind === 'video' ? local.output : null;
  const waitingForBrowser = !audioOut && local?.browserClean && !local?.cleanPcm; // the free cleanup in this tab isn't done yet

  async function run(videoFile) {
    const abort = new AbortController();
    ctl.current = abort;
    setBusy(true); setErr(''); setNote(''); setProg(0);
    try {
      const subtitles = embed ? await jobSubtitles(job) : '';
      let audioFile = null;
      if (audioOut) {
        const res = await fetch(outputUrl(job.id, audioOut.name), { credentials: 'same-origin' });
        if (!res.ok) throw new Error('Couldn’t download the cleaned audio. Please try again.');
        audioFile = await res.blob();
      }
      const pcm = !audioOut && local?.cleanPcm ? local.cleanPcm : null;
      if (!audioOut && !pcm) audioFile = videoFile; // subtitles only: the video keeps its own sound
      const format = choice?.format || subtitledFormat(videoFile, local?.info, job.inputName);
      const { remuxWithCleanAudio } = await import('../media/remux.js');
      const out = noPicture
        ? await buildMedia({ format, audioFile, audioName: audioOut?.name, subtitles, onProgress: setProg, signal: abort.signal })
        : await remuxWithCleanAudio({
          videoFile, pcm, audioFile, audioName: audioOut?.name || '', info: local?.info, format, size: choice?.size, subtitles, onProgress: setProg, signal: abort.signal,
        });
      const suffix = audioOut || pcm ? 'cleaned' : 'subtitled';
      saveBlob(out.blob, choice?.name || `${stem(videoFile?.name || job.inputName)}-${suffix}${out.ext}`);
      if (out.note) setNote(out.note);
    } catch (e) {
      if (e?.name !== 'AbortError') setErr(e.message || 'Couldn’t build the video. Please try again.');
    } finally {
      ctl.current = null;
      setBusy(false);
    }
  }

  const label = busy ? `Building video… ${Math.round(prog * 100)}%` : waitingForBrowser ? 'Waiting for the noise removal in your browser…' : null;
  const what = audioOut ? 'cleaned audio' : local?.cleanPcm ? 'cleaned audio and subtitles' : 'subtitles';
  return (
    <div className="stack" data-testid="remux">
      <div className="row">
        {noPicture ? (
          <button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={() => run(null)} data-testid="remux-btn">
            {label || `Save as ${shownName(choice.name)}`}
          </button>
        ) : file ? (
          <button type="button" className="btn btn-secondary btn-sm" disabled={busy || waitingForBrowser} onClick={() => run(file)} data-testid="remux-btn">
            {label || (choice ? `Save ${shownName(choice.name)}` : `Put the ${what} into ${shownName(file.name)}`)}
          </button>
        ) : (
          <>
            <button type="button" className="btn btn-secondary btn-sm" disabled={waitingForBrowser} onClick={() => picker.current?.click()}>{label || `Put the ${what} into your video…`}</button>
            <input
              ref={picker} type="file" accept="video/*,.mkv,.mov,.mp4,.webm" className="sr-only" tabIndex={-1}
              aria-label="Choose the original video"
              onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ''; if (f) { setFile(f); run(f); } }}
            />
          </>
        )}
        {busy && <button type="button" className="btn btn-ghost btn-sm" onClick={() => ctl.current?.abort()}>Cancel</button>}
        <span className="small muted">This happens on your device; the video isn’t uploaded.</span>
      </div>
      {note && <p className="small banner banner-info mb0">{note}</p>}
      {err && <p className="small banner banner-error mb0" role="alert">{err}</p>}
    </div>
  );
}

/** Failed and canceled jobs cost nothing: say so once (the server's own messages often say it already). */
const withNotCharged = (text) => (/not charged|weren.t charged/i.test(text) ? text : `${text} You weren’t charged.`);

export default function JobCard({ job, local, onChanged }) {
  const { contact } = useAuth();
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const active = ACTIVE.has(job.status);
  const canCancel = job.status === 'uploading' || job.status === 'queued';
  const hasVideoInput = local?.info?.hasVideo || /\.(mp4|mov|m4v|mkv|webm)$/i.test(job.inputName || '');
  const hasVideoOut = job.outputs.some((o) => o.kind === 'video');
  const hasAudioOut = job.outputs.some((o) => o.kind === 'audio');
  // Subtitles put into the video here, in the browser (chosen in the form in this tab); the server never needs the video for it.
  const embedHere = !!local?.embed && job.outputs.some((o) => o.kind === 'captions');
  const showRemux = job.status === 'done' && hasVideoInput && (local?.extracted !== false || embedHere) && local?.keepVideo !== false && !hasVideoOut && (hasAudioOut || embedHere);
  const chosenKind = FORMAT[local?.output?.format]?.kind;
  // A video type was chosen for a recording with no picture: the video (plain black picture) is made here from the cleaned sound.
  const showBlack = job.status === 'done' && chosenKind === 'video' && !hasVideoInput && !hasVideoOut && !!job.outputs.find((o) => o.kind === 'audio');
  // A sound type was chosen: the file keeps the name chosen in the form.
  const audioName = chosenKind === 'audio' ? local.output.name : null;

  // The server names its files its own way (and its header wins over the link's name): the name chosen in the form is applied here.
  async function saveRenamed(o, name) {
    setErr('');
    try {
      const res = await fetch(outputUrl(job.id, o.name), { credentials: 'same-origin' });
      if (!res.ok) throw new Error('Couldn’t download that file. Please try again.');
      saveBlob(await res.blob(), name);
    } catch (e) {
      setErr(e.message || 'Couldn’t download that file. Please try again.');
    }
  }

  async function act(fn) {
    setBusy(true); setErr('');
    try {
      await fn();
      onChanged?.();
    } catch (e) {
      setErr(e.detail || e.message);
    } finally {
      setBusy(false);
    }
  }

  const pct = typeof job.progress === 'number' ? job.progress : 0;
  const step = stageText(job); // '' when the server's step name is not one we know: then only the percentage is shown
  const stepLine = step ? `${step}${pct ? ` · ${Math.round(pct)}%` : ''}` : pct ? `${Math.round(pct)}% done` : 'Just a moment…';
  return (
    <article className="card job" data-testid="job" data-status={job.status} data-job-id={job.id}>
      <div className="job-head">
        <div>
          <div className="job-title">{jobTitle(job, local)}</div>
          <div className="small muted">{kindLabel(job)} · {dateTime(job.createdAt)} · {clock(job.durationSeconds)}</div>
        </div>
        <span className="row">
          {job.free && <span className="badge badge-free" data-testid="free-badge">{job.kind === 'transcribe' ? 'Free' : 'Free subtitles'}</span>}
          <span className={badgeClass(job.status)}>{statusLabel(job.status)}</span>
        </span>
      </div>

      {job.status === 'queued' && (
        <p className="queue-note mb0" role="status" data-testid="queue-note">
          <span className="spinner" aria-hidden="true" /> {queueText(job) || 'In line, you’re up soon'}
        </p>
      )}
      {(job.status === 'running' || job.status === 'cloud_asr') && (
        <div>
          <progress className="bar" max={100} value={pct} aria-label={`Job progress, ${step || statusLabel(job.status)}`} />
          <p className="small muted mb0" role="status">{stepLine}</p>
        </div>
      )}

      {job.status === 'failed' && (
        <>
          <p className="banner banner-error mb0" role="alert">{withNotCharged(job.error || 'This one didn’t work out.')}</p>
          {contact && <p className="small muted mb0">Still stuck? <Link to={`/contact?topic=job&job=${job.id}`} data-testid="job-contact">Tell us about this job</Link>.</p>}
        </>
      )}
      {job.status === 'canceled' && <p className="muted mb0">{withNotCharged(job.error && job.error !== 'Canceled.' ? job.error : 'Canceled.')}</p>}

      {job.status === 'done' && (
        <>
          <ul className="outputs" aria-label="Downloads">
            {job.outputs.map((o) => (
              <li key={o.name}>
                <span><span className="oname">{o.name}</span> <span className="muted small">· {OUT_LABEL[o.kind] || o.kind} · {bytes(o.bytes)}</span></span>
                <a className="btn btn-primary btn-sm" href={outputUrl(job.id, o.name)} download={o.name} data-testid="download"
                  onClick={o.kind === 'audio' && audioName ? (e) => { e.preventDefault(); saveRenamed(o, audioName); } : undefined}>
                  <IconDownload width={16} height={16} /> Download
                </a>
              </li>
            ))}
          </ul>
          {job.notice && <p className="banner banner-info small mb0" role="status" data-testid="job-notice">{job.notice}</p>}
          {showRemux && <RemuxPanel job={job} local={local} />}
          {showBlack && <RemuxPanel job={job} local={local} noPicture />}
          {hasVideoOut && <p className="small muted mb0">The video file above already has the clean sound in it.</p>}
          <div className="row-between">
            <span className="small muted">Cost <b>{money(job.costMicros || 0)}</b> · <Countdown expiresAt={job.expiresAt} /></span>
            {confirmDelete ? (
              <span className="row" data-testid="delete-confirm">
                <span className="small">Delete these files now? They can’t be brought back.</span>
                <button type="button" className="btn btn-danger btn-sm" disabled={busy} onClick={() => act(() => api(`jobs/${job.id}`, { method: 'DELETE' }))}>Yes, delete</button>
                <button type="button" className="btn btn-ghost btn-sm" disabled={busy} onClick={() => setConfirmDelete(false)}>Keep them</button>
              </span>
            ) : (
              <button type="button" className="btn btn-danger-outline btn-sm" disabled={busy} onClick={() => setConfirmDelete(true)}>Delete now</button>
            )}
          </div>
        </>
      )}

      {canCancel && (
        <div><button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={() => act(() => api(`jobs/${job.id}/cancel`, { method: 'POST' }))}>Cancel job</button></div>
      )}
      {err && <p className="banner banner-error small mb0" role="alert">{err}</p>}
    </article>
  );
}

