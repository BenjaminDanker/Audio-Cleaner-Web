import { useMemo, useRef, useState } from 'react';
import { outputUrl } from '../api.js';
import { bytes, shownName } from '../format.js';
import { useFinish } from '../finish.js';
import { canHoldSubtitles, FORMAT } from '../media/convert.js';
import AbPlayer from './AbPlayer.jsx';
import { FinishProgress } from './Progress.jsx';
import { IconDownload } from './Icons.jsx';
import OutputFields, { useOutputChoice } from './OutputPicker.jsx';

const BUSY = new Set(['waiting', 'building', 'saving']);
const CAPS = [['srt', 'SRT'], ['vtt', 'VTT'], ['txt', 'TXT']];

/** "Change and save again": the choices that can change without doing the work again, and Save. */
function SaveAgain({ fin, job }) {
  const st = useFinish(fin);
  const w = fin.effective();
  const avail = fin.availableCaps();
  const anyCaps = Object.values(avail).some(Boolean);
  const inputName = fin.inputName;
  const info = useMemo(() => fin.info || { hasVideo: /\.(mp4|mov|m4v|mkv|webm)$/i.test(inputName), duration: job.durationSeconds || null }, [fin.info, inputName, job.durationSeconds]);
  const file = useMemo(() => ({ name: inputName }), [inputName]);
  const hasMedia = w.sound && !fin.useServerVideo; // the file type and name can still change
  const choice = useOutputChoice({ file: hasMedia ? file : null, info: hasMedia ? info : null, initial: hasMedia ? { name: w.name, format: w.format, size: w.size } : null });
  const [caps, setCaps] = useState({ srt: !!w.caps.srt, vtt: !!w.caps.vtt, txt: !!w.caps.txt });
  const [embed, setEmbed] = useState(!!w.embed);
  const busy = BUSY.has(st.phase);
  const videoFile = hasMedia ? FORMAT[choice.format]?.kind === 'video' : true;
  const canEmbed = w.subs && anyCaps && info.hasVideo && !fin.useServerVideo && info.viaMediabunny !== false && info.viaLibav !== true && videoFile;
  const embedOn = canEmbed && embed;
  const problem = embedOn && hasMedia && !canHoldSubtitles(choice.format) ? 'Subtitles put into the video need an MKV file. Pick MKV, or untick “Put the subtitles into the video”.' : '';
  const picked = Object.entries(caps).some(([id, on]) => on && avail[id]);
  const something = hasMedia || embedOn || (w.subs && picked);
  const valid = !busy && !problem && something && (!hasMedia || choice.valid);

  function save() {
    fin.run({
      save: true,
      wants: {
        ...w,
        format: hasMedia ? choice.format : w.format,
        name: hasMedia ? choice.name.trim() : w.name,
        size: hasMedia ? choice.size : w.size,
        caps: { srt: caps.srt && avail.srt, vtt: caps.vtt && avail.vtt, txt: caps.txt && avail.txt },
        embed: embedOn,
      },
    });
  }

  return (
    <div className="stack again" data-testid="again-fields">
      {hasMedia && <OutputFields choice={choice} info={info} idPrefix={`again-${job.id}`} disabled={busy} />}
      {w.subs && anyCaps && (
        <fieldset className="field" disabled={busy}>
          <legend className="label">Subtitle files</legend>
          <div className="checks">
            {CAPS.filter(([id]) => avail[id]).map(([id, label]) => (
              <label className="check" key={id}>
                <input type="checkbox" checked={caps[id]} onChange={(e) => setCaps({ ...caps, [id]: e.target.checked })} data-testid={`again-cap-${id}`} />
                <span>{label}</span>
              </label>
            ))}
          </div>
        </fieldset>
      )}
      {canEmbed && (
        <label className="check">
          <input
            type="checkbox" checked={embed} disabled={busy} data-testid="again-embed"
            onChange={(e) => {
              setEmbed(e.target.checked);
              if (e.target.checked && hasMedia && !canHoldSubtitles(choice.format) && choice.okIds?.includes('mkv')) choice.pickFormat('mkv');
            }}
          />
          <span><b>Put the subtitles into the video</b></span>
        </label>
      )}
      {problem && <p className="hint hint-bad mb0" role="alert">{problem}</p>}
      <div className="row">
        <button type="button" className="btn btn-primary" onClick={save} disabled={!valid} data-testid="again-save">
          <IconDownload width={18} height={18} /> {busy ? 'Putting it together…' : 'Save'}
        </button>
        <span className="small muted">Saves the files again with these choices.</span>
      </div>
    </div>
  );
}

/**
 * Everything about a finished job, at the bottom of its card: the files it made, one "Download again" for all of them,
 * the listen toggle, and the choices that can still change. The same for a job done on our side and a free one made here.
 */
export default function ResultPanel({ job, fin }) {
  const st = useFinish(fin);
  const picker = useRef(null);
  const [listen, setListen] = useState(false);
  const [again, setAgain] = useState(false);
  const w = fin.effective();
  const files = st.files.length ? st.files : fin.planned(w);
  const busy = BUSY.has(st.phase);
  const needVideo = st.phase === 'needvideo' || (st.phase === 'idle' && fin.needsVideo(w));
  const listenUrl = fin.wavUrl || (fin.serverAudio && w.sound && !w.soundHere ? outputUrl(job.id, fin.serverAudio.name) : '');
  const canListen = !!fin.file && !!listenUrl;
  const silent = fin.pcm && fin.pcm.rms < 1e-5;
  const preview = fin.subs && !fin.subs.empty ? fin.subs.txt.split(/\n+/).filter(Boolean).slice(0, 3).join(' ') : '';
  const pcm = fin.pcm;

  const onDownload = () => {
    if (needVideo) picker.current?.click();
    else fin.again();
  };

  return (
    <div
      className="result stack" data-testid={st.phase === 'ready' ? 'result' : 'result-pending'}
      {...(pcm ? { 'data-rms': pcm.rms.toFixed(6), 'data-peak': pcm.peak.toFixed(4), 'data-frames': pcm.frames, 'data-channels': pcm.channels } : {})}
      {...(fin.subs ? { 'data-cues': fin.subs.cues.length, 'data-language': fin.subs.language } : {})}
    >
      {(st.phase === 'building' || st.phase === 'waiting') && (
        st.phase === 'waiting'
          ? <p className="mb0" role="status" data-testid="waiting"><span className="spinner" aria-hidden="true" /> Waiting for your browser to finish…</p>
          : <FinishProgress progress={st.progress} onCancel={() => fin.cancel()} />
      )}
      {silent && <div className="banner banner-warn">The cleaned audio is nearly silent. The original may only contain noise, or try a gentler setting.</div>}
      {preview && <p className="muted mb0" data-testid="subs-preview">&ldquo;{preview.length > 220 ? `${preview.slice(0, 220)}…` : preview}&rdquo;</p>}

      {files.length > 0 && (
        <ul className="outputs" aria-label="Your files" data-testid="made">
          {files.map((f) => (
            <li key={f.name} data-testid="made-file">
              <span className="oname">{shownName(f.name)}</span>
              <span className="muted small">{f.label}{f.bytes != null ? ` · ${bytes(f.bytes)}` : ''}</span>
            </li>
          ))}
        </ul>
      )}
      {st.note && <p className="small banner banner-info mb0" role="status" data-testid="made-note">{st.note}</p>}
      {job.notice && <p className="banner banner-info small mb0" role="status" data-testid="job-notice">{job.notice}</p>}
      {st.error && <p className="small banner banner-error mb0" role="alert" data-testid="made-error">{st.error}</p>}
      {needVideo && !st.error && <p className="small muted mb0" data-testid="need-video">To put the picture back, choose your original video once. It stays on your device.</p>}

      {files.length > 0 && (
        <div className="row">
          <button type="button" className="btn btn-primary" onClick={onDownload} disabled={busy} data-testid="download-again">
            <IconDownload width={18} height={18} /> {st.phase === 'saving' ? 'Downloading…' : st.phase === 'error' ? 'Try again' : needVideo ? 'Choose the original video' : 'Download again'}
          </button>
          <input
            ref={picker} type="file" accept="video/*,.mkv,.mov,.mp4,.webm" className="sr-only" tabIndex={-1}
            aria-label="Choose the original video"
            onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ''; if (f) fin.setVideo(f); }}
          />
          {files.length > 1 && <span className="small muted">Downloads all {files.length} files. If your browser asks, allow it.</span>}
        </div>
      )}

      {canListen && (
        <details className="more" onToggle={(e) => setListen(e.currentTarget.open)} data-testid="listen">
          <summary>Listen: original or cleaned</summary>
          {listen && <AbPlayer original={fin.file} cleanedUrl={listenUrl} isVideo={!!fin.info?.hasVideo} />}
        </details>
      )}
      <details className="more" onToggle={(e) => setAgain(e.currentTarget.open)} data-testid="again">
        <summary>Change and save again</summary>
        {again && <SaveAgain fin={fin} job={job} />}
      </details>
    </div>
  );
}
