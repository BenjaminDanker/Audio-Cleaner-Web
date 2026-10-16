import { useCallback, useEffect, useMemo, useState } from 'react';
import { stem } from '../format.js';
import {
  checkName, defaultFormatId, FORMAT, formatOfName, listOutputs, listSizes, nameForFormat, SIZES,
} from '../media/convert.js';

// "What should the result be called, and what kind of file?" A name box with a file-type list attached to its right,
// and (for a video) the picture size. Typing another ending changes the list; picking from the list changes the ending.
// Only what this browser can really make for this file is in the list. Used on the Cleanup form (paid cleaning: the choice
// is kept with the job in this tab) and on the free result (saved from there). Nothing here uploads anything.

const defaultName = (file, id) => `${stem(file?.name)}-cleaned.${FORMAT[id]?.ext || 'wav'}`;
const AUDIO_DEFAULT_FOR = (file) => (/\.webm$/i.test(file?.name || '') ? 'ogg' : 'm4a');

/**
 * State for the choice: {name, format, size}. `serverAudio`: our server writes the sound (any audio type is possible,
 * only the video files depend on this browser). `keepVideo` only sets where the choice starts.
 */
export function useOutputChoice({ file, info, keepVideo = true, serverAudio = false }) {
  const [outputs, setOutputs] = useState(null); // null: still asking the browser
  const [format, setFormatState] = useState('wav');
  const [name, setNameState] = useState('');
  const [size, setSize] = useState('same');
  const [sizes, setSizes] = useState(null);
  const [touched, setTouched] = useState(false);

  // another file: start over
  useEffect(() => {
    setOutputs(null);
    setSizes(null);
    setSize('same');
    setTouched(false);
    if (file && info) {
      const id = defaultFormatId(file, info, keepVideo);
      setFormatState(id);
      setNameState(defaultName(file, id));
    }
    // keepVideo only matters for the starting point, handled below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file, info]);

  useEffect(() => {
    if (!file || !info) return undefined;
    let alive = true;
    listOutputs({ info, serverAudio }).then((l) => { if (alive) setOutputs(l); }).catch(() => { if (alive) setOutputs([{ id: 'wav', kind: 'audio', label: 'WAV', ok: true, why: '' }]); });
    return () => { alive = false; };
  }, [file, info, serverAudio]);

  const okIds = useMemo(() => (outputs ? outputs.filter((o) => o.ok).map((o) => o.id) : [format]), [outputs, format]);

  // where it starts follows "keep the video" until the visitor chooses something themselves
  useEffect(() => {
    if (!file || !info || touched) return;
    const id = defaultFormatId(file, info, keepVideo);
    if (outputs && !okIds.includes(id)) return;
    setFormatState(id);
    setNameState(defaultName(file, id));
  }, [file, info, keepVideo, touched, outputs, okIds]);

  // a format this browser cannot make is never kept: fall back to the nearest one that works
  useEffect(() => {
    if (!outputs || okIds.includes(format)) return;
    const kind = FORMAT[format]?.kind;
    const order = [...outputs.filter((o) => o.ok && o.kind === kind), ...outputs.filter((o) => o.ok)];
    const id = order[0]?.id || 'wav';
    setFormatState(id);
    setNameState((n) => nameForFormat(n, id));
  }, [outputs, okIds, format]);

  const picture = !!info?.hasVideo && !info?.viaLibav && FORMAT[format]?.kind === 'video';
  useEffect(() => {
    setSizes(null);
    if (!picture) { setSize('same'); return undefined; }
    let alive = true;
    listSizes(info, format).then((l) => { if (alive) setSizes(l); }).catch(() => { if (alive) setSizes([{ ...SIZES[0], ok: true }]); });
    return () => { alive = false; };
  }, [picture, info, format]);
  useEffect(() => {
    if (sizes && !sizes.find((s) => s.id === size)?.ok) setSize('same');
  }, [sizes, size]);

  const typeName = useCallback((v) => {
    setNameState(v);
    setTouched(true);
    const id = formatOfName(v);
    if (id && okIds.includes(id)) setFormatState(id);
  }, [okIds]);
  const pickFormat = useCallback((id) => {
    setFormatState(id);
    setNameState((n) => nameForFormat(n, id));
    setTouched(true);
  }, []);

  const hint = file && info ? checkName(name, okIds) : '';
  return {
    outputs, sizes, format, name, size, setSize, typeName, pickFormat, hint, valid: !!file && !!info && !!outputs && !hint,
    okIds, picture, isVideoFile: FORMAT[format]?.kind === 'video',
    defaultVideo: () => defaultFormatId(file, info, true),
    defaultAudio: () => AUDIO_DEFAULT_FOR(file),
  };
}

/** Words for the formats that are not in the list. */
function missingNote(outputs) {
  const gone = (outputs || []).filter((o) => !o.ok);
  if (!gone.length) return '';
  const names = gone.map((o) => o.label.replace(/ \(.*\)$/, ''));
  const list = names.length === 1 ? names[0] : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
  return `This browser can’t make ${list} files for this recording, so ${names.length === 1 ? 'that one isn’t' : 'those aren’t'} in the list.`;
}

/**
 * The name box with its list, then the picture size. `idPrefix` keeps ids apart when two are on the page.
 */
export default function OutputFields({ choice, info, idPrefix = 'out', disabled = false, plainBlackNote = true }) {
  const { outputs, sizes, format, name, size, hint } = choice;
  const audio = (outputs || []).filter((o) => o.ok && o.kind === 'audio');
  const video = (outputs || []).filter((o) => o.ok && o.kind === 'video');
  const missing = missingNote(outputs);
  const blackPicture = choice.isVideoFile && !info?.hasVideo;
  const px = SIZES.find((s) => s.id === size)?.px || 0;
  const bigger = px && info?.width && info?.height && px > Math.min(info.width, info.height);
  const onlySame = choice.picture && sizes && sizes.filter((s) => s.ok).length <= 1;
  return (
    <div className="stack" data-testid="output-fields">
      <div className="field">
        <label htmlFor={`${idPrefix}-name`}>Name of the result</label>
        <div className="namefmt">
          <input
            id={`${idPrefix}-name`} className="input" type="text" value={name} data-testid="out-name"
            onChange={(e) => choice.typeName(e.target.value)} disabled={disabled}
            spellCheck={false} autoComplete="off" autoCapitalize="off"
            aria-invalid={hint ? 'true' : 'false'} aria-describedby={`${idPrefix}-hint`}
          />
          <select
            className="select" value={format} data-testid="out-format" aria-label="File type"
            onChange={(e) => choice.pickFormat(e.target.value)} disabled={disabled || !outputs}
          >
            {!outputs && <option value={format}>Checking…</option>}
            {audio.length > 0 && (
              <optgroup label="Sound only">
                {audio.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
              </optgroup>
            )}
            {video.length > 0 && (
              <optgroup label={info?.hasVideo ? 'Video' : 'Video (plain black picture)'}>
                {video.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
              </optgroup>
            )}
          </select>
        </div>
        <span id={`${idPrefix}-hint`} className={hint ? 'hint hint-bad' : 'hint'} data-testid="out-hint" role={hint ? 'alert' : undefined}>
          {hint || 'Tip: type another ending (.mp3, .flac, .mkv…) or pick a file type from the list. Each follows the other.'}
        </span>
      </div>

      {choice.picture && (
        <div className="field">
          <label htmlFor={`${idPrefix}-size`}>Video size</label>
          <select
            id={`${idPrefix}-size`} className="select" value={size} data-testid="out-size"
            onChange={(e) => choice.setSize(e.target.value)} disabled={disabled || !sizes}
          >
            {(sizes || SIZES.slice(0, 1)).filter((s) => s.ok !== false).map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
          </select>
          <span className="hint" data-testid="out-size-note">
            {onlySame
              ? 'This browser can’t re-encode this video, so only the original size is offered.'
              : bigger
                ? 'That’s bigger than the original. The picture is only stretched, so it won’t look sharper and the file gets bigger.'
                : px
                  ? 'The picture is made again at this size on your device, so it takes a little longer.'
                  : 'The picture is copied as it is whenever the new file type can hold it.'}
          </span>
        </div>
      )}
      {blackPicture && plainBlackNote && <p className="small muted mb0" data-testid="out-black-note">There’s no picture in your recording, so the video gets a plain black one.</p>}
      {missing && <p className="small muted mb0" data-testid="out-missing">{missing}</p>}
    </div>
  );
}
