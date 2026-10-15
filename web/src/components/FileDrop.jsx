import { useRef, useState } from 'react';
import { IconUpload } from './Icons.jsx';

// What browsers can open themselves.
export const BROWSER_FORMATS = 'audio/*,video/*,.mp3,.wav,.flac,.ogg,.oga,.opus,.m4a,.aac,.mp4,.m4v,.mov,.webm,.mkv,.aiff,.aif';
// What the server reads (Cleanup page): nearly anything FFmpeg understands. Keep in step with server/acw/jobs.py.
export const SERVER_FORMATS = `${BROWSER_FORMATS},.rf64,.w64,.mp2,.mpa,.m4b,.m4r,.spx,.wma,.aifc,.caf,.au,.snd,.amr,.ac3,.eac3,.dts,.mka,.wv,.ape,.tta,.mpc,.avi,.wmv,.asf,.flv,.f4v,.3gp,.3g2,.ts,.mts,.m2ts,.mpg,.mpeg,.vob,.mxf,.dv,.ogv,.rm,.rmvb`;

/** Drag-and-drop / click-to-choose file picker. */
export default function FileDrop({ onFile, disabled, accept = BROWSER_FORMATS, title = 'Drop an audio or video file here', hint = 'or click to pick one. MP3, WAV, FLAC, M4A, MP4, MOV, WebM, MKV and more.' }) {
  const [over, setOver] = useState(false);
  const input = useRef(null);
  const pick = (files) => {
    const f = files && files[0];
    if (f) onFile(f);
  };
  return (
    <label
      className="dropzone"
      data-over={over}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); if (!disabled) pick(e.dataTransfer.files); }}
    >
      <IconUpload width={34} height={34} />
      <span className="big">{title}</span>
      <span className="muted small">{hint}</span>
      <input
        ref={input}
        type="file"
        accept={accept}
        disabled={disabled}
        onChange={(e) => { pick(e.target.files); e.target.value = ''; }}
        aria-label="Choose an audio or video file"
      />
    </label>
  );
}
