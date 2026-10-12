import { useRef, useState } from 'react';
import { IconUpload } from './Icons.jsx';

const ACCEPT = 'audio/*,video/*,.mp3,.wav,.flac,.ogg,.oga,.opus,.m4a,.aac,.mp4,.m4v,.mov,.webm,.mkv,.ts,.aiff,.aif,.caf,.wma,.avi';

/** Drag-and-drop / click-to-choose file picker. */
export default function FileDrop({ onFile, disabled, title = 'Drop an audio or video file here', hint = 'or click to choose a file. MP3, WAV, FLAC, M4A, MP4, MOV, WebM, MKV and more.' }) {
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
        accept={ACCEPT}
        disabled={disabled}
        onChange={(e) => { pick(e.target.files); e.target.value = ''; }}
        aria-label="Choose an audio or video file"
      />
    </label>
  );
}
