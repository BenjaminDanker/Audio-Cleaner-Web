import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { clock } from '../format.js';

/**
 * A/B player: the original file (audio or video element) and the cleaned audio play side by side; toggling only
 * mutes one and unmutes the other, so you hear both at the same position. The one you are listening to sets the
 * time and the other follows it: a big video the computer can't decode in time then never holds the cleaned sound
 * back (the picture jumps to catch up instead, the way the browser drops late frames).
 */
export default function AbPlayer({ original, cleanedUrl, isVideo }) {
  const origUrl = useMemo(() => URL.createObjectURL(original), [original]);
  useEffect(() => () => URL.revokeObjectURL(origUrl), [origUrl]);

  const master = useRef(null);
  const slave = useRef(null);
  const [mode, setMode] = useState('cleaned');
  const [origOk, setOrigOk] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [dur, setDur] = useState(0);
  const modeRef = useRef(mode);
  modeRef.current = mode;

  const lead = useCallback(
    () => (origOk && mode === 'original' ? master.current : slave.current),
    [origOk, mode],
  );
  const follower = useCallback(
    () => (!origOk ? null : mode === 'original' ? slave.current : master.current),
    [origOk, mode],
  );

  // keep mute state in sync with the chosen mode
  useEffect(() => {
    if (master.current) master.current.muted = mode !== 'original';
    if (slave.current) slave.current.muted = mode !== 'cleaned' && origOk;
  }, [mode, origOk, playing]);

  useEffect(() => {
    const a = lead();
    const b = follower();
    if (!a) return undefined;
    const onPlay = () => { setPlaying(true); if (b) { b.currentTime = a.currentTime; b.play().catch(() => {}); } };
    const onPause = () => { setPlaying(false); b?.pause(); };
    const onSeek = () => { setTime(a.currentTime); if (b) b.currentTime = a.currentTime; };
    // A video follows more loosely than audio: each jump costs it a decode from the previous key frame.
    const slack = b && b.tagName === 'VIDEO' ? 0.5 : 0.15;
    const onTime = () => {
      setTime(a.currentTime);
      if (b && !a.paused && Math.abs(b.currentTime - a.currentTime) > slack) b.currentTime = a.currentTime;
      if (b && !a.paused && b.paused) b.play().catch(() => {});
    };
    const onMeta = () => setDur(Number.isFinite(a.duration) ? a.duration : 0);
    const onEnd = () => { setPlaying(false); b?.pause(); };
    a.addEventListener('play', onPlay);
    a.addEventListener('pause', onPause);
    a.addEventListener('seeked', onSeek);
    a.addEventListener('timeupdate', onTime);
    a.addEventListener('loadedmetadata', onMeta);
    a.addEventListener('ended', onEnd);
    if (a.readyState >= 1) onMeta();
    return () => {
      a.removeEventListener('play', onPlay);
      a.removeEventListener('pause', onPause);
      a.removeEventListener('seeked', onSeek);
      a.removeEventListener('timeupdate', onTime);
      a.removeEventListener('loadedmetadata', onMeta);
      a.removeEventListener('ended', onEnd);
    };
  }, [lead, follower, origUrl, cleanedUrl]);

  // Switching what you hear while playing: the new leader takes over from the old one's position.
  const prevMode = useRef(mode);
  useEffect(() => {
    if (prevMode.current === mode || !origOk) return;
    prevMode.current = mode;
    const a = lead();
    const b = follower();
    if (a && b) {
      a.currentTime = b.currentTime;
      if (!b.paused) a.play().catch(() => {});
    }
  }, [mode, origOk, lead, follower]);

  const toggle = () => {
    const a = lead();
    if (!a) return;
    if (a.paused) a.play().catch(() => {});
    else a.pause();
  };
  const seek = (v) => {
    const a = lead();
    if (!a) return;
    a.currentTime = v;
    setTime(v);
  };
  const onOrigError = () => { setOrigOk(false); setMode('cleaned'); };

  const Media = isVideo ? 'video' : 'audio';
  return (
    <div className="ab">
      <Media
        ref={master}
        src={origUrl}
        preload="auto"
        playsInline={isVideo || undefined}
        onError={onOrigError}
        className={isVideo && origOk ? undefined : 'sr-only'}
        data-testid="ab-original"
      />
      <audio ref={slave} src={cleanedUrl} preload="auto" data-testid="ab-cleaned" />
      {!origOk && <p className="banner banner-info small mb0">Your browser cannot play the original file, so only the cleaned audio is available here.</p>}
      <div className="ab-modes" role="radiogroup" aria-label="What you hear">
        <div className="seg">
          <label>
            <input type="radio" name="ab" value="original" checked={mode === 'original'} disabled={!origOk} onChange={() => setMode('original')} />
            <span>Original</span>
          </label>
          <label>
            <input type="radio" name="ab" value="cleaned" checked={mode === 'cleaned'} onChange={() => setMode('cleaned')} />
            <span>Cleaned</span>
          </label>
        </div>
        <span className="muted small" aria-live="polite">Listening to the {mode} audio</span>
      </div>
      <div className="transport">
        <button type="button" className="btn btn-primary" onClick={toggle} aria-label={playing ? 'Pause' : 'Play'}>{playing ? 'Pause' : 'Play'}</button>
        <input
          type="range"
          min={0}
          max={dur || 0}
          step="0.01"
          value={Math.min(time, dur || 0)}
          onChange={(e) => seek(Number(e.target.value))}
          aria-label="Seek"
          aria-valuetext={`${clock(time)} of ${clock(dur)}`}
        />
        <span className="time">{clock(time)} / {clock(dur)}</span>
      </div>
    </div>
  );
}
