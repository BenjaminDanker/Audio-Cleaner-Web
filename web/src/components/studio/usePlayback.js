import { useCallback, useEffect, useRef, useState } from 'react';
import { dbOf } from '../../editor/tracks.js';
import { MixPlayer } from '../../editor/mix.js';
import { BlendPlayer, gainsFor, matchGain, memorySource } from '../../media/blend.js';
import { AbEngine, ElementEngine, syncPicture } from '../../studio/engine.js';

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/**
 * What plays in the editor and where the playhead is: the mix of what is made (a job of ours, `canMix`), the two signals a free cleanup
 * made here kept (`localBlend`), the original against the result (`localAb`), or the file itself; the picture follows the sound.
 * `live.current.model` is the mix's model as it is now (read while it plays).
 */
export function usePlayback({ session, job, fin, info, S, T, ver, hear, canMix, localBlend, localAb, live }) {
  const [ph, setPh] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playInfo, setPlayInfo] = useState({ buffering: false, error: '' });
  const [fileUrl, setFileUrl] = useState('');
  const [audioUrl, setAudioUrl] = useState('');
  const [abUrl, setAbUrl] = useState('');
  const [, setEngineVer] = useState(0);
  const mediaRef = useRef(null);
  const engineRef = useRef(null);
  const stopAt = useRef(null);

  useEffect(() => {
    if (!session.file || !info?.hasVideo) { setFileUrl(''); return undefined; }
    const u = URL.createObjectURL(session.file);
    setFileUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [session.file, info?.hasVideo]);
  useEffect(() => {
    if (!session.file || info?.hasVideo) { setAudioUrl(''); return undefined; }
    const u = URL.createObjectURL(session.file);
    setAudioUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [session.file, info?.hasVideo]);
  useEffect(() => { setAbUrl(localAb ? fin.wavUrl : ''); }, [localAb, fin?.wavUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    // the engine that plays: the mix of what is made (a job), the kept signals (a card made here), or the file itself
    let eng = null;
    let abEl = null;
    const onEnd = () => { setPlaying(false); setPh((p) => p); };
    const el = mediaRef.current;
    try {
      if (canMix) {
        eng = new MixPlayer({ jobId: job.id, preview: session.server.preview, getModel: () => live.current.model, onState: setPlayInfo, onEnd });
        eng.prepare();
      } else if (localBlend && fin.pairParts) {
        fin.ensurePair().then((pair) => {
          if (engineRef.current !== eng2) return;
          const bp = new BlendPlayer(memorySource(pair), { onEnd, onState: setPlayInfo });
          engineRef.current = bp;
          eng = bp;
          bp.prepare();
          const v0 = live.current.S.tracks.voice[0];
          const g0 = gainsFor(v0 && v0.mode === 1 ? dbOf(v0) : fin.currentStrength());
          bp.setGains(g0.full, g0.input);
          setEngineVer((n) => n + 1);
        }).catch(() => {});
      } else if (localAb && el && abUrl) {
        abEl = new Audio(abUrl);
        abEl.preload = 'auto';
        eng = new AbEngine(el, abEl, { onEnd });
      } else if (el) {
        eng = new ElementEngine(el, { onEnd });
      }
    } catch (e) {
      setPlayInfo({ buffering: false, error: 'Couldn’t get the sound to listen with. Please try again.' });
    }
    const eng2 = eng;
    engineRef.current = eng;
    setPlaying(false);
    return () => { try { (engineRef.current || eng)?.close(); } catch { /* closed */ } engineRef.current = null; abEl?.pause(); };
  }, [canMix, localBlend, localAb, abUrl, job?.id, job?.finishedAt, session.server?.preview, fileUrl, audioUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  // the speaker buttons and the tracks change what is heard at once
  useEffect(() => {
    const eng = engineRef.current;
    if (!eng) return;
    if (eng instanceof MixPlayer) eng.remix();
    else if (eng instanceof AbEngine) eng.setHearCleaned(hear.voice);
    else if (eng instanceof BlendPlayer && fin) {
      const v0 = S.tracks.voice[0];
      const db = v0 && v0.mode === 1 ? dbOf(v0) : fin.currentStrength(); // (the strength being tried, not only the one saved)
      const sums = eng.source.sums ? eng.source.sums() : null;
      const g = hear.voice ? gainsFor(db, { sums }) : { full: 0, input: sums ? matchGain(sums, 1) : 1 };
      eng.setGains(g.full, g.input);
    }
  }, [ver, hear, playing]); // eslint-disable-line react-hooks/exhaustive-deps

  // the playhead follows what plays; the picture follows the sound
  useEffect(() => {
    if (!playing) return undefined;
    const t = setInterval(() => {
      const eng = engineRef.current;
      if (!eng) return;
      const tt = eng.time;
      const end = stopAt.current != null ? stopAt.current : T;
      if (tt >= end - 0.02) { eng.pause(); stopAt.current = null; setPlaying(false); setPh(Math.min(end, T)); return; }
      setPh(tt);
      if (mediaRef.current && !(eng instanceof ElementEngine) && !(eng instanceof AbEngine)) syncPicture(mediaRef.current, eng, true);
    }, 100);
    return () => clearInterval(t);
  }, [playing, T]);
  useEffect(() => {
    const el = mediaRef.current;
    const eng = engineRef.current;
    if (el && eng && !(eng instanceof ElementEngine) && !(eng instanceof AbEngine)) { el.muted = true; syncPicture(el, eng, playing); }
  }, [playing]);

  const seek = useCallback((t) => {
    const x = clamp(t, 0, T);
    setPh(x);
    engineRef.current?.seek(x);
    if (mediaRef.current && !(engineRef.current instanceof ElementEngine) && !(engineRef.current instanceof AbEngine)) mediaRef.current.currentTime = x;
  }, [T]);
  const togglePlay = useCallback(async (from = null, stop = null) => {
    const eng = engineRef.current;
    if (!eng) return;
    if (from != null) { await eng.seek(from); setPh(from); }
    stopAt.current = stop;
    if (eng.playing && from == null) { eng.pause(); setPlaying(false); return; }
    setPlaying(true);
    try { await eng.play(); } catch { setPlaying(false); }
  }, []);

  return { ph, playing, playInfo, fileUrl, audioUrl, mediaRef, engineRef, seek, togglePlay };
}
