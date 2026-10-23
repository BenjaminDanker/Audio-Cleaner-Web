import { shownName } from '../../format.js';
import { fmt } from '../../editor/tracks.js';
import { IcAll, IcMore, IcPause, IcPlay, IcRedo, IcSpark, IcSplit, IcUndo, IcZoomIn, IcZoomOut } from './icons.jsx';
import { tipAttr } from './ui.jsx';

/** The top of the editor: the picture (or the poster of a sound), the file's name and length, the Automatic switch with what it did and
 * the suggestions, and the tools (play, split, undo, zoom, select). Its buttons work through the screen's `data-act` clicks. */
export default function TopPane({ session, job, S, T, info, fileUrl, audioUrl, mediaRef, engineRef, done, localAb, ph, playing, signedIn, pend, reco, zoom, more, fileMeta }) {
  // the lines under the file name
  const autoLine = session.listening
    ? <span data-testid="reco-listening"><span className="spinner" aria-hidden="true" /> Listening to a little of your file to pick the best settings…</span>
    : session.applied && reco
      ? (
        <>
          <span className="auto-text"><b data-testid="reco-status">{session.custom ? 'Set automatically, then changed by you.' : 'Set automatically for this file.'}</b></span>
          {session.custom
            ? <button type="button" className="link" data-act="auto-again" data-testid="reco-apply">Use them again</button>
            : <button type="button" className="link" data-act="auto-undo" data-testid="reco-undo">Undo</button>}
        </>
      )
      : null;
  const sgLine = session.sg.phase === 'running'
    ? (
      <span className="autol" data-testid="sg-progress" role="status">
        <span className="spinner" aria-hidden="true" /> Listening to your file for moments worth a look… {Math.round(session.sg.progress * 100)}% <button type="button" className="link" data-act="sg-cancel" data-testid="sg-cancel">Cancel</button>
      </span>
    )
    : session.auto && session.sg.phase === 'unavailable' && !job ? <span className="autol muted" data-testid="sg-unavailable">{session.sg.why}</span>
      : session.sg.phase === 'canceled' ? <span className="autol muted" data-testid="sg-canceled">Stopped looking. You can still choose parts yourself.</span>
        : session.sg.phase === 'error' ? <span className="autol muted" data-testid="sg-error">Couldn’t look for suggestions in this file. You can still choose parts yourself.</span> : null;

  return (
    <div className="top" id="top">
      <div className={`vid${info?.hasVideo ? '' : ' aud'}`}>
        {info?.hasVideo && fileUrl ? <video ref={mediaRef} src={fileUrl} playsInline preload="auto" muted={done && !localAb} data-testid="video" /> : null}
        {!info?.hasVideo || !fileUrl ? <span data-testid="poster">{session.kind === 'video' ? 'Video' : 'Audio'}</span> : null}
        {info?.hasVideo && <span className="vt js-ph">{fmt(ph)}</span>}
        {!info?.hasVideo && audioUrl && <audio ref={mediaRef} src={audioUrl} preload="auto" />}
      </div>
      <div className="tinfo">
        <div className="tname">
          <h1 className="fname" data-testid="file-name">{shownName(session.fileName)}</h1>
          <p className="fmeta" data-testid="file-meta">{fileMeta} · <button type="button" className="link" data-act="newfile" data-testid="different-file">Different file</button></p>
        </div>
        <div className="autoc">
          <div className="autow" data-testid="auto-box">
            <label className="auto" data-testid="auto-settings">
              <button type="button" className="sw" role="switch" aria-checked={session.auto} aria-labelledby="auto-l" data-act="auto" disabled={!session.hasAudio} {...tipAttr(signedIn ? '' : 'Without an account, Automatic picks among the free options, in your browser. Sign in for the others too.')} />
              <span id="auto-l">Automatic</span>
            </label>
            {pend > 0 && (
              <button type="button" className="chip lk-b" data-act="sgchip" aria-label={`${pend} suggestions. Show the next one`} data-testid="sg-chip">{pend}<IcSpark /></button>
            )}
          </div>
          {session.auto && autoLine && <p className="autol" aria-live="polite" data-testid="reco">{autoLine}</p>}
          {session.auto && session.applied && reco && (
            <details className="auto-why" data-testid="reco-details">
              <summary>Why these choices</summary>
              <p className="muted small mb0"><span data-testid="reco-line">{reco.line}</span> <span data-testid="reco-pick">{reco.pick}</span></p>
              {reco.reasons?.length > 0 && <ul className="reco-why small muted" data-testid="reco-why">{reco.reasons.map((r) => <li key={r}>{r}</li>)}</ul>}
            </details>
          )}
          {sgLine}
        </div>
      </div>
      <div className="tools">
        <button type="button" className="btn btn-primary tp o0" data-act="play" aria-label={playing ? 'Pause' : 'Play'} data-testid="play" disabled={!engineRef.current && !mediaRef.current}>{playing ? <IcPause /> : <IcPlay />}</button>
        <span className="tt o0"><span className="js-ph" data-testid="time">{fmt(ph)}</span><span className="lg"> / {fmt(T)}</span></span>
        <button type="button" className="btn btn-secondary btn-sm tb lk-b o3" data-act="split" data-testid="split"><IcSplit /><span className="tl"><span className="lg">Split at playhead</span><span className="sm">Split</span></span></button>
        <button type="button" className="btn btn-secondary btn-sm tb lk-b o1" data-act="undo" aria-label="Undo" title="Undo" disabled={!S.past.length} data-testid="undo"><IcUndo /></button>
        <button type="button" className="btn btn-secondary btn-sm tb lk-b o1" data-act="redo" aria-label="Redo" title="Redo" disabled={!S.future.length} data-testid="redo"><IcRedo /></button>
        <button type="button" className="btn btn-secondary btn-sm tb o1" data-act="zoom-out" aria-label="Zoom out" title="Zoom out" disabled={zoom <= 1} data-testid="zoom-out"><IcZoomOut /></button>
        <button type="button" className="btn btn-secondary btn-sm tb o1" data-act="zoom-in" aria-label="Zoom in" title="Zoom in" disabled={zoom >= 8} data-testid="zoom-in"><IcZoomIn /></button>
        <button type="button" className="btn btn-secondary btn-sm tb lk-b o2 push" data-act="all" aria-pressed={!S.sel ? 'true' : undefined} data-testid="select-all"><IcAll /><span className="tl"><span className="lg">Select all</span><span className="sm">All</span></span></button>
        <span className="brk" />
        <button type="button" className="btn btn-secondary btn-sm tb lk-b o3 tb-more" data-act="more" aria-pressed={more ? 'true' : undefined} data-testid="select-more"><IcMore /><span className="tl">Select more</span></button>
      </div>
    </div>
  );
}
