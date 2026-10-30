import { useRef } from 'react';
import { Link } from 'react-router';
import { bytes, human } from '../../format.js';
import { fmt, lenTxt, money } from '../../editor/tracks.js';
import { LANGUAGES } from '../../jobs.js';
import OutputFields from '../OutputPicker.jsx';

const LANG = Object.fromEntries(LANGUAGES.map(([c, n]) => [c || 'auto', n]));

/** A step's start-up per run, as seconds of audio: "2.3 s", "1:38". */
const startup = (f) => (f < 60 ? `${Math.round(f * 10) / 10} s` : fmt(f));

/** "Details": what the price is made of, line by line (editor/tracks.js priceOf: the same rules as our side), adding up to the total. */
function Details({ ctx }) {
  const { price, total, deltaMicros, changed } = ctx;
  const L = [];
  const li = (key, a, b) => <li key={key}><span>{a}</span><span>{b}</span></li>;
  for (const l of price?.lines || []) {
    const what = l.item === 'translate' ? `${l.label} · ${l.languages.map((c) => LANG[c] || c).join(', ')} · ${fmt(l.seconds)}` : `${l.label} · ${fmt(l.seconds)}`;
    const plus = l.fixedSeconds > 0 ? ` + ${startup(l.fixedSeconds)} start-up` : '';
    L.push(li(l.item, what + plus, l.free ? 'Free' : money(l.micros)));
  }
  const cv = ctx.conversion;
  const big = cv?.factor > 1 ? ` · counts ${Math.round(cv.factor * 10) / 10}× (bigger or faster than 1080p at 30 fps)` : '';
  if (cv?.free) L.push(li('cv', cv.black ? 'Video with a plain black picture, on our side' : cv.server ? 'Video on our side, the picture copied as it is' : 'File type or size, in your browser', 'Free'));
  else if (cv) L.push(li('cv', `Video conversion${big}`, money(cv.micros)));
  const anyStartup = (price?.lines || []).some((l) => l.fixedSeconds > 0);
  return (
    <div className="pop pl-r" role="dialog" aria-label="Details" data-testid="details">
      <ul className="fl">
        {L.length ? L : li('none', 'Nothing switched on', '')}
        <li className="tot"><span>{changed ? 'Total so far' : 'Total'}</span><span data-testid="details-total">{money(total ?? price?.micros ?? 0)}</span></li>
        {changed && <li className="tot"><span>Save changes</span><span>{money(deltaMicros)}</span></li>}
      </ul>
      {anyStartup && <p className="note">Each step also counts its start-up once per run (loading what it needs), at the same rate as its audio.</p>}
      <p className="note">Each line is its seconds of audio, rounded up to a whole second, at that step&rsquo;s price a minute. No minimum charge.</p>
    </div>
  );
}

/** "Your files": names, labels and sizes, the subtitle preview, Download again's note, the cost and the time left, Delete now. */
function Files({ ctx }) {
  const { files, preview, session, job, expiresIn, delAsk, paid, finState } = ctx;
  const local = !!job?.local;
  return (
    <div className="pop pl-r" role="dialog" aria-label="Your files" data-testid="files">
      {preview && <p className="quote" data-testid="subs-preview">“{preview.length > 220 ? `${preview.slice(0, 220)}…` : preview}”</p>}
      <ul className="fl" data-testid="made">
        {files.map((f) => (
          <li key={f.name} data-testid="made-file"><span>{f.name} <small className="muted">{f.label}</small></span><span className="muted">{f.bytes != null ? bytes(f.bytes) : ''}</span></li>
        ))}
      </ul>
      {files.length > 1 && <p className="note">Downloads all {files.length} files. If your browser asks, allow it.</p>}
      {finState?.note && <p className="note" role="status" data-testid="made-note">{finState.note}</p>}
      {local ? (
        <>
          <p className="note">Kept in this tab only, until you leave or reload the page.</p>
          <button type="button" className="btn btn-ghost btn-sm mt8" data-act="del-yes" data-testid="remove-local">Remove</button>
        </>
      ) : (
        <>
          <p className="note">
            Cost <b>{money(paid)}</b>
            {expiresIn != null && (expiresIn <= 0
              ? ' · Your files are being cleared away.'
              : <> · <span data-testid="expiry">Files are cleared away in {human(Math.max(60, Math.round(expiresIn / 60) * 60))}</span><span data-testid="expiry-rule"> (2 hours after your last change, at most 6 hours after the upload)</span></>)}
          </p>
          {delAsk ? (
            <>
              <p className="note" data-testid="delete-confirm">Delete everything of this file now: your upload, the results and any other versions made from it? They can’t be brought back.</p>
              <div className="delr">
                <button type="button" className="btn btn-danger btn-sm" data-act="del-yes" data-testid="delete-yes">Yes, delete</button>
                <button type="button" className="btn btn-ghost btn-sm" data-act="del-no">Keep them</button>
              </div>
            </>
          ) : <button type="button" className="btn btn-danger-outline btn-sm mt8" data-act="del" data-testid="delete-now">Delete now</button>}
        </>
      )}
      {session.deleteError && <p className="note" role="alert">{session.deleteError}</p>}
    </div>
  );
}

/** The output, under the editor: the file type and name, and the video size (OutputFields as it is), and what downloads by itself. */
export function OutputPane({ ctx }) {
  const { choice, info, disabled, willDownload } = ctx;
  return (
    <section className="pane outp lk" aria-label="Your file" data-testid="output-pane">
      <OutputFields choice={choice} info={info} idPrefix="out" disabled={disabled} />
      {willDownload && <p className="note mt0" data-testid="will-download">When it’s done, these download by themselves: <b>{willDownload}</b>.</p>}
    </section>
  );
}

/** The start row, under the editor and the settings: what the price is made of (or the files, or the progress), the balance, and the one button. */
export default function Bar({ ctx }) {
  const { stage, pop, bar, h, primaryDisabled, wallet } = ctx;
  const picker = useRef(null);
  const link = (v, label, icon) => (
    <button type="button" className="link bl" data-act="pop" data-v={v} aria-label={label} data-testid={`bar-${v}`}><span className="lg">{label}</span><span className="sm"><i>{icon}</i></span></button>
  );
  const { mid, button, building } = bar;
  const needVideo = ctx.needVideo;
  return (
    <div className="pane bb" data-stage={stage} data-testid="bar" data-building={building ? 'yes' : 'no'}>
      <div className="bb-in">
        {pop === 'files' && <Files ctx={ctx} />}
        {pop === 'det' && <Details ctx={ctx} />}
        {mid.kind === 'details' && link('det', 'Details', 'i')}
        {mid.kind === 'files' && link('files', 'Your files', '☰')}
        {mid.kind === 'progress' && (
          <>
            <div className="pg" data-testid="run-progress" role="status">
              <progress className="bar" max={100} value={mid.value ?? undefined} aria-label="Progress" data-testid="job-progress" />
              <span className="muted" data-testid="job-line">{mid.text}</span>
            </div>
            {mid.cancel && <button type="button" className="link bl" data-act="cancel" data-testid="cancel-job">{mid.cancel}</button>}
          </>
        )}
        <span className="bb-sp" />
        {wallet && (
          <Link to="/app/billing" className={`bal${wallet.short ? ' short' : ''}`} data-testid="bar-balance" aria-label={`Balance ${wallet.text}${wallet.short ? ', not enough for this. Add credit' : ''}`}>
            <span className="muted">Balance</span> <b>{wallet.text}</b>
          </Link>
        )}
        <button
          type="button" className="btn btn-primary btn-lg" data-act="primary" disabled={primaryDisabled} data-testid={button.testid} data-state={button.state}
        >
          {button.label}
        </button>
        {needVideo && (
          <input
            ref={picker} type="file" accept="video/*,.mkv,.mov,.mp4,.webm" className="sr-only" tabIndex={-1} aria-label="Choose the original video" data-testid="need-video-input"
            onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ''; if (f) h.onVideo(f); }}
          />
        )}
      </div>
    </div>
  );
}

export { lenTxt };
