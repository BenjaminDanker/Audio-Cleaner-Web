import { useRef } from 'react';
import { bytes, human } from '../../format.js';
import { fmt, lenTxt, money } from '../../editor/tracks.js';
import { FORMAT } from '../../media/convert.js';
import { LANGUAGES } from '../../jobs.js';
import OutputFields from '../OutputPicker.jsx';
import { IcChevron, IcFile } from './icons.jsx';
import { tipAttr } from './ui.jsx';

const LANG = Object.fromEntries(LANGUAGES.map(([c, n]) => [c || 'auto', n]));
const typeLabel = (id) => FORMAT[id]?.label.replace(/ \(.*\)$/, '') || String(id).toUpperCase();

/** "Details": what the price is made of. */
function Details({ ctx }) {
  const { cost: c, total, deltaMicros, changed, serverTotal, partsNote, localPrice } = ctx;
  const L = [];
  const li = (a, b) => <li key={a}><span>{a}</span><span>{b}</span></li>;
  if (c.nPaid) L.push(li(`Noise removal · ${fmt(c.nPaid)}`, money(c.noise)));
  if (c.nFree) L.push(li(`Noise removal, free · ${fmt(c.nFree)}`, 'Free'));
  if (c.rSec) L.push(li(`Voice rebuild · ${fmt(c.rSec)}`, money(c.rebuild)));
  if (c.eSec) L.push(li(`Echo removal · ${fmt(c.eSec)}`, money(c.echo)));
  if (c.lSec) L.push(li(`Leveling · ${fmt(c.lSec)}`, money(c.level)));
  if (c.sStd) L.push(li(`Subtitles, Standard · ${fmt(c.sStd)}`, money(c.sStd * ctx.rates.std)));
  if (c.sExt) L.push(li(`Subtitles, Extra accurate · ${fmt(c.sExt)}`, money(c.sExt * ctx.rates.ext)));
  if (c.trSec) L.push(li(`Translations · ${c.trLangs.map((l) => LANG[l]).join(', ')}`, money(c.tr)));
  if (c.sFree) L.push(li(`Subtitles, free · ${fmt(c.sFree)}`, 'Free'));
  const cv = ctx.conversion;
  const big = cv?.factor > 1 ? ` · counts ${Math.round(cv.factor * 10) / 10}× (bigger or faster than 1080p at 30 fps)` : '';
  if (cv?.free) L.push(li(cv.black ? 'Video with a plain black picture, on our side' : cv.server ? 'Video on our side, the picture copied as it is' : 'File type or size, in your browser', 'Free'));
  else if (cv?.maybe) L.push(li(`Video conversion, only if the picture has to be made again (MKV never needs it)${big}`, `up to ${money(cv.micros)}`));
  else if (cv) L.push(li(`Video conversion${big}`, money(cv.micros)));
  return (
    <div className="pop pl-r" role="dialog" aria-label="Details" data-testid="details">
      <ul className="fl">
        {L.length ? L : li('Nothing switched on', '')}
        <li className="tot"><span>{changed ? 'Total so far' : 'Total'}</span><span data-testid="details-total">{money(serverTotal != null ? serverTotal : total)}</span></li>
        {changed && <li className="tot"><span>Save changes</span><span>{money(deltaMicros)}</span></li>}
      </ul>
      {partsNote && <p className="note">Includes a small amount for each run. Parts are rounded up to whole seconds.</p>}
      {localPrice && <p className="note" data-testid="details-local">Worked out from the price list. Once you sign in, the price also includes a small amount for each run.</p>}
      <p className="note">You only pay for what gets processed. Billed per second of audio, no minimum charge.</p>
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

/** The output: file type and name (OutputFields as it is), the video size, the subtitle files and "into the video". */
function Output({ ctx }) {
  const { choice, info, plan, out, cost: c, tips, disabled, willDownload, tooBigNote, mkvNote } = ctx;
  const sa = plan.wantsText;
  const burnTip = plan.embedOnServer ? tips['burn.server'] : `${tips.burn}${mkvNote ? `, so you’ll get an MKV instead of ${mkvNote}` : ''}`;
  return (
    <div className="pop pl-out" role="dialog" aria-label="Output" data-testid="output-pop">
      <div className="of"><OutputFields choice={choice} info={info} idPrefix="out" disabled={disabled} /></div>
      <div className="of">
        <span className="srl">Subtitle files</span>
        <div className={`pills${sa ? '' : ' dis'}`}>
          {['srt', 'vtt', 'txt'].map((k) => (
            <label className="pill" key={k} {...tipAttr(tips[`sf.${k}`])}>
              <input type="checkbox" data-act="o-sf" value={k} checked={out.sfiles.includes(k)} disabled={!sa || disabled} data-testid={`cap-${k}`} />{k.toUpperCase()}
            </label>
          ))}
          {plan.videoResult && (
            <label className="pill" {...tipAttr(burnTip)}>
              <input type="checkbox" data-act="o-burn" checked={out.burn && plan.canEmbed} disabled={!sa || !plan.canEmbed || disabled} aria-label="Put the subtitles into the video" data-testid="embed" />Put into the video
            </label>
          )}
        </div>
        {sa && c.trLangs.length > 0 && <span className="note mt0">One set of subtitle files per language.</span>}
        {tooBigNote && <span className="note mt0" data-testid="embed-too-big">{tooBigNote}</span>}
      </div>
      {willDownload && <p className="note mt0" data-testid="will-download">When it’s done, these download by themselves: <b>{willDownload}</b>.</p>}
    </div>
  );
}

/** The bar at the bottom: the output, what the price is made of or the files, and the one button. */
export default function Bar({ ctx }) {
  const { stage, pop, choice, bar, h, primaryDisabled } = ctx;
  const picker = useRef(null);
  const link = (v, label, icon) => (
    <button type="button" className="link bl" data-act="pop" data-v={v} aria-label={label} data-testid={`bar-${v}`}><span className="lg">{label}</span><span className="sm"><i>{icon}</i></span></button>
  );
  const stem = choice.name.replace(/\.[^.]+$/, '');
  const { mid, button, building } = bar;
  const needVideo = ctx.needVideo;
  return (
    <div className="bb" data-stage={stage} data-testid="bar" data-building={building ? 'yes' : 'no'}>
      <div className="container bb-in">
        {pop === 'out' && <Output ctx={ctx} />}
        {pop === 'files' && <Files ctx={ctx} />}
        {pop === 'det' && <Details ctx={ctx} />}
        <button type="button" className="btn btn-secondary outb lk-b" data-act="pop" data-v="out" aria-expanded={pop === 'out'} data-testid="output-button">
          <IcFile /><span className="outt"><b>{typeLabel(choice.format)}</b><span className="onm"> · {stem}</span></span><IcChevron />
        </button>
        <span className="bb-sp" />
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
