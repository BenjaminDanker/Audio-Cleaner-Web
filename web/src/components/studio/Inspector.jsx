import { Link } from 'react-router';
import { LANGUAGES } from '../../jobs.js';
import { dbAt, posOf, strengthName } from '../../media/blend.js';
import { human } from '../../format.js';
import {
  echoBlocked, fmt, lenTxt, MAX_TRANSLATIONS, overl, ranges, REM, selTitle, STR, STR_DB, SUBQ, trOf, uni,
} from '../../editor/tracks.js';
import { LONG_SECONDS } from '../../studio/plan.js';
import { IcChevron, IcSpark } from './icons.jsx';
import { Mixed, Seg, tipAttr } from './ui.jsx';

const LANG = Object.fromEntries(LANGUAGES.map(([c, n]) => [c || 'auto', n]));
const LANG_OPTIONS = LANGUAGES.map(([c, n]) => [c || 'auto', n]);

/** The line under the title on a phone: what the selection does. */
function summary(S) {
  const vm = uni(S, 'voice', 'mode');
  let o = '';
  if (vm === null) o = 'Mixed';
  else if (vm === 0) o = 'Voice off';
  else if (vm === 2) o = 'Rebuild';
  else {
    const st = uni(S, 'voice', 'str');
    const k = uni(S, 'voice', 'kind');
    o = (k === 'free' ? 'Free ' : '') + (st === null ? 'Mixed' : STR[st]);
  }
  const q = uni(S, 'subs', 'q');
  if (q) o += `, ${SUBQ[q]}`; else if (q === null) o += ', subtitles mixed';
  return o;
}

/** The translations of the selected Standard / Extra accurate subtitles: one list, or null when the parts differ. */
function uniTr(S) {
  const L = overl(S, 'subs').filter((x) => x.q >= 2);
  const k = (x) => trOf(x).join('.');
  return !L.length ? [] : L.every((x) => k(x) === k(L[0])) ? trOf(L[0]) : null;
}
function trUnion(S) {
  const u = [];
  overl(S, 'subs').forEach((x) => trOf(x).forEach((l) => { if (!u.includes(l)) u.push(l); }));
  return u;
}

const clockMin = (t) => fmt(t);

/**
 * The inspector: everything that changes the selection. Voice (Off, Noise removal with Free | Standard, strength, remover, Voice rebuild),
 * Echo removal, Leveling, Subtitles; above them the suggestion for this part and the part's start and end. Clicks come back through
 * data-act (Studio.jsx); the sliders through the callbacks of `h`.
 */
export default function Inspector({ ctx }) {
  const {
    S, session, tips, sheet, info, verdict, sVerdict, rebuildPaused, locked, lockedWhy, eta, loadLeft, h,
  } = ctx;
  const R = ranges(S);
  const one = R.length === 1 ? R[0] : null;
  const sg = S.sel && S.sel.length === 1 && session.auto && S.sugg.find((g) => g.st === 'new' && g.s === R[0].s && g.e === R[0].e);
  const auto = (tr) => session.auto && overl(S, tr).every((x) => x.auto);
  const tag = (tr) => (auto(tr) ? <span className="atag">Automatic</span> : null);
  const vm = uni(S, 'voice', 'mode');
  const ukind = vm === 1 ? uni(S, 'voice', 'kind') : null;
  const ustr = vm === 1 ? uni(S, 'voice', 'str') : null;
  const urem = vm === 1 ? uni(S, 'voice', 'rem') : null;
  const udb = vm === 1 ? uni(S, 'voice', 'db') : null;
  const fr = ukind === 'free';
  const noFile = !session.file;
  const freeBad = !verdict.ok || noFile;
  const freeWhy = !verdict.ok ? verdict.why : 'Your original file isn’t in this tab, so this can’t be done here.';
  const echoOff = echoBlocked(S);

  // the strength as a number: the slider's, for a selection that has one strength
  const dbNow = ustr !== null && vm === 1 ? (udb != null ? udb : STR_DB[ustr]) : null;
  const sliderMax = ukind === 'free' ? STR_DB[3] : STR_DB[4];

  const modes = [[0, 'Off', ''], [1, 'Noise removal', 'keeps your voice as recorded · takes out most other sounds, game sounds too'], [2, 'Voice rebuild', 'fixes echo, muffled or distorted voices · keeps only speech']];
  const nested = (
    <div className="vbody">
      <div className="sr">
        {ukind === null && <span className="srl"><Mixed show /></span>}
        <Seg
          items={[['free', 'Free', freeBad, freeBad ? freeWhy : tips['k.free']], ['paid', 'Standard', false, tips['k.paid']]]} cur={ukind} act="v-kind" label="Free or Standard" cls="kind"
        />
      </div>
      <div className="sr">
        <span className="srl">Strength<Mixed show={ustr === null} /></span>
        <Seg
          items={[[1, 'Gentle', false, tips['s.1']], [2, 'Balanced', false, tips['s.2']], [3, 'Strong', false, tips['s.3']], [4, 'Max', ukind !== 'paid', ukind !== 'paid' ? `${tips['s.4free']} ${tips['s.4']}` : tips['s.4']]]}
          cur={ustr !== null && udb == null ? ustr : null} act="v-str" label="Strength"
        />
        {dbNow != null && (
          <div className="slr" data-testid="strength-slide">
            <input
              type="range" min={0} max={posOf(sliderMax)} step="0.01" value={posOf(dbNow)} aria-label="Strength" aria-valuetext={strengthName(dbNow)} data-testid="strength-slider"
              onChange={(e) => h.onDb(dbAt(Number(e.target.value)))} onPointerDown={h.slideStart} onKeyDown={h.slideStart} onPointerUp={h.slideEnd} onBlur={h.slideEnd} onKeyUp={h.slideEnd}
            />
            <span className="slt" data-testid="strength-name"><b>{strengthName(dbNow)}</b> ({Math.round(dbNow)} dB)</span>
          </div>
        )}
      </div>
      <div className="sr">
        <span className="srl">Remover{fr ? null : <Mixed show={urem === null} />}</span>
        <Seg items={[0, 1, 2].map((r) => [r, REM[r], false, (fr ? `${tips['r.free']} ` : '') + tips[`r.${r}`]])} cur={fr ? null : urem} act="v-rem" label="Remover" dis={fr} />
        {fr && <span className="pnly">Standard only</span>}
      </div>
      {fr && (
        <div className="fnotes" data-testid="free-notes">
          {info?.channels > 2 && <p className="note mt0">This file has {info.channels} channels; they will be mixed down to mono.</p>}
          {info?.duration > LONG_SECONDS && (
            <p className="note mt0" data-testid="long-warning">
              This is a long recording ({human(info.duration)}). Cleaning it here could take about {human(eta)} and needs a lot of memory.
              Keep this tab open, or choose Denoising for long files.
            </p>
          )}
          {eta ? <p className="note mt0" data-testid="eta">Should take about {human(eta)} on this device{loadLeft ? ' (including getting the noise remover ready)' : ''}.</p> : null}
          {session.second && <p className="note mt0" data-testid="echo-browser-note">{tips['echo.free']}</p>}
          {info?.hasVideo && info?.viaLibav && <p className="note mt0" data-testid="audio-only-note">Your browser can clean the sound of this video, but can’t rebuild this video type, so you’ll get the cleaned sound as a WAV file. Denoising gives you the whole video back.</p>}
        </div>
      )}
    </div>
  );

  const q = uni(S, 'subs', 'q');
  const lg = uni(S, 'subs', 'lang');
  const trs = uniTr(S);
  const picked = trs || trUnion(S);
  const trDis = q === 0 || q === 1;
  const lv = uni(S, 'level', 'lvl');
  const lufs = uni(S, 'level', 'lufs');
  const echoV = uni(S, 'echo', 'on');
  const hasAccurate = overl(S, 'subs').some((x) => x.q === 3);
  const mixedBrowserSubs = overl(S, 'subs').some((x) => x.q === 1);
  const paidSoundSel = overl(S, 'voice').some((x) => x.mode === 2 || (x.mode === 1 && x.kind === 'paid')) || overl(S, 'echo').some((x) => x.on) || overl(S, 'level').some((x) => x.lvl);
  const freeSoundSel = overl(S, 'voice').some((x) => x.mode === 1 && x.kind === 'free');
  const paidSubsSel = overl(S, 'subs').some((x) => x.q >= 2);

  return (
    <>
      <div className="ih">
        <span className="grab" />
        <button type="button" className="ih-main" data-act="sheet" aria-expanded={sheet}>
          <b className="js-title">{selTitle(S)}</b>
          <span className="ih-sum">{summary(S)}</span>
          <span className="chv"><IcChevron /></span>
        </button>
        <button type="button" className="ibtn" data-act="info" aria-haspopup="dialog" aria-label="Compare all the options" data-testid="info-open" {...tipAttr(tips.info)}>i</button>
      </div>
      <div className="ib-body" aria-disabled={locked ? 'true' : undefined}>
        {locked && lockedWhy && <p className="note mt0" data-testid="locked-why">{lockedWhy}</p>}
        {sg && (
          <div className="ig" data-testid="sg-card">
            <IcSpark />
            <span><b>{sg.why}</b> · {sg.todo} suggested</span>
            <button type="button" className="btn btn-sm btn-primary" data-act="sg-apply" data-id={sg.id} data-testid="sg-apply">Apply</button>
            <button type="button" className="x" data-act="sg-dismiss" data-id={sg.id} aria-label="Dismiss" data-testid="sg-dismiss">&times;</button>
          </div>
        )}
        {one && (
          <section className="ctl rng" aria-label="The part" data-testid="range-box">
            {[['s', 'Start', one.s], ['e', 'End', one.e]].map(([w, label, v]) => (
              <div className="rr" key={w}>
                <span className="srl">{label}</span>
                <button type="button" className="btn btn-secondary btn-sm" data-act="nudge" data-w={w} data-d="-1" aria-label={`${label} one second earlier`} data-testid={`nudge-${w}-minus`}>−</button>
                <b className="rv" data-testid={`time-${w}`}>{clockMin(v)}</b>
                <button type="button" className="btn btn-secondary btn-sm" data-act="nudge" data-w={w} data-d="1" aria-label={`${label} one second later`} data-testid={`nudge-${w}-plus`}>+</button>
                <button type="button" className="btn btn-ghost btn-sm" data-act="useph" data-w={w} data-testid={`use-playhead-${w}`}>Use playhead</button>
              </div>
            ))}
            <div className="rr">
              <button type="button" className="btn btn-secondary btn-sm" data-act="playpart" data-testid="play-part">Play part</button>
            </div>
          </section>
        )}
        <section className="ctl l-noise voice">
          <div className="cl"><i className="mk" /><b>Voice</b>{tag('voice')}<Mixed show={vm === null} /></div>
          <div className="vrows" role="radiogroup" aria-label="Voice">
            {modes.map(([v, n, sub]) => {
              const on = vm !== null && vm === v;
              const paused = v === 2 && rebuildPaused;
              return (
                <div key={v} className={`vopt ${v === 1 ? 'l-noise' : v === 2 ? 'l-rebuild' : 'off'}${on ? ' on' : ''}`}>
                  <button
                    type="button" role="radio" className="vr" aria-checked={on} data-act="v-mode" data-v={v} data-testid={`voice-${['off', 'noise', 'rebuild'][v]}`}
                    aria-disabled={paused && !on ? 'true' : undefined} {...tipAttr(paused ? 'Paused right now. Try again later.' : tips[`v.${['off', 'noise', 'rebuild'][v]}`])}
                  >
                    <b>{n}</b>{sub ? <small>{paused ? 'paused right now' : sub}</small> : null}
                  </button>
                  {v === 1 && on ? nested : null}
                </div>
              );
            })}
          </div>
        </section>
        {session.second && (
          <section className="ctl l-echo inl">
            <div className="cl"><i className="mk" /><b {...tipAttr(tips.echo)}>Echo removal</b>{tag('echo')}<Mixed show={echoV === null} /></div>
            <button
              type="button" className="sw" role="switch" aria-checked={echoV === null ? 'mixed' : echoV === 1} aria-label="Echo removal" data-act="e-sw" data-testid="echo-switch"
              aria-disabled={echoOff ? 'true' : undefined} {...tipAttr(echoOff ? tips['echo.free'] : tips.echo)}
            />
          </section>
        )}
        <section className="ctl l-level">
          <div className="cl"><i className="mk" /><b {...tipAttr(tips.level)}>Leveling</b>{tag('level')}<Mixed show={lv === null} /></div>
          <Seg items={[[0, 'Off'], [1, 'Podcast'], [2, 'YouTube'], [3, 'Broadcast'], [4, 'Keep']].map(([k, n]) => [k, n, false, tips[`l.${k}`]])} cur={lv} act="l-lvl" label="Leveling" cls="lv" />
          {lv === 9 && <p className="note mt0">{lufs} LUFS (as before)</p>}
        </section>
        <section className="ctl l-subs">
          <div className="cl"><i className="mk" /><b {...tipAttr(tips.subs)}>Subtitles</b>{tag('subs')}<Mixed show={q === null} /></div>
          <Seg
            items={[0, 1, 2, 3].map((k) => [k, SUBQ[k], k === 1 && (!sVerdict.ok || noFile), k === 1 && (!sVerdict.ok || noFile) ? (!sVerdict.ok ? sVerdict.why : 'Your original file isn’t in this tab, so this can’t be done here.') : tips[`q.${k}`]])}
            cur={q} act="s-q" label="Subtitles"
          />
          <div className="sr">
            <span className="srl" {...tipAttr(q === 0 ? undefined : tips.lang)}>Language<Mixed show={lg === null} /></span>
            <select className="select" data-act="s-lang" aria-label="Language" disabled={q === 0} value={lg === null ? '' : lg} onChange={() => {}} {...(q === 0 ? {} : tipAttr(tips.lang))}>
              {lg === null && <option value="" disabled>Mixed</option>}
              {LANG_OPTIONS.map(([k, n]) => <option key={k} value={k}>{n}</option>)}
            </select>
          </div>
          <div className="sr">
            <span className="srl" {...tipAttr(trDis ? tips['tr.no'] : tips.tr)}>Translate to{trs === null ? <Mixed show /> : null}</span>
            <div className="trs">
              {picked.map((l) => (
                <span className="trc" key={l} data-testid="translation-chip">{LANG[l]}<button type="button" className="x" data-act="tr-del" data-v={l} aria-label={`Remove ${LANG[l]}`}>&times;</button></span>
              ))}
              {picked.length < MAX_TRANSLATIONS && (
                <select className="select" data-act="tr-add" aria-label="Add a translation language" disabled={trDis} value="" onChange={() => {}} data-testid="translate-add" {...(trDis ? {} : tipAttr(tips.tr))}>
                  <option value="">{picked.length ? 'Add another language' : 'No translation'}</option>
                  {LANG_OPTIONS.filter(([k]) => k !== 'auto' && k !== lg && !picked.includes(k)).map(([k, n]) => <option key={k} value={k}>{n}</option>)}
                </select>
              )}
            </div>
          </div>
          {hasAccurate && <span className="hint" data-testid="accurate-note">Extra accurate sends your audio to a speech partner we work with, only for this option. <Link to="/privacy">Privacy details</Link></span>}
          {freeSoundSel && paidSubsSel && <span className="hint" data-testid="browser-subs-note">The noise removal happens on your device. For the subtitles, the original sound is uploaded.</span>}
          {paidSoundSel && mixedBrowserSubs && <span className="hint" data-testid="denoise-local-subs-note">The cleanup is done on our side; the subtitles are made here on your device, from your original recording.</span>}
          {paidSoundSel && paidSubsSel && <span className="hint" data-testid="both-note">We write the transcript from your original recording (it comes out more accurate), and clean the sound separately.</span>}
        </section>
      </div>
    </>
  );
}

export { lenTxt };
