import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { fmt, ranges, STR, SUBQ, TARGETS, TRACKS, trOf } from '../../editor/tracks.js';
import { LANGUAGES } from '../../jobs.js';
import { IcCheck, IcSpark, IcSpeaker, IcSpeakerOff } from './icons.jsx';
import { tipAttr } from './ui.jsx';

const LANG = Object.fromEntries(LANGUAGES.map(([c, n]) => [c || 'auto', n]));
const NICE = [1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600, 7200];
const nice = (x) => NICE.find((n) => n >= x) || 7200;

/** How a section reads in its track. */
function secView(tr, x) {
  if (tr === 'voice') {
    if (x.mode === 0) return { l: 'Off', c: 'off' };
    if (x.mode === 2) return { l: 'Rebuild', c: 'sv-rebuild i3' };
    return { l: (x.kind === 'free' ? 'Free · ' : '') + (x.db != null ? `${Math.round(x.db)} dB` : STR[x.str]), c: `sv-noise i${x.str}${x.kind === 'free' ? ' free' : ''}` };
  }
  if (tr === 'echo') return x.on ? { l: 'Echo', c: 'i3' } : { l: 'Off', c: 'off' };
  if (tr === 'level') return x.lvl ? { l: x.lvl === 9 ? `${x.lufs} LUFS` : TARGETS[x.lvl - 1], sh: x.lvl === 4 ? 'Keep' : '', c: 'i3' } : { l: 'Off', c: 'off' };
  if (!x.q) return { l: 'Off', c: 'off' };
  const t = trOf(x);
  return { l: SUBQ[x.q] + (x.lang !== 'auto' ? ` · ${LANG[x.lang]}` : '') + (t.length ? ` → ${t.map((l) => LANG[l]).join(', ')}` : ''), sh: x.q === 3 ? 'Extra' : x.q === 2 ? 'Std' : '', c: x.q === 1 ? 'free i1' : x.q === 2 ? 'i2' : 'i4' };
}

/** The waveform as SVG paths: the recording's outline (peaks), the distorted stretches, and what is heard (the average level). */
function wavePaths({ data, heard, w, H }) {
  const n = Math.max(40, Math.floor(w / 3.2));
  const step = w / n;
  const mid = H / 2;
  const mx = mid - 3;
  const N = data.N;
  const rngI = (i) => [Math.floor((i * N) / n), Math.max(Math.floor((i * N) / n) + 1, Math.floor(((i + 1) * N) / n))];
  let po = '';
  let pd = '';
  let pc = '';
  for (let i = 0; i < n; i++) {
    const [a, b] = rngI(i);
    let ho = 0;
    let hc = 0;
    let flag = false;
    for (let k = a; k < b && k < N; k++) {
      ho = Math.max(ho, data.o[k]);
      if (heard) hc += heard[k];
      if (data.d[k]) flag = true;
    }
    hc = heard ? hc / Math.max(1, Math.min(b, N) - a) : 0;
    const x = ((i + 0.5) * step).toFixed(1);
    const yo = Math.max(1.5, ho * mx);
    const p = `M${x} ${(mid - yo).toFixed(1)}V${(mid + yo).toFixed(1)}`;
    if (flag) pd += p; else po += p;
    if (heard) { const yc = Math.max(1.5, hc * mx); pc += `M${x} ${(mid - yc).toFixed(1)}V${(mid + yc).toFixed(1)}`; }
  }
  return { po, pd, pc };
}

const Wave = memo(function Wave({ data, heard, w, H, tag, reading }) {
  const paths = useMemo(() => (w && data ? wavePaths({ data, heard, w, H }) : null), [data, heard, w, H]);
  if (reading) return <span className="wtag">Reading the sound of your file…</span>;
  return (
    <>
      {tag ? <span className="wtag" data-testid="hearing">{tag}</span> : null}
      {paths && (
        <>
          <svg className="w base" width={w} height={H} viewBox={`0 0 ${w} ${H}`} aria-hidden="true"><path className="n" d={paths.po} /><path className="d" d={paths.pd} /></svg>
          <svg className="w top" width={w} height={H} viewBox={`0 0 ${w} ${H}`} aria-hidden="true" data-testid="heard-line"><path d={paths.pc} /></svg>
        </>
      )}
      <div className="pm" />
    </>
  );
});

/**
 * The timeline: the ruler with the suggestions' pins, the waveform with the selection's handles, and a row for each track. Positions are
 * percentages of the length; `zoom` makes the rows wider than the screen (they scroll together). The selection is drawn on every row.
 */
export default function Timeline({ ctx }) {
  const { S, session, stage, ph, zoom, tips, pins, hear, hearText, heard, data, mobile, scrollRef, waveRef, reading, prog } = ctx;
  const T = S.T;
  const [size, setSize] = useState({ w: 0, H: 92 });
  const wf = useRef(null);
  useEffect(() => {
    const el = wf.current;
    if (!el) return undefined;
    const measure = () => setSize({ w: Math.floor(el.clientWidth), H: el.clientHeight || (mobile ? 64 : 92) });
    measure();
    const ro = window.ResizeObserver ? new ResizeObserver(measure) : null;
    ro?.observe(el);
    return () => ro?.disconnect();
  }, [mobile, zoom]);
  const { w, H } = size;

  const ticks = useMemo(() => {
    if (!w) return { tk: [], lab: [] };
    const step = nice(T / (w >= 560 ? Math.max(6, Math.round(w / 110)) : 3));
    const tick = nice(step / (w >= 560 ? 3 : 6));
    const tk = [];
    const lab = [];
    for (let t = 0; t <= T + 1e-6; t += tick) tk.push(t);
    for (let t = 0; t <= T + 1e-6; t += step) { if (t !== T && T - t < step * 0.4) continue; lab.push(t); }
    return { tk, lab };
  }, [w, T, zoom]);

  const R = ranges(S);
  const whole = !S.sel;
  const one = !S.sel ? { s: 0, e: T } : S.sel.length === 1 ? S.sel[0] : null;
  const pct = (t) => `${(t / T) * 100}%`;
  const processing = stage === 'processing';
  const trackW = w || 600;

  return (
    <div
      className={`tlw lk${zoom > 1 ? ' zoomed' : ''}`} id="tlw" data-ph="1" data-stage={stage} data-sel={one ? '1' : 'n'} data-ghost={heard ? '1' : undefined}
      style={{
        '--ph': pct(ph), '--zoom': zoom, '--sl': one ? pct(one.s) : '0%', '--sw': one ? `${((one.e - one.s) / T) * 100}%` : '0%',
        '--prog': `${processing ? prog : heard ? 100 : 0}%`,
      }}
    >
      <div className="tsc" ref={scrollRef}>
        <div className="row wr">
          <div className="lab" />
          <div className="trk pl" id="wave" ref={waveRef} data-testid="wave" data-seconds={T}>
            <i className="bands">{R.map((r) => <i key={`${r.s}-${r.e}`} className={`band${r.s <= 0 && r.e >= T ? ' whole' : ''}`} style={{ left: pct(r.s), width: pct(r.e - r.s) }} />)}</i>
            <div className="ruler">
              <div id="ticks">
                {ticks.tk.map((t) => <span key={`k${t}`} className="tk" style={{ left: pct(t) }} />)}
                {ticks.lab.map((t) => <span key={`l${t}`} className={`tkl ${t === 0 ? 'first' : t >= T - 1e-6 ? 'last' : ''}`} style={{ left: pct(t) }}>{fmt(t)}</span>)}
              </div>
              <div id="pins">
                {pins.map((g) => (
                  <button
                    key={g.id} type="button" className={`pin ${g.st === 'applied' ? 'applied' : ''}`} data-act="pin" data-id={g.id} data-testid="pin" style={{ left: pct((g.s + g.e) / 2) }}
                    aria-label={`${g.why}, ${fmt(g.s)} to ${fmt(g.e)}`} title={`${g.why}, ${fmt(g.s)}–${fmt(g.e)}`}
                  >
                    {g.st === 'applied' ? <IcCheck /> : <IcSpark />}
                  </button>
                ))}
              </div>
            </div>
            <div className="wf" id="wf" ref={wf} aria-label="Waveform" data-testid="waveform" data-reading={reading ? 'yes' : 'no'} data-known={data ? 'yes' : 'no'}>
              <Wave data={data} heard={heard} w={w} H={H} tag={stage === 'done' && !reading ? hearText : ''} reading={reading} />
              <div id="sgb">
                {!reading && pins.filter((g) => g.st === 'new').map((g) => <div key={g.id} className="sgb" style={{ left: pct(g.s), width: pct(g.e - g.s) }} />)}
              </div>
            </div>
            <div className="hnd s" id="hs" role="slider" tabIndex={0} aria-label="Start of the selection" aria-valuenow={one ? one.s : undefined} data-h="s" data-testid="handle-start"><i /></div>
            <div className="hnd e" id="he" role="slider" tabIndex={0} aria-label="End of the selection" aria-valuenow={one ? one.e : undefined} data-h="e" data-testid="handle-end"><i /></div>
          </div>
        </div>
        <div id="trows">
          {TRACKS.filter(([tr]) => tr !== 'echo' || S.second).map(([tr, name]) => (
            <div className="row" key={tr} data-testid={`track-${tr}`}>
              <div className="lab">
                {stage === 'done' && tr !== 'subs' && (
                  <button
                    type="button" className={`hear l-${tr}`} data-act="hear" data-tr={tr} aria-pressed={!!hear[tr]} aria-label={`Hear ${name}`} data-testid={`hear-${tr}`}
                    {...tipAttr(tips.hear(name))}
                  >
                    {hear[tr] ? <IcSpeaker /> : <IcSpeakerOff />}
                  </button>
                )}
                {name}
              </div>
              <div className="trk pl tr">
                <i className="bands">{R.map((r) => <i key={`${r.s}-${r.e}`} className={`band${r.s <= 0 && r.e >= T ? ' whole' : ''}`} style={{ left: pct(r.s), width: pct(r.e - r.s) }} />)}</i>
                {S.tracks[tr].map((x) => {
                  const v = secView(tr, x);
                  const wd = (x.e - x.s) / T;
                  const fits = (t) => wd * trackW >= t.length * 6.8 + 18;
                  let lab = v.l;
                  if (!fits(lab) && v.sh && fits(v.sh)) lab = v.sh;
                  return (
                    <button
                      key={`${x.s}-${x.e}`} type="button" data-testid="section" data-tr={tr} data-act="sec" data-s={x.s} data-e={x.e}
                      className={`sec l-${tr} ${v.c}${fits(lab) ? '' : ' tiny'}${x.hard ? ' hard' : ''}`}
                      style={{ left: pct(x.s), width: `${wd * 100}%` }} title={`${name}: ${v.l}, ${fmt(x.s)}–${fmt(x.e)}`}
                    >
                      <b>{lab}</b>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
          <div className="row na"><div className="lab">Music</div><div className="trk pl tr na"><i className="bands" /><span><i>Coming</i></span></div></div>
        </div>
      </div>
    </div>
  );
}

