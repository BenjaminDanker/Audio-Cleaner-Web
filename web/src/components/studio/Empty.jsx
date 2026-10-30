import '../../styles/studio.css';
import { useState } from 'react';
import { Link } from 'react-router';
import { useAuth } from '../../auth.jsx';
import { balanceMoney, shownName } from '../../format.js';
import { usePricing } from '../../hooks.js';
import FileDrop, { SERVER_FORMATS } from '../FileDrop.jsx';
import InfoModal from './InfoModal.jsx';

const LANES = ['Voice', 'Leveling', 'Subtitles', 'Music'];
const CARDS = [
  ['l-noise', 'Voice', 'Noise removal or Voice rebuild'],
  ['l-level', 'Leveling', 'Podcast, YouTube or Broadcast loudness'],
  ['l-subs', 'Subtitles', 'Subtitles and a transcript, with translations'],
];

/** The editor before a file is chosen (Cleanup page): the same layout as with a file, the place to drop one where the picture and the
 * timeline go, so choosing a file only fills it in. `banner` goes above the editor, `aside` beside it (the jobs). */
export default function EmptyStudio({ onFile, reading, problem, banner, aside }) {
  const { balance, signedIn } = useAuth();
  const { data: pricing } = usePricing();
  const [modal, setModal] = useState(false);
  const avail = balance?.available_micros;
  return (
    <div className="studio" data-testid="studio-empty">
      <div className="container ws wide">
        <div className={`lay${aside ? ' with-aside' : ''}`}>
          <div className="work">
            {banner}
            <div className="ed">
              <div className="edl">
                <div className="pane main">
                  <h1 className="fname">Choose a file</h1>
                  <p className="fmeta">Take the noise out of a recording, and get subtitles and a transcript.</p>
                  {!reading && <FileDrop onFile={onFile} accept={SERVER_FORMATS} hint="or click to pick one. Almost any audio or video file: MP3, WAV, M4A, FLAC, WMA, AMR, MP4, MOV, MKV, AVI, WMV, MPEG and more." />}
                  {reading && <p role="status" className="muted"><span className="spinner" aria-hidden="true" /> Reading {shownName(reading)}…</p>}
                  {problem && <div className="banner banner-error" role="alert" data-testid="job-error">{problem}</div>}
                  <div className="ghost-rows ghost" aria-hidden="true">
                    {LANES.map((l) => <div className="ghost-row" key={l}><span>{l}</span><i /></div>)}
                  </div>
                </div>
                <section className="pane outp ghost" aria-label="Your file"><p className="muted mb0">The file type, name and video size go here.</p></section>
              </div>
              <aside className="pane insp still" aria-label="Settings">
                <div className="ih">
                  <b className="ih-main">Settings</b>
                  <button type="button" className="ibtn" aria-haspopup="dialog" aria-label="Compare all the options" data-testid="info-open" onClick={() => setModal(true)}>i</button>
                </div>
                <div className="ib-body ghost">
                  {CARDS.map(([cls, title, line]) => (
                    <section className={`ctl ${cls}`} key={title}>
                      <div className="cl"><i className="mk" /><b>{title}</b></div>
                      <p className="muted small mb0">{line}</p>
                    </section>
                  ))}
                </div>
              </aside>
            </div>
            <div className="pane bb">
              <div className="bb-in">
                <span className="bb-sp" />
                {signedIn && avail != null && (
                  <Link to="/app/billing" className="bal" data-testid="bar-balance"><span className="muted">Balance</span> <b>{balanceMoney(avail)}</b></Link>
                )}
                <button type="button" className="btn btn-primary btn-lg" disabled data-testid="start-empty">Start</button>
              </div>
            </div>
          </div>
          {aside}
        </div>
      </div>
      <InfoModal open={modal} onClose={() => setModal(false)} pricing={pricing} />
    </div>
  );
}
