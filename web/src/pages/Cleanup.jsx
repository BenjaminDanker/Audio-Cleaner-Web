import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { Link } from 'react-router';
import { apiOffline } from '../api.js';
import { useAuth } from '../auth.jsx';
import FileDrop, { SERVER_FORMATS } from '../components/FileDrop.jsx';
import JobRow from '../components/JobRow.jsx';
import { PaidOffline } from '../components/Maintenance.jsx';
import { SignInLink } from '../components/SignIn.jsx';
import Studio from '../components/studio/Studio.jsx';
import { pruneWants, useLocalCards } from '../finish.js';
import { balanceMoney, shownName } from '../format.js';
import { usePageTitle } from '../hooks.js';
import { activeCount } from '../jobs.js';
import { takeDraft } from '../studio/draft.js';
import { Session } from '../studio/session.js';
import { useJobs } from '../useJobs.js';

export default function Cleanup() {
  usePageTitle('Cleanup');
  const { balance, refresh, ready, known, signedIn } = useAuth();
  const offline = useSyncExternalStore(apiOffline.subscribe, apiOffline.get);
  const { jobs, limits, error, reload } = useJobs({ onFinished: refresh }); // (signed out there is no list to ask for)
  const signedOut = ready && known && !signedIn; // (not while the account check has not been answered: that may only be a busy moment)
  const localCards = useLocalCards(); // free runs made in this tab: no job on our side, listed like the others
  const all = useMemo(() => (jobs ? [...localCards, ...jobs] : jobs), [localCards, jobs]);
  const [draft, setDraft] = useState(null); // the file that was chosen: the editor is its page until it runs
  const [reading, setReading] = useState('');
  const [problem, setProblem] = useState('');
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; draft?.dispose(); }; }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Work kept across a sign-in (studio/draft.js): the file and the editor's state come back as they were left, and the copy is deleted.
  useEffect(() => {
    let on = true;
    takeDraft().then(async (rec) => {
      if (!rec || !on) return;
      setReading(rec.name);
      try {
        const s = await Session.open(new File([rec.blob], rec.name, { type: rec.type, lastModified: rec.lastModified }), { draft: rec });
        if (alive.current && on) setDraft(s); else s.dispose();
      } catch (e) {
        setProblem('We couldn’t open your file again. Please choose it once more.');
      } finally {
        setReading('');
      }
    });
    return () => { on = false; };
  }, []);

  // the choices kept for jobs the server no longer lists (their files are cleared away 2 hours after the last change to them) go too
  useEffect(() => {
    if (signedIn && jobs && !error) pruneWants(jobs.map((j) => j.id));
  }, [jobs, error, signedIn]);

  async function choose(f) {
    setProblem('');
    setReading(f.name);
    try {
      const s = await Session.open(f);
      if (alive.current) setDraft(s);
    } catch (e) {
      setProblem(e.message || 'Could not read that file. Please pick another one.');
    } finally {
      setReading('');
    }
  }
  function another() {
    draft?.dispose();
    setDraft(null);
    setProblem('');
  }

  if (draft) {
    return <Studio session={draft} active={activeCount(jobs)} maxActive={limits?.maxActiveJobs || null} onNewFile={another} />;
  }

  return (
    <div className="container page stack-lg">
      <header className="row-between">
        <div>
          <h1 className="mb0">Cleanup</h1>
          <p className="muted mb0">Take the noise out of a recording, and get subtitles and a transcript. Drop a file and we&rsquo;ll show what you can do with it.</p>
        </div>
      </header>

      {offline && <PaidOffline />}
      {signedOut && !offline && (
        <div className="banner banner-info row-between" data-testid="free-no-account">
          <span>The free tools work without an account. Noise removal and subtitles run in your browser, so your file stays on your device. Sign in for the paid options.</span>
          <SignInLink to="/app" className="btn btn-secondary btn-sm">Sign in</SignInLink>
        </div>
      )}

      <section className="card stack" aria-labelledby="newjob-h">
        <h2 id="newjob-h" className="mb0">Add a file</h2>
        {!reading && <FileDrop onFile={choose} accept={SERVER_FORMATS} hint="or click to pick one. Almost any audio or video file: MP3, WAV, M4A, FLAC, WMA, AMR, MP4, MOV, MKV, AVI, WMV, MPEG and more." />}
        {reading && <p role="status" className="muted"><span className="spinner" aria-hidden="true" /> Reading {shownName(reading)}…</p>}
        {problem && <div className="banner banner-error" role="alert" data-testid="job-error">{problem}</div>}
      </section>

      {signedIn ? (
        <section aria-labelledby="jobs-h" className="stack">
          <div className="row-between">
            <h2 id="jobs-h" className="mb0">Your jobs</h2>
            <button type="button" className="btn btn-ghost btn-sm" onClick={reload}>Refresh</button>
          </div>
          {error && <div className="banner banner-error" role="alert">{error}</div>}
          {all === null && <p role="status" className="muted"><span className="spinner" aria-hidden="true" /> Loading your jobs…</p>}
          <p className="muted small mb0" data-testid="jobs-note">Every file stays here, with its results, for 2 hours after your last change to it, and at most 6 hours after the upload. Open one to listen, download it again, change it or delete it.</p>
          {all?.length === 0 && !error && <p className="card muted mb0">Nothing here yet. Drop a file above and it’ll show up.</p>}
          {all?.map((j) => <JobRow key={j.id} job={j} />)}
        </section>
      ) : localCards.length > 0 && (
        <section aria-labelledby="jobs-h" className="stack">
          <h2 id="jobs-h" className="mb0">Made in this tab</h2>
          <p className="muted small mb0" data-testid="jobs-note">These stay in this tab only, until you leave or reload the page. Open one to listen, download it again or change it.</p>
          {localCards.map((j) => <JobRow key={j.id} job={j} />)}
        </section>
      )}

      {signedIn ? (
        <div className="grid grid-2">
          <section className="card stack" aria-labelledby="bal-h">
            <h2 id="bal-h" className="mb0">Balance</h2>
            <div>
              <div className="big-number" data-testid="available">{balanceMoney(balance?.available_micros ?? 0)}</div>
              <div className="muted small">ready to spend</div>
            </div>
            <dl className="kv mb0">
              <dt>Set aside for running jobs</dt><dd data-testid="held">{balanceMoney(balance?.held_micros ?? 0)}</dd>
              <dt>Total balance</dt><dd>{balanceMoney(balance?.balance_micros ?? 0)}</dd>
            </dl>
            <div className="row"><Link to="/app/billing" className="btn btn-primary">Add credit</Link></div>
          </section>
          <GoodToKnow />
        </div>
      ) : <GoodToKnow />}
    </div>
  );
}

function GoodToKnow() {
  return (
    <section className="card stack" aria-labelledby="tips-h">
      <h2 id="tips-h" className="mb0">Good to know</h2>
      <ul className="mb0">
        <li>Free noise removal happens <b>in your browser</b>, so that file never leaves your device.</li>
        <li>Uploading a video for us to work on? Only its <b>sound</b> is uploaded. Your browser puts the clean sound back into the video afterwards.</li>
        <li>Your upload and your results are kept together for <b>2 hours after your last change</b> to a file (at most 6 hours after the upload), so you can listen, download again or change it, and then they are deleted together. You can delete them sooner.</li>
        <li>Choose a part of your file on the timeline to change just that part: a few seconds with an echo, a louder stretch, a muffled answer. You see the price before you save, and a new strength is often free.</li>
        <li>Changing a finished file only redoes the steps that changed, so a new strength is often free. The price is shown before you confirm.</li>
        <li>You pay for what gets processed, by the second. A job that fails or is canceled costs nothing.</li>
      </ul>
    </section>
  );
}
