import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { apiOffline } from '../api.js';
import { useAuth } from '../auth.jsx';
import JobRow from '../components/JobRow.jsx';
import { PaidOffline } from '../components/Maintenance.jsx';
import { SignInLink } from '../components/SignIn.jsx';
import EmptyStudio from '../components/studio/Empty.jsx';
import Studio from '../components/studio/Studio.jsx';
import { pruneWants, useLocalCards } from '../finish.js';
import { usePageTitle } from '../hooks.js';
import { activeCount } from '../jobs.js';
import { takeDraft } from '../studio/draft.js';
import { Session } from '../studio/session.js';
import { useJobs } from '../useJobs.js';

export default function Cleanup() {
  usePageTitle('Cleanup');
  const { refresh, ready, known, signedIn } = useAuth();
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
      const s = await Session.open(f, { signedIn: known ? signedIn : true }); // (before the account check answers: as signed in)
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

  // the free tools' note for visitors without an account (the editor has its own note for when our side is away)
  const freeNote = !offline && signedOut ? (
    <div className="banner banner-info row-between" data-testid="free-no-account">
      <span>The free tools work without an account. Noise removal and subtitles run in your browser, so your file stays on your device. Sign in for the paid options.</span>
      <SignInLink to="/app" className="btn btn-secondary btn-sm">Sign in</SignInLink>
    </div>
  ) : null;

  // the jobs, beside the editor
  const aside = signedIn ? (
    <aside className="jobs-col" aria-labelledby="jobs-h">
      <div className="jh">
        <h2 id="jobs-h">Your jobs</h2>
        <button type="button" className="btn btn-ghost btn-sm" onClick={reload}>Refresh</button>
      </div>
      {error && <div className="banner banner-error" role="alert">{error}</div>}
      {all === null && <p role="status" className="muted"><span className="spinner" aria-hidden="true" /> Loading your jobs…</p>}
      <p className="muted small mb0" data-testid="jobs-note">Every file stays here, with its results, for 2 hours after your last change to it, and at most 6 hours after the upload. Open one to listen, download it again, change it or delete it.</p>
      {all?.length === 0 && !error && <p className="card muted mb0">Nothing here yet. Your files show up here once they run.</p>}
      {all?.map((j) => <JobRow key={j.id} job={j} />)}
    </aside>
  ) : localCards.length > 0 ? (
    <aside className="jobs-col" aria-labelledby="jobs-h">
      <h2 id="jobs-h">Made in this tab</h2>
      <p className="muted small mb0" data-testid="jobs-note">These stay in this tab only, until you leave or reload the page. Open one to listen, download it again or change it.</p>
      {localCards.map((j) => <JobRow key={j.id} job={j} />)}
    </aside>
  ) : null;

  if (draft) {
    return <Studio session={draft} active={activeCount(jobs)} maxActive={limits?.maxActiveJobs || null} onNewFile={another} banner={freeNote} aside={aside} />;
  }
  return <EmptyStudio onFile={choose} reading={reading} problem={problem} banner={offline ? <PaidOffline /> : freeNote} aside={aside} />;
}
