import { useEffect, useRef, useState } from 'react';
import { Link, useLocation, useNavigate, useParams } from 'react-router';
import { api } from '../api.js';
import { useAuth } from '../auth.jsx';
import JobCard from '../components/JobCard.jsx';
import Studio from '../components/studio/Studio.jsx';
import { finishOf, useLocalCards } from '../finish.js';
import { usePageTitle } from '../hooks.js';
import { isLocalJobId } from '../localid.js';
import { activeCount, jobTitle } from '../jobs.js';
import { Session, sessionFor } from '../studio/session.js';
import { useJobs } from '../useJobs.js';

/**
 * One job on a page of its own (/app/job/<id>): the editor, with its progress while it runs and its result when it is done (Your
 * files, Download again, Save changes, Delete now). A job the editor can't open (one made through the API with no audio steps, a job
 * that failed or was canceled) shows as a card. The address works after a reload and with back and forward; it stays good until the
 * job's files are cleared away. "Different file" goes back to the Cleanup page.
 */
export default function Job() {
  const { id } = useParams();
  const { refresh } = useAuth();
  const nav = useNavigate();
  const loc = useLocation();
  const { jobs, limits, error, reload } = useJobs({ onFinished: refresh });
  const [solo, setSolo] = useState(null); // the job asked for by itself, when the list doesn't have it: {job} or {gone: true}
  const [welcome, setWelcome] = useState(() => loc.state?.submitted || '');
  const localCards = useLocalCards(); // a free run made in this tab has no job on our side: it lives in this tab only
  const isLocal = isLocalJobId(id);
  const listed = isLocal ? localCards.find((j) => j.id === id) || null : jobs?.find((j) => j.id === id) || null;
  const job = listed || solo?.job || null;
  usePageTitle(job ? jobTitle(job) : 'Job');

  // the note about the job just started is shown once: a reload or a visit later doesn't repeat it
  // (changing a job again opens another job on this same page: its note comes with the navigation)
  useEffect(() => {
    if (loc.state?.submitted) {
      setWelcome(loc.state.submitted);
      nav(loc.pathname, { replace: true, state: null });
    }
  }, [loc.key]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!loc.state?.submitted) setWelcome(''); // another job: the note of the one before is not for this one
  }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  // another job on the same page (changing a job again opens the new one): ask for the list again, it holds the new job
  const first = useRef(true);
  useEffect(() => {
    setSolo(null);
    if (first.current) first.current = false;
    else reload();
  }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  // the list is only the latest jobs: ask for this one by itself if it isn't in it
  useEffect(() => {
    if (isLocal || jobs === null || listed || solo) return undefined;
    let alive = true;
    api(`jobs/${id}`).then(
      (j) => alive && setSolo(j.status === 'expired' ? { gone: true } : { job: j }),
      () => alive && setSolo({ gone: true }),
    );
    return () => { alive = false; };
  }, [jobs, listed, solo, id, isLocal]);

  const gone = !job && (isLocal ? true : solo?.gone || (error && jobs?.length === 0 && !solo));
  // the editor opens jobs that have audio steps (and any card or job made in this tab); a failed or canceled one shows what happened
  const editable = !!job && !['failed', 'canceled', 'expired'].includes(job.status) && (isLocal || !!sessionFor(job.id) || (!!job.edit && (job.edit.available || job.status !== 'done')));
  const session = editable ? (sessionFor(job.id) || Session.forJob(job, finishOf(job))) : null;
  if (editable && session) {
    return (
      <div data-testid="job-page">
        <Studio
          key={job.id} session={session} job={job} fin={finishOf(job)} welcome={welcome} active={activeCount(jobs)} maxActive={limits?.maxActiveJobs || null}
          onNewFile={() => nav('/app')} onChanged={() => { reload(); refresh(); }} onDeleted={() => nav('/app', { replace: true })}
        />
      </div>
    );
  }
  return (
    <div className="container page stack-lg" data-testid="job-page">
      <header className="row-between">
        <div>
          <Link to="/app" data-testid="all-jobs">&larr; Your jobs</Link>
          <h1 className="mb0">{job ? jobTitle(job) : 'Job'}</h1>
        </div>
        <Link to="/app" className="btn btn-primary" data-testid="start-another">Start another file</Link>
      </header>
      {welcome && <div className="banner banner-success" role="status" data-testid="submitted">{welcome}</div>}
      {!job && !gone && <p role="status" className="muted"><span className="spinner" aria-hidden="true" /> Loading this job…</p>}
      {gone && (
        <div className="card stack" data-testid="job-gone">
          <p className="mb0"><b>This job isn’t here any more.</b></p>
          <p className="muted mb0">Your files are cleared away 2 hours after your last change to a file (at most 6 hours after the upload), or when you delete them, and a free run made in your browser is gone when you leave or reload the page. What you paid is in your <Link to="/app/billing">billing history</Link>.</p>
          <div><Link to="/app" className="btn btn-primary">Start another file</Link></div>
        </div>
      )}
      {job && <JobCard job={job} onChanged={() => { reload(); refresh(); }} onDeleted={() => nav('/app', { replace: true })} />}
    </div>
  );
}
