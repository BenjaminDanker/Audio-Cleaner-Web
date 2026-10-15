import { useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../auth.jsx';
import JobCard from '../components/JobCard.jsx';
import JobForm from '../components/JobForm.jsx';
import { balanceMoney } from '../format.js';
import { usePageTitle } from '../hooks.js';
import { useJobs } from '../useJobs.js';

export default function Dashboard() {
  usePageTitle('Studio');
  const { user, balance, refresh, ensureTos } = useAuth();
  const localJobs = useRef(new Map()); // jobId -> {file, info, extracted} (this tab only)
  const { jobs, error, reload } = useJobs({ onFinished: refresh });

  const submitted = useCallback((job, local) => {
    localJobs.current.set(job.id, local);
    reload();
  }, [reload]);

  return (
    <div className="container page stack-lg">
      <header className="row-between">
        <div>
          <h1 className="mb0">Studio</h1>
          <p className="muted mb0">Clean up and transcribe your longer recordings.</p>
        </div>
      </header>

      {user && !user.tosAccepted && (
        <div className="banner banner-warn row-between" role="status">
          <span>Before your first job, please have a look at the Terms of Service.</span>
          <button type="button" className="btn btn-primary btn-sm" onClick={ensureTos}>Take a look</button>
        </div>
      )}

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
          <div className="row"><Link to="/app/billing" className="btn btn-primary">Add credit</Link><Link to="/transcribe" className="btn btn-ghost">Free transcription</Link></div>
        </section>
        <section className="card stack" aria-labelledby="tips-h">
          <h2 id="tips-h" className="mb0">Good to know</h2>
          <ul className="mb0">
            <li>Uploading a video? Only its <b>sound</b> is uploaded. Your browser puts the clean sound back into the video afterwards.</li>
            <li>Your upload is deleted as soon as it’s processed. Results stay for <b>24 hours</b>.</li>
            <li>You pay for what gets processed, by the second. A job that fails or is canceled costs nothing.</li>
          </ul>
        </section>
      </div>

      <JobForm variant="studio" onSubmitted={submitted} />

      <section aria-labelledby="jobs-h" className="stack">
        <div className="row-between">
          <h2 id="jobs-h" className="mb0">Your jobs</h2>
          <button type="button" className="btn btn-ghost btn-sm" onClick={reload}>Refresh</button>
        </div>
        {error && <div className="banner banner-error" role="alert">{error}</div>}
        {jobs === null && <p role="status" className="muted"><span className="spinner" aria-hidden="true" /> Loading your jobs…</p>}
        {jobs?.length === 0 && !error && <p className="card muted mb0">Nothing here yet. Drop a file above and it’ll show up.</p>}
        {jobs?.map((j) => <JobCard key={j.id} job={j} local={localJobs.current.get(j.id)} onChanged={() => { reload(); refresh(); }} />)}
      </section>
    </div>
  );
}
