import { useEffect, useMemo } from 'react';
import { Link } from 'react-router';
import { useAuth } from '../auth.jsx';
import JobForm from '../components/JobForm.jsx';
import JobRow from '../components/JobRow.jsx';
import { pruneWants, useLocalCards } from '../finish.js';
import { balanceMoney } from '../format.js';
import { usePageTitle } from '../hooks.js';
import { activeCount } from '../jobs.js';
import { useJobs } from '../useJobs.js';

export default function Cleanup() {
  usePageTitle('Cleanup');
  const { balance, refresh } = useAuth();
  const { jobs, limits, error, reload } = useJobs({ onFinished: refresh });
  const localCards = useLocalCards(); // free runs made in this tab: no job on our side, listed like the others
  const all = useMemo(() => (jobs ? [...localCards, ...jobs] : jobs), [localCards, jobs]);

  // the choices kept for jobs the server no longer lists (their files are cleared away 2 hours after they finish) go too
  useEffect(() => {
    if (jobs && !error) pruneWants(jobs.map((j) => j.id));
  }, [jobs, error]);

  return (
    <div className="container page stack-lg">
      <header className="row-between">
        <div>
          <h1 className="mb0">Cleanup</h1>
          <p className="muted mb0">Take the noise out of a recording, and get subtitles and a transcript. Drop a file and we&rsquo;ll show what you can do with it.</p>
        </div>
      </header>

      <JobForm onSubmitted={reload} active={activeCount(jobs)} maxActive={limits?.maxActiveJobs || null} />

      <section aria-labelledby="jobs-h" className="stack">
        <div className="row-between">
          <h2 id="jobs-h" className="mb0">Your jobs</h2>
          <button type="button" className="btn btn-ghost btn-sm" onClick={reload}>Refresh</button>
        </div>
        {error && <div className="banner banner-error" role="alert">{error}</div>}
        {all === null && <p role="status" className="muted"><span className="spinner" aria-hidden="true" /> Loading your jobs…</p>}
        <p className="muted small mb0" data-testid="jobs-note">Every file stays here, with its results, for 2 hours after it’s done. Open one to listen, download it again, change it or delete it.</p>
        {all?.length === 0 && !error && <p className="card muted mb0">Nothing here yet. Drop a file above and it’ll show up.</p>}
        {all?.map((j) => <JobRow key={j.id} job={j} />)}
      </section>

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
        <section className="card stack" aria-labelledby="tips-h">
          <h2 id="tips-h" className="mb0">Good to know</h2>
          <ul className="mb0">
            <li>Free noise removal happens <b>in your browser</b>, so that file never leaves your device.</li>
            <li>Uploading a video for us to work on? Only its <b>sound</b> is uploaded. Your browser puts the clean sound back into the video afterwards.</li>
            <li>Your upload and your results are kept together for <b>2 hours</b> after a file is done, so you can listen, download again or change it, and then they are deleted together. You can delete them sooner.</li>
            <li>Changing a finished file only redoes the steps that changed, so a new strength is often free. The price is shown before you confirm.</li>
            <li>You pay for what gets processed, by the second. A job that fails or is canceled costs nothing.</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
