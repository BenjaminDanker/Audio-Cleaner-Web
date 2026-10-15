import { useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../auth.jsx';
import JobCard from '../components/JobCard.jsx';
import JobForm from '../components/JobForm.jsx';
import { IconLock } from '../components/Icons.jsx';
import { usePageTitle, usePricing } from '../hooks.js';
import { useJobs } from '../useJobs.js';

export default function Transcribe() {
  usePageTitle('Free transcription');
  const { ready, signedIn } = useAuth();
  const { data: pricing } = usePricing();
  const free = pricing?.free;
  return (
    <div className="container page stack-lg">
      <header className="narrow">
        <h1>Free transcription</h1>
        <p className="lead muted">
          Turn speech into text and subtitles (TXT, SRT, VTT).
          {free ? ` Everyone with an account gets ${Math.round(free.transcribeSecondsPerDay / 60)} free minutes a day, for files up to ${Math.round(free.transcribeMaxFileSeconds / 60)} minutes.` : ''}
        </p>
        <p className="row small muted mb0"><IconLock width={18} height={18} /> Your upload is deleted once it’s processed. Results stay for 24 hours, then they’re gone too.</p>
      </header>
      {!ready && <p role="status" className="muted"><span className="spinner" aria-hidden="true" /> Loading…</p>}
      {ready && !signedIn && (
        <div className="card stack narrow">
          <h2 className="mb0">Sign in to transcribe</h2>
          <p className="mb0 muted">An account is how we keep count of your free minutes. It costs nothing and we don’t ask for a card.</p>
          <div><Link to="/login" className="btn btn-primary">Sign in</Link></div>
        </div>
      )}
      {ready && signedIn && <TranscribeApp />}
    </div>
  );
}

function TranscribeApp() {
  const { refresh } = useAuth();
  const { jobs, error, reload } = useJobs({ onFinished: refresh });
  const local = useRef(new Map());
  const submitted = useCallback((job, l) => { local.current.set(job.id, l); reload(); }, [reload]);
  const mine = (jobs || []).filter((j) => j.free);
  return (
    <div className="stack-lg">
      <div className="narrow"><JobForm variant="free" onSubmitted={submitted} /></div>
      <section className="stack narrow" aria-labelledby="fj-h">
        <h2 id="fj-h" className="mb0">Your free transcriptions</h2>
        {error && <div className="banner banner-error" role="alert">{error}</div>}
        {jobs === null && <p role="status" className="muted"><span className="spinner" aria-hidden="true" /> Loading…</p>}
        {jobs && mine.length === 0 && <p className="card muted mb0">Nothing yet. Your first one will show up here.</p>}
        {mine.map((j) => <JobCard key={j.id} job={j} local={local.current.get(j.id)} onChanged={reload} />)}
      </section>
    </div>
  );
}
