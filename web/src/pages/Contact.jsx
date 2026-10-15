import { useEffect, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router';
import { api, waitText } from '../api.js';
import { useAuth } from '../auth.jsx';
import { useNow, usePageTitle } from '../hooks.js';
import NeedsServer from '../components/Maintenance.jsx';
import { SITE } from '../site.js';

const TOPICS = [
  ['question', 'A question'],
  ['billing', 'Billing or a refund'],
  ['job', 'A problem with a job'],
  ['privacy', 'Privacy or my data'],
  ['other', 'Something else'],
];
const MIN = 10;
const MAX = 5000;
const JOB_ID = /^[0-9a-f]{32}$/i;

/** What the visitor reads when the message could not be sent. */
function friendly(err) {
  if (err.status === 422) return 'Something in the form doesn’t look right. Please check it and try again.';
  if (err.status === 404) return 'The contact form isn’t available right now. Please try again a little later.';
  return err.detail || 'We couldn’t send your message just now. Please try again.';
}

export default function Contact() {
  usePageTitle('Contact');
  const { ready, signedIn, user, contact } = useAuth();
  const [params] = useSearchParams();
  const topicParam = params.get('topic');
  const jobParam = (params.get('job') || '').trim();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [topic, setTopic] = useState(TOPICS.some(([k]) => k === topicParam) ? topicParam : JOB_ID.test(jobParam) ? 'job' : 'question');
  const [job, setJob] = useState(JOB_ID.test(jobParam) ? jobParam.toLowerCase() : '');
  const [message, setMessage] = useState('');
  const [trap, setTrap] = useState(''); // a field people never see; only programs fill it
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [sentTo, setSentTo] = useState('');
  const [blockedUntil, setBlockedUntil] = useState(0);
  const touched = useRef(false);
  const alertRef = useRef(null);
  const now = useNow(blockedUntil ? 1000 : 3_600_000);

  // Signed in: start with the account's address (still editable).
  useEffect(() => {
    if (user?.email && !touched.current) setEmail(user.email);
  }, [user?.email]);
  useEffect(() => {
    if (error) alertRef.current?.focus({ preventScroll: false });
  }, [error]);

  if (!ready) return <div className="loading-screen" role="status"><span className="spinner" aria-hidden="true" /> <span className="sr-only">Loading</span></div>;

  // Not known yet (the server did not answer): the "right back" notice while it is away, else a moment of loading.
  // Once the form is open it stays open if the server drops out, so a long message is never thrown away.
  if (contact === null) {
    return <NeedsServer><div className="loading-screen" role="status"><span className="spinner" aria-hidden="true" /> <span className="sr-only">Loading</span></div></NeedsServer>;
  }
  const supportEmail = SITE.supportEmail.trim();
  if (!contact) {
    return (
      <div className="container narrow page">
        <div className="card stack">
          <h1 className="mb0">Contact</h1>
          <p className="mb0">The contact form isn’t switched on right now.{supportEmail ? <> You can email us at <a href={`mailto:${supportEmail}`}>{supportEmail}</a>.</> : ' Please try again a little later.'}</p>
          <div><Link to="/" className="btn btn-secondary">Back to the start</Link></div>
        </div>
      </div>
    );
  }

  const waitLeft = Math.max(0, Math.ceil((blockedUntil - now) / 1000));
  const messageLen = message.trim().length;
  const emailOk = /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email.trim());
  const canSend = emailOk && messageLen >= MIN && messageLen <= MAX && !busy && waitLeft === 0;

  async function submit(e) {
    e.preventDefault();
    if (!canSend) return;
    setBusy(true);
    setError('');
    try {
      await api('contact', {
        method: 'POST',
        redirectOn401: false,
        body: {
          name: name.trim() || undefined,
          email: email.trim(),
          topic,
          message: message.trim(),
          job: topic === 'job' && job.trim() ? job.trim() : undefined,
          website: trap,
        },
      });
      setSentTo(email.trim());
      setMessage('');
    } catch (err) {
      setError(friendly(err));
      if (err.status === 429 && err.retryAfter) setBlockedUntil(Date.now() + Math.min(err.retryAfter, 86400) * 1000);
    } finally {
      setBusy(false);
    }
  }

  if (sentTo) {
    return (
      <div className="container narrow page">
        <div className="card stack">
          <h1 className="mb0">Contact</h1>
          <div className="banner banner-success" role="status" data-testid="contact-done">
            <p><b>Thanks, we got your message and will reply by email.</b></p>
            <p className="mb0">We’ll write to {sentTo}. If nothing has arrived in a couple of days, take a look in your spam folder.</p>
          </div>
          <div className="row">
            <Link to="/" className="btn btn-primary">Back to the start</Link>
            <button type="button" className="btn btn-secondary" onClick={() => { setSentTo(''); setJob(''); }}>Send another message</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container narrow page">
      <div className="card stack">
        <div>
          <h1 className="mb0">Contact</h1>
          <p className="muted mb0">A question, a problem with a job, a refund or a data request: write to us here and we’ll reply by email.</p>
        </div>
        <form className="stack" onSubmit={submit} noValidate data-testid="contact-form">
          <div className="field">
            <label htmlFor="c-topic">What is it about?</label>
            <select id="c-topic" className="select" value={topic} onChange={(e) => setTopic(e.target.value)}>
              {TOPICS.map(([k, label]) => <option key={k} value={k}>{label}</option>)}
            </select>
          </div>
          {topic === 'job' && (
            <div className="field">
              <label htmlFor="c-job">Job number <span className="muted">(optional)</span></label>
              <input id="c-job" className="input" value={job} onChange={(e) => setJob(e.target.value)} maxLength={64} autoComplete="off" spellCheck="false" />
              <span className="hint">The long code of the job, if you have it. It helps us find what happened.</span>
            </div>
          )}
          <div className="field">
            <label htmlFor="c-name">Your name <span className="muted">(optional)</span></label>
            <input id="c-name" className="input" value={name} onChange={(e) => setName(e.target.value)} maxLength={100} autoComplete="name" />
          </div>
          <div className="field">
            <label htmlFor="c-email">Your email address</label>
            <input id="c-email" className="input" type="email" inputMode="email" autoComplete="email" required value={email} onChange={(e) => { touched.current = true; setEmail(e.target.value); }} placeholder="you@example.com" />
            <span className="hint">{signedIn ? 'This is where we’ll reply. We can see which account you wrote from.' : 'This is where we’ll reply.'}</span>
          </div>
          <div className="field">
            <label htmlFor="c-message">Your message</label>
            <textarea id="c-message" className="input" rows={7} required value={message} onChange={(e) => setMessage(e.target.value)} maxLength={MAX} aria-describedby="c-count" />
            <span className="hint" id="c-count">{messageLen < MIN ? `A few words is plenty (at least ${MIN} characters).` : `${messageLen} of ${MAX} characters`}</span>
          </div>
          <div className="hp" aria-hidden="true">
            <label htmlFor="c-extra">Leave this empty</label>
            <input id="c-extra" name="c-extra" tabIndex={-1} autoComplete="off" value={trap} onChange={(e) => setTrap(e.target.value)} />
          </div>
          {error && <div className="banner banner-error" role="alert" tabIndex={-1} ref={alertRef} data-testid="contact-error">{error}</div>}
          {waitLeft > 0 && <p className="muted small mb0" role="status" data-testid="contact-wait">You can send again in {waitText(waitLeft)}.</p>}
          <div className="row">
            <button type="submit" className="btn btn-primary" disabled={!canSend}>{busy ? 'Sending…' : 'Send message'}</button>
          </div>
          <p className="small muted mb0">We use your address only to answer you. Your message is emailed to us and isn’t saved on the website; see the <Link to="/privacy">Privacy Policy</Link>.</p>
        </form>
      </div>
    </div>
  );
}
