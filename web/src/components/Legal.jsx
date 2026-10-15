import { Link } from 'react-router';
import { useAuth } from '../auth.jsx';
import { SITE } from '../site.js';

export const LEGAL_VERSION = '2026-10-15';

/** Wrapper for legal text: version and table of contents. */
export default function Legal({ title, sections, children }) {
  const email = SITE.supportEmail.trim();
  const { contact } = useAuth();
  return (
    <>
      <div className="container page">
        <article className="prose">
          <h1>{title}</h1>
          <p className="muted">
            Version {LEGAL_VERSION}.
          </p>
          <nav className="toc" aria-label="Contents">
            <ol>
              {sections.map(([id, label]) => <li key={id}><a href={`#${id}`}>{label}</a></li>)}
            </ol>
          </nav>
          {children}
          {contact
            ? <p className="small muted">Questions? <Link to="/contact">Write to us</Link>{email && <>, or email <a href={`mailto:${email}`}>{email}</a></>}. Or see <Link to="/">the home page</Link>.</p>
            : email && <p className="small muted">Questions? Email <a href={`mailto:${email}`}>{email}</a>, or see <Link to="/">the home page</Link>.</p>}
        </article>
      </div>
    </>
  );
}
