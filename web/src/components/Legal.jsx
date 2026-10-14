import { Link } from 'react-router-dom';
import { legalComplete, SITE } from '../site.js';

export const LEGAL_VERSION = '2026-10-10';

/** Wrapper for legal text: draft banner (only while the operator details in site.js are incomplete), version, table of contents. */
export default function Legal({ title, sections, children }) {
  const draft = !legalComplete();
  const email = SITE.supportEmail.trim();
  return (
    <>
      {draft && <div className="draft-banner" role="note">Draft — under legal review</div>}
      <div className="container page">
        <article className="prose">
          <h1>{title}</h1>
          <p className="muted">
            Version {LEGAL_VERSION}.
            {draft ? ' This is a draft for review; wording will change before launch and you will be asked to accept the final terms.' : ''}
          </p>
          <nav className="toc" aria-label="Contents">
            <ol>
              {sections.map(([id, label]) => <li key={id}><a href={`#${id}`}>{label}</a></li>)}
            </ol>
          </nav>
          {children}
          {email && <p className="small muted">Questions? Email <a href={`mailto:${email}`}>{email}</a>, or see <Link to="/">the home page</Link>.</p>}
        </article>
      </div>
    </>
  );
}
