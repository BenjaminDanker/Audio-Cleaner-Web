import { Link } from 'react-router-dom';

export const LEGAL_VERSION = '2026-10-10';

/** Wrapper for legal text: draft banner, version, table of contents. */
export default function Legal({ title, sections, children }) {
  return (
    <>
      <div className="draft-banner" role="note">Draft — under legal review</div>
      <div className="container page">
        <article className="prose">
          <h1>{title}</h1>
          <p className="muted">Version {LEGAL_VERSION}. This is a draft for review; wording will change before launch and you will be asked to accept the final terms.</p>
          <nav className="toc" aria-label="Contents">
            <ol>
              {sections.map(([id, label]) => <li key={id}><a href={`#${id}`}>{label}</a></li>)}
            </ol>
          </nav>
          {children}
          <p className="small muted">Questions? See <Link to="/">the home page</Link> for contact details (placeholder: support@example.com).</p>
        </article>
      </div>
    </>
  );
}
