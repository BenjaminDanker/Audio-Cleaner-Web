import { Link } from 'react-router';
import { SITE } from '../site.js';

// The version of the Terms and of the Privacy Policy, filled in by the build (vite.config.js) from what deploy/legal-archive.sh
// decided: the date a text first went live (docs/operations.md, "Legal versions"). Nobody edits a version by hand. Without a deploy
// (npm run dev, the tests) both are "dev".
export const LEGAL_VERSIONS = __LEGAL_VERSIONS__;
export const LEGAL_VERSION = LEGAL_VERSIONS.terms; // the Terms' version, which signed-in users accept

/** Wrapper for legal text: version and table of contents. `version`: this page's own version (LEGAL_VERSIONS.terms or .privacy). */
export default function Legal({ title, sections, version = LEGAL_VERSION, children }) {
  const email = SITE.supportEmail.trim();
  return (
    <>
      <div className="container page">
        <article className="prose">
          <h1>{title}</h1>
          <p className="muted">
            Version {version}.
          </p>
          <nav className="toc" aria-label="Contents">
            <ol>
              {sections.map(([id, label]) => <li key={id}><a href={`#${id}`}>{label}</a></li>)}
            </ol>
          </nav>
          {children}
          <p className="small muted">Questions? <Link to="/contact">Write to us</Link>{email && <>, or email <a href={`mailto:${email}`}>{email}</a></>}. Or see <Link to="/">the home page</Link>.</p>
        </article>
      </div>
    </>
  );
}
