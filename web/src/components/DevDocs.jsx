import { Link } from 'react-router';
import { BASE } from '../api.js';
import { SITE } from '../site.js';

/** Where the API lives, for the examples: this site's address plus its base path, no trailing slash. */
export function apiBase() {
  return `${window.location.origin}${BASE.replace(/\/$/, '')}`;
}

/** A page of developer documentation: a short intro, a contents list (a side column on wide screens) and sections. */
export function DevPage({ title, lead, toc, children }) {
  return (
    <div className="container page">
      <div className="dev-layout">
        <nav className="dev-toc card card-flat" aria-label="On this page">
          <h2>On this page</h2>
          <ol>
            {toc.map(([id, label]) => <li key={id}><a href={`#${id}`}>{label}</a></li>)}
          </ol>
          <p className="small muted mb0">
            <Link to="/developers">Files</Link> · <Link to="/developers/live">Live</Link>
          </p>
        </nav>
        <article className="dev-main">
          <h1>{title}</h1>
          <p className="lead muted">{lead}</p>
          {children}
          <p className="small muted dev-foot">
            Something missing or unclear? {SITE.supportEmail.trim() ? <>Email <a href={`mailto:${SITE.supportEmail.trim()}`}>{SITE.supportEmail.trim()}</a>.</> : <>Use the <Link to="/contact">contact form</Link>.</>}
          </p>
        </article>
      </div>
    </div>
  );
}

export function Section({ id, title, children }) {
  return (
    <section id={id} className="dev-section" aria-labelledby={`${id}-h`}>
      <h2 id={`${id}-h`}>{title}</h2>
      {children}
    </section>
  );
}

/** A code sample. Focusable so it can be scrolled with the keyboard. */
export function Code({ children, label = 'Example' }) {
  return (
    <pre className="code" tabIndex={0} aria-label={label}><code>{children}</code></pre>
  );
}

/** A table that scrolls sideways on a phone. `head` is a list of column names, `rows` a list of lists of cells. */
export function Table({ head, rows, label, codeFirst = true }) {
  return (
    <div className="table-wrap" aria-label={label}>
      <table className="dev-table">
        <thead><tr>{head.map((h) => <th key={h} scope="col">{h}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{r.map((c, j) => (j === 0 ? <th key={j} scope="row">{codeFirst ? <code>{c}</code> : c}</th> : <td key={j} data-label={head.length > 2 ? head[j] : undefined}>{c}</td>))}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
