import { Link } from 'react-router';
import pkg from '../../package.json';
import { COMPONENTS } from '../../scripts/license-list.mjs';
import { useAuth } from '../auth.jsx';
import { usePageTitle } from '../hooks.js';
import { SITE } from '../site.js';

const BASE = import.meta.env.BASE_URL;
const version = (name) => (name ? (pkg.dependencies[name] || pkg.devDependencies[name] || '').replace(/^[\^~]/, '') : '');

export default function Licenses() {
  usePageTitle('Open-source licences');
  const { contact } = useAuth();
  const email = SITE.supportEmail.trim();
  return (
    <div className="container page">
      <article className="prose">
        <h1>Open-source licences</h1>
        <p>{SITE.name} is built on open-source software. We are grateful to the authors. Where a project is dual-licensed we use it under either licence. The full licence text of every component is linked below and also served from <span className="mono">{BASE}licenses/</span>.</p>
        <div className="table-wrap">
          <table className="licenses" data-testid="licenses">
            <thead><tr><th>Component</th><th>Licence</th><th>What it does here</th><th>Licence text</th></tr></thead>
            <tbody>
              {COMPONENTS.map((c) => {
                const v = version(c.pkg);
                return (
                  <tr key={c.name}>
                    <th scope="row">{c.url ? <a href={c.url} rel="noopener noreferrer">{c.name}</a> : c.name}{v ? <span className="muted small"> {v}</span> : null}<div className="small muted">{c.by}</div></th>
                    <td>{c.license}</td>
                    <td>{c.what}</td>
                    <td>
                      <ul className="plain">
                        {c.texts.map((t) => <li key={t.file || t.href}><a href={t.href || `${BASE}licenses/${t.file}`} target="_blank" rel="noopener noreferrer">{t.label}</a></li>)}
                      </ul>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="small muted">The free tools’ models (DeepFilterNet with its WebAssembly module, and Whisper base) are downloaded by your browser from Hugging Face, where they are published with their licence files.</p>
        {(contact || email) && (
          <p className="small muted">Want the source code of one of these, as the licences allow? Ask {contact ? <>through the <Link to="/contact">contact form</Link></> : <>at <a href={`mailto:${email}`}>{email}</a></>} and we will send it.</p>
        )}
        <p className="small muted">Back to <Link to="/">home</Link>.</p>
      </article>
    </div>
  );
}
