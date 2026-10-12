import { useState } from 'react';
import { Link } from 'react-router-dom';
import pkg from '../../package.json';
import { usePageTitle } from '../hooks.js';

const BASE = import.meta.env.BASE_URL;
const v = (name) => (pkg.dependencies[name] || pkg.devDependencies[name] || '').replace(/^[\^~]/, '');

const ITEMS = [
  { name: 'DeepFilterNet (libDF) and DeepFilterNet3 model', by: 'Hendrik Schröter', license: 'MIT OR Apache-2.0', url: 'https://github.com/Rikorose/DeepFilterNet', what: 'Noise suppression: the model and signal-processing library that run in your browser and on our servers.' },
  { name: 'tract', by: 'Sonos', license: 'MIT OR Apache-2.0', url: 'https://github.com/sonos/tract', what: 'Neural-network (ONNX) inference engine compiled into the WebAssembly module.' },
  { name: 'Other Rust crates inside the WebAssembly module', by: 'Various authors', license: 'MIT, Apache-2.0, BSD (permissive)', url: null, what: 'A complete third-party crate list is generated at release time. [to be attached before launch]' },
  { name: 'mediabunny', by: 'Vanilagy and contributors', license: 'MPL-2.0', version: v('mediabunny'), url: 'https://github.com/Vanilagy/mediabunny', what: 'Reads and writes audio and video containers in your browser. Used unmodified; its source is available at the link and licence terms at mozilla.org/MPL/2.0.' },
  { name: 'React and React DOM', by: 'Meta Platforms and contributors', license: 'MIT', version: v('react'), url: 'https://github.com/facebook/react', what: 'User-interface library.' },
  { name: 'React Router', by: 'Remix Software and contributors', license: 'MIT', version: v('react-router-dom'), url: 'https://github.com/remix-run/react-router', what: 'Page routing.' },
  { name: 'Vite', by: 'Evan You, VoidZero and contributors', license: 'MIT', version: v('vite'), url: 'https://github.com/vitejs/vite', what: 'Build tool; small runtime helpers are part of the page bundle.' },
];

function LicenseText({ file, title }) {
  const [text, setText] = useState(null);
  const [open, setOpen] = useState(false);
  const toggle = (e) => {
    setOpen(e.currentTarget.open);
    if (e.currentTarget.open && text == null) {
      fetch(`${BASE}dfn/${file}`).then((r) => (r.ok ? r.text() : Promise.reject(new Error('missing')))).then(setText).catch(() => setText('Licence text unavailable. See the project repository linked above.'));
    }
  };
  return (
    <details onToggle={toggle}>
      <summary>{title}</summary>
      {open && <pre className="license">{text ?? 'Loading…'}</pre>}
    </details>
  );
}

export default function Licenses() {
  usePageTitle('Open-source licences');
  return (
    <div className="container page">
      <article className="prose">
        <h1>Open-source licences</h1>
        <p>Audio Cleaner is built on open-source software. We are grateful to the authors. Where a project is dual-licensed we use it under either licence; the texts below are included as required.</p>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Component</th><th>Licence</th><th>What it does here</th></tr></thead>
            <tbody>
              {ITEMS.map((i) => (
                <tr key={i.name}>
                  <th scope="row">{i.url ? <a href={i.url} rel="noopener noreferrer">{i.name}</a> : i.name}{i.version ? <span className="muted small"> {i.version}</span> : null}<div className="small muted">{i.by}</div></th>
                  <td className="nowrap">{i.license}</td>
                  <td>{i.what}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <h2>Licence texts</h2>
        <LicenseText file="LICENSE-MIT" title="DeepFilterNet — MIT licence" />
        <LicenseText file="LICENSE-APACHE" title="DeepFilterNet — Apache License 2.0" />
        <p className="small muted">The same files are served next to the model at <span className="mono">{BASE}dfn/</span>. The Mozilla Public License 2.0 for mediabunny is at <a href="https://www.mozilla.org/MPL/2.0/" rel="noopener noreferrer">mozilla.org/MPL/2.0</a>.</p>
        <p className="small muted">Back to <Link to="/">home</Link>.</p>
      </article>
    </div>
  );
}
