// The Terms of Service and the Privacy Policy as a visitor reads them, rendered to static HTML, with the SHA-256 of their text.
// The archive of every published version (docs/operations.md, "Legal versions") and the pin that guards the version dates
// (web/legal-versions.json, tests/legal-versions.mjs) both come from here.
//
//   node scripts/legal-snapshot.mjs                 print: <doc> <version> <sha256> <file name>   (nothing is written)
//   node scripts/legal-snapshot.mjs --out DIR       write DIR/<doc>-<version>.html and DIR/<doc>-<version>.html.sha256, and the same
//                                                   as <doc>-<version>-no-contact.html (the contact form off)
//   node scripts/legal-snapshot.mjs --verify        compare with web/legal-versions.json and the version constants; exit 1 on a problem
//   node scripts/legal-snapshot.mjs --pin           write web/legal-versions.json (refuses when a text changed under an unchanged version date)
//   node scripts/legal-snapshot.mjs --hash-of FILE  the text hash of an archived page (FILE or - for stdin): no React needed
//
// Two variants of each page: the contact form on (the text of the Terms and the Privacy Policy links to it) and off (when the server
// cannot send mail: /api/me says contactForm:false, which can flip without a deploy). Both are archived and pinned.
//
// What is hashed: the TEXT of the page (tags dropped, entities decoded, white space normalised, each link's target after its
// words), not the markup, so a style or class change does not need a new version date. The pages are rendered with React's own
// renderToStaticMarkup from the real source files (src/pages/Terms.jsx, Privacy.jsx; SITE from src/site.js); the only stand-in is
// the sign-in state (useAuth), because a visitor's text varies with it in one place: the contact form lines. The render date is never
// part of the hash, so the same text always has the same hash.
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, '..');
export const PIN_FILE = process.env.ACW_LEGAL_PIN || join(WEB, 'legal-versions.json'); // (the variable is for tests)
const AUTH_PY = resolve(WEB, '..', 'server', 'acw', 'auth.py');
const LEGAL_JSX = join(WEB, 'src', 'components', 'Legal.jsx');

export const DOCS = {
  terms: { name: 'Terms of Service', source: 'web/src/pages/Terms.jsx', route: '/terms' },
  privacy: { name: 'Privacy Policy', source: 'web/src/pages/Privacy.jsx', route: '/privacy' },
};

// ---------------------------------------------------------------------------------------------------------- the text

const ENTITIES = {
  amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', ldquo: '\u201c', rdquo: '\u201d', lsquo: '\u2018', rsquo: '\u2019',
  ndash: '\u2013', mdash: '\u2014', hellip: '\u2026', middot: '\u00b7', times: '\u00d7', copy: '\u00a9',
};
function decode(s) {
  return s.replace(/&(?:#x([0-9a-f]+)|#(\d+)|([a-z]+));/gi, (m, hex, dec, name) => {
    if (hex) return String.fromCodePoint(parseInt(hex, 16));
    if (dec) return String.fromCodePoint(parseInt(dec, 10));
    return ENTITIES[name.toLowerCase()] ?? m;
  });
}

/** The text of some markup, one block (paragraph, list item, heading, table row) per line. */
export function textOf(html) {
  const s = html
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/<(script|style)\b[\s\S]*?<\/\1>/gi, '')
    .replace(/<a\b[^>]*?\bhref="([^"]*)"[^>]*>([\s\S]*?)<\/a>/gi, (_, href, inner) => `${inner} [${href}]`)
    .replace(/<\/t[dh]>/gi, ' | ')
    .replace(/<\/(?:p|div|li|h[1-6]|tr|ul|ol|table|nav|article|section|thead|tbody|blockquote|pre)>|<br\s*\/?>/gi, '\n')
    .replace(/<[^>]+>/g, '');
  const lines = decode(s).split('\n').map((l) => l.replace(/\s+/g, ' ').trim()).filter(Boolean);
  return `${lines.join('\n')}\n`;
}

export const sha256 = (s) => createHash('sha256').update(s).digest('hex');

// ------------------------------------------------------------------------------------------------------- the render

let renderer = null;

/** Bundle the real pages with esbuild (the site's own build tool) and load the bundle; the sign-in state is a stand-in. */
async function loadRenderer() {
  if (renderer) return renderer;
  const { build } = await import('esbuild');
  const out = await build({
    stdin: {
      contents: `
        import { createElement } from 'react';
        import { renderToStaticMarkup } from 'react-dom/server';
        import { StaticRouter } from 'react-router';
        import Terms from '../src/pages/Terms.jsx';
        import Privacy from '../src/pages/Privacy.jsx';
        export { SITE } from '../src/site.js';
        const PAGES = { terms: Terms, privacy: Privacy };
        export const render = (doc, route) =>
          renderToStaticMarkup(createElement(StaticRouter, { location: route }, createElement(PAGES[doc])));
      `,
      resolveDir: HERE,
      loader: 'js',
    },
    bundle: true,
    format: 'cjs',
    platform: 'node',
    write: false,
    logLevel: 'silent',
    jsx: 'automatic',
    loader: { '.jsx': 'jsx' },
    define: { 'process.env.NODE_ENV': '"production"', 'import.meta.env.BASE_URL': '"/"' },
    plugins: [{
      name: 'stand-in-for-sign-in-state',
      setup(b) {
        b.onResolve({ filter: /(^|\/)auth\.jsx$/ }, () => ({ path: 'auth', namespace: 'stand-in' }));
        b.onLoad({ filter: /.*/, namespace: 'stand-in' }, () => ({ contents: 'export const useAuth = () => globalThis.__legalAuth;', loader: 'js' }));
      },
    }],
  });
  const mod = { exports: {} };
  new Function('require', 'module', 'exports', out.outputFiles[0].text)(createRequire(import.meta.url), mod, mod.exports);
  renderer = mod.exports;
  return renderer;
}

/** Both pages: { terms: {name, version, markup, text, sha256, noContact: {markup, text, sha256}}, privacy: {...}, site }.
 *  The top level is the page with the contact form on; `noContact` is the same page with it off. */
export async function renderAll() {
  const r = await loadRenderer();
  const out = { site: r.SITE.name };
  for (const [key, doc] of Object.entries(DOCS)) {
    const variants = [true, false].map((contact) => {
      globalThis.__legalAuth = { contact, signedIn: false, user: null };
      const markup = r.render(key, doc.route);
      const text = textOf(markup);
      const version = /^Version (\S+)\.$/m.exec(text)?.[1];
      if (!version) throw new Error(`${doc.source}: no "Version <date>." line in the rendered page`);
      return { name: doc.name, version, markup, text, sha256: sha256(text) };
    });
    if (variants[0].version !== variants[1].version) throw new Error(`${doc.source}: the version date depends on the contact form`);
    const { name, version, ...off } = variants[1];
    out[key] = { ...variants[0], noContact: off };
  }
  return out;
}

/** Every file of the archive: [{ key, doc, file, version, sha256, page, variant }], page being what archiveHtml takes. */
export function variantsOf(r) {
  const list = [];
  for (const key of Object.keys(DOCS)) {
    const d = r[key];
    list.push({ key, doc: key, file: archiveName(key, d.version), version: d.version, sha256: d.sha256, page: d, variant: 'contact form on' });
    list.push({
      key: `${key}-no-contact`, doc: key, file: archiveName(key, d.version, false), version: d.version, sha256: d.noContact.sha256,
      page: { ...d, ...d.noContact }, variant: 'contact form off',
    });
  }
  return list;
}

// ----------------------------------------------------------------------------------------------- the archived page

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
const MARK_OPEN = '<!--legal-text-->';
const MARK_CLOSE = '<!--/legal-text-->';

export const archiveName = (key, version, contact = true) => `${key}-${version}${contact ? '' : '-no-contact'}.html`;

/** A standalone HTML document: the page's text, its version, the render date; the hash is that of the text between the markers. */
export function archiveHtml(key, d, site, { renderedAt, commit, variant = 'contact form on' }) {
  const note = `Archived copy of the ${d.name} of ${site}, version ${d.version} (${variant}), as the website showed it. `
    + `Rendered ${renderedAt}${commit ? ` from commit ${commit}` : ''}. SHA-256 of the text: ${d.sha256}.`;
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="legal-document" content="${key}">
<meta name="legal-variant" content="${esc(variant)}">
<meta name="legal-version" content="${esc(d.version)}">
<meta name="legal-text-sha256" content="${d.sha256}">
<meta name="legal-rendered" content="${esc(renderedAt)}">
<title>${esc(d.name)}, version ${esc(d.version)} (${esc(variant)}, archived) - ${esc(site)}</title>
<style>
body { font: 16px/1.55 system-ui, sans-serif; color: #111; background: #fff; margin: 0; }
.archive-note { background: #f3f3f3; border-bottom: 1px solid #ccc; margin: 0; padding: 10px 16px; font-size: 13px; }
.page { max-width: 46rem; margin: 0 auto; padding: 16px; }
table { border-collapse: collapse; } td, th { border: 1px solid #ccc; padding: 4px 8px; text-align: left; vertical-align: top; }
.muted { color: #555; } .toc { background: #f7f7f7; padding: 8px 16px; }
</style>
</head>
<body>
<p class="archive-note">${esc(note)}</p>
${MARK_OPEN}
${d.markup}
${MARK_CLOSE}
</body>
</html>
`;
}

/** The text hash of an archived page, computed from its content (not taken from the file's own claim). */
export function hashOfArchive(html) {
  const a = html.indexOf(MARK_OPEN);
  const b = html.indexOf(MARK_CLOSE);
  if (a < 0 || b < a) throw new Error('not an archived legal page (no text markers)');
  return sha256(textOf(html.slice(a + MARK_OPEN.length, b)));
}

// ---------------------------------------------------------------------------------------------------------- the pin

export function readPin(file = PIN_FILE) {
  return existsSync(file) ? JSON.parse(readFileSync(file, 'utf8')) : null;
}

export function pinText(r) {
  const entry = (k) => `  "${k}": { "version": "${r[k].version}", "sha256": "${r[k].sha256}", "sha256NoContact": "${r[k].noContact.sha256}" }`;
  return `{\n${Object.keys(DOCS).map(entry).join(',\n')}\n}\n`;
}

/** Why a rendering does not match the pin; [] when it does. */
export function pinProblems(r, pin) {
  const out = [];
  for (const [key, doc] of Object.entries(DOCS)) {
    const p = pin?.[key];
    const cur = r[key];
    if (!p) out.push(`${doc.name}: web/legal-versions.json has no "${key}" entry. Update the pin: npm run legal:pin`);
    else if (p.version !== cur.version) {
      out.push(`${doc.name}: the version date is ${cur.version} but web/legal-versions.json pins ${p.version}. Update the pin: npm run legal:pin`);
    } else if (p.sha256 === undefined || p.sha256NoContact === undefined) {
      out.push(`${doc.name}: web/legal-versions.json lacks the text hash of a variant (sha256, sha256NoContact). Update the pin: npm run legal:pin`);
    } else if (p.sha256 !== cur.sha256 || p.sha256NoContact !== cur.noContact.sha256) {
      const which = [p.sha256 !== cur.sha256 && 'contact form on', p.sha256NoContact !== cur.noContact.sha256 && 'contact form off'].filter(Boolean).join(' and ');
      out.push(`${doc.name}: the text changed (${which}) but its version date is still ${cur.version}. `
        + `Change the version date in ${doc.source}, in LEGAL_VERSION (web/src/components/Legal.jsx) and TOS_VERSION (server/acw/auth.py) where they apply, `
        + 'and update the pin (npm run legal:pin).');
    }
  }
  return out;
}

/** The three places that name the Terms' version have to agree: the page, LEGAL_VERSION, TOS_VERSION. */
export function versionProblems(r) {
  const legal = /export const LEGAL_VERSION = '([^']+)'/.exec(readFileSync(LEGAL_JSX, 'utf8'))?.[1];
  const tos = /^TOS_VERSION = "([^"]+)"/m.exec(readFileSync(AUTH_PY, 'utf8'))?.[1];
  if (legal && tos && legal === tos && legal === r.terms.version) return [];
  return [`The Terms version dates disagree: page ${r.terms.version} (${DOCS.terms.source}), LEGAL_VERSION ${legal ?? 'not found'} `
    + `(web/src/components/Legal.jsx), TOS_VERSION ${tos ?? 'not found'} (server/acw/auth.py). They have to be the same date.`];
}

// ------------------------------------------------------------------------------------------------------------- CLI

function commitOf() {
  try {
    return execFileSync('git', ['-C', WEB, 'rev-parse', '--short', 'HEAD'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  } catch {
    return '';
  }
}

async function main(argv) {
  const flag = (n) => argv.includes(n);
  const arg = (n) => (argv.includes(n) ? argv[argv.indexOf(n) + 1] : undefined);

  if (flag('--hash-of')) {
    const f = arg('--hash-of');
    console.log(hashOfArchive(readFileSync(f === '-' ? 0 : f, 'utf8')));
    return 0;
  }

  const r = await renderAll();

  if (flag('--verify') || flag('--pin')) {
    const problems = versionProblems(r);
    const pinned = pinProblems(r, readPin());
    if (flag('--verify')) {
      for (const p of [...problems, ...pinned]) console.error(`legal versions: ${p}`);
      if (problems.length || pinned.length) return 1;
      console.log(`legal versions: ok (terms ${r.terms.version}, privacy ${r.privacy.version}; pin matches)`);
      return 0;
    }
    // --pin: only a deliberate change goes in (a new version date); a text edit under the old date is exactly what the pin catches.
    const unchanged = pinned.filter((p) => p.includes('version date is still'));
    for (const p of [...problems, ...unchanged]) console.error(`legal versions: ${p}`);
    if (problems.length || unchanged.length) return 1;
    writeFileSync(PIN_FILE, pinText(r));
    console.log(`legal versions: pinned terms ${r.terms.version} ${r.terms.sha256.slice(0, 12)}/${r.terms.noContact.sha256.slice(0, 12)}, privacy ${r.privacy.version} ${r.privacy.sha256.slice(0, 12)}/${r.privacy.noContact.sha256.slice(0, 12)} -> web/legal-versions.json`);
    return 0;
  }

  const out = arg('--out');
  if (out) mkdirSync(out, { recursive: true });
  const renderedAt = process.env.SOURCE_DATE_EPOCH
    ? new Date(Number(process.env.SOURCE_DATE_EPOCH) * 1000).toISOString()
    : new Date().toISOString().replace(/\.\d+Z$/, 'Z');
  const commit = commitOf();
  for (const v of variantsOf(r)) {
    if (out) {
      writeFileSync(join(out, v.file), archiveHtml(v.doc, v.page, r.site, { renderedAt, commit, variant: v.variant }));
      writeFileSync(join(out, `${v.file}.sha256`), `${v.sha256}\n`);
    }
    console.log(`${v.key} ${v.version} ${v.sha256} ${v.file}`);
  }
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main(process.argv.slice(2)).then((code) => process.exit(code), (e) => { console.error(`legal-snapshot: ${e.stack || e.message}`); process.exit(2); });
}
