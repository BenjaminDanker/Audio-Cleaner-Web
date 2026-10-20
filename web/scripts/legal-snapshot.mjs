// The Terms of Service and the Privacy Policy as a visitor reads them, rendered to static HTML, with the SHA-256 of their text,
// and the decision that deploy/legal-archive.sh acts on: which version a text is (docs/operations.md, "Legal versions").
//
//   node scripts/legal-snapshot.mjs                 print: <doc> <sha256>   (nothing is written)
//   node scripts/legal-snapshot.mjs --decide --index FILE|-  [--today YYYY-MM-DD] [--out DIR]
//                                                   FILE lists what the server archive holds, one "<file> <sha256 or ->" per line (the
//                                                   archive is private to the service account: the shell script reads it and passes it
//                                                   in). Prints "<doc> <version> <sha256> <new|archived>" per page: a text whose hash
//                                                   is archived keeps that version; a new text gets today's date (-2, -3 ... when that
//                                                   date is taken). With --out, DIR gets the pages of the NEW versions
//                                                   (<doc>-<version>.html and .html.sha256) and current.json (the versions now live).
//   node scripts/legal-snapshot.mjs --hash-of FILE  the text hash of an archived page (FILE or - for stdin): no React needed
//
// What is hashed: the TEXT of the page (tags dropped, entities decoded, white space normalised, each link's target after its
// words), not the markup, so a style or class change is no new version; and not the "Version ..." line, which is the result of the
// decision. The pages are rendered with React's own renderToStaticMarkup from the real source files (src/pages/Terms.jsx, Privacy.jsx;
// SITE from src/site.js). The render date is never part of the hash, so the same text always has the same hash.
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, '..');

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

const VERSION_LINE = /^Version (\S+)\.$/m;
/** The hash of a page's text. The "Version <x>." line is not part of it: the version is decided from the hash. */
export const textHash = (text) => sha256(text.replace(VERSION_LINE, 'Version.'));

// ------------------------------------------------------------------------------------------------------- the render

const renderers = new Map();
/** What the pages show as their version without a deploy (npm run dev, the tests): vite.config.js says the same. */
export const DEV_VERSIONS = { terms: 'dev', privacy: 'dev' };

/** Bundle the real pages with esbuild (the site's own build tool) and load the bundle; the versions are what the build would define. */
async function loadRenderer(versions) {
  const cacheKey = JSON.stringify(versions);
  if (renderers.has(cacheKey)) return renderers.get(cacheKey);
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
    define: { 'process.env.NODE_ENV': '"production"', 'import.meta.env.BASE_URL': '"/"', __LEGAL_VERSIONS__: cacheKey },
  });
  const mod = { exports: {} };
  new Function('require', 'module', 'exports', out.outputFiles[0].text)(createRequire(import.meta.url), mod, mod.exports);
  renderers.set(cacheKey, mod.exports);
  return mod.exports;
}

/** Both pages: { terms: {name, version, markup, text, sha256}, privacy: {...}, site }, shown with the given versions. */
export async function renderAll(versions = DEV_VERSIONS) {
  const r = await loadRenderer(versions);
  const out = { site: r.SITE.name };
  for (const [key, doc] of Object.entries(DOCS)) {
    const markup = r.render(key, doc.route);
    const text = textOf(markup);
    const shown = VERSION_LINE.exec(text)?.[1];
    if (!shown) throw new Error(`${doc.source}: no "Version <date>." line in the rendered page`);
    if (shown !== versions[key]) throw new Error(`${doc.source}: shows version ${shown}, not ${versions[key]}`);
    out[key] = { name: doc.name, version: shown, markup, text, sha256: textHash(text) };
  }
  return out;
}

// ----------------------------------------------------------------------------------------------- the archived page

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
const MARK_OPEN = '<!--legal-text-->';
const MARK_CLOSE = '<!--/legal-text-->';

export const archiveName = (key, version) => `${key}-${version}.html`;

/** A standalone HTML document: the page's text, its version, the render date; the hash is that of the text between the markers. */
export function archiveHtml(key, d, site, { renderedAt, commit }) {
  const note = `Archived copy of the ${d.name} of ${site}, version ${d.version}, as the website showed it. `
    + `Rendered ${renderedAt}${commit ? ` from commit ${commit}` : ''}. SHA-256 of the text: ${d.sha256}.`;
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="legal-document" content="${key}">
<meta name="legal-version" content="${esc(d.version)}">
<meta name="legal-text-sha256" content="${d.sha256}">
<meta name="legal-rendered" content="${esc(renderedAt)}">
<title>${esc(d.name)}, version ${esc(d.version)} (archived) - ${esc(site)}</title>
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
  return textHash(textOf(html.slice(a + MARK_OPEN.length, b)));
}

// ------------------------------------------------------------------------------------------------------- the decision

const ARCHIVED = /^(terms|privacy)-(\d{4}-\d{2}-\d{2}(?:-\d+)?)\.html$/;

/** The archive listing ("<file> <sha256 or ->" per line, as the shell script prints it) as [{ doc, version, sha256 }];
 *  a file that is not an archived page is ignored, a hash that is not 64 hex digits is null (the page is there; what it says is not known). */
export function parseIndex(text) {
  const out = [];
  for (const line of text.split('\n')) {
    const [file, hash] = line.trim().split(/\s+/);
    const m = ARCHIVED.exec(file ?? '');
    if (m) out.push({ doc: m[1], version: m[2], sha256: /^[0-9a-f]{64}$/.test(hash ?? '') ? hash : null });
  }
  return out;
}

/** The version of each page, no manual step: a text whose hash is archived keeps its version (the newest, should a text come back);
 *  a new text is called today's date, or <date>-2, -3 ... when that date is taken. A version in the archive is never reused,
 *  whatever its file says. { terms: {version, sha256, isNew}, privacy: {...} } */
export function decide(hashes, index, today) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(today)) throw new Error(`not a date: ${today}`);
  const out = {};
  for (const key of Object.keys(DOCS)) {
    const mine = index.filter((e) => e.doc === key);
    const same = mine.filter((e) => e.sha256 === hashes[key]).map((e) => e.version).sort();
    if (same.length) {
      out[key] = { version: same[same.length - 1], sha256: hashes[key], isNew: false };
      continue;
    }
    const taken = new Set(mine.map((e) => e.version));
    let version = today;
    for (let n = 2; taken.has(version); n++) version = `${today}-${n}`;
    out[key] = { version, sha256: hashes[key], isNew: true };
  }
  return out;
}

/** What the server reads (acw/auth.py) and the deploy keeps in data/legal/current.json. */
export function currentJson(decided) {
  const entry = (k) => `  "${k}": { "version": "${decided[k].version}", "sha256": "${decided[k].sha256}" }`;
  return `{\n${Object.keys(DOCS).map(entry).join(',\n')}\n}\n`;
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
  if (!flag('--decide')) {
    for (const key of Object.keys(DOCS)) console.log(`${key} ${r[key].sha256}`);
    return 0;
  }

  const index = arg('--index');
  if (!index) throw new Error('--decide needs --index FILE (or - for stdin)');
  const today = arg('--today') || new Date().toLocaleDateString('en-CA'); // (YYYY-MM-DD, this machine's date)
  const decided = decide({ terms: r.terms.sha256, privacy: r.privacy.sha256 }, parseIndex(readFileSync(index === '-' ? 0 : index, 'utf8')), today);
  const out = arg('--out');
  if (out) {
    mkdirSync(out, { recursive: true });
    const shown = await renderAll({ terms: decided.terms.version, privacy: decided.privacy.version });
    const renderedAt = process.env.SOURCE_DATE_EPOCH
      ? new Date(Number(process.env.SOURCE_DATE_EPOCH) * 1000).toISOString()
      : new Date().toISOString().replace(/\.\d+Z$/, 'Z');
    const commit = commitOf();
    for (const key of Object.keys(DOCS)) {
      if (!decided[key].isNew) continue;
      const file = archiveName(key, decided[key].version);
      writeFileSync(join(out, file), archiveHtml(key, shown[key], shown.site, { renderedAt, commit }));
      writeFileSync(join(out, `${file}.sha256`), `${decided[key].sha256}\n`);
    }
    writeFileSync(join(out, 'current.json'), currentJson(decided));
  }
  for (const key of Object.keys(DOCS)) console.log(`${key} ${decided[key].version} ${decided[key].sha256} ${decided[key].isNew ? 'new' : 'archived'}`);
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main(process.argv.slice(2)).then((code) => process.exit(code), (e) => { console.error(`legal-snapshot: ${e.stack || e.message}`); process.exit(2); });
}
