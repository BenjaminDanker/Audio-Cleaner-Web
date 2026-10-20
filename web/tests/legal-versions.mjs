// The Terms and the Privacy Policy text, as the deploy versions them (docs/operations.md, "Legal versions"): both pages render, their text
// is extracted and hashed the same way every time, the hash does not depend on the version shown, and the version of a text is decided
// from the hash alone (new text -> today's date, -2, -3 for more changes the same day; known text -> its version). No server archive, no
// backend, no browser: the CI web job runs this too.
// Usage: node tests/legal-versions.mjs
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { archiveHtml, archiveName, currentJson, decide, hashOfArchive, parseIndex, renderAll, sha256, textHash, textOf } from '../scripts/legal-snapshot.mjs';

const WEB = fileURLToPath(new URL('..', import.meta.url));
let failed = 0;
async function check(name, fn) {
  try {
    const detail = await fn();
    console.log(`PASS ${name}${detail ? ` -- ${detail}` : ''}`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${name} -- ${String(e.message).split('\n').slice(0, 6).join(' / ')}`);
  }
}

const r = await renderAll();

await check('both pages render with their text, the contact form always on; the text hashes are the same every time', async () => {
  for (const k of ['terms', 'privacy']) {
    assert.ok(r[k].text.length > 5000, `${k}: only ${r[k].text.length} characters of text`);
    assert.match(r[k].sha256, /^[0-9a-f]{64}$/);
    assert.ok(r[k].text.includes('[/contact]'), `${k}: the link to the contact form`);
  }
  const again = await renderAll();
  for (const k of ['terms', 'privacy']) assert.equal(again[k].sha256, r[k].sha256);
  assert.match(r.terms.text, /^Terms of Service\nVersion dev\.\n/);
  assert.match(r.privacy.text, /^Privacy Policy\nVersion dev\.\n/);
  return `terms ${r.terms.sha256.slice(0, 12)}, privacy ${r.privacy.sha256.slice(0, 12)}`;
});

await check('the pages show the versions the build gives them, and the hash does not change with them', async () => {
  const v = await renderAll({ terms: '2030-05-06', privacy: '2030-05-06-2' });
  assert.match(v.terms.text, /^Terms of Service\nVersion 2030-05-06\.\n/);
  assert.match(v.privacy.text, /^Privacy Policy\nVersion 2030-05-06-2\.\n/);
  for (const k of ['terms', 'privacy']) {
    assert.notEqual(v[k].text, r[k].text);
    assert.equal(v[k].sha256, r[k].sha256, `${k}: the version line is not part of the hash`);
  }
});

await check('the hash is of the text: markup and classes do not change it, words and link targets do', () => {
  const a = '<div class="x"><p>Hello &amp; <b>welcome</b>, see <a href="/privacy">the policy</a>.</p><ul><li>one</li><li>two</li></ul></div>';
  const base = sha256(textOf(a));
  assert.equal(sha256(textOf(a.replace('class="x"', 'class="y" id="z"').replace('<b>', '<strong>').replace('</b>', '</strong>'))), base);
  assert.notEqual(sha256(textOf(a.replace('welcome', 'welcome back'))), base);
  assert.notEqual(sha256(textOf(a.replace('/privacy', '/terms'))), base);
  assert.equal(textOf('<p>a &ldquo;b&rdquo; &#x27;c&#39; &lt;d&gt;</p>'), 'a “b” \'c\' <d>\n');
  const withVersion = (v) => textOf(`<h1>Terms</h1><p>Version ${v}.</p><p>Body</p>`);
  assert.equal(textHash(withVersion('2026-01-01')), textHash(withVersion('dev')));
  assert.notEqual(textHash(withVersion('1').replace('Body', 'Other body')), textHash(withVersion('1')));
});

await check('an archived page is a standalone document with its version and a hash that can be recomputed from the file alone', async () => {
  const v = await renderAll({ terms: '2030-05-06', privacy: '2030-05-06' });
  const seen = new Set();
  for (const k of ['terms', 'privacy']) {
    const html = archiveHtml(k, v[k], v.site, { renderedAt: '2026-01-02T03:04:05Z', commit: 'abc1234' });
    seen.add(hashOfArchive(html));
    assert.equal(hashOfArchive(html), r[k].sha256, `${k}: the archived page hashes like the page`);
    assert.ok(html.includes('<meta name="legal-version" content="2030-05-06">') && html.includes('Version 2030-05-06.'));
    assert.ok(html.includes('2026-01-02T03:04:05Z') && html.includes('abc1234') && html.includes(r[k].sha256) && html.startsWith('<!doctype html>'));
    assert.notEqual(hashOfArchive(html.replace('</h1>', ' (edited)</h1>')), r[k].sha256);
  }
  assert.equal(seen.size, 2);
  assert.equal(archiveName('terms', '2030-05-06-2'), 'terms-2030-05-06-2.html');
  assert.throws(() => hashOfArchive('<html>nothing</html>'), /not an archived legal page/);
});

const H1 = 'a'.repeat(64);
const H2 = 'b'.repeat(64);
const H3 = 'c'.repeat(64);
const none = { terms: H1, privacy: H2 };
const idx = (...e) => e.map(([doc, version, sha256]) => ({ doc, version, sha256 }));

await check('a text nobody archived yet is called today\'s date; a text that is archived keeps its version', () => {
  assert.deepEqual(decide(none, [], '2026-10-20'), {
    terms: { version: '2026-10-20', sha256: H1, isNew: true }, privacy: { version: '2026-10-20', sha256: H2, isNew: true },
  });
  const archive = idx(['terms', '2026-10-20', H1], ['privacy', '2026-09-01', H2]);
  const later = decide(none, archive, '2026-12-24');
  assert.deepEqual([later.terms.version, later.terms.isNew, later.privacy.version, later.privacy.isNew], ['2026-10-20', false, '2026-09-01', false]);
  // one page changed, the other did not: only the changed one gets a new version
  const half = decide({ terms: H3, privacy: H2 }, archive, '2026-12-24');
  assert.deepEqual([half.terms.version, half.terms.isNew, half.privacy.version, half.privacy.isNew], ['2026-12-24', true, '2026-09-01', false]);
});

await check('a second and third change on the same day get -2 and -3; a version is never reused; an old text that comes back is its old version again', () => {
  const day = [['terms', '2026-10-20', H1]];
  assert.equal(decide({ terms: H3, privacy: H2 }, idx(...day), '2026-10-20').terms.version, '2026-10-20-2');
  assert.equal(decide({ terms: 'd'.repeat(64), privacy: H2 }, idx(...day, ['terms', '2026-10-20-2', H3]), '2026-10-20').terms.version, '2026-10-20-3');
  // (an archived page whose hash is not known still holds its version)
  assert.equal(decide({ terms: H3, privacy: H2 }, idx(['terms', '2026-10-20', null]), '2026-10-20').terms.version, '2026-10-20-2');
  assert.equal(decide({ terms: H3, privacy: H2 }, idx(['terms', '2026-10-20', H1], ['terms', '2026-10-20-2', 'e'.repeat(64)]), '2026-10-20').terms.version, '2026-10-20-3');
  // the first text again after a change: its own version, not a new one
  const back = decide({ terms: H1, privacy: H2 }, idx(...day, ['terms', '2026-10-21', H3]), '2026-10-22');
  assert.deepEqual([back.terms.version, back.terms.isNew], ['2026-10-20', false]);
  // the other page's versions do not count
  assert.equal(decide({ terms: H3, privacy: H2 }, idx(['privacy', '2026-10-20', H2]), '2026-10-20').terms.version, '2026-10-20');
  assert.throws(() => decide(none, [], 'yesterday'), /not a date/);
});

await check('the archive listing the shell script prints is read as it is: pages only, a lost hash is unknown', () => {
  const listing = [`terms-2026-10-20.html ${H1}`, `terms-2026-10-20-2.html -`, `privacy-2026-10-20.html ${H2}`, 'current.json x', 'terms-2026-10-20.html.sha256 y', 'notes.txt', ''].join('\n');
  assert.deepEqual(parseIndex(listing), idx(['terms', '2026-10-20', H1], ['terms', '2026-10-20-2', null], ['privacy', '2026-10-20', H2]));
});

await check('the command line decides from a listing, writes the new pages and current.json, and leaves the archived ones alone', () => {
  const dir = mkdtempSync(join(tmpdir(), 'legal-test-'));
  try {
    const run = (listing, today) => execFileSync('node', ['scripts/legal-snapshot.mjs', '--decide', '--index', '-', '--today', today, '--out', dir],
      { input: listing, encoding: 'utf8', cwd: WEB }).trim().split('\n').map((l) => l.split(' '));
    const first = run('', '2026-10-20');
    assert.deepEqual(first.map((l) => [l[0], l[1], l[3]]), [['terms', '2026-10-20', 'new'], ['privacy', '2026-10-20', 'new']]);
    assert.deepEqual(readdirSync(dir).sort(), ['current.json', 'privacy-2026-10-20.html', 'privacy-2026-10-20.html.sha256', 'terms-2026-10-20.html', 'terms-2026-10-20.html.sha256']);
    const [terms, privacy] = first;
    assert.equal(readFileSync(join(dir, 'terms-2026-10-20.html.sha256'), 'utf8'), `${terms[2]}\n`);
    assert.equal(hashOfArchive(readFileSync(join(dir, 'terms-2026-10-20.html'), 'utf8')), terms[2]);
    assert.deepEqual(JSON.parse(readFileSync(join(dir, 'current.json'), 'utf8')), {
      terms: { version: '2026-10-20', sha256: terms[2] }, privacy: { version: '2026-10-20', sha256: privacy[2] },
    });
    assert.equal(readFileSync(join(dir, 'current.json'), 'utf8'), currentJson({ terms: { version: '2026-10-20', sha256: terms[2] }, privacy: { version: '2026-10-20', sha256: privacy[2] } }));
    // the same texts a month later: the same versions, nothing new to write
    const empty = mkdtempSync(join(tmpdir(), 'legal-test-'));
    try {
      const again = execFileSync('node', ['scripts/legal-snapshot.mjs', '--decide', '--index', '-', '--today', '2026-11-20', '--out', empty],
        { input: `terms-2026-10-20.html ${terms[2]}\nprivacy-2026-10-20.html ${privacy[2]}\n`, encoding: 'utf8', cwd: WEB });
      assert.equal(again, `terms 2026-10-20 ${terms[2]} archived\nprivacy 2026-10-20 ${privacy[2]} archived\n`);
      assert.deepEqual(readdirSync(empty), ['current.json']);
    } finally {
      rmSync(empty, { recursive: true, force: true });
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

process.exit(failed ? 1 : 0);
