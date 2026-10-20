// The version dates of the Terms and the Privacy Policy: the text of each page has to match web/legal-versions.json (the pin), unless
// its version date was changed; and the Terms' date has to be the same in the page, LEGAL_VERSION (src/components/Legal.jsx) and
// TOS_VERSION (server/acw/auth.py), because that date is what signed-in users accept and what the audit log records. No backend, no browser.
// Fails with: change the version date in the page, LEGAL_VERSION/TOS_VERSION where they apply, and update the pin (npm run legal:pin).
// Usage: node tests/legal-versions.mjs
import assert from 'node:assert/strict';
import { archiveHtml, hashOfArchive, pinProblems, readPin, renderAll, sha256, textOf, variantsOf, versionProblems } from '../scripts/legal-snapshot.mjs';

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

await check('both pages render, with the contact form on and off, with their version date; the text hashes are the same every time', async () => {
  for (const k of ['terms', 'privacy']) {
    assert.match(r[k].version, /^\d{4}-\d{2}-\d{2}/, `${k} version`);
    assert.ok(r[k].text.length > 5000, `${k}: only ${r[k].text.length} characters of text`);
    assert.match(r[k].sha256, /^[0-9a-f]{64}$/);
    assert.match(r[k].noContact.sha256, /^[0-9a-f]{64}$/);
    assert.notEqual(r[k].noContact.sha256, r[k].sha256, `${k}: the contact form should show in the text`);
    assert.ok(r[k].text.includes('[/contact]') && !r[k].noContact.text.includes('[/contact]'), `${k}: the link to the contact form`);
  }
  const again = await renderAll();
  for (const k of ['terms', 'privacy']) {
    assert.equal(again[k].sha256, r[k].sha256);
    assert.equal(again[k].noContact.sha256, r[k].noContact.sha256);
  }
  assert.match(r.terms.text, /^Terms of Service\nVersion /);
  assert.match(r.privacy.text, /^Privacy Policy\nVersion /);
  return `terms ${r.terms.version} ${r.terms.sha256.slice(0, 12)}/${r.terms.noContact.sha256.slice(0, 12)}, privacy ${r.privacy.version} ${r.privacy.sha256.slice(0, 12)}/${r.privacy.noContact.sha256.slice(0, 12)}`;
});

await check('the text of each page matches web/legal-versions.json (a changed text needs a new version date and a new pin)', () => {
  const problems = pinProblems(r, readPin());
  assert.deepEqual(problems, [], problems.join('\n'));
});

await check('the Terms version is the same in the page, LEGAL_VERSION and TOS_VERSION', () => {
  const problems = versionProblems(r);
  assert.deepEqual(problems, [], problems.join('\n'));
});

await check('a text changed under an unchanged version date fails (either variant) and says what to change; a new date without the pin fails too', () => {
  const pin = Object.fromEntries(['terms', 'privacy'].map((k) => [k, { version: r[k].version, sha256: r[k].sha256, sha256NoContact: r[k].noContact.sha256 }])); // (what the real pin would be)
  assert.equal(pinProblems(r, pin).length, 0);
  for (const [field, which] of [['sha256', 'contact form on'], ['sha256NoContact', 'contact form off']]) {
    const same = pinProblems(r, { ...pin, terms: { ...pin.terms, [field]: sha256('an earlier wording') } });
    assert.equal(same.length, 1);
    for (const w of ['Terms of Service', 'the text changed', which, 'still', 'version date', 'LEGAL_VERSION', 'TOS_VERSION', 'update the pin']) assert.ok(same[0].includes(w), `message lacks "${w}": ${same[0]}`);
  }
  const older = { ...pin, privacy: { version: '2020-01-01', sha256: sha256('x'), sha256NoContact: sha256('y') } };
  const stale = pinProblems(r, older);
  assert.equal(stale.length, 1);
  assert.ok(stale[0].includes('Privacy Policy') && stale[0].includes('pins 2020-01-01') && stale[0].includes('Update the pin'), stale[0]);
  assert.equal(pinProblems(r, { terms: pin.terms }).length, 1); // no entry for the other page
  const oneVariant = pinProblems(r, { ...pin, terms: { version: pin.terms.version, sha256: pin.terms.sha256 } });
  assert.ok(oneVariant.length === 1 && oneVariant[0].includes('lacks'), oneVariant.join());
});

await check('a Terms date that differs from LEGAL_VERSION or TOS_VERSION is reported', () => {
  const p = versionProblems({ ...r, terms: { ...r.terms, version: '2000-01-01' } });
  assert.equal(p.length, 1);
  assert.ok(p[0].includes('2000-01-01') && p[0].includes('LEGAL_VERSION') && p[0].includes('TOS_VERSION'), p[0]);
});

await check('the hash is of the text: markup and classes do not change it, words and link targets do', () => {
  const a = '<div class="x"><p>Hello &amp; <b>welcome</b>, see <a href="/privacy">the policy</a>.</p><ul><li>one</li><li>two</li></ul></div>';
  const base = sha256(textOf(a));
  assert.equal(sha256(textOf(a.replace('class="x"', 'class="y" id="z"').replace('<b>', '<strong>').replace('</b>', '</strong>'))), base);
  assert.notEqual(sha256(textOf(a.replace('welcome', 'welcome back'))), base);
  assert.notEqual(sha256(textOf(a.replace('/privacy', '/terms'))), base);
  assert.equal(textOf('<p>a &ldquo;b&rdquo; &#x27;c&#39; &lt;d&gt;</p>'), 'a “b” \'c\' <d>\n');
});

await check('the archive has a page for each variant, named after the version; each carries its version, the render date and a hash that can be recomputed from the file alone', () => {
  const files = variantsOf(r);
  assert.deepEqual(files.map((v) => v.file), [
    `terms-${r.terms.version}.html`, `terms-${r.terms.version}-no-contact.html`, `privacy-${r.privacy.version}.html`, `privacy-${r.privacy.version}-no-contact.html`,
  ]);
  const seen = new Set();
  for (const v of files) {
    const html = archiveHtml(v.doc, v.page, r.site, { renderedAt: '2026-01-02T03:04:05Z', commit: 'abc1234', variant: v.variant });
    seen.add(hashOfArchive(html));
    assert.equal(hashOfArchive(html), v.sha256, v.file);
    assert.ok(html.includes(`<meta name="legal-version" content="${v.version}">`) && html.includes(`<meta name="legal-variant" content="${v.variant}">`));
    assert.ok(html.includes('2026-01-02T03:04:05Z') && html.includes('abc1234') && html.includes(v.sha256) && html.startsWith('<!doctype html>'));
    assert.notEqual(hashOfArchive(html.replace('Version ', 'Versions ')), v.sha256);
  }
  assert.equal(seen.size, 4); // four different texts
  assert.throws(() => hashOfArchive('<html>nothing</html>'), /not an archived legal page/);
});

process.exit(failed ? 1 : 0);
