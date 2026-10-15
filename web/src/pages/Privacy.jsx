import { Link } from 'react-router-dom';
import Legal from '../components/Legal.jsx';
import { usePageTitle } from '../hooks.js';
import { operatorLine, SITE } from '../site.js';

const SECTIONS = [
  ['who', 'Who we are'],
  ['browser', 'The free browser tool'],
  ['collect', 'What we collect'],
  ['files', 'Your files and transcripts'],
  ['live', 'Live audio'],
  ['subprocessors', 'Sub-processors'],
  ['cookies', 'Cookies and tracking'],
  ['retention', 'How long we keep things'],
  ['rights', 'Your rights'],
  ['security', 'Security'],
  ['other', 'Children, transfers and changes'],
];

export default function Privacy() {
  usePageTitle('Privacy Policy');
  const operator = operatorLine() || 'the operator of this site';
  const email = SITE.supportEmail.trim();
  return (
    <Legal title="Privacy Policy" sections={SECTIONS}>
      <h2 id="who">1. Who we are</h2>
      <p>{SITE.name} is operated by {operator}, the controller of your personal data for the account-based services.{email && <> Contact: <a href={`mailto:${email}`}>{email}</a>.</>} The <Link to="/terms">Terms of Service</Link> apply alongside this policy.</p>

      <h2 id="browser">2. The free browser tool</h2>
      <p>The free tool at <Link to="/free">/free</Link> processes your file entirely on your device. Your audio and video are <b>not uploaded</b> and no server call carries your content. The only requests are for the web page, the noise remover itself and the public price list, none of which contain your data.</p>

      <h2 id="collect">3. What we collect</h2>
      <ul>
        <li><b>Account:</b> email address, your name if the sign-in provider shares it, and an identifier from Google if you use it to sign in.</li>
        <li><b>Billing:</b> amounts, dates and status of payments; a Stripe customer reference; your credit ledger. Card details are collected and stored by Stripe, never by us.</li>
        <li><b>Usage:</b> for each job its file name, size, duration, chosen options, status and cost; for live sessions their length and cost; API key names, prefixes and last-used times (the keys themselves are stored only as one-way hashes).</li>
        <li><b>Security logs:</b> IP address and browser type, used for sign-in sessions, abuse prevention and rate limiting.</li>
      </ul>

      <h2 id="files">4. Your files and transcripts</h2>
      <ul>
        <li>For video, your browser extracts the audio track and uploads only that where it can; the video picture never leaves your device in that case.</li>
        <li>Uploaded input files are <b>deleted when processing finishes</b> (or if the job fails or is canceled).</li>
        <li>Results (cleaned audio, captions, transcripts) are available for download for <b>24 hours</b> and then deleted. You can delete them sooner from the Files page.</li>
        <li>We do not listen to your content, and we do <b>not use your audio or transcripts for training</b> models.</li>
        <li><b>Standard</b> transcription is done by us and the audio goes no further. <b>Extra accurate</b> transcription sends the audio to a third-party speech-recognition provider (our sub-processor), which returns the text. It is used only for extra accurate transcription and extra accurate live captions, and only when you choose them. It may process audio in the United States under standard contractual clauses, and it does not use the audio for training. Only choose Extra accurate if you are comfortable with that.</li>
      </ul>

      <h2 id="live">5. Live audio</h2>
      <p>Live audio is streamed to our servers, processed in memory and returned to you. It is <b>not stored</b>. Captions are delivered to you as they are produced and are not saved. If you choose extra accurate live captions, the audio is sent to the speech-recognition provider described in section 6 for that purpose. Standard live captions (English only) are produced by us and the audio goes no further.</p>

      <h2 id="subprocessors">6. Sub-processors and recipients</h2>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Who</th><th>Why</th><th>What they receive</th></tr></thead>
          <tbody>
            <tr><td>Stripe</td><td>Payments, saved cards, receipts, refunds</td><td>Email, payment details, amounts</td></tr>
            <tr><td>A third-party speech-recognition provider</td><td>Extra accurate transcription and extra accurate live captions only, and only when you choose them</td><td>The audio you choose to send. It may be processed in the United States under standard contractual clauses and is not used for training</td></tr>
            <tr><td>Google</td><td>Optional sign-in</td><td>Their own sign-in data; we receive your verified email and name</td></tr>
            <tr><td>Cloudflare (Turnstile)</td><td>Bot check on the email sign-in form, when enabled</td><td>A challenge token and your IP address</td></tr>
            <tr><td>Email delivery provider</td><td>Sign-in links</td><td>Your email address and the message</td></tr>
            <tr><td>Hosting provider</td><td>Running the website and servers</td><td>Technical traffic data</td></tr>
          </tbody>
        </table>
      </div>
      <p>We do not sell personal data and do not share it for advertising.</p>

      <h2 id="cookies">7. Cookies and tracking</h2>
      <p>We use one strictly necessary, HttpOnly session cookie while you are signed in (plus a short-lived cookie during Google sign-in). We use <b>no analytics, advertising trackers, third-party fonts, CDNs or social widgets</b>. The only third-party script is Cloudflare Turnstile, loaded only on the sign-in page when enabled.</p>

      <h2 id="retention">8. How long we keep things</h2>
      <ul>
        <li>Account data: while your account exists, then deleted on request unless we must keep it by law.</li>
        <li>Payment and ledger records: as long as tax and accounting law requires [period to be confirmed].</li>
        <li>Job records (metadata only, no content): [period to be confirmed]. Security logs: [period to be confirmed].</li>
        <li>Uploaded files and results: as described in section 4.</li>
      </ul>

      <h2 id="rights">9. Your rights</h2>
      <p>Depending on where you live (for example under the GDPR/UK GDPR or CCPA) you may have the right to access, correct, delete or export your data, to object to or restrict some processing, and to complain to your data protection authority. You can download a copy of your data or delete your account yourself on the <Link to="/app/account">Account page</Link>; for anything else, email us and we will respond within the time the law requires. Our legal bases are performing our contract with you, our legitimate interest in security and abuse prevention, and legal obligations (such as tax records).</p>

      <h2 id="security">10. Security</h2>
      <p>Connections use TLS. Session tokens, sign-in links and API keys are stored only as hashes. Billing and key management need a signed-in browser session, and state-changing requests are protected against cross-site request forgery. No system is perfectly secure; tell us if you find a problem.</p>

      <h2 id="other">11. Children, transfers and changes</h2>
      <p>The Service is not intended for children under 16. If data is transferred outside your country we rely on appropriate safeguards [to be confirmed]. We will post changes to this policy here with a new version date and, for material changes, notify account holders.</p>
    </Legal>
  );
}
