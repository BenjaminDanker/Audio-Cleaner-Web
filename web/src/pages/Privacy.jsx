import { Link } from 'react-router';
import { useAuth } from '../auth.jsx';
import Legal from '../components/Legal.jsx';
import { usePageTitle } from '../hooks.js';
import { operatorLine, SITE } from '../site.js';

const SECTIONS = [
  ['who', 'Who we are'],
  ['browser', 'Free noise removal and subtitles in your browser'],
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
  const { contact } = useAuth();
  const form = contact ? <Link to="/contact">contact form</Link> : null;
  return (
    <Legal title="Privacy Policy" sections={SECTIONS} version="2026-10-20">
      <h2 id="who">1. Who we are</h2>
      <p>{SITE.name} is operated by {operator}, the controller of your personal data for the account-based services.{(form || email) && <> Contact: {form && <>use the {form}</>}{form && email && ', or '}{email && <a href={`mailto:${email}`}>{email}</a>}.</>} The <Link to="/terms">Terms of Service</Link> apply alongside this policy.</p>
      <p>The Service is not available to people in the EU/EEA, the UK or Switzerland, and we do not offer it there. If a card issued in one of those places is used, we cancel the payment before it is charged (and refund it in full if it ever was).</p>

      <h2 id="browser">2. Free noise removal and subtitles in your browser</h2>
      <p>The free noise removal and the free subtitles on the <Link to="/app">Cleanup page</Link> process your file entirely on your device. Your audio and video are <b>not uploaded</b> and no server call carries your content. The only requests are for the web page, the noise remover and the subtitle maker themselves, and the public price list, none of which contain your data. The noise remover’s and the subtitle maker’s model files come from Hugging Face (huggingface.co), which, like any site your browser connects to, sees your IP address and browser type.</p>

      <h2 id="collect">3. What we collect</h2>
      <ul>
        <li><b>Account:</b> email address, your name if the sign-in provider shares it, and an identifier from Google if you use it to sign in.</li>
        <li><b>Billing:</b> amounts, dates and status of payments; a Stripe customer reference; your credit ledger. Card details are collected and stored by Stripe, never by us.</li>
        <li><b>Usage:</b> for each job its file name, size, duration, chosen options, status and cost; for live sessions their length and cost; API key names, prefixes and last-used times (the keys themselves are stored only as one-way hashes).</li>
        <li><b>Security logs:</b> IP address and browser type, used for sign-in sessions, abuse prevention and rate limiting.</li>
        <li><b>Messages to us:</b> if you write through the contact form we receive your message, the email address you give, your name if you give one and, if you were signed in, which account it came from. It reaches us as an email and is <b>not saved in our database</b>; we only note that a message was sent.</li>
      </ul>

      <h2 id="files">4. Your files and transcripts</h2>
      <ul>
        <li>For video, your browser extracts the audio track and uploads only that where it can; the video picture never leaves your device in that case.</li>
        <li>Uploaded input files, and the copies we keep of them so a result can be changed again quickly (the cleaned sound before your strength is applied, and the transcript), are <b>kept for 2 hours</b> after your last change to the job (finishing is the first; saving an edit, asking for a price, processing and exporting count too), and in any case are <b>deleted 6 hours after the upload</b>. The files of a job that fails or is canceled are deleted at once.</li>
        <li>Results (cleaned audio, captions, transcripts) are available for download for <b>2 hours</b> after your last change to the job (6 hours after the upload at the latest) and are then <b>deleted together with the upload</b>. You can delete them all sooner from the Cleanup page.</li>
        <li>We do not listen to your content, and we do <b>not use your audio or transcripts for training</b> models.</li>
        <li>If you give a file job a <b>callback address</b> (through the API), we send that address a short message when the job ends: its id, status, the list of result files and the cost. No audio or text goes with it.</li>
        <li><b>Standard</b> transcription is done by us and the audio goes no further. <b>Extra accurate</b> transcription sends the audio to a third-party speech-recognition provider (our sub-processor), which returns the text. It is used only for extra accurate transcription and extra accurate live captions, and only when you choose them. When you choose translations, the text of your captions or transcript (never the audio) is sent to the same provider to be translated. It may process audio in the United States, and it does not use the audio for training. Only choose Extra accurate if you are comfortable with that.</li>
      </ul>

      <h2 id="live">5. Live audio</h2>
      <p>Live audio is streamed to our servers, processed in memory and returned to you. It is <b>not stored</b>. Captions are delivered to you as they are produced and are not saved. If you choose extra accurate live captions, the audio is sent to the speech-recognition provider described in section 6 for that purpose. Standard live captions (English only) are produced by us and the audio goes no further.</p>

      <h2 id="subprocessors">6. Sub-processors and recipients</h2>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Who</th><th>Why</th><th>What they receive</th></tr></thead>
          <tbody>
            <tr><td>Stripe</td><td>Payments, saved cards, receipts, refunds</td><td>Email, payment details, amounts</td></tr>
            <tr><td>A third-party speech-recognition provider</td><td>Extra accurate transcription and extra accurate live captions only, and only when you choose them</td><td>The audio you choose to send. It may be processed in the United States and is not used for training</td></tr>
            <tr><td>Google</td><td>Optional sign-in</td><td>Their own sign-in data; we receive your verified email and name</td></tr>
            <tr><td>Email delivery provider</td><td>Sign-in links, and messages sent through the contact form</td><td>Your email address and the message</td></tr>
            <tr><td>Hugging Face</td><td>The free tools’ model files, which your browser downloads from them</td><td>Your IP address and browser type, as with any site your browser connects to</td></tr>
            <tr><td>Hosting provider</td><td>Running the website and servers</td><td>Technical traffic data</td></tr>
          </tbody>
        </table>
      </div>
      <p>We do not sell personal data and do not share it for advertising.</p>

      <h2 id="cookies">7. Cookies and tracking</h2>
      <p>We use one strictly necessary, HttpOnly session cookie while you are signed in (plus a short-lived cookie during Google sign-in). We use <b>no analytics, advertising trackers, third-party fonts or social widgets</b>. The free tools’ model files are downloaded from Hugging Face (section 2). We load no third-party scripts.</p>

      <h2 id="retention">8. How long we keep things</h2>
      <ul>
        <li>Account data: while your account exists, then deleted on request unless we must keep it by law.</li>
        <li>A one-way hash of your email address, kept for good, even after you delete your account, only so the free live trial is not given twice. It cannot be turned back into your address and is not linked to your account.</li>
        <li>Payment and ledger records: 7 years, as tax and accounting law requires (when an account is deleted they are kept without your name or email). Records that go with them, such as that you accepted the Terms, started a payment, turned automatic top-up on or off, or were refunded or credited, are kept just as long.</li>
        <li>Job records and live-session records (metadata only, no content): deleted 12 months after the job or session. File names are removed from your credit history after 12 months; what you paid stays there as a line without the name (date, kind of work, length, amount) for the 7 years above.</li>
        <li>Security logs, including IP addresses and browser types of sign-ins and live sessions: 90 days. The entries themselves (a sign-in, a key created, a contact message sent) are deleted after 12 months, without IP addresses after the first 90 days.</li>
        <li>Sign-in sessions and sign-in links: deleted when they run out, and in any case within 30 days after that. Small bookkeeping rows: payment notification ids about 90 days.</li>
        <li>Messages sent through the contact form: they are not in our database. The email we receive is kept for as long as we need it to answer you and deal with your request.</li>
        <li>Backups of our database: kept for 30 days, so data you deleted can remain in a backup until it expires.</li>
        <li>Uploaded files and results: as described in section 4.</li>
      </ul>

      <h2 id="rights">9. Your rights</h2>
      <p>Depending on where you live (for example under the CCPA) you may have the right to access, correct, delete or export your data, and to object to or restrict some processing. You can download a copy of your data or delete your account yourself on the <Link to="/app/account">Account page</Link>; for anything else, {form || email ? <>{form && <>write to us through the {form} (choose &ldquo;Privacy or my data&rdquo;; the same form helps with refunds and billing questions)</>}{form && email && ', or '}{email && <>email us at <a href={`mailto:${email}`}>{email}</a></>}</> : <>use the <Link to="/app/account">Account page</Link></>} and we will respond within the time the law requires. Our legal bases are performing our contract with you, our legitimate interest in security and abuse prevention, and legal obligations (such as tax records).</p>

      <h2 id="security">10. Security</h2>
      <p>Connections use TLS. Session tokens, sign-in links and API keys are stored only as hashes. Billing and key management need a signed-in browser session, and state-changing requests are protected against cross-site request forgery. No system is perfectly secure; tell us if you find a problem.</p>

      <h2 id="other">11. Children, transfers and changes</h2>
      <p>The Service is not intended for children under 16. Your data is processed in the United States. We do not offer the Service in the EU/EEA, the UK or Switzerland, so we do not transfer the data of people there. We will post changes to this policy here with a new version date and, for material changes, notify account holders.</p>
    </Legal>
  );
}
