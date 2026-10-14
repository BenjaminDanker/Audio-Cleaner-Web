import { Link } from 'react-router-dom';
import Legal from '../components/Legal.jsx';
import { usePageTitle } from '../hooks.js';
import { operatorLine, SITE } from '../site.js';

const SECTIONS = [
  ['about', 'About these terms'],
  ['accounts', 'Accounts'],
  ['services', 'The services'],
  ['credit', 'Prepaid credit and payments'],
  ['autoreload', 'Auto-reload'],
  ['aup', 'Acceptable use'],
  ['content', 'Your content'],
  ['data', 'Data handling'],
  ['availability', 'Availability and changes'],
  ['disclaimer', 'Disclaimers'],
  ['liability', 'Limitation of liability'],
  ['termination', 'Suspension and termination'],
  ['law', 'Governing law and contact'],
];

export default function Terms() {
  usePageTitle('Terms of Service');
  const operator = operatorLine() || 'the operator of this site';
  const jurisdiction = SITE.jurisdiction.trim();
  const email = SITE.supportEmail.trim();
  const address = SITE.legalAddress.trim();
  return (
    <Legal title="Terms of Service" sections={SECTIONS}>
      <h2 id="about">1. About these terms</h2>
      <p>These terms govern your use of {SITE.name} (the &ldquo;Service&rdquo;), operated by {operator} (&ldquo;we&rdquo;, &ldquo;us&rdquo;). By creating an account, buying credit, or using paid features you agree to them. If you do not agree, do not use the paid features. The free browser tool runs on your own device and needs no account.</p>
      <p>We may update these terms. When we do, the version date changes and we will ask you to accept the new version before you can use paid features again. Your continued use of the free browser tool is not conditional on this.</p>

      <h2 id="accounts">2. Accounts</h2>
      <ul>
        <li>You must be at least 16 years old (or the age of digital consent where you live) and able to enter a binding contract.</li>
        <li>Sign in with Google or an email link. Keep your sign-in methods and API keys secret; you are responsible for what happens under your account.</li>
        <li>One person, one account. Do not create accounts to evade limits, free allowances, or a suspension.</li>
        <li>Tell us promptly if you think your account or an API key has been compromised. You can revoke API keys on the Live page at any time.</li>
      </ul>

      <h2 id="services">3. The services</h2>
      <ul>
        <li><b>Free browser denoise.</b> Runs locally in your browser. Your audio is not uploaded. Provided as is, without charge or warranty.</li>
        <li><b>Studio clean.</b> Server-side noise removal, loudness normalisation and related processing of files you upload.</li>
        <li><b>Transcription and subtitles.</b> Standard (our own model, on our servers) or premium (sent to a third-party AI provider, see Privacy Policy). A daily free allowance of standard transcription may be offered; its size can change.</li>
        <li><b>Live.</b> Real-time noise removal and captions for streams, from the browser test page or the OBS plugin, billed by the second of audio processed.</li>
      </ul>
      <p>Prices are shown on the pricing page and before you start a job. We may change prices for future use; changes never apply to a job or live session that has already started.</p>

      <h2 id="credit">4. Prepaid credit and payments</h2>
      <ul>
        <li><b>How it works.</b> You buy credit in US dollars through Stripe (minimum top-up $10, maximum $500 at a time). Using the Service deducts credit at the posted rates. While a job or live session runs we reserve (&ldquo;hold&rdquo;) the estimated cost and release whatever is not used. Jobs that fail or are canceled before they start are not charged.</li>
        <li><b>No cash value.</b> Credit is a prepayment for the Service only. It is not money held on your behalf, not a deposit account, not a gift card or stored-value card, bears no interest, is not transferable, and cannot be exchanged for cash except as described below or where the law requires.</li>
        <li><b>Refunds.</b> You may ask for the unused part of a top-up to be refunded to the original payment method within 14 days of that payment (self-service on the Billing page). After 14 days credit is non-refundable, except where applicable law gives you a right to a refund. Credit you have already used is not refundable.</li>
        <li><b>Forfeiture.</b> If you dispute a legitimate charge with your bank (chargeback), or we find fraud, payment abuse or a serious breach of these terms, we may cancel any remaining credit, suspend the account, and recover amounts owed.</li>
        <li><b>Taxes.</b> Prices may be exclusive of VAT, sales tax or similar taxes, which are added where required.</li>
        <li><b>Payment processing.</b> Payments are handled by Stripe. We do not see or store your full card number.</li>
      </ul>

      <h2 id="autoreload">5. Auto-reload</h2>
      <p>Auto-reload is optional and off unless you turn it on. If you turn it on you authorize us (through Stripe) to charge your saved card the reload amount you chose whenever your balance falls below the threshold you chose, up to the monthly limit you chose per calendar month, until you turn it off. The exact wording you confirm on the Billing page is:</p>
      <blockquote><p>&ldquo;I authorize {SITE.name} to charge my saved card &lt;amount&gt; whenever my balance falls below &lt;threshold&gt;, up to &lt;cap&gt; per calendar month, until I turn this off.&rdquo;</p></blockquote>
      <ul>
        <li>You can change or turn off auto-reload at any time on the Billing page; this stops future reloads. You can also remove your saved card through the card manager.</li>
        <li>If an automatic charge fails, we may retry a limited number of times and then pause auto-reload and tell you on the Billing page.</li>
        <li>Auto-reloaded credit follows the same rules as any other credit, including the 14-day refund window for each charge.</li>
      </ul>

      <h2 id="aup">6. Acceptable use</h2>
      <p>You agree not to use the Service to:</p>
      <ul>
        <li>process, store or distribute content that is illegal where you live or where we operate, including child sexual abuse material, or content that you have no right to process;</li>
        <li>record or process other people&apos;s voices in violation of their privacy rights or applicable wiretapping and consent laws;</li>
        <li>harass, threaten, defraud or impersonate anyone;</li>
        <li>attack, probe or overload the Service, bypass rate limits, free allowances or billing, or resell access without our written permission;</li>
        <li>reverse engineer the Service except where the law allows it, or use automated means other than our documented API keys.</li>
      </ul>
      <p>We apply rate limits to protect the Service for everyone. We may remove content, stop jobs or sessions, and suspend accounts that break these rules, without refund of credit where the breach is serious.</p>

      <h2 id="content">7. Your content</h2>
      <p>You keep all rights in the audio, video and text you provide and in the results. You give us only the limited permission needed to receive, process and temporarily store your files to provide the Service to you. We do not use your audio or transcripts to train models. You confirm that you have the rights needed to process what you upload.</p>

      <h2 id="data">8. Data handling</h2>
      <p>The <Link to="/privacy">Privacy Policy</Link> applies. In short:</p>
      <ul>
        <li>The free browser tool processes files locally; nothing is uploaded.</li>
        <li>Files you upload are deleted when processing finishes. Results are deleted 24 hours after they are ready (or when you delete them).</li>
        <li>Live audio is processed in memory and is not stored.</li>
        <li>Premium transcription sends the audio to OpenAI as our sub-processor. Standard transcription does not leave our servers.</li>
        <li>Payments are processed by Stripe.</li>
      </ul>
      <p>Download your results promptly: we cannot recover deleted files.</p>

      <h2 id="availability">9. Availability and changes</h2>
      <p>We aim to keep the Service running but do not guarantee any uptime, speed or availability. It may be interrupted for maintenance, capacity or reasons beyond our control. Features, limits and free allowances may change. We may stop offering a feature with reasonable notice; unused credit then remains usable on other features or is refunded on request.</p>

      <h2 id="disclaimer">10. Disclaimers</h2>
      <p>The Service is provided &ldquo;as is&rdquo; and &ldquo;as available&rdquo;. Noise removal can alter or remove parts of a recording, and transcripts can contain errors. Check results before relying on them, and keep your originals. Do not rely on the Service for legal, medical, safety-critical or evidentiary purposes. To the extent the law allows we disclaim all warranties not expressly stated here.</p>

      <h2 id="liability">11. Limitation of liability</h2>
      <p>To the maximum extent permitted by law, we are not liable for indirect, incidental, special or consequential damages, or for lost profits, revenue, data or goodwill. Our total liability for any claim relating to the Service is limited to the greater of the amount you paid us in the 12 months before the claim and [USD 50 — to be confirmed]. Nothing in these terms excludes liability that cannot be excluded by law, including for fraud or for death or personal injury caused by negligence, and consumers keep their statutory rights.</p>

      <h2 id="termination">12. Suspension and termination</h2>
      <p>You can stop using the Service at any time and ask us to delete your account. We can suspend or end your access if you breach these terms, if required by law, or to protect the Service or other users. On termination without breach by you, we will refund unused credit on request; credit is forfeited as described in section 4 where the account is closed for fraud, chargebacks or serious breach.</p>

      <h2 id="law">13. Governing law and contact</h2>
      <p>
        These terms are governed by the laws of {jurisdiction || 'the place where the operator is established'}, and the courts {jurisdiction ? <>of {jurisdiction}</> : 'there'} have exclusive jurisdiction, without affecting any mandatory consumer rights you have in your country of residence. If a provision is unenforceable, the rest stays in force. These terms are the whole agreement between you and us about the Service.
      </p>
      {(email || address) && (
        <p>
          Contact: {email && <a href={`mailto:${email}`}>{email}</a>}{email && address ? ', ' : ''}{address}
        </p>
      )}
    </Legal>
  );
}
