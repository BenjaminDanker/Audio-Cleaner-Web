import { useEffect, useMemo, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router';
import { useAuth } from '../auth.jsx';
import { IconBroadcast, IconText, IconWave } from '../components/Icons.jsx';
import PriceTable from '../components/PriceTable.jsx';
import { usePageTitle, usePricing } from '../hooks.js';
import { moneyCents } from '../format.js';
import { SITE } from '../site.js';

/** Decorative before/after waveform (deterministic pseudo-noise, no assets). */
function HeroArt() {
  const { noisy, clean } = useMemo(() => {
    let seed = 7;
    const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    const w = 520;
    const n = 130;
    const env = (i) => {
      const t = i / n;
      return Math.max(0.05, Math.sin(t * Math.PI * 3.2) ** 2 * 0.9 * (0.55 + 0.45 * Math.sin(t * 17)));
    };
    let a = ''; let b = '';
    for (let i = 0; i <= n; i++) {
      const x = (i / n) * w;
      const e = env(i);
      const yn = 60 + (rnd() * 2 - 1) * (22 + 40 * e);
      const yc = 60 + Math.sin(i * 1.7) * e * 44 * (i % 2 ? 1 : -1) * 0.9;
      a += `${i ? 'L' : 'M'}${x.toFixed(1)} ${yn.toFixed(1)}`;
      b += `${i ? 'L' : 'M'}${x.toFixed(1)} ${yc.toFixed(1)}`;
    }
    return { noisy: a, clean: b };
  }, []);
  return (
    <svg className="hero-art" viewBox="0 0 520 292" role="img" aria-label="A noisy waveform above and the same recording after cleaning below">
      <text className="wave-label" x="6" y="14">Before</text>
      <path className="wave-noisy" d={noisy} />
      <line x1="0" y1="146" x2="520" y2="146" stroke="var(--border)" />
      <g transform="translate(0 172)">
        <text className="wave-label" x="6" y="-4">After</text>
        <path className="wave-clean wave-sweep" d={clean} />
      </g>
    </svg>
  );
}

const FAQ = [
  ['Does the free tool upload my audio?', 'No. It runs right in your browser, on your own device, and nothing about your file is sent anywhere. You do need a free account to use it, to keep bots away.'],
  ['How fast is it?', 'Usually a little over a minute of waiting for every ten minutes of audio on a normal laptop. You get a progress bar and an estimate, so you know what to expect.'],
  ['Is this always the best option?', 'No, and we’d rather tell you. YouTube already gives English uploads free automatic captions and free auto-translate, and big English channels get free live captions. OBS and NVIDIA Broadcast have free noise filters that are fine for mild noise, and Macs have Voice Isolation. If you only want captions on your own screen, Windows, Macs, Chrome and phones have free built-in Live Captions. Every option here has a “Good to know” note saying when something you already have is enough. We’re for stubborn noise, Twitch and other sites, files you keep, and fewer mistakes.'],
  ['What is Denoising for?', 'Longer recordings and finished files. We remove the noise, Leveling evens out the voice and sets the loudness (podcast, YouTube or broadcast), and you can add subtitles and a transcript in the same go. Your computer stays free in the meantime.'],
  ['Which files work?', 'On the Cleanup page, almost any audio or video file: MP3, WAV, M4A/AAC, FLAC, OGG/Opus, WMA, AIFF, AMR, AC3, WavPack, MP4, MOV, MKV, WebM, AVI, WMV, FLV, MPEG, MTS/M2TS, 3GP, MXF and more. Noise removal in your browser takes the common ones your browser can open (MP3, WAV, M4A, FLAC, OGG, MP4, MOV, WebM, MKV). With video, the picture is left alone and only the sound is swapped.'],
  ['Why do you only upload the sound of my video?', 'To keep uploads small and quick. Your browser pulls the audio out of the video on your device and sends only that. The cleaned audio goes back into your video on your device too.'],
  ['How does paying work?', 'You add credit, then spend it by the minute of audio. No subscription. You can top up from $10, and switch on auto-reload (with a monthly limit you choose) if you like. Turn it off whenever you want.'],
  ['What happens to my files and my words?', 'Files you upload are deleted as soon as they are processed. Results stay for 24 hours so you can download them, then they are deleted too. Live audio is never stored. We don’t train anything on your audio or your transcripts.'],
  ['What does the free trial cover?', 'Your first two hours of live are free: live noise removal and Standard captions, once per email address. Extra accurate captions are paid, and the free trial is paused for a while whenever live is busy.'],
  ['Where is it available?', 'Audio Cleaner isn’t offered in the EU/EEA, the UK or Switzerland. If a card issued there is used by mistake, the payment is cancelled and the card isn’t charged.'],
  ['Is there an OBS plugin?', 'Yes, for Windows, Mac and Linux: get it on the Live page, where you can also try everything first with your own microphone. Several people? Add it to each microphone, or split a stereo source with one person per side, and name each speaker.'],
];

/**
 * One-time notice passed through router state, e.g. after account deletion. The state is cleared from the history entry
 * right away so a reload or a later visit does not show it again. After a deletion the auth state is refreshed here, once
 * this page (which needs no sign-in) is showing.
 */
function useOneTimeNotice() {
  const location = useLocation();
  const navigate = useNavigate();
  const { refresh } = useAuth();
  const [notice, setNotice] = useState(() => location.state?.notice || null);
  useEffect(() => {
    if (!location.state?.notice) return;
    setNotice(location.state.notice);
    if (location.state.notice === 'account-deleted') refresh();
    navigate({ pathname: location.pathname, search: location.search, hash: location.hash }, { replace: true, state: null });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return [notice, () => setNotice(null)];
}

export default function Landing() {
  usePageTitle('');
  const { hash } = useLocation();
  const { data: pricing } = usePricing();
  const [notice, dismissNotice] = useOneTimeNotice();
  useEffect(() => {
    if (!hash) return undefined;
    // By id, not as a CSS selector: "#1" or "#a b" in the address must not throw.
    let id = hash.slice(1);
    try {
      id = decodeURIComponent(id);
    } catch { /* keep the text as it is */ }
    const timer = setTimeout(() => document.getElementById(id)?.scrollIntoView(), 50);
    return () => clearTimeout(timer);
  }, [hash]);
  const email = SITE.supportEmail.trim();
  const { contact, signedIn, known } = useAuth();
  return (
    <>
      {notice === 'account-deleted' && (
        <div className="container notice-row">
          <div className="banner banner-success row-between" role="status" data-testid="account-deleted">
            <span>Your account has been deleted.</span>
            <button type="button" className="btn btn-ghost btn-sm" onClick={dismissNotice}>Dismiss</button>
          </div>
        </div>
      )}
      <section className="hero">
        <div className="container hero-grid">
          <div>
            <h1>Clean up a recording. Free, right in your browser.</h1>
            <p className="lead">Take the hiss, hum and room noise out of podcasts, calls, lectures and videos. It all happens on your own device, so your file never leaves it.</p>
            <div className="row">
              <Link to="/app" className="btn btn-primary btn-lg">Try it on a file</Link>
              <a href="#pricing" className="btn btn-secondary btn-lg">See what it costs</a>
            </div>
            <p className="muted small">No watermark, no daily limit.</p>
          </div>
          <HeroArt />
        </div>
      </section>

      <section className="section" id="products">
        <div className="container">
          <h2 className="section-title">Pick what you need</h2>
          <p className="section-lead">Anything that runs in your browser is free. When we do the work for you, you pay by the minute, and only for what you use.</p>
          <div className="grid grid-3">
            <article className="card product-card">
              <div className="icon"><IconWave /></div>
              <h3>Clean up a recording</h3>
              <p>Hiss, hum, wind, keyboards and room noise come out; the voice stays natural. Audio or video, almost any format.</p>
              <ul className="ways">
                <li><span className="badge badge-free">Free</span><span>In your browser. Your file never leaves your device.</span></li>
                <li><span className="badge">Paid</span><span><b>Denoising</b>, stronger on heavy noise, and <b>Leveling</b>: the voice evened out, loudness set for podcast, YouTube or broadcast.</span></li>
              </ul>
              <Link to="/app" className="btn btn-primary">Clean a file</Link>
            </article>
            <article className="card product-card">
              <div className="icon"><IconText /></div>
              <h3>Subtitles &amp; transcripts</h3>
              <p>SRT, VTT or plain text, or the subtitles put right into your video. Translations if you want them.</p>
              <ul className="ways">
                <li><span className="badge badge-free">Free</span><span>In your browser, for clear speech.</span></li>
                <li><span className="badge">Paid</span><span><b>Standard</b>, or <b>Extra accurate</b> for accents, noisy rooms and specialist words.</span></li>
              </ul>
              <Link to="/app" className="btn btn-secondary">Get subtitles</Link>
            </article>
            <article className="card product-card">
              <div className="icon"><IconBroadcast /></div>
              <h3>Go live</h3>
              <p>Your mic cleaned up and captions on screen while you stream, in OBS or any browser tab.</p>
              <ul className="ways">
                <li><span className="badge badge-free">Free</span><span>Your first two hours.</span></li>
                <li><span className="badge">Paid</span><span>After that, by the second while you&rsquo;re live.</span></li>
              </ul>
              <Link to="/app/live" className="btn btn-secondary">Try live</Link>
            </article>
          </div>
        </div>
      </section>

      <section className="section alt" id="how">
        <div className="container">
          <h2 className="section-title">How it works</h2>
          <p className="section-lead">Three steps, whichever way you go.</p>
          <ol className="steps">
            <li><h3>Choose free or Denoising</h3><p>Free runs in your browser. Denoising does the work for you with stronger noise removal, and Leveling sets the loudness.</p></li>
            <li><h3>Drop in a file</h3><p>Audio or video. With video, only the sound is uploaded, so it stays small and quick.</p></li>
            <li><h3>Listen, then save</h3><p>Flip between the original and the clean version at the same spot. Then save the audio, or your video with the new sound.</p></li>
          </ol>
        </div>
      </section>

      <section className="section" id="pricing">
        <div className="container narrow stack">
          <h2 className="section-title mb0">What it costs</h2>
          <p className="section-lead">Noise removal and subtitles in your browser are free. For the rest you add credit and spend it by the minute of audio. That&rsquo;s all.</p>
          <PriceTable />
          <div className="card stack">
            <h3 className="mb0">How credit works</h3>
            <ul className="mb0">
              <li><b>Your first two hours of live are free.</b> Noise removal and Standard captions, no card needed.</li>
              <li><b>No subscription.</b> Add credit from {pricing ? moneyCents(pricing.topup.minCents) : '$5'} and use it when you need it.</li>
              <li><b>You only pay for what you use.</b> You see an estimate before you upload. Live is counted by the second.</li>
              <li><b>Auto-reload is optional.</b> It tops you up when you run low, up to a monthly limit you choose. Turn it off any time.</li>
              <li><b>Changed your mind?</b> Unused credit can be refunded within 14 days. The <Link to="/terms">Terms</Link> have the details.</li>
            </ul>
            <div className="row">
              <Link to="/app/billing" className="btn btn-primary">Add credit</Link>
              {known && !signedIn && <Link to="/login" className="btn btn-ghost">Sign in to start</Link>}
            </div>
          </div>
        </div>
      </section>

      <section className="section" id="faq">
        <div className="container narrow faq">
          <h2 className="section-title">Good to know</h2>
          <div>
            {FAQ.map(([q, a]) => (
              <details key={q}><summary>{q}</summary><div><p>{a}</p></div></details>
            ))}
          </div>
          {contact ? <p className="center">Something else? <Link to="/contact">Just ask.</Link></p>
            : email && <p className="center">Something else? <a href={`mailto:${email}`}>Just ask.</a></p>}
          <div className="center">
            <Link to="/app" className="btn btn-primary btn-lg">Try it on a file</Link>
          </div>
        </div>
      </section>
    </>
  );
}
