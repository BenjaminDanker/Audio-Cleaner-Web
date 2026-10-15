import { useEffect, useMemo, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth.jsx';
import { IconBroadcast, IconSparkle, IconText, IconWave } from '../components/Icons.jsx';
import PriceTable from '../components/PriceTable.jsx';
import { findRate, usePageTitle, usePricing } from '../hooks.js';
import { moneyCents, moneyUsd } from '../format.js';
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
  ['How fast is it?', 'Usually about one minute of waiting for every ten minutes of audio on a normal laptop. You get a progress bar and an estimate, so you know what to expect.'],
  ['What is Clean & polish for?', 'Longer recordings and finished files. We remove the noise, even out the voice and set the loudness (podcast, YouTube or broadcast), and you can add subtitles and a transcript in the same go. Your computer stays free in the meantime.'],
  ['Which files work?', 'On the Files page, almost any audio or video file: MP3, WAV, M4A/AAC, FLAC, OGG/Opus, WMA, AIFF, AMR, AC3, WavPack, MP4, MOV, MKV, WebM, AVI, WMV, FLV, MPEG, MTS/M2TS, 3GP, MXF and more. The free cleanup in your browser takes the common ones your browser can open (MP3, WAV, M4A, FLAC, OGG, MP4, MOV, WebM, MKV). With video, the picture is left alone and only the sound is swapped.'],
  ['Why do you only upload the sound of my video?', 'To keep uploads small and quick. Your browser pulls the audio out of the video on your device and sends only that. The cleaned audio goes back into your video on your device too.'],
  ['How does paying work?', 'You add credit, then spend it by the minute of audio. No subscription. You can top up from $10, and switch on auto-reload (with a monthly limit you choose) if you like. Turn it off whenever you want.'],
  ['What happens to my files and my words?', 'Files you upload are deleted as soon as they are processed. Results stay for 24 hours so you can download them, then they are deleted too. Live audio is never stored. We don’t train anything on your audio or your transcripts.'],
  ['Is there an OBS plugin?', 'Not yet, it’s on the way. Until then you can try live noise removal and captions right on the Live page, with your own microphone.'],
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
    if (hash) setTimeout(() => document.querySelector(hash)?.scrollIntoView(), 50);
  }, [hash]);
  const polish = findRate(pricing, 'denoise', 'standard');
  const tx = pricing?.free;
  const live = findRate(pricing, 'stream_denoise', 'light');
  const email = SITE.supportEmail.trim();
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
              <Link to="/free" className="btn btn-primary btn-lg">Try it on a file</Link>
              <a href="#products" className="btn btn-secondary btn-lg">See what else we do</a>
            </div>
            <p className="muted small">Nothing is uploaded. No watermark.</p>
          </div>
          <HeroArt />
        </div>
      </section>

      <section className="section" id="products">
        <div className="container">
          <h2 className="section-title">Four ways to get clean audio</h2>
          <p className="section-lead">Start with the free one. Come back when you have a long file, want subtitles, or go live.</p>
          <div className="grid grid-4">
            <article className="card product-card">
              <div className="icon"><IconWave /></div>
              <h3>Free cleanup <span className="badge badge-free">Free</span></h3>
              <p>Drop in a file and get it back clean, right in your browser. Nothing is uploaded, and the only limit is how patient your computer is.</p>
              <span className="spacer" />
              <Link to="/free" className="btn btn-primary">Try it now</Link>
            </article>
            <article className="card product-card">
              <div className="icon"><IconSparkle /></div>
              <h3>Clean &amp; polish</h3>
              <p>For long recordings and finished files: noise removed, voice evened out and loudness set for podcast, YouTube or broadcast.{polish && <> From {moneyUsd(polish.usdPerHour)} an hour.</>}</p>
              <span className="spacer" />
              <Link to="/app" className="btn btn-secondary">Clean a file</Link>
            </article>
            <article className="card product-card">
              <div className="icon"><IconBroadcast /></div>
              <h3>Live for streamers</h3>
              <p>Clean mic audio and live captions while you stream. You only pay for the minutes you are live.{live && <> From {moneyUsd(live.usdPerHour)} an hour.</>}</p>
              <span className="spacer" />
              <Link to="/app/live" className="btn btn-secondary">Try live</Link>
            </article>
            <article className="card product-card">
              <div className="icon"><IconText /></div>
              <h3>Subtitles &amp; transcripts <span className="badge badge-free">Free every day</span></h3>
              <p>Get TXT, SRT and VTT files from your recording. Choose Standard, or Extra accurate for tricky audio.{tx && <> Free: {Math.round(tx.transcribeSecondsPerDay / 60)} minutes a day.</>}</p>
              <span className="spacer" />
              <Link to="/app" className="btn btn-secondary">Get subtitles</Link>
            </article>
          </div>
        </div>
      </section>

      <section className="section alt" id="how">
        <div className="container">
          <h2 className="section-title">How it works</h2>
          <p className="section-lead">Three steps, whichever way you go.</p>
          <ol className="steps">
            <li><h3>Pick a way</h3><p>The free browser cleanup for quick jobs. Clean &amp; polish for long files, loudness and subtitles.</p></li>
            <li><h3>Drop in a file</h3><p>Audio or video. With video, only the sound is uploaded, so it stays small and quick.</p></li>
            <li><h3>Listen, then save</h3><p>Flip between the original and the clean version at the same spot. Then save the audio, or your video with the new sound.</p></li>
          </ol>
        </div>
      </section>

      <section className="section" id="pricing">
        <div className="container narrow stack">
          <h2 className="section-title mb0">What it costs</h2>
          <p className="section-lead">Noise removal in your browser is free, and so are your first subtitle minutes each day. For the rest you add credit and spend it by the minute of audio. That&rsquo;s all.</p>
          <PriceTable />
          <div className="card stack">
            <h3 className="mb0">How credit works</h3>
            <ul className="mb0">
              <li><b>No subscription.</b> Add credit from {pricing ? moneyCents(pricing.topup.minCents) : '$10'} and use it when you need it.</li>
              <li><b>You only pay for what you use.</b> You see an estimate before you upload. Live is counted by the second.</li>
              <li><b>Auto-reload is optional.</b> It tops you up when you run low, up to a monthly limit you choose. Turn it off any time.</li>
              <li><b>Changed your mind?</b> Unused credit can be refunded within 14 days. The <Link to="/terms">Terms</Link> have the details.</li>
            </ul>
            <div className="row">
              <Link to="/app/billing" className="btn btn-primary">Add credit</Link>
              <Link to="/login" className="btn btn-ghost">Make an account, it&rsquo;s free</Link>
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
          {email && <p className="center">Something else? <a href={`mailto:${email}`}>Just ask.</a></p>}
          <div className="center">
            <Link to="/free" className="btn btn-primary btn-lg">Try it on a file</Link>
          </div>
        </div>
      </section>
    </>
  );
}
