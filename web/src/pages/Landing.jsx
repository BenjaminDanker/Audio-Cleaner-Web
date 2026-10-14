import { useEffect, useMemo, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth.jsx';
import { IconBroadcast, IconDownload, IconLock, IconSparkle, IconText, IconWave } from '../components/Icons.jsx';
import PriceTable from '../components/PriceTable.jsx';
import { findRate, usePageTitle, usePricing } from '../hooks.js';
import { moneyCents, moneyUsd } from '../format.js';

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
  ['Does the free tool upload my audio?', 'No. The free tool runs the DeepFilterNet3 model on your own device inside your browser. After the page and the model have loaded, nothing about your file is sent anywhere.'],
  ['How fast is the browser version?', 'Roughly one minute of processing for every ten minutes of mono audio on a typical laptop, since it uses your own CPU. Stereo takes about as long when your device has a spare core. You get a progress bar and an estimate.'],
  ['What is the difference to Studio clean?', 'Studio clean runs on our servers and adds a clarity chain and loudness normalisation (podcast, YouTube or broadcast targets) on top of noise removal. It is meant for long recordings and finished deliverables, and it can transcribe in the same pass.'],
  ['Which files work?', 'Audio and video: MP3, WAV, FLAC, OGG, AAC/M4A, MP4, MOV, WebM and MKV in current Chrome, Edge, Firefox and Safari. For video, the original picture is kept untouched; only the audio track is replaced.'],
  ['Why does Studio only upload the audio of my video?', 'To save you time and bandwidth. Your browser pulls the audio track out of the video locally and uploads just that. The cleaned audio is put back into your video on your device.'],
  ['How does billing work?', 'Prepaid credit, no subscription. Top up from $10 and spend it per minute of audio. You can turn on optional auto-reload with a monthly cap, and turn it off at any time.'],
  ['What happens to my files?', 'Uploaded files are deleted as soon as processing finishes. Results are kept for 24 hours so you can download them, then deleted. Live audio is processed in memory and never stored. We do not use your audio or transcripts for training.'],
  ['When is the OBS plugin available?', 'It is in development. You can already create API keys and try live denoise and captions from your browser on the Live page.'],
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
  const studio = findRate(pricing, 'denoise', 'standard');
  const tx = pricing?.free;
  const live = findRate(pricing, 'stream_denoise', 'light');
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
            <h1>Clean up your voice. Free, right in your browser.</h1>
            <p className="lead">Remove background noise from podcasts, calls, lectures and videos with the same kind of AI used in professional tools. Your file never leaves your device.</p>
            <div className="row">
              <Link to="/free" className="btn btn-primary btn-lg">Clean a file for free</Link>
              <a href="#products" className="btn btn-secondary btn-lg">See Studio &amp; Live</a>
            </div>
            <p className="muted small">No account. No upload. No watermark.</p>
          </div>
          <HeroArt />
        </div>
      </section>

      <section className="section" id="products">
        <div className="container">
          <h2 className="section-title">Four ways to get clean audio</h2>
          <p className="section-lead">Start free in your browser. Move up when you need longer files, finished loudness, subtitles, or live streaming.</p>
          <div className="grid grid-4">
            <article className="card product-card">
              <div className="icon"><IconWave /></div>
              <h3>Free browser denoise <span className="badge badge-free">Free</span></h3>
              <p>Runs entirely on your device with DeepFilterNet3. No sign-in, no upload, no limits except your own CPU.</p>
              <span className="spacer" />
              <Link to="/free" className="btn btn-primary">Try it now</Link>
            </article>
            <article className="card product-card">
              <div className="icon"><IconSparkle /></div>
              <h3>Studio clean</h3>
              <p>Heavier server-side processing for files: noise removal plus a clarity chain and loudness targets for podcast, YouTube or broadcast.{studio && <> From {moneyUsd(studio.usdPerHour)} per hour.</>}</p>
              <span className="spacer" />
              <Link to="/app" className="btn btn-secondary">Open Studio</Link>
            </article>
            <article className="card product-card">
              <div className="icon"><IconBroadcast /></div>
              <h3>Live for streamers</h3>
              <p>OBS plugin with low-latency denoise and live captions. Pay only for the minutes you are live.{live && <> From {moneyUsd(live.usdPerHour)} per hour.</>}</p>
              <span className="spacer" />
              <Link to="/app/live" className="btn btn-secondary">Set up live</Link>
            </article>
            <article className="card product-card">
              <div className="icon"><IconText /></div>
              <h3>Transcription &amp; subtitles <span className="badge badge-free">Free daily allowance</span></h3>
              <p>Standard on-prem model or premium accuracy. Get TXT, SRT and VTT.{tx && <> Free: {Math.round(tx.transcribeSecondsPerDay / 60)} minutes a day.</>}</p>
              <span className="spacer" />
              <Link to="/transcribe" className="btn btn-secondary">Transcribe</Link>
            </article>
          </div>
        </div>
      </section>

      <section className="section tight" id="desktop">
        <div className="container">
          <div className="card promo">
            <div className="icon" aria-hidden="true"><IconDownload /></div>
            <div className="promo-text">
              <h2 className="mb0">Prefer an app? Get the free desktop version.</h2>
              <p className="muted mb0">Audio Cleaner for Windows and macOS runs the Studio clean chain on your own computer: drop in a batch of audio or video files, nothing is uploaded, and it works offline.</p>
            </div>
            <Link to="/desktop" className="btn btn-secondary">About the desktop app</Link>
          </div>
        </div>
      </section>

      <section className="section alt" id="how">
        <div className="container">
          <h2 className="section-title">How it works</h2>
          <p className="section-lead">Three steps, whichever tool you pick.</p>
          <ol className="steps">
            <li><h3>Choose a tool</h3><p>Free in-browser denoise for quick jobs. Studio for long files, loudness and subtitles.</p></li>
            <li><h3>Drop in a file</h3><p>Audio or video. For video, Studio uploads only the audio track, so uploads stay small.</p></li>
            <li><h3>Compare and download</h3><p>Switch between original and cleaned at the same spot, then save the audio, or your video with the new track.</p></li>
          </ol>
        </div>
      </section>

      <section className="section" id="pricing">
        <div className="container narrow stack">
          <h2 className="section-title mb0">Simple pay-as-you-go pricing</h2>
          <p className="section-lead">The browser tool is free. Everything else is prepaid credit: you pay per minute of audio, nothing else.</p>
          <PriceTable />
          <div className="card stack">
            <h3 className="mb0">How credit works</h3>
            <ul className="mb0">
              <li><b>No subscription.</b> Top up from {pricing ? moneyCents(pricing.topup.minCents) : '$10'} and spend it as you go.</li>
              <li><b>Only what you use.</b> Studio jobs show an estimate before you upload. Live sessions are billed by the second of audio processed.</li>
              <li><b>Optional auto-reload.</b> Top up automatically when your balance runs low, with a monthly cap you choose. Turn it off any time.</li>
              <li><b>Unused credit.</b> Refundable within 14 days of a top-up; see the <Link to="/terms">Terms</Link>.</li>
            </ul>
            <div className="row">
              <Link to="/app/billing" className="btn btn-primary">Add credit</Link>
              <Link to="/login" className="btn btn-ghost">Create a free account</Link>
            </div>
          </div>
        </div>
      </section>

      <section className="section alt" id="privacy">
        <div className="container narrow">
          <div className="row">
            <IconLock width={28} height={28} />
            <h2 className="mb0">Our privacy promise</h2>
          </div>
          <ul className="prose">
            <li><b>The free browser tool never uploads your audio.</b> It makes no server calls except for the page, the model and this price list.</li>
            <li><b>Uploaded files are deleted after processing.</b> Results stay for 24 hours so you can download them, then they are deleted too.</li>
            <li><b>Live audio is never stored.</b> It is processed in memory and discarded.</li>
            <li><b>No training on your data</b>, no ad trackers, no third-party analytics, no external fonts or CDNs.</li>
            <li>Premium transcription is the one place audio leaves our servers: it is sent to OpenAI as a sub-processor, and only if you choose it.</li>
          </ul>
          <p><Link to="/privacy">Read the Privacy Policy</Link></p>
        </div>
      </section>

      <section className="section" id="faq">
        <div className="container narrow faq">
          <h2 className="section-title">Questions</h2>
          <div>
            {FAQ.map(([q, a]) => (
              <details key={q}><summary>{q}</summary><div><p>{a}</p></div></details>
            ))}
          </div>
          <div className="center">
            <Link to="/free" className="btn btn-primary btn-lg">Clean a file for free</Link>
          </div>
        </div>
      </section>
    </>
  );
}
