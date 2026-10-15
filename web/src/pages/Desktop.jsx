import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { IconCheck, IconDownload, IconLock, IconSparkle, IconText, IconUpload, IconWave } from '../components/Icons.jsx';
import { usePageTitle } from '../hooks.js';
import { detectMacChip, detectPlatform } from '../platform.js';
import { SITE } from '../site.js';

const DESKTOP = SITE.desktop;

// ---- release state -----------------------------------------------------------------------------------------------
// public/releases/desktop.json is the single source of truth for "is there a release?". It is also what the desktop
// app's own daily update check reads. Anything unexpected (offline, HTTP error, not JSON, "available" not true) means
// "Coming soon": the page never shows a download link it is not sure about.
const RELEASE_URL = `${import.meta.env.BASE_URL}releases/desktop.json`;
const COMING_SOON = { available: false, version: '', notes: '' };

const cleanText = (v, max) => (typeof v === 'string' ? v.trim().slice(0, max) : '');

function parseRelease(d) {
  if (!d || typeof d !== 'object' || Array.isArray(d)) return COMING_SOON;
  const version = cleanText(d.version, 32);
  return {
    available: d.available === true,
    version: /^[0-9A-Za-z.+_-]+$/.test(version) ? version : '',
    notes: cleanText(d.notes, 300),
  };
}

function useDesktopRelease() {
  const [state, setState] = useState({ loading: true, release: COMING_SOON });
  useEffect(() => {
    const ctl = new AbortController();
    (async () => {
      try {
        const res = await fetch(RELEASE_URL, { cache: 'no-cache', signal: ctl.signal, headers: { Accept: 'application/json' } });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        setState({ loading: false, release: parseRelease(await res.json()) });
      } catch (e) {
        if (e?.name !== 'AbortError') setState({ loading: false, release: COMING_SOON });
      }
    })();
    return () => ctl.abort();
  }, []);
  return state;
}

// ---- content -----------------------------------------------------------------------------------------------------
const FEATURES = [
  { icon: IconSparkle, title: 'The full Studio clean', text: 'Noise removal plus the same voice polish as the website: even levels, gentle compression and a loudness target of your choice.' },
  { icon: IconUpload, title: 'Drop in a whole batch', text: 'Drag in as many recordings as you like and let them run one after another. Audio and video both work.' },
  { icon: IconWave, title: 'Four strengths', text: 'Choose how hard the noise gets pushed down, from gentle (keeps some room tone) up to maximum.' },
  { icon: IconCheck, title: 'Your video stays as it was', text: 'The picture is copied over untouched and your metadata is kept. Only the sound is swapped.' },
  { icon: IconSparkle, title: 'Listen before you save', text: 'Play any result and flip between Original and Cleaned at the same spot.' },
  { icon: IconLock, title: 'Free, offline, all included', text: 'No account, no credit, and nothing more to download once it\'s installed. It works with no internet at all, and your files never leave your computer.' },
];

const DOWNLOADS = [
  { key: 'windows', title: 'Windows', button: 'Download for Windows', detail: 'Installer (.exe) for 64-bit Windows 10 and 11.' },
  { key: 'macArm', title: 'macOS, Apple silicon', button: 'Download for Mac (Apple silicon)', detail: 'Disk image (.dmg) for Macs with an M1 chip or newer.' },
  { key: 'macIntel', title: 'macOS, Intel', button: 'Download for Mac (Intel)', detail: 'Disk image (.dmg) for Macs with an Intel processor.' },
];

const HERO_NOTE = {
  windows: 'Free for Windows 10 and 11 (64-bit).',
  mac: 'Free for macOS 11 or newer, Apple silicon and Intel.',
  mobile: 'The desktop app is for Windows and macOS computers. On this device, use the free browser tool.',
  other: 'Free for Windows 10 or 11 and macOS 11 or newer.',
};

/** Decorative sketch of the app window: drop area, queue with progress, Original / Cleaned switch. */
function AppArt() {
  return (
    <svg className="desk-art" viewBox="0 0 520 292" role="img" aria-label="Sketch of the desktop app: a queue of files being cleaned and an Original or Cleaned switch">
      <rect className="desk-win" x="8" y="8" width="504" height="276" rx="12" />
      <line className="desk-rule" x1="8" y1="40" x2="512" y2="40" />
      <circle className="desk-dot" cx="28" cy="24" r="5" />
      <circle className="desk-dot" cx="46" cy="24" r="5" />
      <circle className="desk-dot" cx="64" cy="24" r="5" />
      <rect className="desk-drop" x="28" y="58" width="464" height="56" rx="10" />
      <text className="desk-label" x="260" y="92" textAnchor="middle">Drop files here</text>
      {[['interview.mp4', 1], ['podcast-ep12.wav', 0.64], ['lecture.m4a', 0.28]].map(([name, done], i) => (
        <g key={name} transform={`translate(28 ${134 + i * 38})`}>
          <rect className="desk-row" x="0" y="0" width="464" height="30" rx="8" />
          <text className="desk-text" x="12" y="19">{name}</text>
          <rect className="desk-track" x="290" y="11" width="160" height="8" rx="4" />
          <rect className="desk-fill" x="290" y="11" width={160 * done} height="8" rx="4" />
        </g>
      ))}
      <g transform="translate(28 252)">
        <rect className="desk-seg" x="0" y="-12" width="180" height="26" rx="8" />
        <rect className="desk-seg-on" x="92" y="-9" width="85" height="20" rx="6" />
        <text className="desk-text" x="16" y="5">Original</text>
        <text className="desk-text desk-text-on" x="108" y="5">Cleaned</text>
      </g>
    </svg>
  );
}

function ComingSoon() {
  return (
    <div className="card stack" data-testid="desktop-coming-soon">
      <div className="row">
        <h3 className="mb0">Coming soon</h3>
        <span className="badge badge-soon">Not released yet</span>
      </div>
      <p className="mb0">The desktop app isn&rsquo;t ready to download yet. Until it is, the same kind of noise removal runs in your browser: free, no sign-up, and your file still never leaves your device.</p>
      <div className="row">
        <Link to="/free" className="btn btn-primary">Clean a file in your browser</Link>
      </div>
    </div>
  );
}

function FirstLaunchNote() {
  return (
    <div className="banner banner-info stack" role="note" data-testid="desktop-first-launch">
      <b>First launch: your system may show a warning</b>
      <p className="mb0">These builds are not code-signed yet, so Windows and macOS do not recognise the publisher. The warning is expected. You only need to get past it once.</p>
      <ul className="mb0">
        <li><b>Windows:</b> if you see &ldquo;Windows protected your PC&rdquo;, click <b>More info</b>, then <b>Run anyway</b>.</li>
        <li><b>macOS:</b> right-click (or Control-click) the app in your Applications folder, choose <b>Open</b>, then confirm with <b>Open</b>. On macOS 15 and later, if there is no Open button, go to System Settings &rarr; Privacy &amp; Security and click <b>Open Anyway</b>.</li>
      </ul>
    </div>
  );
}

export default function Desktop() {
  usePageTitle('Desktop app');
  const { loading, release } = useDesktopRelease();
  const platform = useMemo(() => detectPlatform(), []);
  const [chip, setChip] = useState(null); // 'arm' | 'intel' | null (unknown)
  useEffect(() => {
    if (platform !== 'mac') return undefined;
    let live = true;
    detectMacChip().then((c) => { if (live) setChip(c); });
    return () => { live = false; };
  }, [platform]);

  const recommended = {
    windows: platform === 'windows',
    macArm: platform === 'mac' && chip !== 'intel',
    macIntel: platform === 'mac' && chip !== 'arm',
  };
  // One obvious primary button when we know exactly which file the visitor needs; otherwise it jumps to the list.
  const exactKey = platform === 'windows' ? 'windows' : platform === 'mac' && chip === 'arm' ? 'macArm' : platform === 'mac' && chip === 'intel' ? 'macIntel' : null;
  const exact = exactKey ? DOWNLOADS.find((d) => d.key === exactKey) : null;
  const available = release.available;
  const downloads = DESKTOP.downloads;

  return (
    <>
      <section className="hero">
        <div className="container hero-grid">
          <div>
            <p className="eyebrow">Audio Cleaner for desktop</p>
            <h1>Studio clean for all your files, right on your computer.</h1>
            <p className="lead">A free app for Windows and macOS that takes the noise out of audio and video files and levels them for sharing. Drop in a batch and walk away. It works offline, it has everything, and nothing is uploaded.</p>
            {loading && <p className="muted">Checking the latest release&hellip;</p>}
            {!loading && available && (
              <div className="row">
                {exact
                  ? <a href={downloads[exact.key]} className="btn btn-primary btn-lg" rel="noopener noreferrer"><IconDownload width={20} height={20} /> {exact.button}</a>
                  : <a href="#download" className="btn btn-primary btn-lg"><IconDownload width={20} height={20} /> {platform === 'mac' ? 'Download for Mac' : 'See downloads'}</a>}
                <Link to="/free" className="btn btn-secondary btn-lg">Try it in the browser</Link>
              </div>
            )}
            {!loading && !available && (
              <div className="row">
                <span className="badge badge-soon">Coming soon</span>
                <Link to="/free" className="btn btn-secondary btn-lg">Try it in the browser</Link>
              </div>
            )}
            <p className="muted small">
              {HERO_NOTE[platform]}
              {!loading && available && release.version && <> Version {release.version}.</>}
            </p>
          </div>
          <AppArt />
        </div>
      </section>

      <section className="section" id="download" aria-labelledby="download-title">
        <div className="container">
          <h2 className="section-title" id="download-title">Download</h2>
          <p className="section-lead">Free to download and use. Pick the file for your computer. Every version is listed, so you can always choose another.</p>
          {loading && <div className="card center muted" role="status" aria-busy="true">Checking the latest release&hellip;</div>}
          {!loading && !available && <ComingSoon />}
          {!loading && available && (
            <div className="stack-lg">
              <div className="grid grid-3" data-testid="desktop-downloads">
                {DOWNLOADS.map((d) => (
                  <article key={d.key} className="card dl-card" data-recommended={recommended[d.key]} data-testid={`dl-${d.key}`}>
                    <div className="row-between">
                      <h3 className="mb0">{d.title}</h3>
                      {recommended[d.key] && <span className="badge badge-run">{exactKey ? 'Recommended' : 'Recommended for Mac'}</span>}
                    </div>
                    <p className="muted mb0">{d.detail}</p>
                    <span className="spacer" />
                    <a href={downloads[d.key]} className={`btn ${recommended[d.key] ? 'btn-primary' : 'btn-secondary'}`} rel="noopener noreferrer">
                      <IconDownload width={18} height={18} /> {d.button}
                    </a>
                  </article>
                ))}
              </div>
              <p className="muted small mb0">
                {release.version && <>Version {release.version}. </>}
                {release.notes && <>{release.notes} </>}
                <a href={DESKTOP.releaseNotesUrl} rel="noopener noreferrer">Release notes</a>.
                {' '}Not sure which Mac you have? Open the Apple menu and choose About This Mac: a chip that starts with &ldquo;Apple M&rdquo; means Apple silicon, &ldquo;Intel&rdquo; means Intel.
              </p>
              {!DESKTOP.signed && <FirstLaunchNote />}
            </div>
          )}
        </div>
      </section>

      <section className="section alt" id="features" aria-labelledby="features-title">
        <div className="container">
          <h2 className="section-title" id="features-title">What you get</h2>
          <p className="section-lead">The same Studio clean you know from the website, with no uploads and no credit.</p>
          <div className="grid grid-3">
            {FEATURES.map(({ icon: Icon, title, text }) => (
              <article key={title} className="card product-card">
                <div className="icon"><Icon /></div>
                <h3>{title}</h3>
                <p>{text}</p>
              </article>
            ))}
          </div>
          <p className="center muted small mt0 desk-scope">
            Looking for transcription or live noise removal for streams? Those stay on the website: <Link to="/transcribe">Free transcription</Link> and <Link to="/app/live">Live for streamers</Link>.
          </p>
        </div>
      </section>

      <section className="section" id="requirements" aria-labelledby="requirements-title">
        <div className="container narrow">
          <h2 className="section-title" id="requirements-title">System requirements</h2>
          <p className="section-lead">Make sure your computer is on the list, then pick the matching download above.</p>
          <div className="grid grid-2">
            <div className="card">
              <h3>Windows</h3>
              <ul className="mb0">
                <li>Windows 10 or Windows 11</li>
                <li>64-bit processor (x64)</li>
              </ul>
            </div>
            <div className="card">
              <h3>macOS</h3>
              <ul className="mb0">
                <li>macOS 11 (Big Sur) or newer</li>
                <li>Apple silicon or Intel Mac</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="section alt" id="privacy" aria-labelledby="privacy-title">
        <div className="container narrow">
          <div className="row">
            <IconLock width={28} height={28} />
            <h2 className="mb0" id="privacy-title">Yours alone</h2>
          </div>
          <ul className="prose">
            <li><b>Your files stay on your computer.</b> Everything happens right there and nothing is uploaded.</li>
            <li><b>Works offline.</b> The app never needs the internet, and it doesn&rsquo;t phone home after you install it.</li>
            <li><b>No analytics and no telemetry.</b></li>
          </ul>
          <div className="row">
            <IconText width={20} height={20} />
            <span>The full story is in the <Link to="/privacy">Privacy Policy</Link>.</span>
          </div>
        </div>
      </section>
    </>
  );
}
