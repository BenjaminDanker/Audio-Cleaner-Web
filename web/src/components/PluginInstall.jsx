import { useEffect, useRef, useState } from 'react';
import { API } from '../api.js';
import { bytes } from '../format.js';
import { IconDownload } from './Icons.jsx';

// The plugin files are mirrored on our own server (docs/operations.md, "Release downloads"): the buttons use the
// "latest" addresses, the manifest only tells which version that is.
export const PLUGIN_DOWNLOADS = `${API}downloads/obs-plugin`;

const Code = ({ children }) => <code className="mono install-code">{children}</code>;
const ADD_FILTER = (
  <li>Start OBS. In the <b>Audio Mixer</b>, click the three dots (or right-click) next to your microphone, choose <b>Filters</b>, click <b>+</b> and pick <b>Audio Cleaner</b>. Paste your API key into it.</li>
);

/**
 * One entry per kind of computer. `key` is the plugin manifest's name for the file; `extra` is a second, optional file.
 * The steps assume nothing: where to click, what to type, how to check it worked.
 */
const PLATFORMS = [
  {
    id: 'windows', key: 'windows-x64', label: 'Windows', file: 'zip file',
    steps: (
      <ol className="ol-steps">
        <li>Close OBS.</li>
        <li>Open your <b>Downloads</b> folder and double-click the zip file you just downloaded. Inside it is one folder, <b>audio-cleaner</b>. Leave this window open.</li>
        <li>Open a second File Explorer window (<b>Windows key + E</b>). Click the address bar at the top, type <Code>%ProgramData%\obs-studio</Code> and press <b>Enter</b>.</li>
        <li>If there is no folder called <b>plugins</b> there, make one: right-click an empty spot, choose <b>New</b> → <b>Folder</b> and name it <Code>plugins</Code>.</li>
        <li>Open <b>plugins</b> and drag the <b>audio-cleaner</b> folder from the zip window into it. Windows may ask for permission to copy: click <b>Continue</b>.</li>
        {ADD_FILTER}
      </ol>
    ),
    check: <>If the filter isn&rsquo;t in the list, check that this file exists: <Code>C:\ProgramData\obs-studio\plugins\audio-cleaner\bin\64bit\audio-cleaner.dll</Code>. The plugin doesn&rsquo;t work from <b>Program Files\obs-studio\obs-plugins</b>.</>,
    extra: {
      key: 'windows-setup', label: 'Installer instead',
      text: <>Rather have it done for you? The installer puts the folder in the right place. It isn&rsquo;t signed by Microsoft, so your browser may ask whether to keep it, and Windows shows &ldquo;Windows protected your PC&rdquo;: click <b>More info</b>, then <b>Run anyway</b>.</>,
    },
  },
  {
    id: 'mac', key: 'macos', label: 'Mac', file: '.tar.xz file',
    steps: (
      <ol className="ol-steps">
        <li>Quit OBS (<b>OBS</b> menu → <b>Quit OBS</b>).</li>
        <li>Open your <b>Downloads</b> folder and double-click the file you just downloaded. A file called <b>audio-cleaner.plugin</b> appears next to it.</li>
        <li>In Finder, choose <b>Go</b> → <b>Go to Folder…</b> (<b>Shift + Command + G</b>), type <Code>~/Library/Application Support/obs-studio</Code> and press <b>Return</b>.</li>
        <li>If there is no folder called <b>plugins</b> there, make one: <b>File</b> → <b>New Folder</b>, name it <Code>plugins</Code>.</li>
        <li>Drag <b>audio-cleaner.plugin</b> into the <b>plugins</b> folder.</li>
        <li>The plugin isn&rsquo;t signed by Apple, so macOS won&rsquo;t load it until you allow it. Open <b>Terminal</b> (Applications → Utilities), paste this line and press <b>Return</b>:<br />
          <Code>xattr -dr com.apple.quarantine ~/Library/Application\ Support/obs-studio/plugins/audio-cleaner.plugin</Code></li>
        {ADD_FILTER}
      </ol>
    ),
    check: <>Works on Apple silicon and Intel Macs, with OBS 31 or newer.</>,
  },
  {
    id: 'ubuntu-debian', key: 'ubuntu-debian', label: 'Ubuntu, Debian, Mint', file: '.deb file',
    steps: (
      <ol className="ol-steps">
        <li>Close OBS.</li>
        <li>Double-click the downloaded <b>.deb</b> file and click <b>Install</b> (it asks for your password). Or in a terminal:<br />
          <Code>cd ~/Downloads &amp;&amp; sudo apt install ./audio-cleaner-obs-plugin-*-ubuntu-debian.deb</Code></li>
        {ADD_FILTER}
      </ol>
    ),
    check: <>This is for OBS 31 or newer installed with apt (for example from the OBS Project&rsquo;s PPA). If your OBS is the Flatpak one, use <b>Other Linux</b>.</>,
  },
  {
    id: 'linux-other', key: 'linux-other', label: 'Other Linux, Flatpak', file: '.tar.xz file',
    steps: (
      <ol className="ol-steps">
        <li>Close OBS.</li>
        <li>In a terminal, run:<br />
          <Code>mkdir -p ~/.config/obs-studio/plugins &amp;&amp; tar -xJf ~/Downloads/audio-cleaner-obs-plugin-*-linux-other.tar.xz -C ~/.config/obs-studio/plugins</Code><br />
          For the Flatpak OBS, use <Code>~/.var/app/com.obsproject.Studio/config/obs-studio/plugins</Code> in both places instead.</li>
        {ADD_FILTER}
      </ol>
    ),
    check: <>The file should end up at <Code>…/obs-studio/plugins/audio-cleaner/bin/64bit/audio-cleaner.so</Code>.</>,
  },
];

/** The kind of computer this browser runs on, so its instructions can be the open ones. */
function guessPlatform() {
  const p = `${navigator.userAgentData?.platform || ''} ${navigator.platform || ''} ${navigator.userAgent || ''}`;
  if (/Win/i.test(p)) return 'windows';
  if (/Mac/i.test(p) && !/iPhone|iPad/i.test(p)) return 'mac';
  if (/Ubuntu|Debian|Mint/i.test(p)) return 'ubuntu-debian';
  if (/Linux/i.test(p) && !/Android/i.test(p)) return 'linux-other';
  return null;
}

/** What the plugin download page says: 'loading', 'unavailable' or { version, files }. */
function usePluginManifest() {
  const [state, setState] = useState('loading');
  useEffect(() => {
    const ctl = new AbortController();
    (async () => {
      try {
        const r = await fetch(`${PLUGIN_DOWNLOADS}/manifest.json`, { signal: ctl.signal, cache: 'no-cache' });
        if (!r.ok) throw new Error(String(r.status));
        const m = await r.json();
        if (typeof m?.version !== 'string' || typeof m?.files !== 'object' || m.files === null) throw new Error('shape');
        setState({ version: m.version, files: m.files });
      } catch {
        if (!ctl.signal.aborted) setState('unavailable');
      }
    })();
    return () => ctl.abort();
  }, []);
  return state;
}

export default function PluginInstall() {
  const plugin = usePluginManifest();
  const ready = plugin !== 'loading' && plugin !== 'unavailable';
  const [open, setOpen] = useState(() => guessPlatform());
  const refs = useRef({});
  const size = (key) => (Number.isFinite(plugin.files?.[key]?.size) ? ` (${bytes(plugin.files[key].size)})` : '');

  // Clicking a download opens that computer's steps (and keeps the others out of the way).
  function downloaded(id) {
    setOpen(id);
    setTimeout(() => refs.current[id]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 50);
  }

  return (
    <div className="stack" data-testid="plugin-downloads" aria-labelledby="obs-dl-h" role="group">
      <h3 id="obs-dl-h" className="mb0">Install the Audio Cleaner OBS plugin</h3>
      {plugin === 'loading' && <p className="small muted mb0" role="status">Checking for the newest version&hellip;</p>}
      {plugin === 'unavailable' && <p className="small muted mb0" role="status" data-testid="plugin-unavailable">The plugin download isn&rsquo;t ready yet. Please check back in a little while.</p>}
      {ready && <p className="small muted mb0" data-testid="plugin-version">Newest version: <b>{plugin.version}</b>. Pick your computer, download, then follow the steps under it.</p>}
      {ready && (
        <div className="install-list">
          {PLATFORMS.filter((p) => plugin.files[p.key]).map((p) => (
            <details
              key={p.id} ref={(el) => { refs.current[p.id] = el; }} open={open === p.id} data-testid={`install-${p.id}`}
              onToggle={(e) => { if (e.currentTarget.open) setOpen(p.id); else if (open === p.id) setOpen(null); }}
            >
              <summary>
                <span className="install-name">{p.label}</span>
                <a
                  className="btn btn-secondary btn-sm" href={`${PLUGIN_DOWNLOADS}/latest/${p.id}`} data-testid={`plugin-${p.id}`}
                  onClick={(e) => { e.stopPropagation(); downloaded(p.id); }}
                >
                  <IconDownload width={16} height={16} /> Download{size(p.key)}
                </a>
              </summary>
              <div className="install-body">
                <p className="small muted mb0">After downloading the {p.file}:</p>
                {p.steps}
                {p.check && <p className="small muted mb0">{p.check}</p>}
                {p.extra && plugin.files[p.extra.key] && (
                  <p className="small mb0" data-testid={`plugin-extra-${p.id}`}>
                    {p.extra.text}{' '}
                    <a href={`${PLUGIN_DOWNLOADS}/latest/${p.extra.key}`} data-testid={`plugin-${p.extra.key}`}>Download the installer{size(p.extra.key)}</a>
                  </p>
                )}
              </div>
            </details>
          ))}
        </div>
      )}
      {ready && plugin.files.source && (
        <p className="small muted mb0" data-testid="plugin-source">
          The plugin is free software (GPL 2.0 or later). <a href={`${PLUGIN_DOWNLOADS}/latest/source`}>Its source code</a> is here too.
        </p>
      )}
    </div>
  );
}
