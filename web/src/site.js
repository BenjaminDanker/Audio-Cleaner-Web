// Single place for the operator's identity. The site owner fills these in before launch; nothing else in the
// code base should hard-code a contact address, legal entity or jurisdiction.
//
//   name          shown in the header, footer, page titles and legal texts
//   supportEmail  public contact address (footer, Terms, Privacy). Empty = no contact link is shown.
//   legalName     legal entity that operates the service ("Example Ltd")
//   legalAddress  postal address of that entity
//   jurisdiction  governing law / venue for the Terms ("England and Wales", "the State of Delaware, USA")
//   desktop       the free Windows / macOS app (page: /desktop), see below
//
// While legalName, legalAddress or jurisdiction is empty the legal pages show a "Draft - under legal review"
// banner and use neutral wording ("the operator") instead of the missing detail.
export const SITE = {
  name: 'Audio Cleaner',
  supportEmail: '',
  legalName: '',
  legalAddress: '',
  jurisdiction: '',
  // Desktop app. Whether a release exists (and its version and notes) is NOT decided here: the /desktop page reads
  // public/releases/desktop.json (served at <base>releases/desktop.json, the same file the app's own update check
  // reads) and shows "Coming soon" instead of the download buttons until that file says "available": true. Flip it
  // when the first desktop-v* release is published. Unsigned builds trigger SmartScreen / Gatekeeper warnings on first
  // launch (see docs/desktop.md), so keep `signed` false until the installers are code-signed and notarised; while it
  // is false the page shows a short "first launch" note for Windows and macOS.
  desktop: {
    signed: false,
    downloads: {
      windows: 'https://github.com/BenjaminDanker/Audio-Cleaner-Web/releases/latest/download/AudioCleaner-Setup-x64.exe',
      macArm: 'https://github.com/BenjaminDanker/Audio-Cleaner-Web/releases/latest/download/AudioCleaner-macOS-arm64.dmg',
      macIntel: 'https://github.com/BenjaminDanker/Audio-Cleaner-Web/releases/latest/download/AudioCleaner-macOS-x86_64.dmg',
    },
    releaseNotesUrl: 'https://github.com/BenjaminDanker/Audio-Cleaner-Web/releases/latest',
  },
};

/** True when every field the legal pages depend on has been filled in. */
export const legalComplete = () => !!(SITE.legalName.trim() && SITE.legalAddress.trim() && SITE.jurisdiction.trim());

/** "Example Ltd, 1 Main St" / "Example Ltd" / null when nothing is set. */
export function operatorLine() {
  const name = SITE.legalName.trim();
  const addr = SITE.legalAddress.trim();
  if (name && addr) return `${name}, ${addr}`;
  return name || addr || null;
}
