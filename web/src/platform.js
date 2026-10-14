// Which kind of computer is the visitor on? Only used to highlight the matching download on /desktop; every download
// stays visible. Pure functions of `navigator` (injectable for tests).

/** 'windows' | 'mac' | 'mobile' | 'other' */
export function detectPlatform(nav = typeof navigator === 'undefined' ? null : navigator) {
  if (!nav) return 'other';
  const uad = nav.userAgentData;
  const platform = String(uad?.platform || nav.platform || '');
  const ua = String(nav.userAgent || '');
  // Phones and tablets first: iPadOS reports "MacIntel" (with a touch screen), Android reports "Linux".
  if (uad?.mobile || /Android|iPhone|iPad|iPod/i.test(ua) || (/^Mac/i.test(platform) && (nav.maxTouchPoints || 0) > 1)) return 'mobile';
  if (/^Win/i.test(platform) || /Windows/i.test(ua)) return 'windows';
  if (/^Mac/i.test(platform) || /Mac OS X|Macintosh/i.test(ua)) return 'mac';
  return 'other';
}

/**
 * Apple silicon vs Intel. Browsers deliberately hide this (Safari and Firefox always claim Intel), but Chromium can
 * tell us through User-Agent Client Hints. Resolves 'arm', 'intel' or null (unknown).
 */
export async function detectMacChip(nav = typeof navigator === 'undefined' ? null : navigator) {
  try {
    const v = await nav?.userAgentData?.getHighEntropyValues?.(['architecture']);
    if (v?.architecture === 'arm') return 'arm';
    if (v?.architecture === 'x86') return 'intel';
  } catch {
    /* not available or blocked */
  }
  return null;
}
