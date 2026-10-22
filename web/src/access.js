// What needs an account, in one place. The rule is the price list's own: anything that is priced is done on our side, so it needs an
// account (to hold the money and keep the files); anything that runs in the visitor's browser is free and needs nothing. No feature
// is named here. A feature that runs in the browser has no price, so it works signed out by itself.
//   the editor: needsServer(tracks) (editor/tracks.js) is true when any part of the plan is priced
//   live: liveNeedsServer(choice) is true when any chosen part is something the price list has a row for

import { needsServer } from './editor/tracks.js';

/** Does starting need an account the visitor does not have? `server`: something priced is chosen (needsServer / liveNeedsServer). */
export const needsAccount = (server, signedIn) => !!server && !signedIn;

/** The editor's plan: a priced part is chosen and nobody is signed in. */
export const editorNeedsAccount = (tracks, signedIn) => needsAccount(needsServer(tracks), signedIn);

/**
 * The price list rows a live choice is made of: [[product, tier], ...]. A choice done in the browser (`where: 'browser'`) is made of
 * nothing the price list has; done on our side it is the noise remover, the captions of the chosen quality, and a row for the
 * translations. `captions`: '' | 'local' | 'premium'. `translations`: how many extra languages.
 */
export function liveProducts({ where, denoise = false, captions = '', translations = 0 }) {
  if (where !== 'server') return [];
  const rows = [];
  if (denoise) rows.push(['stream_denoise', 'light']);
  if (captions) rows.push(['stream_captions', captions]);
  if (captions && translations > 0) rows.push(['stream_translate', 'standard']);
  return rows;
}

/** Is anything of this live choice priced (so done on our side)? */
export const liveNeedsServer = (choice) => liveProducts(choice).length > 0;

/**
 * Dollars a minute of the minutes-based rows of a live choice, at the price list (`rateOf(product, tier)` gives a row or null). null: a row
 * is missing. Translations are counted by the speech translated, so they are not part of this sum.
 */
export function livePerMinute(choice, rateOf) {
  let sum = 0;
  for (const [product, tier] of liveProducts(choice)) {
    if (product === 'stream_translate') continue;
    const r = rateOf(product, tier);
    if (!r) return null;
    sum += r.usdPerMinute;
  }
  return sum;
}
