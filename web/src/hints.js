// When to use Rebuild the voice, in one line, with the price per minute like the other options.
// The same words on the Cleanup form and in "Change and save again" (components/JobForm.jsx, components/Result.jsx).
import { rate } from './format.js';

/** `r`: the price-list row of Voice rebuild. */
export const rebuildHint = (r) => `For an echoey room, a muffled voice, phone-quality or damaged sound, or distortion from recording too loud: the whole voice is rebuilt, which also brings back the high sounds that are missing. Slower, ${rate(r.usdPerMinute)} a minute, and on very echoey recordings it can change a word, so listen before you publish.`;
