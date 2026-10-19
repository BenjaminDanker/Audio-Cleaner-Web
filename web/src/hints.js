// When to use Restore highs and when Rebuild the voice, in one line each, with the price per minute like the other options.
// The same words on the Cleanup form and in "Change and save again" (components/JobForm.jsx, components/Result.jsx).
import { rate } from './format.js';

/** `r`: the price-list row of Restore highs (or null while the prices load). */
export const highsHint = (r) => `For a recording that is only missing its high sounds (phone calls, voice messages, small files) and otherwise sounds fine: adds them back and keeps the voice exactly as recorded.${r ? ` ${rate(r.usdPerMinute)} a minute.` : ''}`;

/** `r`: the price-list row of Voice rebuild. */
export const rebuildHint = (r) => `For an echoey room, a muffled voice, phone-quality or damaged sound, or distortion from recording too loud: the whole voice is rebuilt, which also brings back the highs. Slower, ${rate(r.usdPerMinute)} a minute, and on very echoey recordings it can change a word, so listen before you publish.`;
