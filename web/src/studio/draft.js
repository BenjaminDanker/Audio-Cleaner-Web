// Keeping a person's work across the sign-in. A plan that needs an account (anything priced: editor/tracks.js needsServer) cannot start
// signed out, and signing in can leave the site (Google) or open in another tab (the emailed link), so the page is gone by the time
// they are back. Before they go, the chosen file and the editor's state (tracks, selection, suggestions, the output choice) are kept in
// this browser's IndexedDB, only here: nothing is sent anywhere. The Cleanup page puts them back when it opens, and the copy is deleted
// as it is put back, or after a day if nobody comes back. Without a draft nothing here touches IndexedDB at all (a stamp in
// localStorage says whether there is one).

const DB = 'acw-draft';
const STORE = 'draft';
const KEY = 'current';
const STAMP = 'acw.draft.saved'; // when (ms) the copy was kept, in localStorage: lets a page load tell whether there is anything to look for
export const DRAFT_MAX_AGE_MS = 24 * 3600 * 1000;

const stamp = () => { try { const t = Number(localStorage.getItem(STAMP)); return Number.isFinite(t) && t > 0 ? t : 0; } catch { return -1; } }; // -1: no way to tell
const setStamp = (t) => { try { if (t) localStorage.setItem(STAMP, String(t)); else localStorage.removeItem(STAMP); } catch { /* the page then looks in IndexedDB every time */ } };

const openDb = () => new Promise((resolve, reject) => {
  const r = indexedDB.open(DB, 1);
  r.onupgradeneeded = () => r.result.createObjectStore(STORE);
  r.onsuccess = () => resolve(r.result);
  r.onerror = () => reject(r.error || new Error('IndexedDB did not open'));
  r.onblocked = () => reject(new Error('IndexedDB is blocked'));
});
const finished = (tx) => new Promise((resolve, reject) => {
  tx.oncomplete = () => resolve();
  tx.onerror = () => reject(tx.error || new Error('IndexedDB failed'));
  tx.onabort = () => reject(tx.error || new Error('IndexedDB stopped'));
});

const json = (v) => JSON.parse(JSON.stringify(v));
const cloneable = (v) => { try { return structuredClone(v); } catch { return null; } };

/** What is kept for a session: the file itself and everything the editor holds that a sign-in must not lose. */
export function recordOf(session, choice, now = Date.now()) {
  const S = session.S;
  return {
    savedAt: now,
    blob: session.file,
    name: session.file.name,
    type: session.file.type,
    lastModified: session.file.lastModified,
    seconds: session.T,
    lengthUnknown: !!session.lengthUnknown,
    tracks: json(S.tracks),
    sel: S.sel ? json(S.sel) : null,
    sugg: json(S.sugg || []),
    moments: session.sg?.phase || 'idle',
    out: json(session.out),
    choice: choice ? { format: choice.format, name: choice.name, size: choice.size } : null,
    auto: !!session.auto,
    applied: !!session.applied,
    undoTo: session.undoTo || null,
    stats: cloneable(session.stats),
  };
}

/** Keep the session's file and state until the visitor is back. Rejects when this browser cannot keep it (no room, private window). */
export async function keepDraft(session, choice) {
  if (!session.file || !session.S || typeof indexedDB === 'undefined') throw new Error('no place to keep it');
  const db = await openDb();
  try {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(recordOf(session, choice), KEY);
    await finished(tx);
  } finally {
    db.close();
  }
  setStamp(Date.now());
}

async function readAndDelete() {
  if (stamp() === 0 || typeof indexedDB === 'undefined') return null; // never kept one: do not even open the database
  let db;
  try { db = await openDb(); } catch { return null; }
  try {
    const tx = db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);
    const rec = await new Promise((resolve, reject) => {
      const g = store.get(KEY);
      g.onsuccess = () => { store.delete(KEY); resolve(g.result || null); }; // (deleted in the same step: it is put back once)
      g.onerror = () => reject(g.error);
    });
    await finished(tx);
    setStamp(0);
    return rec && rec.blob && Date.now() - rec.savedAt <= DRAFT_MAX_AGE_MS ? rec : null;
  } catch {
    return null;
  } finally {
    db.close();
  }
}

let taking = null;
/**
 * The kept work, once: reading it deletes it (so a second page does not put it back again, and nothing is kept longer than needed).
 * Calls made while one is under way share its answer. null: nothing was kept, it was kept more than a day ago, or it can't be read.
 */
export function takeDraft() {
  if (!taking) taking = readAndDelete().finally(() => { taking = null; });
  return taking;
}

/** Delete a kept copy that is older than a day (called when the site loads; costs a localStorage read when there is none). */
export function purgeOldDraft() {
  const t = stamp();
  if (t === 0 || t === -1 || Date.now() - t <= DRAFT_MAX_AGE_MS) return;
  try { indexedDB.deleteDatabase(DB); } catch { /* nothing to delete */ }
  setStamp(0);
}
