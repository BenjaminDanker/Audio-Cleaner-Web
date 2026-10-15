// How much noise to remove: the same four levels everywhere (website, desktop app, OBS plugin).
// db is DeepFilterNet's attenuation limit; the effect is far from linear, so the steps are closer together
// at the gentle end, where they are easiest to hear.
export const STRENGTHS = [
  { id: 'gentle', label: 'Gentle', db: 12, hint: 'Takes the edge off. Keeps some room sound and music, so it sounds the most natural.' },
  { id: 'balanced', label: 'Balanced', db: 24, hint: 'Most of the noise goes; the room still sounds like a room.' },
  { id: 'strong', label: 'Strong', db: 40, hint: 'Nearly all of the noise goes.' },
  { id: 'max', label: 'Maximum', db: 100, hint: 'Removes as much as it can. Best for noisy places, and good for most voices.' },
];

export const strengthDb = (id) => (STRENGTHS.find((s) => s.id === id) || STRENGTHS[STRENGTHS.length - 1]).db;
