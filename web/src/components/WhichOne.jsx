// "Which one should I pick?" - Noise removal or Voice rebuild, side by side in plain words: the first section of the editor's "All the
// options" window (components/studio/InfoModal.jsx). The words are one list here so they never differ.

export const DENOISING_POINTS = [
  'Keeps your voice exactly as recorded and never changes words.',
  'Laughs, shouts, whispers and singing stay as they were.',
  'Works on any length of recording.',
  'Costs less.',
  'Removes most background sound, including game sounds. The gentler strengths keep some room sound.',
];

export const REBUILD_POINTS = [
  'Makes the voice again from scratch, so it also fixes echo, a muffled or phone-quality voice and distortion from recording too loud.',
  'Very clean, and can sound slightly too clean.',
  'Keeps only speech, so game sounds, music and room sound are all gone.',
  'Laughs, shouts and singing may come out oddly.',
  'On very echoey recordings it can change a word, so listen before you publish.',
  'Files up to 90 minutes. Costs more. Works in many languages.',
];
