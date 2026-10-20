// "Which one should I pick?" - Denoising or Voice rebuild, side by side in plain words. Small and collapsed: next to the two one-line
// hints in the upload form, in "Change and save again" and in the editor's part panel. The words are one list here so they never differ.

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

export default function WhichOne({ id = 'which-one', className = '' }) {
  return (
    <details className={`which-one ${className}`} data-testid={id}>
      <summary>Which one should I pick?</summary>
      <div className="which-cols">
        <section aria-label="Denoising">
          <h4>Denoising</h4>
          <ul>{DENOISING_POINTS.map((t) => <li key={t}>{t}</li>)}</ul>
        </section>
        <section aria-label="Voice rebuild">
          <h4>Voice rebuild</h4>
          <ul>{REBUILD_POINTS.map((t) => <li key={t}>{t}</li>)}</ul>
        </section>
      </div>
    </details>
  );
}
