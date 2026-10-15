import { STRENGTHS } from '../strengths.js';

/** Gentle / Balanced / Strong / Maximum as a row of buttons, with a line about the chosen one. */
export default function StrengthPicker({ name, value, onChange, disabled, label = 'How much noise to remove' }) {
  const current = STRENGTHS.find((s) => s.id === value) || STRENGTHS[STRENGTHS.length - 1];
  return (
    <fieldset className="field" disabled={disabled} data-testid={`strength-${name}`}>
      <legend className="label">{label}</legend>
      <div className="seg" role="radiogroup" aria-label={label}>
        {STRENGTHS.map((s) => (
          <label key={s.id}>
            <input type="radio" name={name} value={s.id} checked={value === s.id} onChange={() => onChange(s.id)} />
            <span>{s.label}</span>
          </label>
        ))}
      </div>
      <span className="hint">{current.hint}</span>
    </fieldset>
  );
}
