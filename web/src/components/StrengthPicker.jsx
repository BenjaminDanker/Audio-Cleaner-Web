import { STRENGTHS } from '../strengths.js';

/** Gentle / Balanced / Strong / Maximum as a row of buttons, with a line about the chosen one. */
export default function StrengthPicker({ name, value, onChange, disabled, label = 'How much noise to remove', customText = '' }) {
  // `value` that is none of the four (a strength in between, from the slider): none is ticked and `customText` says which it is
  const named = STRENGTHS.find((s) => s.id === value);
  const current = named || (customText ? null : STRENGTHS[STRENGTHS.length - 1]);
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
      <span className="hint" data-testid={`${name}-hint`}>{current ? current.hint : customText}</span>
    </fieldset>
  );
}
