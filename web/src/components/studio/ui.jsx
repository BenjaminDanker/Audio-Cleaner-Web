// Small pieces of the editor's markup that several parts use.

/** A tooltip (data-tip) as attributes: none when there is no text. */
export const tipAttr = (t) => (t ? { 'data-tip': t } : {});

/** The prototype's segmented buttons. items: [value, label, disabled?, tooltip?]. `cur`: the pressed value (null: none). Clicks are handled by the screen (data-act / data-v). */
export function Seg({ items, cur, act, label, cls = '', dis = false }) {
  return (
    <div className={`bseg tight${cls ? ` ${cls}` : ''}`} role="group" aria-label={label}>
      {items.map(([v, text, disabled, tp]) => (
        <button
          key={String(v)} type="button" aria-pressed={cur !== null && cur === v} data-act={act} data-v={v}
          aria-disabled={disabled || dis ? 'true' : undefined} {...tipAttr(tp)}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

/** "Mixed": the selected parts differ. */
export const Mixed = ({ show }) => (show ? <em className="mx">Mixed</em> : null);
