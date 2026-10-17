/** The "Recommended for this file / for your room" box: one line with the choices, the reasons under it, and a button to go back to them. */
export default function Recommended({ title, line, reasons = [], listening = false, listeningText, onApply, applyLabel = 'Use the recommended choices', testId = 'reco' }) {
  if (!line && !listening) return null;
  return (
    <section className="reco" data-testid={testId} aria-label={title} aria-live="polite">
      {line ? (
        <>
          <p className="reco-line mb0"><b>{title}:</b> <span data-testid={`${testId}-line`}>{line}</span></p>
          {reasons.length > 0 && (
            <ul className="reco-why small muted" data-testid={`${testId}-why`}>
              {reasons.map((r) => <li key={r}>{r}</li>)}
            </ul>
          )}
          {onApply && <button type="button" className="btn btn-ghost btn-sm" onClick={onApply} data-testid={`${testId}-apply`}>{applyLabel}</button>}
        </>
      ) : (
        <p className="small muted mb0" role="status" data-testid={`${testId}-listening`}><span className="spinner" aria-hidden="true" /> {listeningText}</p>
      )}
    </section>
  );
}
