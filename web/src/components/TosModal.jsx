import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api.js';

const VERSION = '2026-10-10';

/** Modal asking the user to accept the current Terms of Service. */
export default function TosModal({ onClose }) {
  const ref = useRef(null);
  const [checked, setChecked] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const d = ref.current;
    if (d && !d.open) d.showModal();
    return () => {
      if (d?.open) d.close();
    };
  }, []);

  async function accept() {
    setBusy(true);
    setError('');
    try {
      await api('account/tos', { method: 'POST', body: { version: VERSION } });
      onClose(true);
    } catch (e) {
      setError(e.detail || 'That didn’t save. Please try again.');
      setBusy(false);
    }
  }

  return (
    <dialog ref={ref} className="modal" aria-labelledby="tos-title" onCancel={(e) => { e.preventDefault(); onClose(false); }}>
      <form method="dialog" onSubmit={(e) => { e.preventDefault(); if (checked) accept(); }}>
        <h2 id="tos-title">Terms of Service</h2>
        <p>
          Before you go on, please have a look at our <Link to="/terms" target="_blank" rel="noopener">Terms of Service</Link> and{' '}
          <Link to="/privacy" target="_blank" rel="noopener">Privacy Policy</Link> (version {VERSION}). We try to keep them readable.
        </p>
        <label className="check">
          <input type="checkbox" checked={checked} onChange={(e) => setChecked(e.target.checked)} />
          <span>I’ve read them and I agree.</span>
        </label>
        {error && <p className="banner banner-error" role="alert">{error}</p>}
        <div className="modal-actions">
          <button type="button" className="btn btn-ghost" onClick={() => onClose(false)} disabled={busy}>Not now</button>
          <button type="submit" className="btn btn-primary" disabled={!checked || busy}>{busy ? 'Saving…' : 'Agree and continue'}</button>
        </div>
      </form>
    </dialog>
  );
}
