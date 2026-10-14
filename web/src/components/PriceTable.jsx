import { BILLING_NOTE, moneyUsd, rate } from '../format.js';
import { usePricing } from '../hooks.js';

const GROUPS = [
  { title: 'In your browser', products: [] },
  { title: 'Files (server-side)', products: ['denoise', 'transcribe'] },
  { title: 'Live streaming', products: ['stream_denoise', 'stream_captions'] },
];

function minutes(sec) {
  return Math.round(sec / 60);
}

export default function PriceTable() {
  const { data, error, loading } = usePricing();
  if (loading) return <p role="status" className="muted center"><span className="spinner" aria-hidden="true" /> Loading prices…</p>;
  if (error || !data) {
    return <div className="banner banner-warn" role="alert">Prices could not be loaded right now ({error || 'unknown error'}). Please reload the page in a moment.</div>;
  }
  return (
    <div className="stack">
      <div className="table-wrap">
        <table>
          <caption className="sr-only">Prices per minute and per hour of audio</caption>
          <thead>
            <tr><th scope="col">Service</th><th scope="col" className="num">Per minute</th><th scope="col" className="num">Per hour</th><th scope="col">Status</th></tr>
          </thead>
          <tbody>
            <tr className="group"><th scope="rowgroup" colSpan={4}>{GROUPS[0].title}</th></tr>
            <tr>
              <th scope="row">Browser denoise (DeepFilterNet3)</th>
              <td className="num">Free</td><td className="num">Free</td>
              <td><span className="badge badge-free">Free forever</span></td>
            </tr>
            {GROUPS.slice(1).map((g) => (
              <GroupRows key={g.title} group={g} rates={data.rates} free={data.free} />
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted mb0" data-testid="billing-note">{BILLING_NOTE}</p>
    </div>
  );
}

function GroupRows({ group, rates, free }) {
  const rows = rates.filter((r) => group.products.includes(r.product));
  return (
    <>
      <tr className="group"><th scope="rowgroup" colSpan={4}>{group.title}</th></tr>
      {rows.map((r) => (
        <tr key={`${r.product}/${r.tier}`} className={r.available ? undefined : 'soon'}>
          <th scope="row">
            {r.label}
            {r.product === 'transcribe' && r.tier === 'local' && free && (
              <div className="small muted">First {minutes(free.transcribeSecondsPerDay)} min per day free (up to {minutes(free.transcribeMaxFileSeconds)} min per file)</div>
            )}
          </th>
          <td className="num">{rate(r.usdPerMinute)}</td>
          <td className="num">{moneyUsd(r.usdPerHour)}</td>
          <td>{r.available ? <span className="badge badge-ok">Available</span> : <span className="badge badge-soon">Coming soon</span>}</td>
        </tr>
      ))}
    </>
  );
}
