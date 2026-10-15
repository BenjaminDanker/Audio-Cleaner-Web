import { BILLING_NOTE, moneyUsd } from '../format.js';
import { usePricing } from '../hooks.js';

// What each service is, in one plain line. Keyed by "product/tier" (the ids the price list uses); the names and the
// prices themselves come from the server (GET /api/billing/pricing), so they can never disagree with the bill.
const DESCRIPTIONS = {
  'denoise/standard': 'Noise removal plus voice polish and loudness, ready to publish',
  'transcribe/local': 'Subtitles and transcripts (SRT, VTT, TXT)',
  'transcribe/premium': 'The same files, best for accents, noisy rooms and specialist words',
  'stream_denoise/light': 'Cleans up your mic while you talk',
  'stream_denoise/low_latency': 'For tight setups: about 20 ms less delay',
  'stream_captions/local': 'Live captions in English',
  'stream_captions/premium': 'Live captions in other languages too, best for accents and specialist words',
};

const GROUPS = [
  { id: 'free', title: 'Free', products: [] },
  { id: 'files', title: 'Files', products: ['denoise', 'transcribe'] },
  { id: 'live', title: 'Live streaming', products: ['stream_denoise', 'stream_captions'] },
];

const minutes = (sec) => Math.round(sec / 60);

/** "$0.60 / hour" is the headline. Per-minute only when it is a whole number of cents (no "$0.004"). */
function perMinuteNote(usdPerMinute) {
  const cents = usdPerMinute * 100;
  if (cents < 1 - 1e-9 || Math.abs(cents - Math.round(cents)) > 1e-9) return null;
  return `${moneyUsd(usdPerMinute)} / min`;
}

/**
 * The price list. `only` limits it to some groups, e.g. only={['live']} on the Live page; `testId` goes on the table.
 * Rows that are not available are left out. A real table (two columns) so it reads well with a screen reader, and it
 * wraps instead of scrolling sideways on a phone.
 */
export default function PriceTable({ only, testId = 'prices', showNote = true }) {
  const { data, error, loading } = usePricing();
  if (loading) return <p role="status" className="muted center"><span className="spinner" aria-hidden="true" /> Loading prices…</p>;
  if (error || !data) {
    return <div className="banner banner-warn" role="alert">We couldn&rsquo;t load the prices just now. Please reload the page in a moment.</div>;
  }
  const groups = GROUPS.filter((g) => !only || only.includes(g.id));
  return (
    <div className="stack">
      <div className="price-wrap">
        <table className="price-table" data-testid={testId}>
          <caption className="sr-only">What each service costs</caption>
          <thead className="sr-only">
            <tr><th scope="col">Service</th><th scope="col">Price</th></tr>
          </thead>
          <tbody>
            {groups.map((g) => (g.id === 'free'
              ? <FreeRows key={g.id} title={g.title} />
              : <GroupRows key={g.id} group={g} rates={data.rates} free={data.free} />))}
          </tbody>
        </table>
      </div>
      {showNote && <p className="small muted mb0" data-testid="billing-note">{BILLING_NOTE}</p>}
    </div>
  );
}

function FreeRows({ title }) {
  return (
    <>
      <tr className="group"><th scope="rowgroup" colSpan={2}>{title}</th></tr>
      <tr>
        <th scope="row">
          <span className="svc">Clean up a recording in your browser</span>
          <span className="small muted desc">Takes noise out of audio and video on your own device. Nothing is uploaded.</span>
        </th>
        <td className="num"><span className="price-main">Free</span><span className="small muted price-sub">no sign-in needed</span></td>
      </tr>
    </>
  );
}

function GroupRows({ group, rates, free }) {
  const rows = rates.filter((r) => group.products.includes(r.product) && r.available);
  if (rows.length === 0) return null;
  return (
    <>
      <tr className="group"><th scope="rowgroup" colSpan={2}>{group.title}</th></tr>
      {rows.map((r) => {
        const desc = DESCRIPTIONS[`${r.product}/${r.tier}`];
        const sub = perMinuteNote(r.usdPerMinute);
        return (
          <tr key={`${r.product}/${r.tier}`}>
            <th scope="row">
              <span className="svc">{r.label}</span>
              {desc && <span className="small muted desc">{desc}</span>}
              {r.product === 'transcribe' && r.tier === 'local' && free && (
                <span className="small desc free-note">
                  Your first {minutes(free.transcribeSecondsPerDay)} minutes every day are free (files up to {minutes(free.transcribeMaxFileSeconds)} minutes).
                </span>
              )}
            </th>
            <td className="num">
              <span className="price-main">{moneyUsd(r.usdPerHour)} / hour</span>
              {sub && <span className="small muted price-sub">{sub}</span>}
            </td>
          </tr>
        );
      })}
    </>
  );
}
