import { BILLING_NOTE, moneyUsd, rate } from '../format.js';
import { usePricing } from '../hooks.js';

// What each service is, in one plain line. Keyed by "product/tier" (the ids the price list uses); the names and the
// prices themselves come from the server (GET /api/billing/pricing), so they can never disagree with the bill.
const DESCRIPTIONS = {
  'denoise/standard': 'Background noise removed, the voice kept natural',
  'leveling/standard': 'The voice evened out and the loudness set for podcast, YouTube or broadcast; with Denoising or Voice rebuild, or on its own',
  'denoise/rebuild': 'For echoey rooms: the voice is rebuilt, which also removes room echo and restores dull sound',
  'transcribe/local': 'Subtitles and transcripts (SRT, VTT, TXT)',
  'transcribe/premium': 'The same files, best for accents, noisy rooms and specialist words',
  'stream_denoise/light': 'Cleans up your mic while you talk',
  'stream_captions/local': 'Live captions in English',
  'stream_captions/premium': 'Live captions in other languages too, best for accents and specialist words',
  'transcribe_translate/standard': 'The subtitles and transcript in another language, for each language you add',
  'stream_translate/standard': 'Your live captions in another language, for each language you add',
};

// The server's names for two rows say "Transcription"; everywhere else on the site this is "Subtitles & transcripts".
const LABELS = {
  'transcribe/local': 'Subtitles & transcripts',
  'transcribe/premium': 'Subtitles & transcripts, extra accurate',
  'transcribe_translate/standard': 'Translated subtitles, per language',
};

const GROUPS = [
  { id: 'free', title: 'Free', products: [] },
  { id: 'files', title: 'Files', products: ['denoise', 'leveling', 'transcribe', 'transcribe_translate'] },
  { id: 'live', title: 'Live streaming', products: ['stream_denoise', 'stream_captions', 'stream_translate'] },
];


/** "$0.60 / hour" is the headline, the per-minute rate under it: "$0.01 / min", "$0.004 / min". */
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
              : <GroupRows key={g.id} group={g} rates={data.rates} />))}
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
          <span className="svc">Noise removal in your browser</span>
          <span className="small muted desc">Takes noise out of audio and video on your own device. Nothing is uploaded.</span>
        </th>
        <td className="num"><span className="price-main">Free</span></td>
      </tr>
      <tr>
        <th scope="row">
          <span className="svc">Subtitles &amp; transcripts in your browser</span>
          <span className="small muted desc">Made on your own device, nothing is uploaded. Fine for clear speech; the paid ones below are more accurate.</span>
        </th>
        <td className="num"><span className="price-main">Free</span></td>
      </tr>
    </>
  );
}

function GroupRows({ group, rates }) {
  const rows = rates.filter((r) => group.products.includes(r.product) && r.available);
  if (rows.length === 0) return null;
  return (
    <>
      <tr className="group"><th scope="rowgroup" colSpan={2}>{group.title}</th></tr>
      {rows.map((r) => {
        const desc = DESCRIPTIONS[`${r.product}/${r.tier}`];
        return (
          <tr key={`${r.product}/${r.tier}`}>
            <th scope="row">
              <span className="svc">{LABELS[`${r.product}/${r.tier}`] || r.label}</span>
              {desc && <span className="small muted desc">{desc}</span>}
            </th>
            <td className="num">
              <span className="price-main">{rate(r.usdPerMinute)} / min</span>
              <span className="small muted price-sub">{moneyUsd(r.usdPerHour)} / hour</span>
            </td>
          </tr>
        );
      })}
    </>
  );
}
