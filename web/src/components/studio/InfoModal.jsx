import { useEffect, useRef } from 'react';
import { ALTERNATIVES } from '../../alternatives.jsx';
import { priceWords } from '../../studio/copy.js';
import { DENOISING_POINTS, REBUILD_POINTS } from '../WhichOne.jsx';

const Row = ({ a, children }) => <tr><th scope="row">{a}</th>{(Array.isArray(children) ? children : [children]).map((x, i) => <td key={i}>{x}</td>)}</tr>;
const Ul = ({ list }) => <ul>{list.map((x) => <li key={x}>{x}</li>)}</ul>;
const price = (t, free = 'Free') => t || free;

/** The (i) next to the selection's title: every option side by side, where each tooltip only tells about its own option. */
export default function InfoModal({ open, onClose, pricing }) {
  const ref = useRef(null);
  const P = priceWords(pricing);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) { try { d.showModal(); } catch { d.setAttribute('open', ''); } }
    if (!open && d.open) d.close();
  }, [open]);
  useEffect(() => {
    const d = ref.current;
    if (!d) return undefined;
    const onCancel = () => onClose();
    d.addEventListener('close', onCancel);
    return () => d.removeEventListener('close', onCancel);
  }, [onClose]);
  return (
    <dialog ref={ref} className="modal wide" aria-labelledby="dlg-h" data-testid="info-modal" onClick={(ev) => { if (ev.target === ref.current) onClose(); }}>
      {open && (
        <>
          <div className="mh"><h2 id="dlg-h">All the options</h2><button type="button" className="btn btn-secondary btn-sm" data-testid="info-close" onClick={onClose}>Close</button></div>
          <div className="mb">
            <section>
              <h3>Voice: Noise removal or Voice rebuild?</h3>
              <div className="two">
                <div><h4>Noise removal</h4><Ul list={DENOISING_POINTS} /></div>
                <div><h4>Voice rebuild</h4><Ul list={REBUILD_POINTS} /></div>
              </div>
            </section>
            <section>
              <h3>Noise removal: Free or Standard</h3>
              <div className="tw">
                <table className="cmp">
                  <thead><tr><th /><th>Free</th><th>Standard</th></tr></thead>
                  <tbody>
                    <Row a="Where">{['In your browser, on your own device. Nothing is uploaded.', 'On our side. Only the sound is uploaded; a video’s picture stays on your device.']}</Row>
                    <Row a="Strength">{['Gentle, Balanced, Strong', 'Gentle, Balanced, Strong, Max']}</Row>
                    <Row a="Remover">{['One', 'Auto, Light or Heavy']}</Row>
                    <Row a="Speed">{['Depends on your device; the tab has to stay open. The first time it downloads about 22 MB.', 'Fast']}</Row>
                    <Row a="Price">{['Free', price(P.noise, '')]}</Row>
                  </tbody>
                </table>
              </div>
              <dl className="dl">
                <dt>Strength</dt>
                <dd>Gentle takes the edge off and keeps some room sound and music. Balanced: most of the noise goes; the room still sounds like a room. Strong: nearly all of the noise goes. Max removes as much as it can; best for noisy places.</dd>
                <dt>Remover</dt>
                <dd>Auto measures how noisy the recording is and picks the best way. Light is for hiss, hum, a quiet room or a fan. Heavy is for a café, traffic, a crowd or wind: noise almost as loud as the voice.</dd>
              </dl>
            </section>
            <section>
              <h3>Echo removal</h3>
              <p>Removes your speaker echo using the file’s second audio track (your desktop sound). Only for files that have one: in OBS, put the mic on track 1 and the desktop sound on track 2.{P.echo ? ` ${P.echo}.` : ''}</p>
            </section>
            <section>
              <h3>Leveling</h3>
              <p>Evens out the volume and sets how loud the result is, with or without noise removal.{P.level ? ` ${P.level}.` : ''}</p>
              <div className="tw">
                <table className="cmp">
                  <tbody>
                    <Row a="Podcast">−16 LUFS, the usual loudness for podcasts</Row>
                    <Row a="YouTube">−14 LUFS, the loudness YouTube plays videos at</Row>
                    <Row a="Broadcast">−23 LUFS, the TV and radio standard</Row>
                    <Row a="Keep">Evens out the volume but keeps the overall loudness as recorded</Row>
                  </tbody>
                </table>
              </div>
            </section>
            <section>
              <h3>Subtitles: Free, Standard or Extra accurate</h3>
              <div className="tw">
                <table className="cmp">
                  <thead><tr><th /><th>Free</th><th>Standard</th><th>Extra accurate</th></tr></thead>
                  <tbody>
                    <Row a="Where">{['In your browser. Nothing is uploaded.', 'On our side', 'On our side, with a speech partner we work with (your audio goes to them, only for this option)']}</Row>
                    <Row a="Best for">{['Clear speech', 'Most recordings', 'Accents, noisy rooms, specialist words']}</Row>
                    <Row a="Languages">{['The 28 in the list, or detected', 'The 28 in the list, or detected', 'Any language']}</Row>
                    <Row a="Translations">{['No', 'Up to 3 languages', 'Up to 3 languages']}</Row>
                    <Row a="Speed">{['Slower on long files. The first time it downloads about 80 MB.', 'Fast', 'Fast']}</Row>
                    <Row a="Price">{['Free', price(P.std, ''), price(P.ext, '')]}</Row>
                  </tbody>
                </table>
              </div>
              <p>Translations are made from the transcript: one set of subtitle files per language{P.tr ? `, ${P.tr} each` : ''}. Files: SRT and VTT subtitles and a TXT transcript, and the subtitles can go into the video as an MKV file.</p>
            </section>
            <section>
              <h3>Prices</h3>
              <div className="tw">
                <table className="cmp prices">
                  <tbody>
                    <Row a="Noise removal, Free">Free</Row>
                    <Row a="Noise removal, Standard">{P.noise}</Row>
                    <Row a="Voice rebuild">{P.rebuild}</Row>
                    <Row a="Echo removal">{P.echo}</Row>
                    <Row a="Leveling">{P.level}</Row>
                    <Row a="Subtitles, Free">Free</Row>
                    <Row a="Subtitles, Standard">{P.std}</Row>
                    <Row a="Subtitles, Extra accurate">{P.ext}</Row>
                    <Row a="Translations">{P.tr ? `${P.tr} per language` : ''}</Row>
                  </tbody>
                </table>
              </div>
              <p className="small muted">You only pay for what gets processed. Billed per second of audio, no minimum charge. Each part you choose can have its own settings, and you pay for each part with what it uses.</p>
            </section>
            <section>
              <h3>Good to know</h3>
              <Ul list={[ALTERNATIVES.browserDenoise, ALTERNATIVES.denoise, ALTERNATIVES.rebuild, ALTERNATIVES.browserSubs, ALTERNATIVES.accurateSubs, ALTERNATIVES.translations]} />
            </section>
          </div>
        </>
      )}
    </dialog>
  );
}
