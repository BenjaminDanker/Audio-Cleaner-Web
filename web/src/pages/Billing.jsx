import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router';
import { api } from '../api.js';
import { useAuth } from '../auth.jsx';
import { balanceMoney, dateTime, money, moneyCents } from '../format.js';
import { ledgerKind, ledgerText } from '../ledger.js';
import { usePageTitle, usePricing } from '../hooks.js';
import { SITE } from '../site.js';

// Only ever leave the site for Stripe-hosted pages.
const STRIPE_HOSTS = ['checkout.stripe.com', 'billing.stripe.com'];
function goStripe(url) {
  const u = new URL(url);
  if (u.protocol !== 'https:' || !STRIPE_HOSTS.includes(u.hostname)) throw Object.assign(new Error('Unexpected payment page address.'), { detail: 'Unexpected payment page address.' });
  window.location.assign(u.href);
}

const REFUND_DAYS = 14;

const NOT_SERVED = 'Cards issued in the EU/EEA, the UK or Switzerland can’t be used, so this payment was cancelled and the card wasn’t charged.';
const STATUS = {
  succeeded: ['Paid', 'badge badge-ok'], pending: ['Processing', 'badge'], failed: ['Failed', 'badge badge-bad'],
  refunded: ['Refunded', 'badge'], disputed: ['Disputed', 'badge badge-bad'], refused_region: ['Not charged', 'badge'],
};

/** The card processor keeps about 2.9% + 30 cents of every payment; the first refund of a payment takes that off. */
const cardFeeCents = (paymentCents) => Math.ceil(paymentCents * 0.029 + 30);

/** What a refund of this payment should return to the card, as an estimate: unused credit minus the card fee (first refund only). */
function refundEstimate(p, availableMicros) {
  const unused = Math.max(0, Math.min(p.amount_cents - (p.refunded_cents || 0), Math.floor((availableMicros || 0) / 10000)));
  const fee = (p.refunded_cents || 0) === 0 ? cardFeeCents(p.amount_cents) : 0;
  return { unused, fee, back: Math.max(0, unused - fee) };
}

/**
 * "25", "$25", "12.50", "1,250.00" and "12,5" (comma as the decimal mark) are read; anything else (a minus sign,
 * letters, "1.2.3") is NaN, so the form says it is not valid instead of guessing a different amount.
 */
function dollarsToCents(text) {
  let t = String(text).trim().replace(/^\$\s*/, '').replace(/\s+/g, '');
  if (/^\d{1,3}(,\d{3})+(\.\d+)?$/.test(t)) t = t.replace(/,/g, '');
  else if (/^\d+,\d{1,2}$/.test(t)) t = t.replace(',', '.');
  if (!/^(\d+\.?\d*|\.\d+)$/.test(t)) return NaN;
  const n = Number(t);
  return Number.isFinite(n) ? Math.round(n * 100) : NaN;
}

export default function Billing() {
  usePageTitle('Billing');
  const { balance, autoReload, refresh } = useAuth();
  const { data: pricing } = usePricing();
  const [params, setParams] = useSearchParams();
  const [banner, setBanner] = useState(null);
  const [history, setHistory] = useState(null);
  const [histErr, setHistErr] = useState('');
  const startBalance = useRef(null);

  const loadHistory = useCallback(async () => {
    try {
      setHistory(await api('billing/history'));
      setHistErr('');
    } catch (e) {
      setHistErr(e.detail);
    }
  }, []);

  useEffect(() => { loadHistory(); }, [loadHistory]);

  // ?checkout=success|cancel after returning from Stripe. The webhook may take a few seconds.
  useEffect(() => {
    const c = params.get('checkout');
    const paymentId = params.get('p'); // this payment's id (set by the server in the return address)
    if (!c) return undefined;
    setParams({}, { replace: true });
    if (c === 'cancel') {
      setBanner({ kind: 'info', text: 'No problem, checkout was canceled and you weren’t charged.' });
      return undefined;
    }
    if (c !== 'success') return undefined;
    startBalance.current = balance?.available_micros ?? 0;
    setBanner({ kind: 'info', text: 'Payment received. Checking it and updating your balance…', busy: true });
    let tries = 0;
    let stop = false;
    const tick = async () => {
      if (stop) return;
      const me = await refresh();
      tries += 1;
      // Paying is "authorize, then charge once the card's country is checked": a card from a country we don't
      // serve is cancelled, never credited, so look at this payment as well as the balance.
      const mine = paymentId ? await api('billing/history').then((h) => h.payments?.find((x) => x.id === paymentId)).catch(() => null) : null;
      if (mine?.status === 'refused_region') {
        setBanner({ kind: 'warn', text: NOT_SERVED });
        loadHistory();
        return;
      }
      // The payment may already be booked by the time we look (the card processor is usually faster than this page).
      if (mine?.status === 'succeeded' || (me?.balance && me.balance.available_micros > startBalance.current)) {
        const now = me?.balance?.available_micros;
        setBanner({ kind: 'success', text: now != null ? `Thank you! Your balance is now ${balanceMoney(now)}.` : 'Thank you! Your payment went through.' });
        loadHistory();
        return;
      }
      if (tries >= 10) {
        setBanner({ kind: 'warn', text: 'Your payment went through, but the balance hasn’t caught up yet. It can take a minute. Refresh this page in a bit.' });
        return;
      }
      setTimeout(tick, 2000);
    };
    tick();
    return () => { stop = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const unavailable = pricing && pricing.paymentsEnabled === false;
  return (
    <div className="container page stack-lg">
      <header>
        <h1 className="mb0">Billing</h1>
        <p className="muted mb0">Add credit, spend it as you go. No subscription.</p>
      </header>

      {banner && (
        <div className={`banner banner-${banner.kind}`} role="status" data-testid="checkout-banner">
          {banner.busy && <span className="spinner" aria-hidden="true" />} {banner.text}
        </div>
      )}
      {unavailable && <div className="banner banner-warn" role="status" data-testid="payments-off">Payments aren’t switched on yet. You’ll be able to add credit here soon.</div>}

      <div className="grid grid-3">
        <section className="card" aria-labelledby="b1"><h2 id="b1" className="mb0 small muted">Available</h2><div className="big-number" data-testid="available">{balanceMoney(balance?.available_micros ?? 0)}</div></section>
        <section className="card" aria-labelledby="b2"><h2 id="b2" className="mb0 small muted">Set aside</h2><div className="big-number" data-testid="held">{balanceMoney(balance?.held_micros ?? 0)}</div><div className="muted small">Set aside for jobs and live sessions that are running</div></section>
        <section className="card" aria-labelledby="b3"><h2 id="b3" className="mb0 small muted">Total</h2><div className="big-number">{balanceMoney(balance?.balance_micros ?? 0)}</div></section>
      </div>

      <TopUp pricing={pricing} unavailable={unavailable} />
      <AutoReload autoReload={autoReload} pricing={pricing} refresh={refresh} unavailable={unavailable} />
      <History history={history} error={histErr} availableMicros={balance?.available_micros ?? 0} reload={() => { loadHistory(); refresh(); }} unavailable={unavailable} />
    </div>
  );
}

function TopUp({ pricing, unavailable }) {
  const min = pricing?.topup?.minCents ?? 1000;
  const max = pricing?.topup?.maxCents ?? 50000;
  const presets = pricing?.topup?.presetsCents ?? [1000, 2000, 5000, 10000];
  const [choice, setChoice] = useState('2000'); // preset cents or 'custom'
  const [custom, setCustom] = useState('25');
  const [saveCard, setSaveCard] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const cents = choice === 'custom' ? dollarsToCents(custom) : Number(choice);
  const valid = Number.isInteger(cents) && cents >= min && cents <= max;

  async function go(e) {
    e.preventDefault();
    if (!valid || unavailable) return;
    setError('');
    setBusy(true);
    try {
      const { url } = await api('billing/checkout', { method: 'POST', body: { amountCents: cents, saveCard } });
      goStripe(url);
    } catch (err) {
      setError(err.status === 503 ? 'Payments aren’t available yet. Please try again later.' : err.detail);
      setBusy(false);
    }
  }

  return (
    <form className="card stack" onSubmit={go} aria-labelledby="topup-h">
      <h2 id="topup-h" className="mb0">Add credit</h2>
      {pricing?.taxAddedAtCheckout && <p className="muted mb0">Sales tax or VAT is added at checkout where it applies. You get the full amount as credit.</p>}
      <fieldset className="field">
        <legend className="label">Amount</legend>
        <div className="choice-cards">
          {presets.map((c) => (
            <label className="choice" key={c}>
              <input type="radio" name="amount" value={c} checked={choice === String(c)} onChange={() => setChoice(String(c))} />
              <span><b>{moneyCents(c)}</b></span>
            </label>
          ))}
          <label className="choice">
            <input type="radio" name="amount" value="custom" checked={choice === 'custom'} onChange={() => setChoice('custom')} />
            <span><b>Other amount</b><small>{moneyCents(min)} to {moneyCents(max)}</small></span>
          </label>
        </div>
      </fieldset>
      {choice === 'custom' && (
        <div className="field">
          <label htmlFor="custom-amt">Custom amount (USD)</label>
          <div className="inline-input">$<input id="custom-amt" className="input input-money" inputMode="decimal" value={custom} onChange={(e) => setCustom(e.target.value)} aria-invalid={!valid} aria-describedby="custom-hint" /></div>
          <span id="custom-hint" className={valid ? 'hint' : 'hint neg'}>Between {moneyCents(min)} and {moneyCents(max)}.</span>
        </div>
      )}
      <label className="check">
        <input type="checkbox" checked={saveCard} onChange={(e) => setSaveCard(e.target.checked)} />
        <span>Save my card for auto-reload <span className="muted small">(optional; you set the rules below and can turn it off any time)</span></span>
      </label>
      {error && <div className="banner banner-error" role="alert" data-testid="topup-error">{error}</div>}
      <div className="row">
        <button type="submit" className="btn btn-primary btn-lg" disabled={!valid || busy || unavailable} aria-describedby={unavailable ? 'topup-off' : undefined}>{busy ? 'Opening checkout…' : `Pay ${valid ? moneyCents(cents) : ''} with Stripe`}</button>
        <span className="muted small">Payments go through Stripe. We never see your card number.</span>
      </div>
      {unavailable && <p id="topup-off" className="hint mb0" data-testid="topup-off">Adding credit is switched off for now. This button will work as soon as payments are on.</p>}
    </form>
  );
}

function AutoReload({ autoReload, pricing, refresh, unavailable }) {
  const min = pricing?.topup?.minCents ?? 1000;
  const max = pricing?.topup?.maxCents ?? 50000;
  const hasCard = !!autoReload?.hasCard;
  const [threshold, setThreshold] = useState('5');
  const [amount, setAmount] = useState('20');
  const [cap, setCap] = useState('100');
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState('');
  const [msg, setMsg] = useState(null);

  useEffect(() => {
    if (!autoReload || !autoReload.enabled) return; // keep sensible defaults until the user has configured it
    const toD = (c) => String(c / 100);
    if (autoReload.amountCents) setAmount(toD(autoReload.amountCents));
    if (autoReload.monthlyCapCents) setCap(toD(autoReload.monthlyCapCents));
    if (autoReload.thresholdMicros != null) setThreshold(toD(Math.round(autoReload.thresholdMicros / 10000)));
  }, [autoReload?.enabled, autoReload?.amountCents, autoReload?.monthlyCapCents, autoReload?.thresholdMicros]); // eslint-disable-line react-hooks/exhaustive-deps

  const t = dollarsToCents(threshold);
  const a = dollarsToCents(amount);
  const c = dollarsToCents(cap);
  const problems = [];
  if (!(t >= 0 && t <= max)) problems.push(`The “below” amount must be between $0 and ${moneyCents(max)}.`);
  if (!(a >= min && a <= max)) problems.push(`The reload amount must be between ${moneyCents(min)} and ${moneyCents(max)}.`);
  if (!(c >= min && c <= 200000)) problems.push(`The monthly cap must be between ${moneyCents(min)} and ${moneyCents(200000)}.`);
  else if (a > c) problems.push('The monthly cap must be at least one reload.');
  const valid = problems.length === 0;

  async function save(enabled) {
    setMsg(null);
    setBusy(enabled ? 'on' : 'off');
    try {
      await api('billing/auto-reload', {
        method: 'POST',
        body: {
          enabled,
          thresholdCents: Number.isFinite(t) ? t : 500,
          amountCents: Number.isFinite(a) && a >= min ? a : 2000,
          monthlyCapCents: Number.isFinite(c) && c >= min ? c : 10000,
          consent: enabled ? consent : false,
        },
      });
      await refresh();
      setConsent(false);
      setMsg({ kind: 'success', text: enabled ? 'Auto-reload is on.' : 'Auto-reload is off.' });
    } catch (e) {
      setMsg({ kind: 'error', text: e.detail });
    } finally {
      setBusy('');
    }
  }

  async function portal() {
    setBusy('portal');
    setMsg(null);
    try {
      const { url } = await api('billing/portal', { method: 'POST' });
      goStripe(url);
    } catch (e) {
      setMsg({ kind: 'error', text: e.status === 503 ? 'Card management is not available yet.' : e.detail });
      setBusy('');
    }
  }

  const on = !!autoReload?.enabled;
  const failures = autoReload?.failures || 0;
  return (
    <section className="card stack" aria-labelledby="ar-h">
      <div className="row-between">
        <h2 id="ar-h" className="mb0">Auto-reload</h2>
        <span className={on ? 'badge badge-ok' : 'badge'} data-testid="ar-state">{on ? 'On' : 'Off'}</span>
      </div>
      <p className="muted mb0">So you don’t run dry mid-job: when your balance drops below an amount you pick, we top it up from your saved card, up to a monthly limit you choose.</p>
      {failures > 0 && (
        <div className="banner banner-warn" role="alert" data-testid="ar-failures">
          The last {failures} automatic reload{failures === 1 ? '' : 's'} didn’t go through{on ? '' : ', so auto-reload was paused'}. Check your card under “Manage card and receipts”, then save these settings again.
        </div>
      )}
      {!hasCard && <div className="banner banner-info">To use auto-reload, tick &ldquo;Save my card for auto-reload&rdquo; on your next top-up above.</div>}

      <fieldset className="stack" disabled={!hasCard || !!busy}>
        <legend className="sr-only">Auto-reload settings</legend>
        <div className="form-grid">
          <div className="field"><label htmlFor="ar-t">When my balance falls below</label><div className="inline-input">$<input id="ar-t" className="input input-money" inputMode="decimal" value={threshold} onChange={(e) => setThreshold(e.target.value)} /></div></div>
          <div className="field"><label htmlFor="ar-a">Charge my card</label><div className="inline-input">$<input id="ar-a" className="input input-money" inputMode="decimal" value={amount} onChange={(e) => setAmount(e.target.value)} /></div></div>
          <div className="field"><label htmlFor="ar-c">Monthly limit</label><div className="inline-input">$<input id="ar-c" className="input input-money" inputMode="decimal" value={cap} onChange={(e) => setCap(e.target.value)} /></div></div>
        </div>
        {hasCard && problems.map((p) => <p key={p} className="hint neg mb0">{p}</p>)}
        <label className="check">
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} data-testid="ar-consent" />
          <span data-testid="ar-consent-text">
            I authorize {SITE.name} to charge my saved card {valid ? moneyCents(a) : 'the amount I choose'} whenever my balance falls below {valid ? moneyCents(t) : 'the balance I choose'}, up to {valid ? moneyCents(c) : 'the monthly limit I choose'} per calendar month, until I turn this off.
          </span>
        </label>
        <div className="row">
          <button type="button" className="btn btn-primary" disabled={!valid || !consent || unavailable} onClick={() => save(true)}>{on ? 'Update auto-reload' : 'Turn on auto-reload'}</button>
          {on && <button type="button" className="btn btn-danger-outline" onClick={() => save(false)}>Turn off</button>}
        </div>
      </fieldset>
      {msg && <div className={`banner banner-${msg.kind}`} role={msg.kind === 'error' ? 'alert' : 'status'}>{msg.text}</div>}
      <div className="row">
        <button type="button" className="btn btn-secondary" disabled={!!busy || unavailable} onClick={portal}>Manage card and receipts</button>
        <span className="muted small">{unavailable ? 'Not available while payments are off.' : 'Opens Stripe’s secure page.'}</span>
      </div>
    </section>
  );
}

function History({ history, error, reload, unavailable, availableMicros }) {
  const [busy, setBusy] = useState('');
  const [msg, setMsg] = useState(null);
  const [confirmId, setConfirmId] = useState(null);

  async function refund(id) {
    setBusy(id); setMsg(null);
    try {
      const r = await api('billing/refund', { method: 'POST', body: { paymentId: id } });
      setMsg({ kind: 'success', text: `Done. ${moneyCents(r.refundedCents)} is on its way back to your card.` });
      setConfirmId(null);
      reload();
    } catch (e) {
      setMsg({ kind: 'error', text: e.status === 503 ? 'Refunds aren’t available yet.' : e.detail });
    } finally {
      setBusy('');
    }
  }
  const nowS = Date.now() / 1000;
  return (
    <section className="stack" aria-labelledby="hist-h">
      <h2 id="hist-h" className="mb0">History</h2>
      {error && <div className="banner banner-error" role="alert">{error}</div>}
      {msg && <div className={`banner banner-${msg.kind}`} role={msg.kind === 'error' ? 'alert' : 'status'}>{msg.text}</div>}
      {!history && !error && <p role="status" className="muted"><span className="spinner" aria-hidden="true" /> Loading…</p>}
      {history && (
        <>
          <h3 className="mb0">Payments</h3>
          {history.payments.length === 0 ? <p className="card muted mb0">No payments yet.</p> : (
            <div className="table-wrap">
              <table data-testid="payments">
                <thead><tr><th>Date</th><th>Type</th><th>Status</th><th className="num">Amount</th><th className="num">Refunded</th><th><span className="sr-only">Actions</span></th></tr></thead>
                <tbody>
                  {history.payments.map((p) => {
                    const eligible = p.status === 'succeeded' && nowS - p.created_at < REFUND_DAYS * 86400 && (p.refunded_cents || 0) < p.amount_cents;
                    return (
                      <tr key={p.id}>
                        <td className="nowrap">{dateTime(p.created_at)}</td>
                        <td>{p.kind === 'auto_reload' ? 'Auto-reload' : 'Top-up'}</td>
                        <td><span className={(STATUS[p.status] || [p.status, 'badge'])[1]} title={p.status === 'refused_region' ? NOT_SERVED : undefined} data-testid="pay-status">{(STATUS[p.status] || [p.status])[0]}</span></td>
                        <td className="num">{moneyCents(p.amount_cents)}</td>
                        <td className="num">{p.refunded_cents ? moneyCents(p.refunded_cents) : '—'}</td>
                        <td className="right">
                          {eligible && confirmId !== p.id && <button type="button" className="btn btn-secondary btn-sm" disabled={unavailable} aria-describedby={unavailable ? 'refund-off' : undefined} onClick={() => setConfirmId(p.id)}>Refund unused</button>}
                          {eligible && confirmId === p.id && (() => {
                            const est = refundEstimate(p, availableMicros);
                            return (
                              <span className="row" data-testid="refund-confirm">
                                <span className="small" data-testid="refund-estimate">
                                  {est.back > 0
                                    ? <>You&rsquo;ll get back about <b>{moneyCents(est.back)}</b> (unused {moneyCents(est.unused)} minus the card fee{est.fee ? ` of ${moneyCents(est.fee)}` : ''}).</>
                                    : <>Your unused credit ({moneyCents(est.unused)}) doesn&rsquo;t cover the card fee{est.fee ? ` (${moneyCents(est.fee)})` : ''}, so there&rsquo;s nothing to send back yet.</>}
                                </span>
                                <button type="button" className="btn btn-danger btn-sm" disabled={busy === p.id || unavailable || est.back <= 0} onClick={() => refund(p.id)}>{busy === p.id ? 'Refunding…' : 'Yes, refund'}</button>
                                <button type="button" className="btn btn-ghost btn-sm" onClick={() => setConfirmId(null)}>No</button>
                              </span>
                            );
                          })()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          {unavailable && <p id="refund-off" className="hint mb0" data-testid="refund-off">Refunds are switched off for now. The refund buttons will work again as soon as payments are on.</p>}
          <p className="small muted mb0">You can get the unused part of a payment from the last {REFUND_DAYS} days refunded here. The card processor keeps its fee (about 2.9% + 30¢) from the first refund of each payment. The <Link to="/terms">Terms</Link> have the details.</p>
          <h3 className="mb0">Credit history</h3>
          {history.ledger.length === 0 ? <p className="card muted mb0">No activity yet.</p> : (
            <div className="table-wrap">
              <table data-testid="ledger">
                <thead><tr><th>Date</th><th>What</th><th>Type</th><th className="num">Amount</th><th className="num">Balance</th></tr></thead>
                <tbody>
                  {history.ledger.map((l) => (
                    <tr key={l.id}>
                      <td className="nowrap">{dateTime(l.created_at)}</td>
                      <td>{ledgerText(l)}</td>
                      <td><span className="badge">{ledgerKind(l)}</span></td>
                      <td className={`num ${l.amount_micros >= 0 ? 'pos' : 'neg'}`}>{l.amount_micros >= 0 ? '+' : '−'}{money(Math.abs(l.amount_micros))}</td>
                      <td className="num">{money(l.balance_after_micros)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {(history.ledgerHasMore ?? history.ledger.length >= 100) && <p className="small muted mb0">Showing your 100 most recent entries.</p>}
        </>
      )}
    </section>
  );
}
