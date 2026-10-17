// The dashboard page: reads /api/* for the chosen time range and draws the numbers, graphs and tables.
'use strict';

const $ = (id) => document.getElementById(id);
const usd = (v) => `$${(v || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const num = (v) => (v == null ? '-' : Number(v).toLocaleString(undefined, { maximumFractionDigits: 1 }));
const gb = (bytes) => (bytes == null ? null : bytes / 1024 ** 3);
const when = (ts) => new Date(ts * 1000).toLocaleString();
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const charts = {};

function bucketLabel(ts, bucket) {
  const d = new Date(ts * 1000);
  if (bucket >= 86400) return d.toLocaleDateString();
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function chart(id, type, labels, datasets, opts = {}) {
  charts[id]?.destroy();
  charts[id] = new Chart($(id), {
    type,
    data: { labels, datasets },
    options: {
      animation: false, responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'bottom' } },
      ...(type === 'doughnut' ? {} : { scales: { x: { stacked: !!opts.stacked }, y: { stacked: !!opts.stacked, beginAtZero: true } } }),
      ...opts.extra,
    },
  });
  $(id).parentElement.style.height = '320px';
}

function cards(id, items) {
  $(id).innerHTML = items.map(([label, value, cls]) => `<div class="card"><div class="v ${cls || ''}">${esc(value)}</div><div class="l">${esc(label)}</div></div>`).join('');
}

function table(id, head, rows) {
  $(id).innerHTML = rows.length
    ? `<tr>${head.map((h) => `<th class="${h.n ? 'n' : ''}">${esc(h.t)}</th>`).join('')}</tr>` +
      rows.map((r) => `<tr>${r.map((c, i) => `<td class="${head[i].n ? 'n' : ''}">${esc(c)}</td>`).join('')}</tr>`).join('')
    : '<tr><td class="muted">Nothing in this range.</td></tr>';
}

function range() {
  const from = $('from').value ? Math.floor(new Date($('from').value).getTime() / 1000) : null;
  const to = $('to').value ? Math.floor(new Date($('to').value).getTime() / 1000) : Math.floor(Date.now() / 1000);
  return { start: from ?? to - 30 * 86400, end: to };
}

function setPreset(days) {
  const now = new Date();
  const local = (d) => new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
  $('to').value = local(now);
  $('from').value = local(days ? new Date(now.getTime() - days * 86400000) : new Date(Date.UTC(2026, 0, 1)));
  document.querySelectorAll('.presets button').forEach((b) => b.classList.toggle('on', Number(b.dataset.days) === days));
}

async function get(path, r) {
  const res = await fetch(`${path}?start=${r.start}&end=${r.end}`);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

async function load() {
  const r = range();
  $('status').textContent = 'Loading…';
  try {
    const [ov, se, us, er] = await Promise.all([get('/api/overview', r), get('/api/series', r), get('/api/users', r), get('/api/errors', r)]);
    const m = ov.money;
    cards('money', [
      ['Top-ups', usd(m.topupsUsd)], ['Payments', num(m.payments)], ['Spent by users', usd(m.usageUsd)],
      ['Refunds', usd(m.refundsUsd), m.refundsUsd ? 'bad' : ''], ['Disputes', usd(m.disputesUsd), m.disputesUsd ? 'bad' : ''],
      ['Card fees (estimate)', usd(m.feesUsdEstimate)], ['Net (top-ups − refunds − disputes − fees)', usd(m.netUsd)],
      ['Tax collected', usd(m.taxUsd)], ['Credit given away', usd(m.grantsUsd)], ['Failed payments', num(m.failedPayments), m.failedPayments ? 'bad' : ''],
      ['Credit people hold now', usd(m.outstandingCreditUsd)],
    ]);
    const u = ov.users;
    cards('users', [
      ['Accounts', num(u.accounts)], ['Sign-ups', num(u.signups)], ['Active (used something)', num(u.active)],
      ['Spent credit', num(u.spenders)], ['Paid in this range', num(u.payersInRange)], ['Ever paid', num(u.payersEver)],
      ['Never paid (free only)', num(u.neverPaid)], ['Used the live trial', num(u.liveTrialUsers)],
    ]);
    const j = ov.jobs;
    const jobCount = Object.values(j.byType).reduce((a, s) => a + Object.values(s).reduce((x, y) => x + y, 0), 0);
    cards('usage', [
      ['File jobs', num(jobCount)], ['Minutes processed', num(j.minutesProcessed)],
      ['Queue wait, median (s)', num(j.waitP50)], ['Queue wait, 90 % (s)', num(j.waitP90)],
      ['Live sessions', num(ov.live.sessions)], ['Live hours', num(ov.live.hours)], ['Trial live hours', num(ov.live.trialHours)],
    ]);

    const labels = se.points.map((p) => bucketLabel(p.t, se.bucket));
    chart('cMoney', 'bar', labels, [
      { label: 'Top-ups', data: se.points.map((p) => p.topupsUsd) },
      { label: 'Spent', data: se.points.map((p) => p.usageUsd) },
      { label: 'Refunds and disputes', data: se.points.map((p) => p.refundsUsd) },
    ]);
    const prod = Object.entries(ov.revenueByProduct);
    chart('cProducts', 'doughnut', prod.map(([k]) => k), [{ data: prod.map(([, v]) => v) }]);
    chart('cUsers', 'line', labels, [
      { label: 'Sign-ups', data: se.points.map((p) => p.signups) },
      { label: 'Active users', data: se.points.map((p) => p.activeUsers) },
    ]);
    const ubp = Object.entries(ov.usersByProduct);
    chart('cUsersByProduct', 'bar', ubp.map(([k]) => k), [{ label: 'Users', data: ubp.map(([, v]) => v) }], { extra: { indexAxis: 'y' } });
    chart('cUsage', 'bar', labels, [
      { label: 'Jobs done', data: se.points.map((p) => p.jobsDone) },
      { label: 'Jobs failed', data: se.points.map((p) => p.jobsFailed) },
      { label: 'Live hours', data: se.points.map((p) => p.liveHours), type: 'line' },
    ]);

    const mc = se.machine;
    const mlabels = mc.map((p) => bucketLabel(p.t, se.bucket));
    chart('cCpu', 'line', mlabels, [
      { label: 'CPU cores busy', data: mc.map((p) => p.cpu_busy) },
      { label: 'CPU cores', data: mc.map((p) => p.cpu_cores), borderDash: [4, 4], pointRadius: 0 },
      { label: 'GPU % busy', data: mc.map((p) => p.gpu_busy), yAxisID: 'y1' },
    ], { extra: { scales: { y: { beginAtZero: true }, y1: { position: 'right', beginAtZero: true, max: 100 } } } });
    chart('cMem', 'line', mlabels, [
      { label: 'Available on the machine', data: mc.map((p) => gb(p.mem_available)) },
      { label: 'Room under the 12 GB limit', data: mc.map((p) => gb(p.slice_headroom)) },
      { label: 'GPU memory used', data: mc.map((p) => (p.gpu_mem_used == null ? null : p.gpu_mem_used / 1024)) },
    ]);
    chart('cQueue', 'line', mlabels, [
      { label: 'Jobs waiting', data: mc.map((p) => p.jobs_queued) },
      { label: 'Jobs running', data: mc.map((p) => p.jobs_running) },
      { label: 'Live sessions', data: mc.map((p) => p.live_sessions) },
      { label: 'Live capacity used (units)', data: mc.map((p) => p.live_units) },
    ]);
    chart('cDisk', 'line', mlabels, [{ label: 'Disk free', data: mc.map((p) => gb(p.disk_free)) }]);

    table('topUsers', [{ t: 'Email' }, { t: 'Spent', n: 1 }, { t: 'Top-ups', n: 1 }, { t: 'Jobs', n: 1 }, { t: 'Live hours', n: 1 }, { t: 'Balance', n: 1 }, { t: 'Since' }],
      us.users.map((x) => [x.email, usd(x.spentUsd), usd(x.topupsUsd), x.jobs, x.liveHours, usd(x.balanceUsd), new Date(x.since * 1000).toLocaleDateString()]));
    const st = (s, ...names) => names.reduce((a, n) => a + (s[n] || 0), 0);
    table('jobTypes', [{ t: 'Type' }, { t: 'Done', n: 1 }, { t: 'Failed', n: 1 }, { t: 'Canceled or never uploaded', n: 1 }, { t: 'In progress', n: 1 }],
      Object.entries(j.byType).map(([k, s]) => [k, st(s, 'done'), st(s, 'failed'), st(s, 'canceled', 'expired'), st(s, 'uploading', 'queued', 'running', 'cloud_asr')]));
    table('failReasons', [{ t: 'Reason' }, { t: 'Jobs', n: 1 }], Object.entries(er.failureReasons));
    table('liveEnds', [{ t: 'Ended because' }, { t: 'Sessions', n: 1 }], Object.entries(er.liveEndReasons));
    table('alerts', [{ t: 'Alert' }, { t: 'Now' }, { t: 'Last mailed' }, { t: 'Mails', n: 1 }],
      er.alerts.map((a) => [a.subject || a.kind, a.active ? 'ACTIVE' : 'resolved', a.last_sent_at ? when(a.last_sent_at) : '-', a.send_count]));
    table('moneyProblems', [{ t: 'When' }, { t: 'What' }, { t: 'Detail' }], er.moneyProblems.map((p) => [when(p.created_at), p.action, p.detail]));
    table('failedJobs', [{ t: 'When' }, { t: 'User' }, { t: 'Kind' }, { t: 'Minutes', n: 1 }, { t: 'Error' }],
      er.failedJobs.map((f) => [when(f.created_at), f.email, f.kind, num((f.duration_seconds || 0) / 60), f.error]));
    $('status').textContent = `${new Date(r.start * 1000).toLocaleString()} – ${new Date(r.end * 1000).toLocaleString()}`;
  } catch (e) {
    $('status').textContent = `Could not load: ${e.message}`;
  }
}

document.querySelectorAll('.presets button').forEach((b) => b.addEventListener('click', () => { setPreset(Number(b.dataset.days)); load(); }));
$('apply').addEventListener('click', () => { document.querySelectorAll('.presets button').forEach((b) => b.classList.remove('on')); load(); });
setPreset(30);
load();
