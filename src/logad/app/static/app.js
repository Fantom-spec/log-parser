'use strict';
// Plain JS, no build step. Every piece of server/log-derived text goes through textContent (el/append).
const $view = document.getElementById('view'), $nav = document.getElementById('nav');
const el = (t, p = {}, ...k) => { const e = document.createElement(t);
  for (const [a, v] of Object.entries(p)) { if (a === 'class') e.className = v; else e[a] = v; }
  for (const c of k) { if (c == null) continue; e.append(c instanceof Node ? c : document.createTextNode(String(c))); } return e; };
const f = (x, d = 3) => x == null ? '–' : (typeof x === 'number' ? x.toFixed(d) : String(x));
const badge = t => el('span', { class: 'badge ' + t }, t);
const TIERS = ['NORMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];
let timer = null;

async function api(path, body) {
  const opts = body === undefined ? { headers: { 'X-Requested-With': 'logad' } } :
    { method: 'POST', headers: { 'X-Requested-With': 'logad', 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const r = await fetch(path, opts), txt = await r.text();
  let j; try { j = JSON.parse(txt); } catch (e) { j = { error: txt.slice(0, 200) }; }
  if (!r.ok) throw Object.assign(new Error(j.error || r.statusText), { status: r.status, body: j });
  return j;
}
const table = (head, rows, onClick) => { const t = el('table'), h = el('tr'); head.forEach(x => h.append(el('th', {}, x))); t.append(el('thead', {}, h));
  const b = el('tbody'); rows.forEach((r, i) => { const tr = el('tr', onClick ? { class: 'click' } : {}); r.forEach(c => tr.append(el('td', {}, c)));
    if (onClick) tr.onclick = () => onClick(i, tr); b.append(tr); }); t.append(b); return el('div', { class: 'scroll' }, t); };
const stat = (label, value) => el('div', { class: 'card stat' }, el('b', {}, value), label);
const errBox = e => el('div', { class: 'card err' }, 'Error: ' + e.message + (e.body && e.body.hint ? ' — ' + e.body.hint : ''));
const chips = a => el('span', {}, ...a.map(x => el('span', { class: 'chip' }, x)));
const go = h => { location.hash = h; };

// ---------------- Home ----------------
async function home() {
  const s = await api('/api/status'), root = el('div');
  root.append(el('h1', {}, 'SSH login anomaly detection'),
    el('p', { class: 'mut' }, 'Learns what normal SSH login behaviour looks like in 5-minute windows and flags deviations. Alerts come with a risk tier, transparent rules and the evidence behind them.'));
  if (!s.ready) {
    const b = el('button', { class: 'primary' }, 'Open Pipeline and run "full_demo"'); b.onclick = () => go('pipeline');
    root.append(el('div', { class: 'card err' }, el('b', {}, 'No trained model found. '), 'Generate the demo data and train (about a minute). ', b));
    return root;
  }
  const g = el('div', { class: 'grid' });
  [['model', s.model_id], ['windows (dataset)', s.n_windows], ['features', s.n_features], ['grouped by', s.group_by + ', ' + s.window_seconds + 's'],
   ['stored detections', s.counts.detections], ['scored uploads', s.counts.events > 0 ? 'yes' : 'none yet']].forEach(([k, v]) => g.append(stat(k, v)));
  const demo = el('button', { class: 'primary' }, 'Score the demo incident day'); demo.onclick = () => go('score?sample=mixed_incident_day');
  const res = el('button', {}, 'See evaluation results'); res.onclick = () => go('results');
  root.append(g, el('div', { class: 'row' }, demo, res), el('h2', {}, 'Risk tiers'),
    table(['tier', 'model score at least', 'meaning'], TIERS.slice(1).map((t, i, all) => { const v = s.tier_thresholds[t];
      const up = v == null ? null : all.slice(i + 1).find(u => s.tier_thresholds[u] === v);   // a higher tier with the same cut-point wins by score
      const mean = { LOW: 'review: best F1 point on validation', MEDIUM: 'validation precision ≥ 75%', HIGH: 'validation precision ≥ 90%', CRITICAL: 'validation precision ≥ 97%' }[t];
      return [badge(t), v == null ? 'unreachable by score (rules can still raise it)' : f(v),
        up ? el('span', { class: 'mut' }, `${mean}. Same cut-point as ${up}, so scores from ${f(v)} up are ${up}; ${t} is reached only through a rule.`) : mean]; })),
    el('div', { class: 'card' }, el('b', {}, 'Rules that can raise a tier: '), chips(Object.keys(s.rule_floors).map(k => `${k} → at least ${s.rule_floors[k]}`))),
    el('div', { class: 'card mut' }, 'Trained on synthetic SSH logs with planted attack scenarios; treat the numbers as a demonstration of the pipeline, not as real-world accuracy. See Results for the held-out test and the caveats.'));
  return root;
}

// ---------------- Score logs ----------------
function renderScore(out, r) {
  out.replaceChildren();
  const g = el('div', { class: 'grid' });
  g.append(stat('windows scored', r.windows)); TIERS.forEach(t => g.append(el('div', { class: 'card stat' }, el('b', {}, r.by_tier[t]), badge(t))));
  out.append(el('h2', {}, 'Result: ' + r.label), g,
    el('div', { class: 'mut' }, `Parsed ${r.parse.event_lines} of ${r.parse.total_lines} lines as SSH events (coverage ${f(r.parse.coverage_sshd_lines, 2)} of sshd lines; ${r.parse.not_sshd_lines} non-sshd, ${r.parse.malformed_lines} malformed, ${r.parse.unsupported_sshd_lines} unrecognised). Rules hit: ` +
      Object.entries(r.rules_hit).map(([k, v]) => `${k}=${v}`).join(', ') + '.'));
  if (!r.rows.length) { out.append(el('div', { class: 'card' }, 'No window reached the LOW threshold: nothing flagged.')); return; }
  out.append(el('h2', {}, `Flagged windows (${r.shown}, highest score first). Click a row for evidence and the raw log lines.`));
  const detail = {};
  out.append(table(['entity', 'window start (UTC)', 'score', 'tier', 'rules', 'top evidence'],
    r.rows.map(x => [x.entity, x.window_start, f(x.score), badge(x.tier), chips(x.rules), x.evidence[0] || '']),
    async (i, tr) => {
      if (detail[i]) { detail[i].remove(); detail[i] = null; return; }
      const x = r.rows[i], box = el('td', { colSpan: 6 }, el('div', { class: 'mut' }, 'Loading…')); detail[i] = el('tr', {}, box); tr.after(detail[i]);
      try {
        const w = await api(`/api/window?run=${r.run_id}&entity=${encodeURIComponent(x.entity)}&start=${encodeURIComponent(x.window_start)}`);
        const kv = el('div', { class: 'kv' }); Object.entries(x.features).forEach(([k, v]) => kv.append(el('span', {}, k), el('span', {}, v)));
        box.replaceChildren(el('b', {}, 'Why this was flagged (feature deviations, not causal attribution)'), el('ul', {}, ...x.evidence.map(e => el('li', {}, e))),
          el('b', {}, `Log lines in this window (${w.events.length})`), el('pre', {}, w.events.map(e => e.raw_line).join('\n') || '(none)'), el('b', {}, 'Features'), kv);
      } catch (e) { box.replaceChildren(errBox(e)); }
    }));
}

async function score(params) {
  const root = el('div'), samples = (await api('/api/samples')).samples, out = el('div');
  const ta = el('textarea', { rows: 7, placeholder: 'Paste OpenSSH auth.log lines here, or pick a file / a demo log…' });
  const year = el('input', { type: 'number', placeholder: 'year (optional)', style: 'width:130px' });
  const btn = el('button', { class: 'primary' }, 'Score pasted log');
  async function run(body) {
    out.replaceChildren(el('div', { class: 'mut' }, 'Scoring…')); btn.disabled = true;
    if (year.value) body.year = Number(year.value);
    try { renderScore(out, await api('/api/score', body)); } catch (e) { out.replaceChildren(errBox(e)); } finally { btn.disabled = false; }
  }
  btn.onclick = () => run({ text: ta.value, name: 'pasted.log' });
  const file = el('input', { type: 'file', accept: '.log,.txt,text/plain' });
  file.onchange = async () => { const fl = file.files[0]; if (!fl) return; ta.value = await fl.text(); ta.dataset.name = fl.name; };
  const sb = el('div', { class: 'row' }, ...samples.map(s => { const b = el('button', { title: s.description }, s.name.replace(/_/g, ' ')); b.onclick = () => run({ sample: s.name }); return b; }));
  root.append(el('h1', {}, 'Score logs'), el('p', { class: 'mut' }, 'Upload or paste an SSH log, or try a demo log built from the trained users. Nothing leaves this machine.'),
    el('div', { class: 'card' }, el('b', {}, 'Demo logs'), sb, el('div', { class: 'mut' }, samples.map(s => `${s.name.replace(/_/g, ' ')}: ${s.description}`).join('  ·  '))),
    el('div', { class: 'card' }, el('b', {}, 'Your log'), el('div', { class: 'row' }, file, year), ta, el('div', { class: 'row' }, btn)), out);
  if (params.sample) run({ sample: params.sample });
  return root;
}

// ---------------- Results ----------------
async function results() {
  const r = await api('/api/results'), root = el('div');
  root.append(el('h1', {}, 'Results'));
  if (!r.models && !r.scoring) { root.append(el('div', { class: 'card err' }, 'No results yet. Run the pipeline first.')); return root; }
  if (r.models) {
    const rows = Object.entries(r.models.models).map(([k, v]) => { const m = v.metrics, c = v.bootstrap_ci; return [k, f(m.precision), f(m.recall), f(m.f1), `${f(m.pr_auc)} [${f(c.pr_auc.low)}, ${f(c.pr_auc.high)}]`, f(m.false_positive_rate), f(m.alerts_per_day, 1)]; });
    const s = r.models.iforest_seed_stats; if (s) rows.push([`iforest mean of ${s.n_seeds} seeds`, f(s.precision.mean), f(s.recall.mean), f(s.f1.mean), `${f(s.pr_auc.mean)} ± ${f(s.pr_auc.std)}`, f(s.false_positive_rate.mean), f(s.alerts_per_day.mean, 1)]);
    root.append(el('h2', {}, 'Model comparison (held-out test)'), table(['model', 'precision', 'recall', 'F1', 'PR-AUC [95% CI]', 'false-positive rate', 'alerts/day'], rows),
      el('div', { class: 'mut' }, `Test split: ${r.models.class_balance.test.windows} windows, ${r.models.class_balance.test.attack} attack windows. Each threshold was calibrated on the validation split.`));
  }
  if (r.scoring) {
    for (const sp of ['test', 'val']) { const s = r.scoring.splits[sp];
      root.append(el('h2', {}, `Model only vs model + rules — ${sp} (${s.role})`),
        table(['alert decision', 'precision', 'recall', 'F1', 'false-positive rate', 'alerts'], Object.entries(s.configs).map(([k, m]) => [k.replace(/_/g, ' '), f(m.precision), f(m.recall), f(m.f1), f(m.false_positive_rate), m.n_alerts])),
        table(['final tier', 'windows', 'attacks', 'attack share'], Object.entries(s.tiers).map(([t, v]) => [badge(t), v.windows, v.attack_windows, f(v.attack_rate)]))); }
  }
  if (r.ablation) root.append(el('h2', {}, 'Ablation: drop one feature family (change in test PR-AUC)'),
    table(['family dropped', 'Isolation Forest', 'One-Class SVM', 'LOF'], r.ablation.variants.filter(v => v.mode === 'drop').map(v => [v.family, ...['iforest', 'ocsvm', 'lof'].map(m => { const d = v.delta_vs_all[m]; return d ? `${d.delta_pr_auc >= 0 ? '+' : ''}${f(d.delta_pr_auc)} [${f(d.delta_pr_auc_ci[0])}, ${f(d.delta_pr_auc_ci[1])}] ${d.verdict_pr_auc}` : '–'; })])),
    el('div', { class: 'mut' }, r.ablation.caveats.join(' ')));
  if (r.h1) root.append(el('h2', {}, 'H1: per-user vs global baseline'),
    table(['model', 'PR-AUC per-user', 'PR-AUC global', 'delta [95% CI]', 'verdict'], Object.entries(r.h1.models).filter(([m]) => m !== 'rule').map(([m, e]) => [m, f(e.per_user.pr_auc), f(e.global.pr_auc), `${f(e.delta.delta_pr_auc)} [${f(e.delta.delta_pr_auc_ci[0])}, ${f(e.delta.delta_pr_auc_ci[1])}]`, e.delta.verdict_pr_auc])),
    el('div', { class: 'mut' }, 'Features that differ between the two arms: ' + (r.h1.features_that_differ.join(', ') || 'none') + '. ' + r.h1.caveats.join(' ')));
  const cav = (r.scoring && r.scoring.caveats) || []; if (cav.length) root.append(el('div', { class: 'card' }, el('b', {}, 'Caveats'), el('ul', {}, ...cav.map(c => el('li', {}, c)))));
  if (r.figures.length) { const g = el('div', { class: 'figs' }); r.figures.forEach(x => g.append(el('div', {}, el('img', { src: x.url, alt: x.title }), el('div', { class: 'mut' }, x.title)))); root.append(el('h2', {}, 'Figures'), g); }
  return root;
}

// ---------------- Detections explorer ----------------
async function detections() {
  const root = el('div'), out = el('div'), S = { tier: 'FLAGGED', split: '', q: '', missed: '', offset: 0 };
  const sel = (label, key, opts) => { const s = el('select'); opts.forEach(([v, t]) => s.append(el('option', { value: v }, t))); s.onchange = () => { S[key] = s.value; S.offset = 0; load(); }; return el('label', {}, label + ' ', s); };
  const q = el('input', { placeholder: 'entity contains…' }); q.oninput = () => { S.q = q.value; S.offset = 0; clearTimeout(q.t); q.t = setTimeout(load, 250); };
  const mc = el('input', { type: 'checkbox' }); mc.onchange = () => { S.missed = mc.checked ? '1' : ''; S.offset = 0; load(); };
  root.append(el('h1', {}, 'Detections explorer'), el('p', { class: 'mut' }, 'All windows of the evaluated dataset with their tier, rules and (for labelled data) the true label. Use "missed attacks" to see what the system did not flag.'),
    el('div', { class: 'row' }, sel('tier', 'tier', [['FLAGGED', 'flagged (LOW+)'], ['ANY', 'all'], ...TIERS.slice(1).concat('NORMAL').map(t => [t, t])]),
      sel('split', 'split', [['', 'any'], ['train', 'train'], ['val', 'validation'], ['test', 'test']]), q, el('label', {}, mc, ' missed attacks only')), out);
  async function load() {
    out.replaceChildren(el('div', { class: 'mut' }, 'Loading…'));
    try {
      const d = await api('/api/detections?' + new URLSearchParams({ tier: S.tier, split: S.split, q: S.q, missed: S.missed, limit: 100, offset: S.offset }));
      out.replaceChildren(el('div', { class: 'mut' }, `${d.total} matching windows (run ${d.run_id}); showing ${S.offset + 1}–${Math.min(S.offset + 100, d.total)}`));
      const open = {};
      out.append(table(['entity', 'window start', 'split', 'score', 'tier', 'rules', 'label', 'scenario'], d.rows.map(x => [x.entity, x.window_start, x.split, f(x.score), badge(x.tier), chips(x.rules), x.label == null ? '?' : x.label, x.attack_scenario || '']),
        (i, tr) => { if (open[i]) { open[i].remove(); open[i] = null; return; } const x = d.rows[i], kv = el('div', { class: 'kv' });
          Object.entries(x.features).forEach(([k, v]) => kv.append(el('span', {}, k), el('span', {}, v)));
          open[i] = el('tr', {}, el('td', { colSpan: 8 }, el('b', {}, 'Evidence'), el('ul', {}, ...(x.evidence.length ? x.evidence : ['none flagged']).map(e => el('li', {}, e))), el('b', {}, 'Features'), kv)); tr.after(open[i]); }));
      const prev = el('button', {}, '← previous'), next = el('button', {}, 'next →');
      prev.disabled = S.offset === 0; next.disabled = S.offset + 100 >= d.total;
      prev.onclick = () => { S.offset = Math.max(0, S.offset - 100); load(); }; next.onclick = () => { S.offset += 100; load(); };
      out.append(el('div', { class: 'row' }, prev, next));
    } catch (e) { out.replaceChildren(errBox(e)); }
  }
  load(); return root;
}

// ---------------- Pipeline / jobs ----------------
async function pipeline() {
  const j = await api('/api/jobs'), root = el('div'), log = el('pre', {}, 'Pick a job to run. Only one job runs at a time.'), status = el('div', { class: 'mut' });
  let cur = null, since = 0;
  async function poll() {
    if (cur == null) return;
    try { const r = await api(`/api/jobs/${cur}?since=${since}`); since = r.next; if (r.lines.length) { log.append(r.lines.join('\n') + '\n'); log.scrollTop = log.scrollHeight; }
      status.textContent = `job ${r.job_id} (${r.name}): ${r.status}${r.exit_code != null ? ' (exit ' + r.exit_code + ')' : ''}`;
      if (r.status !== 'running') { clearInterval(timer); timer = null; cancel.disabled = true; refreshButtons(false); }
    } catch (e) { status.textContent = 'poll failed: ' + e.message; clearInterval(timer); timer = null; }
  }
  const cancel = el('button', {}, 'Cancel running job'); cancel.disabled = !j.busy;
  cancel.onclick = async () => { try { await api(`/api/jobs/${cur}/cancel`, {}); } catch (e) { status.textContent = e.message; } };
  const btns = [];
  const refreshButtons = busy => btns.forEach(b => { b.disabled = busy; });
  async function start(name) {
    try { const r = await api('/api/jobs', { name }); cur = r.job_id; since = 0; log.textContent = ''; cancel.disabled = false; refreshButtons(true);
      timer = setInterval(poll, 800); poll(); } catch (e) { status.textContent = e.message; }
  }
  const row = el('div', { class: 'row' });
  j.catalog.forEach(c => { const b = el('button', { class: c.name === 'full_demo' ? 'primary' : '', title: c.description }, c.name); b.onclick = () => start(c.name); btns.push(b); row.append(b); });
  refreshButtons(j.busy);
  root.append(el('h1', {}, 'Pipeline'), el('p', { class: 'mut' }, 'Runs the project scripts (fixed allowlist) and streams their output. "full_demo" does everything end to end; the individual steps are for re-running one stage.'),
    row, el('div', { class: 'row' }, cancel, status), log, el('h2', {}, 'History'),
    table(['id', 'job', 'status', 'started', 'finished', 'exit'], j.history.map(h => [h.job_id, h.name, h.status, h.started_at, h.finished_at || '', h.exit_code == null ? '' : h.exit_code]),
      async i => { const h = j.history[i]; const r = await api(`/api/jobs/${h.job_id}`); log.textContent = r.lines.join('\n'); status.textContent = `job ${h.job_id} (${h.name}): ${h.status}`; }));
  return root;
}

// ---------------- Real logs (unlabeled) ----------------
async function real() {
  const l = await api('/api/realdata'), root = el('div'), out = el('div');
  root.append(el('h1', {}, 'Real logs (unlabeled)'), el('p', { class: 'mut' }, 'Analysis of real logs such as Loghub OpenSSH. There are no labels, so this page shows score distributions, flagged windows and sanity checks, never precision or recall.'));
  if (!l.datasets.length) {
    const b = el('button', { class: 'primary' }, 'Open Pipeline'); b.onclick = () => go('pipeline');
    root.append(el('div', { class: 'card' }, 'No real-log analysis yet. In the Pipeline tab run fetch_loghub (needs internet) or place the log at the path under real.log_path in config.yaml, optionally fetch_blocklist, then analyze_real. ', b));
    return root;
  }
  const sel = el('select'); l.datasets.forEach(n => sel.append(el('option', { value: n }, n))); sel.value = l.datasets[0];
  async function show() {
    out.replaceChildren(el('div', { class: 'mut' }, 'Loading…'));
    try {
      const d = await api('/api/realdata/' + encodeURIComponent(sel.value)), g = el('div', { class: 'grid' });
      [['windows', d.n_windows], ['grouped by', d.group_by], ['sshd coverage', f(d.parse.coverage_sshd_lines, 3)], ['event lines', d.parse.event_lines], ['unattributed events', d.unattributed_events]].forEach(([k, v]) => g.append(stat(k, v)));
      out.replaceChildren(el('div', { class: 'card' }, el('b', {}, d.labels)), g,
        el('h2', {}, 'Windows per tier'), table(['split', ...TIERS], Object.entries(d.tiers).map(([sp, c]) => [sp, ...TIERS.map(t => c[t])])),
        el('div', { class: 'mut' }, d.tier_basis),
        el('h2', {}, 'Score distribution (quantiles)'), table(Object.keys(d.score_distribution.quantiles), [Object.values(d.score_distribution.quantiles).map(v => f(v))]),
        d.figure ? el('img', { src: d.figure, alt: 'score distribution', style: 'max-width:100%;background:#fff;border-radius:8px' }) : null,
        el('h2', {}, 'Rules'), el('div', {}, chips(Object.entries(d.rules.hits_total).map(([k, v]) => `${k}: ${v}`)), ' ', d.rules.unavailable.join('; ')),
        el('h2', {}, 'Blocklist overlap'), el('pre', {}, JSON.stringify(d.blocklist_overlap, null, 2)));
      if (d.weak_proxy) out.append(el('h2', {}, 'Weak proxy check'), el('div', { class: 'card err' }, d.weak_proxy.warning),
        el('div', {}, `${d.weak_proxy.definition}: ${d.weak_proxy.proxy_windows} windows; ROC-AUC of score vs proxy = ${f(d.weak_proxy.roc_auc_of_score_vs_proxy)}`),
        table(['tier', 'windows', 'proxy rate'], Object.entries(d.weak_proxy.proxy_rate_by_tier).map(([t, v]) => [badge(t), v.windows, f(v.proxy_rate)])));
      const cd = d.cross_domain;
      out.append(el('h2', {}, 'Cross-domain check (synthetic-trained model on this log)'), cd.available ?
        el('div', {}, el('div', { class: 'card' }, cd.limitation), el('div', {}, `${cd.windows} windows, ${f(cd.share_flagged * 100, 1)}% flagged. Tiers: ` + Object.entries(cd.tiers).map(([k, v]) => `${k} ${v}`).join(', ')),
          table(['feature', 'real median', 'synthetic normal median', 'shift (robust units)'], cd.most_shifted_features.map(x => [x.feature, f(x.real_median), f(x.synthetic_normal_median), f(x.shift_in_robust_units, 1)]))) : el('div', { class: 'mut' }, cd.reason));
      const open = {};
      out.append(el('h2', {}, 'Top flagged windows (click for evidence)'), table(['entity', 'window start', 'score', 'tier', 'rules', 'top evidence'],
        d.top_windows.map(x => [x.entity, x.window_start, f(x.score), badge(x.tier), chips(x.rules), x.evidence[0] || '']),
        (i, tr) => { if (open[i]) { open[i].remove(); open[i] = null; return; } open[i] = el('tr', {}, el('td', { colSpan: 6 }, el('ul', {}, ...(d.top_windows[i].evidence.length ? d.top_windows[i].evidence : ['none']).map(e => el('li', {}, e))))); tr.after(open[i]); }),
        el('div', { class: 'card' }, el('b', {}, 'Caveats'), el('ul', {}, ...d.caveats.map(c => el('li', {}, c)))));
    } catch (e) { out.replaceChildren(errBox(e)); }
  }
  sel.onchange = show; root.append(el('div', { class: 'row' }, 'dataset ', sel), out); show(); return root;
}

// ---------------- Settings / maintenance ----------------
async function settings() {
  const [s, m] = await Promise.all([api('/api/settings'), api('/api/maintenance')]), root = el('div'), msg = el('div', { class: 'mut' });
  const purge = el('button', {}, 'Delete stored uploads and their events');
  purge.onclick = async () => { if (!confirm('Delete all scored uploads from the database?')) return; try { const r = await api('/api/maintenance/purge', {}); msg.textContent = `deleted ${r.deleted_runs} run(s)`; } catch (e) { msg.textContent = e.message; } };
  root.append(el('h1', {}, 'Settings and maintenance'),
    el('div', { class: 'card' }, el('b', {}, 'Scoring configuration (read-only; edit config/thresholds.yaml and re-run calibration)'),
      el('pre', {}, JSON.stringify(s.scoring, null, 2))),
    el('div', { class: 'card' }, el('b', {}, 'Limits'), el('div', {}, `max upload ${s.limits.max_upload_bytes} bytes; at most ${s.limits.max_rows_returned} flagged windows returned per scoring request`)),
    el('div', { class: 'card' }, el('b', {}, 'Security posture'), el('ul', {}, ...s.security.map(x => el('li', {}, x)))),
    el('h2', {}, 'Database'), el('div', { class: 'mut' }, m.db_path),
    table(['table', 'rows'], Object.entries(m.counts)),
    el('div', { class: 'row' }, purge, msg),
    el('h2', {}, 'Recent runs'), table(['run', 'kind', 'dataset', 'model', 'created'], m.runs.map(r => [r.run_id, r.kind, r.dataset, r.model_id, r.created_at])));
  return root;
}

// ---------------- Router ----------------
const PAGES = { home, score, results, detections, real, pipeline, settings };
const NAV = [['home', 'Home'], ['score', 'Score logs'], ['results', 'Results'], ['detections', 'Detections'], ['real', 'Real logs'], ['pipeline', 'Pipeline'], ['settings', 'Settings']];
const links = {};
NAV.forEach(([k, t]) => { links[k] = el('a', { href: '#' + k }, t); $nav.append(links[k]); });
async function route() {
  if (timer) { clearInterval(timer); timer = null; }
  const [name, qs] = (location.hash || '#home').slice(1).split('?'), page = PAGES[name] ? name : 'home';
  Object.entries(links).forEach(([k, a]) => { a.className = k === page ? 'on' : ''; });
  const params = Object.fromEntries(new URLSearchParams(qs || ''));
  $view.replaceChildren(el('div', { class: 'mut' }, 'Loading…'));
  try { $view.replaceChildren(await PAGES[page](params)); } catch (e) { $view.replaceChildren(errBox(e)); }
}
if (typeof window !== 'undefined') window.addEventListener('hashchange', route);
globalThis.__logad = { route, PAGES };
route();
