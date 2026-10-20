"""Static, self-contained HTML dashboard built from the SQLite DB and report files.

No server, no build step, no external requests: data is embedded as escaped JSON and rendered with
textContent only (log-derived text can never become markup). A restrictive CSP is set as defence in depth.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from logad.config import PROJECT_ROOT, resolve_path
from logad.io import load_json
from logad.storage.db import Database

MAX_ROWS = 3000
FIGS = [("pr_curves.png", "all", "Precision-recall, held-out test"),
        ("scoring.png", "all", "Risk tiers and model + rules (test)"),
        ("ablation.png", "ablation", "Feature-family ablation"),
        ("h1.png", "h1", "H1: per-user vs global baseline")]


def _b64(path: Path):
    return base64.b64encode(path.read_bytes()).decode() if path.exists() else None


def _safe_json(obj) -> str:
    s = json.dumps(obj, default=str, allow_nan=False if False else True)
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def collect(cfg: dict, dataset: str, tag: str, reports_root=None, db_path=None) -> dict:
    root = Path(reports_root) if reports_root else PROJECT_ROOT / "reports"
    m, f = root / "metrics" / dataset, root / "figures" / dataset
    data: dict = {"dataset": dataset, "tag": tag}
    for key, p in (("models", m / tag / "metrics.json"), ("scoring", m / tag / "scoring.json"),
                   ("ablation", m / "ablation" / "ablation.json"), ("h1", m / "h1" / "h1.json")):
        data[key] = load_json(p) if p.exists() else None
    data["figures"] = [{"title": t, "png": _b64(f / sub / name)} for name, sub, t in FIGS if (f / sub / name).exists()]
    with Database(db_path or resolve_path(cfg["paths"]["db_path"])) as db:
        run = db.latest_run("replay_dataset", dataset)
        if run is None:
            raise ValueError(f"no replay run for dataset {dataset!r} in the database; run ingest first")
        info = db.conn.execute("SELECT feature_names FROM datasets WHERE dataset=?", (dataset,)).fetchone()
        feats = json.loads(info["feature_names"])
        rows = db.detections(run, None, limit=10**7)
        keep = [r for r in rows if r["tier"] != "NORMAL" or r["label"] == 1][:MAX_ROWS]
        wf = db.window_features_for(dataset, [(r["entity"], r["window_start"]) for r in keep], feats)
        runrow = dict(db.conn.execute("SELECT * FROM runs WHERE run_id=?", (run,)).fetchone())
        data["run"] = {k: runrow[k] for k in ("run_id", "model_id", "model_version", "created_at")}
        data["counts"] = {"windows": len(rows), "shown": len(keep),
                          "by_tier": {t: sum(r["tier"] == t for r in rows) for t in ("NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL")}}
        data["detections"] = [{
            "entity": r["entity"], "window_start": r["window_start"], "split": r["split"], "label": r["label"],
            "scenario": r["attack_scenario"], "score": round(r["score"], 4), "model_tier": r["model_tier"],
            "tier": r["tier"], "rules": json.loads(r["rules"] or "[]"), "evidence": json.loads(r["evidence"] or "[]"),
            "features": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in wf.get((r["entity"], r["window_start"]), {}).items()}} for r in keep]
    return data


def build_dashboard(cfg: dict, dataset: str = "synthetic", tag: str = "all", out_path=None,
                    reports_root=None, db_path=None) -> str:
    data = collect(cfg, dataset, tag, reports_root, db_path)
    out = Path(out_path) if out_path else resolve_path(cfg["paths"]["dashboard_dir"]) / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.replace("__DATA__", _safe_json(data)), encoding="utf-8")
    return str(out)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>logad dashboard</title>
<style>
:root{--bg:#fff;--fg:#1c2330;--mut:#5b6575;--card:#f4f6f9;--line:#d8dde5;--acc:#2b6cb0;
--low:#b8860b;--med:#d9730d;--high:#c53030;--crit:#7b2cbf;--ok:#2f855a}
@media (prefers-color-scheme:dark){:root{--bg:#12161d;--fg:#e4e8ee;--mut:#9aa5b5;--card:#1b212b;--line:#2d3644;--acc:#63a4ff;
--low:#e0b84a;--med:#f0963c;--high:#f06868;--crit:#c58bff;--ok:#56c28a}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
main{max-width:1180px;margin:0 auto;padding:20px 16px 60px}h1{margin:0 0 4px;font-size:22px}h2{margin:30px 0 8px;font-size:17px}
.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:8px 0}
.grid{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}.stat b{display:block;font-size:20px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{color:var(--mut);font-weight:600;cursor:default}.scroll{overflow-x:auto}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600;color:#fff}
.NORMAL{background:var(--ok)}.LOW{background:var(--low)}.MEDIUM{background:var(--med)}.HIGH{background:var(--high)}.CRITICAL{background:var(--crit)}
.figs{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}.figs img{width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
.ctl{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}select,input,button{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
tr.row{cursor:pointer}tr.row:hover{background:var(--card)}ul{margin:4px 0 4px 18px;padding:0}.miss{border-left:3px solid var(--high)}
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:2px 14px;font-size:12px}.kv span:nth-child(odd){color:var(--mut)}
.v-supported,.v-dropping_helps,.v-only_better{color:var(--ok);font-weight:600}.v-reversed,.v-dropping_hurts,.v-only_worse{color:var(--high);font-weight:600}
</style></head><body><main>
<h1>SSH login anomaly detection</h1><div class="mut" id="sub"></div>
<div id="root"></div></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const el=(t,p={},...k)=>{const e=document.createElement(t);for(const[a,v]of Object.entries(p)){if(a==='class')e.className=v;else e[a]=v}
  for(const c of k){if(c==null)continue;e.append(c instanceof Node?c:document.createTextNode(String(c)))}return e};
const f=(x,d=3)=>x==null?'–':(typeof x==='number'?x.toFixed(d):String(x));
const badge=t=>el('span',{class:'badge '+t},t);
function table(head,rows){const t=el('table'),h=el('tr');head.forEach(x=>h.append(el('th',{},x)));t.append(el('thead',{},h));
  const b=el('tbody');rows.forEach(r=>{const tr=el('tr');r.forEach(c=>tr.append(el('td',{},c)));b.append(tr)});t.append(b);return el('div',{class:'scroll'},t)}
const root=document.getElementById('root');
const sec=(title,...k)=>{root.append(el('h2',{},title));k.forEach(x=>root.append(x))};
const ci=d=>d?`${d.delta_pr_auc>=0?'+':''}${f(d.delta_pr_auc)}  [${f(d.delta_pr_auc_ci[0])}, ${f(d.delta_pr_auc_ci[1])}]`:'–';
const verdict=d=>d?el('span',{class:'v-'+d.verdict_pr_auc},d.verdict_pr_auc):'–';

document.getElementById('sub').textContent=`dataset ${D.dataset} · model ${D.run.model_id} (${D.run.model_version}) · run ${D.run.run_id}`;

// overview
const bt=D.counts.by_tier, thr=D.scoring?D.scoring.tier_thresholds:{};
const stats=el('div',{class:'grid'});
[['windows',D.counts.windows],...Object.entries(bt).map(([k,v])=>[k,v])].forEach(([k,v])=>stats.append(el('div',{class:'card stat'},el('b',{},v),k)));
sec('Overview',stats,el('div',{class:'mut'},'Tier cut-points on the model score: '+Object.entries(thr).map(([k,v])=>`${k} ≥ ${v==null?'unreachable':f(v)}`).join(' · ')));
if(D.scoring){const c=el('div',{class:'card'});c.append(el('b',{},'Read this first'));const u=el('ul');D.scoring.caveats.forEach(x=>u.append(el('li',{},x)));c.append(u);root.append(c)}

// models
if(D.models){const rows=Object.entries(D.models.models).map(([k,v])=>{const m=v.metrics,ci=v.bootstrap_ci;return[k,f(m.precision),f(m.recall),f(m.f1),
  `${f(m.pr_auc)} [${f(ci.pr_auc.low)}, ${f(ci.pr_auc.high)}]`,f(m.false_positive_rate),f(m.alerts_per_day,1)]});
  const s=D.models.iforest_seed_stats;
  if(s)rows.push([`iforest mean over ${s.n_seeds} seeds`,f(s.precision.mean),f(s.recall.mean),f(s.f1.mean),`${f(s.pr_auc.mean)} ± ${f(s.pr_auc.std)}`,f(s.false_positive_rate.mean),f(s.alerts_per_day.mean,1)]);
  sec('Models (held-out test)',table(['model','precision','recall','F1','PR-AUC [95% CI]','FPR','alerts/day'],rows),
    el('div',{class:'mut'},`Test: ${D.models.class_balance.test.windows} windows, ${D.models.class_balance.test.attack} attacks. Thresholds calibrated on validation.`))}

// scoring
if(D.scoring){for(const sp of ['test','val']){const s=D.scoring.splits[sp];
  sec(`Model only vs model + rules (${sp}: ${s.role})`,table(['decision','precision','recall','F1','FPR','alerts'],
    Object.entries(s.configs).map(([k,m])=>[k,f(m.precision),f(m.recall),f(m.f1),f(m.false_positive_rate),m.n_alerts])),
    table(['final tier','windows','attacks','attack rate'],Object.entries(s.tiers).map(([t,v])=>[badge(t),v.windows,v.attack_windows,f(v.attack_rate)])),
    table(['rule','hits','attack hits','precision','recall'],Object.entries(s.rules).map(([r,v])=>[r,v.hits,v.attack_hits,f(v.precision),f(v.recall)])))}}

// ablation + H1
if(D.ablation){const rows=D.ablation.variants.filter(v=>v.mode==='drop').map(v=>[v.family,...['iforest','ocsvm','lof'].map(m=>{const d=v.delta_vs_all[m];return d?el('span',{},ci(d)+' ',verdict(d)):'–'})]);
  sec('Ablation: drop one feature family (Δ test PR-AUC vs all features)',table(['family dropped','Isolation Forest','One-Class SVM','LOF'],rows),
    el('div',{class:'mut'},D.ablation.caveats.join(' ')))}
if(D.h1){const rows=Object.entries(D.h1.models).filter(([m])=>m!=='rule').map(([m,e])=>[m,f(e.per_user.pr_auc),f(e.global.pr_auc),ci(e.delta),verdict(e.delta)]);
  sec('H1: per-user vs global baseline',table(['model','PR-AUC per-user','PR-AUC global','Δ [95% CI]','verdict'],rows),
    el('div',{class:'mut'},`Features that differ between arms: ${D.h1.features_that_differ.join(', ')||'none'}. `+D.h1.caveats.join(' ')))}

// figures
if(D.figures.length){const g=el('div',{class:'figs'});D.figures.forEach(x=>g.append(el('div',{},el('img',{src:'data:image/png;base64,'+x.png,alt:x.title}),el('div',{class:'mut'},x.title))));sec('Figures',g)}

// explorer
const ex={tier:'ANY',split:'ANY',rule:'ANY',q:'',missed:false,sort:'score',shown:150};
const selOpts=(vals)=>vals.map(v=>el('option',{value:v},v));
const ctl=el('div',{class:'ctl'});
const mk=(label,key,vals)=>{const s=el('select');selOpts(vals).forEach(o=>s.append(o));s.onchange=()=>{ex[key]=s.value;ex.shown=150;draw()};ctl.append(el('label',{},label+' ',s))};
mk('tier','tier',['ANY','LOW','MEDIUM','HIGH','CRITICAL','NORMAL']);
mk('split','split',['ANY',...new Set(D.detections.map(d=>d.split).filter(Boolean))]);
mk('rule','rule',['ANY','R1','R2','R3','NONE']);
const q=el('input',{placeholder:'entity contains…'});q.oninput=()=>{ex.q=q.value.toLowerCase();ex.shown=150;draw()};ctl.append(q);
const mc=el('input',{type:'checkbox'});mc.onchange=()=>{ex.missed=mc.checked;ex.shown=150;draw()};ctl.append(el('label',{},mc,' missed attacks only (label=1, tier NORMAL)'));
const out=el('div'),more=el('button',{},'show more');more.onclick=()=>{ex.shown+=300;draw()};
function draw(){let r=D.detections.filter(d=>(ex.tier==='ANY'||d.tier===ex.tier)&&(ex.split==='ANY'||d.split===ex.split)&&
  (ex.rule==='ANY'||(ex.rule==='NONE'?d.rules.length===0:d.rules.includes(ex.rule)))&&(!ex.q||d.entity.toLowerCase().includes(ex.q))&&(!ex.missed||(d.label===1&&d.tier==='NORMAL')));
  out.replaceChildren();const t=el('table'),h=el('tr');['entity','window start (UTC)','split','score','model tier','tier','rules','label','scenario'].forEach(x=>h.append(el('th',{},x)));t.append(el('thead',{},h));
  const b=el('tbody');r.slice(0,ex.shown).forEach(d=>{const tr=el('tr',{class:'row'+(d.label===1&&d.tier==='NORMAL'?' miss':'')});
    [d.entity,d.window_start,d.split,f(d.score,4),badge(d.model_tier),badge(d.tier),d.rules.join(', '),d.label==null?'?':d.label,d.scenario||''].forEach(c=>tr.append(el('td',{},c)));
    let open=null;tr.onclick=()=>{if(open){open.remove();open=null;return}
      const kv=el('div',{class:'kv'});Object.entries(d.features).forEach(([k,v])=>{kv.append(el('span',{},k),el('span',{},v))});
      const ev=el('ul');(d.evidence.length?d.evidence:['no flagged evidence']).forEach(x=>ev.append(el('li',{},x)));
      open=el('tr',{},el('td',{colSpan:9},el('b',{},'Evidence (feature deviations, not causal attribution)'),ev,el('b',{},'Features'),kv));tr.after(open)};b.append(tr)});
  t.append(b);out.append(el('div',{class:'mut'},`${r.length} matching windows (showing ${Math.min(ex.shown,r.length)}; ${D.counts.shown} of ${D.counts.windows} windows are listed: flagged or missed attacks)`),el('div',{class:'scroll'},t));
  if(r.length>ex.shown)out.append(more)}
sec('Detections explorer',ctl,out);draw();
</script></body></html>
"""
