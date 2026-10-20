// Runs the real static/app.js against a live server with a minimal DOM shim.
// usage: node ui_shim.js BASE_URL APP_JS [route]   env LOG_TEXT = text to paste and score
const fs = require('fs');
const [,, BASE, APP, ROUTE] = process.argv;
class Node { constructor(){ this.children = []; this.parent = null; }
  append(...k){ for (const c of k){ const n = typeof c === 'string' ? new Text(c) : c; n.parent = this; this.children.push(n);} }
  replaceChildren(...k){ this.children = []; this.append(...k); }
  remove(){ if (this.parent) this.parent.children = this.parent.children.filter(c => c !== this); }
  after(n){ const p = this.parent, i = p.children.indexOf(this); n.parent = p; p.children.splice(i + 1, 0, n); } }
class Text extends Node { constructor(t){ super(); this.text = String(t); } }
class Element extends Node { constructor(tag){ super(); this.tag = tag; this.className = ''; this.dataset = {}; }
  set innerHTML(v){ throw new Error('innerHTML is forbidden'); } set outerHTML(v){ throw new Error('outerHTML is forbidden'); }
  set textContent(v){ this.children = [new Text(v)]; } get textContent(){ return all(this).filter(n => n instanceof Text).map(n => n.text).join(''); } }
const all = n => [n, ...n.children.flatMap(all)];
const byId = { view: new Element('main'), nav: new Element('nav') };
global.Node = Node; global.window = { addEventListener(){} }; global.location = { hash: ROUTE || '' }; global.confirm = () => true;
global.document = { getElementById: id => byId[id], createElement: t => new Element(t), createTextNode: t => new Text(t) };
const realFetch = global.fetch; global.fetch = (p, o) => realFetch(BASE + p, o);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const texts = () => all(byId.view).filter(n => n instanceof Text).map(n => n.text);
const has = s => texts().some(t => t.includes(s));
async function until(pred, what, ms = 15000) { const t0 = Date.now(); while (!pred()) { if (Date.now() - t0 > ms) throw new Error('timeout waiting for ' + what + '; saw: ' + texts().slice(0, 40).join(' | ')); await sleep(50); } }
(async () => {
  (0, eval)(fs.readFileSync(APP, 'utf8'));
  const out = { steps: [] };
  const L = globalThis.__logad;
  await until(() => ROUTE ? has('Score logs') : has('SSH login anomaly detection'), 'first page'); out.steps.push('home');
  if (!ROUTE) {
    for (const r of ['results', 'detections', 'real', 'ip', 'settings', 'pipeline']) { location.hash = '#' + r; await L.route(); await sleep(250); if (r === 'real') { await until(() => has('windows per tier') || has('Windows per tier') || has('Error:'), 'real page'); out.realOk = has('Windows per tier') && !has('Error:'); out.realNoPR = !texts().some(x => /^(precision|recall|F1)$/i.test(x.trim())); } out.steps.push(r); }
    location.hash = '#score'; await L.route();
  }
  const view = () => all(byId.view);
  if (process.env.LOG_TEXT) {
    const ta = view().find(n => n.tag === 'textarea'); ta.value = process.env.LOG_TEXT;
    view().find(n => n.tag === 'button' && n.children.some(c => c.text === 'Score pasted log')).onclick();
  } else if (!ROUTE) {
    view().find(n => n.tag === 'button' && n.children.some(c => c.text === 'mixed incident day')).onclick();
  }
  await until(() => has('Flagged windows') || has('nothing flagged') || has('Error:'), 'score result');
  out.flagged = has('Flagged windows');
  const rows = view().filter(n => n.tag === 'tr' && n.className === 'click' && n.onclick);
  if (rows.length) { await rows[0].onclick(); await until(() => has('Log lines in this window') || has('Error:'), 'detail'); out.detail = has('Log lines in this window'); }
  out.texts = texts(); out.imgTags = view().filter(n => n.tag === 'img').length; out.scriptTags = view().filter(n => n.tag === 'script').length;
  out.hasError = has('Error:');
  console.log(JSON.stringify(out)); process.exit(0);
})().catch(e => { console.error(String(e.stack || e)); process.exit(1); });
