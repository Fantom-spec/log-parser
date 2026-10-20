// Minimal DOM shim: runs the dashboard script, clicks everything, and fails on any HTML-string injection path.
const fs = require('fs');
const [,, htmlPath] = process.argv;
const html = fs.readFileSync(htmlPath, 'utf8');
const data = html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];
class Node { constructor(){ this.children = []; this.parent = null; }
  append(...k){ for (const c of k){ const n = typeof c === 'string' ? new Text(c) : c; n.parent = this; this.children.push(n);} }
  replaceChildren(){ this.children = []; }
  remove(){ if (this.parent) this.parent.children = this.parent.children.filter(c => c !== this); }
  after(n){ const p = this.parent, i = p.children.indexOf(this); n.parent = p; p.children.splice(i + 1, 0, n); } }
class Text extends Node { constructor(t){ super(); this.text = String(t); } }
class Element extends Node { constructor(tag){ super(); this.tag = tag; this.className = ''; }
  set innerHTML(v){ throw new Error('innerHTML assignment is forbidden'); }
  set outerHTML(v){ throw new Error('outerHTML assignment is forbidden'); }
  set textContent(v){ this.children = [new Text(v)]; } }
global.Node = Node;
const byId = { data: Object.assign(new Element('script'), {}), root: new Element('main'), sub: new Element('div') };
Object.defineProperty(byId.data, 'textContent', { get: () => data });
global.document = { getElementById: id => byId[id], createElement: t => new Element(t), createTextNode: t => new Text(t) };
eval(code);
const all = [];
(function walk(n){ all.push(n); n.children.forEach(walk); })(byId.root);
const texts = () => { const out = []; (function w(n){ if (n instanceof Text) out.push(n.text); n.children && n.children.forEach(w); })(byId.root); return out; };
// exercise every interaction
all.filter(n => n.onclick).forEach(n => { n.onclick(); n.onclick(); });
all.filter(n => n.tag === 'select' && n.onchange).forEach(n => { n.value = 'CRITICAL'; n.onchange(); n.value = 'ANY'; n.onchange(); });
all.filter(n => n.tag === 'input' && n.oninput).forEach(n => { n.value = 'x'; n.oninput(); n.value = ''; n.oninput(); });
const imgs = []; (function w(n){ if (n.tag === 'img') imgs.push(n.src); n.children.forEach(w); })(byId.root);
console.log(JSON.stringify({ texts: texts(), imgs: imgs.map(s => s.slice(0, 22)), nodes: all.length }));
