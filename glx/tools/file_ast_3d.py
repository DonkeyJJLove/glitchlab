# glitchlab/glx/tools/file_ast_3d.py
from __future__ import annotations
"""
File AST 3D — interaktywny podgląd grafu jednego pliku .py w 3D (ForceGraph3D).

Użycie:
  python -m glx.tools.file_ast_3d --file glitchlab/core/demo_pkg/pipeline.py
  python -m glx.tools.file_ast_3d --file path/to/file.py --output .glx/graphs/this_file_3d.html

Cechy:
- Bez zależności poza standardową biblioteką (render w przeglądarce).
- Sterowanie w UI: layout (force/sphere/grid/radial/tower), 2D/3D, siła/odległość linków,
  rozmiar węzłów, widoczność etykiet, filtry typów, „show unresolved names”, wyszukiwanie (z hopami).
- Seedowane Z, by uniknąć „płaskiego” układu; można przełączyć na 2D.

Struktura węzłów:
- file:<abs_posix_path>   (etykieta: nazwa pliku)
- func:<qualname>         (etykieta: qualname())
- class:<qualname>        (etykieta: ClassName)
- module:<name>           (etykieta: nazwa modułu z importów)
- name:<callee>           (etykieta: nazwa wywołania, jeśli nie zmapowano do definicji)

Krawędzie:
- define: file → defs (func/class)
- call:   caller(def/file) → callee(name lub (opcjonalnie) dopasowany def)
- import: file → module
"""

import ast
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Model i ekstrakcja AST dla jednego pliku (bez zależności zewnętrznych)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    id: str
    kind: str
    label: str
    meta: Dict[str, object] = field(default_factory=dict)

@dataclass
class Edge:
    src: str
    dst: str
    kind: str

@dataclass
class Graph:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    version: str = "v1"
    meta: Dict[str, object] = field(default_factory=dict)

    def add_node(self, nid: str, kind: str, label: str, **meta) -> None:
        if nid not in self.nodes:
            self.nodes[nid] = Node(nid, kind, label, dict(meta))

    def add_edge(self, src: str, dst: str, kind: str) -> None:
        self.edges.append(Edge(src, dst, kind))


def _abs_posix(p: Path) -> str:
    return p.resolve().as_posix()


class _FileVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: List[str] = []
        self.defs: Dict[str, Tuple[str, int]] = {}  # qualname -> (kind, line)
        self.calls: List[Tuple[str, str, int]] = []  # (scope or <module>, callee, line)
        self.imports: List[str] = []

    def _q(self, name: str) -> str:
        return ".".join([*self.scope, name]) if self.scope else name

    def _cur(self) -> str:
        return ".".join(self.scope) if self.scope else "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = "method" if self.scope and self.scope[-1][:1].isupper() else "function"
        q = self._q(node.name)
        self.defs[q] = (kind, node.lineno)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        kind = "async_method" if self.scope and self.scope[-1][:1].isupper() else "async_function"
        q = self._q(node.name)
        self.defs[q] = (kind, node.lineno)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        q = self._q(node.name)
        self.defs[q] = ("class", node.lineno)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        callee = _attr_to_dotted(node.func) or (getattr(ast, "unparse", None) and ast.unparse(node.func)) or ""
        self.calls.append((self._cur(), callee, node.lineno))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self.imports.append(a.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.append(node.module.split(".")[0])
        else:
            for a in node.names:
                self.imports.append(a.name.split(".")[0])


def _attr_to_dotted(n: ast.AST) -> Optional[str]:
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        base = _attr_to_dotted(n.value)
        return f"{base}.{n.attr}" if base else n.attr
    if isinstance(n, ast.Call):
        return _attr_to_dotted(n.func)
    return None


def build_file_graph(py_path: Path) -> Graph:
    g = Graph()
    file_abs = _abs_posix(py_path)
    file_id = f"file:{file_abs}"
    g.add_node(file_id, "file", py_path.name, path=file_abs)

    try:
        src = py_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        src = py_path.read_text(encoding="latin-1", errors="replace")

    v = _FileVisitor()
    tree = ast.parse(src, filename=file_abs)
    v.visit(tree)

    # defs
    tail_to_qual: Dict[str, str] = {}
    for qual, (kind, line) in v.defs.items():
        if kind == "class":
            nid = f"class:{qual}"
            label = qual
        else:
            nid = f"func:{qual}"
            label = f"{qual}()"
        g.add_node(nid, "class" if kind == "class" else "func", label, file=file_abs, line=line)
        g.add_edge(file_id, nid, "define")
        tail = qual.split(".")[-1]
        if tail not in tail_to_qual:
            tail_to_qual[tail] = nid  # mapuj nazwy końcowe do node-id

    # imports
    for m in sorted(set(v.imports)):
        mid = f"module:{m}"
        g.add_node(mid, "module", m)
        g.add_edge(file_id, mid, "import")

    # calls (resolve do lokalnych defs po nazwie końcowej, w przeciwnym razie -> name:<callee>)
    for caller, callee, line in v.calls:
        if not callee:
            continue
        src_id = file_id
        if caller != "<module>":
            # próbuj dopasować do istniejącej definicji
            if caller in v.defs:
                src_id = f"{'class' if v.defs[caller][0]=='class' else 'func'}:{caller}"
            else:
                # jeżeli to w metodzie (np. Class.method), spróbuj dopasować
                parts = caller.split(".")
                for j in range(len(parts), 0, -1):
                    q = ".".join(parts[:j])
                    if q in v.defs:
                        src_id = f"{'class' if v.defs[q][0]=='class' else 'func'}:{q}"
                        break

        dst_id = tail_to_qual.get(callee.split(".")[-1])
        if not dst_id:
            dst_id = f"name:{callee}"
            g.add_node(dst_id, "name", callee)
        g.add_edge(src_id, dst_id, "call")

    g.meta = {
        "file": file_abs,
        "generated_by": "glx.tools.file_ast_3d",
        "nodes_count": len(g.nodes),
        "edges_count": len(g.edges),
    }
    return g


def graph_to_payload(g: Graph) -> Dict[str, object]:
    return {
        "version": g.version,
        "meta": g.meta,
        "nodes": [
            {"id": n.id, "kind": n.kind, "label": n.label, "meta": n.meta}
            for n in sorted(g.nodes.values(), key=lambda x: x.id)
        ],
        "edges": [{"src": e.src, "dst": e.dst, "kind": e.kind} for e in g.edges],
    }


# ──────────────────────────────────────────────────────────────────────────────
# HTML (UMD: Three + SpriteText + ForceGraph3D) z bogatym HUD do modyfikacji
# ──────────────────────────────────────────────────────────────────────────────

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>File AST 3D — __TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  :root { color-scheme: dark; }
  html, body { margin:0; padding:0; height:100%; background:#0b0e14; color:#e6e6e6; font: 14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, sans-serif; }
  #scene { width:100%; height:100%; }
  .hud {
    position: fixed; top: 8px; left: 8px; right: 8px;
    display:flex; flex-wrap:wrap; gap:8px; align-items:center;
    background: rgba(0,0,0,0.35); border:1px solid rgba(255,255,255,0.15); padding:8px 10px; border-radius:10px;
    backdrop-filter: blur(4px); z-index:10;
  }
  .hud .group { display:flex; gap:6px; align-items:center; background:#141826; padding:6px 8px; border-radius:8px; border:1px solid #232b43; }
  .hud label { opacity: 0.9; }
  .hud .badge { padding:2px 8px; border-radius:8px; background:#1e2430; border:1px solid #2b3242; white-space:nowrap; }
  .hud input[type="range"]{ width:120px; }
  .hud select, .hud input[type="text"], .hud button {
    background:#1a1f2b; border:1px solid #2b3242; color:#e6e6e6; border-radius:6px; padding:5px 6px;
  }
  .hud button { cursor:pointer; }
  .panel {
    position: fixed; bottom: 10px; left: 10px; max-width: 420px;
    background: rgba(0,0,0,0.35); border:1px solid rgba(255,255,255,0.15);
    padding:10px; border-radius:10px; backdrop-filter: blur(4px);
  }
  .panel pre { margin:0; font-size:12px; white-space:pre-wrap; }
</style>
<script src="https://unpkg.com/three@0.150.1/build/three.min.js"></script>
<script src="https://unpkg.com/three-spritetext@1.9.4/dist/three-spritetext.min.js"></script>
<script src="https://unpkg.com/3d-force-graph@1.73.2/dist/3d-force-graph.min.js"></script>
</head>
<body>
<div class="hud">
  <div class="badge" id="stats">Nodes: 0 • Edges: 0</div>

  <div class="group">
    <label>layout
      <select id="layout">
        <option value="force">force</option>
        <option value="sphere">sphere</option>
        <option value="grid">grid</option>
        <option value="radial">radial</option>
        <option value="tower">tower</option>
      </select>
    </label>
    <label>dim
      <select id="dim"><option>3D</option><option>2D</option></select>
    </label>
    <label><input type="checkbox" id="labels" checked> labels</label>
    <label><input type="checkbox" id="showNames" checked> unresolved names</label>
  </div>

  <div class="group">
    <label>charge <input type="range" id="charge" min="-500" max="-10" step="10" value="-180"></label>
    <label>link dist <input type="range" id="dist" min="20" max="220" step="10" value="90"></label>
    <label>size <input type="range" id="size" min="1" max="10" step="1" value="6"></label>
    <label>link α <input type="range" id="lopa" min="0" max="1" step="0.05" value="0.15"></label>
  </div>

  <div class="group">
    <label>kinds:</label>
    <label><input type="checkbox" class="kind" value="file" checked> file</label>
    <label><input type="checkbox" class="kind" value="class" checked> class</label>
    <label><input type="checkbox" class="kind" value="func" checked> func</label>
    <label><input type="checkbox" class="kind" value="module" checked> module</label>
    <label><input type="checkbox" class="kind" value="name" checked> name</label>
  </div>

  <div class="group">
    <input type="text" id="search" placeholder="filter (glob/regex fragment)" style="width:180px"/>
    <label>hops
      <select id="hops"><option>0</option><option selected>1</option><option>2</option></select>
    </label>
    <button id="apply">apply</button>
    <button id="clear">clear</button>
    <button id="reset">reset view</button>
    <button id="export">export JSON</button>
  </div>

  <div style="margin-left:auto" class="badge"><small>__TITLE__</small></div>
</div>
<div id="scene"></div>

<div class="panel" id="info" style="display:none">
  <strong>Node</strong>
  <pre id="infoPre"></pre>
</div>

<script>
const PAYLOAD = __DATA_JSON__;

// Normalize → ForceGraph format
function normalize(payload) {
  const nodes = (payload.nodes||[]).map(n => ({
    id: n.id,
    name: n.label || n.id,
    kind: n.kind || 'default',
    meta: n.meta || {}
  }));
  const links = (payload.edges||[]).map(e => ({
    source: e.src, target: e.dst, kind: e.kind || 'link'
  }));
  // degree
  const deg = Object.create(null);
  links.forEach(l => { deg[l.source] = (deg[l.source]||0)+1; deg[l.target] = (deg[l.target]||0)+1; });
  nodes.forEach(n => n.degree = deg[n.id] || 0);
  return {nodes, links};
}

const RAW = normalize(PAYLOAD);
let CURRENT = { nodes: RAW.nodes.slice(), links: RAW.links.slice() };

const COLORS = {
  file:   0x34d399, // emerald-400
  class:  0xa78bfa, // violet-400
  func:   0xf472b6, // pink-400
  module: 0x60a5fa, // blue-400
  name:   0x9ca3af, // gray-400
  default:0x93c5fd
};
function colorOf(n){ return COLORS[n.kind] || COLORS.default; }

const el = document.getElementById('scene');
const gStats = document.getElementById('stats');
const gLayout = document.getElementById('layout');
const gDim = document.getElementById('dim');
const gLabels = document.getElementById('labels');
const gShowNames = document.getElementById('showNames');
const gCharge = document.getElementById('charge');
const gDist = document.getElementById('dist');
const gSize = document.getElementById('size');
const gLOpa = document.getElementById('lopa');
const gKinds = Array.from(document.querySelectorAll('.kind'));
const gSearch = document.getElementById('search');
const gHops = document.getElementById('hops');
const gApply = document.getElementById('apply');
const gClear = document.getElementById('clear');
const gReset = document.getElementById('reset');
const gExport = document.getElementById('export');
const gInfo = document.getElementById('info');
const gInfoPre = document.getElementById('infoPre');

const Graph = ForceGraph3D()(el)
  .graphData(CURRENT)
  .numDimensions(3)
  .backgroundColor('#0b0e14')
  .nodeRelSize(parseInt(gSize.value,10))
  .nodeOpacity(0.95)
  .linkOpacity(parseFloat(gLOpa.value))
  .linkColor(() => 'rgba(255,255,255,0.35)')
  .onNodeHover(n => { el.style.cursor = n ? 'pointer' : null; })
  .onNodeClick((node) => {
    const dist = 70;
    const camera = Graph.camera();
    const controls = Graph.controls();
    const vec = new THREE.Vector3(node.x, node.y, node.z);
    const dir = vec.sub(camera.position).normalize();
    const newPos = vec.add(dir.multiplyScalar(-dist));
    controls.target.set(node.x, node.y, node.z);
    camera.position.set(newPos.x, newPos.y, newPos.z);
    // info
    gInfo.style.display = 'block';
    gInfoPre.textContent = JSON.stringify(node, null, 2);
  });

// Node object (SpriteText or sphere)
function rebuildNodeObjects(){
  const haveSprite = typeof SpriteText !== 'undefined';
  const size = parseInt(gSize.value,10);
  Graph.nodeThreeObject(node => {
    if (!gLabels.checked || !haveSprite) {
      const r = Math.max(2, 0.8 * size + Math.log2(2 + node.degree));
      const geom = new THREE.SphereGeometry(r, 16, 16);
      const mat = new THREE.MeshBasicMaterial({ color: colorOf(node) });
      return new THREE.Mesh(geom, mat);
    }
    const spr = new SpriteText(node.name);
    spr.material.depthWrite = false;
    spr.color = '#'+colorOf(node).toString(16).padStart(6,'0');
    spr.textHeight = 0.9 * size + Math.min(10, node.degree);
    return spr;
  });
}
rebuildNodeObjects();

// forces
Graph.d3Force('charge').strength(parseInt(gCharge.value,10));
Graph.d3Force('link').distance(() => parseInt(gDist.value,10));

// Seed 3D depth so it’s not flat
(function seedDepth(data){
  const R = 120;
  data.nodes.forEach(n => {
    if (typeof n.x !== 'number') n.x = (Math.random()-0.5)*R;
    if (typeof n.y !== 'number') n.y = (Math.random()-0.5)*R;
    if (typeof n.z !== 'number') n.z = (Math.random()-0.5)*R;
  });
})(CURRENT);

function updateStats(data=CURRENT){
  gStats.textContent = `Nodes: ${data.nodes.length} • Edges: ${data.links.length}`;
}
updateStats();

// Filtering and search
function applyFilter(){
  const keepKinds = new Set(gKinds.filter(x=>x.checked).map(x=>x.value));
  const showNames = gShowNames.checked;
  const q = gSearch.value.trim();
  const hops = parseInt(gHops.value,10);

  // base mask
  let nodes = RAW.nodes.filter(n => keepKinds.has(n.kind) && (showNames || n.kind!=='name'));

  // text filter (substring or regex if begins/ends with /)
  function matches(n){
    if (!q) return true;
    if (q.startsWith('/') && q.endsWith('/')) {
      try { return new RegExp(q.slice(1,-1),'i').test(n.name) || new RegExp(q.slice(1,-1),'i').test(n.id); }
      catch(e){ return true; }
    }
    return (n.name||'').toLowerCase().includes(q.toLowerCase()) || (n.id||'').toLowerCase().includes(q.toLowerCase());
  }
  const base = new Set(nodes.filter(matches).map(n=>n.id));

  // hops expansion
  let keep = new Set(base);
  if (hops > 0) {
    const adj = new Map(); // id -> array of neighbor ids
    RAW.links.forEach(l => {
      if (!adj.has(l.source)) adj.set(l.source, []);
      if (!adj.has(l.target)) adj.set(l.target, []);
      adj.get(l.source).push(l.target);
      adj.get(l.target).push(l.source);
    });
    let frontier = Array.from(base);
    for (let d=0; d<hops; d++){
      const next = [];
      frontier.forEach(id => {
        (adj.get(id)||[]).forEach(nb => { if (!keep.has(nb)) { keep.add(nb); next.push(nb); } });
      });
      frontier = next;
    }
  }
  // compose subset
  const nodeSet = new Set(nodes.map(n=>n.id).filter(id => keep.has(id)));
  const subNodes = RAW.nodes.filter(n => nodeSet.has(n.id));
  const subLinks = RAW.links.filter(l => nodeSet.has(l.source) && nodeSet.has(l.target));

  CURRENT = {nodes: subNodes, links: subLinks};
  Graph.graphData(CURRENT);
  updateStats(CURRENT);
}
gApply.addEventListener('click', applyFilter);
gClear.addEventListener('click', () => { gSearch.value=''; applyFilter(); });
gKinds.forEach(cb => cb.addEventListener('change', applyFilter));
gShowNames.addEventListener('change', applyFilter);

// Layout switching
function toForce(){
  Graph.graphData(CURRENT).cooldownTicks(220).resetCountdown();
}
function toSphere(){
  const N = CURRENT.nodes.length;
  const r = 140;
  CURRENT.nodes.forEach((n,i)=>{
    const phi = Math.acos(-1 + (2*i)/N);
    const theta = Math.sqrt(N*Math.PI) * phi;
    n.x = r * Math.cos(theta) * Math.sin(phi);
    n.y = r * Math.sin(theta) * Math.sin(phi);
    n.z = r * Math.cos(phi);
  });
  Graph.graphData(CURRENT).cooldownTicks(0);
}
function toGrid(){
  const n = CURRENT.nodes.length;
  const cols = Math.ceil(Math.sqrt(n));
  const cell = 40, zstep = 20;
  CURRENT.nodes.forEach((n,i)=>{
    const r = Math.floor(i/cols), c = i%cols;
    n.x = (c - cols/2)*cell;
    n.y = (r - Math.ceil(n/cols)/2)*cell;
    n.z = (n.kind==='file'?2: (n.kind==='class'?1:(n.kind==='func'?0:(n.kind==='module'?-1:-2)))) * zstep;
  });
  Graph.graphData(CURRENT).cooldownTicks(0);
}
function toRadial(){
  const groups = new Map();
  CURRENT.nodes.forEach(n => {
    const k = n.kind; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(n);
  });
  const R0 = 60, step = 40;
  const kinds = Array.from(groups.keys()).sort();
  kinds.forEach((k,ki)=>{
    const ring = groups.get(k), R = R0 + ki*step, N = ring.length;
    ring.forEach((n,i)=>{
      const a = (2*Math.PI*i)/Math.max(1,N);
      n.x = R*Math.cos(a); n.y = R*Math.sin(a); n.z = (ki - kinds.length/2)*12;
    });
  });
  Graph.graphData(CURRENT).cooldownTicks(0);
}
function toTower(){
  const order = {file:0, class:1, func:2, module:3, name:4};
  const layers = new Map();
  CURRENT.nodes.forEach(n=>{
    const z = (order[n.kind]||5);
    if (!layers.has(z)) layers.set(z, []);
    layers.get(z).push(n);
  });
  const zs = Array.from(layers.keys()).sort((a,b)=>a-b);
  const cell = 36, zgap = 36;
  zs.forEach((z,zi)=>{
    const L = layers.get(z), cols = Math.ceil(Math.sqrt(L.length));
    L.forEach((n,i)=>{
      const r = Math.floor(i/cols), c = i%cols;
      n.x = (c - cols/2)*cell; n.y = (r - Math.ceil(L.length/cols)/2)*cell; n.z = (zi - zs.length/2)*zgap;
    });
  });
  Graph.graphData(CURRENT).cooldownTicks(0);
}

gLayout.addEventListener('change', () => {
  const v = gLayout.value;
  if (v==='force') toForce();
  else if (v==='sphere') toSphere();
  else if (v==='grid') toGrid();
  else if (v==='radial') toRadial();
  else if (v==='tower') toTower();
});

gDim.addEventListener('change', () => {
  Graph.numDimensions(gDim.value==='2D' ? 2 : 3);
  Graph.resetCountdown();
});

// Tunables
gCharge.addEventListener('input', () => {
  Graph.d3Force('charge').strength(parseInt(gCharge.value,10));
  Graph.resetCountdown();
});
gDist.addEventListener('input', () => {
  const d = parseInt(gDist.value,10);
  Graph.d3Force('link').distance(() => d);
  Graph.resetCountdown();
});
gSize.addEventListener('input', () => { Graph.nodeRelSize(parseInt(gSize.value,10)); rebuildNodeObjects(); Graph.refresh(); });
gLOpa.addEventListener('input', () => { Graph.linkOpacity(parseFloat(gLOpa.value)); });

// Reset & export
gReset.addEventListener('click', () => { Graph.zoomToFit(400, 60, () => true); });
gExport.addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(PAYLOAD, null, 2)], {type:'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'file_ast_graph.json'; a.click(); URL.revokeObjectURL(a.href);
});

// Initial view
setTimeout(()=>Graph.zoomToFit(400,80,()=>true), 200);
</script>
</body></html>
"""

# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_glx_graphs_dir(file_path: Path) -> Path:
    # Prefer: REPO/.glx/graphs (jeśli w drzewie istnieje ".glx"), else obok pliku
    for p in [file_path.parent] + list(file_path.parents):
        glx = p / ".glx" / "graphs"
        if glx.parent.exists():
            glx.mkdir(parents=True, exist_ok=True)
            return glx
    glx = file_path.parent / ".glx" / "graphs"
    glx.mkdir(parents=True, exist_ok=True)
    return glx

def _write_html(payload: Dict[str, object], out_path: Path, title: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__DATA_JSON__", json.dumps(payload, ensure_ascii=False)) \
               .replace("__TITLE__", title)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="3D AST of a single Python file (interactive)")
    ap.add_argument("--file", required=True, help="ścieżka do pliku .py do wizualizacji")
    ap.add_argument("--output", default=None, help="ścieżka wyjściowa .html (domyślnie: <repo>/.glx/graphs/file_ast_3d_<stem>.html)")
    ap.add_argument("--title", default=None, help="tytuł w HTML (opcjonalnie)")
    args = ap.parse_args(argv)

    f = Path(args.file).resolve()
    if not f.exists():
        print(f"[error] file not found: {f}", file=sys.stderr)
        return 2

    g = build_file_graph(f)
    payload = graph_to_payload(g)

    title = args.title or f"{f.name} • AST graph"
    out = Path(args.output).resolve() if args.output else (_resolve_glx_graphs_dir(f) / f"file_ast_3d_{f.stem}.html")
    _write_html(payload, out, title)
    print(str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
