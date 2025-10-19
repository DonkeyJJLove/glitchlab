# glx/tools/file_scope_viz.py
# -*- coding: utf-8 -*-
"""
File Scope Viz — „autentyczna” wersja:
- Projektowy graf 3D (nodes/edges z project_graph.json albo budowany ad hoc).
- Mozaika jednego pliku (preferuje realne artefakty: --mosaic-input / --delta-input),
  z fallbackiem na szybki pseudo-core/delta wyliczany lokalnie.
- KPI AST z prawdziwego indeksu (glitchlab.analysis.ast_index) z łagodnym fallbackiem.

Użycie (z ROOT projektu — tam gdzie leży folder 'glitchlab/'):
  python -m glx.tools.file_scope_viz --repo-root .\glitchlab\ --file .\glitchlab\analysis\astmap.py
  # lub jawnie podając artefakty:
  python -m glx.tools.file_scope_viz --repo-root .\glitchlab\ ^
      --graph-input .\glitchlab\.glx\graphs\project_graph.json ^
      --mosaic-input .\glitchlab\.glx\graphs\mosaic_meta.json ^
      --file .\glitchlab\analysis\astmap.py --grid auto

Plik wyjściowy: <repo>/.glx/graphs/file_scope_viz.html (chyba że podasz --output).
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# HTML (jedna instancja THREE, SpriteText → 3d-force-graph; UI do zmiany grid/metryki)
# ──────────────────────────────────────────────────────────────────────────────
HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>File Scope Viz — Graph + Mosaic</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  :root { --bg:#0b0e14; --fg:#e6e6e6; --panel:#0f1320; --muted:#9ca3af; --line:#2b3242; }
  html, body { margin:0; padding:0; height:100%; background:var(--bg); color:var(--fg); font-family: Inter, Arial, sans-serif; }
  #app { display:grid; grid-template-rows:auto 1fr; height:100%; }
  .hud {
    display:flex; gap:12px; align-items:center; padding:10px 12px; border-bottom:1px solid var(--line);
    background: rgba(0,0,0,0.35); backdrop-filter: blur(4px);
  }
  .hud .badge { padding:2px 8px; border-radius:8px; background:#121829; border:1px solid var(--line); font-size:12px; }
  .hud input, .hud select, .hud button {
    background:#121829; border:1px solid var(--line); color:var(--fg); border-radius:8px; padding:6px 8px;
  }
  .hud .spacer { flex: 1 1 auto; }
  .panes { display:grid; grid-template-columns: 1.0fr 0.9fr; gap:8px; padding:8px; height:100%; }
  .panel { border:1px solid var(--line); border-radius:10px; overflow:hidden; background:var(--panel); display:flex; flex-direction:column; min-height:0; }
  .panel h3 { margin:0; padding:8px 10px; font-size:14px; border-bottom:1px solid var(--line); color:#93c5fd;}
  .panel .body { flex:1; position: relative; min-height:0; }
  #graph3d { position:absolute; inset:0; }
  #mosaicWrap { display:grid; grid-template-rows:auto 1fr; height:100%; }
  canvas { background:#0b0e14; display:block; width:100%; height:100%; image-rendering: pixelated; }
  .row { display:flex; gap:10px; align-items:center; padding:8px; border-bottom:1px solid var(--line); }
  .kpi { display:flex; gap:14px; align-items:center; font-size:12px; color:var(--muted);}
  .kpi .pill { padding:2px 8px; border-radius:999px; border:1px solid var(--line); background:#111628; color:#cbd5e1;}
  .legend { font-size:12px; color:var(--muted); margin-left:auto; }
</style>
<script src="https://unpkg.com/three@0.150.1/build/three.min.js"></script>
<script src="https://unpkg.com/three-spritetext@1.9.4/dist/three-spritetext.min.js"></script>
<script src="https://unpkg.com/3d-force-graph@1.73.2/dist/3d-force-graph.min.js"></script>
</head>
<body>
<div id="app">
  <div class="hud">
    <div class="badge" id="statsGraph">Graph: 0 nodes • 0 edges</div>
    <div class="badge" id="statsMosaic">Mosaic: 0×0 • roi 0%</div>
    <div class="kpi" id="kpiAst">
      <span class="pill" id="kS">S=0</span>
      <span class="pill" id="kH">H=0</span>
      <span class="pill" id="kZ">Z=0</span>
      <span class="pill" id="kDefs">defs=0</span>
      <span class="pill" id="kCalls">calls=0</span>
      <span class="pill" id="kImports">imports=0</span>
    </div>
    <div class="spacer"></div>
    <label><input type="checkbox" id="showLabels" checked/> labels</label>
    <label>kind:
      <select id="kindFilter">
        <option value="">(all)</option>
        <option>module</option><option>file</option><option>func</option><option>topic</option><option>name</option>
      </select>
    </label>
    <button id="resetGraph">reset view</button>
  </div>

  <div class="panes">
    <div class="panel">
      <h3>Project Graph 3D</h3>
      <div class="body"><div id="graph3d"></div></div>
    </div>
    <div class="panel">
      <h3>Mosaic: <span id="fileName"></span></h3>
      <div class="body" id="mosaicWrap">
        <div class="row">
          <label>metric:
            <select id="metricSel">
              <option value="edge">edge (intensity)</option>
              <option value="entropy">entropy</option>
              <option value="roi">roi</option>
              <option value="degree">grid degree</option>
            </select>
          </label>
          <label>grid:
            <select id="gridSel">
              <option value="auto">auto</option>
              <option value="8x8">8×8</option>
              <option value="12x12">12×12</option>
              <option value="16x16">16×16</option>
            </select>
          </label>
          <label>contrast <input type="range" id="contrast" min="0.5" max="4" step="0.1" value="1"/></label>
          <div class="legend">click tile → details in console</div>
        </div>
        <canvas id="mCanvas"></canvas>
      </div>
    </div>
  </div>
</div>

<script>
const PAYLOAD = __PAYLOAD_JSON__;

// ------------- GRAPH 3D ----------------
function normalizeGraph(payload) {
  const nodes = (payload.nodes || []).map(n => ({ id:n.id, name:n.label||n.id, kind:n.kind||'default', meta:n.meta||{} }));
  const links = (payload.edges || []).map(e => ({ source:e.src, target:e.dst, kind:e.kind||'link' }));
  const deg = Object.create(null); links.forEach(l => { deg[l.source]=(deg[l.source]||0)+1; deg[l.target]=(deg[l.target]||0)+1; });
  nodes.forEach(n => n.degree = deg[n.id] || 0);
  return {nodes, links};
}
const GRAPH = normalizeGraph(PAYLOAD.graph);
document.getElementById('statsGraph').textContent = `Graph: ${GRAPH.nodes.length} nodes • ${GRAPH.links.length} edges`;

const COLORS = { module:0x60a5fa, file:0x34d399, func:0xf472b6, topic:0xfbbf24, name:0x9ca3af, default:0x93c5fd };
function colorFor(n){ return COLORS[n.kind] || COLORS.default; }

const gEl = document.getElementById('graph3d');
const showLabels = document.getElementById('showLabels');
const kindSel = document.getElementById('kindFilter');
const resetBtn = document.getElementById('resetGraph');

const Graph = ForceGraph3D()(gEl)
 .graphData(GRAPH)
 .nodeColor(n => colorFor(n))
 .nodeOpacity(0.95)
 .nodeRelSize(4)
 .linkOpacity(0.12)
 .linkColor(()=>'rgba(255,255,255,0.35)')
 .backgroundColor('#0b0e14');

const haveSprite = (typeof SpriteText !== 'undefined');
Graph.nodeThreeObject(node => {
  if (!showLabels.checked || !haveSprite) {
    const geom = new THREE.SphereGeometry(Math.max(2, 1 + Math.log2(2+node.degree)), 16, 16);
    const mat = new THREE.MeshBasicMaterial({ color: colorFor(node) });
    return new THREE.Mesh(geom, mat);
  }
  const sprite = new SpriteText(node.name);
  sprite.material.depthWrite = false;
  sprite.color = '#' + colorFor(node).toString(16).padStart(6,'0');
  sprite.textHeight = 8 + Math.min(10, node.degree);
  return sprite;
});
showLabels.addEventListener('change', () => Graph.refresh());
kindSel.addEventListener('change', () => {
  const k = kindSel.value;
  if (!k) return Graph.graphData(GRAPH);
  const keep = new Set(GRAPH.nodes.filter(n => n.kind===k).map(n => n.id));
  const nodes = GRAPH.nodes.filter(n => keep.has(n.id));
  const links = GRAPH.links.filter(l => keep.has(l.source) && keep.has(l.target));
  Graph.graphData({nodes, links});
});
resetBtn.addEventListener('click', ()=> Graph.zoomToFit(400, 60, node=>true));
setTimeout(()=> Graph.zoomToFit(400, 80, n => n.degree>=0), 200);

// ------------- MOSAIC ----------------
const fileName = document.getElementById('fileName');
fileName.textContent = PAYLOAD.file.name || PAYLOAD.file.path || 'file';

const mCanvas = document.getElementById('mCanvas');
const ctx = mCanvas.getContext('2d');
const metricSel = document.getElementById('metricSel');
const gridSel = document.getElementById('gridSel');
const contrastInput = document.getElementById('contrast');

function setCanvasSize(){
  const rect = mCanvas.getBoundingClientRect();
  mCanvas.width = Math.max(300, Math.floor(rect.width));
  mCanvas.height = Math.max(300, Math.floor(rect.height - 2));
}
window.addEventListener('resize', ()=>{ setCanvasSize(); drawMosaic(); });

function clamp01(x){ return x<0?0:(x>1?1:x); }
function colorForVal(v){
  const t=clamp01(v);
  const r = Math.floor(255*clamp01(1.5*t - 0.2));
  const g = Math.floor(255*clamp01(1.5*t));
  const b = Math.floor(255*clamp01(1.2*(1-t)));
  return `rgb(${r},${g},${b})`;
}

let Mosaic = PAYLOAD.mosaic; // {rows, cols, cells:[{edge,entropy,roi,degree,...}], roi_coverage}
document.getElementById('statsMosaic').textContent = `Mosaic: ${Mosaic.rows}×${Mosaic.cols} • roi ${(100*(Mosaic.roi_coverage||0)).toFixed(1)}%`;

function pickGrid(){
  const opt = gridSel.value;
  if (opt==='auto') return [Mosaic.rows,Mosaic.cols];
  const [a,b] = opt.split('x').map(s=>parseInt(s,10));
  return [a,b];
}

mCanvas.addEventListener('click', (ev)=>{
  const rect = mCanvas.getBoundingClientRect();
  const [rows,cols] = pickGrid();
  const cw = mCanvas.width/cols, ch = mCanvas.height/rows;
  const cx = Math.floor((ev.clientX - rect.left)/cw);
  const cy = Math.floor((ev.clientY - rect.top)/ch);
  const idx = cy*cols + cx;
  const cell = Mosaic.cells[idx];
  if (cell) { console.log('Tile', idx, cell); }
});

function drawMosaic(){
  setCanvasSize();
  const metric = metricSel.value;
  const [rows, cols] = pickGrid();
  const srcRows = Mosaic.rows, srcCols = Mosaic.cols;
  const cw = mCanvas.width/cols, ch = mCanvas.height/rows;
  const cst = parseFloat(contrastInput.value||'1');
  ctx.clearRect(0,0,mCanvas.width,mCanvas.height);
  for (let r=0; r<rows; r++){
    for (let c=0; c<cols; c++){
      const rr = Math.min(srcRows-1, Math.max(0, Math.round((r/(rows-1||1))*(srcRows-1))));
      const cc = Math.min(srcCols-1, Math.max(0, Math.round((c/(cols-1||1))*(srcCols-1))));
      const idx = rr*srcCols + cc;
      const cell = Mosaic.cells[idx] || {};
      let v = parseFloat(cell[metric]||0);
      v = Math.pow(clamp01(v), cst);
      ctx.fillStyle = colorForVal(v);
      ctx.fillRect(Math.floor(c*cw), Math.floor(r*ch), Math.ceil(cw), Math.ceil(ch));
      if ((cell.roi||0)>0.5){
        ctx.strokeStyle = 'rgba(255,255,255,0.9)';
        ctx.lineWidth = 2;
        ctx.strokeRect(Math.floor(c*cw)+0.5, Math.floor(r*ch)+0.5, Math.ceil(cw)-1, Math.ceil(ch)-1);
      }
    }
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  for (let r=1; r<rows; r++){ const y = Math.floor(r*ch)+0.5; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(mCanvas.width,y); ctx.stroke(); }
  for (let c=1; c<cols; c++){ const x = Math.floor(c*cw)+0.5; ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,mCanvas.height); ctx.stroke(); }
}
metricSel.addEventListener('change', drawMosaic);
gridSel.addEventListener('change', drawMosaic);
contrastInput.addEventListener('input', drawMosaic);

setCanvasSize();
drawMosaic();

// KPIs
(function(){
  const k = PAYLOAD.ast || {};
  const setTxt = (id, t) => { const el=document.getElementById(id); if (el) el.textContent=t; }
  setTxt('kS', `S=${k.S||0}`); setTxt('kH', `H=${k.H||0}`); setTxt('kZ', `Z=${k.Z||0}`);
  setTxt('kDefs', `defs=${k.defs||0}`); setTxt('kCalls', `calls=${k.calls||0}`); setTxt('kImports', `imports=${k.imports||0}`);
})();
</script>
</body>
</html>
"""

# ──────────────────────────────────────────────────────────────────────────────
# IO helpers
# ──────────────────────────────────────────────────────────────────────────────
def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def _atomic_write_text(path: Path, data: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8", newline="\n") as tmp:
        tmp.write(data); tmp.flush(); os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    return path

# ──────────────────────────────────────────────────────────────────────────────
# Project graph loader / builder
# ──────────────────────────────────────────────────────────────────────────────
def _load_or_build_graph(repo_root: Path, input_path: Optional[Path]) -> Dict[str, Any]:
    if input_path and input_path.exists():
        return _read_json(input_path)
    art = repo_root / ".glx" / "graphs" / "project_graph.json"
    if art.exists():
        return _read_json(art)
    try:
        from glitchlab.analysis.project_graph import build_project_graph, save_project_graph  # type: ignore
    except Exception:
        try:
            from analysis.project_graph import build_project_graph, save_project_graph  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Brak project_graph.json. Podaj --graph-input albo zbuduj artefakt:\n"
                "  python -m glitchlab.analysis.project_graph build --repo-root . --write\n"
                "lub\n"
                "  python -m analysis.project_graph build --repo-root . --write"
            ) from e
    g = build_project_graph(repo_root)
    save_project_graph(g, repo_root)
    return {
        "version": g.version,
        "meta": g.meta,
        "nodes": [{"id": n.id, "kind": n.kind, "label": n.label, "meta": n.meta} for n in sorted(g.nodes.values(), key=lambda x: x.id)],
        "edges": [{"src": e.src, "dst": e.dst, "kind": e.kind} for e in g.edges],
    }

# ──────────────────────────────────────────────────────────────────────────────
# AST KPIs — autentycznie z ast_index jeśli dostępny
# ──────────────────────────────────────────────────────────────────────────────
def _ast_kpis_real(file_path: Path) -> Optional[Dict[str, int]]:
    try:
        try:
            from glitchlab.analysis.ast_index import ast_index_of_file  # type: ignore
        except Exception:
            from analysis.ast_index import ast_index_of_file  # type: ignore
    except Exception:
        return None
    try:
        idx = ast_index_of_file(str(file_path))
    except Exception:
        idx = None
    if not idx:
        return None
    defs_n = len(getattr(idx, "defs", {}) or {})
    calls_n = len(getattr(idx, "calls", []) or [])
    imports_n = len(getattr(idx, "imports", []) or [])
    return {
        "S": int(getattr(idx, "S", 0) or 0),
        "H": int(getattr(idx, "H", 0) or 0),
        "Z": int(getattr(idx, "Z", 0) or 0),
        "defs": int(defs_n),
        "calls": int(calls_n),
        "imports": int(imports_n),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Fallback AST KPIs — kompatybilny z Python 3.9 (bez ast.Match)
# ──────────────────────────────────────────────────────────────────────────────
def _ast_kpis_fallback(src: str) -> Dict[str, int]:
    """
    Minimalny, szybki fallback:
      - S: wagi dla definicji/kontrolnych konstrukcji
      - H: wagi dla wywołań/importów/przypisań
      - Z: maksymalna głębokość drzewa AST (rekursja po dzieciach)
    Zawiera ochronę przed brakiem ast.Match w Py<3.10.
    """
    try:
        tree = ast.parse(src)
    except Exception:
        return {"S":0,"H":0,"Z":0,"defs":0,"calls":0,"imports":0}

    HAS_MATCH = hasattr(ast, "Match")
    CONTROL = (ast.If, ast.For, ast.While, ast.With, ast.Try) + ((ast.Match,) if HAS_MATCH else ())
    S=H=0; defs=calls=imports=0

    # głębokość AST
    max_depth = 0
    def _walk(n: ast.AST, d: int):
        nonlocal S,H,defs,calls,imports,max_depth
        max_depth = max(max_depth, d)
        # S
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)): S += 3; defs += 1
        if isinstance(n, ast.ClassDef): S += 4; defs += 1
        if isinstance(n, CONTROL): S += 2
        # H
        if isinstance(n, ast.Call): H += 3; calls += 1
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)): H += 2
        if isinstance(n, (ast.Import, ast.ImportFrom)): H += 2; imports += 1
        for ch in ast.iter_child_nodes(n):
            _walk(ch, d+1)
    _walk(tree, 0)

    # heurystyczne Z: zbijamy do umiarkowanej skali (moduł depth bywa duży)
    Z = max(1, int(round(max_depth / 3)))  # „co ~3 poziomy” jako jeden przyrost Z
    return {"S": int(S), "H": int(H), "Z": int(Z), "defs": int(defs), "calls": int(calls), "imports": int(imports)}

# ──────────────────────────────────────────────────────────────────────────────
# Mosaic extractors — prefer artefakty, fallback do pseudo
# ──────────────────────────────────────────────────────────────────────────────
def _coerce_int(x: Any, d: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return d

def _norm_path(p: str) -> str:
    return str(Path(p)).replace("\\", "/")

def _extract_mosaic_from_mosaic_meta(mosaic_meta: Mapping[str, Any], file_path: Path) -> Optional[Dict[str, Any]]:
    target = _norm_path(str(file_path))
    pf = mosaic_meta.get("per_file")
    if isinstance(pf, Mapping):
        if target in pf:
            return dict(pf[target])  # type: ignore
        base = _norm_path(file_path.name)
        for k, v in pf.items():
            if _norm_path(k).endswith("/" + base):
                return dict(v) if isinstance(v, Mapping) else None  # type: ignore
    ent = mosaic_meta.get("entries")
    if isinstance(ent, list):
        for item in ent:
            if not isinstance(item, Mapping):
                continue
            f = item.get("file") or {}
            p = _norm_path(f.get("path") or f.get("name") or "")
            if not p:
                continue
            if p == target or p.endswith("/" + file_path.name):
                return dict(item)  # type: ignore
    if "file" in mosaic_meta and isinstance(mosaic_meta.get("file"), Mapping):
        p = _norm_path(mosaic_meta["file"].get("path") or mosaic_meta["file"].get("name") or "")
        if p and (p == target or p.endswith("/" + file_path.name)):
            return dict(mosaic_meta)  # type: ignore
    return None

def _extract_mosaic_from_delta(delta: Mapping[str, Any], file_path: Path) -> Optional[Dict[str, Any]]:
    target = _norm_path(str(file_path))
    if "grid" in delta and "edges" in delta:
        f = delta.get("file") or {}
        p = _norm_path(f.get("path") or f.get("name") or "")
        if not p or p == target or p.endswith("/" + file_path.name):
            return dict(delta)  # type: ignore
    files = delta.get("files") or delta.get("changed_files")
    if isinstance(files, list):
        for it in files:
            if not isinstance(it, Mapping):
                continue
            p = _norm_path(it.get("path") or it.get("file") or "")
            if p and (p == target or p.endswith("/" + file_path.name)):
                return dict(it)  # type: ignore
    pf = delta.get("per_file")
    if isinstance(pf, Mapping):
        if target in pf:
            return dict(pf[target])  # type: ignore
        base = _norm_path(file_path.name)
        for k, v in pf.items():
            if _norm_path(k).endswith("/" + base):
                return dict(v) if isinstance(v, Mapping) else None  # type: ignore
    return None

# Pseudo core/delta (gdy nie mamy artefaktów)
_KEYWORDS = {
    "def": 5.0, "class": 7.0, "return": 2.0, "yield": 3.0,
    "if": 3.0, "elif": 2.0, "else": 1.0,
    "for": 3.0, "while": 3.0, "try": 4.0, "except": 4.0, "with": 2.0, "match": 4.0, "case": 2.0,
    "import": 2.0, "from": 1.5, "as": 0.5,
    "lambda": 3.0, "await": 2.5, "async": 2.5,
}
def _entropy_of_text(text: str) -> float:
    if not text:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    ps = [v / n for v in freq.values()]
    H = -sum(p * math.log2(p) for p in ps if p > 0)
    Hmax = math.log2(min(256, max(2, len(freq))))
    v = (H / (Hmax or 1.0))
    return 0.0 if v < 0 else (1.0 if v > 1.0 else v)

def _chunk_score(lines: List[str]) -> Tuple[float, float]:
    score = 0.0; ent = 0.0
    for ln in lines:
        low = ln.strip().lower()
        for k, w in _KEYWORDS.items():
            if k in low:
                score += w * low.count(k)
        score += 0.2 * low.count("(")
        score += 0.2 * low.count(".")
        leading = len(ln) - len(ln.lstrip(" "))
        if leading >= 8: score += 0.2
        if leading >= 16: score += 0.2
        ent += _entropy_of_text(low)
    if lines:
        ent /= len(lines)
    return score, ent

def _grid_for_lines(n_lines: int) -> Tuple[int, int]:
    tiles = max(4, math.ceil(n_lines / 12))
    side = int(math.ceil(math.sqrt(tiles)))
    return side, side  # rows, cols

def _build_pseudo_delta_for_file(file_path: Path, grid: Optional[str]=None) -> Dict[str, Any]:
    src = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = src.splitlines() or [""]
    if grid and "x" in grid:
        r,c = grid.lower().split("x",1)
        rows, cols = max(1,int(r)), max(1,int(c))
    else:
        rows, cols = _grid_for_lines(len(lines))
    N = rows*cols
    step = max(1, math.ceil(len(lines) / N))
    edges: List[float] = []; entrs: List[float] = []
    for i in range(N):
        L0 = i*step
        chunk = lines[L0:L0+step]
        sc, en = _chunk_score(chunk)
        edges.append(sc); entrs.append(en)
    mx = max(edges) if edges else 1.0
    edges01 = [ (x/mx) if mx>0 else 0.0 for x in edges ]
    order = sorted(range(N), key=lambda i: edges01[i], reverse=True)
    k = max(1, int(0.1*N))
    roi_idx = order[:k]
    bs = {}
    for r in range(rows):
        for c in range(cols):
            idx = r*cols + c
            bs[(c, r)] = {"edges": float(edges01[idx]), "entropy": float(entrs[idx])}
    W,H = 768, 768
    sx, sy = W/cols, H/rows
    cells = [ {"id": r*cols+c, "center":[ int(round((c+0.5)*sx)), int(round((r+0.5)*sy)) ]} for r in range(rows) for c in range(cols) ]
    core = {"mode":"square","size":[W,H],"cells":cells}
    return {
        "file": {"path": str(file_path), "name": file_path.name, "lines": len(lines)},
        "grid": {"rows": rows, "cols": cols},
        "core": core,
        "edges": edges01,
        "block_stats": { f"{k[0]},{k[1]}": v for k,v in bs.items() },
        "roi_indices": roi_idx
    }

def _mosaic_from_any(spec: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    grid = spec.get("grid") or {}
    rows = _coerce_int(grid.get("rows"), 0)
    cols = _coerce_int(grid.get("cols"), 0)
    if rows <= 0 or cols <= 0:
        core = spec.get("core") or {}
        cells = core.get("cells") or []
        if isinstance(cells, list) and cells:
            n = len(cells)
            side = int(round(math.sqrt(n)))
            rows = side; cols = side
    if rows <= 0 or cols <= 0:
        return None
    N = rows*cols
    edges = list(spec.get("edges") or [0.0]*N)
    roi = set(int(i) for i in spec.get("roi_indices") or [])
    def deg(idx: int) -> int:
        r, c = divmod(idx, cols)
        d = 0
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            rr,cc = r+dr,c+dc
            if 0<=rr<rows and 0<=cc<cols: d+=1
        return d
    bs = spec.get("block_stats") or {}
    def entropy_at(i: int) -> float:
        r, c = divmod(i, cols)
        key = f"{c},{r}"
        val = bs.get(key)
        if isinstance(val, Mapping) and "entropy" in val:
            try:
                return float(val["entropy"])
            except Exception:
                return 0.0
        return 0.0
    cells = []
    for i in range(N):
        cells.append({
            "id": i,
            "row": i//cols, "col": i%cols,
            "edge": float(edges[i] if i < len(edges) else 0.0),
            "entropy": float(entropy_at(i)),
            "roi": 1.0 if i in roi else 0.0,
            "degree": deg(i),
        })
    roi_cov = (len(roi)/N) if N>0 else 0.0
    return {"rows": rows, "cols": cols, "cells": cells, "roi_coverage": roi_cov}

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Unified File Scope Viz (authentic): Project Graph 3D + File Mosaic + AST KPIs")
    ap.add_argument("--repo-root", "--repo", dest="repo_root", default=".", help="katalog ROOT projektu (ten z folderem glitchlab/)")
    ap.add_argument("--graph-input", default=None, help="ścieżka do project_graph.json (opcjonalnie)")
    ap.add_argument("--mosaic-input", default=None, help="ścieżka do mosaic_meta.json (opcjonalnie)")
    ap.add_argument("--delta-input", default=None, help="ścieżka do delta_report.json / file-delta.json (opcjonalnie)")
    ap.add_argument("--file", "-f", required=True, help="ścieżka do pliku źródłowego, np. glitchlab/analysis/astmap.py")
    ap.add_argument("--grid", default="auto", help="preset siatki mozaiki: auto | 8x8 | 12x12 | 16x16 (fallback)")
    ap.add_argument("--output", "-o", default=None, help="ścieżka wyjściowa .html (domyślnie .glx/graphs/file_scope_viz.html)")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f("[err] nie znaleziono pliku: {file_path}"), file=sys.stderr)
        return 2

    graph = _load_or_build_graph(root, Path(args.graph_input).resolve() if args.graph_input else None)

    mosaic_spec: Optional[Dict[str, Any]] = None
    if args.mosaic_input:
        mp = Path(args.mosaic_input).resolve()
        if mp.exists():
            mm = _read_json(mp)
            entry = _extract_mosaic_from_mosaic_meta(mm, file_path)
            if entry:
                mosaic_spec = entry
    if mosaic_spec is None and args.delta_input:
        dp = Path(args.delta_input).resolve()
        if dp.exists():
            delta = _read_json(dp)
            entry = _extract_mosaic_from_delta(delta, file_path)
            if entry:
                mosaic_spec = entry
    if mosaic_spec is None:
        preset = args.grid if args.grid and args.grid.lower() != "auto" else None
        mosaic_spec = _build_pseudo_delta_for_file(file_path, grid=preset)

    mosaic = _mosaic_from_any(mosaic_spec or {})
    if mosaic is None:
        print("[err] nie udało się zbudować mozaiki (brak grid/edges).", file=sys.stderr)
        return 3

    kpis = _ast_kpis_real(file_path)
    if kpis is None:
        src = file_path.read_text(encoding="utf-8", errors="ignore")
        kpis = _ast_kpis_fallback(src)

    payload = {
        "graph": graph,
        "file": { "path": str(file_path), "name": file_path.name },
        "mosaic": mosaic,
        "ast": kpis,
        "meta": {
            "source": {
                "graph": args.graph_input or str(root / ".glx" / "graphs" / "project_graph.json"),
                "mosaic": args.mosaic_input or args.delta_input or "pseudo",
            }
        }
    }

    out = Path(args.output).resolve() if args.output else (root/".glx"/"graphs"/"file_scope_viz.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__PAYLOAD_JSON__", json.dumps(payload, ensure_ascii=False))
    _atomic_write_text(out, html)
    print(str(out))
    return 0

if __name__ == "__main__":
    sys.exit(main())
