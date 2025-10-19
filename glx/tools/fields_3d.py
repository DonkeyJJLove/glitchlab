#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D wizualizacja pól (fields) na grafie projektu (Force-Graph + Three.js, UMD)
Użycie:
  python -m glitchlab.glx.tools.fields_3d --repo-root .\glitchlab ^
      --field-input .\glitchlab\.glx\graphs\fields\dist_shz.json ^
      --output .\glitchlab\.glx\graphs\project_graph_fields_3d.html

Opcje:
  --graph-input  opcjonalna ścieżka do project_graph.json
  --field-input  ścieżka do fields/<name>.json (np. dist_shz.json) [wymagane]
  --invert       odwróć skalę (przydane dla *dist*, by patrzeć jak *sim*)
  --size-by      degree|field (skala rozmiaru węzła)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Project Graph • Fields 3D</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  html, body { margin:0; padding:0; height:100%; background:#0b0e14; color:#e6e6e6; font-family: Inter, ui-sans-serif, system-ui, Arial, sans-serif; }
  #scene { width:100%; height:100%; }
  .hud {
    position: fixed; top: 8px; left: 8px; right: 8px; display:flex; gap:12px; align-items:center; flex-wrap: wrap;
    background: rgba(0,0,0,0.35); border:1px solid rgba(255,255,255,0.15); padding:8px 10px; border-radius:10px; backdrop-filter: blur(4px);
  }
  .hud .badge { padding:2px 6px; border-radius:6px; background:#1e2430; border:1px solid #2b3242; }
  .hud button, .hud select, .hud input[type="checkbox"], .hud input[type="range"] {
    background:#111827; border:1px solid #2b3242; color:#e6e6e6; border-radius:6px; padding:5px 8px;
  }
  .hud .group { display:flex; align-items:center; gap:6px; }
  .legend {
    position: fixed; bottom: 12px; left: 12px; right: 12px; display:flex; gap:14px; align-items:flex-end;
    pointer-events: none;
  }
  .legend .bar {
    height: 10px; flex: 1 1 auto; border-radius: 6px; border:1px solid rgba(255,255,255,0.15);
    background: linear-gradient(90deg, #0d0887 0%, #6a00a8 16%, #b12a90 32%, #e16462 48%, #fca636 64%, #f0f921 100%);
    box-shadow: 0 0 0 1px rgba(0,0,0,0.2) inset;
  }
  .legend .ticks { display:flex; justify-content:space-between; width:100%; font-size:12px; color:#cbd5e1; margin-top:4px;}
  .legend .title { font-size:12px; color:#9ca3af; margin-bottom:4px;}
  .hud a { color:#93c5fd; text-decoration:none; }
  .kpi { font-size:12px; color:#cbd5e1; }
</style>
<!-- UMD (brak CORS problemów lokalnie) -->
<script src="https://unpkg.com/three@0.150.1/build/three.min.js"></script>
<script src="https://unpkg.com/three-spritetext@1.9.4/dist/three-spritetext.min.js"></script>
<script src="https://unpkg.com/3d-force-graph@1.73.2/dist/3d-force-graph.min.js"></script>
</head>
<body>
<div class="hud">
  <div class="badge" id="stats">Nodes: 0 • Edges: 0</div>
  <div class="group"><label><input type="checkbox" id="labels" checked/> labels</label></div>
  <div class="group">size-by:
    <select id="sizeBy">
      <option value="degree">degree</option>
      <option value="field">field</option>
    </select>
  </div>
  <div class="group">threshold:
    <input type="range" id="thresh" min="0" max="1" step="0.01" value="0"/>
    <span id="threshV" class="kpi">0.00</span>
  </div>
  <div class="group">top-K:
    <input type="range" id="topk" min="0" max="100" step="1" value="100"/>
    <span id="topkV" class="kpi">all</span>
  </div>
  <div class="group"><label><input type="checkbox" id="invert"/> invert</label></div>
  <button id="reset">reset view</button>
  <div style="margin-left:auto"><small>GlitchLab • fields-3D</small></div>
</div>

<div id="scene"></div>

<div class="legend">
  <div style="min-width:220px;">
    <div class="title" id="legendTitle">field: –</div>
    <div class="bar"></div>
    <div class="ticks"><span id="tickMin">0</span><span id="tickMid">0.5</span><span id="tickMax">1</span></div>
  </div>
</div>

<script>
const GRAPH = __DATA_GRAPH__;
const FIELD = __DATA_FIELD__;
const SETTINGS = __DATA_SETTINGS__;

const COLORS = {
  default: 0x93c5fd
};

// Turbo-like ramp in JS (0..1)
function rampTurbo(t) {
  // Piecewise polynomial approximation (simplified)
  const r = Math.round(255 * Math.min(1, Math.max(0, 0.13572138 + 4.61539260*t - 42.66032258*t*t + 132.13108234*t*t*t - 152.94239396*t*t*t*t + 59.28637943*t*t*t*t*t)));
  const g = Math.round(255 * Math.min(1, Math.max(0, 0.09140261 + 2.30097675*t - 4.51317663*t*t - 2.00344700*t*t*t + 18.22835057*t*t*t*t - 14.12650088*t*t*t*t*t)));
  const b = Math.round(255 * Math.min(1, Math.max(0, 0.10667330 + 12.64194608*t - 60.58204836*t*t + 110.36276771*t*t*t - 89.90310912*t*t*t*t + 27.34824973*t*t*t*t*t)));
  return (r<<16) + (g<<8) + b;
}
function clamp01(x){ return x<0?0:(x>1?1:x); }
function quantile(arr, q){
  if(!arr.length) return 0;
  const a = Array.from(arr).sort((x,y)=>x-y);
  if(q<=0) return a[0];
  if(q>=1) return a[a.length-1];
  const pos = (a.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return a[base] + (rest>0 ? rest*(a[base+1]-a[base]) : 0);
}

function normalizeField(valuesMap) {
  const vals = Object.values(valuesMap).filter(v => Number.isFinite(v));
  if(!vals.length) return { map: valuesMap, min:0, max:1 };
  const mn = Math.min(...vals);
  const mx = Math.max(...vals);
  const scale = (mx>mn)?(mx-mn):1;
  const out = {};
  for(const [k,v] of Object.entries(valuesMap)) out[k] = (v - mn) / scale;
  return { map: out, min: mn, max: mx };
}

function normalizeData(graph, fieldObj, invertDefault){
  const nodes = (graph.nodes || []).map(n => ({
    id: n.id, name: n.label || n.id, kind: n.kind || 'default', meta: n.meta || {}
  }));
  const links = (graph.edges || graph.links || []).map(e => ({
    source: e.src || e.source, target: e.dst || e.target, kind: e.kind || 'link'
  }));
  // calc degree
  const deg = Object.create(null);
  links.forEach(l => { deg[l.source]=(deg[l.source]||0)+1; deg[l.target]=(deg[l.target]||0)+1; });
  nodes.forEach(n => n.degree = deg[n.id] || 0);

  // field map
  const fieldName = (fieldObj.field && fieldObj.field.name) ? fieldObj.field.name : (fieldObj.name || "field");
  const rawMap = (fieldObj.values && typeof fieldObj.values === 'object') ? fieldObj.values : {};
  const norm = normalizeField(rawMap);
  const inv = !!invertDefault;
  const map = norm.map;
  // attach field value normalized (0..1)
  nodes.forEach(n => {
    const v = map[n.id];
    n.field = (typeof v === 'number' && Number.isFinite(v)) ? clamp01(inv ? 1 - v : v) : null;
  });

  // legend ticks use raw (pre-normalization) stats for interpretability
  const rawVals = Object.values(rawMap).filter(v => Number.isFinite(v));
  const L = {
    fieldName,
    rawMin: rawVals.length? Math.min(...rawVals) : 0,
    rawMid: rawVals.length? quantile(rawVals, 0.5) : 0.5,
    rawMax: rawVals.length? Math.max(...rawVals) : 1,
    nonNull: rawVals.length
  };

  return { nodes, links, legend: L };
}

const RAW = normalizeData(GRAPH, FIELD, SETTINGS.invert_default);
const gStats = document.getElementById('stats');
const gLabels = document.getElementById('labels');
const gSizeBy = document.getElementById('sizeBy');
const gThresh = document.getElementById('thresh');
const gThreshV = document.getElementById('threshV');
const gTopK = document.getElementById('topk');
const gTopKV = document.getElementById('topkV');
const gInvert = document.getElementById('invert');
const gReset = document.getElementById('reset');

document.getElementById('legendTitle').textContent = `field: ${RAW.legend.fieldName}`;
document.getElementById('tickMin').textContent = RAW.legend.rawMin.toFixed(3);
document.getElementById('tickMid').textContent = RAW.legend.rawMid.toFixed(3);
document.getElementById('tickMax').textContent = RAW.legend.rawMax.toFixed(3);
gSizeBy.value = SETTINGS.size_by;
gInvert.checked = SETTINGS.invert_default === true;

function colorFor(node){
  if(node.field == null) return COLORS.default;
  return rampTurbo(node.field);
}
function nodeSize(node){
  if(gSizeBy.value === 'field'){
    const f = node.field==null?0.2:(0.2 + 1.2*node.field);
    return 3 + 10*f;
  } else {
    return 3 + Math.log2(2 + node.degree);
  }
}

function buildGraph(){
  const el = document.getElementById('scene');
  const Graph = ForceGraph3D()(el)
    .graphData({nodes: RAW.nodes, links: RAW.links})
    .nodeOpacity(0.95)
    .nodeRelSize(4)
    .linkOpacity(0.12)
    .linkColor(() => 'rgba(255,255,255,0.35)')
    .backgroundColor('#0b0e14')
    .nodeColor(n => colorFor(n))
    .onNodeHover(n => { el.style.cursor = n ? 'pointer' : null; })
    .onNodeClick((node) => {
      const dist = 80;
      const camera = Graph.camera();
      const controls = Graph.controls();
      const vec = new THREE.Vector3(node.x, node.y, node.z);
      const dir = vec.sub(camera.position).normalize();
      const newPos = vec.add(dir.multiplyScalar(-dist));
      controls.target.set(node.x, node.y, node.z);
      camera.position.set(newPos.x, newPos.y, newPos.z);
    });

  const haveSprite = typeof SpriteText !== 'undefined';
  Graph.nodeThreeObject(node => {
    if (!gLabels.checked || !haveSprite) {
      const geom = new THREE.SphereGeometry(Math.max(2, 1 + Math.log2(2+node.degree)), 16, 16);
      const mat = new THREE.MeshBasicMaterial({ color: colorFor(node) });
      return new THREE.Mesh(geom, mat);
    }
    const txt = (node.name || node.id) + (node.field!=null? ` (${node.field.toFixed(2)})` : '');
    const sprite = new SpriteText(txt);
    sprite.material.depthWrite = false;
    sprite.color = '#' + colorFor(node).toString(16).padStart(6, '0');
    sprite.textHeight = 8 + Math.min(10, node.degree);
    return sprite;
  });

  function updateStats(nodes, links){
    const nn = nodes? nodes.length : RAW.nodes.length;
    const ee = links? links.length : RAW.links.length;
    gStats.textContent = `Nodes: ${nn} • Edges: ${ee}`;
  }
  updateStats();

  function applyFilter(){
    const thr = parseFloat(gThresh.value || "0");
    const topkPercent = parseInt(gTopK.value || "100", 10);
    gThreshV.textContent = thr.toFixed(2);
    gTopKV.textContent = topkPercent>=100 ? "all" : `${topkPercent}%`;

    let keep = RAW.nodes;
    if(thr > 0){
      keep = keep.filter(n => (n.field ?? 0) >= thr);
    }
    if(topkPercent < 100){
      const sorted = [...keep].sort((a,b) => (b.field??-1) - (a.field??-1));
      const k = Math.max(1, Math.floor(sorted.length * (topkPercent/100)));
      keep = sorted.slice(0, k);
    }
    const keepSet = new Set(keep.map(n => n.id));
    const links = RAW.links.filter(l => keepSet.has(l.source) && keepSet.has(l.target));
    Graph.graphData({nodes: keep, links});
    updateStats(keep, links);
  }

  gThresh.addEventListener('input', applyFilter);
  gTopK.addEventListener('input', applyFilter);
  gLabels.addEventListener('change', () => Graph.refresh());
  gSizeBy.addEventListener('change', () => Graph.nodeRelSize(4).refresh());
  gInvert.addEventListener('change', () => {
    // re-apply inversion and rebuild field scale
    RAW.nodes.forEach(n => {
      if(n.field == null) return;
      n.field = 1.0 - n.field;
    });
    applyFilter();
  });
  gReset.addEventListener('click', () => Graph.zoomToFit(400, 60, node => true));

  // initial
  setTimeout(() => Graph.zoomToFit(400, 80, node => node.degree >= 0), 250);
  return Graph;
}

buildGraph();
</script>
</body>
</html>
"""

def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def _load_graph(repo_root: Path, graph_input: Optional[Path]) -> Dict[str, Any]:
    if graph_input and graph_input.exists():
        return _load_json(graph_input)
    artifact = (repo_root / ".glx" / "graphs" / "project_graph.json")
    if artifact.exists():
        return _load_json(artifact)
    raise RuntimeError(
        "Nie znaleziono project_graph.json. Zbuduj najpierw artefakt:\n"
        "  python -m glitchlab.analysis.project_graph build --repo-root . --write"
    )

def _load_field(field_input: Path) -> Dict[str, Any]:
    if not field_input.exists():
        raise RuntimeError(f"Nie znaleziono pliku pola: {field_input}")
    obj = _load_json(field_input)
    # akceptujemy {"values":{nid:val}, "field":{name:...}} lub {"name":...,"values":...}
    if "values" not in obj:
        raise RuntimeError("Plik pola musi zawierać sekcję 'values' (mapa node_id->value).")
    return obj

def _write_html(payload_graph: Dict[str, Any],
                payload_field: Dict[str, Any],
                invert_default: bool,
                size_by: str,
                out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_graph = json.dumps({
        "nodes": payload_graph.get("nodes", []),
        "edges": payload_graph.get("edges", payload_graph.get("links", [])),
        "meta": payload_graph.get("meta", {})
    }, ensure_ascii=False)
    data_field = json.dumps(payload_field, ensure_ascii=False)
    settings = json.dumps({
        "invert_default": bool(invert_default),
        "size_by": ("field" if size_by == "field" else "degree")
    }, ensure_ascii=False)
    html = (HTML_TEMPLATE
            .replace("__DATA_GRAPH__", data_graph)
            .replace("__DATA_FIELD__", data_field)
            .replace("__DATA_SETTINGS__", settings))
    out_path.write_text(html, encoding="utf-8")
    return out_path

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="3D wizualizacja pola na grafie projektu")
    ap.add_argument("--repo-root", "--repo", dest="repo_root", default=".", help="root repo (ten z katalogiem .glx/)")
    ap.add_argument("--graph-input", help="ścieżka do project_graph.json (opcjonalnie)")
    ap.add_argument("--field-input", required=True, help="ścieżka do fields/<name>.json (np. dist_shz.json)")
    ap.add_argument("--invert", action="store_true", help="odwróć skalę pola (np. dla dystansu)")
    ap.add_argument("--size-by", choices=["degree", "field"], default="degree", help="rozmiar węzła wg stopnia lub wartości pola")
    ap.add_argument("--output", default=None, help="ścieżka wyjściowa .html (domyślnie .glx/graphs/project_graph_fields_3d.html)")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    g_input = Path(args.graph_input).resolve() if args.graph_input else None
    f_input = Path(args.field_input).resolve()
    graph = _load_graph(root, g_input)
    field = _load_field(f_input)
    default_out = root / ".glx" / "graphs" / "project_graph_fields_3d.html"
    out = Path(args.output).resolve() if args.output else default_out
    res = _write_html(graph, field, args.invert, args.size_by, out)
    print(str(res))
    return 0

if __name__ == "__main__":
    sys.exit(main())
