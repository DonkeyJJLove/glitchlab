# glx/tools/project_graph_3d.py
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Any, Dict, Optional

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Project Graph 3D</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  html, body { margin:0; padding:0; height:100%; background:#0b0e14; color:#e6e6e6; font-family: Arial, sans-serif; }
  #scene { width:100%; height:100%; }
  .hud {
    position: fixed; top: 8px; left: 8px; right: 8px; display:flex; gap:12px; align-items:center;
    background: rgba(0,0,0,0.35); border:1px solid rgba(255,255,255,0.15); padding:8px 10px; border-radius:8px;
    backdrop-filter: blur(4px);
  }
  .hud .badge { padding:2px 6px; border-radius:6px; background:#1e2430; border:1px solid #2b3242; }
  .hud button, .hud select, .hud input[type="checkbox"] {
    background:#1a1f2b; border:1px solid #2b3242; color:#e6e6e6; border-radius:6px; padding:5px 8px;
  }
  .hud a { color:#93c5fd; text-decoration:none; }
</style>
<!-- One and only one THREE build -->
<script src="https://unpkg.com/three@0.150.1/build/three.min.js"></script>
<!-- SpriteText must be loaded BEFORE 3d-force-graph in UMD mode -->
<script src="https://unpkg.com/three-spritetext@1.9.4/dist/three-spritetext.min.js"></script>
<script src="https://unpkg.com/3d-force-graph@1.73.2/dist/3d-force-graph.min.js"></script>
</head>
<body>
<div class="hud">
  <div class="badge" id="stats">Nodes: 0 • Edges: 0</div>
  <label><input type="checkbox" id="labels" checked/> labels</label>
  <label>filter kind:
    <select id="kindFilter">
      <option value="">(all)</option>
      <option>module</option>
      <option>file</option>
      <option>func</option>
      <option>topic</option>
      <option>name</option>
    </select>
  </label>
  <button id="reset">reset view</button>
  <div style="margin-left:auto"><small>GlitchLab • 3D Graph</small></div>
</div>
<div id="scene"></div>

<script>
const DATA = __DATA_JSON__;

const COLORS = {
  module: 0x60a5fa,  // blue-400
  file:   0x34d399,  // emerald-400
  func:   0xf472b6,  // pink-400
  topic:  0xfbbf24,  // amber-400
  name:   0x9ca3af,  // gray-400 (placeholder for unresolved call targets)
  default:0x93c5fd
};

function normalizeData(payload) {
  // Map transport {nodes:[{id,kind,label,meta}], edges:[{src,dst,kind}]} -> 3d-force-graph {nodes:[{id,name,kind,...}], links:[{source,target}]}
  const nodes = (payload.nodes || []).map(n => ({
    id: n.id,
    name: n.label || n.id,
    kind: n.kind || 'default',
    meta: n.meta || {},
  }));
  const links = (payload.edges || []).map(e => ({
    source: e.src,
    target: e.dst,
    kind: e.kind || 'link'
  }));
  // degree calc
  const deg = Object.create(null);
  links.forEach(l => { deg[l.source] = (deg[l.source]||0)+1; deg[l.target] = (deg[l.target]||0)+1; });
  nodes.forEach(n => { n.degree = deg[n.id] || 0; });
  return { nodes, links };
}

const RAW = normalizeData(DATA);
const gStats = document.getElementById('stats');
const gLabels = document.getElementById('labels');
const gKind = document.getElementById('kindFilter');
const gReset = document.getElementById('reset');

function colorFor(node) {
  return COLORS[node.kind] || COLORS.default;
}

function buildGraph(data) {
  const el = document.getElementById('scene');
  const Graph = ForceGraph3D()(el)
    .graphData(data)
    .nodeAutoColorBy('kind') // backup palette, we override below
    .nodeColor(n => colorFor(n))
    .nodeOpacity(0.95)
    .nodeRelSize(4)
    .linkOpacity(0.12)
    .linkColor(() => 'rgba(255,255,255,0.35)')
    .backgroundColor('#0b0e14')
    .onNodeHover((n) => {
      el.style.cursor = n ? 'pointer' : null;
    })
    .onNodeClick((node, event) => {
      // focus
      const dist = 80;
      const camera = Graph.camera();
      const controls = Graph.controls();
      const vec = new THREE.Vector3(node.x, node.y, node.z);
      const dir = vec.sub(camera.position).normalize();
      const newPos = vec.add(dir.multiplyScalar(-dist));
      controls.target.set(node.x, node.y, node.z);
      camera.position.set(newPos.x, newPos.y, newPos.z);
    });

  // node object (SpriteText or sphere fallback)
  const haveSprite = typeof SpriteText !== 'undefined';
  Graph.nodeThreeObject(node => {
    if (!gLabels.checked || !haveSprite) {
      const geom = new THREE.SphereGeometry(Math.max(2, 1 + Math.log2(2+node.degree)), 16, 16);
      const mat = new THREE.MeshBasicMaterial({ color: colorFor(node) });
      return new THREE.Mesh(geom, mat);
    }
    const sprite = new SpriteText(node.name);
    sprite.material.depthWrite = false;
    sprite.color = '#' + colorFor(node).toString(16).padStart(6, '0');
    sprite.textHeight = 8 + Math.min(10, node.degree); // slightly scale with degree
    return sprite;
  });

  // HUD stats
  function updateStats(subset) {
    const nn = subset ? subset.nodes.length : data.nodes.length;
    const ee = subset ? subset.links.length : data.links.length;
    gStats.textContent = `Nodes: ${nn} • Edges: ${ee}`;
  }
  updateStats();

  // filtering
  function applyFilter() {
    const k = gKind.value;
    if (!k) {
      Graph.graphData(data);
      updateStats();
      return;
    }
    const keep = new Set(data.nodes.filter(n => n.kind === k).map(n => n.id));
    const nodes = data.nodes.filter(n => keep.has(n.id));
    const links = data.links.filter(l => keep.has(l.source) && keep.has(l.target));
    const subset = { nodes, links };
    Graph.graphData(subset);
    updateStats(subset);
  }
  gKind.addEventListener('change', applyFilter);
  gLabels.addEventListener('change', () => Graph.refresh());
  gReset.addEventListener('click', () => {
    Graph.zoomToFit(400, 60, node => true);
  });

  // initial focus
  setTimeout(() => Graph.zoomToFit(400, 80, node => node.degree >= 0), 200);
  return Graph;
}

const graph = buildGraph(RAW);
</script>
</body>
</html>
"""


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _load_or_build_graph(repo_root: Path, input_path: Optional[Path]) -> Dict[str, Any]:
    # 1) prefer explicit --input
    if input_path and input_path.exists():
        return _load_json(input_path)

    # 2) try standard artifact
    artifact = (repo_root / ".glx" / "graphs" / "project_graph.json")
    if artifact.exists():
        return _load_json(artifact)

    # 3) try to build using analysis.project_graph
    try:
        from glitchlab.analysis.project_graph import build_project_graph, save_project_graph  # type: ignore
    except Exception:
        try:
            # relative import fallback if running outside package context
            from analysis.project_graph import build_project_graph, save_project_graph  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Nie znaleziono artefaktu project_graph.json i nie mogę zbudować grafu, "
                f"bo nie udało się zaimportować 'glitchlab.analysis.project_graph' ani 'analysis.project_graph'.\n"
                "➡ Upewnij się, że uruchamiasz z ROOT projektu (katalog zawiera folder 'glitchlab/'), "
                "albo podaj --input do gotowego JSON.\n"
                "Przykład budowy artefaktu:\n"
                "  python -m glitchlab.analysis.project_graph build --repo-root . --write"
            ) from e

    g = build_project_graph(repo_root)
    save_project_graph(g, repo_root)
    return {
        "version": g.version,
        "meta": g.meta,
        "nodes": [
            {"id": n.id, "kind": n.kind, "label": n.label, "meta": n.meta}
            for n in sorted(g.nodes.values(), key=lambda x: x.id)
        ],
        "edges": [{"src": e.src, "dst": e.dst, "kind": e.kind} for e in g.edges],
    }


def _write_html(payload: Dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # wstawiamy JSON „inline”
    data_json = json.dumps(payload, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Render 3D Project Graph (Force-Graph + Three.js)")
    ap.add_argument("--repo-root", "--repo", dest="repo_root", default=".",
                    help="katalog root projektu (ten z folderem glitchlab/)")
    ap.add_argument("--input", help="ścieżka do project_graph.json (opcjonalnie)")
    ap.add_argument("--output", default=None,
                    help="ścieżka wyjściowa .html (domyślnie .glx/graphs/project_graph_3d.html)")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    input_path = Path(args.input).resolve() if args.input else None
    payload = _load_or_build_graph(root, input_path)

    default_out = root / ".glx" / "graphs" / "project_graph_3d.html"
    out = Path(args.output).resolve() if args.output else default_out
    res = _write_html(payload, out)
    print(str(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
