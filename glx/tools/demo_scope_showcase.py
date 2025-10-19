# glx/tools/demo_scope_showcase.py
from __future__ import annotations
import json, sys, re, ast, textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
#  HTML (3D graf + overlay „aur” soczewek). Jedno ładowanie THREE + SpriteText.
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>GlitchLab • Meta-Lenses (Demo)</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  html, body { margin:0; padding:0; height:100%; background:#0b0e14; color:#e6e6e6; font-family: Inter, Arial, sans-serif; }
  #scene { width:100%; height:100%; }
  .hud {
    position: fixed; top: 8px; left: 8px; right: 8px; display:flex; gap:12px; align-items:center; flex-wrap: wrap;
    background: rgba(0,0,0,0.35); border:1px solid rgba(255,255,255,0.15); padding:8px 10px; border-radius:10px;
    backdrop-filter: blur(4px);
  }
  .hud .badge { padding:2px 8px; border-radius:8px; background:#1e2430; border:1px solid #2b3242; }
  .hud label { display:inline-flex; gap:6px; align-items:center; padding:2px 6px; border-radius:8px; background:#121621; border:1px solid #232a3a; }
  .legend { display:flex; gap:10px; align-items:center; margin-left:auto; }
  .dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
  .p { background:#22d3ee; } .m { background:#f472b6; } .f { background:#22c55e; } .fn { background:#fb923c; } .b { background:#f59e0b; }
</style>
<!-- UMD: jedna instancja THREE + SpriteText PRZED 3d-force-graph -->
<script src="https://unpkg.com/three@0.150.1/build/three.min.js"></script>
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
  <label><input type="checkbox" id="lensProject" checked /><span>project</span></label>
  <label><input type="checkbox" id="lensModule" checked /><span>module</span></label>
  <label><input type="checkbox" id="lensFile" checked /><span>file</span></label>
  <label><input type="checkbox" id="lensFunc" checked /><span>func</span></label>
  <label><input type="checkbox" id="lensBus" checked /><span>bus</span></label>
  <button id="reset">reset view</button>
  <div class="legend">
    <span class="badge">Lenses:</span>
    <span class="dot p"></span><small>project</small>
    <span class="dot m"></span><small>module</small>
    <span class="dot f"></span><small>file</small>
    <span class="dot fn"></span><small>func</small>
    <span class="dot b"></span><small>bus</small>
  </div>
</div>
<div id="scene"></div>

<script>
const PAYLOAD = __DATA_JSON__;

const COLORS = {
  module: 0x60a5fa,  // blue-400
  file:   0x34d399,  // emerald-400
  func:   0xf472b6,  // pink-400
  topic:  0xfbbf24,  // amber-400
  name:   0x9ca3af,  // gray-400
  default:0x93c5fd
};

const LENS_COLORS = {
  project: 0x22d3ee, // cyan-400
  module:  0xf472b6, // pink-400
  file:    0x22c55e, // green-500
  func:    0xfb923c, // orange-400
  bus:     0xf59e0b  // amber-500
};

function normalizeData(payload) {
  const nodes = (payload.nodes || []).map(n => ({
    id: n.id,
    name: n.label || n.id,
    kind: n.kind || 'default',
    meta: n.meta || {},
    lens: n.lens || {}
  }));
  const links = (payload.edges || []).map(e => ({
    source: e.src, target: e.dst, kind: e.kind || 'link'
  }));
  const deg = Object.create(null);
  links.forEach(l => { deg[l.source]=(deg[l.source]||0)+1; deg[l.target]=(deg[l.target]||0)+1; });
  nodes.forEach(n => n.degree = deg[n.id] || 0);
  return {nodes, links};
}

const RAW = normalizeData(PAYLOAD);

const el = document.getElementById('scene');
const gStats = document.getElementById('stats');
const gLabels = document.getElementById('labels');
const gKind = document.getElementById('kindFilter');
const gReset = document.getElementById('reset');
const toggles = {
  project: document.getElementById('lensProject'),
  module:  document.getElementById('lensModule'),
  file:    document.getElementById('lensFile'),
  func:    document.getElementById('lensFunc'),
  bus:     document.getElementById('lensBus'),
};

function colorFor(node) { return COLORS[node.kind] || COLORS.default; }

const Graph = ForceGraph3D()(el)
  .graphData(RAW)
  .nodeAutoColorBy('kind')
  .nodeColor(n => colorFor(n))
  .nodeOpacity(0.95)
  .nodeRelSize(4)
  .linkOpacity(0.12)
  .linkColor(() => 'rgba(255,255,255,0.35)')
  .backgroundColor('#0b0e14')
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

Graph.nodeThreeObjectExtend(true);

function makeHalo(radius, colorHex) {
  const geom = new THREE.SphereGeometry(radius, 16, 16);
  const mat = new THREE.MeshBasicMaterial({ color: colorHex, transparent: true, opacity: 0.25 });
  return new THREE.Mesh(geom, mat);
}

Graph.nodeThreeObject(node => {
  const group = new THREE.Group();
  const haveSprite = typeof SpriteText !== 'undefined';

  if (gLabels.checked && haveSprite) {
    const sprite = new SpriteText(node.name);
    sprite.material.depthWrite = false;
    sprite.color = '#' + colorFor(node).toString(16).padStart(6, '0');
    sprite.textHeight = 8 + Math.min(10, node.degree);
    group.add(sprite);
  } else {
    const geom = new THREE.SphereGeometry(Math.max(2, 1 + Math.log2(2+node.degree)), 16, 16);
    const mat = new THREE.MeshBasicMaterial({ color: colorFor(node) });
    group.add(new THREE.Mesh(geom, mat));
  }

  const lens = node.lens || {};
  const baseR = 6 + Math.min(10, node.degree);

  if (toggles.project.checked && lens.project) group.add(makeHalo(baseR+7, LENS_COLORS.project));
  if (toggles.module.checked  && lens.module)  group.add(makeHalo(baseR+5, LENS_COLORS.module));
  if (toggles.file.checked    && lens.file)    group.add(makeHalo(baseR+3, LENS_COLORS.file));
  if (toggles.func.checked    && lens.func)    group.add(makeHalo(baseR+1, LENS_COLORS.func));
  if (toggles.bus.checked     && lens.bus)     group.add(makeHalo(baseR+9, LENS_COLORS.bus));

  return group;
});

function updateStats(subset) {
  const nn = subset ? subset.nodes.length : RAW.nodes.length;
  const ee = subset ? subset.links.length : RAW.links.length;
  gStats.textContent = `Nodes: ${nn} • Edges: ${ee}`;
}
updateStats();

function applyFilter() {
  const k = gKind.value;
  if (!k) { Graph.graphData(RAW); updateStats(); return; }
  const keep = new Set(RAW.nodes.filter(n => n.kind === k).map(n => n.id));
  const nodes = RAW.nodes.filter(n => keep.has(n.id));
  const links = RAW.links.filter(l => keep.has(l.source) && keep.has(l.target));
  Graph.graphData({nodes, links});
  updateStats({nodes, links});
}

gKind.addEventListener('change', applyFilter);
gLabels.addEventListener('change', () => Graph.refresh());
Object.values(toggles).forEach(cb => cb.addEventListener('change', () => Graph.refresh()));
gReset.addEventListener('click', () => Graph.zoomToFit(400, 60, n => true));
setTimeout(() => Graph.zoomToFit(400, 80, n => n.degree >= 0), 200);
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
#  LOCAL GRAPH HELPERS (payload-based; zero importów z pakietu)
# ─────────────────────────────────────────────────────────────────────────────

TOPIC_TAG_RE = re.compile(r"#\s*glx:topic\.(publish|subscribe|request_reply)\s*=\s*([^\n#]+)", re.IGNORECASE)

def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def _default_artifact(repo_root: Path) -> Path:
    # Akceptujemy zarówno "<root>/.glx/graphs", jak i "glitchlab/.glx/graphs"
    a = repo_root / ".glx" / "graphs" / "project_graph.json"
    if a.exists(): return a
    b = repo_root / "glitchlab" / ".glx" / "graphs" / "project_graph.json"
    return b

def _norm(p: str) -> str:
    return p.replace("\\", "/")

def _neighbors(edges: List[Dict[str, str]], allowed: Set[str]) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = {}
    for e in edges:
        if e.get("kind") not in allowed:
            continue
        s, t = e.get("src"), e.get("dst")
        if not s or not t: continue
        adj.setdefault(s, set()).add(t)
        adj.setdefault(t, set()).add(s)
    return adj

def _match_node(n: Dict[str, Any], pats: List[str]) -> bool:
    if not pats: return False
    vals = [str(n.get("id","")), str(n.get("label","")), str(n.get("meta",{}).get("path",""))]
    for pat in pats:
        try:
            if re.search(pat, " ".join(vals)):
                return True
        except re.error:
            # traktuj jako podciąg
            if any(pat in v for v in vals):
                return True
    return False

def _bfs_ids(nodes: List[Dict[str,Any]], edges: List[Dict[str,str]],
             allowed_kinds: Set[str], allowed_edges: Set[str],
             center_pats: List[str], depth: int, max_nodes: int) -> Set[str]:
    # wybór kotwic
    anchors = [n["id"] for n in nodes
               if (n.get("kind") in allowed_kinds) and _match_node(n, center_pats)]
    if not anchors:
        # fallback: pierwszy węzeł dozwolonego typu o najwyższym "degree"
        deg: Dict[str, int] = {}
        for e in edges:
            if e.get("kind") in allowed_edges:
                s, t = e.get("src"), e.get("dst")
                deg[s] = deg.get(s, 0) + 1
                deg[t] = deg.get(t, 0) + 1
        best = None
        best_d = -1
        for n in nodes:
            if n.get("kind") in allowed_kinds:
                d = deg.get(n["id"], 0)
                if d > best_d:
                    best, best_d = n["id"], d
        anchors = [best] if best else []

    adj = _neighbors(edges, allowed_edges)
    keep: Set[str] = set()
    frontier: List[Tuple[str,int]] = [(a,0) for a in anchors]
    while frontier and len(keep) < max_nodes:
        nid, d = frontier.pop(0)
        if nid in keep: continue
        # sprawdzamy rodzaj węzła
        nk = next((n.get("kind") for n in nodes if n["id"] == nid), None)
        if nk not in allowed_kinds: continue
        keep.add(nid)
        if d < depth:
            for nb in adj.get(nid, ()):
                if nb not in keep:
                    frontier.append((nb, d+1))
        if len(keep) >= max_nodes: break
    return keep

def _embed_lenses(payload: Dict[str, Any], membership: Dict[str, Set[str]]) -> Dict[str, Any]:
    # do każdego node dodajemy n.lens{...}
    for n in payload.get("nodes", []):
        nid = n["id"]
        n["lens"] = {
            "project": nid in membership.get("project", set()),
            "module":  nid in membership.get("module", set()),
            "file":    nid in membership.get("file", set()),
            "func":    nid in membership.get("func", set()),
            "bus":     nid in membership.get("bus", set()),
        }
    return payload

# ─────────────────────────────────────────────────────────────────────────────
#  MINI DEMO PAKIET (opcjonalnie) + fallback budowy małego grafu demo
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_demo_pkg(repo_root: Path) -> List[Path]:
    base = repo_root / "glitchlab" / "core" / "demo_pkg"
    base.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    files: Dict[str, str] = {
        "__init__.py": "from .pipeline import load, process, save\n",
        "utils.py": textwrap.dedent("""\
            # glx:topic.subscribe=glx.demo.events
            def clean(data):
                return [d for d in data if d is not None]
            def transform(data):
                return [str(x).strip().upper() for x in data]
        """),
        "pipeline.py": textwrap.dedent("""\
            # glx:topic.publish=glx.demo.events
            from .utils import clean, transform
            def load():
                return ["  a ", None, "b", "  c"]
            def process(data):
                tmp = clean(data); out = transform(tmp); return out
            def save(items):
                for it in items: print("SAVE:", it)
        """),
        "bus.py": textwrap.dedent("""\
            # glx:topic.request_reply=glx.demo.req->glx.demo.res
            def handle(req: str) -> str:
                return f"ok:{req}"
        """),
    }
    for name, content in files.items():
        p = base / name
        if not p.exists():
            p.write_text(content, encoding="utf-8"); created.append(p)
    return created

def _module_id_from_path(repo_root: Path, p: Path) -> str:
    rel = _norm(str(p.resolve().relative_to(repo_root)))
    if rel.endswith(".py"): rel = rel[:-3]
    return rel.replace("/", ".")

def _parse_topics(file_path: Path) -> Dict[str, List[str]]:
    txt = file_path.read_text(encoding="utf-8", errors="ignore")
    topics = {"publish": [], "subscribe": [], "request_reply": []}
    for m in TOPIC_TAG_RE.finditer(txt):
        kind = m.group(1).lower().strip()
        raw = m.group(2).strip()
        vals = [v.strip() for v in re.split(r"[,\s]+", raw) if v.strip()]
        if kind == "request_reply":
            for v in vals:
                if "->" in v:
                    topics[kind].append(v)
        else:
            topics[kind].extend(vals)
    return topics

def _ast_demo_graph(repo_root: Path, demo_dir: Path) -> Dict[str, Any]:
    """
    Fallback: zbuduj mały graf tylko z demo_pkg (mod->file->func + calls + topic edges).
    """
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, str]] = []

    py_files = sorted([p for p in demo_dir.rglob("*.py")])
    for fp in py_files:
        mod = _module_id_from_path(repo_root, fp)
        mod_id = f"module:{mod}"
        file_id = f"file:{_norm(str(fp))}"
        # nody
        if mod_id not in nodes:
            nodes[mod_id] = {"id": mod_id, "kind":"module", "label": mod, "meta":{"path": _norm(str(fp))}}
        nodes[file_id] = {"id": file_id, "kind":"file", "label": fp.name, "meta":{"path": _norm(str(fp))}}
        edges.append({"src": mod_id, "dst": file_id, "kind":"define"})
        # AST
        try:
            tree = ast.parse(fp.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        # funkcje
        class V(ast.NodeVisitor):
            def __init__(self): self.scope: List[str]=[]; self.calls: List[Tuple[str,str]]= []
            def _cur(self): return ".".join(self.scope) if self.scope else "<module>"
            def visit_FunctionDef(self, n: ast.FunctionDef):
                q = ".".join([*self.scope, n.name]) if self.scope else n.name
                fn_id = f"func:{mod}:{q}"
                nodes[fn_id] = {"id": fn_id, "kind":"func", "label": f"{q}()", "meta":{"path": _norm(str(fp)), "line": n.lineno}}
                edges.append({"src": file_id, "dst": fn_id, "kind":"define"})
                self.scope.append(n.name); self.generic_visit(n); self.scope.pop()
            def visit_AsyncFunctionDef(self, n: ast.AsyncFunctionDef): self.visit_FunctionDef(n)  # noqa
            def visit_Call(self, n: ast.Call):
                callee = ""
                if isinstance(n.func, ast.Attribute): callee = n.func.attr
                elif isinstance(n.func, ast.Name): callee = n.func.id
                self.calls.append((self._cur(), callee))
                self.generic_visit(n)
        v = V(); v.visit(tree)
        for caller, callee in v.calls:
            src = file_id if caller == "<module>" else f"func:{mod}:{caller}"
            name_id = f"name:{callee}"
            nodes.setdefault(name_id, {"id": name_id, "kind":"name", "label": callee, "meta":{}})
            edges.append({"src": src, "dst": name_id, "kind":"call"})
        # topic tags
        topics = _parse_topics(fp)
        for t in topics.get("publish", []):
            tid = f"topic:{t}"
            nodes.setdefault(tid, {"id": tid, "kind":"topic", "label": t, "meta":{}})
            edges.append({"src": file_id, "dst": tid, "kind":"define"})
            edges.append({"src": file_id, "dst": tid, "kind":"link"})
        for t in topics.get("subscribe", []):
            tid = f"topic:{t}"
            nodes.setdefault(tid, {"id": tid, "kind":"topic", "label": t, "meta":{}})
            edges.append({"src": file_id, "dst": tid, "kind":"use"})
        for rr in topics.get("request_reply", []):
            a, b = rr.split("->", 1)
            ta, tb = f"topic:{a}", f"topic:{b}"
            nodes.setdefault(ta, {"id": ta, "kind":"topic", "label": a, "meta":{}})
            nodes.setdefault(tb, {"id": tb, "kind":"topic", "label": b, "meta":{}})
            edges.append({"src": file_id, "dst": ta, "kind":"define"})
            edges.append({"src": file_id, "dst": tb, "kind":"define"})
            edges.append({"src": ta, "dst": tb, "kind":"rpc"})
    payload = {
        "version":"v1",
        "meta":{"generated_by":"demo_scope_showcase", "repo_root": _norm(str(repo_root)),
                "nodes_count": len(nodes), "edges_count": len(edges)},
        "nodes": [nodes[k] for k in sorted(nodes.keys())],
        "edges": edges
    }
    return payload

# ─────────────────────────────────────────────────────────────────────────────
#  SOCZEWKI (payload → membership)
# ─────────────────────────────────────────────────────────────────────────────

LEVEL_ALLOWED_KINDS = {
    "project": {"module", "topic", "project"},
    "module":  {"module", "topic"},
    "file":    {"file", "module", "topic"},
    "func":    {"func", "topic"},
    "bus":     {"topic", "module", "file"},
}
ALLOWED_EDGES_DEFAULT = {"import","define","call","link","use","rpc"}

def _lens_membership(payload: Dict[str, Any], level: str, centers: List[str], depth: int, max_nodes: int=400) -> Set[str]:
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    allowed_kinds = LEVEL_ALLOWED_KINDS.get(level, {"module","file","func","topic"})
    return _bfs_ids(nodes, edges, allowed_kinds, ALLOWED_EDGES_DEFAULT, centers, depth, max_nodes)

def _compute_all_lenses(payload: Dict[str, Any], repo_root: Path, center_mod_sub: str, center_file_sub: str) -> Dict[str, Set[str]]:
    # Heurystyczne centry dla demo_pkg
    demo_mod = center_mod_sub  # np. "glitchlab.core.demo_pkg"
    demo_file_pat = center_file_sub  # np. "/core/demo_pkg/pipeline.py"
    demo_func_pat = demo_file_pat + ":process"  # tekstowy wzorzec; funkcje oznaczamy id "func:mod:qual"
    return {
        "project": _lens_membership(payload, "project", [demo_mod], depth=2),
        "module":  _lens_membership(payload, "module",  [demo_mod], depth=2),
        "file":    _lens_membership(payload, "file",    [demo_file_pat], depth=2),
        "func":    _lens_membership(payload, "func",    [demo_func_pat], depth=1),
        "bus":     _lens_membership(payload, "bus",     [r"^topic:"], depth=3),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  IO
# ─────────────────────────────────────────────────────────────────────────────

def _write_html(payload: Dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(payload, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    out_path.write_text(html, encoding="utf-8")
    return out_path

def _load_or_fallback(repo_root: Path, input_json: Optional[Path], ensure_demo: bool) -> Dict[str, Any]:
    # 1) --input
    if input_json and input_json.exists():
        return _load_json(input_json)
    # 2) standard artefakt
    art = _default_artifact(repo_root)
    if art.exists():
        return _load_json(art)
    # 3) fallback: jeśli wolno, zbuduj mini-graf TYLKO dla demo_pkg
    demo_dir = repo_root / "glitchlab" / "core" / "demo_pkg"
    if ensure_demo and demo_dir.exists():
        return _ast_demo_graph(repo_root, demo_dir)
    raise RuntimeError(
        "Brak wejścia. Podaj --input do project_graph.json albo uruchom z ROOT, "
        "gdzie istnieje '.glx/graphs/project_graph.json'. Możesz też dodać --write-demo, "
        "żebym zbudował mini-graf dla 'glitchlab/core/demo_pkg/'."
    )

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="GlitchLab • Meta-Lenses demo (3D) — bez zależności od importów pakietu")
    ap.add_argument("--repo-root", "--repo", dest="repo_root", default=".",
                    help="ROOT projektu (ten z folderem 'glitchlab/').")
    ap.add_argument("--input", help="Ścieżka do project_graph.json (opcjonalnie).")
    ap.add_argument("--output", help="HTML wyjściowy (domyślnie: .glx/graphs/demo_scope_showcase.html).")
    ap.add_argument("--write-demo", action="store_true",
                    help="Utwórz demo paket 'glitchlab/core/demo_pkg' jeśli nie istnieje (fallback budowy małego grafu).")
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    input_p = Path(args.input).resolve() if args.input else None

    # 1) opcjonalnie stwórz paczkę demo
    if args.write_demo:
        created = _ensure_demo_pkg(repo_root)
        if created:
            print("[demo] utworzono pliki:", *map(str, created), sep="\n  ")

    # 2) wczytaj lub zbuduj payload
    payload = _load_or_fallback(repo_root, input_p, ensure_demo=args.write_demo)

    # 3) podgraf (tylko demo_pkg) – żeby pokazać różne poziomy abstrakcji na małym wycinku
    sub_nodes: List[Dict[str, Any]] = []
    keep_ids: Set[str] = set()
    demo_substr = "/core/demo_pkg/"
    for n in payload.get("nodes", []):
        path = _norm(str(n.get("meta",{}).get("path","")))
        if demo_substr in path or ("demo_pkg" in str(n.get("label","")) and n.get("kind")=="module"):
            keep_ids.add(n["id"])
    # jeśli nic nie znaleźliśmy, pokazuj cały payload (użytkownik podał własny center)
    if not keep_ids:
        sub_nodes = payload.get("nodes", [])
        sub_edges = payload.get("edges", [])
    else:
        node_ids = keep_ids
        sub_nodes = [n for n in payload.get("nodes", []) if n["id"] in node_ids]
        sub_edges = [e for e in payload.get("edges", []) if e.get("src") in node_ids and e.get("dst") in node_ids]
        payload = {"version":payload.get("version","v1"), "meta":payload.get("meta",{}),
                   "nodes": sub_nodes, "edges": sub_edges}

    # 4) policz członkostwa soczewek i wstrzyknij do payload
    demo_mod = "glitchlab.core.demo_pkg"
    demo_file = _norm(str(repo_root / "glitchlab" / "core" / "demo_pkg" / "pipeline.py"))
    memberships = _compute_all_lenses(payload, repo_root, demo_mod, demo_file)
    payload = _embed_lenses(payload, memberships)

    # 5) zapis HTML
    default_out = (repo_root / ".glx" / "graphs" / "demo_scope_showcase.html")
    # jeśli graf w subfolderze glitchlab/.glx istnieje, preferuj go dla spójności
    alt_dir = (repo_root / "glitchlab" / ".glx" / "graphs")
    out = Path(args.output).resolve() if args.output else (alt_dir if alt_dir.exists() else default_out)
    if out.is_dir():
        out = out / "demo_scope_showcase.html"
    out_path = _write_html(payload, out)
    print(str(out_path))
    return 0

if __name__ == "__main__":
    sys.exit(main())
