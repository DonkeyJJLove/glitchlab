#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AST ⇄ Mozaika • 3D Dashboard (UMD, bez CORS)
Użycie (przykład):
  python -m glitchlab.glx.tools.ast_mosaic_dashboard --repo-root .\glitchlab --file core\astmap.py ^
         --rows 32 --cols 24 --out .\glitchlab\.glx\graphs\ast_mosaic_dashboard.html

Co pokazuje:
- Widok AST (graf wywołań / definicji) w 3D — ForceGraph + etykiety.
- Widok Mozaiki (3D grid z kolumnami-kostkami): rzędy=linie, kolumny~wcięcia (indenty),
  wysokość/kolor = wybrana metryka (def/call/control/len) lub miks wagowy.

Stałe menu (po lewej) — żywe przełączanie:
- Dataset: AST | Mosaic
- Mosaic: rows, cols, height scale, wybór metryki lub miks (def/call/ctrl/len) z suwakami wag
- Etykiety (labels), filtr top-K, próg
- Legenda min/med/max aktualizowana dla aktywnej metryki

Wszystko działa z samym stdlib + UMD CDN (Three, SpriteText, ForceGraph).
"""

from __future__ import annotations
import argparse, json, sys, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>AST ⇄ Mosaic • 3D Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  :root {
    --bg:#0b0e14; --panel:#0e1420; --panel2:#121a2b; --bdr:#2b3242; --fg:#e6e6e6; --mut:#9ca3af;
    --accent:#60a5fa; --ok:#34d399; --warn:#fbbf24; --pink:#f472b6;
  }
  html,body {height:100%;margin:0;background:var(--bg);color:var(--fg);font-family:Inter,system-ui,Arial,sans-serif;}
  .root {display:flex;height:100%;width:100%;}
  .side {width:320px;min-width:280px;max-width:420px;background:linear-gradient(180deg,var(--panel),var(--panel2));
         border-right:1px solid var(--bdr);padding:14px 14px 10px;box-sizing:border-box;overflow:auto}
  .side h1{margin:0 0 10px;font-size:16px;color:#cbd5e1}
  .side h2{margin:14px 0 6px;font-size:13px;color:#cbd5e1;border-bottom:1px dashed #243048;padding-bottom:4px}
  .side .row{display:flex;align-items:center;gap:8px;margin:6px 0}
  .side label{font-size:13px;color:#cbd5e1}
  .side input[type=range], .side select, .side input[type=number]{
    width:100%;background:#111827;border:1px solid var(--bdr);color:var(--fg);border-radius:6px;padding:4px 6px;
  }
  .side input[type=checkbox]{transform:translateY(1px)}
  .badge{display:inline-block;padding:2px 6px;border-radius:6px;background:#1e2430;border:1px solid var(--bdr);font-size:12px;color:#cbd5e1}
  .help{font-size:12px;color:var(--mut);line-height:1.35}
  .legend{font-size:12px;color:#cbd5e1;margin-top:8px}
  .bar{height:10px;border-radius:6px;border:1px solid rgba(255,255,255,.15);
       background:linear-gradient(90deg,#0d0887,#6a00a8,#b12a90,#e16462,#fca636,#f0f921)}
  .ticks{display:flex;justify-content:space-between;color:#9ca3af}
  .scene{flex:1;position:relative}
  #stage{position:absolute;inset:0}
  .hud{position:absolute;top:8px;left:8px;right:8px;display:flex;gap:10px;align-items:center;
       background:rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.15);padding:6px 8px;border-radius:8px;backdrop-filter:blur(4px)}
</style>
<!-- UMD scripts (no ESM/CORS issues) -->
<script src="https://unpkg.com/three@0.150.1/build/three.min.js"></script>
<script src="https://unpkg.com/three-spritetext@1.9.4/dist/three-spritetext.min.js"></script>
<script src="https://unpkg.com/3d-force-graph@1.73.2/dist/3d-force-graph.min.js"></script>
</head>
<body>
<div class="root">
  <div class="side">
    <h1>AST ⇄ Mosaic • 3D</h1>
    <div class="row">
      <span class="badge" id="kpi-file"></span>
    </div>
    <div class="row">
      <span class="badge" id="kpi-stats">Nodes: 0 • Edges: 0</span>
    </div>

    <h2>Dataset</h2>
    <div class="row"><label><input type="radio" name="dataset" value="ast" id="ds-ast" checked> AST (call/defs graph)</label></div>
    <div class="row"><label><input type="radio" name="dataset" value="mosaic" id="ds-mosaic"> Mosaic (indent×lines)</label></div>
    <div class="row help">AST: graf definicji i (lokalnych) wywołań. Mosaic: rzędy≈linie, kolumny≈poziom wcięcia; kolor/wysokość = metryka.</div>

    <h2>Mosaic controls</h2>
    <div class="row"><label style="width:90px">rows</label><input id="rows" type="range" min="8" max="128" step="1"><span id="rowsV" class="badge">–</span></div>
    <div class="row"><label style="width:90px">cols</label><input id="cols" type="range" min="8" max="64" step="1"><span id="colsV" class="badge">–</span></div>
    <div class="row"><label style="width:90px">height</label><input id="hscale" type="range" min="0.2" max="3" step="0.05"><span id="hscaleV" class="badge">–</span></div>
    <div class="row"><label style="width:90px">metric</label>
      <select id="metric">
        <option value="mix">mix (weighted)</option>
        <option value="def">defs density</option>
        <option value="call">calls density</option>
        <option value="ctrl">control density</option>
        <option value="len">line length</option>
      </select>
    </div>
    <div class="row help">„density” liczone na oknie bloku (rzęd+kol), normalizowane do [0,1] w danym widoku.</div>
    <div class="row"><label style="width:90px">w(def)</label><input id="w-def" type="range" min="0" max="2" step="0.05"><span id="w-defV" class="badge">–</span></div>
    <div class="row"><label style="width:90px">w(call)</label><input id="w-call" type="range" min="0" max="2" step="0.05"><span id="w-callV" class="badge">–</span></div>
    <div class="row"><label style="width:90px">w(ctrl)</label><input id="w-ctrl" type="range" min="0" max="2" step="0.05"><span id="w-ctrlV" class="badge">–</span></div>
    <div class="row"><label style="width:90px">w(len)</label><input id="w-len" type="range" min="0" max="2" step="0.05"><span id="w-lenV" class="badge">–</span></div>

    <h2>View</h2>
    <div class="row"><label><input type="checkbox" id="labels" checked> labels</label></div>
    <div class="row"><label style="width:90px">threshold</label><input id="thr" type="range" min="0" max="1" step="0.01"><span id="thrV" class="badge">–</span></div>
    <div class="row"><label style="width:90px">top-K %</label><input id="topk" type="range" min="1" max="100" step="1" value="100"><span id="topkV" class="badge">all</span></div>

    <h2>Legend</h2>
    <div class="bar"></div>
    <div class="ticks"><span id="lg-min">0</span><span id="lg-med">0.5</span><span id="lg-max">1</span></div>
    <div class="legend" id="lg-caption">metric: –</div>

    <h2>About</h2>
    <div class="help">
      • AST: węzły = definicje (function/class); krawędzie = lokalne call-y („A→B”).<br/>
      • Mosaic: siatka (rows×cols), kolumna≈poziom wcięcia linii; kolor/wysokość = mix(def,call,ctrl,len).<br/>
      • Wszystkie przeliczenia dzieją się w przeglądarce bez reloadu strony.
    </div>
  </div>
  <div class="scene">
    <div class="hud">
      <span class="badge">AST ⇄ Mosaic • GlitchLab</span>
      <span class="help" id="meta-info"></span>
    </div>
    <div id="stage"></div>
  </div>
</div>

<script>
const SRC = __DATA_SRC__;
const AST = __DATA_AST__;
const MOS = __DATA_MOSAIC__;
const CFG = __DATA_CFG__;

const q = (id)=>document.getElementById(id);
const DS_AST = q('ds-ast'), DS_MOS = q('ds-mosaic');
const KPI = q('kpi-stats'), KPI_FILE = q('kpi-file'), MI = q('meta-info');
const R = q('rows'), RV=q('rowsV'), C=q('cols'), CV=q('colsV'), HS=q('hscale'), HSV=q('hscaleV');
const M=q('metric'), WD=q('w-def'), WDV=q('w-defV'), WC=q('w-call'), WCV=q('w-callV'), WX=q('w-ctrl'), WXV=q('w-ctrlV'), WL=q('w-len'), WLV=q('w-lenV');
const LAB=q('labels'), THR=q('thr'), THRV=q('thrV'), TK=q('topk'), TKV=q('topkV'), LGMIN=q('lg-min'), LGMED=q('lg-med'), LGMAX=q('lg-max'), LGC=q('lg-caption');

KPI_FILE.textContent = SRC.file;
R.value = CFG.rows; RV.textContent = CFG.rows;
C.value = CFG.cols; CV.textContent = CFG.cols;
HS.value = CFG.hscale; HSV.textContent = CFG.hscale.toFixed(2);
M.value = CFG.metric;
WD.value = CFG.w.def; WDV.textContent = CFG.w.def.toFixed(2);
WC.value = CFG.w.call; WCV.textContent = CFG.w.call.toFixed(2);
WX.value = CFG.w.ctrl; WXV.textContent = CFG.w.ctrl.toFixed(2);
WL.value = CFG.w.len;  WLV.textContent = CFG.w.len.toFixed(2);
THR.value = 0; THRV.textContent = "0.00";
TK.value = 100; TKV.textContent = "all";
LAB.checked = true;

function clamp01(x){return x<0?0:(x>1?1:x);}
function quantile(a,q){
  if(!a.length) return 0; const s=[...a].sort((x,y)=>x-y);
  const p=(s.length-1)*q, b=Math.floor(p), r=p-b; return s[b]+(r>0? r*(s[b+1]-s[b]) : 0);
}
function rampTurbo(t){
  const r=Math.round(255*Math.min(1,Math.max(0,0.13572138+4.61539260*t-42.66032258*t*t+132.13108234*t*t*t-152.94239396*t*t*t*t+59.28637943*t*t*t*t*t)));
  const g=Math.round(255*Math.min(1,Math.max(0,0.09140261+2.30097675*t-4.51317663*t*t-2.00344700*t*t*t+18.22835057*t*t*t*t-14.12650088*t*t*t*t*t)));
  const b=Math.round(255*Math.min(1,Math.max(0,0.10667330+12.64194608*t-60.58204836*t*t+110.36276771*t*t*t-89.90310912*t*t*t*t+27.34824973*t*t*t*t*t)));
  return (r<<16)+(g<<8)+b;
}

function initAST(){
  const nodes = AST.nodes.map(n => ({...n}));
  const links = AST.links.map(l => ({...l}));
  const deg = Object.create(null);
  links.forEach(l=>{deg[l.source]=(deg[l.source]||0)+1;deg[l.target]=(deg[l.target]||0)+1;});
  nodes.forEach(n => n.degree = deg[n.id]||0);

  const el = document.getElementById('stage');
  el.innerHTML = "";
  const Graph = ForceGraph3D()(el)
    .graphData({nodes, links})
    .nodeOpacity(0.95)
    .nodeRelSize(4)
    .linkOpacity(0.12)
    .linkColor(()=> 'rgba(255,255,255,0.35)')
    .backgroundColor('#0b0e14')
    .nodeColor(n => n.kind==='class'?0x60a5fa: (n.kind==='func'?0xf472b6:0x93c5fd))
    .nodeThreeObject(node => {
      if(!LAB.checked || typeof SpriteText==='undefined'){
        const geom=new THREE.SphereGeometry(Math.max(2,1+Math.log2(2+node.degree)),16,16);
        const mat = new THREE.MeshBasicMaterial({ color: node.kind==='class'?0x60a5fa:(node.kind==='func'?0xf472b6:0x93c5fd) });
        return new THREE.Mesh(geom, mat);
      }
      const sprite = new SpriteText(node.name);
      sprite.material.depthWrite=false;
      sprite.color = node.kind==='class'?'#60a5fa':(node.kind==='func'?'#f472b6':'#93c5fd');
      sprite.textHeight = 8 + Math.min(10, node.degree);
      return sprite;
    })
    .onNodeClick(node=>{
      const dist=80, cam=Graph.camera(), ctl=Graph.controls();
      const v=new THREE.Vector3(node.x,node.y,node.z), dir=v.sub(cam.position).normalize(), np=v.add(dir.multiplyScalar(-dist));
      ctl.target.set(node.x,node.y,node.z); cam.position.set(np.x,np.y,np.z);
    });

  KPI.textContent = `Nodes: ${nodes.length} • Edges: ${links.length}`;
  MI.textContent = `AST: defs=${AST.meta.defs}, calls=${AST.meta.calls}, ctrl=${AST.meta.ctrl}, imports=${AST.meta.imports}`;
  setTimeout(()=>Graph.zoomToFit(400,80, n=>true), 250);
}

function computeMosaicValues(rows, cols, weights, metric){
  const N = rows*cols;
  const vals = new Array(N).fill(0);
  const M = MOS.map; // "r,c" -> {def,call,ctrl,len}
  let arr = [];
  for(let r=0;r<rows;r++){
    for(let c=0;c<cols;c++){
      const k = r+','+c;
      const o = M[k] || {def:0,call:0,ctrl:0,len:0};
      const v = (metric==='def')?o.def :
                (metric==='call')?o.call :
                (metric==='ctrl')?o.ctrl :
                (metric==='len')?o.len :
                (weights.def*o.def + weights.call*o.call + weights.ctrl*o.ctrl + weights.len*o.len) /
                Math.max(1e-9, (weights.def+weights.call+weights.ctrl+weights.len));
      vals[r*cols + c] = v;
      arr.push(v);
    }
  }
  // min-max normalization
  const mn = Math.min(...arr), mx=Math.max(...arr), scale = (mx>mn)?(mx-mn):1;
  const norm = vals.map(v => (v-mn)/scale);
  return {norm, mn, med: quantile(arr,0.5), mx};
}

let mosaicScene = null;
function initMosaic(){
  const rows = parseInt(R.value,10);
  const cols = parseInt(C.value,10);
  const weights = {def:parseFloat(WD.value), call:parseFloat(WC.value), ctrl:parseFloat(WX.value), len:parseFloat(WL.value)};
  const metric = M.value;
  const {norm, mn, med, mx} = computeMosaicValues(rows, cols, weights, metric);

  const el = document.getElementById('stage');
  el.innerHTML = "";
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, el.clientWidth/el.clientHeight, 0.1, 2000);
  camera.position.set(cols*0.8, rows*0.9, Math.max(120, rows+cols));
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(el.clientWidth, el.clientHeight);
  renderer.setClearColor(0x0b0e14, 1);
  el.appendChild(renderer.domElement);

  // lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));

  // grid of boxes
  const group = new THREE.Group();
  const hs = parseFloat(HS.value);
  const size = 0.9;
  for(let r=0;r<rows;r++){
    for(let c=0;c<cols;c++){
      const v = norm[r*cols+c];
      const h = Math.max(0.1, v*10*hs);
      const geom = new THREE.BoxGeometry(size, h, size);
      const col = rampTurbo(v);
      const mat = new THREE.MeshBasicMaterial({color: col});
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(c - cols/2, h/2, r - rows/2);
      group.add(mesh);
      if(LAB.checked && typeof SpriteText !== 'undefined' && v>0.85){
        const sp = new SpriteText(v.toFixed(2)); sp.material.depthWrite=false;
        sp.color = '#ffffff'; sp.textHeight = 2.5; sp.position.set(c - cols/2, h + 1.5, r - rows/2);
        group.add(sp);
      }
    }
  }
  scene.add(group);

  // basic orbit controls (naive)
  let isDragging=false, px=0, py=0, rx=0.6, ry=-0.6, dist=Math.max(40, Math.max(rows,cols)*1.2);
  function updateCam(){
    const x = dist*Math.cos(rx)*Math.cos(ry);
    const y = dist*Math.sin(ry);
    const z = dist*Math.sin(rx)*Math.cos(ry);
    camera.position.set(x,y,z);
    camera.lookAt(0,0,0);
  }
  updateCam();

  el.onmousedown = (e)=>{isDragging=true; px=e.clientX; py=e.clientY;}
  el.onmouseup = ()=>{isDragging=false;}
  el.onmousemove = (e)=>{
    if(!isDragging) return;
    const dx=(e.clientX-px)/200, dy=(e.clientY-py)/200; px=e.clientX; py=e.clientY;
    rx+=dx; ry += dy; ry = Math.max(-1.2, Math.min(1.2, ry)); updateCam();
  }
  el.onwheel = (e)=>{
    dist *= (e.deltaY>0)? 1.1 : 0.9; dist = Math.max(10, Math.min(2000, dist)); updateCam();
  }
  window.addEventListener('resize', ()=>{
    renderer.setSize(el.clientWidth, el.clientHeight);
    camera.aspect = el.clientWidth/el.clientHeight; camera.updateProjectionMatrix();
  });

  function animate(){ requestAnimationFrame(animate); renderer.render(scene, camera); }
  animate();

  KPI.textContent = `Cells: ${rows*cols} • metric: ${metric}`;
  MI.textContent = `Mosaic: mn=${mn.toFixed(3)} med=${med.toFixed(3)} mx=${mx.toFixed(3)} hs=${hs.toFixed(2)}`;
  LGMIN.textContent = mn.toFixed(3); LGMED.textContent = med.toFixed(3); LGMAX.textContent = mx.toFixed(3);
  LGC.textContent = `metric: ${metric}${metric==='mix'?' (weighted)': ''}`;
  mosaicScene = {scene, camera, renderer};
}

function refresh(){
  if(DS_AST.checked) initAST(); else initMosaic();
}

[DS_AST,DS_MOS,R,C,HS,M,WD,WC,WX,WL,LAB,THR,TK].forEach(el=>{
  el.addEventListener('input', ()=>{
    if(el===R) RV.textContent = R.value;
    if(el===C) CV.textContent = C.value;
    if(el===HS) HSV.textContent = parseFloat(HS.value).toFixed(2);
    WDV.textContent = parseFloat(WD.value).toFixed(2);
    WCV.textContent = parseFloat(WC.value).toFixed(2);
    WXV.textContent = parseFloat(WX.value).toFixed(2);
    WLV.textContent = parseFloat(WL.value).toFixed(2);
    THRV.textContent = parseFloat(THR.value).toFixed(2);
    const v = parseInt(TK.value,10); TKV.textContent = (v>=100?'all':v+'%');
    refresh();
  });
});

refresh();
</script>
</body>
</html>
"""

# ------------------------ Python side: data preparation ------------------------

def _read_source(repo_root: Path, file_arg: str) -> Tuple[str, str]:
    p = Path(file_arg)
    if not p.is_absolute():
        p = (repo_root / file_arg).resolve()
    if not p.exists():
        raise FileNotFoundError(f"file not found: {p}")
    return p.read_text(encoding="utf-8", errors="ignore"), str(p)

def _indent_level(line: str) -> int:
    # tabs -> 4-spaces heuristic
    s = line.replace("\t", "    ")
    return len(s) - len(s.lstrip(" "))

def _ast_model(src: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Bardzo lekki model AST (stdlib), odporny na Python 3.9 (bez ast.Match jeśli brak).
    - nodes: definicje (class/func) z prostymi identyfikatorami
    - links: call A->B, jeżeli B jest zdefiniowany w tym samym pliku (po nazwie prostej)
    """
    import ast
    Match = getattr(ast, "Match", ())  # 3.10+
    class V(ast.NodeVisitor):
        def __init__(self):
            self.scope: List[str] = []
            self.nodes: Dict[str, Dict[str, Any]] = {}
            self.links: List[Dict[str, str]] = []
            self.calls: List[Tuple[str, str, int]] = []  # (scope, name, line)
            self.ctrl = 0
            self.imports = 0
        def _cur(self) -> str: return ".".join(self.scope) if self.scope else "<module>"
        def visit_FunctionDef(self, n: ast.FunctionDef):
            q = (self._cur()+"."+n.name) if self.scope else n.name
            self.nodes[q] = {"id": q, "name": n.name, "kind":"func"}
            self.scope.append(n.name); self.generic_visit(n); self.scope.pop()
        def visit_AsyncFunctionDef(self, n: "ast.AsyncFunctionDef"):
            q = (self._cur()+"."+n.name) if self.scope else n.name
            self.nodes[q] = {"id": q, "name": n.name, "kind":"func"}
            self.scope.append(n.name); self.generic_visit(n); self.scope.pop()
        def visit_ClassDef(self, n: ast.ClassDef):
            q = (self._cur()+"."+n.name) if self.scope else n.name
            self.nodes[q] = {"id": q, "name": n.name, "kind":"class"}
            self.scope.append(n.name); self.generic_visit(n); self.scope.pop()
        def visit_Call(self, n: ast.Call):
            # próbujemy uzyskać proste Name lub Attribute(...).attr
            def dotted(x: ast.AST) -> str:
                if isinstance(x, ast.Name): return x.id
                if isinstance(x, ast.Attribute):
                    base = dotted(x.value)
                    return f"{base}.{x.attr}" if base else x.attr
                return ""
            name = dotted(n.func) or "call"
            self.calls.append((self._cur(), name.split(".")[-1], getattr(n, "lineno", 0)))
            self.generic_visit(n)
        def visit_Import(self, n: ast.Import): self.imports += 1; self.generic_visit(n)
        def visit_ImportFrom(self, n: ast.ImportFrom): self.imports += 1; self.generic_visit(n)
        def generic_visit(self, n: ast.AST):
            if isinstance(n, (ast.If, ast.For, ast.While, ast.With, ast.Try) + ((Match,) if Match else ())):
                self.ctrl += 1
            super().generic_visit(n)
    tree = ast.parse(src)
    v = V(); v.visit(tree)
    # prosty resolver call->def po samej nazwie
    defs_by_name = { d["name"]: d["id"] for d in v.nodes.values() if d["kind"] in ("func","class") }
    for scope, name, line in v.calls:
        if name in defs_by_name and scope != defs_by_name[name]:
            v.links.append({"source": scope if scope in v.nodes else list(v.nodes.keys())[0], "target": defs_by_name[name]})
    return (
        {"nodes": list(v.nodes.values()), "links": v.links,
         "meta": {"defs": sum(1 for x in v.nodes.values() if x["kind"]=="func") + sum(1 for x in v.nodes.values() if x["kind"]=="class"),
                  "calls": len(v.calls), "ctrl": v.ctrl, "imports": v.imports}},
        {"per_line_calls":[l for _,__,l in v.calls]}
    )

def _mosaic_from_src(src: str, rows: int, cols: int) -> Dict[str, Any]:
    """
    Grid: rzędy ~ linie (row = linia * rows / nlines), kolumny ~ poziom wcięcia (indent//2)
    Komórka gromadzi 4 liczniki: def/call/ctrl/len (zastępcze wskaźniki „gęstości”).
    """
    lines = src.splitlines()
    n = max(1, len(lines))
    # Analiza heurystyczna per-linia:
    #  - def: linia zaczyna się od 'def ' lub 'class '
    #  - call: obecność '(...' i nawiasów (przybliżenie) albo 'Name(' wzorcem regex
    #  - ctrl: if/for/while/with/try/except/match (bezpieczeństwo 3.9: 'match' tylko jako tekst)
    #  - len: długość linii (clamp)
    re_call = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\s*\(")
    re_ctrl = re.compile(r"\b(if|for|while|with|try|except|match|case)\b")
    grid: Dict[Tuple[int,int], Dict[str,float]] = {}
    for i, ln in enumerate(lines, start=1):
        r = min(rows-1, (i-1)*rows//n)
        indent = _indent_level(ln)
        c = min(cols-1, indent//2)
        cell = grid.setdefault((r,c), {"def":0.0,"call":0.0,"ctrl":0.0,"len":0.0})
        s = ln.lstrip()
        if s.startswith("def ") or s.startswith("class "): cell["def"] += 1.0
        if re_call.search(ln): cell["call"] += 1.0
        if re_ctrl.search(s): cell["ctrl"] += 1.0
        cell["len"] += max(0.0, min(1.0, len(ln)/120.0))
    # normalizacja lokalna do [0,1] per kanał
    chans = ["def","call","ctrl","len"]
    for ch in chans:
        vals = [v[ch] for v in grid.values()]
        mx = max(vals) if vals else 1.0
        for k in grid.keys():
            if mx>0: grid[k][ch] = grid[k][ch]/mx
    mp = { f"{r},{c}": grid.get((r,c), {"def":0,"call":0,"ctrl":0,"len":0}) for r in range(rows) for c in range(cols) }
    return {"map": mp, "meta":{"rows":rows,"cols":cols,"lines":n}}

def _write_html(out_path: Path, payload: Dict[str, Any]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = (HTML
      .replace("__DATA_SRC__", json.dumps(payload["src"], ensure_ascii=False))
      .replace("__DATA_AST__", json.dumps(payload["ast"], ensure_ascii=False))
      .replace("__DATA_MOSAIC__", json.dumps(payload["mosaic"], ensure_ascii=False))
      .replace("__DATA_CFG__", json.dumps(payload["cfg"], ensure_ascii=False)))
    out_path.write_text(html, encoding="utf-8")
    return out_path

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AST ⇄ Mosaic 3D Dashboard")
    ap.add_argument("--repo-root","--repo",dest="repo_root", default=".", help="root repo (np. katalog z glitchlab/)")
    ap.add_argument("--file", required=True, help="ścieżka pliku źródłowego (względna do --repo-root lub absolutna)")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--cols", type=int, default=24)
    ap.add_argument("--hscale", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="ścieżka wyjściowa HTML (domyślnie .glx/graphs/ast_mosaic_dashboard.html)")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    src, file_path = _read_source(root, args.file)
    ast_graph, _aux = _ast_model(src)
    mosaic = _mosaic_from_src(src, max(8,args.rows), max(8,args.cols))
    cfg = {"rows": max(8,args.rows), "cols": max(8,args.cols), "hscale": float(args.hscale),
           "metric":"mix", "w":{"def":1.2,"call":1.0,"ctrl":1.0,"len":0.4}}
    payload = {
        "src": {"file": file_path},
        "ast": ast_graph,
        "mosaic": mosaic,
        "cfg": cfg,
    }
    default_out = root/".glx"/"graphs"/"ast_mosaic_dashboard.html"
    out = Path(args.out).resolve() if args.out else default_out
    res = _write_html(out, payload)
    print(str(res))
    return 0

if __name__ == "__main__":
    sys.exit(main())
