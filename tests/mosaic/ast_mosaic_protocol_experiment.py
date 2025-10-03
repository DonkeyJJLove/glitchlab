# demos/ast_mosaic_protocol_experiment.py
# -*- coding: utf-8 -*-
"""
GlitchLab – AST×Mozaika 12x12 (hex) : kompresja AST + Φ/Ψ + protokół kontekstu
Uruchom:  python demos/ast_mosaic_protocol_experiment.py
Wymaga:   numpy, matplotlib
"""
from __future__ import annotations
import ast, json, math, hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

# ====== 0) USTAWIENIA ======
PLOT = True   # rysuj 2× subplot (AST kompasy/hex-grid)
ROWS, COLS = 12, 12  # plaster miodu ~ 12x12

# ====== 1) AST: PARSOWANIE + KOMPRESJA (Merkle-AST) ======
@dataclass
class AstNodeInfo:
    id: int
    kind: str
    label: str
    parent: Optional[int]
    depth: int
    children: List[int] = field(default_factory=list)
    hash: str = ""
    meta: Tuple[float, float, float, float, float, float] = (0,0,0,0,0,0)
    pos3d: Tuple[float, float, float] = (0.0,0.0,0.0)
    count: int = 1  # po kompresji (ile zwinęto poddrzew tego samego typu)

def node_label(n: ast.AST) -> str:
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)): return f"def {n.name}"
    if isinstance(n, ast.ClassDef): return f"class {n.name}"
    if isinstance(n, ast.Assign):
        t = n.targets[0] if n.targets else None
        return f"{getattr(t,'id','assign')} = …"
    if isinstance(n, ast.Name): return n.id
    return type(n).__name__

def build_ast_info(src: str) -> Dict[int, AstNodeInfo]:
    root = ast.parse(src)
    nodes: Dict[int, AstNodeInfo] = {}
    nid = 0
    def add(a: ast.AST, depth: int, parent: Optional[int]) -> int:
        nonlocal nid
        me = nid; nid+=1
        k = a.__class__.__name__
        nodes[me] = AstNodeInfo(me, k, node_label(a), parent, depth)
        if parent is not None:
            nodes[parent].children.append(me)
        for ch in ast.iter_child_nodes(a):
            add(ch, depth+1, me)
        return me
    add(root, 0, None)

    # pozycje: radial by depth (kompas 3D)
    by_depth: Dict[int, List[int]] = {}
    for i,n in nodes.items(): by_depth.setdefault(n.depth,[]).append(i)
    for d,ids in by_depth.items():
        ids.sort()
        R = 6.0 + 2.0*d
        for j,i in enumerate(ids):
            ang = 2*math.pi*j/max(1,len(ids))
            nodes[i].pos3d = (R*math.cos(ang), R*math.sin(ang), 2.0*d)

    # meta-heurystyki (spójne z Twoimi demo-metaglifami)
    rng = np.random.default_rng(1337)
    for n in nodes.values():
        L,S,Sel,Stab,Cau,H = rng.uniform(0.25,0.85,size=6)
        if n.kind in ("If","Compare"): Sel,Cau = 0.85,0.80
        if n.kind in ("Expr","Call"): L,S = 0.65,0.55
        if n.kind in ("Return",): Stab = 0.90
        if n.kind in ("Assign",): Sel = 0.70
        nodes[n.id].meta = (float(L),float(S),float(Sel),float(Stab),float(Cau),float(H))

    # Merkle-hash poddrzew
    def merkle(i: int) -> str:
        n = nodes[i]
        child_hashes = "".join(sorted(merkle(c) for c in n.children))
        s = f"{n.kind}|{n.label}|{child_hashes}"
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
        nodes[i].hash = h
        return h
    _ = merkle(0)

    # kompresja: policz powtórzenia hashy; zbij „izomorficzne” poddrzewa
    freq: Dict[str,int] = {}
    for n in nodes.values(): freq[n.hash] = freq.get(n.hash,0)+1
    for n in nodes.values(): n.count = freq[n.hash]

    return nodes

def d_ast_distance(nodesA: Dict[int,AstNodeInfo], nodesB: Dict[int,AstNodeInfo]) -> float:
    # lekka odległość: różnica rozkładu hashy + różnica głębokości
    from collections import Counter
    ca = Counter(n.hash for n in nodesA.values())
    cb = Counter(n.hash for n in nodesB.values())
    keys = set(ca)|set(cb)
    dist_hash = sum(abs(ca[k]-cb[k]) for k in keys)
    da = np.array([n.depth for n in nodesA.values()], float)
    db = np.array([n.depth for n in nodesB.values()], float)
    dist_depth = abs(da.mean() - (db.mean() if db.size else 0.0))
    return float(dist_hash + 0.05*dist_depth)

# ====== 2) HEKS-MOZAIKA 12×12 ======
# axial (q,r) → 2D; generujemy maski i cechy
@dataclass
class Hex:
    q: int; r: int
    center: Tuple[float,float]
    feats: Dict[str,float] = field(default_factory=dict)
    roi: int = 0

@dataclass
class Mosaic:
    rows:int; cols:int
    hexes: List[Hex]
    index_by_qr: Dict[Tuple[int,int], int]
    layers: Dict[str, np.ndarray]  # np arrays of shape (N,)

def build_hex_mosaic(rows=ROWS, cols=COLS, seed=7) -> Mosaic:
    # układ „offset odd-r”
    def axial_to_xy(q:int,r:int, size:float=1.0)->Tuple[float,float]:
        x = size * (math.sqrt(3)*q + math.sqrt(3)/2 * r)
        y = size * (3/2 * r)
        return (x,y)
    rng = np.random.default_rng(seed)
    hexes: List[Hex] = []
    idx: Dict[Tuple[int,int],int] = {}
    k = 0
    for r in range(rows):
        for q in range(cols):
            cx,cy = axial_to_xy(q, r, size=1.0)
            edge = np.clip(0.45 + 0.45*np.sin(0.3*cx+0.2*cy) + 0.08*rng.standard_normal(), 0, 1)
            var  = np.clip(0.4 + 0.6*np.cos(0.2*cx-0.15*cy), 0, 1)
            hexes.append(Hex(q,r,(cx,cy), feats={"edge":float(edge),"var":float(var)}, roi=0))
            idx[(q,r)] = k; k+=1
    N = len(hexes)
    layers = {
        "edge": np.array([h.feats["edge"] for h in hexes], dtype=float),
        "var":  np.array([h.feats["var"]  for h in hexes], dtype=float),
        "ssim": np.ones(N, dtype=float)
    }
    # domyślne ROI: prostokątny „blok” w centrum siatki
    cx_all = np.array([h.center[0] for h in hexes]); cy_all=np.array([h.center[1] for h in hexes])
    x0,x1 = np.quantile(cx_all,[0.35,0.65])
    y0,y1 = np.quantile(cy_all,[0.35,0.65])
    roi = ((cx_all>=x0)&(cx_all<=x1)&(cy_all>=y0)&(cy_all<=y1)).astype(int)
    for i,h in enumerate(hexes): h.roi = int(roi[i])
    layers["roi"] = roi.astype(float)
    return Mosaic(rows, cols, hexes, idx, layers)

def region_ids(region:str, M:Mosaic, edge_thr=0.55)->Set[int]:
    N = len(M.hexes)
    if region=="ALL": return set(range(N))
    if region=="roi": return {i for i,h in enumerate(M.hexes) if h.roi>0}
    if region=="edges": return {i for i,h in enumerate(M.hexes) if h.feats["edge"]>edge_thr}
    if region=="~edges": return {i for i,h in enumerate(M.hexes) if h.feats["edge"]<=edge_thr}
    if region.startswith("ssim<"):
        t=float(region.split("<",1)[1]);
        return {i for i,v in enumerate(M.layers["ssim"]) if v<t}
    return set()

def region_feats(M:Mosaic, ids:Set[int])->np.ndarray:
    if not ids: return np.zeros(6,dtype=float)
    ed = np.array([M.hexes[i].feats["edge"] for i in ids])
    fL   = float(1.0 - ed.mean())
    fS   = float(0.5 + 0.5 * ed.std())
    fSel = float((ed>0.6).mean())
    fSt  = float(1.0 - ed.std())
    fC   = float(min(1.0, 0.35 + 0.6 * ed.mean()))
    fH   = float(0.4 + 0.5 * ed.std())
    return np.array([fL,fS,fSel,fSt,fC,fH], dtype=float)

# ====== 3) SPRZĘŻENIE Φ/Ψ ======
def project_phi(nodes: Dict[int,AstNodeInfo], M:Mosaic, lam:float=0.0, gamma:float=0.7,
                focus: Optional[int]=None, eta:float=0.0)->np.ndarray:
    """
    Raster z AST: rozrzucamy energię/entropię na heksy (przez pozycje/lambda).
    lam  – skala: 0=detal (poziomy), 1=centroidy grup (tu heurystycznie: spłaszczamy do warstwy depth)
    gamma – udział komponenty AST względem bazowego 'edge'
    eta – boost dla wybranego węzła focus
    Zwraca: v ∈ R^N,  v = (1-γ)*edge + γ*heat(AST)
    """
    N = len(M.hexes)
    heat = np.zeros(N, dtype=float)

    # bounding 2D sceny AST (XY)
    xs = np.array([n.pos3d[0] for n in nodes.values()]); ys=np.array([n.pos3d[1] for n in nodes.values()])
    x0,x1 = float(xs.min()),float(xs.max()); y0,y1=float(ys.min()),float(ys.max())
    if abs(x1-x0)<1e-9: x1=x0+1.0
    if abs(y1-y0)<1e-9: y1=y0+1.0
    # bounding mozaiki (XY)
    hx = np.array([h.center[0] for h in M.hexes]); hy=np.array([h.center[1] for h in M.hexes])
    a0,a1 = float(hx.min()),float(hx.max()); b0,b1=float(hy.min()),float(hy.max())

    def to_hex_index(x:float,y:float)->int:
        # najbliższe centrum
        d2 = (hx - x)**2 + (hy - y)**2
        return int(np.argmin(d2))

    # „grupowanie” po lam: im większa λ, tym mocniej przyciągamy do średnich poziomów depth
    by_depth: Dict[int,List[int]] = {}
    for i,n in nodes.items(): by_depth.setdefault(n.depth,[]).append(i)
    centroid_by_depth: Dict[int,Tuple[float,float]] = {}
    for d, ids in by_depth.items():
        cx = float(np.mean([nodes[i].pos3d[0] for i in ids]))
        cy = float(np.mean([nodes[i].pos3d[1] for i in ids]))
        centroid_by_depth[d]=(cx,cy)

    for i,n in nodes.items():
        x,y,_ = n.pos3d
        cx,cy = centroid_by_depth[n.depth]
        X = (1.0-lam)*x + lam*cx
        Y = (1.0-lam)*y + lam*cy
        # przeskalowanie do przestrzeni heksów
        Xh = a0 + (X - x0) / (x1 - x0) * (a1 - a0)
        Yh = b0 + (Y - y0) / (y1 - y0) * (b1 - b0)
        j = to_hex_index(Xh, Yh)
        L,S,Sel,Stab,Cau,H = n.meta
        w = 0.6*np.linalg.norm(n.meta) + 0.4*H
        if focus is not None and i==focus: w *= (1.0 + max(0.0, eta))
        heat[j] += float(w * max(1, n.count))  # powtarzalność wzmacnia

    # normalizacja
    if heat.max()>1e-12: heat = (heat-heat.min())/(heat.max()-heat.min())
    base = M.layers["edge"].copy()
    return (1.0-gamma)*base + gamma*heat

def fuse_meta(node_meta: np.ndarray, reg_meta: np.ndarray, lam: float, beta: float) -> np.ndarray:
    return (1.0 - lam) * node_meta + lam * (beta * reg_meta)

def apply_psi_update(nodes: Dict[int,AstNodeInfo], M:Mosaic, delta:float=0.2):
    """miękka aktualizacja meta-wektorów z mozaiki"""
    ids_all = set(range(len(M.hexes)))
    feats_all = region_feats(M, ids_all)
    for n in nodes.values():
        # heurystyka: If/Compare → region=ALL; Expr → edges/~edges wg parzystości id; Assign → roi
        if n.kind=="Assign": ids = region_ids("roi", M)
        elif n.kind=="Expr": ids = region_ids("edges" if n.id%2==0 else "~edges", M)
        else: ids = ids_all
        reg = feats_all if not ids else region_feats(M, ids)
        m = np.array(n.meta, dtype=float)
        nodes[n.id].meta = tuple((1.0-delta)*m + delta*reg)

# ====== 4) METRYKI / FUNKCJA CELU ======
def d_phi_cost(M:Mosaic, ast_comp:np.ndarray, denoise_ids:Set[int], blur_ids:Set[int]) -> float:
    # kara: wysokie edge w denoise i niskie edge w blur
    edge = M.layers["edge"]
    cost = 0.1 * float(edge[list(denoise_ids)].sum()) + 0.1 * float((1.0 - edge[list(blur_ids)]).sum())
    # zgodność ast_comp z base edge (im bardziej zbieżne – tym lepiej)
    align = float(np.mean((ast_comp - edge)**2))
    return float(cost + 0.2*align)

def invariants(M:Mosaic, denoise_ids:Set[int], blur_ids:Set[int], thr=0.55)->Dict[str,str]:
    I1="OK"
    I3="OK" if all(M.hexes[i].feats["edge"]<=thr for i in denoise_ids) else "WARN"
    roi = region_ids("roi", M); edges = region_ids("edges", M, thr)
    leak = len(roi & edges)/max(1,len(roi))
    I2=f"boundary_overlap≈{leak:.2f}"; I4="check after Φ"
    return {"I1":I1,"I2":I2,"I3":I3,"I4":I4}

# ====== 5) PROTOKÓŁ KONTEXTU ======
def export_protocol(nodes:Dict[int,AstNodeInfo], M:Mosaic, ast_comp:np.ndarray, J:float) -> Dict:
    ast_json = [{
        "id": n.id, "kind": n.kind, "label": n.label, "parent": n.parent,
        "depth": n.depth, "children": n.children, "hash": n.hash, "count": n.count,
        "meta": list(map(float,n.meta)), "pos3d": list(map(float,n.pos3d))
    } for n in nodes.values()]
    proto = {
        "ast": {"nodes": ast_json},
        "mosaic": {
            "rows": M.rows, "cols": M.cols,
            "layers": {k: v.tolist() for k,v in M.layers.items()},
            "hex_centers": [h.center for h in M.hexes]
        },
        "phi": {"ast_component": ast_comp.tolist()},
        "metrics": {"J": float(J)},
        "version": "v5-protocol-0.1"
    }
    return proto

# ====== 6) GŁÓWNY PRZEPŁYW ======
EXAMPLE = """\
def pipeline(img):
    R  = (120, 80, 200, 160)
    E  = edges(img, method='Sobel', thresh=0.55)
    D  = denoise_nlm(img, strength=0.35)
    B  = gaussian_blur(img, sigma=1.8)
    Z  = blend(img, B, 0.5)
    M  = metric_ssim(img, Z)
    return blend(D, B, 0.5)
"""

def main(show=PLOT):
    # AST
    nodes = build_ast_info(EXAMPLE)
    # MOZAIKA
    M = build_hex_mosaic()
    # Φ: raster AST→mozaika
    ast_comp = project_phi(nodes, M, lam=0.25, gamma=0.7, focus=None, eta=0.0)
    # Ψ: delikatny feedback i ponowna projekcja
    apply_psi_update(nodes, M, delta=0.15)
    ast_comp = project_phi(nodes, M, lam=0.35, gamma=0.7, focus=None, eta=0.0)

    # METRYKI i J
    denoise_ids = region_ids("~edges", M)
    blur_ids    = region_ids("edges",  M)
    J = d_phi_cost(M, ast_comp, denoise_ids, blur_ids)

    # PROTOKÓŁ
    proto = export_protocol(nodes, M, ast_comp, J)
    print(json.dumps(proto["metrics"], indent=2))
    print("protocol-size", len(json.dumps(proto)))

    if show:
        import matplotlib.pyplot as plt
        from matplotlib.patches import RegularPolygon
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
        # AST „kompas” 2D (rzut XY + wysokość=meta_energy)
        for n in nodes.values():
            x,y,z = n.pos3d
            h = 0.8 + 1.6*np.linalg.norm(n.meta)/np.sqrt(6)
            ax1.plot([x,x],[y,y+h], color="#334155", lw=2.0)
            ax1.scatter([x],[y+h], s=18, c=[(0.2,0.2,0.7,0.9)])
            if n.depth<=3: ax1.text(x, y+h+0.2, n.kind, fontsize=8, ha="center")
        ax1.set_title("AST – rzut kompasu (h~‖meta‖)")
        ax1.set_aspect("equal", "box")

        # heks-mozaika: kolor = base edge, obwódka = wkład AST
        centers = np.array([h.center for h in M.hexes])
        base = M.layers["edge"]; contrib = ast_comp
        bmin,bmax = base.min(), base.max()
        for i,(cx,cy) in enumerate(centers):
            col = ( (base[i]-bmin)/(bmax-bmin+1e-9), 0.2, 1.0-(base[i]-bmin)/(bmax-bmin+1e-9), 0.95 )
            hex = RegularPolygon((cx,cy), numVertices=6, radius=0.95, orientation=np.radians(30),
                                 facecolor=col, edgecolor=(0,0,0, 0.25+0.7*contrib[i]), lw=1.0)
            ax2.add_patch(hex)
        ax2.autoscale_view()
        ax2.set_aspect("equal","box")
        ax2.set_title("Mozaika 12×12 (edge + wkład AST)")
        plt.tight_layout(); plt.show()

    return proto

if __name__ == "__main__":
    main()
