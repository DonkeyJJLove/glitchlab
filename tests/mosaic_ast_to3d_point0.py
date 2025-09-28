# -*- coding: utf-8 -*-
"""
GlitchLab – Punkt 0 w 3D: AST ⇄ Mozaika (hex 12×12 – stykające się bokami) ⇄ Lattice BCC (14-NN ~ truncated octahedra)

• AST 3D: metaglif (wysokość = ||meta||, kolor = entropia).
• Mozaika HEX 12×12: axial (q,r), pointy-top, spacing gwarantujący STYK BOKÓW.
• Lattice 3D (Voronoi BCC): centra komórek + krawędzie 14-sąsiedztwa (aproksymacja ściętych ośmiościanów).
• Φ (AST→Mozaika), Ψ (Mozaika→AST), „punkt 0” (centroid regionu).
• Miary: CR_AST (Merkle), CR_TO (snap do komórek BCC), Align3D, J_phi.

Wymagania: numpy, matplotlib
Uruchom:  python demos/mosaic_ast_to3d_point0.py
"""

from __future__ import annotations
import ast, math, hashlib, json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from mpl_toolkits.mplot3d import Axes3D  # noqa


# ─────────────────────────────────────────────
# 0) AST: parsowanie, metaglif, Merkle
# ─────────────────────────────────────────────

@dataclass
class AstNode:
    id: int
    kind: str
    label: str
    depth: int
    parent: Optional[int]
    children: List[int] = field(default_factory=list)
    meta: Tuple[float,float,float,float,float,float] = (0,0,0,0,0,0)
    pos3d: Tuple[float,float,float] = (0.0,0.0,0.0)
    h: str = ""

    @property
    def energy(self) -> float:  # ||meta||
        return float(np.linalg.norm(self.meta))

    @property
    def entropy(self) -> float:  # H
        return float(self.meta[-1])

EXAMPLE = r"""
def pipeline(img):
    R  = (120, 80, 200, 160)
    E  = edges(img, method='Sobel', thresh=0.55)
    D  = denoise_nlm(img, strength=0.35)
    B  = gaussian_blur(img, sigma=1.8)
    Z  = blend(img, B, 0.5)
    M  = metric_ssim(img, Z)
    return blend(D, B, 0.5)
"""

def _label(n: ast.AST) -> str:
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)): return f"def {n.name}"
    if isinstance(n, ast.ClassDef): return f"class {n.name}"
    if isinstance(n, ast.Assign):
        t = n.targets[0] if n.targets else None
        return f"{getattr(t,'id','assign')} = …"
    if isinstance(n, ast.Name): return n.id
    return type(n).__name__

def build_ast(src: str) -> Dict[int, AstNode]:
    root = ast.parse(src)
    nodes: Dict[int, AstNode] = {}
    nid = 0
    def add(a: ast.AST, d: int, p: Optional[int]):
        nonlocal nid
        me = nid; nid += 1
        nodes[me] = AstNode(me, a.__class__.__name__, _label(a), d, p)
        if p is not None: nodes[p].children.append(me)
        for ch in ast.iter_child_nodes(a): add(ch, d+1, me)
    add(root, 0, None)

    # pozycjonowanie radialne (kompas); Z = depth*2
    by_d: Dict[int,List[int]] = {}
    for i,n in nodes.items(): by_d.setdefault(n.depth,[]).append(i)
    for d, ids in by_d.items():
        ids.sort()
        R = 6.0 + 2.0*d
        for j,i in enumerate(ids):
            a = 2*math.pi*j/max(1,len(ids))
            nodes[i].pos3d = (R*math.cos(a), R*math.sin(a), 2.0*d)

    # meta – deterministyczne heurystyki (stabilne)
    for n in nodes.values():
        L=S=Sel=Stab=Cau=H = 0.55
        if n.kind in ("If","Compare"): Sel, Cau, H = 0.85, 0.80, 0.60
        if n.kind in ("Expr","Call"):  L, S = 0.65, 0.58
        if n.kind in ("Return",):      Stab = 0.90
        if n.kind in ("Assign",):      Sel = 0.70
        nodes[n.id].meta = (L,S,Sel,Stab,Cau,H)

    # Merkle (hash poddrzew)
    def merkle(i:int)->str:
        ch = "".join(sorted(merkle(c) for c in nodes[i].children))
        s = f"{nodes[i].kind}|{nodes[i].label}|{ch}"
        h = hashlib.sha256(s.encode()).hexdigest()[:16]
        nodes[i].h = h; return h
    _ = merkle(0)
    return nodes


# ─────────────────────────────────────────────
# 1) HEX 12×12 – stykające (axial q,r; pointy-top)
# ─────────────────────────────────────────────

HEX_R = 1.0  # radius to VERTEX (pointy-top). Apothegm = √3/2 * R.
# DYSTANSE DLA STYKU BOKAMI (pointy-top):
# Δx = √3 * R ; Δy = 1.5 * R  (dokładnie te wartości → brak szczelin)
def axial_to_xy(q:int, r:int, R:float=HEX_R)->Tuple[float,float]:
    return ( (math.sqrt(3.0)*R)*(q + 0.5*(r&1)) , (1.5*R)*r )

@dataclass
class Hex:
    q:int; r:int
    center: Tuple[float,float]
    edge: float
    roi: int

@dataclass
class Mosaic:
    rows:int; cols:int
    hexes: List[Hex]

def build_hex_mosaic(rows=12, cols=12)->Mosaic:
    hexes: List[Hex] = []
    # ciągła „tekstura” edge – zależna tylko od (x,y), bez aliasingu
    for r in range(rows):
        for q in range(cols):
            x,y = axial_to_xy(q,r,HEX_R)
            edge = 0.50 + 0.45*math.tanh(0.17*x - 0.15*y)  # płynne pole
            hexes.append(Hex(q,r,(x,y), edge, 0))
    # ROI: centralny romb
    xs = np.array([h.center[0] for h in hexes]); ys = np.array([h.center[1] for h in hexes])
    x0,x1 = np.quantile(xs,[0.35,0.65]); y0,y1 = np.quantile(ys,[0.35,0.65])
    for h in hexes:
        h.roi = int(x0<=h.center[0]<=x1 and y0<=h.center[1]<=y1)
    return Mosaic(rows, cols, hexes)


# ─────────────────────────────────────────────
# 2) Lattice 3D: Voronoi BCC (14-NN ~ truncated octahedra)
# ─────────────────────────────────────────────

BCC_HEX = [(+1,+1,+1),(+1,+1,-1),(+1,-1,+1),(+1,-1,-1),
           (-1,+1,+1),(-1,+1,-1),(-1,-1,+1),(-1,-1,-1)]
BCC_SQR = [(+2,0,0),(-2,0,0),(0,+2,0),(0,-2,0),(0,0,+2),(0,0,-2)]

@dataclass
class Cell:
    xyz: Tuple[int,int,int]
    center: Tuple[float,float,float]

@dataclass
class BCC:
    cells: List[Cell]
    index: Dict[Tuple[int,int,int], int]
    neighbors: Dict[int, List[int]]

def build_bcc(nx=6, ny=6, nz=4, scale=1.0)->BCC:
    cells=[]; index={}
    def to_center(x,y,z): return (scale*x/2.0, scale*y/2.0, scale*z/2.0)
    k=0
    for z in range(-nz,nz+1):
        for y in range(-ny,ny+1):
            for x in range(-nx,nx+1):
                if (x+y+z)%2==0:
                    cells.append(Cell((x,y,z), to_center(x,y,z)))
                    index[(x,y,z)] = k; k+=1
    neigh: Dict[int,List[int]]={}
    for i,c in enumerate(cells):
        x,y,z = c.xyz
        ids=[]
        for dx,dy,dz in (BCC_HEX + BCC_SQR):
            j = index.get((x+dx,y+dy,z+dz))
            if j is not None: ids.append(j)
        neigh[i]=ids
    return BCC(cells,index,neigh)


# ─────────────────────────────────────────────
# 3) Φ / Ψ / „punkt 0” / metryki
# ─────────────────────────────────────────────

def region_ids_for_node(n: AstNode, M: Mosaic, thr=0.6) -> Set[int]:
    ids=set()
    for i,h in enumerate(M.hexes):
        if n.kind=="Assign":
            if h.roi: ids.add(i)
        elif n.kind in ("Expr","Call"):
            # parzyste → „edges”, nieparzyste → „~edges”
            if (n.id%2)==0 and h.edge>thr: ids.add(i)
            if (n.id%2)==1 and h.edge<=thr: ids.add(i)
        elif n.kind in ("Return","If"):
            ids.add(i)
    return ids

def centroid_hex(ids:Set[int], M:Mosaic)->Tuple[float,float,float]:
    if not ids:
        # środek sceny mozaiki
        xs = [h.center[0] for h in M.hexes]; ys = [h.center[1] for h in M.hexes]
        return (float(np.mean(xs)), float(np.mean(ys)), 0.0)
    xs=[M.hexes[i].center[0] for i in ids]
    ys=[M.hexes[i].center[1] for i in ids]
    zs=[0.10 + 2.0*M.hexes[i].edge for i in ids]
    return (float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs)))

def phi_ast_to_hex(nodes:Dict[int,AstNode], M:Mosaic, lam=0.35, gamma=0.7,
                   boost:Optional[int]=None, eta=0.0)->np.ndarray:
    # raster ciepła: węzły → najbliższe centra hex po przesunięciu w kierunku centroidu warstwy
    # 1) centroidy warstw depth
    by_d: Dict[int,List[int]] = {}
    for i,n in nodes.items(): by_d.setdefault(n.depth,[]).append(i)
    layer_centroid = {d: (float(np.mean([nodes[i].pos3d[0] for i in ids])),
                          float(np.mean([nodes[i].pos3d[1] for i in ids])))
                      for d,ids in by_d.items()}
    centers = np.array([h.center for h in M.hexes])
    heat = np.zeros(len(M.hexes), float)
    for i,n in nodes.items():
        cx,cy = layer_centroid[n.depth]
        X = (1.0-lam)*n.pos3d[0] + lam*cx
        Y = (1.0-lam)*n.pos3d[1] + lam*cy
        j = int(np.argmin((centers[:,0]-X)**2 + (centers[:,1]-Y)**2))
        w = 0.6*np.linalg.norm(n.meta) + 0.4*n.entropy
        if boost is not None and boost==i: w *= (1.0+max(0.0,eta))
        heat[j] += w
    # normalizacja + blend z polem edge (γ)
    if heat.max()>1e-12:
        heat = (heat-heat.min())/(heat.max()-heat.min())
    base = np.array([h.edge for h in M.hexes])
    return (1.0-gamma)*base + gamma*heat

def psi_hex_to_ast(nodes:Dict[int,AstNode], M:Mosaic, delta=0.2):
    # miękki feedback: ROI vs ~ROI → poprawki meta wektorów
    ids_roi = {i for i,h in enumerate(M.hexes) if h.roi}
    ids_bg  = set(range(len(M.hexes))) - ids_roi
    def feats(ids:Set[int])->np.ndarray:
        if not ids: return np.zeros(6,float)
        ed = np.array([M.hexes[i].edge for i in ids])
        fL   = 1.0-ed.mean()
        fS   = 0.5+0.5*ed.std()
        fSel = ( ed>0.6 ).mean()
        fSt  = 1.0-ed.std()
        fC   = min(1.0, 0.35+0.6*ed.mean())
        fH   = 0.45+0.5*ed.std()
        return np.array([fL,fS,fSel,fSt,fC,fH], float)
    m_roi, m_bg = feats(ids_roi), feats(ids_bg)
    for n in nodes.values():
        target = m_roi if (n.id%3==0) else m_bg
        m = np.array(n.meta, float)
        nodes[n.id].meta = tuple((1.0-delta)*m + delta*target)

def merkle_compression(nodes:Dict[int,AstNode])->float:
    from collections import Counter
    cnt = Counter(n.h for n in nodes.values())
    return len(nodes)/max(1,len(cnt))  # CR_AST

def snap_ast_to_bcc(nodes:Dict[int,AstNode], L:BCC)->Tuple[Dict[int,int], float, float]:
    centers = np.array([c.center for c in L.cells])
    occ: Dict[int,int] = {}
    assign: Dict[int,int] = {}
    for i,n in nodes.items():
        p = np.array([n.pos3d[0], n.pos3d[1], n.pos3d[2]])
        j = int(np.argmin(np.sum((centers-p)**2, axis=1)))
        assign[i]=j; occ[j]=occ.get(j,0)+1
    CR_TO = len(nodes)/max(1, len(occ))
    # Align3D – wariancja occupancy po 14-NN
    var_local=[]
    for j,c in enumerate(L.cells):
        vals = [occ.get(t,0) for t in L.neighbors[j]] + [occ.get(j,0)]
        if vals: var_local.append(np.var(vals))
    Align3D = float(np.mean(var_local)) if var_local else 0.0
    return assign, CR_TO, Align3D

def phi_cost(M:Mosaic, ast_comp:np.ndarray)->float:
    # kara: mocny wkład na krawędziach (ryzyko blur/denoise konfliktu) + zbieżność do pola bazowego
    edges = np.array([h.edge for h in M.hexes])
    ids_e = np.where(edges>0.6)[0]
    ids_n = np.where(edges<=0.6)[0]
    leak = float(ast_comp[ids_e].mean())              # zbyt duży wkład w pasie krawędzi
    miss = float((1.0-ast_comp[ids_n]).mean())        # zbyt mały wkład gdzie „~edges”
    align = float(np.mean((ast_comp-edges)**2))       # niespójność z bazą
    return 0.4*leak + 0.3*miss + 0.3*align


# ─────────────────────────────────────────────
# 4) Rysowanie (3 panele)
# ─────────────────────────────────────────────

def draw_ast(ax, nodes:Dict[int,AstNode]):
    # krawędzie
    for n in nodes.values():
        x0,y0,z0 = n.pos3d
        for cid in n.children:
            x1,y1,z1 = nodes[cid].pos3d
            ax.plot([x0,x1],[y0,y1],[z0,z1], color="#7b8fa1", lw=1.0)
    # metaglif
    cmap = plt.get_cmap("plasma")
    for n in nodes.values():
        x,y,z = n.pos3d
        h = 1.1 + 2.1*(n.energy/np.sqrt(6))
        c = cmap(n.entropy)
        ax.plot([x,x],[y,y],[z,z+h], color=c, lw=3.0)
        ax.scatter([x],[y],[z+h], s=24, c=[c], edgecolors="black", depthshade=True)
        if n.depth<=3: ax.text(x,y,z+h+0.35, n.kind, fontsize=8, ha="center")
    ax.set_title("AST 3D – metaglif")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

def draw_hex(ax, M: Mosaic, ast_comp: Optional[np.ndarray] = None):
    # heksy pointy-top, ORIENTATION=0 rad, PROMIEŃ = HEX_R → styk BOKÓW
    vals = np.array([h.edge for h in M.hexes])
    vmin, vmax = float(vals.min()), float(vals.max())

    def norm(v):
        if vmax - vmin < 1e-12:
            return 0.0
        return (v - vmin) / (vmax - vmin)

    # rysowanie heksów
    for i, h in enumerate(M.hexes):
        cx, cy = h.center
        base = norm(h.edge)
        face = (base, 0.2, 1.0 - base, 0.95)
        ealpha = 0.28 if ast_comp is None else 0.25 + 0.65 * float(ast_comp[i])

        # ⬇⬇⬇ KLUCZOWA ZMIANA: wszystkie parametry nazwane
        poly = RegularPolygon(
            xy=(cx, cy),
            numVertices=6,
            radius=HEX_R,
            orientation=0.0,
            facecolor=face,
            edgecolor=(0, 0, 0, ealpha),
            linewidth=1.0,
            antialiased=True,
            snap=True,
        )
        ax.add_patch(poly)

        if h.roi:
            ring = RegularPolygon(
                xy=(cx, cy),
                numVertices=6,
                radius=HEX_R * 0.86,
                orientation=0.0,
                facecolor=(0, 0, 0, 0),
                edgecolor=(1, 1, 1, 0.55),
                linewidth=0.6,
                antialiased=True,
                snap=True,
            )
            ax.add_patch(ring)

    # zakresy bez „oddechu” osi (żeby nie wprowadzać wizualnych przerw)
    xs = np.array([h.center[0] for h in M.hexes], dtype=float)
    ys = np.array([h.center[1] for h in M.hexes], dtype=float)
    ax.set_xlim(xs.min() - HEX_R * 1.05, xs.max() + HEX_R * 1.05)
    ax.set_ylim(ys.min() - HEX_R * 1.05, ys.max() + HEX_R * 1.05)

    # geometrycznie „sztywno” – brak odkształceń
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.0)
    ax.autoscale(enable=False)

    ax.set_title("Mozaika HEX 12×12 – stykające się bokami")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
def draw_bcc(ax, L:BCC, assign:Optional[Dict[int,int]]=None):
    # krawędzie 14-NN
    for j,c in enumerate(L.cells):
        x0,y0,z0 = c.center
        for t in L.neighbors[j]:
            x1,y1,z1 = L.cells[t].center
            ax.plot([x0,x1],[y0,y1],[z0,z1], color="#9aa5b1", lw=0.45, alpha=0.7)
    # centra; jeśli mamy przypisania węzłów AST, dociąż kolor
    occ = {}
    if assign:
        for nid,cid in assign.items(): occ[cid]=occ.get(cid,0)+1
    for j,c in enumerate(L.cells):
        k = occ.get(j,0)
        ax.scatter([c.center[0]],[c.center[1]],[c.center[2]],
                   s=10+6*k, c=[(0.15,0.5,1.0,0.85)], edgecolors="black", depthshade=True)
    ax.set_title("Lattice 3D – Voronoi BCC (14-NN)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=22, azim=-48)


# ─────────────────────────────────────────────
# 5) MAIN (spięcie + protokół)
# ─────────────────────────────────────────────

def main():
    # AST
    nodes = build_ast(EXAMPLE)

    # Lattice 3D
    L = build_bcc(nx=6,ny=6,nz=4)
    assign, CR_TO, Align3D = snap_ast_to_bcc(nodes, L)

    # HEX mozaika – gwarancja styku boków (axial, pointy-top)
    M = build_hex_mosaic(12,12)

    # Φ i Ψ
    ast_comp = phi_ast_to_hex(nodes, M, lam=0.35, gamma=0.7)
    psi_hex_to_ast(nodes, M, delta=0.18)
    ast_comp = phi_ast_to_hex(nodes, M, lam=0.42, gamma=0.7)

    # Miary
    CR_AST = merkle_compression(nodes)
    J_phi  = phi_cost(M, ast_comp)
    scores = {
        "CR_AST": float(CR_AST),
        "CR_TO":  float(CR_TO),
        "Align3D": float(Align3D),
        "J_phi": float(J_phi),
        "J_total": float(0.5*J_phi + 0.25/CR_AST + 0.25/CR_TO)
    }
    print(json.dumps(scores, indent=2))

    # Protokół kontekstu (gotowe do cache)
    proto = {
        "version": "v5-to3d-hex",
        "ast": {"nodes":[{
            "id":n.id,"kind":n.kind,"label":n.label,"depth":n.depth,
            "parent":n.parent,"children":n.children,"hash":n.h,
            "meta":list(map(float,n.meta)),"pos3d":list(map(float,n.pos3d))
        } for n in nodes.values()]},
        "mosaic": {
            "hex_centers":[h.center for h in M.hexes],
            "edge":[float(h.edge) for h in M.hexes],
            "roi":[int(h.roi) for h in M.hexes],
            "axial":"pointy-top", "hex_R": HEX_R,
            "spacing":{"dx": math.sqrt(3.0)*HEX_R, "dy": 1.5*HEX_R}
        },
        "bcc": {
            "cells":[{"xyz":c.xyz, "center":c.center} for c in L.cells],
            "neighbors": L.neighbors,
            "assign": assign
        },
        "phi":{"ast_component": ast_comp.tolist()},
        "metrics": scores
    }

    # Rysunek (3 panele)
    fig = plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(1,3,1, projection="3d")
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3, projection="3d")

    draw_ast(ax1, nodes)
    draw_hex(ax2, M, ast_comp=ast_comp)
    draw_bcc(ax3, L, assign=assign)

    fig.suptitle("GlitchLab – Punkt 0 (3D): AST ⇄ HEX (stykające) ⇄ BCC/TO lattice", y=0.98)
    plt.tight_layout(); plt.show()
    return proto

if __name__=="__main__":
    main()
