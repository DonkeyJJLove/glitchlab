# -*- coding: utf-8 -*-
"""
GlitchLab — AST × Mozaika (heks) × TO-lattice (truncated octahedron, Voronoi BCC)
- Heksy stykają się bokami (axial q,r z poprawną geometrią).
- Wspólna rama 3D: komórki ściętego ośmiościanu (TO) aproksymowane kratą BCC (14-NN).
- Kompresja: Merkle-AST + „snap” węzłów AST do komórek TO (CR_AST, CR_TO).
- Sprzężenia Φ/Ψ i funkcja oceny J.

Uruchom:
  python demos/ast_mosaic_to_lattice_experiment.py
Wymaga:
  numpy, matplotlib
"""
from __future__ import annotations
import ast, math, json, hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

# ============ 0) PARAMETRY ============
HEX_R = 1.0        # promień heksa (inscribed) — ważny dla styków
ROWS, COLS = 12,12 # mozaika 12×12
SHOW = True

# ============ 1) AST: parsowanie + Merkle + „kompas 3D” ============
@dataclass
class AstNodeInfo:
    id: int
    kind: str
    label: str
    depth: int
    parent: Optional[int]
    children: List[int] = field(default_factory=list)
    h: str = ""                                  # Merkle hash poddrzewa
    meta: Tuple[float,float,float,float,float,float]=(0,0,0,0,0,0)
    pos3d: Tuple[float,float,float]=(0.0,0.0,0.0)

def _label(n: ast.AST)->str:
    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)): return f"def {n.name}"
    if isinstance(n,ast.ClassDef): return f"class {n.name}"
    if isinstance(n,ast.Assign):
        t = n.targets[0] if n.targets else None
        return f"{getattr(t,'id','assign')} = …"
    if isinstance(n,ast.Name): return n.id
    return type(n).__name__

def build_ast(src:str, seed:int=123)->Dict[int,AstNodeInfo]:
    T = ast.parse(src)
    nodes: Dict[int,AstNodeInfo] = {}
    nid=0
    def add(a:ast.AST, d:int, p:Optional[int]):
        nonlocal nid
        me=nid; nid+=1
        nodes[me]=AstNodeInfo(me, a.__class__.__name__, _label(a), d, p)
        if p is not None: nodes[p].children.append(me)
        for ch in ast.iter_child_nodes(a): add(ch, d+1, me)
    add(T,0,None)

    # pozycje (kompas 3D): radial by depth
    by_d: Dict[int,List[int]]={}
    for i,n in nodes.items(): by_d.setdefault(n.depth,[]).append(i)
    for d,ids in by_d.items():
        ids.sort()
        R = 6.0 + 2.0*d
        for j,i in enumerate(ids):
            ang = 2*math.pi*j/max(1,len(ids))
            nodes[i].pos3d = (R*math.cos(ang), R*math.sin(ang), 2.0*d)

    # meta (heurystyki stabilne, nie losowe)
    def meta_for(n:AstNodeInfo)->Tuple[float,...]:
        L=S=Sel=Stab=Cau=H=0.55
        if n.kind in ("If","Compare"): Sel,Cau,H = 0.85,0.80,0.60
        if n.kind in ("Return",): Stab = 0.90
        if n.kind in ("Assign",): Sel,Stab = 0.70,0.72
        if n.kind in ("Call","Expr"): L,S = 0.65,0.60
        return (L,S,Sel,Stab,Cau,H)
    for n in nodes.values(): nodes[n.id].meta = meta_for(n)

    # Merkle hash
    def merkle(i:int)->str:
        ch = "".join(sorted(merkle(c) for c in nodes[i].children))
        s = f"{nodes[i].kind}|{nodes[i].label}|{ch}"
        h = hashlib.sha256(s.encode()).hexdigest()[:16]
        nodes[i].h = h
        return h
    _ = merkle(0)
    return nodes

# ============ 2) TO-lattice (Voronoi BCC) i heks-mozaika ============
# BCC w integerach: punkt „even” (x+y+z parzyste) i „odd” (przesunięte o (0.5,0.5,0.5)).
# Najbliżsi sąsiedzi: 8 wierzchołków sześcianu (hexy), drugi pierścień: 6 osiowych (kwadraty) → łącznie 14.
BCC_NEIGH_HEX = [(+1,+1,+1),(+1,+1,-1),(+1,-1,+1),(+1,-1,-1),
                 (-1,+1,+1),(-1,+1,-1),(-1,-1,+1),(-1,-1,-1)]
BCC_NEIGH_SQR = [(+2,0,0),(-2,0,0),(0,+2,0),(0,-2,0),(0,0,+2),(0,0,-2)]

@dataclass
class TOCell:
    xyz: Tuple[int,int,int]      # integer coords BCC (skalowane)
    center: Tuple[float,float,float]
    feats: Dict[str,float]

@dataclass
class TOLattice:
    cells: List[TOCell]
    index: Dict[Tuple[int,int,int], int]
    neighbors: Dict[int, List[int]]  # 14-lista sąsiadów

def build_to_lattice(nx=8,ny=8,nz=4, scale=1.0)->TOLattice:
    cells=[]; index={}
    def to_center(x,y,z):
        return (scale*x/2.0, scale*y/2.0, scale*z/2.0)  # /2 → bo są kroki ±2 i ±1
    k=0
    for z in range(-nz,nz+1):
        for y in range(-ny,ny+1):
            for x in range(-nx,nx+1):
                if (x+y+z)%2==0:   # BCC-even warstwa
                    cx,cy,cz = to_center(x,y,z)
                    # cechy emulujemy prostym polem
                    edge = 0.5 + 0.4*math.tanh(0.1*cx - 0.08*cy + 0.05*cz)
                    var  = 0.4 + 0.6*math.sin(0.07*cx + 0.11*cy - 0.06*cz)**2
                    cells.append(TOCell((x,y,z),(cx,cy,cz),{"edge":edge,"var":var}))
                    index[(x,y,z)] = k; k+=1
    # sąsiedzi (14)
    neigh: Dict[int,List[int]]={}
    for i,c in enumerate(cells):
        x,y,z = c.xyz
        ids=[]
        for dx,dy,dz in BCC_NEIGH_HEX + BCC_NEIGH_SQR:
            key=(x+dx,y+dy,z+dz)
            j=index.get(key)
            if j is not None: ids.append(j)
        neigh[i]=ids
    return TOLattice(cells,index,neigh)

# heks-mozaika 12×12 — *prawdziwe* stykające się heksy (axial q,r)
@dataclass
class Hex:
    q:int; r:int
    center: Tuple[float,float]
    feats: Dict[str,float]; roi:int

@dataclass
class Mosaic:
    hexes: List[Hex]
    edge: np.ndarray  # (N,)
    roi:  np.ndarray  # (N,)

def axial_to_xy(q:int, r:int, R:float=HEX_R)->Tuple[float,float]:
    step_x = math.sqrt(3.0)*R
    step_y = 1.5*R
    x = step_x * (q + 0.5*(r&1))
    y = step_y * r
    return (x,y)

def build_hex_mosaic(rows=ROWS, cols=COLS)->Mosaic:
    hexes=[]; edge=[]; roi=[]
    # generujemy z pola z TO-lattice: projekcja z=const (użyję syntetycznej funkcji)
    for r in range(rows):
        for q in range(cols):
            x,y = axial_to_xy(q,r,HEX_R)
            # tekstura „edge”: ciągła, bez szpar (tylko od x,y)
            e = 0.5 + 0.45*math.tanh(0.18*x - 0.16*y)
            hexes.append(Hex(q,r,(x,y),{"edge":e},0))
            edge.append(e); roi.append(0)
    # ROI: centralny romb ~30% środka
    xs=np.array([h.center[0] for h in hexes]); ys=np.array([h.center[1] for h in hexes])
    x0,x1 = np.quantile(xs,[0.35,0.65]); y0,y1 = np.quantile(ys,[0.35,0.65])
    Rmask = ((xs>=x0)&(xs<=x1)&(ys>=y0)&(ys<=y1)).astype(int)
    for i,h in enumerate(hexes): h.roi=int(Rmask[i])
    return Mosaic(hexes, np.array(edge,float), Rmask.astype(float))

# ============ 3) Projekcja Φ i feedback Ψ ============
def phi_project_ast_to_hex(nodes:Dict[int,AstNodeInfo], M:Mosaic, lam=0.3, gamma=0.7,
                           boost_id:Optional[int]=None, eta=0.0)->np.ndarray:
    # bounding AST XY
    xs=np.array([n.pos3d[0] for n in nodes.values()]); ys=np.array([n.pos3d[1] for n in nodes.values()])
    x0,x1=float(xs.min()),float(xs.max()); y0,y1=float(ys.min()),float(ys.max())
    if x1-x0<1e-9: x1=x0+1;
    if y1-y0<1e-9: y1=y0+1
    # centroidy warstw (poziomy depth)
    by_d:Dict[int,List[int]]={};
    for i,n in nodes.items(): by_d.setdefault(n.depth,[]).append(i)
    layer_centroids={d: (float(np.mean([nodes[i].pos3d[0] for i in ids])),
                         float(np.mean([nodes[i].pos3d[1] for i in ids])))
                     for d,ids in by_d.items()}
    # raster
    H=len(M.hexes); heat=np.zeros(H,float)
    centers=np.array([h.center for h in M.hexes])
    for i,n in nodes.items():
        # przesuwamy w stronę centroidu warstwy (lam)
        cx,cy = layer_centroids[n.depth]
        X=(1.0-lam)*n.pos3d[0] + lam*cx
        Y=(1.0-lam)*n.pos3d[1] + lam*cy
        # mapowanie do heksów: najbliższe centrum
        # skalowanie współrzędnych AST → [min,max] heksów (tu wystarcza nearest)
        # (bo heksy są równomiernie rozłożone)
        j = int(np.argmin((centers[:,0]-X)**2 + (centers[:,1]-Y)**2))
        L,S,Sel,Stab,Cau,Hm = n.meta
        w = 0.6*np.linalg.norm(n.meta) + 0.4*Hm
        if boost_id is not None and i==boost_id: w *= (1.0+max(0.0,eta))
        heat[j]+=w
    if heat.max()>1e-12: heat=(heat-heat.min())/(heat.max()-heat.min())
    base=M.edge
    return (1.0-gamma)*base + gamma*heat

def psi_feedback(nodes:Dict[int,AstNodeInfo], M:Mosaic, delta=0.2):
    # miękka aktualizacja meta z cech regionów (roi vs ~roi) — prosty przykład
    ids_roi = {i for i,h in enumerate(M.hexes) if h.roi>0}
    ids_nroi = set(range(len(M.hexes))) - ids_roi
    def feats(ids:Set[int])->np.ndarray:
        if not ids: return np.zeros(6,float)
        ed=np.array([M.hexes[i].feats["edge"] for i in ids])
        fL=1.0-ed.mean(); fS=0.5+0.5*ed.std(); fSel=(ed>0.6).mean()
        fSt=1.0-ed.std(); fC=min(1.0, 0.35+0.6*ed.mean()); fH=0.45+0.5*ed.std()
        return np.array([fL,fS,fSel,fSt,fC,fH],float)
    m_roi, m_nroi = feats(ids_roi), feats(ids_nroi)
    for n in nodes.values():
        target = m_roi if (n.id%3==0) else m_nroi
        m=np.array(n.meta,float)
        nodes[n.id].meta = tuple((1.0-delta)*m + delta*target)

# ============ 4) Kompresja i metryki ============
def compression_merkle(nodes:Dict[int,AstNodeInfo])->float:
    from collections import Counter
    cnt = Counter(n.h for n in nodes.values())
    return len(nodes)/max(1,len(cnt))  # CR_AST

def to_snap_and_compress(nodes:Dict[int,AstNodeInfo], L:TOLattice)->Tuple[Dict[int,int], float, float]:
    # snap: rzut pos3d na środek najbliższej komórki; policz CR_TO i Align3D
    centers = np.array([c.center for c in L.cells])
    occ: Dict[int,int]={}  # cell_id -> licznik
    assign: Dict[int,int]={}  # node_id -> cell_id
    for i,n in nodes.items():
        p=np.array([n.pos3d[0],n.pos3d[1],n.pos3d[2]])
        j=int(np.argmin(np.sum((centers-p)**2,axis=1)))
        assign[i]=j; occ[j]=occ.get(j,0)+1
    CR_TO = len(nodes)/max(1,len(occ))
    # Align3D: dywergencja jednolitości po 14-NN: rozrzut occupancy po sąsiedztwach
    var_acc=[]
    for j,c in enumerate(L.cells):
        neigh=L.neighbors[j]
        vals=[occ.get(t,0) for t in neigh]+[occ.get(j,0)]
        if len(vals)>0: var_acc.append(np.var(vals))
    Align3D = float(np.mean(var_acc))  # im mniejsze tym bardziej jednorodne
    return assign, CR_TO, Align3D

def phi_cost(edge_base:np.ndarray, ast_comp:np.ndarray, M:Mosaic)->float:
    # kara za „denoise na krawędziach” i słabą zgodność z bazą
    ids_edges = {i for i,h in enumerate(M.hexes) if h.feats["edge"]>0.6}
    ids_ne = set(range(len(M.hexes))) - ids_edges
    leak = float(np.mean(ast_comp[list(ids_edges)]))     # duży wkład na krawędziach = ryzyko blur/denoise-konfliktu
    miss = float(np.mean(1.0-ast_comp[list(ids_ne)]))    # niski wkład gdzie powinien być
    align = float(np.mean((ast_comp-edge_base)**2))      # zbieżność z bazą
    return 0.4*leak + 0.3*miss + 0.3*align

# ============ 5) Protokół kontekstu ============
def export_protocol(nodes, M, L, assign, ast_comp, scores)->Dict:
    return {
        "version":"v5-protocol-TO-0.1",
        "ast":{"nodes":[{
            "id":n.id,"kind":n.kind,"label":n.label,"depth":n.depth,"parent":n.parent,
            "children":n.children,"hash":n.h,"meta":list(map(float,n.meta)),"pos3d":list(map(float,n.pos3d))
        } for n in nodes.values()]},
        "mosaic":{
            "hex_centers":[h.center for h in M.hexes],
            "edge":M.edge.tolist(),"roi":M.roi.tolist()
        },
        "to_lattice":{
            "cells":[{"center":c.center,"xyz":c.xyz} for c in L.cells],
            "assign":assign
        },
        "phi":{"ast_component":ast_comp.tolist()},
        "metrics":scores
    }

# ============ 6) DEMO MAIN ============
EXAMPLE = """\
def pipeline(img):
    R  = (120,80,200,160)
    E  = edges(img, method='Sobel', thresh=0.55)
    D  = denoise_nlm(img, strength=0.35)
    B  = gaussian_blur(img, sigma=1.8)
    Z  = blend(img, B, 0.5)
    M  = metric_ssim(img, Z)
    return blend(D, B, 0.5)
"""

def main(show=SHOW):
    # AST
    nodes = build_ast(EXAMPLE)
    CR_AST = compression_merkle(nodes)

    # TO-lattice (3D, wspólna rama)
    L = build_to_lattice(nx=6,ny=6,nz=4, scale=1.0)
    assign, CR_TO, Align3D = to_snap_and_compress(nodes, L)

    # Heks-mozaika (12×12) — stykające się heksy
    M = build_hex_mosaic()

    # Φ i Ψ
    ast_comp = phi_project_ast_to_hex(nodes, M, lam=0.3, gamma=0.7)
    psi_feedback(nodes, M, delta=0.15)
    ast_comp = phi_project_ast_to_hex(nodes, M, lam=0.4, gamma=0.7)

    # Ocena
    J_phi = phi_cost(M.edge, ast_comp, M)
    scores = {
        "CR_AST": float(CR_AST),
        "CR_TO": float(CR_TO),
        "Align3D": float(Align3D),
        "J_phi": float(J_phi),
        "J_total": float(0.5*J_phi + 0.25/CR_AST + 0.25/CR_TO)  # preferuj większą kompresję, mniejszy koszt Φ
    }
    print(json.dumps(scores, indent=2))

    # Protokół (do GUI/HUD)
    proto = export_protocol(nodes, M, L, assign, ast_comp, scores)
    # (tu byśmy zapisali do ctx.cache jako JSON)

    if show:
        # Rysunek: lewy — AST (rzut XY), prawy — heksy (stykające)
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
        for n in nodes.values():
            x,y,z=n.pos3d
            h=0.9+1.5*np.linalg.norm(n.meta)/np.sqrt(6)
            ax1.plot([x,x],[y,y+h],color="#374151",lw=2.0)
            ax1.scatter([x],[y+h],s=18,c=[(0.2,0.2,0.8,0.95)])
            if n.depth<=3: ax1.text(x,y+h+0.2,n.kind,fontsize=8,ha="center")
        ax1.set_aspect("equal","box")
        ax1.set_title("AST – kompas (rzut XY; h~‖meta‖)")

        centers=np.array([h.center for h in M.hexes])
        bmin,bmax=M.edge.min(),M.edge.max()
        for i,hx in enumerate(M.hexes):
            cx,cy=hx.center
            # stykające: radius = HEX_R, orientation=30°
            face = ( (M.edge[i]-bmin)/(bmax-bmin+1e-9), 0.2, 1.0-(M.edge[i]-bmin)/(bmax-bmin+1e-9), 0.96 )
            border_alpha = 0.25 + 0.7*ast_comp[i]
            poly = RegularPolygon((cx,cy), numVertices=6, radius=HEX_R, orientation=np.radians(30),
                                  facecolor=face, edgecolor=(0,0,0,border_alpha), lw=1.0)
            ax2.add_patch(poly)
        ax2.autoscale_view()
        ax2.set_aspect("equal","box")
        ax2.set_title("Mozaika 12×12 — heksy stykające (edge + wkład AST)")
        plt.tight_layout(); plt.show()

    return proto

if __name__ == "__main__":
    main()
