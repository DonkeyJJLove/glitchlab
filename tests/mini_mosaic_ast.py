# -*- coding: utf-8 -*-
"""
mini_mosaic_ast.py — lekki, czytelny szkic metaprzestrzeni (AST ⟷ Mozaika)
Autor: GlitchLab (demo)

Co pokazuje:
- Minimalny AST (lista węzłów + rodzice).
- Minimalna mozaika (siatka kafelków z cechą edge_density oraz warstwami: roi, ssim).
- Φ (AST→Mozaika): prosty plan — Denoise w ~edges, Blur w edges, Repair tam gdzie ssim spada.
- Ψ (Mozaika→AST): 1 reguła — jeśli dużo kafelków z niskim ssim, dołóż Repair(region=ssim<τ).
- Inwarianty I1–I4: szybkie sprawdzenie/druk (pass/warn).
- Wizualizacja: AST (2D radial) + mozaika (heatmapa SSIM) z konturem ROI i pasem krawędzi.

Wymagania: numpy, matplotlib
"""

from __future__ import annotations
import math, numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple

# -------------------------------
# 1) Minimalne modele
# -------------------------------

@dataclass
class ASTNode:
    id: int
    kind: str                  # 'Load','EdgeMap','Denoise','Blur','Blend','Metric','Repair','ROI'
    params: Dict = field(default_factory=dict)
    parent: Optional[int] = None
    children: List[int] = field(default_factory=list)

@dataclass
class Mosaic:
    rows: int
    cols: int
    edge: np.ndarray           # (R*C,) edge_density in [0,1]
    ssim: np.ndarray           # (R*C,) start=1.0
    roi: np.ndarray            # (R*C,) 0/1
    def ids(self) -> range: return range(self.rows*self.cols)

# -------------------------------
# 2) Budowa mini-świata
# -------------------------------

def build_mini_ast() -> Dict[int, ASTNode]:
    # Proste drzewko: Load → EdgeMap → {Denoise, Blur} → Blend → Metric
    nodes: Dict[int, ASTNode] = {}
    def add(i, kind, parent=None, **params):
        nodes[i] = ASTNode(i, kind, params, parent, [])
        if parent is not None: nodes[parent].children.append(i)
    add(0,'Load')
    add(1,'EdgeMap', parent=0, method='Sobel', thresh=0.55)
    add(2,'Denoise', parent=1, algo='NLM', strength=0.35, region='~edges')
    add(3,'Blur',    parent=1, name='Gaussian', sigma=1.8, region='edges')
    add(4,'Blend',   parent=0, alpha=0.5)           # traktujemy jako łącznik
    add(5,'Metric',  parent=4, name='SSIM')
    # symboliczne ROI (przyda się do I2 pokazowo)
    add(6,'ROI',     parent=0, shape='rect', rect=(0.3,0.3,0.6,0.6))  # x0,y0,x1,y1 w [0,1]
    return nodes

def build_mini_mosaic(R=10, C=14, seed=7) -> Mosaic:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:R, 0:C]
    diag = 1.0 - np.abs(xx-yy)/max(R,C)
    edge = np.clip(0.45 + 0.5*diag + 0.06*rng.standard_normal((R,C)),0,1).reshape(-1)
    ssim = np.ones(R*C)
    roi  = np.zeros(R*C)
    # prostokątny ROI (środek)
    r0, r1 = int(0.3*R), int(0.6*R)
    c0, c1 = int(0.3*C), int(0.6*C)
    for r in range(r0,r1):
        for c in range(c0,c1):
            roi[r*C+c]=1.0
    return Mosaic(R,C,edge,ssim,roi)

# -------------------------------
# 3) Φ: projekcja AST → mozaika
# -------------------------------

def region_to_ids(region:str, M:Mosaic, edge_thr=0.55) -> Set[int]:
    region = (region or 'ALL').strip()
    if region=='ALL': return set(M.ids())
    if region=='edges':   return {i for i,v in enumerate(M.edge) if v>edge_thr}
    if region=='~edges':  return {i for i,v in enumerate(M.edge) if v<=edge_thr}
    if region=='roi':     return {i for i,v in enumerate(M.roi)  if v>0.5}
    if region.startswith('ssim<'):
        t=float(region.split('<',1)[1])
        return {i for i,v in enumerate(M.ssim) if v<t}
    return set()

def phi_apply(nodes:Dict[int,ASTNode], M:Mosaic, edge_thr=0.55) -> List[str]:
    """Symuluje wpływ węzłów na warstwę SSIM (lekka, czytelna heurystyka)."""
    log=[]
    for n in nodes.values():
        if n.kind=='Denoise':
            ids = region_to_ids(n.params.get('region','~edges'), M, edge_thr)
            k   = n.params.get('strength',0.3)
            for i in ids:
                M.ssim[i] = np.clip(M.ssim[i] + 0.18*(1-M.edge[i])*k, 0,1)
            log.append(f"Denoise @~edges (+SSIM)")
        if n.kind=='Blur':
            ids = region_to_ids(n.params.get('region','edges'), M, edge_thr)
            sig = n.params.get('sigma',1.6)
            for i in ids:
                drop = 0.10*(M.edge[i])*sig/2
                M.ssim[i] = np.clip(M.ssim[i] - drop + 0.05*M.roi[i], 0,1)
            log.append(f"Blur @edges (-SSIM edge, feather in ROI)")
        if n.kind=='Repair':
            ids = region_to_ids(n.params.get('region','ssim<0.8'), M, edge_thr)
            lim = n.params.get('limit',0.2)
            for i in ids:
                M.ssim[i] = np.clip(M.ssim[i] + min(lim, 0.25*(0.9-M.ssim[i])), 0,1)
            log.append(f"Repair @ssim<thr (+SSIM)")
    return log

# -------------------------------
# 4) Ψ: podnoszenie mozaiki → AST
# -------------------------------

def psi_suggest(M:Mosaic, thr=0.80, frac=0.18) -> Optional[ASTNode]:
    low = [i for i,v in enumerate(M.ssim) if v<thr]
    if len(low)/len(M.ssim) > frac:
        return ASTNode(id=99, kind='Repair', params={'limit':0.25, 'region':f'ssim<{thr}'})
    return None

# -------------------------------
# 5) Inwarianty / stałe strukturalne (I1–I4)
# -------------------------------

def invariants_report(nodes:Dict[int,ASTNode], M:Mosaic, edge_thr=0.55) -> List[str]:
    rep=[]
    # I1: operacje nie poza nośnikiem (tu: czy filtrowanie nie deklaruje regionu sprzecznego)
    ok_I1=True
    for n in nodes.values():
        if n.kind in ('Denoise','Blur') and n.params.get('region') not in ('edges','~edges','roi',None):
            ok_I1=False
    rep.append(f"I1(types/regions): {'OK' if ok_I1 else 'WARN'}")
    # I2: spójność — jeśli filtr działa na ROI, złagodź granice (tu: tylko sygnał)
    edge_band = region_to_ids('edges', M, edge_thr)
    roi_ids   = region_to_ids('roi', M, edge_thr)
    boundary_overlap = len(edge_band & roi_ids)/max(1,len(roi_ids))
    rep.append(f"I2(sheaf continuity @ROI boundary): boundary_overlap≈{boundary_overlap:.2f} (expect small leak)")
    # I3: lokalność — denoise nie na krawędziach
    denoise_ok = True
    for n in nodes.values():
        if n.kind=='Denoise':
            sel = region_to_ids(n.params.get('region','~edges'), M, edge_thr)
            denoise_ok &= all(M.edge[i]<=edge_thr for i in sel)
    rep.append(f"I3(locality denoise/~edges): {'OK' if denoise_ok else 'WARN'}")
    # I4: monotoniczność — metryka nie powinna spadać globalnie (sprawdzimy po Φ)
    rep.append("I4(monotonicity SSIM): will check after Φ (Δglobal≥0)")
    return rep

# -------------------------------
# 6) Wizualizacja
# -------------------------------

def plot_ast(nodes:Dict[int,ASTNode], ax):
    # prosty layout radialny wg głębokości
    def depth(nid):
        d=0; p=nodes[nid].parent
        while p is not None: d+=1; p=nodes[p].parent
        return d
    layers: Dict[int,List[int]]={}
    for nid in nodes: layers.setdefault(depth(nid),[]).append(nid)
    for d in layers: layers[d].sort()
    pos: Dict[int,Tuple[float,float]]={}
    for d, ids in layers.items():
        R=0.8*(d+1)/max(1,len(layers))
        for j,nid in enumerate(ids):
            a=2*math.pi*j/max(1,len(ids))
            pos[nid]=(0.5+R*math.cos(a), 0.5+R*math.sin(a))
    # krawędzie
    for n in nodes.values():
        for ch in n.children:
            x0,y0=pos[n.id]; x1,y1=pos[ch]
            ax.plot([x0,x1],[y0,y1], color='0.7', lw=1.2, zorder=1)
    # węzły + etykiety
    for n in nodes.values():
        x,y = pos[n.id]
        ax.scatter([x],[y], s=80, c='white', edgecolors='black', zorder=2)
        ax.text(x, y, n.kind, ha='center', va='center', fontsize=8, zorder=3)
    ax.set_title("AST (radial 2D)")
    ax.axis('off')

def plot_mosaic(M:Mosaic, ax, title="Mosaic SSIM"):
    img = M.ssim.reshape(M.rows, M.cols)
    im  = ax.imshow(img, vmin=0.6, vmax=1.02, origin='upper')
    ax.set_title(title)
    # kontur ROI
    roi = M.roi.reshape(M.rows,M.cols)
    ax.contour(roi, levels=[0.5], colors='white', linewidths=1.2)
    # pas krawędzi (edge>thr) jako półtransparentna maska
    ax.imshow((M.edge.reshape(M.rows,M.cols)>0.55), alpha=0.12, cmap='Greys', origin='upper')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# -------------------------------
# 7) Demo end-to-end (czytelne)
# -------------------------------

def main():
    nodes = build_mini_ast()
    M     = build_mini_mosaic(R=10, C=14)

    print("== Invariants (before Φ) ==")
    for line in invariants_report(nodes, M): print("  -", line)

    ssim_before = M.ssim.copy()
    log = phi_apply(nodes, M)         # Φ: zastosuj plan
    d_ssim = float(M.ssim.mean() - ssim_before.mean())

    print("\n== Φ log ==")
    for l in log: print("  -", l)
    print(f"Δ SSIM (global mean): {d_ssim:+.3f}")

    patch = psi_suggest(M, thr=0.80, frac=0.18)  # Ψ
    if patch:
        print("\n== Ψ suggestion ==")
        print(f"  - {patch.kind} with {patch.params} in region {patch.params.get('region')} (would be added to AST)")
        # pokaż efekt hipotetycznie:
        nodes_s = {**nodes, patch.id: patch}
        phi_apply({patch.id: patch}, M)

    # I4: sprawdź monotoniczność po Φ
    ok_I4 = M.ssim.mean() >= ssim_before.mean() - 1e-6
    print("\n== Invariants (after Φ) ==")
    print(f"  - I4(monotonicity SSIM global): {'OK' if ok_I4 else 'WARN'}")

    # Wykresy
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,5))
    plot_ast(nodes, ax1)
    plot_mosaic(M, ax2, title="Mosaic SSIM (after Φ/Ψ)")
    plt.suptitle("Metaprzestrzeń: AST ↔ Mozaika (lekki szkic)", y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
