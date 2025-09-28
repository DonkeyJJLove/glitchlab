# -*- coding: utf-8 -*-
"""
demos/mosaic_ast_3d_demo.py
---------------------------------
GlitchLab – 3D demo metaprzestrzeni: AST ↔ Mozaika (+ projekcja Φ)

Co pokazuje:
- AST 3D ("kompas"): Z = głębokość, X/Y = układ radialny w obrębie warstwy.
- Mozaika 3D: siatka R×C z kaflami jako słupki bar3d (wysokość/kolor = edge_density).
- Projekcja Φ: "Denoise(~edges)" i "Blur(edges)" rysują wiązki z węzłów AST do (pod)zbiorów kafli.
- Raport: szybkie inwarianty I1–I4 i metryki zgodności (d_AST, d_M, d_Φ).

Jak to się ma do GlitchLab:
- AST tutaj to abstrakt pipeline'u; w GlitchLab przechowywany jako JSON (cache["ast/json"]).
- Mozaika to warstwa diagnostyczna; w GlitchLab trafia do cache jako mosaic/* (tiles, features, layers).
- Φ/Ψ w repo idą do core.agent.phi / core.agent.psi; GUI (HUD/GraphView) tylko renderuje i przełącza warstwy.

Autor: Zespół GlitchLab (demo edukacyjne)
"""

from __future__ import annotations
import ast, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (import side-effect for 3D)


# ────────────────────────────────────────────────────────────────────────────────
# 1) AST → struktura i pozycjonowanie 3D (kompas)
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class AstNode:
    id: int
    label: str
    depth: int
    parent: Optional[int]
    children: List[int] = field(default_factory=list)
    pos3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)

EXAMPLE_SRC = r"""
def pipeline(img):
    # szkic – nie wykonujemy, tylko parsujemy
    R = (120, 80, 200, 160)  # ROI
    E = edges(img, method='Sobel', thresh=0.55)
    D = denoise_nlm(img, strength=0.35)          # ~edges
    B = gaussian_blur(img, sigma=1.8)            # edges
    M = metric_ssim(img, blend(img, B, 0.5))
    return blend(D, B, 0.5)
"""

def build_ast_3d(py_src: str) -> Dict[int, AstNode]:
    root = ast.parse(py_src)
    nodes: Dict[int, AstNode] = {}
    nid = 0

    def add(a: ast.AST, depth: int, parent: Optional[int]) -> int:
        nonlocal nid
        me = nid; nid += 1
        lab = a.__class__.__name__
        nodes[me] = AstNode(me, lab, depth, parent)
        if parent is not None:
            nodes[parent].children.append(me)
        for ch in ast.iter_child_nodes(a):
            add(ch, depth+1, me)
        return me

    add(root, 0, None)

    # radial by depth: każdy poziom na pierścieniu, Z=depth
    by_depth: Dict[int, List[int]] = {}
    for i, n in nodes.items():
        by_depth.setdefault(n.depth, []).append(i)

    for d, ids in by_depth.items():
        ids.sort()
        R = 6.0 + d*2.0
        for j, i in enumerate(ids):
            a = 2*math.pi*j/max(1, len(ids))
            x = R*math.cos(a)
            y = R*math.sin(a)
            z = d*2.0
            nodes[i].pos3d = (x, y, z)

    return nodes


# ────────────────────────────────────────────────────────────────────────────────
# 2) Mozaika 3D (R×C słupków) + cechy
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class Mosaic:
    rows: int
    cols: int
    edge: np.ndarray   # (N,) w [0,1]
    ssim: np.ndarray   # (N,) baseline = 1.0
    roi:  np.ndarray   # (N,) 0/1

def build_mosaic(rows=10, cols=14, seed=7) -> Mosaic:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols]
    # pas krawędzi wzdłuż przekątnej
    diag = 1.0 - np.abs(xx-yy)/max(rows, cols)
    edge = np.clip(0.45 + 0.5*diag + 0.06*rng.standard_normal((rows, cols)), 0, 1).reshape(-1)
    ssim = np.ones(rows*cols)
    roi  = np.zeros(rows*cols)
    r0, r1 = int(0.30*rows), int(0.62*rows)
    c0, c1 = int(0.30*cols), int(0.62*cols)
    for r in range(r0, r1):
        for c in range(c0, c1):
            roi[r*cols+c] = 1.0
    return Mosaic(rows, cols, edge, ssim, roi)


# ────────────────────────────────────────────────────────────────────────────────
# 3) Projekcja Φ (AST→Mozaika) – wybór kafli i logika linii
# ────────────────────────────────────────────────────────────────────────────────

def region_ids(region: str, M: Mosaic, edge_thr=0.55) -> Set[int]:
    if region == 'ALL': return set(range(M.rows*M.cols))
    if region == 'edges':   return {i for i,v in enumerate(M.edge) if v>edge_thr}
    if region == '~edges':  return {i for i,v in enumerate(M.edge) if v<=edge_thr}
    if region == 'roi':     return {i for i,v in enumerate(M.roi)  if v>0.5}
    if region.startswith('ssim<'):
        thr = float(region.split('<',1)[1])
        return {i for i,v in enumerate(M.ssim) if v<thr}
    return set()

def ast_interest_nodes(nodes: Dict[int, AstNode]) -> Dict[str, int]:
    """
    Heurystycznie wskaż 3 węzły 'tematyczne' do połączenia z mozaiką.
    (W realnym GlitchLab robi to parser AST pipeline'u i rejestr filtrów).
    """
    by_label = {}
    for n in nodes.values():
        by_label.setdefault(n.label, []).append(n.id)
    # wybierz po 1 sztuce (gdy brak – None)
    pick = lambda lbl: by_label.get(lbl, [None])[0]
    return {
        'Denoise': pick('Expr'),  # demo: Expr ~ denoise
        'Blur':    pick('Expr'),  # drugi Expr ~ blur (tylko do wizualizacji)
        'If':      pick('If'),
        'Return':  pick('Return'),
    }

def phi_demo_plans(M: Mosaic) -> Dict[str, Set[int]]:
    """Dwa regiony do pokazania projekcji: edges (dla blur) i ~edges (dla denoise)."""
    return {
        'denoise_region': region_ids('~edges', M),
        'blur_region':    region_ids('edges',  M),
        'roi_region':     region_ids('roi',    M),
    }


# ────────────────────────────────────────────────────────────────────────────────
# 4) Pseudometryki / inwarianty (lekkie)
# ────────────────────────────────────────────────────────────────────────────────

def d_ast(nodes: Dict[int, AstNode]) -> float:
    E = sum(len(n.children) for n in nodes.values())
    depth_pen = sum(n.depth**1.15 for n in nodes.values())
    return float(E + 0.02*depth_pen)

def d_mosaic(M: Mosaic) -> float:
    return float(np.var(M.edge) + 0.5*np.var(M.ssim))

def d_phi_cost(M: Mosaic, denoise_ids:Set[int], blur_ids:Set[int], thr=0.55) -> float:
    # kara: denoise na krawędziach + blur na nie-krawędziach
    cost  = sum(M.edge[i] for i in denoise_ids)*0.1
    cost += sum(1.0-M.edge[i] for i in blur_ids)*0.1
    return float(cost)

def invariants_summary(M: Mosaic, denoise_ids:Set[int], blur_ids:Set[int], thr=0.55) -> Dict[str,str]:
    I1 = "OK"  # w tym demo regio selekcje są poprawne typowo
    I3 = "OK" if all(M.edge[i]<=thr for i in denoise_ids) else "WARN"
    # granica ROI dotyka pas krawędzi? informacyjnie
    roi_set   = set(region_ids('roi', M))
    edges_set = set(region_ids('edges', M, thr))
    overlap   = len(roi_set & edges_set)/max(1,len(roi_set))
    I2 = f"boundary_overlap≈{overlap:.2f} (expect small leak control)"
    I4 = "check after Φ (Δ mean SSIM ≥ 0)"
    return {"I1":I1, "I2":I2, "I3":I3, "I4":I4}


# ────────────────────────────────────────────────────────────────────────────────
# 5) Rysowanie 3D
# ────────────────────────────────────────────────────────────────────────────────

def draw_ast_3d(ax, nodes: Dict[int, AstNode], title="AST → Kompas 3D"):
    # krawędzie
    for n in nodes.values():
        x0,y0,z0 = n.pos3d
        for cid in n.children:
            x1,y1,z1 = nodes[cid].pos3d
            ax.plot([x0,x1],[y0,y1],[z0,z1], color="#7b8fa1", lw=1.0)
    # węzły
    for n in nodes.values():
        x,y,z = n.pos3d
        ax.scatter([x],[y],[z], s=20, c="white", edgecolors="black", depthshade=True)
        if n.depth<=3:
            ax.text(x, y, z+0.6, n.label, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (depth)")

def draw_mosaic_3d(ax, M: Mosaic, title="Mosaic 3D (edge density)"):
    R,C = M.rows, M.cols
    xs, ys, zs, dx, dy, dz, colors = [], [], [], [], [], [], []
    # słupek w (c,r) o wysokości proporcjonalnej do edge
    for r in range(R):
        for c in range(C):
            i = r*C+c
            h = 2.0*M.edge[i] + 0.1  # [0.1..2.1]
            xs.append(c); ys.append(r); zs.append(0.0)
            dx.append(0.85); dy.append(0.85); dz.append(h)
            colors.append( (M.edge[i], 0.2, 1.0-M.edge[i], 0.9) )  # RGBA
    ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True, color=colors, linewidth=0.1)
    # ROI kontur jako „rama” (drut)
    roi = M.roi.reshape(R,C)
    rr, cc = np.where(roi>0.5)
    if len(rr)>0:
        rmin,rmax = rr.min(), rr.max()
        cmin,cmax = cc.min(), cc.max()
        z = 2.3
        ax.plot([cmin, cmax, cmax, cmin, cmin],
                [rmin, rmin, rmax, rmax, rmin],
                [z,z,z,z,z], color="white", lw=1.8, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("cols"); ax.set_ylabel("rows"); ax.set_zlabel("edge→height")
    ax.view_init(elev=25, azim=-60)

def draw_phi_links(ax, src_pos: Tuple[float,float,float], M: Mosaic, tile_ids: Set[int], max_lines=50, color="#ff8a00"):
    """Wiązki z jednego węzła AST do centów kafli (losowo–rzadko, by nie śmiecić)."""
    R,C = M.rows, M.cols
    ids = list(tile_ids)
    if len(ids) > max_lines:
        ids = list(np.random.default_rng(0).choice(ids, size=max_lines, replace=False))
    x0,y0,z0 = src_pos
    for i in ids:
        r, c = divmod(i, C)
        # centroid słupka (w mozaice)
        x1 = c + 0.42
        y1 = r + 0.42
        z1 = 2.2  # nad wierzchołkiem
        ax.plot([x0,x1], [y0,y1], [z0,z1], color=color, lw=0.7, alpha=0.7)


# ────────────────────────────────────────────────────────────────────────────────
# 6) Main – spinamy całość
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # Buduj struktury
    nodes = build_ast_3d(EXAMPLE_SRC)
    M      = build_mosaic(rows=10, cols=14)
    picks  = ast_interest_nodes(nodes)
    plans  = phi_demo_plans(M)

    # Raport
    denoise_ids = plans['denoise_region']
    blur_ids    = plans['blur_region']
    inv = invariants_summary(M, denoise_ids, blur_ids)
    print("== Invariants ==")
    for k,v in inv.items(): print(f"  - {k}: {v}")
    print(f"d_AST={d_ast(nodes):.3f} | d_M={d_mosaic(M):.3f} | d_Φ≈{d_phi_cost(M, denoise_ids, blur_ids):.3f}")

    # Rysuj scenę 3D (2 osie: AST i Mozaika)
    fig = plt.figure(figsize=(13,6))
    ax_ast = fig.add_subplot(1,2,1, projection='3d')
    ax_mos = fig.add_subplot(1,2,2, projection='3d')

    draw_ast_3d(ax_ast, nodes)
    draw_mosaic_3d(ax_mos, M)

    # Projekcja Φ – linki z "denoise" (nie-krawędzie) i "blur" (krawędzie)
    # (tu bierzemy dwa różne węzły Expr jako „uchwyty” – w realnym GL mapping jest po nazwie filtra)
    expr_nodes = [n for n in nodes.values() if n.label=="Expr"]
    if expr_nodes:
        src_denoise = expr_nodes[0].pos3d
        src_blur    = expr_nodes[-1].pos3d if len(expr_nodes)>1 else expr_nodes[0].pos3d
        draw_phi_links(ax_ast, src_denoise, M, denoise_ids, color="#16a34a")  # zielone
        draw_phi_links(ax_ast, src_blur,    M, blur_ids,    color="#e11d48")  # różowe

    # Podpisz, by było jasne co oglądamy
    fig.suptitle("GlitchLab – 3D metaprzestrzeń: AST (kompas) ↔ Mozaika (kratownica) + projekcja Φ", y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
