# -*- coding: utf-8 -*-
"""
demos/mosaic_ast_metalegend_demo.py
-----------------------------------
GlitchLab – "punkt 0" (możowanie) między AST i Mozaiką.
Każdy węzeł AST ma *metastrukturę* (stałe strukturalne) → wektor w meta-przestrzeni.
W punkcie 0 ten wektor "współgra" z mozaiką (region/warstwa), tworząc jedną, samopiszącą się strukturę.

Wizualizacja:
- 3D AST (kompas): Z = głębokość; na każdym węźle pionowy "meta-słupek" (wysokość=energia meta, kolor=entropia).
- 3D Mozaika: kratownica R×C (bar3d); wysokość/kolor = edge_density; ROI jako rama.
- "Punkt 0": dla każdego węzła liczymy centroid kafli, na których węzeł *działa* (region), i rysujemy wiązkę
  AST_node → fusion_point (centroid na mozaice), a w fusion_point stawiamy świecący marker.
- Legenda/infografika: komiksowe klocki z opisem meta-wymiarów oraz schemat Φ/Ψ i "punktu 0".

Autor: GlitchLab (demo edukacyjne)
"""

from __future__ import annotations
import ast, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D projection side-effect)


# ────────────────────────────────────────────────────────────────────────────────
# 1) AST: struktura + meta-wymiary node'ów
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class AstNode:
    id: int
    label: str
    depth: int
    parent: Optional[int]
    children: List[int] = field(default_factory=list)
    pos3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # meta: 6-wymiarowy wektor stałych strukturalnych w [0,1]
    # (locality, scale, selectivity, stability, causality, entropy)
    meta: Tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)

    @property
    def meta_energy(self) -> float:
        return float(np.linalg.norm(self.meta))

    @property
    def meta_entropy(self) -> float:
        return float(self.meta[-1])


EXAMPLE_SRC = r"""
def pipeline(img):
    R  = (120, 80, 200, 160)            # ROI
    E  = edges(img, method='Sobel', thresh=0.55)
    D  = denoise_nlm(img, strength=0.35)      # ~edges
    B  = gaussian_blur(img, sigma=1.8)        # edges
    Z  = blend(img, B, 0.5)
    M  = metric_ssim(img, Z)
    return blend(D, B, 0.5)
"""


def build_ast_with_meta(py_src: str) -> Dict[int, AstNode]:
    """Buduje AST i przypisuje meta-wektory na podstawie typu węzła (heurystyki demo)."""
    root = ast.parse(py_src)
    nodes: Dict[int, AstNode] = {}
    nid = 0

    def add(a: ast.AST, depth: int, parent: Optional[int]) -> int:
        nonlocal nid
        me = nid;
        nid += 1
        lab = a.__class__.__name__
        n = AstNode(me, lab, depth, parent)
        nodes[me] = n
        if parent is not None:
            nodes[parent].children.append(me)
        for ch in ast.iter_child_nodes(a):
            add(ch, depth + 1, me)
        return me

    add(root, 0, None)

    # Pozycjonowanie radialne
    by_depth: Dict[int, List[int]] = {}
    for i, n in nodes.items(): by_depth.setdefault(n.depth, []).append(i)
    for d, ids in by_depth.items():
        ids.sort()
        R = 6.0 + d * 2.0
        for j, i in enumerate(ids):
            a = 2 * math.pi * j / max(1, len(ids))
            nodes[i].pos3d = (R * math.cos(a), R * math.sin(a), d * 2.0)

    # Meta-heurystyki: im "bardziej kontrolny" node, tym wyższa selectivity/causality itp.
    rng = np.random.default_rng(42)
    for n in nodes.values():
        loc, sca, sel, stab, cau, ent = rng.uniform(0.25, 0.85, size=6)
        if n.label in ("If", "Compare"):
            sel, cau = 0.85, 0.80
        if n.label in ("Expr", "Call"):
            loc, sca = 0.65, 0.55
        if n.label in ("Return",):
            stab = 0.9
        if n.label in ("Assign",):
            sel = 0.7
        # delikatna normalizacja
        nodes[n.id].meta = (loc, sca, sel, stab, cau, ent)

    return nodes


# ────────────────────────────────────────────────────────────────────────────────
# 2) Mozaika: kratownica 3D (edge/ROI/SSIM)
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class Mosaic:
    rows: int
    cols: int
    edge: np.ndarray  # (N,) [0,1]
    ssim: np.ndarray  # (N,) [0,1] (baseline 1)
    roi: np.ndarray  # (N,) 0/1


def build_mosaic(rows=10, cols=14, seed=7) -> Mosaic:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols]
    diag = 1.0 - np.abs(xx - yy) / max(rows, cols)
    edge = np.clip(0.45 + 0.5 * diag + 0.06 * rng.standard_normal((rows, cols)), 0, 1).reshape(-1)
    ssim = np.ones(rows * cols)
    roi = np.zeros(rows * cols)
    r0, r1 = int(0.3 * rows), int(0.62 * rows)
    c0, c1 = int(0.3 * cols), int(0.62 * cols)
    for r in range(r0, r1):
        for c in range(c0, c1):
            roi[r * cols + c] = 1.0
    return Mosaic(rows, cols, edge, ssim, roi)


# ────────────────────────────────────────────────────────────────────────────────
# 3) Regiony/Φ: do jakich kafli "celuje" dany węzeł (symbolicznie)
# ────────────────────────────────────────────────────────────────────────────────

def region_for_node(n: AstNode, M: Mosaic, edge_thr=0.55) -> Set[int]:
    """Symboliczna mapa: część Expr'ów traktujemy jak Denoise(~edges) / Blur(edges)."""
    if n.label == "Assign":  # ROI
        return {i for i, v in enumerate(M.roi) if v > 0.5}
    if n.label == "Expr":
        # heurystyka: parzyste → blur(edges), nieparzyste → denoise(~edges)
        if (n.id % 2) == 0:
            return {i for i, v in enumerate(M.edge) if v > edge_thr}
        else:
            return {i for i, v in enumerate(M.edge) if v <= edge_thr}
    if n.label in ("Return", "If"):
        return set(range(M.rows * M.cols))  # global
    return set()


def centroid_of_tiles(ids: Set[int], M: Mosaic) -> Tuple[float, float, float]:
    """Centroid w przestrzeni mozaiki (x:kolumna, y:wiersz, z: 'edge-height')."""
    if not ids:
        return (M.cols * 0.5, M.rows * 0.5, 0.0)
    rows, cols = M.rows, M.cols
    cs = np.array([i % cols for i in ids], dtype=float)
    rs = np.array([i // cols for i in ids], dtype=float)
    zs = np.array([2.0 * M.edge[i] + 0.1 for i in ids], dtype=float)
    return (float(cs.mean() + 0.42), float(rs.mean() + 0.42), float(zs.mean()))


# ────────────────────────────────────────────────────────────────────────────────
# 4) Rysowanie – AST 3D, Mozaika 3D, wiązki i „punkty 0”, legenda-infografika
# ────────────────────────────────────────────────────────────────────────────────

def draw_ast_3d(ax, nodes: Dict[int, AstNode]):
    # krawędzie
    for n in nodes.values():
        x0, y0, z0 = n.pos3d
        for cid in n.children:
            x1, y1, z1 = nodes[cid].pos3d
            ax.plot([x0, x1], [y0, y1], [z0, z1], color="#7b8fa1", lw=1.0)
    # węzły jako "meta-słupki"
    cmap = plt.get_cmap("plasma")
    for n in nodes.values():
        x, y, z = n.pos3d
        h = 1.2 + 2.2 * (n.meta_energy / np.sqrt(6))  # norm meta max≈√6
        c = cmap(n.meta_entropy)  # kolor = entropia
        ax.plot([x, x], [y, y], [z, z + h], color=c, lw=3.0)  # słupek
        ax.scatter([x], [y], [z + h], s=25, c=[c], edgecolors="black", depthshade=True)
        if n.depth <= 3:
            ax.text(x, y, z + h + 0.4, n.label, fontsize=8, ha="center")
    ax.set_title("AST → Kompas 3D z metaglifami")
    ax.set_xlabel("X");
    ax.set_ylabel("Y");
    ax.set_zlabel("Z (depth/meta)")


def draw_mosaic_3d(ax, M: Mosaic):
    R, C = M.rows, M.cols
    xs, ys, zs, dx, dy, dz, colors = [], [], [], [], [], [], []
    for r in range(R):
        for c in range(C):
            i = r * C + c
            h = 2.0 * M.edge[i] + 0.1
            xs.append(c);
            ys.append(r);
            zs.append(0.0)
            dx.append(0.85);
            dy.append(0.85);
            dz.append(h)
            colors.append((M.edge[i], 0.2, 1.0 - M.edge[i], 0.95))
    ax.bar3d(xs, ys, zs, dx, dy, dz, color=colors, linewidth=0.1, shade=True)
    # ROI jako rama
    roi = M.roi.reshape(R, C)
    rr, cc = np.where(roi > 0.5)
    if len(rr) > 0:
        rmin, rmax = rr.min(), rr.max()
        cmin, cmax = cc.min(), cc.max()
        z = 2.4
        ax.plot([cmin, cmax, cmax, cmin, cmin], [rmin, rmin, rmax, rmax, rmin], [z, z, z, z, z], color="white", lw=1.8)
    ax.set_title("Mozaika 3D (edge density)")
    ax.set_xlabel("cols");
    ax.set_ylabel("rows");
    ax.set_zlabel("edge→height")
    ax.view_init(elev=24, azim=-58)


def draw_fusion_links(ax_ast, ax_mos, nodes: Dict[int, AstNode], M: Mosaic):
    """Rysuje wiązki AST→punkt0 oraz markery punktów 0 na mozaice."""
    rng = np.random.default_rng(0)
    for n in nodes.values():
        ids = region_for_node(n, M)
        if not ids: continue
        # punkt 0 (centroid)
        cx, cy, cz = centroid_of_tiles(ids, M)
        # „świat mozaiki” jest w innych jednostkach – narysuj marker na osi mozaiki:
        ax_mos.scatter([cx], [cy], [cz + 0.05], s=35, c="#ffd166", edgecolors="black", depthshade=True, zorder=10)
        # wiązka z AST: cienka linia (kolor od entropii)
        c = plt.get_cmap("plasma")(n.meta_entropy)
        x0, y0, z0 = n.pos3d
        # aby było wyraźnie, końcówkę linii pozycjonujemy w *ramce AST*, ale w kierunku mozaiki:
        # ułóż "pseudo-most": AST→(średni wektor w stronę sceny mozaiki)
        # (prosto: przeskaluj koord. mozaiki do chmury wokół AST)
        x1, y1, z1 = x0 + 0.15 * (cx - M.cols * 0.5), y0 + 0.15 * (cy - M.rows * 0.5), z0 + 0.3
        ax_ast.plot([x0, x1], [y0, y1], [z0, x1 * 0 + z1], color=c, lw=0.9, alpha=0.85)


def draw_infographic(ax):
    """Infografika – klocki znaczeń, nie 'lista'."""
    ax.axis("off")
    # bloki (rounded) + krótkie hasła
    blocks = [
        ("Locality (L)", "gdzie działa\n(ROI / ~edges / global)"),
        ("Scale (S)", "na jakiej skali\n(tile / multi-scale)"),
        ("Selectivity (Sel)", "jak wybiera\n(progi, reguły)"),
        ("Stability (Stab)", "czy utrzymuje\nspójność (I2)"),
        ("Causality (Cau)", "wpływ na resztę\n(sterowanie Ψ)"),
        ("Entropy (H)", "stopień wolności\n/ niepewność"),
    ]
    x0, y0, dx, dy = 0.05, 0.55, 0.28, 0.12
    for i, (title, desc) in enumerate(blocks):
        xi = x0 + (i % 3) * dx
        yi = y0 - (i // 3) * dy
        box = FancyBboxPatch((xi, yi), dx - 0.02, dy - 0.02, boxstyle="round,pad=0.02,rounding_size=0.02",
                             linewidth=1.2, edgecolor="#334155", facecolor="#e2e8f0")
        ax.add_patch(box)
        ax.text(xi + 0.01, yi + dy - 0.05, title, fontsize=10, weight="bold", color="#0f172a")
        ax.text(xi + 0.01, yi + 0.02, desc, fontsize=9, color="#1f2937")
    # Schemat Φ / Ψ + „punkt 0”
    ax.text(0.05, 0.40, "Φ: AST → Mozaika", fontsize=11, weight="bold", color="#0f766e")
    ax.text(0.05, 0.35, "node.meta  ⟶  selektor(region)  ⟶  centroid tiles  =  punkt 0", fontsize=9)
    ax.text(0.05, 0.28, "Ψ: Mozaika → AST", fontsize=11, weight="bold", color="#7c2d12")
    ax.text(0.05, 0.23, "warstwy/metryki  ⟶  reguły podnoszenia  ⟶  nowy/zmieniony node", fontsize=9)
    # Wzór energii / kolorystyki meta-glifu
    ax.text(0.05, 0.12, "Metaglif węzła:", fontsize=11, weight="bold")
    ax.text(0.05, 0.08, "wysokość = ‖(L,S,Sel,Stab,Cau,H)‖   |   kolor = H (entropia)", fontsize=9)
    # I1–I4 piktogramy
    ax.text(0.62, 0.40, "Inwarianty:", fontsize=11, weight="bold")
    ax.text(0.62, 0.34, "I1  typy/nośniki  ✓", fontsize=9)
    ax.text(0.62, 0.29, "I2  spójność (sheaf)  ⇄", fontsize=9)
    ax.text(0.62, 0.24, "I3  lokalność/leak ≤ δ", fontsize=9)
    ax.text(0.62, 0.19, "I4  monotoniczność celu", fontsize=9)


# ────────────────────────────────────────────────────────────────────────────────
# 5) Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    nodes = build_ast_with_meta(EXAMPLE_SRC)
    M = build_mosaic(rows=10, cols=14)

    # Scena: 3 panele – AST 3D, Mozaika 3D, infografika
    fig = plt.figure(figsize=(14, 6))
    ax_ast = fig.add_subplot(1, 3, 1, projection="3d")
    ax_mos = fig.add_subplot(1, 3, 2, projection="3d")
    ax_inf = fig.add_subplot(1, 3, 3)

    draw_ast_3d(ax_ast, nodes)
    draw_mosaic_3d(ax_mos, M)
    draw_fusion_links(ax_ast, ax_mos, nodes, M)
    draw_infographic(ax_inf)

    fig.suptitle("GlitchLab – Punkt 0: możowanie metastruktur AST z mozaiką", y=0.98)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
