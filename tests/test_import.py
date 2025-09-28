# -*- coding: utf-8 -*-
"""
mosaic_ast_experiment.py
---------------------------------
Eksperymentalny prototyp "Mozaikowego Drzewa AST" dla GlitchLab.

Co demonstruje:
- Parsowanie AST z podanego źródła Pythona (ast.parse) i budowę grafu DAG.
- Rzutowanie AST do przestrzeni 3D (tzw. "kompas AST"): Z=poziom/głębokość, X/Y=porządek/topologia.
- Zdefiniowanie mozaiki kafelków (grid), cech per-kafelek i warstw (edge, ssim, roi).
- Funktor Φ (AST→Mozaika): selekcja kafli, operacje per-kafelek (blur/denoise/sharpen/blend/roi).
- Funktor Ψ (Mozaika→AST): generowanie pod-uzasadnionych poprawek do AST gdy per-kafelek SSIM spada.
- Pseudometryki d_AST, d_M, d_Φ (szacunkowe i lekkie).
- Wizualizacja: 3D kompas AST + 2D heatmapa mozaiki (warstwa 'ssim' po projekcji).

Autor: GlitchLab – eksperymentalny model meta-poziomu (AST ⟷ Mozaika).
"""

from __future__ import annotations
import ast
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# =========================================================
# 1) AST → struktura + pozycjonowanie 3D (kompas)
# =========================================================

@dataclass
class AstNode:
    id: int
    label: str
    depth: int
    parent: Optional[int]
    children: List[int] = field(default_factory=list)
    # współrzędne do wizualizacji (ustalimy po zbudowaniu drzewa)
    pos3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # "wektor strukturalny" – mała sygnatura cech
    sig: Tuple[int, int] = (0, 0)  # (arity, label_hash_bucket)


def _label_of(node: ast.AST) -> str:
    return node.__class__.__name__


def build_ast_graph(py_src: str) -> Dict[int, AstNode]:
    """
    Tworzy DAG z ast.parse(py_src). Każdy węzeł ma: id, label, depth, parent, children.
    Następnie przydzielamy proste położenia 3D: Z = depth, X/Y = indeksy porządkowe.
    """
    root_ast = ast.parse(py_src)
    nodes: Dict[int, AstNode] = {}
    next_id = 0

    def add_node(py_node: ast.AST, depth: int, parent: Optional[int]) -> int:
        nonlocal next_id
        nid = next_id
        next_id += 1
        lab = _label_of(py_node)
        n = AstNode(id=nid, label=lab, depth=depth, parent=parent)
        nodes[nid] = n
        if parent is not None:
            nodes[parent].children.append(nid)
        # rekurencja po dzieciach
        for child in ast.iter_child_nodes(py_node):
            add_node(child, depth + 1, nid)
        return nid

    add_node(root_ast, depth=0, parent=None)

    # Pozycjonowanie: prosty layout – porządek DFS nadaje X, kolumna na głębokości -> Y,
    # Z = depth (kompas "w głąb"), Y w oparciu o indeks w danej głębokości.
    by_depth: Dict[int, List[int]] = {}
    for nid, n in nodes.items():
        by_depth.setdefault(n.depth, []).append(nid)

    for d, ids in by_depth.items():
        ids.sort()  # deterministycznie
        for j, nid in enumerate(ids):
            n = nodes[nid]
            # X – pozycja DFS (zwykły id wystarcza), Y – indeks w warstwie, Z – głębokość
            x = float(nid)
            y = float(j)
            z = float(n.depth)
            n.pos3d = (x, y, z)
            # sygnatura: arity + hash bucket labelu
            n.sig = (len(n.children), (hash(n.label) % 97))

    return nodes


# =========================================================
# 2) Mozaika – grid kafelków, cechy, warstwy
# =========================================================

@dataclass
class Tile:
    """Reprezentuje kafelek: maska (indeks), cechy, waga, etykieta."""
    index: int
    feats: Dict[str, float]
    weight: float = 1.0
    label: Optional[str] = None


@dataclass
class Mosaic:
    tiles: List[Tile]
    shape: Tuple[int, int]  # (rows, cols)
    layers: Dict[str, np.ndarray]  # nazwa -> wartości per-tile (len = R*C)
    adj: Dict[int, List[int]]      # sąsiedztwo kafelków

    def tile_ids(self) -> Iterable[int]:
        return range(len(self.tiles))


def build_grid_mosaic(rows: int = 12, cols: int = 12, seed: int = 7) -> Mosaic:
    """
    Tworzy sztuczną mozaikę (grid R×C). Featy per-tile: edge_density, var, band_hi.
    Te featy to syntetyczne funkcje od położenia i lekkiego szumu – wystarczą do demo Φ/Ψ.
    """
    rng = np.random.default_rng(seed)
    tiles: List[Tile] = []
    N = rows * cols

    # syntetyczne pola: "krawędzie" silniejsze w pasie przekątnej, wariancja w centrum
    grid_y, grid_x = np.mgrid[0:rows, 0:cols]
    g = (np.abs(grid_x - grid_y) / max(rows, cols))  # pas diagonalny
    edge_density = 0.4 + 0.6 * (1.0 - g) + 0.05 * rng.standard_normal(size=(rows, cols))
    edge_density = np.clip(edge_density, 0, 1)

    yy = (grid_y - rows / 2.0) / (rows / 2.0)
    xx = (grid_x - cols / 2.0) / (cols / 2.0)
    r2 = xx**2 + yy**2
    local_var = np.clip(1.0 - r2 + 0.05 * rng.standard_normal(size=(rows, cols)), 0, 1)

    band_hi = np.clip(0.5 + 0.5 * np.sin(2 * math.pi * grid_x / max(2, cols - 1))
                      + 0.05 * rng.standard_normal(size=(rows, cols)), 0, 1)

    for i in range(N):
        r = i // cols
        c = i % cols
        feats = {
            "edge_density": float(edge_density[r, c]),
            "local_var": float(local_var[r, c]),
            "band_hi": float(band_hi[r, c]),
        }
        tiles.append(Tile(index=i, feats=feats))

    layers = {
        "edge": edge_density.reshape(-1),
        "ssim": np.ones(N),            # baseline SSIM = 1.0
        "roi": np.zeros(N),            # ROI mask (0/1)
        "diff": np.zeros(N),           # magnitude of change (później uzupełnimy)
    }

    # sąsiedztwo 4-kierunkowe
    adj: Dict[int, List[int]] = {i: [] for i in range(N)}
    for i in range(N):
        r = i // cols
        c = i % cols
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                adj[i].append(rr * cols + cc)

    return Mosaic(tiles=tiles, shape=(rows, cols), layers=layers, adj=adj)


# =========================================================
# 3) AST↦Mozaika (Φ) – selektory i akcje per-tile
# =========================================================

@dataclass
class NodePlan:
    """Abstrakcyjne polecenie Φ dla pojedynczego węzła."""
    action: str                      # 'blur'|'denoise'|'sharpen'|'blend'|'set_roi'|...
    tile_ids: Set[int]               # zbiór wybranych kafli
    params: Dict[str, Any] = field(default_factory=dict)


def region_to_tiles(region_expr: str, M: Mosaic, thr_map: Dict[str, float]) -> Set[int]:
    """
    Prosty parser regionów:
      - 'ALL' – wszystkie kafle
      - 'edges' – edge > thr_map['edge']
      - '~edges' – negacja
      - 'roi' – kafle z warstwy 'roi' > 0
      - 'ssim<0.8' – warunek na warstwie ssim
    """
    region_expr = (region_expr or "ALL").strip()
    if region_expr == "ALL":
        return set(M.tile_ids())

    if region_expr == "edges":
        tau = thr_map.get("edge", 0.5)
        e = M.layers["edge"]
        return {i for i, v in enumerate(e) if v > tau}

    if region_expr == "~edges":
        tau = thr_map.get("edge", 0.5)
        e = M.layers["edge"]
        return {i for i, v in enumerate(e) if v <= tau}

    if region_expr == "roi":
        r = M.layers["roi"]
        return {i for i, v in enumerate(r) if v > 0.5}

    if region_expr.startswith("ssim<"):
        try:
            val = float(region_expr.split("<", 1)[1])
        except Exception:
            val = 0.8
        s = M.layers["ssim"]
        return {i for i, v in enumerate(s) if v < val}

    # fallback: nic
    return set()


def phi_project(nodes: Dict[int, AstNode], M: Mosaic,
                thr_map: Dict[str, float]) -> List[NodePlan]:
    """
    Przepisuje węzły AST na proste akcje mozaikowe.
    Używa nazw etykiet AST (np. 'FunctionDef','If','Compare','Return','Expr', ...)
    oraz heurystyk parametryzowanych przez thr_map.
    """
    plans: List[NodePlan] = []
    # Syntetycznie: gdy w AST znajdziemy:
    #  - "If" z "Compare ssim < τ" → plan naprawczy na ssim<τ (LocalContrast)
    #  - "Expr/Call" zawierający 'Gaussian' → blur na 'edges'
    #  - "Expr/Call" zawierający 'NLM'/'Denoise' → denoise na '~edges'
    #  - "Assign" z identyfikatorem 'ROI' → set_roi na zadanym prostokącie (symulujemy)
    for n in nodes.values():
        lab = n.label

        # prosty ROI – jeśli w kodzie widzimy Assign do nazwy 'R'/'ROI', ustawmy warstwę roi
        if lab == "Assign":
            # heurystycznie: 1/3 kafli w środku jako ROI
            rows, cols = M.shape
            roi = np.zeros(rows * cols)
            r0, r1 = rows // 3, 2 * rows // 3
            c0, c1 = cols // 3, 2 * cols // 3
            for r in range(r0, r1):
                for c in range(c0, c1):
                    roi[r * cols + c] = 1.0
            M.layers["roi"] = roi
            plans.append(NodePlan("set_roi", set(np.nonzero(roi)[0].tolist())))
            continue

        if lab == "If":
            # naprawa: "ssim<tau" (heurystycznie: weź region 'ssim<0.8')
            tiles = region_to_tiles("ssim<0.8", M, thr_map)
            if tiles:
                plans.append(NodePlan("repair_contrast", tiles, {"limit": 0.2}))
            continue

        if lab == "Expr":
            # heurystyka: patrzmy na podpis "dzieci" – jeśli węzeł niżej to np. Call('Gaussian'/'NLM')
            # Tu bez parsowania atrybutów wywołań – syntetyczna wersja:
            # - co ~3-cia Expr → Gaussian blur na krawędziach
            # - co ~5-ta Expr → Denoise na ~edges
            if (n.id % 5) == 1:
                tiles = region_to_tiles("~edges", M, thr_map)
                if tiles:
                    plans.append(NodePlan("denoise", tiles, {"strength": 0.35, "algo": "NLM"}))
            if (n.id % 3) == 2:
                tiles = region_to_tiles("edges", M, thr_map)
                if tiles:
                    plans.append(NodePlan("blur", tiles, {"sigma": 1.8}))
            continue

        if lab == "Return":
            # końcowa "blend" – syntetycznie mieszamy wynik z oryginałem
            plans.append(NodePlan("blend", set(M.tile_ids()), {"alpha": 0.5}))
            continue

    return plans


def phi_execute(M: Mosaic, plans: List[NodePlan]) -> None:
    """
    Symuluje efekty planów per-tile, aktualizuje: layers['ssim'], layers['diff'].
    Nie dotyka realnych pikseli – operuje na cechach syntetycznych.
    """
    ssim = M.layers["ssim"].copy()
    diff = M.layers["diff"].copy()
    edge = M.layers["edge"]
    roi = M.layers["roi"]

    for plan in plans:
        if plan.action == "set_roi":
            # już zrobione w phi_project – tu nic
            continue

        if plan.action == "blur":
            # blur na krawędziach: minimalnie redukuj edge i ssim w pasie krawędzi, ale popraw w ROI
            sig = float(plan.params.get("sigma", 2.0))
            for i in plan.tile_ids:
                # większy wpływ gdy edge większe
                k = min(0.15, 0.08 + 0.08 * edge[i] * sig / 2.0)
                ssim[i] = np.clip(ssim[i] - 0.10 * k + 0.05 * roi[i], 0.0, 1.0)
                diff[i] += 0.3 * k

        if plan.action == "denoise":
            # denoise na ~edges: zwiększ ssim (głównie poza krawędziami), niewielka zmiana diff
            strength = float(plan.params.get("strength", 0.3))
            for i in plan.tile_ids:
                k = min(0.25, 0.12 + 0.15 * (1.0 - edge[i]) * strength)
                ssim[i] = np.clip(ssim[i] + k, 0.0, 1.0)
                diff[i] += 0.1 * k

        if plan.action == "repair_contrast":
            # lokalny kontrast: tam gdzie ssim niskie – podbij
            limit = float(plan.params.get("limit", 0.2))
            for i in plan.tile_ids:
                k = min(limit, 0.25 * (0.85 - ssim[i]))
                ssim[i] = np.clip(ssim[i] + k, 0.0, 1.0)
                diff[i] += 0.05 * k

        if plan.action == "blend":
            # mieszanie finalne: delikatnie 'uspokój' skrajne wartości
            alpha = float(plan.params.get("alpha", 0.5))
            ssim = alpha * ssim + (1.0 - alpha) * np.clip(ssim, 0.85, 1.0)
            diff *= (0.9 + 0.1 * alpha)

    M.layers["ssim"] = ssim
    M.layers["diff"] = diff


# =========================================================
# 4) Mozaika↦AST (Ψ) – prosta reguła naprawcza
# =========================================================

@dataclass
class AstPatch:
    kind: str
    params: Dict[str, Any]
    region_expr: str


def psi_lift(M: Mosaic, ssim_thr: float = 0.78, frac_thr: float = 0.2) -> List[AstPatch]:
    """
    Jeśli odsetek kafelków z ssim < ssim_thr przekracza frac_thr – zaproponuj naprawę.
    """
    s = M.layers["ssim"]
    low_idx = [i for i, v in enumerate(s) if v < ssim_thr]
    frac = len(low_idx) / max(1, len(s))
    patches: List[AstPatch] = []
    if frac > frac_thr:
        patches.append(AstPatch(kind="Repair.LocalContrast",
                                params={"limit": 0.25},
                                region_expr=f"ssim<{ssim_thr:.2f}"))
    return patches


# =========================================================
# 5) Pseudometryki d_AST, d_M, d_Φ
# =========================================================

def d_ast(nodes: Dict[int, AstNode]) -> float:
    """Prosta 'energia' drzewa: liczba krawędzi + karne wagi za głębokie ścieżki."""
    E = sum(len(n.children) for n in nodes.values())
    depth_pen = sum(n.depth**1.2 for n in nodes.values())
    return float(E + 0.02 * depth_pen)


def d_mosaic(M: Mosaic) -> float:
    """Miara 'chropowatości' mozaiki: wariancja edge + penalty za rozrzut SSIM."""
    edge = M.layers["edge"]
    ssim = M.layers["ssim"]
    return float(np.var(edge) + 0.5 * np.var(ssim))


def d_phi(plans: List[NodePlan], M: Mosaic) -> float:
    """Koszt dopasowania Φ: kara za działanie 'nie tam gdzie trzeba' (heurystyka)."""
    edge = M.layers["edge"]
    cost = 0.0
    for p in plans:
        if p.action == "blur":
            # blur poza krawędziami – źle
            cost += sum(1.0 - edge[i] for i in p.tile_ids) * 0.1
        if p.action == "denoise":
            # denoise na krawędziach – źle
            cost += sum(edge[i] for i in p.tile_ids) * 0.1
    return float(cost)


# =========================================================
# 6) Wizualizacja: kompas 3D AST + heatmap mozaiki
# =========================================================

def plot_ast_compass(nodes: Dict[int, AstNode], ax=None):
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    # rysuj segmenty rodzic→dziecko
    for n in nodes.values():
        x0, y0, z0 = n.pos3d
        for cid in n.children:
            x1, y1, z1 = nodes[cid].pos3d
            ax.plot([x0, x1], [y0, y1], [z0, z1], linewidth=1.2)
    # punkty i etykiety
    for n in nodes.values():
        x, y, z = n.pos3d
        ax.scatter([x], [y], [z], s=20)
        if n.depth <= 4:  # by nie spamować
            ax.text(x, y, z + 0.25, n.label, fontsize=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (depth)")
    ax.set_title("AST → Kompas 3D")


def plot_mosaic_layer(M: Mosaic, layer: str = "ssim", ax=None, vmin=None, vmax=None):
    if ax is None:
        ax = plt.gca()
    rows, cols = M.shape
    data = M.layers[layer].reshape(rows, cols)
    im = ax.imshow(data, origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(f"Mosaic layer: {layer}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# =========================================================
# 7) Przykładowe źródło (AST), main() – demo end-to-end
# =========================================================

EXAMPLE_SRC = r"""
def pipeline(img):
    # syntetyczny szkic – nie wykonujemy, tylko badamy AST
    R = (120, 80, 200, 160)  # ROI
    if True:
        x = 1
    y = gaussian_blur(img, sigma=2.0)
    z = denoise_nlm(y, strength=0.3)
    return blend(img, z, alpha=0.5)
"""

def main():
    print(">> Buduję AST z przykładowego źródła…")
    nodes = build_ast_graph(EXAMPLE_SRC)
    print(f"   Węzłów: {len(nodes)} | d_AST={d_ast(nodes):.3f}")

    print(">> Tworzę mozaikę…")
    M = build_grid_mosaic(rows=12, cols=12, seed=13)
    print(f"   Tiles: {len(M.tiles)} | d_M={d_mosaic(M):.3f}")

    print(">> Projekcja Φ (AST→Mosaic)…")
    thr_map = {"edge": 0.55}
    plans = phi_project(nodes, M, thr_map)
    cost_phi_pre = d_phi(plans, M)
    phi_execute(M, plans)
    cost_phi_post = d_phi(plans, M)

    # Ψ – lift (M→AST)
    patches = psi_lift(M, ssim_thr=0.80, frac_thr=0.18)

    print("   Plany Φ:")
    for p in plans:
        print(f"     - {p.action:>14s} | tiles={len(p.tile_ids)} | params={p.params}")
    print(f"   d_Φ (przed symulacją): {cost_phi_pre:.3f} | po: {cost_phi_post:.3f}")

    if patches:
        print("   Propozycje Ψ (łatki AST):")
        for pt in patches:
            print(f"     - {pt.kind} @ {pt.region_expr} | params={pt.params}")
    else:
        print("   Ψ: brak koniecznych poprawek (ssim OK).")

    # Wizualizacja: kompas 3D + heatmapa SSIM
    fig = plt.figure(figsize=(11, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.0])
    ax0 = fig.add_subplot(gs[0], projection="3d")
    plot_ast_compass(nodes, ax=ax0)

    ax1 = fig.add_subplot(gs[1])
    plot_mosaic_layer(M, layer="ssim", ax=ax1, vmin=0.6, vmax=1.05)
    ax1.set_xlabel("cols")
    ax1.set_ylabel("rows")
    plt.suptitle("Mozaikowe Drzewo AST – Φ/Ψ demo", y=0.98)
    plt.tight_layout()
    plt.show()

    # Podsumowanie metryk
    print("\n== PODSUMOWANIE ==")
    print(f"d_AST = {d_ast(nodes):.3f} | d_M = {d_mosaic(M):.3f} | d_Φ = {d_phi(plans, M):.3f}")
    ssim = M.layers['ssim']
    print(f"SSIM per-tile: min={ssim.min():.3f}, med={np.median(ssim):.3f}, max={ssim.max():.3f}")


if __name__ == "__main__":
    main()
