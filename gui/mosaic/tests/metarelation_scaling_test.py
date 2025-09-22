# -*- coding: utf-8 -*-
"""
demos/ast_mosaic_meta_evolution.py
----------------------------------
Interaktywny demo-lab:
- dowolny kod Pythona -> AST (parsowanie na żywo z edytora)
- wizualizacja AST (3D) w skali szarości, pozycje: detal → centroidy grup (λ)
- mozaika 10x14 (3D bar chart) w skali szarości + ROI-A/ROI-B
- wybór węzła AST i regionu -> fuzja metastruktury (jedna arytmetyka relacji)
- przerywane linie dla relacji, jaskrawe kolory dla akcentów
- „róża metryczna” (⟨L,S⟩, ⟨Sel,Stab⟩, ⟨Cau,H⟩) jako wynik fuzji

Uruchom:
    python demos/ast_mosaic_meta_evolution.py

Wymaga:
    Python 3.8+, tkinter, numpy, matplotlib
"""

from __future__ import annotations
import ast
import math
import tkinter as tk
from tkinter import ttk, messagebox

from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# ─────────────────────────────────────────────────────────────
# 1) Model danych
# ─────────────────────────────────────────────────────────────

@dataclass
class AstNodeMeta:
    name: str
    kind: str
    pos_det: Tuple[float, float, float]      # pozycja detalu (λ=0)
    group: str                                # nazwa grupy (do centroidu przy λ→1)
    meta: Tuple[float, float, float, float, float, float]  # (L,S,Sel,Stab,Cau,H) ∈ [0,1]^6

    @property
    def energy(self) -> float:
        return float(np.linalg.norm(self.meta))

    @property
    def entropy(self) -> float:
        return float(self.meta[-1])


@dataclass
class Mosaic:
    rows: int
    cols: int
    edge: np.ndarray  # (rows*cols,) ∈ [0,1]
    roiA: np.ndarray  # (rows*cols,) 0/1
    roiB: np.ndarray  # (rows*cols,) 0/1


# ─────────────────────────────────────────────────────────────
# 2) Mozaika demo + regiony + cechy
# ─────────────────────────────────────────────────────────────

def build_demo_mosaic(rows=10, cols=14) -> Mosaic:
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:rows, 0:cols]
    diag = 1.0 - np.abs(xx - yy) / max(rows, cols)
    edge = np.clip(0.45 + 0.5 * diag + 0.06 * rng.standard_normal((rows, cols)), 0, 1).reshape(-1)

    roiA = np.zeros(rows * cols)
    roiB = np.zeros(rows * cols)
    # ROI A – prostokąt po lewej
    for r in range(int(0.25 * rows), int(0.65 * rows)):
        for c in range(int(0.12 * cols), int(0.42 * cols)):
            roiA[r * cols + c] = 1.0
    # ROI B – prostokąt po prawej
    for r in range(int(0.35 * rows), int(0.85 * rows)):
        for c in range(int(0.55 * cols), int(0.90 * cols)):
            roiB[r * cols + c] = 1.0
    return Mosaic(rows, cols, edge, roiA, roiB)


def region_indices(M: Mosaic, key: str) -> Set[int]:
    if key == "ROI-A": return {i for i, v in enumerate(M.roiA) if v > 0.5}
    if key == "ROI-B": return {i for i, v in enumerate(M.roiB) if v > 0.5}
    return set(range(M.rows * M.cols))  # ALL


def region_centroid(M: Mosaic, ids: Set[int]) -> Tuple[float, float, float]:
    if not ids: return (M.cols * 0.5, M.rows * 0.5, 0.0)
    cols = np.array([i % M.cols for i in ids], dtype=float)
    rows = np.array([i // M.cols for i in ids], dtype=float)
    z = np.array([2.0 * M.edge[i] + 0.1 for i in ids], dtype=float)
    return float(cols.mean()), float(rows.mean()), float(z.mean())


def region_feats(M: Mosaic, ids: Set[int]) -> np.ndarray:
    """
    ψ(region) → 6D, kompatybilne z (L,S,Sel,Stab,Cau,H)
    """
    if not ids: return np.zeros(6, dtype=float)
    ed = np.array([M.edge[i] for i in ids])
    fL   = float(1.0 - np.mean(ed))               # lokalność ↑, gdy gładko
    fS   = float(0.5 + 0.5 * np.std(ed))          # skala ↑, gdy zmienność
    fSel = float(np.mean(ed > 0.6))               # selektywność = udział krawędzi
    fSt  = float(1.0 - np.std(ed))                # stabilność = 1 - wariancja
    fC   = float(min(1.0, 0.3 + 0.7 * np.mean(ed)))  # kauzalność ~ „siła” regionu
    fH   = float(0.4 + 0.5 * np.std(ed))          # entropia ~ niepewność
    return np.array([fL, fS, fSel, fSt, fC, fH], dtype=float)


def fuse_meta(node_meta: np.ndarray, reg_meta: np.ndarray, lam: float, beta: float = 1.0) -> np.ndarray:
    """
    Jedna arytmetyka jednej relacji:
    m_fused(λ) = (1−λ)·m_node + λ·(β·ψ(region) ⊙ align)  ;  align = 1 (tu)
    """
    align = np.ones_like(node_meta)
    return (1.0 - lam) * node_meta + lam * (beta * reg_meta * align)


# ─────────────────────────────────────────────────────────────
# 3) AST: parsowanie, pozycjonowanie, metahurystyki
# ─────────────────────────────────────────────────────────────

def _node_label(n: ast.AST) -> str:
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"def {n.name}"
    if isinstance(n, ast.ClassDef):
        return f"class {n.name}"
    if isinstance(n, ast.Assign):
        # first simple target
        try:
            t = n.targets[0]
            if isinstance(t, ast.Name): return f"{t.id} = …"
        except Exception:
            pass
        return "assign"
    if isinstance(n, ast.Name):
        return n.id
    return type(n).__name__


def _group_of(n: ast.AST) -> str:
    """
    Grupy do supergrafu:
      - 'G:def:<fn>' dla węzłów wewnątrz funkcji,
      - 'G:class:<cls>' dla węzłów wewnątrz klasy,
      - 'G:top' dla top-level,
      - dodatkowo kubełkowanie po głębokości: sufiks '/Dk'.
    """
    # śledzimy kontener (najbliższy FunctionDef/ClassDef)
    parent_fn = getattr(n, "_parent_fn", None)
    parent_cls = getattr(n, "_parent_cls", None)
    if parent_fn: base = f"G:def:{parent_fn}"
    elif parent_cls: base = f"G:class:{parent_cls}"
    else: base = "G:top"
    d = getattr(n, "_depth", 0)
    return f"{base}/D{d//2}"  # kubełki co 2 poziomy


def _attach_parents_and_depths(tree: ast.AST):
    """
    Wzbogacamy nody o: _parent, _depth, _parent_fn, _parent_cls
    """
    def walk(n, parent=None, depth=0, fn=None, cls=None):
        setattr(n, "_parent", parent)
        setattr(n, "_depth", depth)
        setattr(n, "_parent_fn", fn)
        setattr(n, "_parent_cls", cls)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = n.name
        if isinstance(n, ast.ClassDef):
            cls = n.name
        for _f, v in ast.iter_fields(n):
            if isinstance(v, ast.AST):
                walk(v, n, depth + 1, fn, cls)
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, ast.AST):
                        walk(it, n, depth + 1, fn, cls)
    walk(tree)


def _coords_for_tree(tree: ast.AST) -> Dict[ast.AST, Tuple[float, float, float]]:
    """
    Ustalamy pozycje 'detalu' (λ=0): x=indeks na poziomie, y=typ (bucket per type),
    z=głębokość (poziom).
    """
    # poziomy
    per_level: Dict[int, List[ast.AST]] = {}
    type_bucket: Dict[str, int] = {}
    def walk(n):
        d = getattr(n, "_depth", 0)
        per_level.setdefault(d, []).append(n)
        for ch in ast.iter_child_nodes(n):
            walk(ch)
    walk(tree)

    # bucketowanie typów
    b = 0
    for n in ast.walk(tree):
        t = type(n).__name__
        if t not in type_bucket:
            type_bucket[t] = b
            b += 1

    # indeks na poziomie
    order_on_level: Dict[ast.AST, int] = {}
    for d, nds in per_level.items():
        for i, n in enumerate(nds):
            order_on_level[n] = i

    coords: Dict[ast.AST, Tuple[float, float, float]] = {}
    for n in ast.walk(tree):
        x = 2.0 * order_on_level.get(n, 0)
        y = 2.0 * type_bucket[type(n).__name__]
        z = 2.0 * getattr(n, "_depth", 0)
        coords[n] = (x, y, z)
    return coords


def _meta_for_node(n: ast.AST) -> Tuple[float, float, float, float, float, float]:
    """
    Heurystyki meta (L,S,Sel,Stab,Cau,H) ∈ [0,1] bazujące na typie/roli.
    Tu prosto i deterministycznie: nadaje charakter.
    """
    rng = np.random.default_rng(abs(hash((type(n).__name__, getattr(n, "_depth", 0)))) % (2**32))
    L, S, Sel, Stab, Cau, H = rng.uniform(0.30, 0.85, size=6)
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        Stab = 0.85; H = 0.35; Sel = 0.55
    if isinstance(n, ast.If):
        Sel = 0.80; H = 0.55
    if isinstance(n, ast.For):
        S = 0.70; Cau = 0.60
    if isinstance(n, ast.Call):
        Sel = 0.65; Cau = 0.55
    if isinstance(n, ast.Assign):
        L = 0.55; Stab = 0.70
    return float(L), float(S), float(Sel), float(Stab), float(Cau), float(H)


def ast_nodes_from_code(code: str) -> List[AstNodeMeta]:
    """
    Parsuje kod -> AST -> lista węzłów z pozycją detalu, grupą i metą.
    """
    tree = ast.parse(code)
    _attach_parents_and_depths(tree)
    coords = _coords_for_tree(tree)

    nodes: List[AstNodeMeta] = []
    for n in ast.walk(tree):
        # wybierz tylko 'istotne' nody (w celu listy do wyboru)
        if isinstance(n, (ast.Module, ast.Load, ast.Store, ast.Del)):
            continue
        name = _node_label(n)
        kind = type(n).__name__
        group = _group_of(n)
        nodes.append(
            AstNodeMeta(
                name=name,
                kind=kind,
                pos_det=coords[n],
                group=group,
                meta=_meta_for_node(n)
            )
        )
    return nodes


# ─────────────────────────────────────────────────────────────
# 4) Geometria meta: λ → pozycje, centroidy grup
# ─────────────────────────────────────────────────────────────

def group_centroids(nodes: List[AstNodeMeta]) -> Dict[str, Tuple[float, float, float]]:
    by_g: Dict[str, List[Tuple[float, float, float]]] = {}
    for n in nodes:
        by_g.setdefault(n.group, []).append(n.pos_det)
    cents: Dict[str, Tuple[float, float, float]] = {}
    for g, pts in by_g.items():
        xs, ys, zs = list(zip(*pts))
        cents[g] = (float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs)))
    return cents


def coords_for_lambda(nodes: List[AstNodeMeta], lam: float) -> Dict[str, Tuple[float, float, float]]:
    cents = group_centroids(nodes)
    pos: Dict[str, Tuple[float, float, float]] = {}
    for n in nodes:
        p = n.pos_det; q = cents[n.group]
        pos[n.name] = (p[0] + (q[0] - p[0]) * lam,
                       p[1] + (q[1] - p[1]) * lam,
                       p[2] + (q[2] - p[2]) * lam)
    return pos


# ─────────────────────────────────────────────────────────────
# 5) Rysowanie: AST, Mozaika, Fuzja
# ─────────────────────────────────────────────────────────────

def draw_ast(ax, nodes: List[AstNodeMeta], pos: Dict[str, Tuple[float, float, float]]):
    cmap = plt.get_cmap("Greys")
    for n in nodes:
        x, y, z = pos[n.name]
        h = 0.9 + 1.9 * (n.energy / np.sqrt(6))
        c = cmap(0.35 + 0.55 * n.entropy)  # szarość zależna od H
        ax.plot([x, x], [y, y], [z, z + h], color=c, lw=2.0, alpha=0.9)
        ax.scatter([x], [y], [z + h], s=26, c=[c], edgecolors="black", depthshade=True)
        ax.text(x, y, z + h + 0.35, n.name, fontsize=7, ha="center", color="black")


def draw_mosaic(ax, M: Mosaic):
    xs, ys, zs, dx, dy, dz, cols = [], [], [], [], [], [], []
    for r in range(M.rows):
        for c in range(M.cols):
            i = r * M.cols + c
            h = 2.0 * M.edge[i] + 0.1
            xs.append(c); ys.append(r); zs.append(0.0)
            dx.append(0.85); dy.append(0.85); dz.append(h)
            g = M.edge[i]
            cols.append((g, g, g, 0.92))  # skala szarości
    ax.bar3d(xs, ys, zs, dx, dy, dz, color=cols, linewidth=0.1, shade=True)
    ax.set_xlabel("cols"); ax.set_ylabel("rows"); ax.set_zlabel("edge→height")
    ax.view_init(elev=25, azim=-58)


def draw_region_frame(ax, M: Mosaic, key: str, z=2.45, color="lime"):
    ids = region_indices(M, key)
    if not ids: return
    rr = np.array([i // M.cols for i in ids]); cc = np.array([i % M.cols for i in ids])
    rmin, rmax = rr.min(), rr.max(); cmin, cmax = cc.min(), cc.max()
    ax.plot([cmin, cmax, cmax, cmin, cmin],
            [rmin, rmin, rmax, rmax, rmin],
            [z, z, z, z, z], color=color, lw=1.6, linestyle="--")


def draw_fusion(ax_ast, ax_mos,
                node: AstNodeMeta, pos: Dict[str, Tuple[float, float, float]],
                M: Mosaic, reg_key: str, lam: float, fused: np.ndarray):
    # centroid regionu (punkt 0)
    ids = region_indices(M, reg_key)
    cx, cy, cz = region_centroid(M, ids)
    colors_reg = {"ROI-A": "lime", "ROI-B": "magenta", "ALL": "orange"}
    reg_color = colors_reg.get(reg_key, "orange")
    ax_mos.scatter([cx], [cy], [cz + 0.05], s=52, c=reg_color,
                   edgecolors="black", depthshade=True, zorder=10)

    # wiązka node → centroid regionu (linia przerywana)
    x0, y0, z0 = pos[node.name]
    ax_ast.plot([x0, cx], [y0, cy], [z0, cz],
                linestyle="--", color=reg_color, lw=1.6, alpha=0.95)

    # róża metryczna (3 pary): ⟨L,S⟩, ⟨Sel,Stab⟩, ⟨Cau,H⟩
    base = np.array([x0, y0, z0 + 0.28])
    pairs = [(0, 1), (2, 3), (4, 5)]
    rose_colors = ["cyan", "orange", "yellow"]
    labels = ["⟨L,S⟩", "⟨Sel,Stab⟩", "⟨Cau,H⟩"]
    scale = 1.2
    for k, (i, j) in enumerate(pairs):
        val = float(0.5 * (fused[i] + fused[j]))
        vec = np.array([(1 if k == 0 else 0),
                        (1 if k == 1 else 0),
                        0.9])  # orty + lekko w górę
        tip = base + scale * val * vec
        ax_ast.plot([base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]],
                    linestyle="--", color=rose_colors[k], lw=2.0)
        ax_ast.text(tip[0], tip[1], tip[2] + 0.08, labels[k],
                    fontsize=7, color=rose_colors[k])

    # formuła
    ax_ast.text(x0, y0, z0 - 0.6,
                r"$m_{\mathrm{fused}}(\lambda)=(1-\lambda)\,m_{\mathrm{node}}+\lambda\,\beta\,\psi(\mathrm{region})$",
                fontsize=7, ha="center", color="black")


def draw_infographic(ax):
    ax.axis("off")
    ax.text(0.52, 2.25, "Infografika relacji", fontsize=11, weight="bold")
    ax.text(0.52, 2.00, "■  Szarości: struktura bazowa (AST, Mozaika)", fontsize=9, color="black")
    ax.text(0.52, 1.75, "◆  Centroid ROI: lime/magenta/orange", fontsize=9, color="lime")
    ax.text(0.52, 1.50, "— —    przerywane: relacje node ↔ ROI & róża metryczna", fontsize=9, color="magenta")
    ax.text(0.52, 1.25, "✦  Róża: pary ⟨L,S⟩, ⟨Sel,Stab⟩, ⟨Cau,H⟩ po FUZJI", fontsize=9, color="cyan")
    ax.text(0.52, 1.00, "Jedna arytmetyka relacji:", fontsize=10, weight="bold")
    ax.text(0.52, 0.75, "m_fused(λ) = (1−λ)·m_node + λ·β·ψ(region)", fontsize=9)
    ax.text(0.52, 0.50, "λ skaluje meta–warstwę (detal → supergraf grup) bez zmiany mechaniki fuzji", fontsize=9)


# ─────────────────────────────────────────────────────────────
# 6) GUI (Tkinter + Matplotlib)
# ─────────────────────────────────────────────────────────────

DEFAULT_SNIPPET = """\
# proste demo, edytuj i naciśnij [Render]
def f(x):
    y = x
    if y > 0:
        y = y - 1
    return y + 1

class K:
    def __init__(self, a):
        self.a = a

def g(n):
    s = 0
    for i in range(n):
        s += i
    return s

z = f(3) + g(4)
"""

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AST ⟷ Mozaika — meta-evolution (λ) • one-relation arithmetic")
        self.geometry("1380x900")

        self.M = build_demo_mosaic()
        self.nodes: List[AstNodeMeta] = []
        self.pos: Dict[str, Tuple[float, float, float]] = {}
        self.node_by_label: Dict[str, AstNodeMeta] = {}

        # ——— układ: lewy (edytor + sterowanie), prawy (figura)
        left = ttk.Frame(self); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=8, pady=8)
        right = ttk.Frame(self); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # edytor
        ttk.Label(left, text="Kod Pythona (Ctrl+Enter = Render):").pack(anchor="w")
        self.txt = tk.Text(left, width=64, height=32, wrap="none", font=("Consolas", 10))
        self.txt.pack(fill=tk.BOTH, expand=True)
        self.txt.insert("1.0", DEFAULT_SNIPPET)
        self.txt.bind("<Control-Return>", lambda e: self.render())

        # sterowanie
        ctrl = ttk.LabelFrame(left, text="Sterowanie")
        ctrl.pack(fill=tk.X, pady=6)

        # λ i β
        row0 = ttk.Frame(ctrl); row0.pack(fill=tk.X, pady=2)
        ttk.Label(row0, text="λ (meta skala)").pack(side=tk.LEFT)
        self.s_lambda = tk.Scale(row0, from_=0.0, to=1.0, resolution=0.02, orient=tk.HORIZONTAL, length=220,
                                 command=lambda _v: self.repaint())
        self.s_lambda.set(0.0); self.s_lambda.pack(side=tk.LEFT, padx=6)

        ttk.Label(row0, text="β (wzmocnienie regionu)").pack(side=tk.LEFT, padx=(10, 2))
        self.s_beta = tk.Scale(row0, from_=0.2, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, length=140,
                               command=lambda _v: self.repaint())
        self.s_beta.set(1.0); self.s_beta.pack(side=tk.LEFT)

        # region
        row1 = ttk.Frame(ctrl); row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Region").pack(side=tk.LEFT)
        self.cmb_region = ttk.Combobox(row1, values=["ROI-A", "ROI-B", "ALL"], width=10, state="readonly")
        self.cmb_region.set("ROI-A"); self.cmb_region.pack(side=tk.LEFT, padx=6)
        self.cmb_region.bind("<<ComboboxSelected>>", lambda _e: self.repaint())

        # węzeł AST
        row2 = ttk.Frame(ctrl); row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Węzeł AST").pack(side=tk.LEFT)
        self.cmb_node = ttk.Combobox(row2, values=[], width=32, state="readonly")
        self.cmb_node.pack(side=tk.LEFT, padx=6)
        self.cmb_node.bind("<<ComboboxSelected>>", lambda _e: self.repaint())

        # przyciski
        row3 = ttk.Frame(ctrl); row3.pack(fill=tk.X, pady=4)
        ttk.Button(row3, text="Render", command=self.render).pack(side=tk.LEFT, padx=2)
        ttk.Button(row3, text="Reset widoków", command=self.reset_views).pack(side=tk.LEFT, padx=6)
        ttk.Button(row3, text="Aa+", command=lambda: self._font_step(+1)).pack(side=tk.RIGHT, padx=2)
        ttk.Button(row3, text="Aa−", command=lambda: self._font_step(-1)).pack(side=tk.RIGHT, padx=2)

        # figury
        self.fig = plt.Figure(figsize=(9.6, 6.8))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[12, 1], width_ratios=[1, 1], hspace=0.25, wspace=0.25)
        self.ax_ast = self.fig.add_subplot(gs[0, 0], projection="3d")
        self.ax_mos = self.fig.add_subplot(gs[0, 1], projection="3d")
        self.ax_inf = self.fig.add_subplot(gs[1, :])

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()

        # pierwszy render
        self.render()

    def _font_step(self, delta: int):
        try:
            f = self.txt["font"]
            fam, size = f.split()[0], int(f.split()[1])
        except Exception:
            fam, size = "Consolas", 10
        size = max(6, size + delta)
        self.txt.configure(font=(fam, size))

    # ——— logika
    def render(self):
        code = self.txt.get("1.0", "end-1c")
        try:
            self.nodes = ast_nodes_from_code(code)
        except SyntaxError as e:
            messagebox.showerror("Błąd składni", str(e))
            return
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się sparsować AST: {e}")
            return

        # odśwież listę węzłów do wyboru
        labels = [f"{n.name}  ·  [{n.kind}]" for n in self.nodes]
        self.node_by_label = {lab: n for lab, n in zip(labels, self.nodes)}
        self.cmb_node["values"] = labels
        if labels:
            cur = self.cmb_node.get()
            if cur not in labels:
                # heurystyka: wybierz „centralny” węzeł (największa energia)
                pick = max(self.nodes, key=lambda nn: nn.energy)
                # znajdź jego etykietę
                for lab, nn in self.node_by_label.items():
                    if nn is pick:
                        self.cmb_node.set(lab); break

        # inicjalne pozycje
        self.repaint()

    def reset_views(self):
        self.ax_ast.view_init(elev=22, azim=-48)
        self.ax_mos.view_init(elev=25, azim=-58)
        self.canvas.draw_idle()

    def repaint(self):
        if not self.nodes:
            return
        lam = float(self.s_lambda.get())
        beta = float(self.s_beta.get())
        reg_key = self.cmb_region.get() or "ROI-A"

        pos = coords_for_lambda(self.nodes, lam)
        self.pos = pos

        # czyszczenie osi
        self.ax_ast.cla(); self.ax_mos.cla(); self.ax_inf.cla()

        # rysuj AST i mozaikę
        draw_ast(self.ax_ast, self.nodes, pos)
        draw_mosaic(self.ax_mos, self.M)
        draw_region_frame(self.ax_mos, self.M, "ROI-A", color="lime")
        draw_region_frame(self.ax_mos, self.M, "ROI-B", color="magenta")
        draw_infographic(self.ax_inf)

        # wybór węzła
        sel_label = self.cmb_node.get()
        node = None
        if sel_label and sel_label in self.node_by_label:
            node = self.node_by_label[sel_label]
        else:
            node = self.nodes[0]

        # fuzja meta
        ids = region_indices(self.M, reg_key)
        reg_vec = region_feats(self.M, ids)
        fused = fuse_meta(np.array(node.meta), reg_vec, lam, beta=beta)
        draw_fusion(self.ax_ast, self.ax_mos, node, pos, self.M, reg_key, lam, fused)

        # limity osi AST
        xs, ys, zs = zip(*pos.values())
        self.ax_ast.set_xlim(min(xs) - 1, max(xs) + 1)
        self.ax_ast.set_ylim(min(ys) - 1, max(ys) + 1)
        self.ax_ast.set_zlim(min(zs) - 1, max(zs) + 3.5)
        self.ax_ast.set_title(f"AST — λ={lam:.2f}  ·  node={node.name} [{node.kind}]  ·  reg={reg_key}  ·  β={beta:.1f}")

        # odśwież
        self.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────
# 7) MAIN
# ─────────────────────────────────────────────────────────────

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
