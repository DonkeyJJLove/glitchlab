# -*- coding: utf-8 -*-
"""
demos/ast_mosaic_coupled.py
---------------------------
AST ⇄ Mozaika: sprzężenie dwukierunkowe + interaktywność.

• Edytuj kod po lewej, [Ctrl+Enter] lub [Render] → AST i Mozaika się przeliczają.
• Klik na mozaice:
    - LPM: ustaw ROI-A na klikniętej komórce,
    - PPM: ustaw ROI-B.
• Klik na węźle AST: wybór węzła (do fuzji; opcjonalnie wpływa na mozaikę przez η).
• Suwaki:
    λ – meta skala (detal → centroidy grup),
    β – siła regionu (fuzja),
    γ – sprzężenie AST→Mozaika (ile mozaika pochodzi z AST),
    Δ – feedback Mozaika→AST (miękka aktualizacja meta-wektorów),
    η – wzmacnianie wkładu wybranego węzła w raster AST→Mozaika.

Wymagania: Python 3.8+, tkinter, numpy, matplotlib
Uruchom:   python demos/ast_mosaic_coupled.py
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
# 1) Struktury danych
# ─────────────────────────────────────────────────────────────

@dataclass
class AstNodeMeta:
    name: str
    kind: str
    pos_det: Tuple[float, float, float]                      # pozycja przy λ=0
    group: str                                               # grupa dla centroidu przy λ→1
    meta0: Tuple[float, float, float, float, float, float]   # bazowy meta-wektor (L,S,Sel,Stab,Cau,H)
    meta: np.ndarray                                         # aktualny meta-wektor (z feedbackiem)

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
    base_edge: np.ndarray    # stała „tekstura” (rows*cols,)
    edge: np.ndarray         # aktualna mapa (po sprzężeniu z AST)
    roiA: np.ndarray         # maska ROI-A (rows*cols,)
    roiB: np.ndarray         # maska ROI-B (rows*cols,)


# ─────────────────────────────────────────────────────────────
# 2) AST: parsowanie, grupy, meta
# ─────────────────────────────────────────────────────────────

def _node_label(n: ast.AST) -> str:
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"def {n.name}"
    if isinstance(n, ast.ClassDef):
        return f"class {n.name}"
    if isinstance(n, ast.Assign):
        try:
            t = n.targets[0]
            if isinstance(t, ast.Name): return f"{t.id} = …"
        except Exception:
            pass
        return "assign"
    if isinstance(n, ast.Name): return n.id
    return type(n).__name__


def _attach_parents_and_depths(tree: ast.AST):
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


def _group_of(n: ast.AST) -> str:
    if getattr(n, "_parent_fn", None): base = f"G:def:{getattr(n, '_parent_fn')}"
    elif getattr(n, "_parent_cls", None): base = f"G:class:{getattr(n, '_parent_cls')}"
    else: base = "G:top"
    d = getattr(n, "_depth", 0)
    return f"{base}/D{d//2}"


def _coords_for_tree(tree: ast.AST) -> Dict[ast.AST, Tuple[float, float, float]]:
    per_level: Dict[int, List[ast.AST]] = {}
    type_bucket: Dict[str, int] = {}
    def walk(n):
        d = getattr(n, "_depth", 0)
        per_level.setdefault(d, []).append(n)
        for ch in ast.iter_child_nodes(n):
            walk(ch)
    walk(tree)

    b = 0
    for n in ast.walk(tree):
        t = type(n).__name__
        if t not in type_bucket:
            type_bucket[t] = b; b += 1

    order_on_level: Dict[ast.AST, int] = {}
    for d, nds in per_level.items():
        for i, n in enumerate(nds): order_on_level[n] = i

    coords: Dict[ast.AST, Tuple[float, float, float]] = {}
    for n in ast.walk(tree):
        x = 2.0 * order_on_level.get(n, 0)
        y = 2.0 * type_bucket[type(n).__name__]
        z = 2.0 * getattr(n, "_depth", 0)
        coords[n] = (x, y, z)
    return coords


def _meta_for_node(n: ast.AST) -> Tuple[float, float, float, float, float, float]:
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
    tree = ast.parse(code)
    _attach_parents_and_depths(tree)
    coords = _coords_for_tree(tree)
    nodes: List[AstNodeMeta] = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.Load, ast.Store, ast.Del)):
            continue
        name = _node_label(n)
        kind = type(n).__name__
        group = _group_of(n)
        m0 = _meta_for_node(n)
        nodes.append(
            AstNodeMeta(
                name=name,
                kind=kind,
                pos_det=coords[n],
                group=group,
                meta0=m0,
                meta=np.array(m0, dtype=float)
            )
        )
    return nodes


# ─────────────────────────────────────────────────────────────
# 3) Geometria meta: λ → pozycje
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
# 4) Mozaika: konstrukcja z AST + sprzężenie
# ─────────────────────────────────────────────────────────────

def build_base_mosaic(rows=10, cols=14) -> Mosaic:
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:rows, 0:cols]
    diag = 1.0 - np.abs(xx - yy) / max(rows, cols)
    base = np.clip(0.45 + 0.5 * diag + 0.06 * rng.standard_normal((rows, cols)), 0, 1)
    base = base.reshape(-1)

    roiA = np.zeros(rows * cols)
    roiB = np.zeros(rows * cols)
    return Mosaic(rows, cols, base_edge=base.copy(), edge=base.copy(), roiA=roiA, roiB=roiB)


def _norm01(a: np.ndarray) -> np.ndarray:
    if a.size == 0: return a
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo < 1e-12: return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _sobel_mag(img2d: np.ndarray) -> np.ndarray:
    Kx = np.array([[+1, 0, -1],
                   [+2, 0, -2],
                   [+1, 0, -1]], dtype=float)
    Ky = np.array([[+1, +2, +1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=float)
    pad = np.pad(img2d, 1, mode="edge")
    gx = np.zeros_like(img2d); gy = np.zeros_like(img2d)
    for r in range(img2d.shape[0]):
        for c in range(img2d.shape[1]):
            roi = pad[r:r+3, c:c+3]
            gx[r, c] = float(np.sum(roi * Kx))
            gy[r, c] = float(np.sum(roi * Ky))
    return np.sqrt(gx*gx + gy*gy)


def mosaic_from_ast(nodes: List[AstNodeMeta], rows: int, cols: int,
                    lam: float, gamma: float,
                    selected: Optional[str] = None, eta: float = 0.0) -> np.ndarray:
    """
    Buduje mozaikę z AST (pozycje wg λ, wagi = energia/entropia).
    gamma (γ) ∈ [0,1] skaluje udział komponenty AST (reszta z base_edge).
    selected + eta: opcjonalny „boost” wybranego węzła (η≥0) w akumulacji ciepła.
    Zwraca "edge" (rows*cols,) po sprzężeniu z AST (przed blendem w App).
    """
    if not nodes:
        return np.zeros(rows * cols)

    # pozycje 2D (x,y) zależne od λ
    cents = group_centroids(nodes)
    def _pos_lam(n):
        p = np.array([n.pos_det[0], n.pos_det[1]], dtype=float)
        q = np.array([cents[n.group][0], cents[n.group][1]], dtype=float)
        return p + (q - p) * lam
    pts = np.array([_pos_lam(n) for n in nodes], dtype=float)

    x0, x1 = float(np.min(pts[:,0])), float(np.max(pts[:,0]))
    y0, y1 = float(np.min(pts[:,1])), float(np.max(pts[:,1]))
    if x1 - x0 < 1e-9: x1 = x0 + 1.0
    if y1 - y0 < 1e-9: y1 = y0 + 1.0
    nx = ((pts[:,0] - x0) / (x1 - x0) * (cols - 1)).clip(0, cols - 1)
    ny = ((pts[:,1] - y0) / (y1 - y0) * (rows - 1)).clip(0, rows - 1)

    heat = np.zeros((rows, cols), dtype=float)
    for k, n in enumerate(nodes):
        r = int(round(float(ny[k]))); c = int(round(float(nx[k])))
        w = 0.6 * n.energy + 0.4 * n.entropy
        if selected and n.name == selected:
            w *= (1.0 + max(0.0, float(eta)))  # boost wybranego
        heat[r, c] += float(w)

    heat = _norm01(heat)
    edges = _sobel_mag(heat)
    edges = _norm01(edges)
    return (edges * gamma).reshape(-1)


def region_indices(M: Mosaic, key: str) -> Set[int]:
    if key == "ROI-A": return {i for i, v in enumerate(M.roiA) if v > 0.5}
    if key == "ROI-B": return {i for i, v in enumerate(M.roiB) if v > 0.5}
    return set(range(M.rows * M.cols))


def region_centroid(M: Mosaic, ids: Set[int]) -> Tuple[float, float, float]:
    if not ids: return (M.cols * 0.5, M.rows * 0.5, 0.0)
    cols = np.array([i % M.cols for i in ids], dtype=float)
    rows = np.array([i // M.cols for i in ids], dtype=float)
    z = np.array([2.0 * M.edge[i] + 0.1 for i in ids], dtype=float)
    return float(cols.mean()), float(rows.mean()), float(z.mean())


def region_feats(M: Mosaic, ids: Set[int]) -> np.ndarray:
    if not ids: return np.zeros(6, dtype=float)
    ed = np.array([M.edge[i] for i in ids])
    fL   = float(1.0 - np.mean(ed))
    fS   = float(0.5 + 0.5 * np.std(ed))
    fSel = float(np.mean(ed > 0.6))
    fSt  = float(1.0 - np.std(ed))
    fC   = float(min(1.0, 0.3 + 0.7 * np.mean(ed)))
    fH   = float(0.4 + 0.5 * np.std(ed))
    return np.array([fL, fS, fSel, fSt, fC, fH], dtype=float)


def fuse_meta(node_meta: np.ndarray, reg_meta: np.ndarray, lam: float, beta: float) -> np.ndarray:
    align = np.ones_like(node_meta)
    return (1.0 - lam) * node_meta + lam * (beta * reg_meta * align)


# ─────────────────────────────────────────────────────────────
# 5) Rysowanie (szarości dla metawarstwy; relacje jaskrawo, przerywane)
# ─────────────────────────────────────────────────────────────

def draw_ast(ax, nodes: List[AstNodeMeta], pos: Dict[str, Tuple[float, float, float]],
             pick=False):
    cmap = plt.get_cmap("Greys")
    for n in nodes:
        x, y, z = pos[n.name]
        h = 0.9 + 1.9 * (n.energy / np.sqrt(6))
        c = cmap(0.35 + 0.55 * n.entropy)
        ax.plot([x, x], [y, y], [z, z + h], color=c, lw=2.0, alpha=0.9)
        scat = ax.scatter([x], [y], [z + h], s=30, c=[c], edgecolors="black",
                          depthshade=True, picker=pick, pickradius=6)
        scat._glitch_name = n.name  # tag do pickingu
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
            cols.append((g, g, g, 0.92))
    ax.bar3d(xs, ys, zs, dx, dy, dz, color=cols, linewidth=0.1, shade=True)
    ax.set_xlabel("cols"); ax.set_ylabel("rows"); ax.set_zlabel("edge→height")
    ax.view_init(elev=25, azim=-58)


def draw_region_frame(ax, M: Mosaic, key: str, color="lime"):
    ids = region_indices(M, key)
    if not ids: return
    rr = np.array([i // M.cols for i in ids]); cc = np.array([i % M.cols for i in ids])
    rmin, rmax = rr.min(), rr.max(); cmin, cmax = cc.min(), cc.max()
    z = 2.45
    ax.plot([cmin, cmax, cmax, cmin, cmin],
            [rmin, rmin, rmax, rmax, rmin],
            [z, z, z, z, z], color=color, lw=1.6, linestyle="--")


def draw_fusion(ax_ast, ax_mos,
                node: AstNodeMeta, pos: Dict[str, Tuple[float, float, float]],
                M: Mosaic, reg_key: str, lam: float, fused: np.ndarray):
    ids = region_indices(M, reg_key)
    cx, cy, cz = region_centroid(M, ids)
    colors_reg = {"ROI-A": "lime", "ROI-B": "magenta", "ALL": "orange"}
    reg_color = colors_reg.get(reg_key, "orange")
    ax_mos.scatter([cx], [cy], [cz + 0.05], s=60, c=reg_color,
        edgecolors="black", depthshade=True, zorder=10)

    x0, y0, z0 = pos[node.name]
    ax_ast.plot([x0, cx], [y0, cy], [z0, cz],
        linestyle="--", color=reg_color, lw=1.8, alpha=0.95)

    base = np.array([x0, y0, z0 + 0.28])
    pairs = [(0, 1), (2, 3), (4, 5)]
    rose_colors = ["cyan", "orange", "yellow"]
    labels = ["⟨L,S⟩", "⟨Sel,Stab⟩", "⟨Cau,H⟩"]
    scale = 1.2
    for k, (i, j) in enumerate(pairs):
        val = float(0.5 * (fused[i] + fused[j]))
        vec = np.array([(1 if k == 0 else 0),
                        (1 if k == 1 else 0),
                        0.9])
        tip = base + scale * val * vec
        ax_ast.plot([base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]],
                    linestyle="--", color=rose_colors[k], lw=2.0)
        ax_ast.text(tip[0], tip[1], tip[2] + 0.08, labels[k],
                    fontsize=7, color=rose_colors[k])

    ax_ast.text(x0, y0, z0 - 0.6,
        r"$m_{\mathrm{fused}}(\lambda)=(1-\lambda)\,m_{\mathrm{node}}+\lambda\,\beta\,\psi(\mathrm{region})$",
        fontsize=7, ha="center", color="black")


def draw_infographic(ax):
    ax.axis("off")
    ax.text(0.02, 0.92, "Infografika sprzężenia", fontsize=11, weight="bold")
    ax.text(0.02, 0.82, "■ Szarości: warstwa meta (AST, Mozaika)", fontsize=9, color="black")
    ax.text(0.02, 0.74, "◆ ROI: lime / magenta / orange", fontsize=9, color="lime")
    ax.text(0.02, 0.66, "— — przerywane: relacje node ↔ ROI i róża metryczna", fontsize=9, color="magenta")
    ax.text(0.02, 0.58, "γ: AST→Mozaika;  Δ: Mozaika→AST (soft feedback);  η: boost wybranego węzła", fontsize=9)
    ax.text(0.02, 0.46, "Jedna arytmetyka relacji:", fontsize=10, weight="bold")
    ax.text(0.05, 0.38, "m_fused(λ) = (1−λ)·m_node + λ·β·ψ(region)", fontsize=9)
    ax.text(0.05, 0.30, "λ: detal → centroidy grup (supergraf)", fontsize=9)


# ─────────────────────────────────────────────────────────────
# 6) GUI (Tkinter + Matplotlib)
# ─────────────────────────────────────────────────────────────

DEFAULT_SNIPPET = """\
# Edytuj kod i naciśnij Ctrl+Enter (lub 'Render')
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
        if i % 2 == 0:
            s += i
    return s

z = f(3) + g(4)
"""

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AST ⇄ Mozaika — sprzężenie meta (λ, β, γ, Δ, η)")
        self.geometry("1440x940")

        # model
        self.nodes: List[AstNodeMeta] = []
        self.pos: Dict[str, Tuple[float, float, float]] = {}
        self.node_by_label: Dict[str, AstNodeMeta] = {}
        self.M = build_base_mosaic()
        self.reg_key = "ROI-A"

        # layout
        left = ttk.Frame(self); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=8, pady=8)
        right = ttk.Frame(self); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # editor
        ttk.Label(left, text="Kod Pythona (Ctrl+Enter = Render):").pack(anchor="w")
        self.txt = tk.Text(left, width=66, height=32, wrap="none", font=("Consolas", 10))
        self.txt.pack(fill=tk.BOTH, expand=True)
        self.txt.insert("1.0", DEFAULT_SNIPPET)
        self.txt.bind("<Control-Return>", lambda e: self.render())

        # controls
        ctrl = ttk.LabelFrame(left, text="Sterowanie")
        ctrl.pack(fill=tk.X, pady=6)

        row0 = ttk.Frame(ctrl); row0.pack(fill=tk.X, pady=2)
        ttk.Label(row0, text="λ").pack(side=tk.LEFT)
        # ZMIANA: λ teraz wywołuje pełny rebuild mozaiki (a nie tylko repaint)
        self.s_lambda = tk.Scale(row0, from_=0.0, to=1.0, resolution=0.02, orient=tk.HORIZONTAL, length=220,
                                 command=lambda _v: self.on_lambda_changed())
        self.s_lambda.set(0.0); self.s_lambda.pack(side=tk.LEFT, padx=4)

        ttk.Label(row0, text="β").pack(side=tk.LEFT)
        self.s_beta = tk.Scale(row0, from_=0.2, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, length=140,
                               command=lambda _v: self.repaint())
        self.s_beta.set(1.0); self.s_beta.pack(side=tk.LEFT, padx=6)

        row1 = ttk.Frame(ctrl); row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="γ (AST→Mozaika)").pack(side=tk.LEFT)
        self.s_gamma = tk.Scale(row1, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=220,
                                command=lambda _v: self.recompute_mosaic_from_ast())
        self.s_gamma.set(0.7); self.s_gamma.pack(side=tk.LEFT, padx=4)

        ttk.Label(row1, text="Δ (Mozaika→AST)").pack(side=tk.LEFT)
        # ZMIANA: Δ po zmianie aktualizuje meta i natychmiast przebudowuje mozaikę
        self.s_delta = tk.Scale(row1, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=140,
                                command=lambda _v: self.apply_feedback_and_recompute())
        self.s_delta.set(0.0); self.s_delta.pack(side=tk.LEFT, padx=6)

        row2 = ttk.Frame(ctrl); row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="η (boost wybranego węzła)").pack(side=tk.LEFT)
        self.s_eta = tk.Scale(row2, from_=0.0, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, length=220,
                              command=lambda _v: self.recompute_mosaic_from_ast())
        self.s_eta.set(0.0); self.s_eta.pack(side=tk.LEFT, padx=4)

        ttk.Label(row2, text="Region").pack(side=tk.LEFT)
        self.cmb_region = ttk.Combobox(row2, values=["ROI-A", "ROI-B", "ALL"], width=10, state="readonly")
        self.cmb_region.set("ROI-A"); self.cmb_region.pack(side=tk.LEFT, padx=6)
        self.cmb_region.bind("<<ComboboxSelected>>", lambda _e: self.on_region_changed())

        row3 = ttk.Frame(ctrl); row3.pack(fill=tk.X, pady=4)
        ttk.Label(row3, text="Węzeł AST").pack(side=tk.LEFT)
        self.cmb_node = ttk.Combobox(row3, values=[], width=34, state="readonly")
        self.cmb_node.pack(side=tk.LEFT, padx=6)
        # ZMIANA: wybór węzła (gdy η>0) też przelicza mozaikę
        self.cmb_node.bind("<<ComboboxSelected>>", lambda _e: self.on_node_changed())

        row4 = ttk.Frame(ctrl); row4.pack(fill=tk.X, pady=4)
        ttk.Button(row4, text="Render", command=self.render).pack(side=tk.LEFT, padx=2)
        ttk.Button(row4, text="Reset widoków", command=self.reset_views).pack(side=tk.LEFT, padx=6)
        ttk.Button(row4, text="Aa+", command=lambda: self._font_step(+1)).pack(side=tk.RIGHT, padx=2)
        ttk.Button(row4, text="Aa−", command=lambda: self._font_step(-1)).pack(side=tk.RIGHT, padx=2)

        # figure
        self.fig = plt.Figure(figsize=(10.4, 7.4))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[12, 1], width_ratios=[1, 1], hspace=0.25, wspace=0.28)
        self.ax_ast = self.fig.add_subplot(gs[0, 0], projection="3d")
        self.ax_mos = self.fig.add_subplot(gs[0, 1], projection="3d")
        self.ax_inf = self.fig.add_subplot(gs[1, :])

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()

        # picking/click handlers
        self.canvas.mpl_connect("pick_event", self.on_pick_ast)
        self.canvas.mpl_connect("button_press_event", self.on_click_any)

        # initial render
        self.render()

    # ------------- helpers -------------
    def _font_step(self, delta: int):
        try:
            fam, size = self.txt["font"].split()[0], int(self.txt["font"].split()[1])
        except Exception:
            fam, size = "Consolas", 10
        self.txt.configure(font=(fam, max(6, size + delta)))

    def reset_views(self):
        self.ax_ast.view_init(elev=22, azim=-48)
        self.ax_mos.view_init(elev=25, azim=-58)
        self.canvas.draw_idle()

    # ------------- core flow -------------
    def render(self):
        code = self.txt.get("1.0", "end-1c")
        try:
            self.nodes = ast_nodes_from_code(code)
        except SyntaxError as e:
            messagebox.showerror("Błąd składni", str(e)); return
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się sparsować AST: {e}"); return

        labels = [f"{n.name}  ·  [{n.kind}]" for n in self.nodes]
        self.node_by_label = {lab: n for lab, n in zip(labels, self.nodes)}
        self.cmb_node["values"] = labels
        if labels:
            cur = self.cmb_node.get()
            if cur not in labels:
                pick = max(self.nodes, key=lambda nn: nn.energy)
                for lab, nn in self.node_by_label.items():
                    if nn is pick: self.cmb_node.set(lab); break

        # po każdej zmianie kodu: odtwórz meta = meta0
        for n in self.nodes:
            n.meta = np.array(n.meta0, dtype=float)

        # pełny rebuild mozaiki (uwzględni λ, γ, η)
        self.recompute_mosaic_from_ast()
        # odmalowanie
        self.repaint()

    def on_lambda_changed(self):
        # λ wpływa na pozycje → musi przebudować mozaikę
        self.recompute_mosaic_from_ast()

    def on_node_changed(self):
        # jeśli η>0, wybór węzła zmienia wkład do mozaiki
        self.recompute_mosaic_from_ast()
        self.repaint()

    def recompute_mosaic_from_ast(self):
        lam = float(self.s_lambda.get())
        gamma = float(self.s_gamma.get())
        eta = float(self.s_eta.get())
        # który węzeł jest wybrany (dla η)
        sel_label = self.cmb_node.get()
        selected_name = None
        if sel_label and sel_label in self.node_by_label:
            selected_name = self.node_by_label[sel_label].name

        ast_comp = mosaic_from_ast(self.nodes, self.M.rows, self.M.cols, lam, gamma,
                                   selected=selected_name, eta=eta)  # (N,)
        a = _norm01(ast_comp.reshape(self.M.rows, self.M.cols))
        b = _norm01(self.M.base_edge.reshape(self.M.rows, self.M.cols))
        # blend z bazą
        edge = (1.0 - gamma) * b + gamma * a
        self.M.edge = edge.reshape(-1)
        self.repaint()

    def apply_feedback_and_recompute(self):
        # delikatny feedback z mozaiki do meta wektorów (Δ)
        delta = float(self.s_delta.get())
        if delta > 0.0 and self.nodes:
            lam = float(self.s_lambda.get())
            pos = coords_for_lambda(self.nodes, lam)
            rows, cols = self.M.rows, self.M.cols
            edge2d = self.M.edge.reshape(rows, cols)

            def feats_at(x, y):
                xs = np.array([p[0] for p in pos.values()], dtype=float)
                ys = np.array([p[1] for p in pos.values()], dtype=float)
                x0, x1 = float(xs.min()), float(xs.max())
                y0, y1 = float(ys.min()), float(ys.max())
                if x1 - x0 < 1e-9: x1 = x0 + 1.0
                if y1 - y0 < 1e-9: y1 = y0 + 1.0
                cc = int(round((x - x0) / (x1 - x0) * (cols - 1)))
                rr = int(round((y - y0) / (y1 - y0) * (rows - 1)))
                rr = max(0, min(rows - 1, rr)); cc = max(0, min(cols - 1, cc))
                r0, r1 = max(0, rr - 1), min(rows, rr + 2)
                c0, c1 = max(0, cc - 1), min(cols, cc + 2)
                ids = [r * cols + c for r in range(r0, r1) for c in range(c0, c1)]
                return region_feats(self.M, set(ids))

            for n in self.nodes:
                x, y, _z = pos[n.name]
                psi = feats_at(x, y)
                n.meta = (1.0 - delta) * n.meta + delta * psi

        # po feedbacku też przebuduj mozaikę (bo zmieniły się energie/entropie)
        self.recompute_mosaic_from_ast()

    def on_region_changed(self):
        self.reg_key = self.cmb_region.get() or "ROI-A"
        self.repaint()

    # ------------- interakcje myszą -------------
    def on_pick_ast(self, event):
        art = event.artist
        tag = getattr(art, "_glitch_name", None)
        if not tag: return
        for lab, n in self.node_by_label.items():
            if n.name == tag:
                self.cmb_node.set(lab)
                self.on_node_changed()
                break

    def on_click_any(self, event):
        if event.inaxes != self.ax_mos: return
        if event.xdata is None or event.ydata is None: return
        col = int(round(event.xdata)); row = int(round(event.ydata))
        if (row < 0 or row >= self.M.rows or col < 0 or col >= self.M.cols): return

        r0, r1 = max(0, row - 2), min(self.M.rows, row + 2)
        c0, c1 = max(0, col - 2), min(self.M.cols, col + 2)
        mask = np.zeros(self.M.rows * self.M.cols)
        for r in range(r0, r1):
            for c in range(c0, c1):
                mask[r * self.M.cols + c] = 1.0

        if event.button == 1:   # LPM
            self.M.roiA = mask
            if self.reg_key == "ROI-A":
                self.repaint()
        elif event.button == 3: # PPM
            self.M.roiB = mask
            if self.reg_key == "ROI-B":
                self.repaint()

    # ------------- repaint -------------
    def repaint(self):
        if not self.nodes:
            return
        lam = float(self.s_lambda.get())
        beta = float(self.s_beta.get())
        reg_key = self.reg_key

        pos = coords_for_lambda(self.nodes, lam)
        self.pos = pos

        self.ax_ast.cla(); self.ax_mos.cla(); self.ax_inf.cla()

        draw_ast(self.ax_ast, self.nodes, pos, pick=True)
        draw_mosaic(self.ax_mos, self.M)
        draw_region_frame(self.ax_mos, self.M, "ROI-A", color="lime")
        draw_region_frame(self.ax_mos, self.M, "ROI-B", color="magenta")
        draw_infographic(self.ax_inf)

        sel_label = self.cmb_node.get()
        node = None
        if sel_label and sel_label in self.node_by_label:
            node = self.node_by_label[sel_label]
        else:
            node = self.nodes[0]

        ids = region_indices(self.M, reg_key)
        reg_vec = region_feats(self.M, ids)
        fused = fuse_meta(np.array(node.meta), reg_vec, lam, beta=beta)
        draw_fusion(self.ax_ast, self.ax_mos, node, pos, self.M, reg_key, lam, fused)

        xs, ys, zs = zip(*pos.values())
        self.ax_ast.set_xlim(min(xs) - 1, max(xs) + 1)
        self.ax_ast.set_ylim(min(ys) - 1, max(ys) + 1)
        self.ax_ast.set_zlim(min(zs) - 1, max(zs) + 3.5)
        self.ax_ast.set_title(
            f"AST — λ={lam:.2f} · node={node.name} [{node.kind}] · reg={reg_key} · "
            f"β={self.s_beta.get():.1f} · γ={self.s_gamma.get():.2f} · Δ={self.s_delta.get():.2f} · η={self.s_eta.get():.1f}"
        )

        self.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────
# 7) MAIN
# ─────────────────────────────────────────────────────────────

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
