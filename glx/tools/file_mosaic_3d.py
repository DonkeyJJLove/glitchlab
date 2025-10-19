# glitchlab/glx/tools/file_mosaic_3d.py
# -*- coding: utf-8 -*-
"""
File-level 3D viz (offline, no CORS) — Plotly backend — z relacjami Mozaika ↔ AST
i stałym HUD-em do przełączania trybów.

Co pokazujemy:
- Prawa część: 3D graf (file / definicje / połączenia wywołań).
- Lewa część: mozaika „Z” z naszej matematyki (tu: znormalizowana głębokość AST per linia).
- Warstwa relacji: kontury spanów definicji na mozaice + łącza def → pasmo mozaiki.
- HUD: stałe przyciski do przełączania Surface/Bars, pokazywania samego grafu, samej mozaiki,
       obydwu warstw oraz relacji.

Użycie (Windows PowerShell):
  python -m glitchlab.glx.tools.file_mosaic_3d --repo-root .\\glitchlab --file core\\astmap.py
Opcje:
  --rows 32 --cols 32
  --max-rel 40
  --out  .glx/graphs/file_mosaic_3d.html
"""

from __future__ import annotations
import argparse, ast, math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import networkx as nx
from plotly.offline import plot as plot_offline
import plotly.graph_objects as go

# (opcjonalnie) bogatszy indeks jeśli masz nasz moduł; nie jest wymagany
try:
    from glitchlab.analysis.ast_index import ast_index_of_source  # type: ignore
except Exception:
    ast_index_of_source = None  # type: ignore

HEIGHT_SCALE_BARS = 1.2  # musi odpowiadać _mosaic_trace_bars()
# ──────────────────────────────────────────────────────────────────────────────
# AST lite (Py3.9; bez ast.Match w typach) — deterministyczne i szybkie
# ──────────────────────────────────────────────────────────────────────────────

def _get_end_lineno(node: ast.AST) -> int:
    end_ln = getattr(node, "end_lineno", None)
    if isinstance(end_ln, int):
        return end_ln
    max_ln = getattr(node, "lineno", 1)
    for ch in ast.walk(node):
        ln = getattr(ch, "lineno", None)
        if isinstance(ln, int) and ln > max_ln:
            max_ln = ln
    return max_ln


@dataclass(frozen=True)
class DefRec:
    qualname: str
    kind: str  # function|async_function|class|method|async_method
    lineno: int
    end_lineno: int


@dataclass(frozen=True)
class CallRec:
    scope: str
    func: str
    lineno: int


@dataclass
class AstLite:
    defs: Dict[str, DefRec]
    calls: List[CallRec]
    lines: int
    depth_per_line: Dict[int, int]
    calls_per_line: Dict[int, int]


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: List[str] = []
        self.defs: Dict[str, DefRec] = {}
        self.calls: List[CallRec] = []
        self.depth_per_line: Dict[int, int] = {}
        self.calls_per_line: Dict[int, int] = {}
        self._depth = 0

    def _push(self, name: str) -> None:
        self.scope.append(name)

    def _pop(self) -> None:
        if self.scope:
            self.scope.pop()

    def _cur(self) -> str:
        return ".".join(self.scope) if self.scope else "<module>"

    def generic_visit(self, node: ast.AST) -> None:
        self._depth += 1
        ln = getattr(node, "lineno", None)
        if isinstance(ln, int):
            self.depth_per_line[ln] = max(self.depth_per_line.get(ln, 0), self._depth)
        super().generic_visit(node)
        self._depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = "method" if self.scope and self.scope[-1][:1].isupper() else "function"
        q = ".".join(self.scope + [node.name]) if self.scope else node.name
        self.defs[q] = DefRec(q, kind, node.lineno, _get_end_lineno(node))
        self._push(node.name)
        self.generic_visit(node)
        self._pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        kind = "async_method" if self.scope and self.scope[-1][:1].isupper() else "async_function"
        q = ".".join(self.scope + [node.name]) if self.scope else node.name
        self.defs[q] = DefRec(q, kind, node.lineno, _get_end_lineno(node))
        self._push(node.name)
        self.generic_visit(node)
        self._pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        q = ".".join(self.scope + [node.name]) if self.scope else node.name
        self.defs[q] = DefRec(q, "class", node.lineno, _get_end_lineno(node))
        self._push(node.name)
        self.generic_visit(node)
        self._pop()

    def visit_Call(self, node: ast.Call) -> None:
        def dotted(n: ast.AST) -> Optional[str]:
            if isinstance(n, ast.Name): return n.id
            if isinstance(n, ast.Attribute):
                b = dotted(n.value); return f"{b}.{n.attr}" if b else n.attr
            if isinstance(n, ast.Call): return dotted(n.func)
            return None
        f = dotted(node.func) or "<lambda>"
        ln = getattr(node, "lineno", -1)
        self.calls.append(CallRec(self._cur(), f, ln))
        if isinstance(ln, int) and ln > 0:
            self.calls_per_line[ln] = self.calls_per_line.get(ln, 0) + 1
        self.generic_visit(node)


def astlite_of_source(src: str) -> AstLite:
    t = ast.parse(src)
    v = _Visitor(); v.visit(t)
    # lekka „wygładzona” głębokość
    n = len(src.splitlines())
    if n >= 3:
        for i in range(2, n):
            a = v.depth_per_line.get(i - 1, 0)
            b = v.depth_per_line.get(i, 0)
            c = v.depth_per_line.get(i + 1, 0)
            v.depth_per_line[i] = max(v.depth_per_line.get(i, 0), (a + b + c) // 2)
    return AstLite(defs=v.defs, calls=v.calls, lines=n,
                   depth_per_line=v.depth_per_line, calls_per_line=v.calls_per_line)


# ──────────────────────────────────────────────────────────────────────────────
# Graf (NX) i meta
# ──────────────────────────────────────────────────────────────────────────────

def build_graph_for_file(src: str, file_label: str) -> tuple[nx.Graph, Dict[str, Dict[str, Any]], AstLite]:
    al = astlite_of_source(src)
    G = nx.Graph()
    nmeta: Dict[str, Dict[str, Any]] = {}

    file_id = f"file:{file_label}"
    G.add_node(file_id)
    nmeta[file_id] = {"label": Path(file_label).name, "kind": "file", "size": 16, "span": (1, al.lines)}

    for q, d in al.defs.items():
        nid = f"def:{q}"
        G.add_node(nid)
        span = max(1, d.end_lineno - d.lineno + 1)
        nmeta[nid] = {
            "label": q, "kind": d.kind, "size": max(6, min(18, 6 + int(math.log2(2 + span)) * 2)),
            "span": (d.lineno, d.end_lineno)
        }
        G.add_edge(file_id, nid)

    # calls → łącz do znanych defs, w przeciwnym razie do ext:*
    name_map = {q.split(".")[-1]: f"def:{q}" for q in al.defs.keys()}
    for c in al.calls:
        target = name_map.get(c.func.split(".")[-1])
        src_node = f"def:{c.scope}" if c.scope != "<module>" and f"def:{c.scope}" in G else file_id
        if target:
            G.add_edge(src_node, target)
        else:
            ext_id = f"ext:{c.func}"
            if ext_id not in G:
                G.add_node(ext_id)
                nmeta[ext_id] = {"label": c.func, "kind": "ext", "size": 5, "span": (c.lineno, c.lineno)}
                G.add_edge(file_id, ext_id)
            G.add_edge(src_node, ext_id)

    # layout 3D (deterministyczny seed)
    pos = nx.spring_layout(G, dim=3, seed=42, iterations=200)
    for nid, p in pos.items():
        nmeta.setdefault(nid, {"label": nid, "kind": "node", "size": 6, "span": (1, 1)})
        nmeta[nid]["pos"] = (float(p[0]), float(p[1]), float(p[2]))

    # „fizyka” (hover)
    max_deg = max((G.degree(n) for n in G.nodes), default=1) or 1
    for nid, m in nmeta.items():
        deg = G.degree(nid)
        span = m.get("span", (1, 1))
        nlines = max(1, al.lines)
        lam_span = (span[1] - span[0] + 1) / nlines
        lam_cpl = deg / max_deg
        m["lambda_span"] = float(round(lam_span, 3))
        m["lambda_coupling"] = float(round(lam_cpl, 3))

    return G, nmeta, al


# ──────────────────────────────────────────────────────────────────────────────
# Mozaika (Z) i metryki relacji
# ──────────────────────────────────────────────────────────────────────────────

def mosaic_grid_from_source(al: AstLite, rows: int = 24, cols: int = 24) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Raster mozaiki: dla każdego wiersza bierzemy max głębokości AST w odpowiadającym mu
    zakresie linii i kopiujemy ją na wszystkie kolumny (dla widoczności).
    """
    nlines = max(1, al.lines)
    depth = np.zeros(nlines + 1, dtype=np.float32)
    for ln, d in al.depth_per_line.items():
        if 0 <= ln <= nlines:
            depth[ln] = max(depth[ln], float(d))

    Z = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        y0 = int(r * nlines / rows) + 1
        y1 = int((r + 1) * nlines / rows) + 1
        val = float(depth[y0:y1].max()) if y1 > y0 else float(depth[min(nlines, y0)])
        Z[r, :] = val

    mx = float(Z.max()) if Z.size else 1.0
    if mx > 0: Z /= mx
    return Z, {"rows": rows, "cols": cols, "nlines": nlines, "max_depth": float(mx)}


# ──────────────────────────────────────────────────────────────────────────────
# Mozaika: dwie reprezentacje — Surface (gładka) i Bars (kafelkowa)
# ──────────────────────────────────────────────────────────────────────────────

def _mosaic_trace_surface(Z: np.ndarray) -> go.Surface:
    """Gładka powierzchnia — jak „stare” heatmapy 3D."""
    rows, cols = Z.shape
    yy = np.linspace(-1.2, 1.2, rows)
    xx = np.linspace(-1.2, 1.2, cols)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = (Z * 1.2) - 0.1
    XX = XX - 2.4  # przesunięcie w lewo, graf stoi po prawej
    return go.Surface(
        x=XX, y=YY, z=ZZ, colorscale="Viridis", opacity=0.78,
        colorbar=dict(title="intensity"), showscale=True,
        name="mosaic_surface", visible=False  # domyślnie wyłączamy (będą Bars)
    )


def _mosaic_trace_bars(Z: np.ndarray, height_scale: float = 1.2) -> go.Mesh3d:
    """
    Kafelkowa mozaika 3D (Turbo) — każdy blok to prostopadłościan o wysokości ~ Z[r,c].
    Zwracamy pojedynczy Mesh3d (oszczędnie pamięciowo).
    """
    rows, cols = Z.shape
    # siatka w przestrzeni (jak Surface): X w [-2.4, 0], Y w [-1.2, 1.2]
    x_left, x_right = -2.4, 0.0
    y_bottom, y_top = -1.2, 1.2
    dx = (x_right - x_left) / cols
    dy = (y_top - y_bottom) / rows

    xs, ys, zs = [], [], []
    i_idx, j_idx, k_idx = [], [], []
    intens = []  # kolor/intensywność na wierzchołek (stała dla klocka)

    def add_box(x0, x1, y0, y1, h, color_val):
        base = len(xs)
        vx = [x0, x1, x1, x0, x0, x1, x1, x0]
        vy = [y0, y0, y1, y1, y0, y0, y1, y1]
        vz = [0, 0, 0, 0, h, h, h, h]
        xs.extend(vx); ys.extend(vy); zs.extend(vz)
        intens.extend([color_val]*8)
        faces = [
            (4, 5, 6), (4, 6, 7),     # top
            (0, 1, 5), (0, 5, 4),     # bok1
            (1, 2, 6), (1, 6, 5),     # bok2
            (2, 3, 7), (2, 7, 6),     # bok3
            (3, 0, 4), (3, 4, 7),     # bok4
        ]
        for a, b, c in faces:
            i_idx.append(base + a)
            j_idx.append(base + b)
            k_idx.append(base + c)

    Z = np.asarray(Z, dtype=np.float32)
    zmax = float(Z.max()) if Z.size else 1.0
    Zn = (Z / zmax) if zmax > 0 else Z

    for r in range(rows):
        y0 = y_bottom + r*dy
        y1 = y0 + dy
        for c in range(cols):
            x0 = x_left + c*dx
            x1 = x0 + dx
            v = float(Zn[r, c])
            h = max(0.02, v * height_scale)
            add_box(x0, x1, y0, y1, h, v)

    return go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=i_idx, j=j_idx, k=k_idx,
        intensity=intens, colorscale="Turbo", showscale=True,
        opacity=0.90, name="mosaic_bars", visible=True
    )




# Kolory węzłów grafu
COLORS = {
    "file": "rgb(52, 211, 153)",        # emerald-400
    "class": "rgb(250, 204, 21)",       # amber-400
    "function": "rgb(96, 165, 250)",    # blue-400
    "async_function": "rgb(147, 197, 253)",
    "method": "rgb(244, 114, 182)",     # pink-400
    "async_method": "rgb(251, 113, 133)",
    "ext": "rgb(156, 163, 175)",        # gray-400
    "node": "rgb(147, 197, 253)",
}

# ──────────────────────────────────────────────────────────────────────────────
# Relacje i współczynniki
# ──────────────────────────────────────────────────────────────────────────────

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    if a.size != b.size or a.size == 0: return 0.0
    a = a - a.mean(); b = b - b.mean()
    sa = float(np.sqrt((a*a).sum())); sb = float(np.sqrt((b*b).sum()))
    if sa == 0.0 or sb == 0.0: return 0.0
    return float(np.clip((a*b).sum()/(sa*sb), -1.0, 1.0))

def compute_relation_coeffs(al: AstLite, Z: np.ndarray, rows: int) -> Dict[str, float]:
    """ρ(depth,calls), ρ(depth,in_def) oraz ρ(Z_row, calls_row)"""
    n = max(1, al.lines)
    depth = np.zeros(n, dtype=np.float32)
    calls = np.zeros(n, dtype=np.float32)
    in_def = np.zeros(n, dtype=np.float32)

    for ln, d in al.depth_per_line.items():
        if 1 <= ln <= n: depth[ln-1] = max(depth[ln-1], float(d))
    for ln, c in al.calls_per_line.items():
        if 1 <= ln <= n: calls[ln-1] += float(c)
    for d in al.defs.values():
        l0, l1 = max(1, d.lineno), min(n, d.end_lineno)
        in_def[l0-1:l1] = 1.0

    if depth.max() > 0: depth /= depth.max()
    if calls.max() > 0: calls /= calls.max()

    r1 = _pearson(depth, calls)
    r2 = _pearson(depth, in_def)

    row_calls = np.zeros(rows, dtype=np.float32)
    for r in range(rows):
        y0 = int(r * n / rows); y1 = int((r + 1) * n / rows)
        row_calls[r] = calls[y0:y1].sum() if y1 > y0 else calls[min(n-1, y0)]
    if row_calls.max() > 0: row_calls /= row_calls.max()

    Zrow = Z.mean(axis=1) if Z.size else np.zeros(rows, dtype=np.float32)
    r3 = _pearson(Zrow, row_calls)

    return {
        "rho_depth_calls": round(float(r1), 3),
        "rho_depth_in_def": round(float(r2), 3),
        "rho_Zrow_calls": round(float(r3), 3),
        "defs": len(al.defs),
        "calls": int(calls.sum()),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Ślady Plotly: graf, prostokąty spanów, linki def→mozaika
# ──────────────────────────────────────────────────────────────────────────────

def _node_traces(G: nx.Graph, nmeta: Dict[str, Dict[str, Any]]) -> tuple[go.Scatter3d, go.Scatter3d]:
    # nodes
    xs, ys, zs, sizes, colors, labels, hover = [], [], [], [], [], [], []
    for nid, m in nmeta.items():
        x, y, z = m["pos"]
        xs.append(x); ys.append(y); zs.append(z)
        sizes.append(m.get("size", 8))
        k = m.get("kind", "node")
        colors.append(COLORS.get(k, COLORS["node"]))
        label = m.get("label", nid)
        span = m.get("span", (None, None))
        lam_s = m.get("lambda_span", 0.0); lam_c = m.get("lambda_coupling", 0.0)
        labels.append(label)
        hover.append(
            f"{label}<br>"
            f"<span style='font-size:11px;color:#9aa'>[{k}] lines {span[0]}–{span[1]} • "
            f"λ_span={lam_s} • λ_cpl={lam_c}</span>"
        )

    nodes = go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers+text",
        marker=dict(size=sizes, color=colors, opacity=0.95),
        text=labels, textposition="top center",
        hovertext=hover, hoverinfo="text",
        name="nodes", visible=True
    )

    # edges
    xe, ye, ze = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = nmeta[u]["pos"]; x1, y1, z1 = nmeta[v]["pos"]
        xe += [x0, x1, None]; ye += [y0, y1, None]; ze += [z0, z1, None]
    edges = go.Scatter3d(
        x=xe, y=ye, z=ze, mode="lines",
        line=dict(color="rgba(255,255,255,0.25)", width=2),
        hoverinfo="skip", name="edges", visible=True
    )
    return edges, nodes

def _row_to_y(r: int, rows: int) -> float:
    y_bottom, y_top = -1.2, 1.2
    return y_bottom + (r + 0.5) * ((y_top - y_bottom) / rows)

def _def_spans_to_relations(nmeta: Dict[str, Dict[str, Any]],
                            Z: np.ndarray, rows: int, cols: int, nlines: int,
                            max_rel: int = 40) -> tuple[go.Scatter3d, go.Scatter3d]:
    """Kontury spanów + kotwice def→pasmo mozaiki (koordy zgodne z Bars/Surface)."""
    x_left, x_right = -2.4, 0.0
    z_eps = 0.02

    # wybierz największe spany (czytelność)
    defs = [(nid, m) for nid, m in nmeta.items()
            if m.get("kind") in ("class", "function", "async_function", "method", "async_method")]
    defs.sort(key=lambda kv: (kv[1]["span"][1] - kv[1]["span"][0] + 1), reverse=True)
    defs = defs[:max_rel]

    xr, yr, zr = [], [], []     # rectangles (batched with None)
    xl, yl, zl = [], [], []     # links

    for nid, m in defs:
        l0, l1 = m["span"]
        r0 = max(0, min(rows - 1, int((l0 - 1) * rows / max(1, nlines))))
        r1 = max(0, min(rows - 1, int((l1 - 1) * rows / max(1, nlines))))
        if r1 < r0: r0, r1 = r1, r0
        y0, y1 = _row_to_y(r0, rows), _row_to_y(r1, rows)

        # wysokość = „szczyt” bary w środku pasa
        v = float(Z[min(rows - 1, (r0 + r1) // 2), cols // 2])
        z_row = max(0.02, v * HEIGHT_SCALE_BARS) + z_eps

        # prostokąt z czterech krawędzi
        xr += [x_left, x_right, None, x_right, x_right, None, x_right, x_left, None, x_left, x_left, None]
        yr += [y0, y0, None, y0, y1, None, y1, y1, None, y1, y0, None]
        zr += [z_row] * 12

        # link def → środek pasa mozaiki
        nx_, ny_, nz_ = m["pos"]
        xm, ym, zm = ((x_left + x_right) / 2.0, (y0 + y1) / 2.0, z_row)
        xl += [nx_, xm, None]; yl += [ny_, ym, None]; zl += [nz_, zm, None]

    rects = go.Scatter3d(
        x=xr, y=yr, z=zr, mode="lines",
        line=dict(color="rgba(250,204,21,0.55)", width=3),
        hoverinfo="skip", name="def_spans", visible=False
    )
    links = go.Scatter3d(
        x=xl, y=yl, z=zl, mode="lines",
        line=dict(color="rgba(147,197,253,0.55)", width=2),
        hoverinfo="skip", name="relations", visible=False
    )
    return rects, links

# ──────────────────────────────────────────────────────────────────────────────
# Składanie figury + HUD
# ──────────────────────────────────────────────────────────────────────────────

def build_figure(G: nx.Graph, nmeta: Dict[str, Dict[str, Any]],
                 Z: np.ndarray, al: AstLite, title: str,
                 max_rel: int = 40, init_surface: bool = False) -> go.Figure:
    edges_t, nodes_t = _node_traces(G, nmeta)
    bars_t = _mosaic_trace_bars(Z, height_scale=HEIGHT_SCALE_BARS)
    surf_t = _mosaic_trace_surface(Z)

    rows, cols = Z.shape
    rects_t, links_t = _def_spans_to_relations(nmeta, Z, rows, cols, al.lines, max_rel=max(1, max_rel))

    # Inicjalna widoczność (domyślnie Bars; gdy init_surface=True — Surface)
    if init_surface:
        bars_t.visible = False
        surf_t.visible = True
    else:
        bars_t.visible = True
        surf_t.visible = False

    coeffs = compute_relation_coeffs(al, Z, rows)
    coeff_txt = (
        f"ρ(depth,calls)={coeffs['rho_depth_calls']} • "
        f"ρ(depth,in_def)={coeffs['rho_depth_in_def']} • "
        f"ρ(Zrow,calls_row)={coeffs['rho_Zrow_calls']}<br>"
        f"defs={coeffs['defs']} • calls≈{coeffs['calls']}"
    )

    fig = go.Figure(data=[edges_t, nodes_t, bars_t, surf_t, rects_t, links_t])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            bgcolor="#0b0e14",
            aspectmode="data",
        ),
        paper_bgcolor="#0b0e14",
        font=dict(color="#e6e6e6"),
        margin=dict(l=0, r=0, t=54, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", x=0.02, y=0.98),
        updatemenus=[dict(
            type="buttons", direction="right", x=0.02, y=1.06,
            buttons=[
                dict(label="Both+Relations", method="update",
                     args=[{"visible": [True, True, bars_t.visible, surf_t.visible, True, True]}]),
                dict(label="Both", method="update",
                     args=[{"visible": [True, True, bars_t.visible, surf_t.visible, False, False]}]),
                dict(label="Graph", method="update",
                     args=[{"visible": [True, True, False, False, False, False]}]),
                dict(label="Mosaic Bars", method="update",
                     args=[{"visible": [False, False, True, False, False, False]}]),
                dict(label="Mosaic Surface", method="update",
                     args=[{"visible": [False, False, False, True, False, False]}]),
            ],
            bgcolor="rgba(0,0,0,0.45)", bordercolor="rgba(255,255,255,0.2)",
        )],
        annotations=[
            dict(
                text="<b>Right:</b> call graph • <b>Left:</b> 3D mosaic (AST depth)<br>"
                     "Relacje: żółte kontury = spany definicji; niebieskie = def→pasmo.",
                x=0.5, y=1.12, xref="paper", yref="paper", showarrow=False,
                font=dict(size=12, color="#9aa")
            ),
            dict(
                text=f"<b>Coefficients:</b> {coeff_txt}",
                x=0.5, y=1.08, xref="paper", yref="paper", showarrow=False,
                font=dict(size=12, color="#cbd5e1")
            ),
        ],
    )
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_file(repo_root: Path, user_path: str) -> Path:
    p = Path(user_path)
    if not p.is_absolute():
        cand1 = repo_root / p
        if cand1.exists(): return cand1.resolve()
        cand2 = repo_root / "glitchlab" / p
        if cand2.exists(): return cand2.resolve()
    return p.resolve()

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="File-level 3D viz (Plotly, offline) — Mosaic↔AST relations")
    ap.add_argument("--repo-root", "--repo", dest="repo_root", default=".",
                    help="ROOT projektu (z folderem glitchlab/)")
    ap.add_argument("--file", required=True, help="plik (np. core/astmap.py lub pełna ścieżka)")
    ap.add_argument("--rows", type=int, default=24, help="mozaika: liczba wierszy")
    ap.add_argument("--cols", type=int, default=24, help="mozaika: liczba kolumn")
    ap.add_argument("--max-rel", type=int, default=40, help="maks. liczba definicji do rysowania relacji")
    ap.add_argument("--mosaic", choices=["bars", "surface"], default="bars",
                    help="domyślny tryb mozaiki (HUD pozwala przełączać)")
    ap.add_argument("--out", default=None, help="ścieżka HTML (domyślnie .glx/graphs/file_mosaic_3d.html)")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    fpath = _resolve_file(root, args.file)
    if not fpath.exists():
        print(f"[ERR] file not found: {fpath}")
        return 2

    src = fpath.read_text(encoding="utf-8", errors="ignore")
    G, nmeta, al = build_graph_for_file(src, file_label=str(fpath))
    Z, _meta = mosaic_grid_from_source(al, rows=max(4, args.rows), cols=max(4, args.cols))

    title = f"File 3D Viz — {fpath.name}  •  nodes:{G.number_of_nodes()} edges:{G.number_of_edges()}"
    fig = build_figure(
        G, nmeta, Z, al, title,
        max_rel=max(1, args.max_rel),
        init_surface=(args.mosaic == "surface"),
    )

    out = Path(args.out).resolve() if args.out else (root / ".glx" / "graphs" / "file_mosaic_3d.html").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_offline(fig, filename=str(out), auto_open=False, include_plotlyjs="inline")
    print(str(out))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

