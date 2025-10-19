# glitchlab/analysis/phi_psi_bridge.py
# -*- coding: utf-8 -*-
# Python 3.9+
"""
Most: metasoczewka / graf projektu → mozaika (Φ/Ψ).

Funkcje:
- scope_to_mosaic(ScopeResult, ...)        → AnalysisMosaic
- graph_to_mosaic(ProjectGraph, ...)       → AnalysisMosaic
- project_delta_to_mosaic(delta_report)    → AnalysisMosaic  (kompatybilność wsteczna)

Założenia:
- Layout kafli = równy grid (deterministyczny). Układ stabilny po (kind, label, id).
- 'edge' można podać jako mapa node_id→value; w przeciwnym razie liczony z (ważonego) stopnia.
- 'roi' można podać jako zbiór node_id; w przeciwnym razie – domyślna (środek 30–62%) w adapterze.

Zależności:
- analysis.mosaic_adapter.core_to_analysis_mosaic
- analysis.project_graph.ProjectGraph
- glitchlab.gui.mosaic.hybrid_ast_mosaic (pośrednio przez adapter)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, Set

import numpy as np

# ── Adapter Φ/Ψ ───────────────────────────────────────────────────────────────
from .mosaic_adapter import core_to_analysis_mosaic

# ── Modele grafu projektu ────────────────────────────────────────────────────
try:
    from .project_graph import ProjectGraph, Node, Edge  # type: ignore
except Exception:  # pragma: no cover
    ProjectGraph = Any  # type: ignore
    Node = Any          # type: ignore
    Edge = Any          # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# API publiczne
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "scope_to_mosaic",
    "graph_to_mosaic",
    "project_delta_to_mosaic",
]


# ──────────────────────────────────────────────────────────────────────────────
# Konfiguracja layoutu
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LayoutConfig:
    """
    Konfiguracja generowania siatki kafli pod mozaikę:
    - canvas_size: (W,H) — rozmiar płótna dla równomiernych centrów
    - prefer_hex:  czy wymusić 'hex' (adapter i tak poradzi sobie z kind="grid"/"hex")
    - rows_cols:   wymuszone (rows, cols); gdy None — auto z liczby węzłów i proporcji W/H
    """
    canvas_size: Tuple[int, int] = (1200, 800)
    prefer_hex: bool = False
    rows_cols: Optional[Tuple[int, int]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Most: ScopeResult → Mosaic
# ──────────────────────────────────────────────────────────────────────────────

def scope_to_mosaic(
    scope_result: "Any",
    *,
    edge_scores: Optional[Mapping[str, float]] = None,
    roi_nodes: Optional[Iterable[str]] = None,
    layout: Optional[LayoutConfig] = None,
    kind: Optional[str] = None,
) -> "object":
    """
    Rzutuje wynik metasoczewki (ScopeResult) na AnalysisMosaic.

    Parametry:
      - edge_scores:     mapowanie node_id→edge∈[0,1]; gdy None → (ważony) stopień w podgrafie.
      - roi_nodes:       iterowalne node_id; gdy None → adapter użyje domyślnej ROI (środek).
      - layout:          konfiguracja siatki (canvas, rows/cols).
      - kind:            'grid'/'hex' lub None (heurystyka z layoutu).
    """
    sub = getattr(scope_result, "graph_sub", None) or getattr(scope_result, "graph", None)
    if sub is None:
        raise ValueError("ScopeResult nie zawiera subgrafu ('graph_sub' lub 'graph').")
    return graph_to_mosaic(
        sub,
        nodes=None,  # cały subgraf widoku
        edge_scores=edge_scores,
        roi_nodes=roi_nodes or set(getattr(scope_result, "anchors", []) or []),
        layout=layout,
        kind=kind,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Most: ProjectGraph (+ maski/metyki) → Mosaic
# ──────────────────────────────────────────────────────────────────────────────

def graph_to_mosaic(
    g: ProjectGraph,
    *,
    nodes: Optional[Iterable[str]] = None,
    edge_scores: Optional[Mapping[str, float]] = None,
    roi_nodes: Optional[Iterable[str]] = None,
    layout: Optional[LayoutConfig] = None,
    kind: Optional[str] = None,
) -> "object":
    """
    Rzutuje (pod)graf projektu na AnalysisMosaic.

    - nodes:       jeśli podane, ogranicza do danego zbioru node_id; w przeciwnym razie bierze wszystkie.
    - edge_scores: mapowanie node_id→edge∈[0,1]; brak → użyj (ważonego) stopnia (degree).
    - roi_nodes:   zbiór node_id w ROI; brak → adapter użyje domyślnej ROI.
    - layout:      konfiguracja siatki (W,H oraz opcjonalnie rows/cols).
    - kind:        'grid'/'hex' lub None (z layoutu).
    """
    layout = layout or LayoutConfig()
    node_ids = _materialize_node_ids(g, nodes)
    if not node_ids:
        # pusta mozaika
        core = _grid_from_nodes([], layout)
        return core_to_analysis_mosaic(core, edges_per_cell=np.zeros(0), roi_mask=np.zeros(0), kind=kind)

    ordered_nodes = _deterministic_node_order(g, node_ids)
    core = _grid_from_nodes(ordered_nodes, layout)

    # edge vector
    edge_vec = _edge_vector_for_nodes(g, ordered_nodes, edge_scores=edge_scores)

    # roi vector (opcjonalnie)
    roi_vec = _roi_vector_for_nodes(ordered_nodes, roi_nodes) if roi_nodes is not None else None

    return core_to_analysis_mosaic(
        core,
        edges_per_cell=edge_vec,
        block_stats=None,
        metric_name="edges",
        roi_mask=roi_vec,
        kind=(kind or ("hex" if layout.prefer_hex else "grid")),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Implementacja layoutu i wektorów
# ──────────────────────────────────────────────────────────────────────────────

_KIND_ORDER = {
    "project": -1,  # umowny korzeń
    "module": 0,
    "file": 1,
    "func": 2,
    "topic": 3,
}  # pozostałe → 99

def _materialize_node_ids(g: ProjectGraph, nodes: Optional[Iterable[str]]) -> List[str]:
    if nodes is None:
        return list(g.nodes.keys())
    out = []
    for nid in nodes:
        if nid in g.nodes:
            out.append(nid)
    return out

def _deterministic_node_order(g: ProjectGraph, node_ids: List[str]) -> List[Node]:
    def key(n: Node) -> Tuple[int, str, str]:
        ko = _KIND_ORDER.get(str(n.kind).lower(), 99)
        label = str(getattr(n, "label", "") or "")
        return (ko, label.lower(), str(n.id))
    nodes = [g.nodes[nid] for nid in node_ids]
    nodes.sort(key=key)
    return nodes

def _auto_rows_cols(n: int, canvas: Tuple[int, int]) -> Tuple[int, int]:
    if n <= 0:
        return (0, 0)
    W, H = max(1, int(canvas[0])), max(1, int(canvas[1]))
    ratio = float(W) / float(H)
    cols = max(1, int(math.ceil(math.sqrt(n * ratio))))
    rows = max(1, int(math.ceil(n / cols)))
    return (rows, cols)

def _grid_from_nodes(nodes: List[Node], layout: LayoutConfig) -> Dict[str, Any]:
    W, H = layout.canvas_size
    rows, cols = layout.rows_cols or _auto_rows_cols(len(nodes), layout.canvas_size)
    cells: List[Dict[str, Any]] = []
    if rows > 0 and cols > 0:
        sx = W / max(1, cols)
        sy = H / max(1, rows)
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx >= len(nodes):
                    break
                cx = int(round((c + 0.5) * sx))
                cy = int(round((r + 0.5) * sy))
                cells.append({"id": idx, "center": (cx, cy), "meta": {"node_id": nodes[idx].id}})
    return {"mode": ("hex" if layout.prefer_hex else "square"), "size": (W, H), "cells": cells, "raster": None}

def _weighted_degree(g: ProjectGraph, nid: str) -> float:
    deg = 0.0
    for e in g.edges:
        if e.src == nid or e.dst == nid:
            w = getattr(e, "weight", None)
            try:
                deg += float(w) if w is not None else 1.0
            except Exception:
                deg += 1.0
    return deg

def _edge_vector_for_nodes(
    g: ProjectGraph,
    nodes: List[Node],
    *,
    edge_scores: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    N = len(nodes)
    vec = np.zeros(N, dtype=float)
    if edge_scores:
        for i, n in enumerate(nodes):
            try:
                vec[i] = float(edge_scores.get(n.id, 0.0))
            except Exception:
                vec[i] = 0.0
    else:
        # fallback: (ważony) stopień w subgrafie
        # UWAGA: traktujemy 'g' jako już zawężony podgraf — jeśli przekazano pełny graf,
        #        a chcemy liczyć w subgrafie, należy go wcześniej uciąć do 'nodes'.
        for i, n in enumerate(nodes):
            vec[i] = _weighted_degree(g, n.id)
    # normalizacja do [0,1]
    mx = float(np.max(vec)) if vec.size else 0.0
    if mx > 0.0:
        vec = vec / mx
    vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)
    vec = np.clip(vec, 0.0, 1.0)
    return vec

def _roi_vector_for_nodes(nodes: List[Node], roi_nodes: Optional[Iterable[str]]) -> np.ndarray:
    N = len(nodes)
    out = np.zeros(N, dtype=float)
    if roi_nodes is None:
        return out
    roi_set = {str(x) for x in roi_nodes}
    for i, n in enumerate(nodes):
        if n.id in roi_set:
            out[i] = 1.0
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Część Δ (kompatybilność): delta_report → Mosaic
# ──────────────────────────────────────────────────────────────────────────────

# Zachowujemy wcześniejsze API, ale kod porządkujemy jako osobną sekcję.

Core = Dict[str, Any]
BlockStats = Mapping[Tuple[int, int], Mapping[str, float]]

def _try_get(d: Mapping[str, Any], *path: str, default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _coerce_block_key(k: Any) -> Optional[Tuple[int, int]]:
    if isinstance(k, (tuple, list)) and len(k) == 2:
        try:
            return (int(k[0]), int(k[1]))
        except Exception:
            return None
    if isinstance(k, str):
        for sep in (",", ";", "|", " "):
            if sep in k:
                a, b = k.split(sep, 1)
                try:
                    return (int(a.strip()), int(b.strip()))
                except Exception:
                    return None
    return None

def _parse_block_stats(obj: Any) -> BlockStats:
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    if obj is None:
        return out
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            ij = _coerce_block_key(k)
            if ij is None or not isinstance(v, Mapping):
                continue
            rec = {str(kk): float(vv) for kk, vv in v.items() if isinstance(vv, (int, float))}
            out[ij] = rec
        return out
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) == 3 and isinstance(item[2], Mapping):
                try:
                    i, j = int(item[0]), int(item[1])
                except Exception:
                    continue
                rec = {str(kk): float(vv) for kk, vv in item[2].items() if isinstance(vv, (int, float))}
                out[(i, j)] = rec
        return out
    return out

def _grid_from_rc(rows: int, cols: int, size: Tuple[int, int]) -> Core:
    W, H = int(size[0]), int(size[1])
    rows = max(0, int(rows))
    cols = max(0, int(cols))
    cells: List[Dict[str, Any]] = []
    if rows > 0 and cols > 0:
        sx = W / max(1, cols)
        sy = H / max(1, rows)
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                cx = int(round((c + 0.5) * sx))
                cy = int(round((r + 0.5) * sy))
                cells.append({"id": idx, "center": (cx, cy)})
    return dict(mode="square", size=(W, H), cells=cells, raster=None)

def _coerce_core(delta: Mapping[str, Any]) -> Core:
    for key in ("core", "mosaic", "repo_mosaic", "mosaic_core"):
        core = _try_get(delta, key)
        if isinstance(core, Mapping) and "cells" in core and "size" in core:
            mode = str(core.get("mode", "square")).lower()
            size = tuple(core.get("size", (1024, 768)))  # type: ignore
            cells = list(core.get("cells", []))          # type: ignore
            return dict(mode=mode, size=size, cells=cells, raster=None)
    rows = int(_try_get(delta, "grid", "rows", default=_try_get(delta, "layout", "rows", default=0)) or 0)
    cols = int(_try_get(delta, "grid", "cols", default=_try_get(delta, "layout", "cols", default=0)) or 0)
    n = int(_try_get(delta, "cells", "count", default=_try_get(delta, "files", "count", default=rows * cols)) or 0)
    if rows == 0 and cols == 0 and n > 0:
        W = int(_try_get(delta, "size", 0, default=1024) or 1024)
        H = int(_try_get(delta, "size", 1, default=768) or 768)
        ratio = max(1.0, float(W) / max(1, H))
        cols = max(1, int(round(math.sqrt(n * ratio))))
        rows = max(1, int(math.ceil(n / cols)))
    size = tuple(_try_get(delta, "size", default=(1024, 768)))  # type: ignore
    return _grid_from_rc(rows, cols, size)

def _coerce_edges_per_cell(delta: Mapping[str, Any], N: int) -> Optional[np.ndarray]:
    for key in ("edges_per_cell", "edge", "edges"):
        arr = _try_get(delta, key)
        if isinstance(arr, (list, tuple)) and len(arr) > 0:
            v = np.asarray(arr, dtype=float).reshape(-1)
            if v.size >= N:
                v = v[:N]
            if v.size == N:
                v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0)
                v = np.clip(v, 0.0, 1.0)
                return v
    cells = _try_get(delta, "cells")
    if isinstance(cells, (list, tuple)) and cells:
        v = np.zeros(N, dtype=float)
        any_set = False
        for i, c in enumerate(cells[:N]):
            if isinstance(c, Mapping) and "edge" in c:
                try:
                    v[i] = float(c["edge"])
                    any_set = True
                except Exception:
                    pass
        if any_set:
            v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0)
            v = np.clip(v, 0.0, 1.0)
            return v
    files = _try_get(delta, "files") or _try_get(delta, "changed_files")
    if isinstance(files, (list, tuple)) and files:
        v = np.zeros(N, dtype=float)
        mx = 1e-9
        for i, item in enumerate(files[:N]):
            if not isinstance(item, Mapping):
                continue
            if "edge" in item:
                try:
                    val = float(item["edge"])
                except Exception:
                    val = 0.0
            elif "impact" in item:
                try:
                    val = float(item["impact"])
                except Exception:
                    val = 0.0
            else:
                dS = float(item.get("dS", 0.0) or 0.0)
                dH = float(item.get("dH", 0.0) or 0.0)
                dZ = float(item.get("dZ", 0.0) or 0.0)
                val = abs(dS) + abs(dH) + 0.25 * abs(dZ)
            v[i] = max(0.0, val)
            mx = max(mx, v[i])
        if mx > 0:
            v /= mx
        v = np.clip(v, 0.0, 1.0)
        return v
    return None

def _coerce_roi(delta: Mapping[str, Any], N: int) -> Optional[np.ndarray]:
    roi = _try_get(delta, "roi") or _try_get(delta, "roi_mask")
    if roi is None:
        idxs = _try_get(delta, "roi_indices") or _try_get(delta, "hot", "indices")
        if isinstance(idxs, (list, tuple)) and idxs:
            out = np.zeros(N, dtype=float)
            for x in idxs:
                try:
                    i = int(x)
                    if 0 <= i < N:
                        out[i] = 1.0
                except Exception:
                    pass
            return out
        return None
    arr = np.asarray(list(roi))
    if arr.ndim == 1 and arr.size == N and arr.dtype.kind in "fcui":
        out = np.clip(arr.astype(float, copy=False), 0.0, 1.0)
        return out
    out = np.zeros(N, dtype=float)
    for x in arr.reshape(-1):
        try:
            i = int(x)
            if 0 <= i < N:
                out[i] = 1.0
        except Exception:
            pass
    return out

def project_delta_to_mosaic(delta_report: Mapping[str, Any]) -> "object":
    """
    Rzutuje elastycznie różne warianty delta_report -> AnalysisMosaic.
    """
    core = _coerce_core(delta_report)
    rows = cols = 0
    try:
        xs = sorted({int(c["center"][0]) for c in core.get("cells", [])})
        ys = sorted({int(c["center"][1]) for c in core.get("cells", [])})
        cols = len(xs); rows = len(ys)
    except Exception:
        rows = int(_try_get(delta_report, "grid", "rows", default=0) or 0)
        cols = int(_try_get(delta_report, "grid", "cols", default=0) or 0)
    N = max(0, rows * cols)

    edges_vec = _coerce_edges_per_cell(delta_report, N)
    block_stats = _parse_block_stats(
        _try_get(delta_report, "block_stats")
        or _try_get(delta_report, "blocks", "stats")
        or _try_get(delta_report, "mosaic", "block_stats")
    )
    roi_mask = _coerce_roi(delta_report, N)

    return core_to_analysis_mosaic(
        core,
        edges_per_cell=edges_vec,
        block_stats=(block_stats if edges_vec is None else None),
        metric_name="edges",
        roi_mask=roi_mask,
        kind=None,
    )
