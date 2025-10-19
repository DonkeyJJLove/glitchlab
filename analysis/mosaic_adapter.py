# glitchlab/analysis/mosaic_adapter.py
# Adapter core ↔ analysis: CoreMosaic(dict) → AnalysisMosaic(dataclass)
# Python 3.9+

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# ── Łagodne importy dataclass Mosaic (nowa struktura katalogów) ───────────────
AnalysisMosaic = None  # type: ignore
for _mod in (
    "glitchlab.src.mosaic.hybrid_ast_mosaic",
    "glitchlab.mosaic.hybrid_ast_mosaic",
    "glitchlab.gui.mosaic.hybrid_ast_mosaic",
    "mosaic.hybrid_ast_mosaic",
    "app.mosaic.hybrid_ast_mosaic",
):
    try:
        AnalysisMosaic = __import__(_mod, fromlist=["Mosaic"]).Mosaic  # type: ignore
        break
    except Exception:
        continue

CoreMosaic = Mapping[str, object]  # {"mode","cell_px","size","cells","raster", ...}

__all__ = [
    "grid_dims_from_core",
    "edges_vector_from_block_stats",
    "core_to_analysis_mosaic",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: wymiar siatki i mapowanie bloków → komórki
# ──────────────────────────────────────────────────────────────────────────────

def _unique_sorted(values: Iterable[int]) -> List[int]:
    """Zwraca posortowaną listę unikalnych wartości int (deterministycznie)."""
    return sorted({int(v) for v in values})


def grid_dims_from_core(core: CoreMosaic) -> Tuple[int, int]:
    """
    Wyznacza (rows=ny, cols=nx) siatki komórek na podstawie centrów komórek z core.
    Zgodnie z core.mosaic: id = j*nx + i (row-major), a centra są równomierne w wierszach/kolumnach.
    """
    cells: List[dict] = list(core.get("cells", []))  # type: ignore
    if not cells:
        return (0, 0)
    xs = _unique_sorted(c["center"][0] for c in cells)
    ys = _unique_sorted(c["center"][1] for c in cells)
    nx = len(xs)
    ny = len(ys)
    # sanity – liczba komórek powinna się zgadzać
    if nx * ny != len(cells):
        lab = core.get("raster")
        if isinstance(lab, np.ndarray) and lab.ndim == 2:
            # Szacunek nx,ny po liczbie etykiet i proporcjach obrazu
            n_cells = int(len(cells))
            H, W = lab.shape
            ratio = W / max(1, H)
            nx = int(round(np.sqrt(n_cells * ratio)))
            ny = max(1, n_cells // max(1, nx))
    return (ny, nx)


def _block_grid_dims(block_stats: Mapping[Tuple[int, int], Mapping[str, float]]) -> Tuple[int, int]:
    """Z kluczy (bi,bj) wyznacza rozmiar siatki bloków (bx, by)."""
    if not block_stats:
        return (0, 0)
    xs = [int(i) for (i, _) in block_stats.keys()]
    ys = [int(j) for (_, j) in block_stats.keys()]
    bx = (max(xs) + 1) if xs else 0
    by = (max(ys) + 1) if ys else 0
    return (bx, by)


def edges_vector_from_block_stats(
    core: CoreMosaic,
    block_stats: Mapping[Tuple[int, int], Mapping[str, float]],
    *,
    metric_name: str = "edges",
) -> np.ndarray:
    """
    Buduje wektor edge[rows*cols] ∈ [0,1] na podstawie `block_stats[(bi,bj)][metric_name]`.
    Wartość dla komórki bierzemy z bloku zawierającego jej środek (tak jak w core.mosaic_project_blocks).
    """
    (H, W) = tuple(core.get("size", (0, 0)))  # type: ignore
    cells: List[dict] = list(core.get("cells", []))  # type: ignore
    rows, cols = grid_dims_from_core(core)
    N = rows * cols
    if N == 0 or not cells:
        return np.zeros(0, dtype=float)

    bx, by = _block_grid_dims(block_stats)
    if bx <= 0 or by <= 0:
        return np.zeros(N, dtype=float)

    bw = max(1, int(W) // bx)
    bh = max(1, int(H) // by)

    out = np.zeros(N, dtype=float)
    for c in cells:
        cid: int = int(c["id"])
        cx, cy = map(int, c["center"])
        bi = int(np.clip(cx // bw, 0, bx - 1))
        bj = int(np.clip(cy // bh, 0, by - 1))
        cell_stats = block_stats.get((bi, bj), {})
        val = float(cell_stats.get(metric_name, cell_stats.get("edge", 0.0)))
        out[cid] = val

    # normalizacja do [0,1] i sanity na NaN/Inf
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    if out.size and (out.min() < 0.0 or out.max() > 1.0):
        lo, hi = float(out.min()), float(out.max())
        if hi > lo:
            out = (out - lo) / (hi - lo)
        else:
            out = np.clip(out, 0.0, 1.0)
    return out.astype(float, copy=False)


# ──────────────────────────────────────────────────────────────────────────────
# Główny adapter: CoreMosaic → AnalysisMosaic
# ──────────────────────────────────────────────────────────────────────────────

def _default_roi(rows: int, cols: int) -> np.ndarray:
    """
    Domyślna ROI jak w analysis.build_mosaic_grid: prostokątny środek 30–62% wymiaru.
    """
    roi = np.zeros(rows * cols, dtype=float)
    r0, r1 = int(0.30 * rows), int(0.62 * rows)
    c0, c1 = int(0.30 * cols), int(0.62 * cols)
    for r in range(max(0, r0), min(rows, r1)):
        base = r * cols
        for c in range(max(0, c0), min(cols, c1)):
            roi[base + c] = 1.0
    return roi


def core_to_analysis_mosaic(
    core: CoreMosaic,
    edges_per_cell: Optional[Union[Sequence[float], np.ndarray]] = None,
    *,
    block_stats: Optional[Mapping[Tuple[int, int], Mapping[str, float]]] = None,
    metric_name: str = "edges",
    roi_mask: Optional[Union[Sequence[float], Sequence[int], np.ndarray]] = None,
    kind: Optional[str] = None,
) -> "AnalysisMosaic":
    """
    Konwertuje wynik `glitchlab.core.mosaic.mosaic_map(..)` do `hybrid_ast_mosaic.Mosaic`.

    Priorytet źródła `edge` (w tej kolejności):
      1) `edges_per_cell` (oczekiwana długość rows*cols, wartości [0,1]),
      2) `block_stats[(bi,bj)][metric_name]` odwzorowane przez centra komórek,
      3) wektor zerowy.

    ROI:
      - jeśli `roi_mask` podane: akceptuje listę bool/0..1 lub indeksy kafli,
      - w przeciwnym razie używa domyślnego centralnego prostokąta (30–62%).

    kind:
      - domyślnie wynika z `core['mode']`: "hex" → "hex", inaczej "grid".
    """
    if AnalysisMosaic is None:  # pragma: no cover
        raise RuntimeError("analysis.hybrid_ast_mosaic.Mosaic nie jest dostępny w środowisku importu")

    rows, cols = grid_dims_from_core(core)
    N = rows * cols
    if N <= 0:
        # pusta mozaika – zwróć minimalny obiekt analityczny
        return AnalysisMosaic(rows=0, cols=0,
                              edge=np.zeros(0, dtype=float),
                              ssim=np.zeros(0, dtype=float),
                              roi=np.zeros(0, dtype=float),
                              kind="grid")

    # --- edge: priorytetowane źródła ---
    if edges_per_cell is not None:
        edge = np.asarray(list(edges_per_cell), dtype=float).reshape(-1)
        if edge.size != N:
            raise ValueError(f"edges_per_cell length mismatch: got {edge.size}, expected {N}")
    elif block_stats:
        edge = edges_vector_from_block_stats(core, block_stats, metric_name=metric_name)
        if edge.size != N:
            edge = np.zeros(N, dtype=float)
    else:
        edge = np.zeros(N, dtype=float)

    # sanity: NaN/Inf → clamp do [0,1]
    edge = np.nan_to_num(edge, nan=0.0, posinf=1.0, neginf=0.0)
    edge = np.clip(edge, 0.0, 1.0).astype(float, copy=False)

    # --- ssim: placeholder jedynek (na razie nieużywane w metrykach) ---
    ssim = np.ones(N, dtype=float)

    # --- roi: z maski/indeksów albo domyślna ---
    if roi_mask is None:
        roi = _default_roi(rows, cols)
    else:
        arr = np.asarray(list(roi_mask))
        if arr.ndim == 1 and arr.size == N and arr.dtype.kind in "fcui":
            roi = np.clip(arr.astype(float, copy=False), 0.0, 1.0)
        else:
            roi = np.zeros(N, dtype=float)
            for idx in arr.reshape(-1):
                ii = int(idx)
                if 0 <= ii < N:
                    roi[ii] = 1.0

    # --- kind: z core lub nadpisany parametrem ---
    core_mode = str(core.get("mode", "square") or "square").lower()
    kind_eff = (kind or ("hex" if core_mode == "hex" else "grid")).lower()
    if kind_eff not in ("grid", "hex"):
        kind_eff = "grid"

    return AnalysisMosaic(
        rows=int(rows),
        cols=int(cols),
        edge=edge,
        ssim=ssim,
        roi=roi,
        kind=kind_eff,
        hex_centers=None,
        hex_R=None,
    )
