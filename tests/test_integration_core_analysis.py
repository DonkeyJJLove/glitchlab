# tests/test_integration_core_analysis.py
# -*- coding: utf-8 -*-
"""
Smoke/integration dla „core ⇄ analysis”:
- adapter core_to_analysis_mosaic: spójność wymiarów i zakresów,
- mosaic_profile: I1 (α+β≈1), zakresy H,Z oraz stabilność,
- exporters.export_mosaic_overlay: overlay RGB zgodny rozmiarem i typem.

Uruchamianie:
    pytest -q
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Importy z łagodnym skipem (repo może być w trakcie wdrożeń etapami)
# ──────────────────────────────────────────────────────────────────────────────

core_ok = analysis_ok = exporter_ok = False

try:
    from glitchlab.core.mosaic import mosaic_map as core_mosaic_map  # CoreMosaic (dict)
    core_ok = True
except Exception:  # pragma: no cover
    core_ok = False

try:
    # AnalysisMosaic (dataclass) + profil
    from glitchlab.gui.mosaic.hybrid_ast_mosaic import Mosaic as AnalysisMosaic
    from glitchlab.gui.mosaic.hybrid_ast_mosaic import mosaic_profile
    analysis_ok = True
except Exception:  # pragma: no cover
    analysis_ok = False

try:
    # Adapter core → analysis
    from glitchlab.analysis.mosaic_adapter import core_to_analysis_mosaic  # type: ignore
    analysis_ok = analysis_ok and True
except Exception:  # pragma: no cover
    core_to_analysis_mosaic = None  # type: ignore
    analysis_ok = False

try:
    from glitchlab.analysis.exporters import export_mosaic_overlay  # type: ignore
    exporter_ok = True
except Exception:  # pragma: no cover
    export_mosaic_overlay = None  # type: ignore
    exporter_ok = False


@pytest.mark.skipif(not (core_ok and analysis_ok), reason="Warstwy core/analysis jeszcze niekompletne")
class TestCoreAnalysisAdapter:
    """Testy adaptera oraz profilu analitycznego mozaiki."""

    def _build_core(self, H: int = 96, W: int = 128, cell_px: int = 16) -> Dict[str, Any]:
        # CoreMosaic (dict)
        core = core_mosaic_map((H, W), mode="square", cell_px=cell_px)
        assert isinstance(core, dict)
        assert "cells" in core and "raster" in core and "size" in core
        return core

    def _edges_per_cell(self, core: Dict[str, Any]) -> List[float]:
        """Deterministyczny gradient [0..1] po komórkach (kolumnami)."""
        cells: List[Dict[str, Any]] = core["cells"]
        # Oszacuj nx, ny z pól komórek (id = j*nx+i) → wyznaczamy nx z max id + 1 / liczby wierszy
        # Lepsze: skorzystaj z metadanych square grid
        nx_ny = self._grid_dims_from_core(core)
        nx, ny = nx_ny
        vals: List[float] = []
        for j in range(ny):
            for i in range(nx):
                # prosty, powtarzalny profil kolumnowy
                v = 0.1 + 0.8 * (i / max(1, nx - 1))
                vals.append(float(np.clip(v, 0.0, 1.0)))
        # Przycięcie do realnej liczby komórek (ostatni rząd/kolumna może być mniejszy)
        return vals[: len(cells)]

    def _grid_dims_from_core(self, core: Dict[str, Any]) -> Tuple[int, int]:
        """Wyznacza (nx, ny) z core; używa układu id = j*nx+i."""
        cells: List[Dict[str, Any]] = core["cells"]
        if not cells:
            return (0, 0)
        # Szukamy największego id w pierwszym wierszu (y stałe) — ale prościej:
        # Inferuj przez bounding w pikselach + cell_px (bardziej odporne).
        cell_px = int(core.get("cell_px", 16))
        H, W = core.get("size", (0, 0))
        nx = int(math.ceil(W / float(cell_px))) if cell_px > 0 else 0
        ny = int(math.ceil(H / float(cell_px))) if cell_px > 0 else 0
        if nx * ny < len(cells):
            # fallback: spróbuj odgadnąć z max id
            max_id = max(int(c["id"]) for c in cells)
            # heurystyka: najbliższy prostokąt >= max_id+1
            area = max_id + 1
            # preferuj nx bliskie sqrt(area) z uwzględnieniem proporcji W/H
            asp = (W + 1e-6) / (H + 1e-6) if H and W else 1.0
            nx_guess = max(1, int(round(math.sqrt(area * asp))))
            ny_guess = max(1, int(math.ceil(area / nx_guess)))
            nx, ny = nx_guess, ny_guess
        return (nx, ny)

    def test_core_to_analysis_mosaic_basic(self):
        core = self._build_core(H=96, W=128, cell_px=16)
        edges = self._edges_per_cell(core)

        A = core_to_analysis_mosaic(core, edges_per_cell=edges, roi_mask=None, kind="grid")  # type: ignore
        assert isinstance(A, AnalysisMosaic)
        N = A.rows * A.cols
        assert N >= 1
        assert A.edge.shape == (N,)
        # Zakres [0..1]
        assert float(np.min(A.edge)) >= 0.0 - 1e-9
        assert float(np.max(A.edge)) <= 1.0 + 1e-9
        # SSIM i ROI też zgodne wymiarem
        assert A.ssim.shape == (N,)
        assert A.roi.shape == (N,)

    def test_mosaic_profile_invariants(self):
        core = self._build_core(H=96, W=128, cell_px=16)
        edges = self._edges_per_cell(core)
        A = core_to_analysis_mosaic(core, edges_per_cell=edges, roi_mask=None, kind="grid")  # type: ignore

        S, H, Z, a, b = mosaic_profile(A, thr=0.55)
        # Zakresy i inwarianty
        assert S == A.rows + A.cols
        assert 0 <= H <= (A.rows * A.cols)
        assert Z >= 0
        assert abs((a + b) - 1.0) < 1e-9  # I1: α+β≈1

        # Determinizm: powtórne wywołanie z tą samą mozaiką daje to samo
        S2, H2, Z2, a2, b2 = mosaic_profile(A, thr=0.55)
        assert (S, H, Z) == (S2, H2, Z2)
        assert abs(a - a2) < 1e-12 and abs(b - b2) < 1e-12


@pytest.mark.skipif(not (core_ok and exporter_ok), reason="Eksporter lub core niedostępny")
class TestExportersOverlay:
    """Testy overlay na podstawie block_stats → core overlay."""

    def _build_core(self, H: int = 120, W: int = 160, cell_px: int = 16) -> Dict[str, Any]:
        core = core_mosaic_map((H, W), mode="square", cell_px=cell_px)
        return core

    def _grid_dims_from_core(self, core: Dict[str, Any]) -> Tuple[int, int]:
        cell_px = int(core.get("cell_px", 16))
        H, W = core.get("size", (0, 0))
        nx = int(math.ceil(W / float(cell_px))) if cell_px > 0 else 0
        ny = int(math.ceil(H / float(cell_px))) if cell_px > 0 else 0
        return (nx, ny)

    def _make_block_stats(self, core: Dict[str, Any]) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Buduje stabilny block_stats (bx=nx, by=ny) z gradientem."""
        nx, ny = self._grid_dims_from_core(core)
        stats: Dict[Tuple[int, int], Dict[str, float]] = {}
        for bj in range(ny):
            for bi in range(nx):
                t = (bi / max(1, nx - 1)) if nx > 1 else 0.0
                u = (bj / max(1, ny - 1)) if ny > 1 else 0.0
                # deterministyczne kanały
                entropy = 0.2 + 0.7 * t
                edges = 0.15 + 0.8 * (0.5 * t + 0.5 * u)
                mean = 0.3 + 0.4 * (1.0 - u)
                stats[(bi, bj)] = {"entropy": float(np.clip(entropy, 0.0, 1.0)),
                                   "edges": float(np.clip(edges, 0.0, 1.0)),
                                   "mean": float(np.clip(mean, 0.0, 1.0))}
        return stats

    def test_export_mosaic_overlay_rgb(self):
        core = self._build_core(H=120, W=160, cell_px=16)
        H, W = core.get("size", (0, 0))
        img = np.zeros((H, W, 3), dtype=np.uint8)  # czarne tło
        block_stats = self._make_block_stats(core)

        ov = export_mosaic_overlay(img, core, block_stats)  # type: ignore
        assert isinstance(ov, np.ndarray)
        assert ov.dtype == np.uint8
        assert ov.shape == img.shape
        # powinno coś pokolorować (nie wszystko zero)
        assert int(ov.sum()) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Dodatkowy smoke: CLI parser istnieje i akceptuje podstawowe flagi (bez uruchamiania GIT)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not analysis_ok, reason="Warstwa analysis niedostępna")
def test_build_cli_parses_minimal():
    from glitchlab.mosaic.hybrid_ast_mosaic import build_cli
    p = build_cli()
    # parsowanie trybu run (bez wykonywania funkcji cmd)
    args = p.parse_args(["--mosaic", "grid", "--rows", "4", "--cols", "5", "run"])
    assert args.mosaic == "grid"
    assert args.rows == 4
    assert args.cols == 5
