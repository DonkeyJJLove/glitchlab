# glitchlab/core/metrics/basic.py

from __future__ import annotations

"""
Lekki interfejs metryk w warstwie `core`. Preferuje delegację do
`glitchlab.analysis.metrics`, ale zawiera bezpieczne fallbacki,
aby moduł działał także bez warstwy analysis.
"""

from typing import Dict, Tuple
import numpy as np

try:
    # Preferowany wariant: użyj implementacji z analysis.*
    from backup.analysis import (  # type: ignore
        to_gray_f32 as _to_gray_f32,
        downsample_max_side as _downsample_max_side,
        compute_entropy as _compute_entropy,
        edge_density as _edge_density,
        contrast_rms as _contrast_rms,
        block_stats as _block_stats,
    )
    _USE_ANALYSIS_IMPL = True
except Exception:
    _USE_ANALYSIS_IMPL = False
    # Fallbacki minimalne (bez SciPy/OpenCV)
    from PIL import Image

    def _to_gray_f32(arr: np.ndarray) -> np.ndarray:
        if arr.ndim not in (2, 3):
            raise ValueError("to_gray_f32: expected 2D gray or 3D RGB")
        if arr.dtype == np.uint8:
            a = arr.astype(np.float32) / 255.0
        else:
            a = arr.astype(np.float32, copy=False)
        if a.ndim == 3:
            if a.shape[-1] != 3:
                raise ValueError("to_gray_f32: last dim must be 3 (RGB)")
            g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
            return np.clip(g, 0.0, 1.0, out=g)
        return np.clip(a, 0.0, 1.0, out=a)

    def _downsample_max_side(arr: np.ndarray, max_side: int = 1024) -> np.ndarray:
        if arr.ndim == 3 and arr.shape[-1] == 3:
            mode = "RGB"
        elif arr.ndim == 2:
            mode = "L"
        else:
            raise ValueError("downsample_max_side: expected (H,W) or (H,W,3)")
        H, W = arr.shape[:2]
        m = max(H, W)
        if m <= max_side:
            return arr
        scale = max_side / float(m)
        new_size = (max(1, int(round(W * scale))), max(1, int(round(H * scale))))
        if arr.dtype != np.uint8:
            a = np.clip(arr, 0.0, 1.0)
            if mode == "RGB":
                im = Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), mode=mode)
            else:
                im = Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), mode="L")
        else:
            im = Image.fromarray(arr, mode=mode)
        im = im.resize(new_size, resample=Image.BICUBIC)
        out = np.asarray(im)
        if mode == "L":
            out = out.astype(np.float32) / 255.0
        else:
            out = out.astype(np.uint8)
        return out

    def _compute_entropy(arr: np.ndarray, bins: int = 256) -> float:
        g = _to_gray_f32(arr)
        h, _ = np.histogram(g, bins=bins, range=(0.0, 1.0))
        p = h.astype(np.float64)
        s = p.sum()
        if s <= 0:
            return 0.0
        p /= s
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def _edge_density(arr: np.ndarray) -> float:
        g = _to_gray_f32(arr)
        gx = np.zeros_like(g, dtype=np.float32)
        gy = np.zeros_like(g, dtype=np.float32)
        gx[:, 1:] = g[:, 1:] - g[:, :-1]
        gy[1:, :] = g[1:, :] - g[:-1, :]
        ed = (np.abs(gx) + np.abs(gy)).mean() if g.size else 0.0
        return float(np.clip(ed, 0.0, 1.0))

    def _contrast_rms(arr: np.ndarray) -> float:
        g = _to_gray_f32(arr)
        mu = float(g.mean()) if g.size else 0.0
        rms = float(np.sqrt(np.mean((g - mu) ** 2))) if g.size else 0.0
        return float(np.clip(rms, 0.0, 1.0))

    def _block_stats(
        arr: np.ndarray, block: int = 16, max_side: int = 1024, bins: int = 64
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        g = _to_gray_f32(arr)
        # downsample dla szybkości (dotrzymując proporcji)
        H, W = g.shape
        m = max(H, W)
        if m > max_side:
            scale = max_side / float(m)
            new_size = (max(1, int(round(W * scale))), max(1, int(round(H * scale))))
            im = Image.fromarray((np.clip(g, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L")
            g = np.asarray(im.resize(new_size, resample=Image.BICUBIC), dtype=np.float32) / 255.0
            H, W = g.shape

        bx = int(np.ceil(W / block))
        by = int(np.ceil(H / block))
        out: Dict[Tuple[int, int], Dict[str, float]] = {}
        # szybkie gradienty
        gx = np.zeros_like(g, dtype=np.float32)
        gy = np.zeros_like(g, dtype=np.float32)
        gx[:, 1:] = g[:, 1:] - g[:, :-1]
        gy[1:, :] = g[1:, :] - g[:-1, :]
        abs_edges = np.abs(gx) + np.abs(gy)

        for j in range(by):
            for i in range(bx):
                x0 = i * block
                y0 = j * block
                x1 = min(W, x0 + block)
                y1 = min(H, y0 + block)
                tile = g[y1 - (y1 - y0) : y1, x1 - (x1 - x0) : x1]
                if tile.size == 0:
                    val = {"entropy": 0.0, "edges": 0.0, "mean": 0.0, "variance": 0.0}
                else:
                    h, _ = np.histogram(tile, bins=bins, range=(0.0, 1.0))
                    p = h.astype(np.float64)
                    s = p.sum()
                    if s > 0:
                        p /= s
                        p = p[p > 0]
                        ent = float(-(p * np.log2(p)).sum())
                    else:
                        ent = 0.0
                    edges = float(abs_edges[y0:y1, x0:x1].mean())
                    mean = float(tile.mean())
                    var = float(tile.var())
                    val = {
                        "entropy": ent,
                        "edges": float(np.clip(edges, 0.0, 1.0)),
                        "mean": float(np.clip(mean, 0.0, 1.0)),
                        "variance": float(max(0.0, var)),
                    }
                out[(i, j)] = val
        return out


# Public API (re-exports)
to_gray_f32 = _to_gray_f32
downsample_max_side = _downsample_max_side
compute_entropy = _compute_entropy
edge_density = _edge_density
contrast_rms = _contrast_rms
block_stats = _block_stats

__all__ = [
    "to_gray_f32",
    "downsample_max_side",
    "compute_entropy",
    "edge_density",
    "contrast_rms",
    "block_stats",
]
