# glitchlab/analysis/metrics.py
"""
---
version: 2
kind: module
id: "analysis-metrics"
created_at: "2025-09-11"
name: "glitchlab.analysis.metrics"
author: "GlitchLab v2"
role: "Image & Quality Metrics Library"
description: >
  Zestaw szybkich metryk obrazu (entropy, edge_density, contrast_rms, block_stats)
  oraz opcjonalne narzędzia do normalizacji i łączenia sygnałów jakości wykorzystywanych przez
  metasoczewki/fields. Nowe API *nie zmienia* dotychczasowego kontraktu ani wyników funkcji.
inputs:
  arr: {dtype: "uint8|float32", shape: "(H,W)|(H,W,3)", colorspace: "RGB/Gray"}
  params:
    block: {type: int, default: 16}
    max_side: {type: int, default: 1024}
outputs:
  global:
    compute_entropy: {type: float, units: "bits"}
    edge_density: {type: float, range: "[0,1]"}
    contrast_rms: {type: float, range: "[0,1]"}
  blocks:
    block_stats: {type: dict[(bx,by)->{entropy:float,edges:float,mean:float,variance:float}]}
quality_api:
  utilities:
    normalize_array_01(x, robust=True, q_lo=5, q_hi=95) -> np.ndarray[0..1]
    normalize_field_map(field_map, robust=True, q_lo=5, q_hi=95) -> dict[node->0..1]
    blend_fields(field_maps, weights=None, normalize_each=True, robust=True) -> dict[node->0..1]
    nan_safe(value_or_array) -> value_or_array_without_nans
policy:
  deterministic: true
  side_effects: false
constraints:
  - "no SciPy/OpenCV"
  - "clamp/NaN-safe wyniki"
license: "Proprietary"
---
"""
from __future__ import annotations

from typing import Dict, Tuple, Mapping, Iterable, Optional
import numpy as np
from PIL import Image

__all__ = [
    # istniejące API (nie zmieniamy)
    "to_gray_f32",
    "downsample_max_side",
    "compute_entropy",
    "edge_density",
    "contrast_rms",
    "block_stats",
    # nowe, opcjonalne narzędzia do jakości
    "nan_safe",
    "normalize_array_01",
    "normalize_field_map",
    "blend_fields",
]

# -----------------------------
# Konwersje i pomocnicze
# -----------------------------

def to_gray_f32(arr: np.ndarray) -> np.ndarray:
    """
    uint8 (H,W[,3]) lub float (H,W[,3]) -> gray float32 [0,1]
    """
    if arr.ndim not in (2, 3):
        raise ValueError("to_gray_f32: expected 2D gray or 3D RGB")
    if arr.dtype == np.uint8:
        a = arr.astype(np.float32) / 255.0
    else:
        a = arr.astype(np.float32, copy=False)

    if a.ndim == 3:
        if a.shape[-1] != 3:
            raise ValueError("to_gray_f32: for 3D inputs, last dim must be 3 (RGB)")
        g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        return np.clip(g, 0.0, 1.0, out=g)
    return np.clip(a, 0.0, 1.0, out=a)


def downsample_max_side(arr: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """
    Jeśli potrzeba, skaluje obraz tak, by max(H,W) == max_side (bicubic, Pillow).
    Zachowuje dtype (uint8/float32). Działa dla Gray i RGB.
    """
    H, W = arr.shape[:2]
    m = max(H, W)
    if m <= max_side:
        return arr
    scale = max_side / float(m)
    new_size = (max(1, int(round(W * scale))), max(1, int(round(H * scale))))

    if arr.dtype != np.uint8:
        # konwertujemy tymczasowo do uint8, potem z powrotem do float32 [0,1]
        tmp = np.clip(arr, 0.0, 1.0) if arr.dtype != np.uint8 else arr
        if tmp.dtype != np.uint8:
            tmp = (tmp * 255.0 + 0.5).astype(np.uint8)
        mode = "L" if arr.ndim == 2 else "RGB"
        im = Image.fromarray(tmp, mode=mode).resize(new_size, resample=Image.BICUBIC)
        out = np.asarray(im, dtype=np.float32) / 255.0
        return out
    else:
        mode = "L" if arr.ndim == 2 else "RGB"
        im = Image.fromarray(arr, mode=mode).resize(new_size, resample=Image.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


# -----------------------------
# Metryki globalne
# -----------------------------

def compute_entropy(arr: np.ndarray, bins: int = 256) -> float:
    """
    Shannon entropy (bits) na gray [0,1] (histogram z 'bins' koszami).
    """
    g = to_gray_f32(arr)
    # histogram w [0,1]
    hist, _ = np.histogram(g, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    # unikamy log(0)
    p = p[p > 0]
    H = -np.sum(p * (np.log2(p)))
    # maksymalna entropia = log2(bins); nie normalizujemy aby zachować skale "bitową"
    return float(np.clip(H, 0.0, np.log2(bins)))


def edge_density(arr: np.ndarray) -> float:
    """
    Średnia gęstość krawędzi: E[|∇x| + |∇y|] na gray [0,1].
    """
    g = to_gray_f32(arr)
    gx = np.zeros_like(g, dtype=np.float32)
    gy = np.zeros_like(g, dtype=np.float32)
    gx[:, :-1] = g[:, 1:] - g[:, :-1]
    gy[:-1, :] = g[1:, :] - g[:-1, :]
    mag = np.abs(gx) + np.abs(gy)
    # opcjonalne docięcie, by mieściło się ~[0,1]
    return float(np.clip(mag.mean(), 0.0, 1.0))


def contrast_rms(arr: np.ndarray) -> float:
    """
    RMS kontrast: sqrt(mean((gray - mean(gray))^2)) – zakres ~[0,1].
    """
    g = to_gray_f32(arr)
    mu = float(g.mean()) if g.size > 0 else 0.0
    d = g - mu
    rms = float(np.sqrt(np.mean(d * d))) if g.size > 0 else 0.0
    return float(np.clip(rms, 0.0, 1.0))


# -----------------------------
# Metryki blokowe (dla mozaiki)
# -----------------------------

def block_stats(
    arr: np.ndarray,
    block: int = 16,
    max_side: int = 1024,
    bins: int = 64,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Statystyki kafelkowe: entropy/edges/mean/variance dla bloków block×block.
    Dla przyspieszenia najpierw downsample do max_side.
    Zwraca mapę {(bx,by) -> {...}} gdzie bx,by to indeksy w siatce bloków.
    """
    if block < 4:
        raise ValueError("block must be >= 4")

    # downsample (zachowujemy typ wyjścia z downsample: u8 lub f32)
    small = downsample_max_side(arr, max_side=max_side)
    g = to_gray_f32(small)
    H, W = g.shape
    out: Dict[Tuple[int, int], Dict[str, float]] = {}

    # liczba bloków
    bx_count = int(np.ceil(W / float(block)))
    by_count = int(np.ceil(H / float(block)))

    # prealokacje do histogramu
    hist_bins = bins
    for by in range(by_count):
        y0 = by * block
        y1 = min(H, y0 + block)
        for bx in range(bx_count):
            x0 = bx * block
            x1 = min(W, x0 + block)
            tile = g[y0:y1, x0:x1]
            if tile.size == 0:
                continue

            # mean/variance
            m = float(tile.mean())
            v = float(tile.var())

            # entropy
            h_hist, _ = np.histogram(tile, bins=hist_bins, range=(0.0, 1.0))
            p = h_hist.astype(np.float64)
            s = p.sum()
            if s > 0:
                p /= s
                p = p[p > 0]
                H_bits = -np.sum(p * np.log2(p))
                H_bits = float(np.clip(H_bits, 0.0, np.log2(hist_bins)))
            else:
                H_bits = 0.0

            # edges
            gx = np.zeros_like(tile)
            gy = np.zeros_like(tile)
            gx[:, :-1] = tile[:, 1:] - tile[:, :-1]
            gy[:-1, :] = tile[1:, :] - tile[:-1, :]
            ed = float(np.clip((np.abs(gx) + np.abs(gy)).mean(), 0.0, 1.0))

            out[(bx, by)] = {
                "entropy": H_bits,
                "edges": ed,
                "mean": float(np.clip(m, 0.0, 1.0)),
                "variance": float(np.clip(v, 0.0, 1.0)),
            }

    return out


# ============================================================
# Nowe: opcjonalne narzędzia jakości (normalizacja i blending)
# ============================================================

def nan_safe(x):
    """
    Zastępuje NaN/Inf bezpiecznymi wartościami. Działa dla skalarów i ndarray.
    - NaN -> 0.0
    - +Inf -> 1.0
    - -Inf -> 0.0
    """
    if isinstance(x, (int, float)):
        if np.isnan(x):  # type: ignore[arg-type]
            return 0.0
        if x == float("inf"):
            return 1.0
        if x == float("-inf"):
            return 0.0
        return float(x)
    arr = np.asarray(x, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr


def _robust_minmax_params(arr: np.ndarray, q_lo: float = 5.0, q_hi: float = 95.0) -> Tuple[float, float]:
    """
    Zwraca (lo, hi) na bazie kwantyli — odporne na outliery.
    Gdy hi<=lo, poszerza przedział minimalnie, by uniknąć dzielenia przez zero.
    """
    if arr.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(arr, max(0.0, min(100.0, q_lo))))
    hi = float(np.percentile(arr, max(0.0, min(100.0, q_hi))))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = 1.0
    if hi <= lo:
        hi = lo + 1e-9
    return (lo, hi)


def normalize_array_01(
    x: Iterable[float] | np.ndarray,
    *,
    robust: bool = True,
    q_lo: float = 5.0,
    q_hi: float = 95.0,
) -> np.ndarray:
    """
    Normalizacja do [0,1]:
      - jeśli robust=True: używa kwantyli (q_lo,q_hi), następnie przycina do [0,1],
      - jeśli robust=False: używa min/max z całej próbki,
      - zawsze NaN-safe.
    """
    arr = nan_safe(x).astype(np.float64)
    if arr.size == 0:
        return arr
    if robust:
        lo, hi = _robust_minmax_params(arr, q_lo=q_lo, q_hi=q_hi)
    else:
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1e-9
    y = (arr - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0, out=y)


def normalize_field_map(
    field_map: Mapping[str, float],
    *,
    robust: bool = True,
    q_lo: float = 5.0,
    q_hi: float = 95.0,
) -> Dict[str, float]:
    """
    Normalizuje mapę {node_id -> value} do [0,1] (NaN-safe, Inf-safe).
    Używa normalize_array_01 na wartościach i odtwarza mapę.
    """
    if not field_map:
        return {}
    keys = list(field_map.keys())
    vals = np.array([field_map[k] for k in keys], dtype=np.float64)
    norm = normalize_array_01(vals, robust=robust, q_lo=q_lo, q_hi=q_hi)
    return {k: float(v) for k, v in zip(keys, norm.tolist())}


def blend_fields(
    field_maps: Mapping[str, Mapping[str, float]],
    weights: Optional[Mapping[str, float]] = None,
    *,
    normalize_each: bool = True,
    robust: bool = True,
    q_lo: float = 5.0,
    q_hi: float = 95.0,
) -> Dict[str, float]:
    """
    Łączy wiele pól w jeden wynik {node_id->score in [0,1]}:
      - dopasowuje klucze po unii node_id,
      - opcjonalnie normalizuje każde pole do [0,1] (robust kwantylami),
      - stosuje wagi (domyślnie równe),
      - wynik to ważona średnia, NaN/Inf-safe i *zawsze* przycięta do [0,1].

    Przykład:
      blend_fields(
        {"pagerank": pr_map, "degree": deg_map, "churn": ch_map},
        weights={"pagerank": 0.5, "degree": 0.3, "churn": 0.2}
      )
    """
    if not field_maps:
        return {}

    # zbuduj unijny zbiór węzłów
    all_nodes: set[str] = set()
    for m in field_maps.values():
        all_nodes.update(m.keys())

    # domyślne wagi
    if weights is None:
        weights = {name: 1.0 for name in field_maps.keys()}
    # normalizacja wag na wypadek dowolnych skal
    w_vec = np.array([max(0.0, float(weights.get(name, 0.0))) for name in field_maps.keys()], dtype=np.float64)
    w_sum = float(np.sum(w_vec))
    if w_sum <= 0:
        # gdy same zera → równomierne
        w_vec = np.ones_like(w_vec)
        w_sum = float(np.sum(w_vec))
    w_norm = (w_vec / w_sum).tolist()

    # (opcjonalnie) przeskaluj każde pole do [0,1]
    field_names = list(field_maps.keys())
    processed: Dict[str, Dict[str, float]] = {}
    for name in field_names:
        m = field_maps[name]
        processed[name] = normalize_field_map(m, robust=robust, q_lo=q_lo, q_hi=q_hi) if normalize_each else {k: float(nan_safe(v)) for k, v in m.items()}

    # składanie: ważona średnia; brakujące wartości → 0.0
    out: Dict[str, float] = {}
    for nid in all_nodes:
        acc = 0.0
        for j, name in enumerate(field_names):
            v = float(processed[name].get(nid, 0.0))
            acc += w_norm[j] * v
        out[nid] = float(np.clip(acc, 0.0, 1.0))
    return out
