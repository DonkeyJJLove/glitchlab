# glitchlab/core/utils.py
# -*- coding: utf-8 -*-
"""
Podstawowe narzędzia backendu:
- Ctx: kontekst przetwarzania (maski, amplitude, RNG, meta, cache)
- Normalizacja obrazu (uint8 RGB), konwersja do szarości
- Mapy krawędzi + budowa maski krawędzi (z opcjonalną dylacją)
- Generatory amplitudy (linear_x/y, radial, perlin, z maski)
- Pomocnicze resize/clip i RNG

Uwaga: brak SciPy – do dylacji używamy Pillow (ImageFilter.MaxFilter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple

import numpy as np
from PIL import Image, ImageFilter

# opcjonalny Perlin (pnoise2); jeśli brak, mamy fallback sinusowy
try:
    import noise as _noise  # pip install noise
except Exception:
    _noise = None


# =============================================================================
# Kontekst
# =============================================================================

@dataclass
class Ctx:
    """Kontekst przetwarzania jednego pipeline’u."""
    masks: Dict[str, np.ndarray] = field(default_factory=dict)   # str -> (H,W) float32 [0,1]
    amplitude: Optional[np.ndarray] = None                       # (H,W) float32 (dowolna skala dodatnia)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(7))
    seed: int = 7
    meta: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)

    def reseed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            seed = self.seed
        else:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)


# =============================================================================
# Konwersje/normalizacje
# =============================================================================

def normalize_image(a: np.ndarray) -> np.ndarray:
    """
    Zwraca obraz w formacie (H,W,3) uint8.
    Obsługa wejść: L/LA/RGB/RGBA/float/int.
    """
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.ndim == 3 and a.shape[2] == 4:  # RGBA -> RGB na białym tle
        alpha = a[..., 3:4].astype(np.float32) / 255.0
        rgb = a[..., :3].astype(np.float32)
        a = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            scale = 255.0 if a.max() <= 1.0 else 1.0
            a = np.clip(a * scale, 0, 255).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("normalize_image: oczekiwano (H,W,3) po normalizacji")
    return np.ascontiguousarray(a)


def to_gray(arr: np.ndarray) -> np.ndarray:
    """Konwersja do szarości float32 (0..255)."""
    arr = normalize_image(arr)
    g = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return g.astype(np.float32)


# =============================================================================
# Maski / krawędzie
# =============================================================================

def compute_edges(arr: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """
    Prosty gradient (różnice sąsiednie) → mapa krawędzi float32 0..1.
    """
    g = to_gray(arr) / 255.0
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:] = np.abs(g[:, 1:] - g[:, :-1])
    gy[1:, :] = np.abs(g[1:, :] - g[:-1, :])
    e = np.clip((gx + gy) * gain, 0.0, 1.0)
    if (e.max() - e.min()) > 1e-8:
        e = (e - e.min()) / (e.max() - e.min())
    return e.astype(np.float32)


def _dilate_mask_pillow(mask: np.ndarray, ksize: int, iters: int) -> np.ndarray:
    """
    Dylacja binarna 2D przez Pillow (MaxFilter). mask musi być float 0..1.
    """
    if iters <= 0 or ksize <= 1:
        return mask
    if ksize % 2 == 0:
        ksize += 1  # MaxFilter wymaga nieparzystego
    im = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    for _ in range(iters):
        im = im.filter(ImageFilter.MaxFilter(ksize))
    out = np.asarray(im, dtype=np.uint8).astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)


def build_edge_mask(arr: np.ndarray, thresh: float = 60, dilate: int = 0, ksize: int = 3) -> np.ndarray:
    """
    Buduje maskę krawędzi: progowanie gradientu + opcjonalna dylacja.
    - thresh: próg w skali 0..255 (po zeskalowaniu gradientu do 0..255)
    - dilate: liczba iteracji dylacji
    - ksize : rozmiar okna MaxFilter (nieparzysty)
    """
    e = compute_edges(arr, gain=1.2)  # 0..1
    m = (e * 255.0 >= float(thresh)).astype(np.float32)
    if dilate > 0:
        m = _dilate_mask_pillow(m, ksize=max(3, int(ksize)), iters=int(dilate))
    return m  # 0..1 float32


def resize_mask_to(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    """
    Skalowanie maski do rozmiaru (H,W) przez Pillow (BICUBIC).
    """
    H, W = shape_hw
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L")
    im = im.resize((W, H), Image.BICUBIC)
    return (np.asarray(im, dtype=np.uint8).astype(np.float32) / 255.0).clip(0, 1)


# =============================================================================
# Amplitudy (pola sterujące)
# =============================================================================

def _perlin2d(width: int, height: int, scale: float, octaves: int = 4,
              persistence: float = 0.5, lacunarity: float = 2.0, base: int = 0) -> np.ndarray:
    """
    2D Perlin (z `noise`) albo fallback sinusowy, gdy brak pakietu.
    Zwraca float32 0..1.
    """
    if _noise is None or scale <= 0:
        # fallback: łagodna siatka sinus/cos
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        f = 2.0 * np.pi / max(16.0, scale or 16.0)
        z = np.sin(xx * f) * np.cos(yy * f)
        z = (z - z.min()) / (z.max() - z.min() + 1e-8)
        return z.astype(np.float32)

    z = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            z[y, x] = _noise.pnoise2(
                x / scale, y / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024, repeaty=1024,
                base=base,
            )
    z = (z - z.min()) / (z.max() - z.min() + 1e-8)
    return z.astype(np.float32)


def make_amplitude(shape_hw: Tuple[int, int],
                   kind: str = "none",
                   strength: float = 1.0,
                   ctx: Optional[Ctx] = None,
                   **kwargs) -> np.ndarray:
    """
    Buduje pole amplitudy (H,W) float32 dodatnie.
    kind:
      - 'none'      : jedynki
      - 'linear_x'  : rampa w osi X (0..1)
      - 'linear_y'  : rampa w osi Y (0..1)
      - 'radial'    : kołowo od środka (0..1)
      - 'perlin'    : szum perlin (wymaga pip noise; jest fallback)
      - 'mask'      : z maski ctx.masks[mask_key] (param: mask_key)
    strength: mnożnik końcowy (>0)
    kwargs:
      - dla perlin: scale, octaves, persistence, lacunarity, base
      - dla mask : mask_key
    """
    H, W = shape_hw
    if strength <= 0:
        strength = 1.0

    if kind == "none":
        amp = np.ones((H, W), dtype=np.float32)

    elif kind == "linear_x":
        x = np.linspace(0.0, 1.0, W, dtype=np.float32)
        amp = np.tile(x[None, :], (H, 1))

    elif kind == "linear_y":
        y = np.linspace(0.0, 1.0, H, dtype=np.float32)
        amp = np.tile(y[:, None], (1, W))

    elif kind == "radial":
        yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
        r = np.sqrt(xx * xx + yy * yy)
        r = (r - r.min()) / (r.max() - r.min() + 1e-8)
        amp = 1.0 - r  # większa amplituda w centrum

    elif kind == "perlin":
        scale = float(kwargs.get("scale", 96.0))
        octaves = int(kwargs.get("octaves", 4))
        persistence = float(kwargs.get("persistence", 0.5))
        lacunarity = float(kwargs.get("lacunarity", 2.0))
        base = int(kwargs.get("base", 0 if ctx is None else ctx.seed))
        amp = _perlin2d(W, H, scale, octaves, persistence, lacunarity, base)

    elif kind == "mask":
        mask_key = kwargs.get("mask_key")
        if ctx is None or not isinstance(mask_key, str) or mask_key not in ctx.masks:
            raise KeyError("make_amplitude(kind='mask'): wymagany ctx.masks[mask_key]")
        amp = resize_mask_to(ctx.masks[mask_key].astype(np.float32), (H, W))

    else:
        raise ValueError(f"make_amplitude: nieznany kind='{kind}'")

    # skala dodatnia
    amp = np.clip(amp, 0.0, None).astype(np.float32)
    if (amp.max() - amp.min()) > 1e-8:
        amp = (amp - amp.min()) / (amp.max() - amp.min())
    amp = np.clip(amp * float(strength), 1e-6, None)
    return amp


# =============================================================================
# RNG i pomocnicze
# =============================================================================

def build_rng(seed: int = 7) -> np.random.Generator:
    """Deterministyczny generator liczb losowych."""
    return np.random.default_rng(int(seed))


def set_seed(ctx: Ctx, seed: int) -> None:
    ctx.reseed(seed)


__all__ = [
    "Ctx",
    "normalize_image",
    "to_gray",
    "compute_edges",
    "build_edge_mask",
    "resize_mask_to",
    "make_amplitude",
    "build_rng",
    "set_seed",
]
