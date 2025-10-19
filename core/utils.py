# glitchlab/core/utils.py
"""
---
version: 2
kind: module
id: "core-utils"
created_at: "2025-09-11"
name: "glitchlab.core.utils"
author: "GlitchLab v2"
role: "Core Utilities (image/mask conversions & helpers)"
description: >
  Lekki zestaw funkcji pomocniczych wspólny dla core/filters/GUI: konwersje RGB↔Gray,
  stabilna konwersja do uint8, bezpieczne przycinanie [0,1], zmiana rozmiaru (Pillow),
  deterministyczny RNG i prosty BoxBlur dla masek. Bez SciPy/OpenCV.

inputs:
  image_rgb?: {dtype: "uint8|float32", shape: "(H,W,3)", desc: "obraz RGB"}
  image_gray?: {dtype: "uint8|float32", shape: "(H,W)",   desc: "obraz w odcieniach szarości"}
  size_hw?: {type: "(int,int)", desc: "nowy rozmiar (H,W) dla resize_u8"}
  seed?: {type: "int|null", desc: "ziarno RNG dla make_rng"}
  blur.radius?: {type: "float", min: 0, desc: "promień BoxBlur dla box_blur"}

outputs:
  gray_f32?: {dtype: "float32", shape: "(H,W)", range: "[0,1]"}
  rgb_u8?:   {dtype: "uint8",   shape: "(H,W,3)"}
  resized?:  {dtype: "uint8",   shape: "(H,W[,3])"}
  rng?:      {type: "np.random.Generator"}

interfaces:
  exports:
    - "to_gray_f32_u8"
    - "to_u8_rgb"
    - "resize_u8"
    - "make_rng"
    - "safe_clip01"
    - "box_blur"
  depends_on: ["numpy","Pillow"]
  used_by: ["glitchlab.core.pipeline","glitchlab.core.roi","glitchlab.filters","glitchlab.app"]

contracts:
  - "funkcje są czyste i deterministyczne; nie mutują wejść"
  - "Gray zawsze jako float32 w [0,1]"
  - "to_u8_rgb zwraca uint8 (H,W,3); clamp domyślnie ON"
  - "resize_u8 obsługuje 'nearest'|'bilinear'|'bicubic'"

constraints:
  - "no SciPy/OpenCV"
  - "wejścia muszą mieć poprawny kształt i dtype"

hud:
  note: "Moduł nie zapisuje telemetrii; pośrednio wspiera HUD przez pipeline/analizę."

tests_smoke:
  - "to_gray_f32_u8(np.zeros((16,16,3),uint8)) → (16,16) f32 [0,1]"
  - "to_u8_rgb(np.zeros((8,8,3),float32)) → (8,8,3) uint8"
  - "resize_u8(np.zeros((8,8,3),uint8),(4,4)) → (4,4,3) uint8"
license: "Proprietary"
---
"""

from __future__ import annotations

from typing import Optional, Tuple, Mapping, Any
import numpy as np
from PIL import Image, ImageFilter

__all__ = [
    "to_gray_f32_u8",
    "to_u8_rgb",
    "resize_u8",
    "make_rng",
    "safe_clip01",
    "box_blur",
]


# --------------------------------------------------------------------------------------
# Type helpers
# --------------------------------------------------------------------------------------

def to_gray_f32_u8(img_u8: np.ndarray) -> np.ndarray:
    """Gray f32 [0,1] from uint8 RGB."""
    if img_u8.dtype != np.uint8 or img_u8.ndim != 3 or img_u8.shape[-1] != 3:
        raise ValueError("to_gray_f32_u8: expected uint8 RGB (H,W,3)")
    f = img_u8.astype(np.float32) / 255.0
    g = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return np.clip(g, 0.0, 1.0)


def _sobel_mag_gray01(g: np.ndarray) -> np.ndarray:
    """Sobel magnitude for gray f32 [0,1], kernel 3x3, returns [0,1]."""
    assert g.ndim == 2
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    pad = np.pad(g, 1, mode="edge")
    H, W = g.shape
    gx = np.zeros_like(g, dtype=np.float32)
    gy = np.zeros_like(g, dtype=np.float32)
    for i in range(3):
        for j in range(3):
            sl = pad[i:i + H, j:j + W]
            gx += sl * Kx[i, j]
            gy += sl * Ky[i, j]
    mag = np.sqrt(gx * gx + gy * gy) * (1.0 / 8.0)
    return np.clip(mag, 0.0, 1.0)


def compute_edges(img_u8: np.ndarray, *, ksize: int = 3,
                  thresh: Optional[float] = None,
                  dilate: int = 0) -> np.ndarray:
    """
    Zwraca maskę krawędzi f32 [0,1], ten sam rozmiar co obraz.
    - ksize: obecnie tylko 3 (zachowane dla kompatybilności)
    - thresh: jeśli podano, binarzuje (0/1) wg progu [0..1]
    - dilate: piksele rozszerzania (0 = bez); realizowane MaxFilter (Pillow)
    """
    if img_u8.ndim == 3 and img_u8.shape[-1] == 3 and img_u8.dtype == np.uint8:
        g = to_gray_f32_u8(img_u8)
    elif img_u8.ndim == 2:
        g = np.clip(img_u8.astype(np.float32), 0, 255) / 255.0
    else:
        raise ValueError("compute_edges: expected uint8 RGB or grayscale")
    if ksize != 3:
        # placeholder: trzymamy interfejs, ale realnie używamy 3x3
        pass
    mag = _sobel_mag_gray01(g)

    if thresh is not None:
        m = (mag >= float(thresh)).astype(np.float32)
    else:
        m = mag.astype(np.float32)

    if dilate and dilate > 0:
        # Pillow MaxFilter: rozmiar okna musi być nieparzysty
        win = int(dilate) * 2 + 1
        u8 = (np.clip(m, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        im = Image.fromarray(u8, mode="L").filter(ImageFilter.MaxFilter(win))
        m = np.asarray(im, dtype=np.float32) / 255.0

    return np.clip(m, 0.0, 1.0)


def resize_mask_to(mask_f32: np.ndarray,
                   like_or_hw: Tuple[int, int] | np.ndarray,
                   *, method: str = "bicubic") -> np.ndarray:
    """
    Skaluje maskę f32 [0,1] do (H,W) celu lub rozmiaru obrazu referencyjnego.
    """
    if isinstance(like_or_hw, np.ndarray):
        H, W = like_or_hw.shape[:2]
    else:
        H, W = like_or_hw
    mode = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC}.get(method.lower(), Image.BICUBIC)
    u8 = (np.clip(mask_f32.astype(np.float32), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    im = Image.fromarray(u8, mode="L").resize((W, H), mode)
    out = np.asarray(im, dtype=np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)


def make_amplitude(shape_or_img: Tuple[int, int] | np.ndarray,
                   *, kind: str = "none", strength: float = 1.0,
                   scale: float = 96.0, octaves: int = 3,
                   persistence: float = 0.5, lacunarity: float = 2.0,
                   center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Generuje mapę amplitudy f32 [0,1] (H,W):
      - none:      ones
      - linear_x:  0..1 po osi X
      - linear_y:  0..1 po osi Y
      - radial:    0..1 od centrum (domyślnie środek)
      - perlin:    lekka, bezbiblioteczna "value-noise" z oktawami
    'strength' skaluje wynik (potem i tak filtry mogą mieć use_amp).
    """
    if isinstance(shape_or_img, np.ndarray):
        H, W = shape_or_img.shape[:2]
    else:
        H, W = shape_or_img
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if kind == "none":
        base = np.ones((H, W), np.float32)
    elif kind == "linear_x":
        base = xx / max(W - 1, 1)
    elif kind == "linear_y":
        base = yy / max(H - 1, 1)
    elif kind == "radial":
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        if center is not None:
            cx, cy = float(center[0]), float(center[1])
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        base = r / (np.sqrt(cx * cx + cy * cy) + 1e-6)
        base = np.clip(1.0 - base, 0.0, 1.0)  # 1 w środku, 0 na brzegu
    elif kind == "perlin":
        # szybki "value noise": losowa siatka -> bilinear upsample; nakładamy oktawy
        rng = np.random.default_rng(12345)

        def octave(freq_px: float) -> np.ndarray:
            gH = max(1, int(max(H, 1) / max(freq_px, 1.0)))
            gW = max(1, int(max(W, 1) / max(freq_px, 1.0)))
            grid = rng.random((gH, gW), dtype=np.float32)
            im = Image.fromarray((grid * 255).astype(np.uint8), "L").resize((W, H), Image.BILINEAR)
            return np.asarray(im, np.float32) / 255.0

        amp = 0.0
        total = 0.0
        freq = max(scale, 8.0)
        amp_w = 1.0
        for _ in range(int(max(1, octaves))):
            amp += octave(freq) * amp_w
            total += amp_w
            freq = max(4.0, freq / max(lacunarity, 1.0001))
            amp_w *= float(persistence)
        base = (amp / max(total, 1e-6)).astype(np.float32)
        base = np.clip(base, 0.0, 1.0)
    else:
        base = np.ones((H, W), np.float32)

    out = np.clip(base * float(strength), 0.0, 1.0)
    return out


def to_gray_f32_u8(img_u8: np.ndarray) -> np.ndarray:
    """
    RGB uint8 (H,W,3) -> gray float32 [0,1].
    Jeśli wejście jest float w [0,1] i ma 3 kanały, rzutuje na gray bez skalowania.
    """
    if img_u8.ndim == 2:
        a = img_u8
        if a.dtype == np.uint8:
            g = a.astype(np.float32) / 255.0
        else:
            g = a.astype(np.float32, copy=False)
        return np.clip(g, 0.0, 1.0, out=g)

    if img_u8.ndim != 3 or img_u8.shape[-1] != 3:
        raise ValueError("to_gray_f32_u8: expected (H,W,3) or (H,W) array")

    if img_u8.dtype == np.uint8:
        a = img_u8.astype(np.float32) / 255.0
    else:
        a = img_u8.astype(np.float32, copy=False)

    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return np.clip(y, 0.0, 1.0, out=y)


def to_u8_rgb(img_f: np.ndarray, *, clamp: bool = True) -> np.ndarray:
    """
    float (H,W,3) w ~[0,1] -> uint8 (H,W,3). Jeśli clamp=True, przycina do [0,1].
    Przyjmuje też uint8 i zwraca kopię (stabilizator typu).
    """
    if img_f.ndim != 3 or img_f.shape[-1] != 3:
        raise ValueError("to_u8_rgb: expected (H,W,3)")

    if img_f.dtype == np.uint8:
        return img_f.copy()

    a = img_f.astype(np.float32, copy=False)
    if clamp:
        np.clip(a, 0.0, 1.0, out=a)
    return (a * 255.0 + 0.5).astype(np.uint8)


def resize_u8(img_u8: np.ndarray, size_hw: Tuple[int, int], *, method: str = "bicubic") -> np.ndarray:
    """
    Zmiana rozmiaru RGB/Gray uint8 do (H,W). Metody: 'nearest'|'bilinear'|'bicubic'.
    """
    if img_u8.ndim not in (2, 3):
        raise ValueError("resize_u8: expected 2D gray or 3D RGB array")
    h, w = int(size_hw[0]), int(size_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError("resize_u8: size must be positive")

    if img_u8.ndim == 2:
        mode = "L"
    else:
        if img_u8.shape[-1] != 3:
            raise ValueError("resize_u8: only RGB supported for 3D arrays")
        mode = "RGB"

    method = (method or "bicubic").lower()
    if method == "nearest":
        resample = Image.NEAREST
    elif method == "bilinear":
        resample = Image.BILINEAR
    else:
        resample = Image.BICUBIC

    im = Image.fromarray(img_u8, mode=mode)
    im = im.resize((w, h), resample=resample)
    arr = np.asarray(im, dtype=np.uint8)
    if img_u8.ndim == 2:
        return arr
    # PIL może zwrócić shape (h,w,3) lub (h,w) jeśli obraz degeneruje – upewnij się o 3 kanałach
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def make_rng(seed: Optional[int]) -> np.random.Generator:
    """Deterministyczny RNG (NumPy PCG64) dla podanego ziarna."""
    return np.random.default_rng(seed)


def safe_clip01(a: np.ndarray) -> np.ndarray:
    """Bezpieczne przycięcie do [0,1] z rzutowaniem do float32."""
    f = a.astype(np.float32, copy=False)
    return np.clip(f, 0.0, 1.0, out=f)


def box_blur(mask_f32: np.ndarray, radius: float) -> np.ndarray:
    """
    BoxBlur dla maski float32 [0,1]. Zwraca float32 [0,1].
    Dla radius<=0 zwraca wejście (kopię jeśli potrzeba).
    """
    if radius <= 0:
        return mask_f32.astype(np.float32, copy=True)
    im = Image.fromarray((safe_clip01(mask_f32) * 255.0 + 0.5).astype(np.uint8), mode="L")
    im = im.filter(ImageFilter.BoxBlur(radius=float(radius)))
    out = np.asarray(im, dtype=np.uint8).astype(np.float32) / 255.0
    return safe_clip01(out)
