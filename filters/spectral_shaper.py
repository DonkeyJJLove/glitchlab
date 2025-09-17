# glitchlab/filters/spectral_shaper.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Any, Dict, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

# Rejestr filtrów
try:
    from glitchlab.core.registry import register
except Exception as e:  # pragma: no cover
    def register(_name: str):
        def deco(f): return f
        return deco


def _to_float_img(arr: np.ndarray) -> np.ndarray:
    """uint8[H,W,(3|4)] -> float32[H,W,3] w zakresie 0..1"""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.shape[-1] == 4:
        # zrób premultiply (bezpieczniej do FFT)
        rgb = a[..., :3].astype(np.float32)
        alpha = (a[..., 3:4].astype(np.float32) / 255.0)
        rgb = np.where(alpha > 0, rgb * alpha, rgb)
        a = rgb
    return (a.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    a = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.shape[-1] != 3:
        a = a[..., :3]
    return a


def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    # gładkie przejście (0..1)
    t = np.clip((x - edge0) / max(1e-6, (edge1 - edge0)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _make_radial_mask(h: int, w: int) -> np.ndarray:
    yc = (h - 1) / 2.0
    xc = (w - 1) / 2.0
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    ry = (y - yc) / max(1.0, yc)
    rx = (x - xc) / max(1.0, xc)
    r = np.sqrt(rx * rx + ry * ry)  # 0..~1.4 (rog)
    return r


def _make_angle(h: int, w: int) -> np.ndarray:
    yc = (h - 1) / 2.0
    xc = (w - 1) / 2.0
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    ang = np.degrees(np.arctan2(y - yc, x - xc))  # -180..180
    return ang


def _build_spectral_mask(h: int, w: int, mode: str,
                         low: float, high: float,
                         angle_deg: float, ang_width: float,
                         soft: float) -> np.ndarray:
    r = _make_radial_mask(h, w)  # radius
    m_rad = np.zeros((h, w), dtype=np.float32)

    s = max(0.0, float(soft))
    # pasmo radialne
    if mode == "ring" or mode == "bandpass" or mode == "direction":
        # bandpass-like
        edge = max(1e-4, 0.5 * s)
        lo_in, lo_out = low - edge, low + edge
        hi_out, hi_in = high - edge, high + edge
        m_low = _smoothstep(r, lo_in, lo_out)
        m_high = 1.0 - _smoothstep(r, hi_in, hi_out)
        m_rad = np.clip(m_low * m_high, 0.0, 1.0)
    elif mode == "bandstop":
        edge = max(1e-4, 0.5 * s)
        lo_in, lo_out = low - edge, low + edge
        hi_out, hi_in = high - edge, high + edge
        m_low = _smoothstep(r, lo_in, lo_out)
        m_high = 1.0 - _smoothstep(r, hi_in, hi_out)
        m_bp = np.clip(m_low * m_high, 0.0, 1.0)
        m_rad = 1.0 - m_bp
    else:
        m_rad = np.ones((h, w), dtype=np.float32)

    if mode == "direction":
        ang = _make_angle(h, w)
        # okno kierunkowe ± ang_width/2 wokół angle_deg
        d = np.abs(((ang - angle_deg + 180.0) % 360.0) - 180.0)  # 0..180
        half = max(1.0, ang_width * 0.5)
        edge = max(0.1, soft * 15.0)  # miękkie przejście kątowe
        m_dir = 1.0 - _smoothstep(d, half, half + edge)
        m = np.clip(m_rad * m_dir, 0.0, 1.0)
    else:
        m = m_rad

    # wyzeruj DC (środek) – unikamy offsetu jasności
    m[h // 2, w // 2] = 0.0
    return m.astype(np.float32)


def _apply_fft_boost(img_f: np.ndarray, mask: np.ndarray, boost: float) -> np.ndarray:
    """
    img_f: float32 [H,W,3] 0..1
    mask: float32 [H,W]    0..1 (w przestrzeni częstotliwości)
    """
    h, w, c = img_f.shape
    out = np.empty_like(img_f)
    eps = 1e-6
    # mnożnik w dziedzinie częstotliwości: 1 + boost*mask
    mult = 1.0 + float(boost) * mask
    for ch in range(c):
        F = np.fft.fftshift(np.fft.fft2(img_f[..., ch]))
        F2 = F * mult
        res = np.fft.ifft2(np.fft.ifftshift(F2))
        out[..., ch] = np.real(res)
    out = np.clip(out, 0.0, 1.0)
    return out


def _apply_spatial_mask(src: np.ndarray, dst: np.ndarray, mask_u8: Optional[np.ndarray]) -> np.ndarray:
    """
    Jeśli jest maska przestrzenna (0..255), zrób lokalny miks dst/src.
    """
    if mask_u8 is None:
        return dst
    m = np.asarray(mask_u8).astype(np.float32) / 255.0
    if m.ndim == 2:
        m = m[..., None]
    m = np.clip(m, 0.0, 1.0)
    return src * (1.0 - m) + dst * m


@register("spectral_shaper")
def spectral_shaper(arr: Any, ctx: Any = None, params: Optional[Dict[str, Any]] = None, **kw: Any) -> Any:
    """
    Filtr częstotliwościowy:
      mode:        ring|bandpass|bandstop|direction
      low, high:   granice pasma (0..~1.4, sensownie 0..1.0)
      angle_deg:   kąt centralny dla 'direction'
      ang_width:   szerokość kątowa dla 'direction' (stopnie)
      boost:       siła modyfikacji (np. 0.8)
      soft:        zmiękczenie krawędzi masek (0..1)
      blend:       miks z oryginałem w dziedzinie przestrzennej (0..1)
      mask_key:    nazwa maski z ctx.masks (jeśli dostępna)
      use_amp:     skaler (jeśli chcesz modulować przez "amplitude")
      clamp:       jeżeli True – klamruj wynik 0..1
    """
    if np is None:
        raise RuntimeError("NumPy is required by 'spectral_shaper'.")

    p = dict(params or {})
    p.update(kw or {})

    mode = str(p.get("mode", "ring")).lower()
    low = float(p.get("low", 0.15))
    high = float(p.get("high", 0.45))
    angle_deg = float(p.get("angle_deg", 0.0))
    ang_width = float(p.get("ang_width", 20.0))
    boost = float(p.get("boost", 0.8))
    soft = float(p.get("soft", 0.08))
    blend = float(p.get("blend", 0.0))
    mask_key = p.get("mask_key", None)
    use_amp = float(p.get("use_amp", 1.0))
    clamp = bool(p.get("clamp", True))

    # amplituda globalna (jeśli ctx ma)
    try:
        amp = float(getattr(getattr(ctx, "cfg", None), "get", lambda *_: 1.0)("amplitude", {}).get("strength", 1.0))
    except Exception:
        amp = 1.0
    boost = boost * max(0.0, use_amp) * float(amp)

    img_f = _to_float_img(arr)
    h, w, _ = img_f.shape

    # maska częstotliwościowa
    m_spec = _build_spectral_mask(h, w, mode, low, high, angle_deg, ang_width, soft)

    # przetwarzanie
    out_f = _apply_fft_boost(img_f, m_spec, boost)

    # miks w przestrzeni
    out_f = (1.0 - blend) * img_f + blend * out_f

    # opcjonalny miks przez maskę przestrzenną z ctx
    mask_u8 = None
    if mask_key:
        try:
            if hasattr(ctx, "masks") and mask_key in ctx.masks:
                mask_u8 = ctx.masks[mask_key]
        except Exception:
            pass
    out_f = _apply_spatial_mask(img_f, out_f, mask_u8)

    if clamp:
        out_f = np.clip(out_f, 0.0, 1.0)

    return _to_uint8(out_f)
