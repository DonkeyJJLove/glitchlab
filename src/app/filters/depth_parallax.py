# -*- coding: utf-8 -*-
"""
depth_displace — pseudo-3D przesunięcie paralaksy sterowane mapą "głębi".
Wspiera: maski ROI (mask_key), amplitude (use_amp), stereo (anaglyph),
cieniowanie "height shading" oraz diagnostykę do HUD.

Diag (HUD):
  diag/depth_displace/depth    – mapa głębi (gray)
  diag/depth_displace/dx, /dy  – znormalizowane przemieszczenia
  diag/depth_displace/shade    – faktor cieniowania (jeśli shading)
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional

try:
    from glitchlab.core.registry import register  # normal
except Exception:  # pragma: no cover
    from core.registry import register  # type: ignore

# (opcjonalny) perlin
try:
    import noise as _noise  # pip install noise
except Exception:
    _noise = None

DOC = "Pseudo-3D parallax using a depth field; supports mask, amplitude, stereo and shading."
DEFAULTS: Dict[str, Any] = {
    "depth_map": "noise_fractal",  # noise_fractal|perlin|sine
    "scale": 56.0,  # px – siła przesunięcia (poziomo)
    "freq": 110.0,  # częstotliwość dla generatora głębi
    "octaves": 5,  # oktawy (dla perlin)
    "vertical": 0.15,  # udział pionowego przesunięcia (0..1)
    "stereo": True,  # anaglyph R/B offset
    "stereo_px": 2,  # px różnicy kanałów R/B
    "shading": True,  # cieniowanie na gradiencie głębi
    "shade_gain": 0.25,  # 0..1
    "mask_key": None,  # ROI
    "use_amp": 1.0,  # float|bool – wpływ ctx.amplitude
    "clamp": True,  # końcowe przycięcie do u8
}


# ----------------------- helpers (samowystarczalne) -----------------------

def _to_u8(f32: np.ndarray) -> np.ndarray:
    x = np.clip(f32, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _fit_hw(m: np.ndarray, H: int, W: int) -> np.ndarray:
    mh, mw = m.shape[:2]
    out = np.zeros((H, W), dtype=np.float32)
    h = min(H, mh);
    w = min(W, mw)
    out[:h, :w] = m[:h, :w].astype(np.float32)
    if h < H: out[h:, :w] = out[h - 1:h, :w]
    if w < W: out[:H, w:] = out[:H, w - 1:w]
    return out


def _resolve_mask(ctx, mask_key: Optional[str], H: int, W: int) -> Optional[np.ndarray]:
    if not mask_key or not getattr(ctx, "masks", None):
        return None
    m = ctx.masks.get(mask_key)
    if m is None:
        return None
    m = np.asarray(m, dtype=np.float32)
    if m.shape != (H, W):
        m = _fit_hw(m, H, W)
    return np.clip(m, 0.0, 1.0)


def _amplitude_map(ctx, H: int, W: int, use_amp) -> np.ndarray:
    if not hasattr(ctx, "amplitude") or ctx.amplitude is None:
        return np.ones((H, W), dtype=np.float32)
    amp = np.asarray(ctx.amplitude, dtype=np.float32)
    if amp.shape != (H, W):
        amp = _fit_hw(amp, H, W)
    amp -= amp.min()
    amp /= (amp.max() + 1e-12)
    base = 0.25 + 0.75 * amp
    if isinstance(use_amp, bool):
        return base if use_amp else np.ones((H, W), dtype=np.float32)
    return base * float(max(0.0, use_amp))


def _depth_field(h: int, w: int, kind: str, freq: float, octaves: int, seed: int) -> np.ndarray:
    kind = (kind or "noise_fractal").lower()
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    if kind in ("noise_fractal", "perlin") and _noise is not None:
        scale = max(8.0, float(freq))
        z = np.zeros((h, w), dtype=np.float32)
        fx = fy = 1.0 / scale
        for j in range(h):
            yv = j * fy
            for i in range(w):
                xv = i * fx
                z[j, i] = _noise.pnoise2(
                    xv, yv,
                    octaves=int(max(1, octaves)),
                    persistence=0.5, lacunarity=2.0,
                    repeatx=1024, repeaty=1024, base=int(seed),
                )
        z -= z.min()
        z /= (z.max() + 1e-12)
        return z.astype(np.float32)

    # fallback: sin/cos pole
    f = 2.0 * np.pi / max(16.0, float(freq) or 16.0)
    z = 0.5 * (np.sin(xx * f) + np.cos(yy * f))
    z -= z.min()
    z /= (z.max() + 1e-12)
    return z.astype(np.float32)


def _emit_diag(ctx, depth: np.ndarray, dx: np.ndarray, dy: np.ndarray, shade: Optional[np.ndarray]) -> None:
    try:
        ctx.cache["diag/depth_displace/depth"] = _to_u8(np.stack([depth, depth, depth], axis=-1))
        # dx,dy w 0..1 (0.5 = brak ruchu)
        dxn = 0.5 + 0.5 * np.tanh(dx / (np.abs(dx).mean() + 1e-6))
        dyn = 0.5 + 0.5 * np.tanh(dy / (np.abs(dy).mean() + 1e-6))
        ctx.cache["diag/depth_displace/dx"] = _to_u8(np.stack([dxn, dxn, dxn], axis=-1))
        ctx.cache["diag/depth_displace/dy"] = _to_u8(np.stack([dyn, dyn, dyn], axis=-1))
        if shade is not None:
            s3 = np.clip(shade.astype(np.float32), 0.0, 1.0)
            if s3.ndim == 2: s3 = s3[..., None]
            s3 = np.repeat(s3, 3, axis=-1)
            ctx.cache["diag/depth_displace/shade"] = _to_u8(s3)
    except Exception:
        pass


# -------------------------------- main --------------------------------

@register("depth_displace", defaults=DEFAULTS, doc=DOC)
def depth_displace(img: np.ndarray, ctx, **p) -> np.ndarray:
    """
    Parallax przesunięcie wg mapy głębi. Zwraca u8 RGB.
    """
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("depth_displace: expected RGB-like image (H,W,C>=3)")
    H, W, _ = a.shape

    depth_map = str(p.get("depth_map", DEFAULTS["depth_map"]))
    scale = float(p.get("scale", DEFAULTS["scale"]))
    freq = float(p.get("freq", DEFAULTS["freq"]))
    octaves = int(p.get("octaves", DEFAULTS["octaves"]))
    vertical = float(p.get("vertical", DEFAULTS["vertical"]))
    stereo = bool(p.get("stereo", DEFAULTS["stereo"]))
    stereo_px = int(p.get("stereo_px", DEFAULTS["stereo_px"]))
    shading = bool(p.get("shading", DEFAULTS["shading"]))
    shade_gain = float(p.get("shade_gain", DEFAULTS["shade_gain"]))
    mask_key = p.get("mask_key", DEFAULTS["mask_key"])
    use_amp = p.get("use_amp", DEFAULTS["use_amp"])
    clamp = bool(p.get("clamp", DEFAULTS["clamp"]))

    seed = int(getattr(ctx, "seed", 7))
    depth = _depth_field(H, W, depth_map, freq=freq, octaves=octaves, seed=seed)

    amp_map = _amplitude_map(ctx, H, W, use_amp)

    m = _resolve_mask(ctx, mask_key, H, W)
    if m is None:
        m = np.ones((H, W), dtype=np.float32)

    # przemieszczenia
    s = float(scale)
    dx = (depth - 0.5) * 2.0 * s * amp_map * m
    dy = (depth - 0.5) * 2.0 * (s * float(max(0.0, vertical))) * amp_map * m

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xs = np.clip((xx + np.rint(dx)).astype(np.int32), 0, W - 1)
    ys = np.clip((yy + np.rint(dy)).astype(np.int32), 0, H - 1)

    out = np.empty_like(a)
    # G zawsze środkowy
    out[..., 1] = a[ys, xs, 1]

    if stereo:
        xs_r = np.clip(xs + int(abs(stereo_px)), 0, W - 1)
        xs_b = np.clip(xs - int(abs(stereo_px)), 0, W - 1)
        out[..., 0] = a[ys, xs_r, 0]  # R
        out[..., 2] = a[ys, xs_b, 2]  # B
    else:
        out[..., 0] = a[ys, xs, 0]
        out[..., 2] = a[ys, xs, 2]

    shade = None
    if shading:
        gx = np.zeros_like(depth, dtype=np.float32)
        gy = np.zeros_like(depth, dtype=np.float32)
        gx[:, 1:] = np.abs(depth[:, 1:] - depth[:, :-1])
        gy[1:, :] = np.abs(depth[1:, :] - depth[:-1, :])
        grad = np.clip(gx + gy, 0.0, None)
        if (grad.max() - grad.min()) > 1e-8:
            grad = (grad - grad.min()) / (grad.max() + 1e-12)
        k = float(np.clip(shade_gain, 0.0, 1.0))
        shade = (1.0 - k * grad)  # 2D
        out = np.clip(out.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)

    _emit_diag(ctx, depth, dx, dy, shade)

    if clamp:
        return out
    # bez clamp – zachowujemy raw (tu i tak u8)
    return out
