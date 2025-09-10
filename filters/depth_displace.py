# -*- coding: utf-8 -*-
"""
depth_displace — pseudo-3D przesunięcie paralaksy sterowane mapą "głębi".
"""

from __future__ import annotations
import numpy as np

from glitchlab.core.registry import register  # standaryzujemy
from glitchlab.core.utils import make_amplitude

try:
    import noise as _noise
except Exception:
    _noise = None


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
                    octaves=int(octaves),
                    persistence=0.5, lacunarity=2.0,
                    repeatx=1024, repeaty=1024, base=int(seed),
                )
        z = (z - z.min()) / (z.max() - z.min() + 1e-8)
        return z.astype(np.float32)

    f = 2.0 * np.pi / max(16.0, float(freq) or 16.0)
    z = 0.5 * (np.sin(xx * f) + np.cos(yy * f))
    z = (z - z.min()) / (z.max() - z.min() + 1e-8)
    return z.astype(np.float32)


def _apply_mask(base: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return base
    m = np.clip(mask, 0.0, 1.0).astype(np.float32)
    return base * m


@register("depth_displace")
def depth_displace(
    img: np.ndarray,
    ctx,
    depth_map: str = "noise_fractal",
    scale: float = 56.0,
    freq: float = 110.0,
    octaves: int = 5,
    vertical: float = 0.15,
    stereo: bool = True,
    stereo_px: int = 2,
    shading: bool = True,
    shade_gain: float = 0.25,
    mask_key: str | None = None,
):
    a = img.astype(np.uint8, copy=False)
    h, w, c = a.shape
    if c != 3:
        raise ValueError("depth_displace: oczekiwano (H,W,3)")

    depth = _depth_field(h, w, depth_map, freq=freq, octaves=int(octaves), seed=getattr(ctx, "seed", 7))

    if getattr(ctx, "amplitude", None) is None:
        ctx.amplitude = make_amplitude((h, w), kind="none", strength=1.0, ctx=ctx)
    amp = ctx.amplitude.astype(np.float32)
    if (amp.max() - amp.min()) > 1e-8:
        amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)

    m = None
    if isinstance(mask_key, str) and hasattr(ctx, "masks") and mask_key in ctx.masks:
        mk = ctx.masks[mask_key].astype(np.float32)
        if mk.shape != (h, w):
            from glitchlab.core.utils import resize_mask_to
            mk = resize_mask_to(mk, (h, w))
        m = np.clip(mk, 0.0, 1.0)

    s = float(scale)
    dx = (depth - 0.5) * 2.0 * s * amp
    dy = (depth - 0.5) * 2.0 * (s * float(vertical)) * amp if vertical > 0 else np.zeros_like(dx)

    if m is not None:
        dx = _apply_mask(dx, m)
        dy = _apply_mask(dy, m)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xs = np.clip((xx + np.rint(dx)).astype(np.int32), 0, w - 1)
    ys = np.clip((yy + np.rint(dy)).astype(np.int32), 0, h - 1)

    out = np.empty_like(a)
    out[..., 1] = a[ys, xs, 1]

    if stereo:
        xs_r = np.clip(xs + int(abs(stereo_px)), 0, w - 1)
        xs_b = np.clip(xs - int(abs(stereo_px)), 0, w - 1)
        out[..., 0] = a[ys, xs_r, 0]
        out[..., 2] = a[ys, xs_b, 2]
    else:
        out[..., 0] = a[ys, xs, 0]
        out[..., 2] = a[ys, xs, 2]

    if shading:
        gx = np.zeros_like(depth, dtype=np.float32)
        gy = np.zeros_like(depth, dtype=np.float32)
        gx[:, 1:] = np.abs(depth[:, 1:] - depth[:, :-1])
        gy[1:, :] = np.abs(depth[1:, :] - depth[:-1, :])
        grad = np.clip(gx + gy, 0.0, None)
        if (grad.max() - grad.min()) > 1e-8:
            grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        k = max(0.0, min(1.0, float(shade_gain)))
        shade = (1.0 - k * grad)[..., None]
        out = np.clip(out.astype(np.float32) * shade, 0, 255).astype(np.uint8)

    return out
