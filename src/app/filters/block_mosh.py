# glitchlab/filters/block_mosh.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional, Tuple

try:
    from glitchlab.core.registry import register  # normal
except Exception:  # pragma: no cover
    from core.registry import register  # type: ignore

DOC = (
    "Block Mosh (simple): losowe przesuwanie bloków (roll) z maską i amplitude. "
    "Obsługuje tryb per_channel, wrap/clamp, mix. Diagnostyka: diag/bm_select, diag/bm_dx, diag/bm_dy, diag/bm_alpha."
)

DEFAULTS: Dict[str, Any] = {
    "size": 24,  # px
    "p": 0.33,  # wybór bloku
    "max_shift": 8,  # px
    "per_channel": False,  # True: przesuwa kanały niezależnie
    "wrap": True,  # True: roll; False: clamp w obrębie patcha
    "mix": 1.0,  # 0..1 blend z oryginałem
    "mask_key": None,  # ROI
    "use_amp": 1.0,  # float|bool (0..2 typowo)
    "amp_influence": 1.0,  # skala wpływu amplitude na p/siłę
    "clamp": True,  # końcowe przycięcie do u8
}


def _to_f32(u8: np.ndarray) -> np.ndarray:
    return u8.astype(np.float32) / 255.0


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


def _resolve_mask(ctx, mask_key: Optional[str], H: int, W: int) -> np.ndarray:
    m = None
    if mask_key and getattr(ctx, "masks", None):
        m = ctx.masks.get(mask_key)
    if m is None:
        return np.ones((H, W), dtype=np.float32)
    m = np.asarray(m).astype(np.float32)
    if m.shape != (H, W): m = _fit_hw(m, H, W)
    return np.clip(m, 0.0, 1.0)


def _amplitude_map(ctx, H: int, W: int, use_amp) -> np.ndarray:
    if not hasattr(ctx, "amplitude") or ctx.amplitude is None:
        return np.ones((H, W), dtype=np.float32)
    amp = np.asarray(ctx.amplitude).astype(np.float32)
    if amp.shape != (H, W): amp = _fit_hw(amp, H, W)
    amp -= amp.min();
    amp /= (amp.max() + 1e-12)
    base = 0.25 + 0.75 * amp
    if isinstance(use_amp, bool):
        return base if use_amp else np.ones((H, W), dtype=np.float32)
    return base * float(max(0.0, use_amp))


def _shift_patch_u8(patch: np.ndarray, dx: int, dy: int, wrap: bool) -> np.ndarray:
    # patch u8 (H,W,3)
    if dx == 0 and dy == 0:
        return patch.copy()
    if wrap:
        return np.roll(np.roll(patch, dy, axis=0), dx, axis=1)
    H, W, _ = patch.shape
    out = patch.copy()
    y_src0 = max(0, -dy);
    y_src1 = min(H, H - dy)
    x_src0 = max(0, -dx);
    x_src1 = min(W, W - dx)
    y_dst0 = max(0, dy);
    y_dst1 = min(H, H + dy)
    x_dst0 = max(0, dx);
    x_dst1 = min(W, W + dx)
    if y_src0 < y_src1 and x_src0 < x_src1:
        out[y_dst0:y_dst1, x_dst0:x_dst1, :] = patch[y_src0:y_src1, x_src0:x_src1, :]
    return out


@register("block_mosh", defaults=DEFAULTS, doc=DOC)
def block_mosh(img: np.ndarray, ctx, **p) -> np.ndarray:
    """
    Prosty block-mosh: przesuwa wybrane bloki (roll). Wspiera maskę i amplitude.
    Zapisuje diagnostyki (wybór bloków, dx/dy, alpha miksu).
    """
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("block_mosh: expected RGB-like image (H,W,C>=3)")
    H, W, _ = a.shape
    rng = getattr(ctx, "rng", np.random.default_rng(7))

    size = max(4, int(p.get("size", DEFAULTS["size"])))
    prob = float(p.get("p", DEFAULTS["p"]))
    max_shift = max(0, int(p.get("max_shift", DEFAULTS["max_shift"])))
    per_channel = bool(p.get("per_channel", DEFAULTS["per_channel"]))
    wrap = bool(p.get("wrap", DEFAULTS["wrap"]))
    mix = float(np.clip(p.get("mix", DEFAULTS["mix"]), 0.0, 1.0))
    mask_key = p.get("mask_key", DEFAULTS["mask_key"])
    use_amp = p.get("use_amp", DEFAULTS["use_amp"])
    amp_infl = float(max(0.0, p.get("amp_influence", DEFAULTS["amp_influence"])))
    clamp = bool(p.get("clamp", DEFAULTS["clamp"]))

    mask_map = _resolve_mask(ctx, mask_key, H, W)
    amp_map = _amplitude_map(ctx, H, W, use_amp)

    # diagnostyki
    sel_map = np.zeros((H, W), dtype=np.float32)
    dx_map = np.zeros((H, W), dtype=np.float32) + 0.5
    dy_map = np.zeros((H, W), dtype=np.float32) + 0.5
    alpha_map = np.zeros((H, W), dtype=np.float32)

    out = a.copy()
    bys = list(range(0, H, size))
    bxs = list(range(0, W, size))

    for by in bys:
        y2 = min(H, by + size)
        for bx in bxs:
            x2 = min(W, bx + size)

            m_avg = float(mask_map[by:y2, bx:x2].mean())
            a_avg = float(amp_map[by:y2, bx:x2].mean())
            weight = np.clip(m_avg * (0.25 + 0.75 * a_avg) * amp_infl, 0.0, 2.0)

            p_eff = np.clip(prob * (0.5 + 0.5 * weight), 0.0, 1.0)
            if rng.random() >= p_eff:
                continue

            patch = out[by:y2, bx:x2, :].copy()
            base_patch = patch.copy()

            # losowe przesunięcie (skalowane weightem)
            dx = int(rng.integers(-max_shift, max_shift + 1)) if max_shift > 0 else 0
            dy = int(rng.integers(-max_shift, max_shift + 1)) if max_shift > 0 else 0
            dx = int(round(dx * weight))
            dy = int(round(dy * weight))

            if per_channel:
                # niezależny roll kanałów
                for ch in range(3):
                    patch[..., ch:ch + 1] = _shift_patch_u8(patch[..., ch:ch + 1], dx, dy, wrap)
            else:
                patch = _shift_patch_u8(patch, dx, dy, wrap)

            # alpha/mix
            alpha = np.clip(mix * (0.5 + 0.5 * weight), 0.0, 1.0)
            alpha_map[by:y2, bx:x2] = alpha
            patch_f = _to_f32(patch)
            base_f = _to_f32(base_patch)
            blended = base_f * (1.0 - alpha) + patch_f * alpha
            out[by:y2, bx:x2, :] = _to_u8(blended)

            # diagnostyka
            sel_map[by:y2, bx:x2] = np.maximum(sel_map[by:y2, bx:x2], float(p_eff))
            if max_shift > 0:
                dxx = 0.5 + 0.5 * float(np.clip(dx, -max_shift, max_shift)) / float(max_shift)
                dyy = 0.5 + 0.5 * float(np.clip(dy, -max_shift, max_shift)) / float(max_shift)
            else:
                dxx = dyy = 0.5
            dx_map[by:y2, bx:x2] = dxx
            dy_map[by:y2, bx:x2] = dyy

    # HUD
    if getattr(ctx, "cache", None) is not None:
        def _u(g: np.ndarray) -> np.ndarray:
            g = np.clip(g.astype(np.float32), 0.0, 1.0)
            u = (g * 255.0 + 0.5).astype(np.uint8)
            return np.stack([u, u, u], axis=-1)

        ctx.cache["diag/bm_select"] = _u(sel_map)
        ctx.cache["diag/bm_dx"] = _u(dx_map)
        ctx.cache["diag/bm_dy"] = _u(dy_map)
        ctx.cache["diag/bm_alpha"] = _u(alpha_map)

    return out if not clamp else _to_u8(_to_f32(out))
