import numpy as np
from glitchlab.core.registry import register

try:
    import noise  # opcjonalnie: pip install noise
except ImportError:
    noise = None


def _to_rgb_uint8(a: np.ndarray) -> np.ndarray:
    """Ujednolica tablicę do (H,W,3) uint8 (z RGBA/grayscale/float/int)."""
    a = np.asarray(a)
    if a.ndim == 2:  # grayscale -> RGB
        a = np.stack([a, a, a], axis=-1)
    if a.ndim == 3 and a.shape[2] == 4:  # RGBA -> RGB (premultiply do białego tła)
        alpha = a[..., 3:4].astype(np.float32) / 255.0
        rgb = a[..., :3].astype(np.float32)
        a = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


@register("depth_displace")
def depth_displace(
    img,
    ctx,
    depth_map="noise_fractal",
    scale=40,
    octaves=4,
    freq=70.0,
    stereo=True,
    shading=True,
):
    """
    Pseudo-3D displacement (2D) z cieniowaniem i lekkim stereo RGB (anaglif).
    Zwraca zawsze (H,W,3) uint8.
    """
    img = _to_rgb_uint8(img)
    h, w, c = img.shape

    # --- MAPA GŁĘBI ---
    if depth_map == "noise_fractal" and noise is not None:
        depth = np.empty((h, w), dtype=np.float32)
        base = getattr(ctx, "seed", 0)
        for y in range(h):
            for x in range(w):
                depth[y, x] = noise.pnoise2(
                    x / freq, y / freq, octaves=octaves,
                    repeatx=w, repeaty=h, base=base
                )
    else:
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        depth = 0.5 * (np.sin(xx / 25.0) + np.cos(yy / 35.0))

    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # --- 2D pola przesunięć ---
    dx = np.round((depth - 0.5) * 2.0 * scale).astype(np.int32)     # [-scale, +scale]
    dy = np.round((0.5 - depth) * 0.5 * scale).astype(np.int32)     # delikatne Y

    X = np.tile(np.arange(w, dtype=np.int32), (h, 1))
    Y = np.tile(np.arange(h, dtype=np.int32).reshape(-1, 1), (1, w))

    src_x = np.clip(X + dx, 0, w - 1)
    src_y = np.clip(Y + dy, 0, h - 1)

    out = np.empty_like(img)
    for ch in range(c):
        out[:, :, ch] = img[src_y, src_x, ch]

    # shading z gradientu głębi (prosty lambert)
    if shading:
        gy, gx = np.gradient(depth.astype(np.float32))
        grad = np.sqrt(gx * gx + gy * gy)
        shade = 0.7 + 0.3 * (1.0 - np.clip(2.0 * grad, 0.0, 1.0))  # 0.7..1.0
        out = np.clip(out.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)

    if stereo and out.shape[2] >= 3:
        out[:, :, 0] = np.roll(out[:, :, 0], +2, axis=1)  # R
        out[:, :, 2] = np.roll(out[:, :, 2], -2, axis=1)  # B

    return out
