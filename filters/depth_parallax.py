import numpy as np
from glitchlab.core.registry import register

try:
    import noise
except ImportError:
    noise = None


@register("depth_parallax")
def depth_parallax(img, ctx, scale=50, freq=80.0, octaves=4, shading=True, stereo=True):
    """
    Stronger pseudo-3D effect with parallax, shading and stereoscopy.

    Args:
        img (np.ndarray): Input image (H, W, C).
        ctx: Context object.
        scale (int): Maximum displacement (pixels).
        freq (float): Noise frequency (lower = larger features).
        octaves (int): Detail in noise.
        shading (bool): Apply depth-based darkening.
        stereo (bool): Shift RGB channels differently.
    """
    h, w, c = img.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # depth map
    if noise is not None:
        depth = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                depth[y, x] = noise.pnoise2(
                    x / freq, y / freq,
                    octaves=octaves,
                    repeatx=w, repeaty=h,
                    base=ctx.seed if hasattr(ctx, "seed") else 0
                )
    else:
        depth = np.sin(xx / 25.0) * np.cos(yy / 25.0)

    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # parallax displacement: warping relative to center
    cx, cy = w // 2, h // 2
    dx = ((xx - cx) / w * depth * scale).astype(int)
    dy = ((yy - cy) / h * depth * scale).astype(int)

    out = np.zeros_like(img)
    for i in range(c):
        for y in range(h):
            src_y = np.clip(y + dy[y, :], 0, h-1)
            src_x = np.clip(np.arange(w) + dx[y, :], 0, w-1)
            out[y, :, i] = img[src_y, src_x, i]

    # shading (depth as shadow)
    if shading:
        out = (out * (0.6 + 0.4 * depth[..., None])).astype(np.uint8)

    # stereoscopic RGB parallax
    if stereo and c >= 3:
        out[:, :, 0] = np.roll(out[:, :, 0], 5, axis=1)   # red shift
        out[:, :, 2] = np.roll(out[:, :, 2], -5, axis=1)  # blue shift

    return out
