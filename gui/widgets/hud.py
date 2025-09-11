# -*- coding: utf-8 -*-
"""
Robust HUD widget for GlitchLab
-------------------------------
Renders ONLY image-like entries from `ctx.cache` and ignores scalars/strings/dicts.
- Accepts: NumPy arrays (H,W), (H,W,3|4), (3,H,W), PIL.Image.Image
- Normalizes float/bool/object inputs safely, clamps to uint8 RGB.
- Horizontal scroll strip with thumbnails + key labels.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageTk


THUMB_MAX_W = 260
THUMB_MAX_H = 160
TILE_PAD_X = 8
TILE_PAD_Y = 8


def _is_img_like(v: Any) -> bool:
    # quick prefilter
    if isinstance(v, Image.Image):
        return True
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            return False
        if v.ndim == 2:
            return True
        if v.ndim == 3 and (v.shape[2] in (1, 3, 4) or v.shape[0] in (1, 3, 4)):
            return True
    return False


def _to_rgb_u8(arr: np.ndarray) -> np.ndarray:
    """
    Convert array (H,W[,C]) to RGB uint8 safely.
    - float -> auto scale using min/max (with epsilon)
    - bool -> 0/255
    - 2D -> gray replicated to 3 channels
    - channel-first -> transpose to HWC
    - alpha (4th) channel is dropped
    """
    if arr.dtype == object:
        raise ValueError("object array not supported")
    a = np.asarray(arr)

    # bring to HWC
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[2] not in (1, 3, 4):
        # assume CHW -> HWC
        a = np.transpose(a, (1, 2, 0))

    if a.ndim == 2:
        a = a[:, :, None]

    if a.ndim != 3:
        raise ValueError(f"Unsupported ndarray shape {a.shape}")

    # drop alpha if present
    if a.shape[2] == 4:
        a = a[:, :, :3]
    elif a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.shape[2] != 3:
        # try to coerce to 3
        a = a[:, :, :3] if a.shape[2] > 3 else np.pad(a, ((0,0),(0,0),(0,3-a.shape[2])), mode="edge")

    # dtype handling
    if a.dtype == np.uint8:
        out = a
    elif a.dtype == np.bool_:
        out = a.astype(np.uint8) * 255
    else:
        x = a.astype(np.float32)

        # heuristic: if values are within [0,1.5] treat as [0,1] domain
        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        if np.isfinite(x_min) and np.isfinite(x_max):
            if x_min >= -0.05 and x_max <= 1.5:
                x = np.clip(x, 0.0, 1.0) * 255.0
            else:
                # generic min-max scale
                rng = (x_max - x_min) if (x_max - x_min) > 1e-12 else 1.0
                x = (x - x_min) / rng * 255.0
        else:
            x = np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0)
            x = np.clip(x, 0.0, 1.0) * 255.0
        out = (x + 0.5).astype(np.uint8)

    # ensure contiguous HWC
    return np.ascontiguousarray(out)


def _ensure_pil(v: Any) -> Optional[Image.Image]:
    """Return PIL.Image.Image or None if not renderable."""
    try:
        if isinstance(v, Image.Image):
            im = v.convert("RGB") if v.mode != "RGB" else v
            return im
        if isinstance(v, np.ndarray) and _is_img_like(v):
            rgb = _to_rgb_u8(v)
            return Image.fromarray(rgb, "RGB")
    except Exception:
        return None
    return None


class Hud(tk.Frame):
    def __init__(self, master: Optional[tk.Misc] = None):
        super().__init__(master, bg="#1e1e1e")
        self._photos: List[ImageTk.PhotoImage] = []
        self._tiles: List[tk.Widget] = []

        # scrollable strip
        self.canvas = tk.Canvas(self, bg="#1e1e1e", highlightthickness=0, height=THUMB_MAX_H + 2*TILE_PAD_Y + 20)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.hbar.set)
        self.inner = tk.Frame(self.canvas, bg="#1e1e1e")
        self.window = self.canvas.create_window(0, 0, anchor="nw", window=self.inner)

        self.canvas.pack(fill="both", expand=True, side="top")
        self.hbar.pack(fill="x", side="bottom")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    # ---- public API -------------------------------------------------

    def clear(self) -> None:
        for w in self._tiles:
            try:
                w.destroy()
            except Exception:
                pass
        self._tiles.clear()
        self._photos.clear()

    def render_from_cache(self, ctx) -> None:
        """Render image-like diagnostics from ctx.cache (dict)."""
        self.clear()

        cache = getattr(ctx, "cache", {}) or {}
        # stable order: diagnostic first, then stage, then the rest
        def _sort_key(k: str) -> Tuple[int, str]:
            if k.startswith("diag/"): 
                return (0, k)
            if k.startswith("stage/"):
                return (1, k)
            return (2, k)

        keys = sorted(list(cache.keys()), key=_sort_key)
        x = TILE_PAD_X
        for k in keys:
            v = cache.get(k, None)
            im = _ensure_pil(v)
            if im is None:
                # not an image-like entry: skip silently
                continue

            # thumbnail
            im = im.copy()
            im.thumbnail((THUMB_MAX_W, THUMB_MAX_H), Image.BICUBIC)
            ph = ImageTk.PhotoImage(im)
            self._photos.append(ph)  # keep reference

            # tile
            tile = tk.Frame(self.inner, bg="#2b2b2b")
            lbl = tk.Label(tile, image=ph, bg="#2b2b2b")
            lbl.pack(padx=4, pady=4)

            cap = tk.Label(tile, text=k, fg="#c8c8c8", bg="#2b2b2b", font=("Segoe UI", 8))
            cap.pack(fill="x", padx=4, pady=(0, 6))

            tile.place(x=x, y=TILE_PAD_Y)
            self._tiles.append(tile)

            x += im.width + 2*TILE_PAD_X + 8

        # resize inner width
        iw = max(x, self.canvas.winfo_width())
        ih = THUMB_MAX_H + 2*TILE_PAD_Y + 24
        self.inner.configure(width=iw, height=ih)
        self.canvas.configure(scrollregion=(0, 0, iw, ih))

    # ---- geometry mgmt ----------------------------------------------

    def _on_inner_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox(self.window))

    def _on_canvas_configure(self, event):
        # keep inner height; expand width to canvas but keep scrollable
        self.canvas.itemconfig(self.window, height=event.height)
        # width is set during render
        pass
