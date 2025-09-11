
# -*- coding: utf-8 -*-
"""
HUD (diagnostic viewer)
-----------------------
Robust, crash-proof renderer for ctx.cache diagnostics.
- Accepts grayscale or RGB, uint8 or float32 in [0..1].
- Accepts ndarray with shape (H,W), (H,W,1), (H,W,3), (H,W,4).
- Gracefully skips unsupported entries and shows a placeholder instead of crashing.
- Can be called with either a full ctx object or a raw cache dict.

Public API expected by the app:
    Hud(parent)
    .render_from_cache(ctx_or_cache_or_None)
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Iterable, Tuple, Optional

import numpy as np
from PIL import Image, ImageTk, ImageDraw

THUMB_W = 256
THUMB_H = 256
PAD = 8

class Hud(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        # simple scrollable canvas with an inner frame to put thumbnails in a grid
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#202020")
        self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vscroll.set)
        self.inner = ttk.Frame(self.canvas)

        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vscroll.pack(side="right", fill="y")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self._imgs: list[ImageTk.PhotoImage] = []  # hold refs

        # Initial message
        self._show_empty()

    # --------------- layout helpers ---------------
    def _on_inner_configure(self, event) -> None:
        # update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        # keep inner width equal to canvas width
        cw = event.width
        self.canvas.itemconfigure(self.canvas_window, width=cw)

    # --------------- public API ---------------
    def render_from_cache(self, ctx_or_cache: Optional[Any]) -> None:
        """Render thumbnails from ctx.cache or a raw dict."""
        # clear previous
        for child in self.inner.winfo_children():
            child.destroy()
        self._imgs.clear()

        if ctx_or_cache is None:
            self._show_empty("(brak danych)")
            return

        cache: Dict[str, Any]
        amp = None
        masks: Dict[str, np.ndarray] = {}

        # accept either a ctx or a dict-like
        if hasattr(ctx_or_cache, "cache"):
            cache = getattr(ctx_or_cache, "cache", {}) or {}
            amp = getattr(ctx_or_cache, "amplitude", None)
            masks = getattr(ctx_or_cache, "masks", {}) or {}
        elif isinstance(ctx_or_cache, dict):
            cache = ctx_or_cache
        else:
            cache = {}

        # Compose entries to show: amplitude & masks first, then diagnostic keys.
        entries: list[Tuple[str, Any]] = []

        if isinstance(amp, np.ndarray):
            entries.append(("amplitude", amp))

        if isinstance(masks, dict):
            for k, v in sorted(masks.items()):
                entries.append((f"mask:{k}", v))

        # Prefer diag/*, then stage/0/* thumbnails, finally anything that looks like an image
        diag_items = [(k, v) for k, v in cache.items() if isinstance(k, str) and k.startswith(("diag/", "spectral_", "phase_", "pixel_sort/trigger"))]
        stage_items = [(k, v) for k, v in cache.items() if isinstance(k, str) and k.startswith("stage/")]
        other_items = [(k, v) for k, v in cache.items() if (k, v) not in diag_items and (k, v) not in stage_items]

        # sort for stability
        for arr in (diag_items, stage_items, other_items):
            arr.sort(key=lambda kv: kv[0])

        entries.extend(diag_items + stage_items + other_items)

        if not entries:
            self._show_empty("(brak map diagnostycznych)")
            return

        # Render grid (2 columns on narrow, 3+ columns on wide)
        self._render_grid(entries)

    # --------------- rendering ---------------
    def _render_grid(self, entries: Iterable[Tuple[str, Any]]) -> None:
        # Decide number of columns based on current width
        try:
            width = max(1, int(self.winfo_width()))
        except Exception:
            width = 800
        cols = max(2, width // (THUMB_W + 2*PAD))

        r = c = 0
        for key, val in entries:
            try:
                im = self._to_image(val)
                imt = self._thumb(im, (THUMB_W, THUMB_H))
                p = ImageTk.PhotoImage(imt)
            except Exception as e:
                # fallback: draw an error tile
                imt = self._error_tile(str(e))
                p = ImageTk.PhotoImage(imt)
            self._imgs.append(p)  # keep ref

            frm = ttk.Frame(self.inner, padding=PAD)
            lab = ttk.Label(frm, image=p, background="#202020")
            cap = ttk.Label(frm, text=key, foreground="#c8c8c8")
            lab.pack()
            cap.pack(anchor="w")
            frm.grid(row=r, column=c, sticky="nw")
            c += 1
            if c >= cols:
                c = 0
                r += 1

    def _thumb(self, im: Image.Image, box: Tuple[int, int]) -> Image.Image:
        w, h = im.size
        bw, bh = map(int, box)
        scale = min(bw / max(1, w), bh / max(1, h))
        nw, nh = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
        return im.resize((nw, nh), Image.Resampling.LANCZOS)

    def _to_image(self, val: Any) -> Image.Image:
        # Already a PIL image
        if isinstance(val, Image.Image):
            return val.convert("RGB")

        # NumPy array
        if isinstance(val, np.ndarray):
            arr = val
            # handle booleans
            if arr.dtype == np.bool_:
                arr = arr.astype(np.uint8) * 255

            # float → normalize to [0..255]
            if arr.dtype.kind in "fc":
                # try to detect if already in 0..255 or 0..1
                a = np.asarray(arr, dtype=np.float32)
                a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
                # First normalize to 0..1 if out of range
                amin, amax = float(np.min(a)), float(np.max(a))
                if amax - amin > 1e-12:
                    a = (a - amin) / (amax - amin)
                a = np.clip(a, 0.0, 1.0)
                arr = (a * 255.0 + 0.5).astype(np.uint8)

            # int dtypes other than u8 → cast
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8, copy=False)

            # shape handling
            if arr.ndim == 2:
                # grayscale → RGB
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3:
                if arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                elif arr.shape[2] >= 3:
                    arr = arr[..., :3]
                else:
                    # unexpected last dim
                    arr = np.repeat(arr, 3, axis=2) if arr.shape[2] == 0 else arr
            else:
                # flatten best effort
                flat = arr.ravel()
                n = (flat.size // 3) * 3
                flat = flat[:n]
                if n == 0:
                    return self._error_tile("(pusta macierz)")
                h = max(1, int(np.sqrt(n // 3)))
                w = max(1, (n // 3) // h)
                arr = flat.reshape((h, w, 3))

            return Image.fromarray(arr, "RGB")

        # Fallback: render value as text on tile
        return self._text_tile(str(val))

    def _text_tile(self, text: str) -> Image.Image:
        im = Image.new("RGB", (THUMB_W, THUMB_H), (32, 32, 32))
        d = ImageDraw.Draw(im)
        d.text((10, 10), text, fill=(230, 230, 230))
        return im

    def _error_tile(self, msg: str) -> Image.Image:
        im = Image.new("RGB", (THUMB_W, THUMB_H), (48, 24, 24))
        d = ImageDraw.Draw(im)
        d.text((10, 10), "HUD error", fill=(255, 240, 240))
        d.text((10, 30), msg[:120], fill=(240, 200, 200))
        return im

    def _show_empty(self, msg: str = "(HUD idle)") -> None:
        for child in self.inner.winfo_children():
            child.destroy()
        self._imgs.clear()
        lab = ttk.Label(self.inner, text=msg, padding=PAD)
        lab.grid(row=0, column=0, sticky="nw")
