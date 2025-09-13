
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from typing import Any, Optional


def _to_rgb_image(value: Any) -> Optional[Image.Image]:
    """Accepts np arrays (H,W), (H,W,1), (H,W,3), uint8/float; strings -> None (text slot)."""
    try:
        if isinstance(value, Image.Image):
            im = value
        elif isinstance(value, np.ndarray):
            arr = value
            if arr.ndim == 2:
                # grayscale -> RGB
                a = arr.astype(np.float32)
                if a.max() <= 1.001: a = a * 255.0
                a = np.clip(a, 0, 255).astype(np.uint8)
                rgb = np.stack([a, a, a], axis=-1)
                im = Image.fromarray(rgb, "RGB")
            elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                a = arr.astype(np.float32)
                if a.max() <= 1.001: a = a * 255.0
                a = np.clip(a, 0, 255).astype(np.uint8)
                if a.shape[-1] == 1:
                    a = np.repeat(a, 3, axis=-1)
                if a.shape[-1] == 4:
                    a = a[..., :3]
                im = Image.fromarray(a, "RGB")
            else:
                return None
        elif isinstance(value, (bytes, bytearray)):
            try:
                from io import BytesIO
                im = Image.open(BytesIO(value)).convert("RGB")
            except Exception:
                return None
        else:
            return None
        return im
    except Exception:
        return None


class _HudSlot(ttk.Frame):
    def __init__(self, parent, title: str):
        super().__init__(parent, padding=4)
        self.var_title = tk.StringVar(value=title)
        self.var_text = tk.StringVar(value="")
        self.label = ttk.Label(self, textvariable=self.var_title, font=("", 10, "bold"))
        self.label.pack(anchor="w")
        self.canvas = tk.Label(self, bd=1, relief="sunken")  # for image
        self.canvas.pack(fill="both", expand=True, pady=(2, 0))
        self.text = ttk.Label(self, textvariable=self.var_text, foreground="#444", justify="left", anchor="w")
        self.text.pack(fill="x", pady=(2, 0))
        self._img_ref = None  # keep PhotoImage alive

    def set_title(self, text: str):
        self.var_title.set(text)

    def set_text(self, text: str):
        self.var_text.set(text)

    def set_image(self, value: Any, max_size=(256, 256)):
        im = _to_rgb_image(value)
        if im is None:
            self._img_ref = None
            self.canvas.configure(image="")
            return
        im2 = im.copy()
        im2.thumbnail(max_size, Image.BICUBIC)
        tkimg = ImageTk.PhotoImage(im2)
        self.canvas.configure(image=tkimg)
        self._img_ref = tkimg


class Hud(ttk.Frame):
    """
    3-slotowy HUD dla diagnostyki.
    Oczekuje obrazów/map w ctx.cache pod przewidywalnymi kluczami, ale jest odporny na braki.
    """
    def __init__(self, parent):
        super().__init__(parent)
        grid = ttk.Frame(self); grid.pack(fill="both", expand=True)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(2, weight=1)
        self.s1 = _HudSlot(grid, "Slot 1"); self.s1.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.s2 = _HudSlot(grid, "Slot 2"); self.s2.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        self.s3 = _HudSlot(grid, "Slot 3"); self.s3.grid(row=0, column=2, sticky="nsew", padx=4, pady=4)

    def render_from_cache(self, ctx):
        cache = getattr(ctx, "cache", {}) or {}

        # Priorytet: amplitude, edge mask, fft/mag
        picked1 = cache.get("cfg/amplitude/vis") or cache.get("amplitude/vis") or cache.get("stage/0/fft_mag") \
                  or cache.get("spectral_shaper/mag")
        picked2 = cache.get("edge_mask/vis") or cache.get("pixel_sort/trigger")
        picked3 = cache.get("spectral_shaper/mask") or cache.get("phase_glitch/noise")

        # Titles
        self.s1.set_title("Amplitude / FFT")
        self.s2.set_title("Edges / Trigger")
        self.s3.set_title("Mask / Phase noise")

        # Robust assignment
        if isinstance(picked1, str):
            self.s1.set_text(picked1); self.s1.set_image(None)
        else:
            self.s1.set_text(""); self.s1.set_image(picked1)

        if isinstance(picked2, str):
            self.s2.set_text(picked2); self.s2.set_image(None)
        else:
            self.s2.set_text(""); self.s2.set_image(picked2)

        if isinstance(picked3, str):
            self.s3.set_text(picked3); self.s3.set_image(None)
        else:
            self.s3.set_text(""); self.s3.set_image(picked3)
