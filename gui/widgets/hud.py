# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from PIL import Image, ImageTk

class Hud(ttk.Frame):
    """Lekki HUD na miniatury diagnostyczne pobierane z ctx.cache.
       Zasada: wybierz do 4 najciekawszych wpisów (diag/*, *phase_glitch*, *spectral_shaper*).
    """
    def __init__(self, master):
        super().__init__(master)
        self._items: List[Tuple[str, tk.Label, Optional[ImageTk.PhotoImage]]] = []
        self._grid = ttk.Frame(self); self._grid.pack(fill="both", expand=True)
        self._no = ttk.Label(self, text="Diagnostics (n/a)"); self._no.pack()
        self._thumbs_refs: List[ImageTk.PhotoImage] = []  # keep references

    def render_from_cache(self, ctx: Optional[Any]) -> None:
        for w in self._grid.winfo_children(): w.destroy()
        self._thumbs_refs.clear()
        if ctx is None:
            self._no.configure(text="Diagnostics (n/a)")
            return
        cache: Dict[str, Any] = getattr(ctx, "cache", {}) or {}

        # wybór kandydatów
        keys = []
        for k in cache.keys():
            if isinstance(k, str) and (k.startswith("diag/") or "phase_glitch" in k or "spectral_shaper" in k):
                keys.append(k)
        keys = sorted(keys)[:4]
        if not keys:
            self._no.configure(text="Diagnostics (n/a)")
            return
        else:
            self._no.configure(text="")

        # render
        col = 0
        for k in keys:
            v = cache.get(k)
            if not isinstance(v, np.ndarray): 
                continue
            im = self._to_image(v)
            tkimg = ImageTk.PhotoImage(im)
            lbl = ttk.Label(self._grid, text=k, compound="top", image=tkimg)
            lbl.grid(row=0, column=col, padx=8, pady=8, sticky="n")
            self._thumbs_refs.append(tkimg)
            col += 1

    # helpers
    def _to_image(self, arr: np.ndarray) -> Image.Image:
        if arr.ndim == 2:
            u8 = np.clip(arr, 0, 255).astype(np.uint8)
            rgb = np.stack([u8,u8,u8], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] in (1,3,4):
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[2] == 1:
                rgb = np.repeat(arr, 3, axis=2)
            else:
                rgb = arr[..., :3]
        else:
            # awaryjnie: min/max norm na pierwszym kanale
            x = arr.astype(np.float32)
            x -= x.min(); x /= (x.max() + 1e-8)
            u8 = (x * 255.0 + 0.5).astype(np.uint8)
            rgb = np.stack([u8,u8,u8], axis=-1)

        im = Image.fromarray(rgb, "RGB")
        # downscale do 256x256 max
        w, h = im.size
        s = min(256.0/max(1,w), 256.0/max(1,h), 1.0)
        if s < 1.0:
            im = im.resize((int(w*s), int(h*s)), Image.Resampling.BILINEAR)
        return im
