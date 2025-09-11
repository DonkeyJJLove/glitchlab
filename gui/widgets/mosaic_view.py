# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Optional, Any, Dict
import numpy as np
from PIL import Image, ImageTk

class MosaicMini(ttk.Frame):
    """Mini-podgląd mozaiki. Jeśli brak overlay w ctx.cache['stage/0/mosaic'], 
       używa pierwszego dostępnego obrazu z klucza 'diag/*' jako fallback.
    """
    def __init__(self, master):
        super().__init__(master)
        self._lab = ttk.Label(self, text="Masks & Amplitude", anchor="w")
        self._lab.pack(anchor="w", padx=6, pady=(6,2))
        self._img = ttk.Label(self, text="(n/a)")
        self._img.pack(padx=6, pady=(0,6))
        self._ref = None

    def set_overlay(self, img_arr: Optional[np.ndarray]) -> None:
        if img_arr is None:
            self._img.configure(text="(n/a)", image="")
            self._ref = None
            return
        im = self._to_image(img_arr)
        tkimg = ImageTk.PhotoImage(im)
        self._img.configure(image=tkimg, text="")
        self._ref = tkimg

    def set_from_cache(self, cache: Dict[str, Any]) -> None:
        arr = cache.get("stage/0/mosaic")
        if arr is None:
            for k,v in cache.items():
                if isinstance(k, str) and k.startswith("diag/") and isinstance(v, np.ndarray):
                    arr = v; break
        self.set_overlay(arr)

    def _to_image(self, arr: np.ndarray) -> Image.Image:
        if arr.ndim == 2:
            u8 = np.clip(arr, 0, 255).astype(np.uint8)
            rgb = np.stack([u8,u8,u8], axis=-1)
        else:
            a = arr[..., :3]
            if a.dtype != np.uint8:
                a = np.clip(a, 0, 255).astype(np.uint8)
            rgb = a
        im = Image.fromarray(rgb, "RGB")
        # max height 160
        w, h = im.size
        s = min(1.0, 160.0 / max(1, h))
        if s < 1.0:
            im = im.resize((int(w*s), int(h*s)), Image.Resampling.BILINEAR)
        return im
