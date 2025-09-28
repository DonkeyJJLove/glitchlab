# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageTk

def _to_u8(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    x = img
    if x.dtype in (np.float32, np.float64):
        x = (x.clip(0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = x.astype(np.uint8, copy=False)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.shape[-1] == 4:
        x = x[..., :3]
    return x

def _thumb(arr: np.ndarray, maxwh: int = 128) -> Image.Image:
    arr = _to_u8(arr)
    if arr is None: return None
    im = Image.fromarray(arr, "RGB")
    im.thumbnail((maxwh, maxwh), Image.LANCZOS)
    return im

class PipelinePreview(ttk.Frame):
    """
    Grid of thumbnails for per-step results.
    API:
      set_images([(title, np.ndarray), ...])
      clear()
    """
    def __init__(self, parent, columns: int = 4, thumb: int = 128):
        super().__init__(parent)
        self.columns = columns
        self.thumb = thumb
        self._cells: List[Tuple[ttk.Label, ImageTk.PhotoImage]] = []
        self._grid = ttk.Frame(self)
        self._grid.pack(fill="both", expand=True)

    def clear(self):
        for w, _ in self._cells:
            w.destroy()
        self._cells.clear()

    def set_images(self, items: List[Tuple[str, np.ndarray]]):
        self.clear()
        if not items: return
        r = c = 0
        for title, arr in items:
            im = _thumb(arr, self.thumb)
            if im is None: continue
            ph = ImageTk.PhotoImage(im)
            cell = ttk.Frame(self._grid, relief="flat", padding=2)
            lbl = ttk.Label(cell, text=title, anchor="center")
            cv = ttk.Label(cell, image=ph)
            cv.image = ph  # keep ref
            lbl.pack(fill="x")
            cv.pack()
            cell.grid(row=r, column=c, padx=4, pady=4, sticky="n")
            self._cells.append((cv, ph))
            c += 1
            if c >= self.columns:
                c = 0; r += 1
