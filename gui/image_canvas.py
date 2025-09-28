# glitchlab/gui/image_canvas.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Optional
import numpy as np
from PIL import Image, ImageTk


def _to_u8(img: np.ndarray) -> np.ndarray:
    if img is None: return None
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


class ImageCanvas(ttk.Frame):
    """
    Scrollable image canvas with zoom and optional crosshair.
    API: set_image, zoom_in/out/to, fit, center, set_crosshair(bool)
    """

    def __init__(self, parent, background="#202020"):
        super().__init__(parent)
        self.bg = background
        self._img_u8: Optional[np.ndarray] = None
        self._pil: Optional[Image.Image] = None
        self._tk: Optional[ImageTk.PhotoImage] = None
        self._scale = 1.0
        self._crosshair = False

        self.canvas = tk.Canvas(self, bg=self.bg, highlightthickness=0)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._img_id = None
        self._cross_ids = []

        # Mouse bindings
        self.canvas.bind("<MouseWheel>", self._on_wheel)  # Windows
        self.canvas.bind("<Button-4>", self._on_wheel)  # Linux up
        self.canvas.bind("<Button-5>", self._on_wheel)  # Linux down
        self.canvas.bind("<ButtonPress-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._do_pan)
        self.canvas.bind("<Motion>", self._on_motion)

    def set_image(self, img: np.ndarray):
        self._img_u8 = _to_u8(img)
        self._pil = None if self._img_u8 is None else Image.fromarray(self._img_u8, "RGB")
        self._render()

    def zoom_in(self, factor: float = 1.25):
        self._scale = min(64.0, self._scale * factor);
        self._render()

    def zoom_out(self, factor: float = 1.25):
        self._scale = max(0.05, self._scale / factor);
        self._render()

    def zoom_to(self, scale: float):
        self._scale = max(0.05, min(64.0, float(scale)));
        self._render()

    def center(self):
        self.canvas.update_idletasks()
        self.canvas.xview_moveto(0.0);
        self.canvas.yview_moveto(0.0)

    def fit(self, margin: int = 16):
        if self._pil is None: return
        cw = max(32, self.canvas.winfo_width() - margin * 2)
        ch = max(32, self.canvas.winfo_height() - margin * 2)
        iw, ih = self._pil.size
        if iw <= 0 or ih <= 0: return
        sx = cw / iw;
        sy = ch / ih
        self._scale = max(0.05, min(64.0, min(sx, sy)))
        self._render()

    def set_crosshair(self, on: bool = True):
        self._crosshair = bool(on);
        self._draw_cross(None)

    # internal
    def _render(self):
        self.canvas.delete("all")
        self._img_id = None
        self._cross_ids.clear()
        if self._pil is None: return
        iw, ih = self._pil.size
        w = max(1, int(iw * self._scale))
        h = max(1, int(ih * self._scale))
        im = self._pil if (w == iw and h == ih) else self._pil.resize((w, h), Image.NEAREST)
        self._tk = ImageTk.PhotoImage(im)
        self._img_id = self.canvas.create_image(0, 0, anchor="nw", image=self._tk)
        self.canvas.configure(scrollregion=(0, 0, w, h))
        self._draw_cross(None)

    def _on_wheel(self, event):
        # Positive for zoom-in, negative for zoom-out
        delta = 1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else -1
        if delta > 0:
            self.zoom_in(1.10)
        else:
            self.zoom_out(1.10)

    def _start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _do_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_motion(self, event):
        if self._crosshair: self._draw_cross((event.x, event.y))

    def _draw_cross(self, xy):
        for cid in self._cross_ids:
            self.canvas.delete(cid)
        self._cross_ids.clear()
        if not self._crosshair: return
        w = self.canvas.winfo_width();
        h = self.canvas.winfo_height()
        if xy is None:
            x, y = w // 2, h // 2
        else:
            x, y = xy
        self._cross_ids.append(self.canvas.create_line(0, y, w, y, fill="#ffffff"))
        self._cross_ids.append(self.canvas.create_line(x, 0, x, h, fill="#ffffff"))
