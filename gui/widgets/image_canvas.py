# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class ImageCanvas(ttk.Frame):
    """Canvas z płynnym zoom/pan i trybem Fit/1:1. Zero zależności zewnętrznych."""

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg="#1e1e1e", highlightthickness=0)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._img = None
        self._tk = None
        self._scale = 1.0
        self._fit_mode = True  # Fit-to-window
        self._img_id = None

        # Events
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<MouseWheel>", self._on_wheel)            # Windows
        self.canvas.bind("<Button-4>", self._on_wheel)              # Linux up
        self.canvas.bind("<Button-5>", self._on_wheel)              # Linux down
        self.canvas.bind("<ButtonPress-2>", self._start_pan)        # MMB
        self.canvas.bind("<B2-Motion>", self._on_pan)
        self.canvas.bind("<Double-Button-1>", self._toggle_fit)

    # Public API
    def set_image(self, pil_image: Image.Image) -> None:
        self._img = pil_image.copy()
        self._scale = 1.0
        self._fit_mode = True
        self._render()

    def fit(self) -> None:
        self._fit_mode = True
        self._render()

    def set_zoom(self, scale: float) -> None:
        self._scale = max(0.05, min(32.0, float(scale)))
        self._fit_mode = False
        self._render()

    # Internals
    def _toggle_fit(self, event=None):
        self._fit_mode = not self._fit_mode
        self._render()

    def _on_resize(self, event):
        if self._fit_mode:
            self._render()

    def _on_wheel(self, event):
        if self._img is None:
            return
        # Normalize delta
        delta = 0
        if getattr(event, "num", None) == 4:
            delta = 120
        elif getattr(event, "num", None) == 5:
            delta = -120
        else:
            delta = int(getattr(event, "delta", 0))

        if delta == 0:
            return

        factor = 1.1 if delta > 0 else 1/1.1
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        self._fit_mode = False
        old = self._scale
        self._scale = max(0.05, min(32.0, self._scale * factor))
        self._render(anchor=(cx, cy), keep_point=(event.x, event.y), old_scale=old)

    def _start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _on_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _render(self, anchor=None, keep_point=None, old_scale=None):
        if self._img is None:
            self.canvas.delete("all")
            return

        im = self._img
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        if W < 2 or H < 2:
            return

        if self._fit_mode:
            iw, ih = im.size
            s = min(W/max(1, iw), H/max(1, ih))
            s = max(0.05, min(32.0, s))
        else:
            s = self._scale

        iw, ih = im.size
        tw, th = int(iw * s), int(ih * s)
        if tw < 1 or th < 1:
            return
        try:
            imr = im.resize((tw, th), Image.Resampling.BICUBIC)
        except Exception:
            imr = im.resize((tw, th))

        self._tk = ImageTk.PhotoImage(imr)
        self.canvas.configure(scrollregion=(0, 0, tw, th))
        if self._img_id is None:
            self._img_id = self.canvas.create_image(0, 0, image=self._tk, anchor="nw")
        else:
            self.canvas.itemconfigure(self._img_id, image=self._tk)

        # center if fit, otherwise keep point under cursor
        if self._fit_mode:
            self.canvas.xview_moveto(max(0.0, (tw - W) / 2.0 / max(1, tw)))
            self.canvas.yview_moveto(max(0.0, (th - H) / 2.0 / max(1, th)))
        else:
            if anchor and keep_point and old_scale:
                # keep the canvas point under the mouse after zoom change
                ax, ay = anchor
                kx, ky = keep_point
                # proportion of position in scrollregion
                rx = (ax * (self._scale/old_scale) - kx) / max(1, tw)
                ry = (ay * (self._scale/old_scale) - ky) / max(1, th)
                self.canvas.xview_moveto(min(max(0.0, rx), 1.0))
                self.canvas.yview_moveto(min(max(0.0, ry), 1.0))
