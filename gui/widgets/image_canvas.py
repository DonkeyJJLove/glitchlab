
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFont
from typing import Optional

class ImageCanvas(ttk.Frame):
    """
    Viewer z: dopasowaniem, 100%, zoom +/- (kółko), panning (drag), wycentrowaniem,
    krzyżykowym kursorem i miarkami (rulers) na górze/lewej.
    API kompatybilne: set_image(PIL.Image), get_image().
    """
    def __init__(self, master, show_toolbar: bool = True, show_rulers: bool = True, **kw):
        super().__init__(master, **kw)
        self.bg = "#1a1a1a"
        self.fg = "#bbb"
        self.ruler_size = 22 if show_rulers else 0
        self._im: Optional[Image.Image] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._scale: float = 1.0
        self._ox: float = 0.0   # pan offset in screen px
        self._oy: float = 0.0
        self._drag_from: Optional[tuple[int,int]] = None
        self._cursor_xy: Optional[tuple[int,int]] = None

        # layout: rulers + canvas + toolbar
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        # rulers
        if self.ruler_size:
            self.ruler_top = tk.Canvas(self, height=self.ruler_size, bg=self.bg, highlightthickness=0)
            self.ruler_left = tk.Canvas(self, width=self.ruler_size, bg=self.bg, highlightthickness=0)
            self.ruler_top.grid(row=0, column=1, sticky="ew")
            self.ruler_left.grid(row=1, column=0, sticky="ns")
        else:
            self.ruler_top = None
            self.ruler_left = None

        # canvas
        self.cv = tk.Canvas(self, bg=self.bg, highlightthickness=0, cursor="tcross")
        self.cv.grid(row=1, column=1, sticky="nsew")

        # toolbar
        if show_toolbar:
            tb = ttk.Frame(self); tb.grid(row=2, column=1, sticky="w")
            ttk.Button(tb, text="Fit", command=self.zoom_fit).pack(side="left")
            ttk.Button(tb, text="100%", command=self.zoom_100).pack(side="left", padx=(6,0))
            ttk.Button(tb, text="−", command=lambda: self.step_zoom(0.9)).pack(side="left", padx=(6,0))
            ttk.Button(tb, text="+", command=lambda: self.step_zoom(1.1)).pack(side="left")
            ttk.Button(tb, text="Center", command=self.center).pack(side="left", padx=(6,0))

        # events
        self.cv.bind("<Configure>", lambda e: self._redraw())
        self.cv.bind("<ButtonPress-1>", self._on_down)
        self.cv.bind("<B1-Motion>", self._on_drag)
        self.cv.bind("<ButtonRelease-1>", self._on_up)
        # wheel
        self.cv.bind("<MouseWheel>", self._on_wheel)          # Windows
        self.cv.bind("<Button-4>", self._on_wheel_linux)      # Linux
        self.cv.bind("<Button-5>", self._on_wheel_linux)
        self.cv.bind("<Motion>", self._on_motion)

    # public API
    def set_image(self, im: Image.Image) -> None:
        if im is None:
            self._im = None
            self._photo = None
            self._redraw()
            return
        if not isinstance(im, Image.Image):
            raise TypeError("set_image expects PIL.Image.Image")
        self._im = im
        # reset zoom to Fit on first load
        self.zoom_fit()

    def get_image(self) -> Optional[Image.Image]:
        return self._im

    def zoom_fit(self) -> None:
        if self._im is None: return
        W = max(1, self.cv.winfo_width())
        H = max(1, self.cv.winfo_height())
        iw, ih = self._im.size
        sx = (W - 2) / float(iw)
        sy = (H - 2) / float(ih)
        self._scale = max(0.01, min(sx, sy))
        self.center()
        self._redraw()

    def zoom_100(self) -> None:
        self._scale = 1.0
        self.center()
        self._redraw()

    def step_zoom(self, factor: float) -> None:
        self._scale = max(0.01, min(32.0, self._scale * float(factor)))
        self._redraw()

    def center(self) -> None:
        self._ox = self.cv.winfo_width() / 2.0
        self._oy = self.cv.winfo_height() / 2.0
        self._redraw()

    # internals
    def _on_down(self, e):
        self._drag_from = (e.x, e.y)

    def _on_drag(self, e):
        if not self._drag_from: return
        dx = e.x - self._drag_from[0]
        dy = e.y - self._drag_from[1]
        self._ox += dx
        self._oy += dy
        self._drag_from = (e.x, e.y)
        self._redraw()

    def _on_up(self, _e):
        self._drag_from = None

    def _on_wheel(self, e):
        factor = 1.1 if e.delta > 0 else 1.0/1.1
        self._zoom_at(e.x, e.y, factor)

    def _on_wheel_linux(self, e):
        # Button-4: up, Button-5: down
        factor = 1.1 if e.num == 4 else 1.0/1.1
        self._zoom_at(e.x, e.y, factor)

    def _zoom_at(self, x, y, factor):
        if self._im is None: return
        old_scale = self._scale
        self._scale = max(0.01, min(32.0, self._scale * factor))
        # zoom at cursor: adjust pan so that image point under cursor stays
        if abs(self._scale - old_scale) > 1e-6:
            # image coords under cursor before
            ix, iy = self._screen_to_image(x, y)
            # after zoom new screen pos of that image coord:
            sx, sy = self._image_to_screen(ix, iy)
            self._ox += (x - sx)
            self._oy += (y - sy)
        self._redraw()

    def _on_motion(self, e):
        self._cursor_xy = (e.x, e.y)
        self._draw_crosshair()
        self._draw_rulers()

    # coordinate transforms
    def _image_to_screen(self, ix, iy):
        if self._im is None: return (0,0)
        iw, ih = self._im.size
        sx = (ix - iw/2.0) * self._scale + self._ox
        sy = (iy - ih/2.0) * self._scale + self._oy
        return sx, sy

    def _screen_to_image(self, sx, sy):
        if self._im is None: return (0,0)
        iw, ih = self._im.size
        ix = (sx - self._ox) / (self._scale + 1e-9) + iw/2.0
        iy = (sy - self._oy) / (self._scale + 1e-9) + ih/2.0
        return ix, iy

    def _redraw(self):
        self.cv.delete("all")
        if self._im is None:
            return
        iw, ih = self._im.size
        # scaled image
        sw = max(1, int(round(iw * self._scale)))
        sh = max(1, int(round(ih * self._scale)))
        im2 = self._im.resize((sw, sh), Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC)
        self._photo = ImageTk.PhotoImage(im2)
        x = self._ox - sw/2.0
        y = self._oy - sh/2.0
        self.cv.create_image(x, y, image=self._photo, anchor="nw")
        self._draw_crosshair()
        self._draw_rulers()

    def _draw_crosshair(self):
        self.cv.delete("cross")
        if self._cursor_xy is None: return
        x, y = self._cursor_xy
        w = self.cv.winfo_width(); h = self.cv.winfo_height()
        self.cv.create_line(0, y, w, y, fill="#606060", tags="cross")
        self.cv.create_line(x, 0, x, h, fill="#606060", tags="cross")

    def _draw_rulers(self):
        if not self.ruler_size or self._im is None:
            return
        iw, ih = self._im.size
        # top
        self.ruler_top.delete("all")
        self.ruler_left.delete("all")
        self.ruler_top.create_rectangle(0,0,self.ruler_top.winfo_width(), self.ruler_size, fill=self.bg, outline="")
        self.ruler_left.create_rectangle(0,0,self.ruler_size, self.ruler_left.winfo_height(), fill=self.bg, outline="")
        # ticks every 100 px in image coords (scaled)
        step = 100
        # horiz
        W = self.ruler_top.winfo_width()
        for i in range(0, iw+1, step):
            sx, sy = self._image_to_screen(i, 0)
            if 0 <= sx <= W:
                self.ruler_top.create_line(sx, self.ruler_size, sx, self.ruler_size-7, fill=self.fg)
                self.ruler_top.create_text(sx+2, self.ruler_size-9, text=str(i), anchor="nw", fill=self.fg, font=("TkDefaultFont", 8))
        # vert
        H = self.ruler_left.winfo_height()
        for j in range(0, ih+1, step):
            sx, sy = self._image_to_screen(0, j)
            if 0 <= sy <= H:
                self.ruler_left.create_line(self.ruler_size, sy, self.ruler_size-7, sy, fill=self.fg)
                self.ruler_left.create_text(2, sy+2, text=str(j), anchor="nw", fill=self.fg, font=("TkDefaultFont", 8))
