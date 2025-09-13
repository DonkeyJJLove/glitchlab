# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from typing import Callable, Optional, Tuple

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk


__all__ = ["ImageCanvas", "OverviewMini", "MaskPreview"]


class ImageCanvas(ttk.Frame):
    """
    Prosty viewer z płynnym zoom/pan.
    Skalowanie: screen_xy = img_xy * zoom + offset
    gdzie offset przesuwa lewy-górny róg obrazu w układzie canvas.

    Public API (używane przez GUI):
      - set_image(img_u8 | PIL.Image)
      - set_zoom(z: float), get_zoom() -> float
      - fit(), center()
      - set_crosshair(flag: bool)
      - get_viewport_pixels() -> (x0,y0,x1,y1) w pikselach obrazu (clipped)
      - set_view_center(x: float, y: float)   # w pikselach obrazu
      - on_view_changed: Optional[Callable[[], None]]  # callback po zmianie widoku
    """
    BG = "#2b2b2b"

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.cv = tk.Canvas(self, bg=self.BG, highlightthickness=0, bd=0)
        self.cv.pack(fill="both", expand=True)

        # stan
        self._img_np: Optional[np.ndarray] = None      # uint8 HxWx3
        self._img_pil: Optional[Image.Image] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._img_item: Optional[int] = None

        self._zoom: float = 1.0
        self._min_zoom: float = 0.05
        self._max_zoom: float = 16.0
        self._offset: Tuple[float, float] = (0.0, 0.0)   # w px canvas

        self._is_panning: bool = False
        self._pan_anchor: Tuple[int, int] = (0, 0)

        self._crosshair: bool = False
        self._cross_items: Tuple[Optional[int], Optional[int]] = (None, None)

        # callback (np. do OverviewMini)
        self.on_view_changed: Optional[Callable[[], None]] = None

        # zdarzenia
        self.cv.bind("<Configure>", lambda e: self._redraw())
        self.cv.bind("<Button-1>", self._on_btn1)
        self.cv.bind("<B1-Motion>", self._on_drag)
        self.cv.bind("<ButtonRelease-1>", self._on_release)
        # scroll: Windows/Mac różnie
        self.cv.bind("<MouseWheel>", self._on_wheel)      # Windows / macOS (sometimes)
        self.cv.bind("<Button-4>", self._on_wheel_up)     # Linux
        self.cv.bind("<Button-5>", self._on_wheel_down)   # Linux

    # ---------------- public API ----------------
    def set_image(self, img) -> None:
        """
        img: np.ndarray uint8 HxWx3 lub PIL.Image (RGB).
        """
        if isinstance(img, Image.Image):
            pil = img.convert("RGB")
            arr = np.array(pil, dtype=np.uint8)
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                arr = np.repeat(img[..., None], 3, axis=2).astype(np.uint8)
                pil = Image.fromarray(arr, "RGB")
            else:
                arr = img.astype(np.uint8)
                pil = Image.fromarray(arr, "RGB")
        else:
            self._img_np = None
            self._img_pil = None
            self._photo = None
            self._img_item = None
            self._redraw()
            return

        self._img_np = arr
        self._img_pil = pil
        self._ensure_photo()
        # domyślnie dopasuj do okna (ale zachowaj bieżący zoom jeśli już był obraz)
        if self._zoom is None or self._img_item is None:
            self._zoom = 1.0
        self.fit()
        self._emit_view_changed()

    def get_zoom(self) -> float:
        return float(self._zoom)

    def set_zoom(self, z: float) -> None:
        z = max(self._min_zoom, min(self._max_zoom, float(z)))
        if abs(z - self._zoom) < 1e-6:
            return
        # zoom względem środka canvasa
        cx = self.cv.winfo_width() / 2.0
        cy = self.cv.winfo_height() / 2.0
        self._zoom_around_point(z / self._zoom, (cx, cy))
        self._emit_view_changed()

    def set_crosshair(self, flag: bool) -> None:
        self._crosshair = bool(flag)
        self._redraw()

    def fit(self) -> None:
        """Dopasuj cały obraz do okna (z marginesem)."""
        if self._img_pil is None:
            return
        cw = max(1, self.cv.winfo_width())
        ch = max(1, self.cv.winfo_height())
        iw, ih = self._img_pil.size
        if iw == 0 or ih == 0:
            return
        scale = min(cw / iw, ch / ih)
        # niewielki margines
        scale *= 0.98
        scale = max(self._min_zoom, min(self._max_zoom, scale))
        self._zoom = scale
        # wycentruj
        self.center()
        self._emit_view_changed()

    def center(self) -> None:
        """Wycentruj obraz."""
        if self._img_pil is None:
            return
        cw = self.cv.winfo_width()
        ch = self.cv.winfo_height()
        iw, ih = self._img_pil.size
        x = (cw - iw * self._zoom) / 2.0
        y = (ch - ih * self._zoom) / 2.0
        self._offset = (x, y)
        self._redraw()
        self._emit_view_changed()

    def set_view_center(self, x_img: float, y_img: float) -> None:
        """Ustaw widok tak, aby środek wskazywał na (x_img,y_img) w pikselach obrazu."""
        if self._img_pil is None:
            return
        cw = self.cv.winfo_width()
        ch = self.cv.winfo_height()
        iw, ih = self._img_pil.size
        x_img = max(0.0, min(iw, float(x_img)))
        y_img = max(0.0, min(ih, float(y_img)))
        # po transformacji punkt (x_img,y_img) ma trafić w (cw/2, ch/2)
        ox = cw / 2.0 - x_img * self._zoom
        oy = ch / 2.0 - y_img * self._zoom
        self._offset = (ox, oy)
        self._redraw()
        self._emit_view_changed()

    def get_viewport_pixels(self) -> Tuple[int, int, int, int]:
        """
        Zwróć prostokąt aktualnego viewportu w pikselach obrazu (x0,y0,x1,y1), przycięty do granic obrazu.
        Gdy brak obrazu —(0,0,0,0).
        """
        if self._img_pil is None:
            return (0, 0, 0, 0)
        cw = self.cv.winfo_width()
        ch = self.cv.winfo_height()
        iw, ih = self._img_pil.size
        ox, oy = self._offset
        # szukamy img_x = (screen_x - ox) / zoom dla screen_x in [0,cw]
        x0 = int(max(0, math.floor((0 - ox) / self._zoom)))
        y0 = int(max(0, math.floor((0 - oy) / self._zoom)))
        x1 = int(min(iw, math.ceil((cw - ox) / self._zoom)))
        y1 = int(min(ih, math.ceil((ch - oy) / self._zoom)))
        if x1 < x0: x0, x1 = x1, x0
        if y1 < y0: y0, y1 = y1, y0
        x0 = max(0, min(iw, x0))
        y0 = max(0, min(ih, y0))
        x1 = max(0, min(iw, x1))
        y1 = max(0, min(ih, y1))
        return (x0, y0, x1, y1)

    # ---------------- events ----------------
    def _on_btn1(self, e):
        self._is_panning = True
        self._pan_anchor = (e.x, e.y)

    def _on_drag(self, e):
        if not self._is_panning:
            return
        ax, ay = self._pan_anchor
        dx = e.x - ax
        dy = e.y - ay
        ox, oy = self._offset
        self._offset = (ox + dx, oy + dy)
        self._pan_anchor = (e.x, e.y)
        self._redraw()
        self._emit_view_changed()

    def _on_release(self, _e):
        self._is_panning = False

    def _on_wheel(self, e):
        # Windows: e.delta multiple of 120
        step = 1.0 + (0.1 * (1 if e.delta > 0 else -1))
        self._zoom_around_point(step, (e.x, e.y))
        self._emit_view_changed()

    def _on_wheel_up(self, e):
        self._zoom_around_point(1.1, (e.x, e.y))
        self._emit_view_changed()

    def _on_wheel_down(self, e):
        self._zoom_around_point(1/1.1, (e.x, e.y))
        self._emit_view_changed()

    # ---------------- internals ----------------
    def _zoom_around_point(self, factor: float, pivot_screen_xy: Tuple[float, float]) -> None:
        if self._img_pil is None:
            return
        # clamp
        new_zoom = max(self._min_zoom, min(self._max_zoom, self._zoom * float(factor)))
        factor = new_zoom / self._zoom
        px, py = pivot_screen_xy
        ox, oy = self._offset
        # (px,py) = img_xy * zoom + offset  → zachowaj (px,py) stałe
        self._offset = (px - (px - ox) * factor, py - (py - oy) * factor)
        self._zoom = new_zoom
        self._redraw()

    def _ensure_photo(self):
        if self._img_pil is None:
            self._photo = None
            return
        self._photo = ImageTk.PhotoImage(self._img_pil)

    def _redraw(self):
        self.cv.delete("all")
        if self._img_pil is None or self._photo is None:
            # nic nie rysujemy, puste tło
            return

        iw, ih = self._img_pil.size
        ox, oy = self._offset

        # W Tkinter najlepiej skalować poprzez create_image + scale w macierzach?
        # Tu robimy soft-scale przez PIL przy ekstremach zoomu dla lepszej jakości.
        # Dla zoomu 0.2..4.0 zostawiamy oryginał + transformację Canvas (tk nie ma macierzy),
        # więc stosujemy PIL-resize do rozmiaru ekranu.
        disp_w = max(1, int(round(iw * self._zoom)))
        disp_h = max(1, int(round(ih * self._zoom)))
        pil = self._img_pil if (disp_w == iw and disp_h == ih) else self._img_pil.resize((disp_w, disp_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)

        self._img_item = self.cv.create_image(ox, oy, anchor="nw", image=self._photo)

        # crosshair (opcjonalnie)
        if self._crosshair:
            cw = self.cv.winfo_width()
            ch = self.cv.winfo_height()
            cx = cw // 2
            cy = ch // 2
            self.cv.create_line(0, cy, cw, cy, fill="#7090ff", stipple="gray12")
            self.cv.create_line(cx, 0, cx, ch, fill="#7090ff", stipple="gray12")

    def _emit_view_changed(self):
        if callable(self.on_view_changed):
            try:
                self.on_view_changed()
            except Exception:
                pass


# ---------------- Aux widgets for Global tab ----------------

class OverviewMini(ttk.Frame):
    """
    Minimap całego obrazu + prostokąt aktualnego viewportu.
    Klik/drag w minimapie przewija główny obraz.
    """
    def __init__(self, parent, canvas: ImageCanvas, width: int = 220, height: int = 160):
        super().__init__(parent)
        self.canvas_ref = canvas
        self.width = width
        self.height = height

        self.cv = tk.Canvas(self, width=width, height=height, bg="#1e1e1e", highlightthickness=1, highlightbackground="#444")
        self.cv.pack(fill="both", expand=False)

        self._thumb: Optional[Image.Image] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._rect_item: Optional[int] = None

        # po zmianie widoku w głównym canvasie – odśwież prostokąt
        self.canvas_ref.on_view_changed = self._refresh  # nadpisujemy/ustawiamy callback
        self.cv.bind("<Configure>", lambda e: self._rebuild_thumb())
        self.cv.bind("<Button-1>", self._on_click)
        self.cv.bind("<B1-Motion>", self._on_click)

    def _rebuild_thumb(self):
        img = self.canvas_ref._img_pil  # używamy publicznie niedostępnego, ale to tylko miniaturka
        if img is None:
            self.cv.delete("all")
            self._thumb = None
            self._photo = None
            return
        W = max(1, self.cv.winfo_width())
        H = max(1, self.cv.winfo_height())
        iw, ih = img.size
        scale = min(W / iw, H / ih)
        tw = max(1, int(iw * scale))
        th = max(1, int(ih * scale))
        self._thumb = img.resize((tw, th), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(self._thumb)
        self._refresh()

    def _refresh(self):
        self.cv.delete("all")
        if self._photo is None:
            self._rebuild_thumb()
            if self._photo is None:
                return
        W = self.cv.winfo_width()
        H = self.cv.winfo_height()
        tw = self._photo.width()
        th = self._photo.height()
        # wycentruj miniaturę
        x0 = (W - tw) // 2
        y0 = (H - th) // 2
        self.cv.create_image(x0, y0, anchor="nw", image=self._photo)

        # prostokąt viewportu
        x_img0, y_img0, x_img1, y_img1 = self.canvas_ref.get_viewport_pixels()
        img = self.canvas_ref._img_pil
        if img is None or x_img1 <= x_img0 or y_img1 <= y_img0:
            return
        iw, ih = img.size
        scale = min(self._photo.width() / iw, self._photo.height() / ih)
        rx0 = x0 + int(x_img0 * scale)
        ry0 = y0 + int(y_img0 * scale)
        rx1 = x0 + int(x_img1 * scale)
        ry1 = y0 + int(y_img1 * scale)
        self._rect_item = self.cv.create_rectangle(rx0, ry0, rx1, ry1, outline="#80ff80")

    def _on_click(self, e):
        img = self.canvas_ref._img_pil
        if img is None or self._photo is None:
            return
        W = self.cv.winfo_width()
        H = self.cv.winfo_height()
        tw = self._photo.width()
        th = self._photo.height()
        x0 = (W - tw) // 2
        y0 = (H - th) // 2
        # zamień klik z minimapy na piksele obrazu
        sx = e.x - x0
        sy = e.y - y0
        if sx < 0 or sy < 0 or sx > tw or sy > th:
            return
        iw, ih = img.size
        scale = min(self._photo.width() / iw, self._photo.height() / ih)
        x_img = sx / scale
        y_img = sy / scale
        self.canvas_ref.set_view_center(x_img, y_img)


class MaskPreview(ttk.Frame):
    """
    Podgląd wybranej maski (2D ndarray) w małym okienku.
    API:
      - set_array(arr: np.ndarray|None)
    """
    def __init__(self, parent, width: int = 220, height: int = 120):
        super().__init__(parent)
        self.cv = tk.Canvas(self, width=width, height=height, bg="#1e1e1e", highlightthickness=1, highlightbackground="#444")
        self.cv.pack(fill="both", expand=False)
        self._photo: Optional[ImageTk.PhotoImage] = None
        self.cv.bind("<Configure>", lambda e: self._resize_to_canvas())

        self._arr: Optional[np.ndarray] = None

    def set_array(self, arr: Optional[np.ndarray]) -> None:
        self._arr = None if arr is None else np.array(arr, copy=False)
        self._resize_to_canvas()

    def _resize_to_canvas(self):
        self.cv.delete("all")
        if self._arr is None:
            return
        a = self._arr
        # normalizacja do 0..255
        if a.ndim == 3:
            a = a[..., 0]
        mn = float(np.nanmin(a))
        mx = float(np.nanmax(a))
        den = (mx - mn) if (mx - mn) > 1e-12 else 1.0
        u8 = np.clip((a - mn) * (255.0 / den), 0, 255).astype(np.uint8)
        pil = Image.fromarray(u8, "L").convert("RGB")

        W = max(1, self.cv.winfo_width())
        H = max(1, self.cv.winfo_height())
        iw, ih = pil.size
        scale = min(W / iw, H / ih)
        tw = max(1, int(iw * scale))
        th = max(1, int(ih * scale))
        pil = pil.resize((tw, th), Image.NEAREST)
        self._photo = ImageTk.PhotoImage(pil)
        x0 = (W - tw) // 2
        y0 = (H - th) // 2
        self.cv.create_image(x0, y0, anchor="nw", image=self._photo)
