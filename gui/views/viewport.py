# glitchlab/gui/views/viewport.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional, Tuple, Any

import numpy as np

try:
    # preferowany: nowy widget z /widgets
    from glitchlab.gui.widgets.image_canvas import ImageCanvas
except Exception:
    ImageCanvas = None  # type: ignore


def _np_to_u8(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    if isinstance(img, np.ndarray) and img.dtype == np.uint8:
        return img
    try:
        return np.clip(img, 0, 255).astype(np.uint8)
    except Exception:
        return None


class Viewport(ttk.Frame):
    """
    Wrapper na ImageCanvas (jeśli dostępny) z normalizacją API:
      - set_image(np.ndarray|PIL.Image)
      - get_zoom()/set_zoom()/zoom_to()
      - fit()/zoom_fit(), center()
      - screen_to_image(x,y), get_origin()
      - set_crosshair(on)
    """

    def __init__(self, parent: tk.Misc, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas: Any

        if ImageCanvas is not None:
            # UWAGA: to jest ttk.Frame -> NIE podajemy 'background'/'highlightthickness'
            try:
                self.canvas = ImageCanvas(self)
            except tk.TclError:
                # na wszelki wypadek drugi raz, bez kwargs (tu i tak nie przekazujemy)
                self.canvas = ImageCanvas(self)
        else:
            # Fallback: zwykły Canvas – tu można ustawić tło
            self.canvas = tk.Canvas(self, background="#101010", highlightthickness=0)

        self.canvas.pack(fill="both", expand=True)

        # pomocniczy cache dla fallbacków API
        self._zoom: float = 1.0
        self._origin: Tuple[float, float] = (0.0, 0.0)
        self._crosshair_enabled: bool = False

    # ---------- API ----------

    def set_image(self, img: Any) -> None:
        """Obsługuje ndarray (H×W×C, uint8) oraz PIL.Image.Image."""
        if hasattr(self.canvas, "set_image") or hasattr(self.canvas, "display"):
            _img = img if _is_pil(img) else _np_to_u8(img)
            for name in ("set_image", "display", "show", "update_image", "set_array"):
                fn = getattr(self.canvas, name, None)
                if callable(fn):
                    try:
                        fn(_img)
                        return
                    except Exception:
                        # spróbuj kolejnej nazwy
                        pass
        # fallback na czystym Canvas
        try:
            from PIL import Image, ImageTk
            pil = img if _is_pil(img) else Image.fromarray(_np_to_u8(img))
            # trzymaj referencję, żeby GC nie ubił obrazka
            self._tk_img = ImageTk.PhotoImage(pil)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self._tk_img, anchor="nw")
        except Exception:
            try:
                self.canvas.delete("all")
            except Exception:
                pass

    def get_zoom(self) -> float:
        for name in ("get_zoom", "zoom"):
            fn = getattr(self.canvas, name, None)
            if callable(fn):
                try:
                    z = float(fn())
                    self._zoom = z
                    return z
                except Exception:
                    pass
        return self._zoom

    def set_zoom(self, z: float) -> None:
        for name in ("set_zoom", "zoom_to"):
            fn = getattr(self.canvas, name, None)
            if callable(fn):
                try:
                    fn(float(z))
                    self._zoom = float(z)
                    return
                except Exception:
                    pass
        self._zoom = float(z)

    def zoom_to(self, z: float) -> None:
        self.set_zoom(z)

    def fit(self) -> None:
        for name in ("fit", "zoom_fit"):
            fn = getattr(self.canvas, name, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass

    def zoom_fit(self) -> None:
        self.fit()

    def center(self) -> None:
        fn = getattr(self.canvas, "center", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    def screen_to_image(self, x: int, y: int) -> Tuple[int, int]:
        for name in ("screen_to_image", "to_image", "view_to_image", "coords_to_image"):
            fn = getattr(self.canvas, name, None)
            if callable(fn):
                try:
                    ix, iy = fn(x, y)
                    return int(ix), int(iy)
                except Exception:
                    pass
        # fallback przez zoom/origin
        z = max(1e-6, float(self.get_zoom()))
        ox, oy = self.get_origin()
        return int(round(ox + x / z)), int(round(oy + y / z))

    def get_origin(self) -> Tuple[float, float]:
        for name in ("get_origin", "origin"):
            fn = getattr(self.canvas, name, None)
            if callable(fn):
                try:
                    ox, oy = fn()
                    self._origin = (float(ox), float(oy))
                    return self._origin
                except Exception:
                    pass
        return self._origin

    def set_crosshair(self, enabled: bool) -> None:
        fn = getattr(self.canvas, "set_crosshair", None)
        if callable(fn):
            try:
                fn(bool(enabled))
                self._crosshair_enabled = bool(enabled)
                return
            except Exception:
                pass
        self._crosshair_enabled = bool(enabled)


def _is_pil(obj: Any) -> bool:
    try:
        from PIL import Image
        return isinstance(obj, Image.Image)
    except Exception:
        return False
