"""
ruler.py
========

This module provides a lightweight ``Ruler`` widget for the GlitchLab GUI.
The original implementation was defined inline in ``gui/app.py`` as
``_Ruler``; it is now factored out into its own module to simplify
``app.py`` and promote modularity.  The ``Ruler`` draws tick marks
and labels along the top or left edge of an image canvas to indicate
pixel coordinates.  It listens for viewport changes and redraws
itself accordingly.

The widget is intentionally simple: it does not support all features
from the v2 monolith (e.g. inverted axes), but it covers the core
functionality used by the existing app.  If no view transform is
available, it simply draws ticks every 64 pixels; otherwise it
consults the provided transformation callback.

Usage example::

    from glitchlab.gui.widgets.ruler import Ruler
    h_ruler = Ruler(parent, orient="horizontal", length=800)
    v_ruler = Ruler(parent, orient="vertical", length=600)
    h_ruler.pack(side="top", fill="x")
    v_ruler.pack(side="left", fill="y")
    h_ruler.update_ticks((0,0), (800,600), (1,1))

Call ``update_ticks(origin, size, scale)`` whenever the viewport
changes.  ``origin`` is the (x,y) coordinate in image space at the
top‐left of the view; ``size`` is (width,height) of the view in
pixels; ``scale`` is (sx,sy) such that one canvas pixel equals
``1/sx`` image pixels (i.e. scale from image to screen).  In the
current app, ``scale`` is used to compute tick labels.
"""
# glitchlab/gui/widgets/ruler.py
from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Optional, Tuple


class Ruler(tk.Canvas):
    """
    Lekka linijka do viewer'a.

    Parametry
    ---------
    parent : tk.Misc
        Rodzic.
    orientation : {"x","y"}
        Oś linijki: "x" (góra) lub "y" (lewo).
    bg, fg, tick, font : opcjonalne kolory i czcionka
    height, width : sugerowane wymiary (Canvas standard)
    **kwargs : dowolne inne opcje Canvas (BEZ 'orientation' — obsługujemy sami)

    API (wykorzystywane przez app.py; wszystkie są bezpieczne w użyciu):
        - set_zoom(scale: float)               # px_canvas per px_image (zoom)
        - set_origin(origin_px: float)         # początek osi w pikselach obrazu
        - set_length(length_px: int)           # długość obrazu w px (do podpisów)
        - set_cursor(pos_px: Optional[float])  # wskaźnik kursora (px obrazu) lub None
        - set_theme(bg: str, fg: str, tick: str)
        - redraw()                             # ręczne odrysowanie (zwykle automatyczne)
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        orientation: str = "x",
        bg: str = "#2b2b2b",
        fg: str = "#d0d0d0",
        tick: str = "#a0a0a0",
        font: Optional[Tuple[str, int]] = None,
        **kwargs,
    ) -> None:
        # NIE przekazujemy 'orientation' do Canvas
        kwargs = dict(kwargs)  # kopia
        kwargs.setdefault("background", bg)
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(parent, **kwargs)

        self.orientation = "y" if str(orientation).lower().startswith("y") else "x"
        self.bg = bg
        self.fg = fg
        self.tick = tick
        self.font = font or ("TkDefaultFont", 8)

        # Stan rysowania
        self.zoom = 1.0          # px_canvas / px_image
        self.origin = 0.0        # image-px na lewym/górnym brzegu viewer'a
        self.length_px = 0       # pełna długość obrazu (px)
        self.cursor: Optional[float] = None  # pozycja kursora w image-px

        # Reakcja na zmianę rozmiaru
        self.bind("<Configure>", lambda e: self.redraw())

    # ---------------- API ---------------- #

    def set_zoom(self, scale: float) -> None:
        self.zoom = max(1e-6, float(scale))
        self.redraw()

    def set_origin(self, origin_px: float) -> None:
        self.origin = float(origin_px)
        self.redraw()

    def set_length(self, length_px: int) -> None:
        self.length_px = max(0, int(length_px))
        self.redraw()

    def set_cursor(self, pos_px: Optional[float]) -> None:
        self.cursor = None if pos_px is None else float(pos_px)
        self.redraw()

    def set_theme(self, bg: Optional[str] = None, fg: Optional[str] = None, tick: Optional[str] = None) -> None:
        if bg:
            self.bg = bg
            self.configure(background=bg)
        if fg:
            self.fg = fg
        if tick:
            self.tick = tick
        self.redraw()

    # ---------------- Rysowanie ---------------- #

    def redraw(self) -> None:
        self.delete("all")
        W = max(1, self.winfo_width())
        H = max(1, self.winfo_height())

        # Tło
        self.create_rectangle(0, 0, W, H, fill=self.bg, outline=self.bg)

        # Ustal „ładny” krok w pikselach obrazu, tak aby na ekranie odległość między
        # głównymi kreskami wynosiła ~80–140 px.
        desired_px_on_canvas = 110
        step_img = self._nice_step(max(1e-6, desired_px_on_canvas / self.zoom))

        # Pozycja pierwszej kreski (w px obrazu)
        start_img = self._ceil_to(self.origin, step_img)  # pierwszy >= origin
        # Iteruj po kreskach w zakresie widocznym
        end_img = self.origin + (W if self.orientation == "x" else H) / self.zoom + step_img

        # parametry kreskowania
        major_len = 0.55
        mid_len = 0.35
        minor_len = 0.22

        # Rysuj siatkę główną i podpisy
        img = start_img
        while img <= end_img:
            x_canvas = (img - self.origin) * self.zoom
            if self.orientation == "x":
                self._tick_x(x_canvas, H, major_len)
                self._label_x(x_canvas, H, img)
            else:
                y_canvas = (img - self.origin) * self.zoom
                self._tick_y(y_canvas, W, major_len)
                self._label_y(y_canvas, W, img)
            # Pomiędzy głównymi znaczkami narysuj pomocnicze (5 lub 10 podziałek)
            sub = 10 if step_img >= 10 else 5
            sub_step_img = step_img / sub
            for i in range(1, sub):
                sub_img = img + i * sub_step_img
                if sub_img >= end_img:
                    break
                c = (sub_img - self.origin) * self.zoom
                if self.orientation == "x":
                    self._tick_x(c, H, mid_len if i in (5,) else minor_len)
                else:
                    self._tick_y(c, W, mid_len if i in (5,) else minor_len)
            img += step_img

        # Wskaźnik kursora
        if self.cursor is not None:
            c = (self.cursor - self.origin) * self.zoom
            if self.orientation == "x":
                self.create_line(c, 0, c, H, fill=self.fg)
            else:
                self.create_line(0, c, W, c, fill=self.fg)

    # ----- pomoce grafiki -----

    def _tick_x(self, x: float, H: int, scale: float) -> None:
        y0 = 0
        y1 = int(H * scale)
        self.create_line(x, y0, x, y1, fill=self.tick)

    def _label_x(self, x: float, H: int, value_img_px: float) -> None:
        s = str(int(round(value_img_px)))
        self.create_text(x + 2, H - 2, anchor="sw", text=s, fill=self.fg, font=self.font)

    def _tick_y(self, y: float, W: int, scale: float) -> None:
        x0 = int(W * (1.0 - scale))
        x1 = W
        self.create_line(x0, y, x1, y, fill=self.tick)

    def _label_y(self, y: float, W: int, value_img_px: float) -> None:
        s = str(int(round(value_img_px)))
        self.create_text(W - 2, y - 2, anchor="se", text=s, fill=self.fg, font=self.font)

    # ----- matematyka „ładnego” kroku -----

    @staticmethod
    def _nice_step(raw_step: float) -> float:
        """Zwraca krok z rodziny 1/2/5×10^k najbliższy `raw_step`."""
        if raw_step <= 0:
            return 1.0
        k = math.floor(math.log10(raw_step))
        base = 10 ** k
        for m in (1, 2, 5, 10):
            step = m * base
            if step >= raw_step:
                return step
        return 10 * base

    @staticmethod
    def _ceil_to(x: float, step: float) -> float:
        return math.floor(x / step) * step
