# glitchlab/gui/widgets/ruler.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import tkinter as tk
from typing import Optional, Tuple


class Ruler(tk.Canvas):
    """
    Lekka linijka do viewer'a obrazu (oś X – u góry, oś Y – po lewej).

    API (kanoniczne):
      - set_zoom(scale: float)               # px_canvas per px_image (zoom)
      - set_origin(origin_px: float)         # image-px na lewym/górnym brzegu widoku
      - set_length(length_px: int)           # pełna długość obrazu (px) do etykiet
      - set_cursor(pos_px: Optional[float])  # wskaźnik kursora (w px obrazu) lub None
      - set_theme(bg: str, fg: str, tick: str)
      - redraw()                             # ręczne odrysowanie (zwykle niepotrzebne)

    API (zgodne wstecz z dotychczasowym app.py):
      - set_zoom_and_size(zoom: float, length_px: int)
      - update_marker_view_px(view_px: int)  # view_px w pikselach ekranu → przelicza na px obrazu

    Parametry konstruktora:
      orientation: "x" (górny) lub "y" (lewy). Nie jest przekazywany do Canvas!
      height/width możesz podać normalnie jak do Canvas; domyślnie dobrane pod oś.

    Uwaga wydajnościowa:
      Wbudowany lekki cache stanu zapobiega zbędnym redraw przy braku zmian.
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
        # NIE przekazujemy 'orientation' do Canvas (unik heiß: unknown option "-orientation")
        kw = dict(kwargs)
        kw.setdefault("background", bg)
        kw.setdefault("highlightthickness", 0)

        # Sensowne wymiary domyślne
        if orientation.lower().startswith("x"):
            kw.setdefault("height", 26)
        else:
            kw.setdefault("width", 28)

        super().__init__(parent, **kw)

        self.orientation = "y" if str(orientation).lower().startswith("y") else "x"
        self.bg = bg
        self.fg = fg
        self.tick = tick
        self.font = font or ("TkDefaultFont", 8)

        # Stan bieżący
        self.zoom = 1.0          # px_canvas / px_image
        self.origin = 0.0        # image-px na lewym/górnym brzegu widoku
        self.length_px = 0       # pełna długość obrazu (px)
        self.cursor: Optional[float] = None  # pozycja kursora (px obrazu)

        # Cache do wstrzemięźliwego rysowania
        self._last_sig: Optional[Tuple[str, int, int, float, float, int, Optional[int]]] = None

        # Reakcja na zmianę rozmiaru
        self.bind("<Configure>", lambda _e: self.redraw())

    # ---------- Public API (kanoniczne) ----------

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
        changed = False
        if bg and bg != self.bg:
            self.bg = bg
            self.configure(background=bg)
            changed = True
        if fg and fg != self.fg:
            self.fg = fg
            changed = True
        if tick and tick != self.tick:
            self.tick = tick
            changed = True
        if changed:
            self.redraw()

    # ---------- Public API (zgodne z dawnym app.py) ----------

    def set_zoom_and_size(self, zoom: float, length_px: int) -> None:
        """Alias: wygodne ustawienie zoom + długość obrazu."""
        self.zoom = max(1e-6, float(zoom))
        self.length_px = max(0, int(length_px))
        self.redraw()

    def update_marker_view_px(self, view_px: int) -> None:
        """
        view_px – pozycja kursora w pikselach „widoku/canvas”.
        Przelicza na współrzędną obrazu i ustawia wskaźnik.
        """
        try:
            img_px = self.origin + float(view_px) / max(self.zoom, 1e-6)
        except Exception:
            img_px = None
        self.set_cursor(img_px if img_px is not None else None)

    # ---------- Rysowanie ----------

    def redraw(self) -> None:
        # sygnatura stanu — jeśli nic się nie zmieniło, pomijamy rysowanie
        W = max(1, self.winfo_width())
        H = max(1, self.winfo_height())
        sig = (self.orientation, W, H, round(self.zoom, 6), round(self.origin, 3),
               int(self.length_px), int(self.cursor) if self.cursor is not None else None)
        if sig == self._last_sig:
            return
        self._last_sig = sig

        self.delete("all")

        # Tło
        self.create_rectangle(0, 0, W, H, fill=self.bg, outline=self.bg)

        # Wygodny cel: odległość ~100 px między głównymi kreskami na ekranie
        desired_canvas = 110
        step_img = self._nice_step(max(1e-6, desired_canvas / self.zoom))

        # zakres w obraz-px aktualnie widoczny na linijce
        span_img = (W if self.orientation == "x" else H) / self.zoom
        start_img = self._floor_to(self.origin, step_img)
        end_img = self.origin + span_img + step_img

        # długości kresek (proporcje)
        major = 0.58
        mid = 0.36
        minor = 0.22

        img = start_img
        while img <= end_img:
            c = (img - self.origin) * self.zoom
            if self.orientation == "x":
                self._tick_x(c, H, major)
                self._label_x(c, H, img)
            else:
                self._tick_y(c, W, major)
                self._label_y(c, W, img)

            # podziałki pomocnicze (5/10)
            sub = 10 if step_img >= 10 else 5
            sub_step = step_img / sub
            for i in range(1, sub):
                sub_img = img + i * sub_step
                if sub_img >= end_img:
                    break
                cc = (sub_img - self.origin) * self.zoom
                if self.orientation == "x":
                    self._tick_x(cc, H, mid if (sub == 10 and i == 5) else minor)
                else:
                    self._tick_y(cc, W, mid if (sub == 10 and i == 5) else minor)
            img += step_img

        # wskaźnik kursora (jeśli jest)
        if self.cursor is not None:
            cc = (self.cursor - self.origin) * self.zoom
            if self.orientation == "x":
                self.create_line(cc, 0, cc, H, fill=self.fg)
            else:
                self.create_line(0, cc, W, cc, fill=self.fg)

    # ----- pomoc rysunku -----

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
    def _floor_to(x: float, step: float) -> float:
        return math.floor(x / step) * step
