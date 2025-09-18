# glitchlab/gui/widgets/overlay_renderer.py
# -*- coding: utf-8 -*-
"""
overlay_renderer — wspólne utilsy do rysowania overlay’ów na tk.Canvas.

Założenia:
- Nie trzyma stanu obrazu; działa wyłącznie na podanym `tk.Canvas`.
- Skaluje grubości linii/uchwyty względem zoomu (param `px` = 1 px w przestrzeni ekranu).
- Grupuje elementy przez tagi (np. "tool_rect", "measure"), by łatwo je czyścić.
- Zapewnia czytelne etykiety (tekst + tło) i uchwyty (drag handles).

API (najważniejsze):
    px = OverlayRenderer.pixel(world_zoom: float) -> float
    draw_rect(cv, x0, y0, x1, y1, tag="tool_rect", zoom=1.0, style=None) -> dict[str,int]
    draw_ellipse(cv, x0, y0, x1, y1, tag="tool_ellipse", zoom=1.0, style=None) -> dict[str,int]
    draw_line_with_label(cv, x0, y0, x1, y1, label, tag="measure", zoom=1.0, style=None) -> dict[str,int]
    draw_handles(cv, bbox, tag="handles", zoom=1.0, style=None) -> list[tuple[str,int,int,int]]
    hit_test_handles(x, y, handles, tol=6) -> Optional[str]
    clear(cv, tag)

Style – domyślne:
    stroke="#65b0ff", fill="", dash=(4,2), handle_fill="#fffb", label_bg="#000b", label_fg="#fff"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Any
import tkinter as tk


# ──────────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OverlayStyle:
    stroke: str = "#65b0ff"
    stroke_minor: str = "#9ac9ff"
    fill: str = ""
    dash: Tuple[int, int] = (4, 2)
    width_px: float = 1.5
    handle_size_px: float = 6.0
    handle_fill: str = "#66aaffcc"   # półprzezr.
    handle_stroke: str = "#66aaff"
    label_bg: str = "#000000b0"
    label_fg: str = "#ffffff"
    label_pad_px: float = 4.0
    font: Tuple[str, int] = ("", 9)


DEFAULT_STYLE = OverlayStyle()


# ──────────────────────────────────────────────────────────────────────────────
# Utilsy
# ──────────────────────────────────────────────────────────────────────────────

class OverlayRenderer:
    """Statyczne utilsy do rysowania overlay na tk.Canvas."""

    # ── skala 1-pixelowa (względem zoom) ────────────────────────────────────
    @staticmethod
    def pixel(zoom: float) -> float:
        """Zwraca „1px” w koordach canvasa tak, by linie były czytelne przy różnych zoomach."""
        zoom = float(max(1e-6, zoom))
        # Utrzymujemy mniej więcej stałą grubość na ekranie
        return 1.0 / zoom

    # ── czyszczenie grupy ────────────────────────────────────────────────────
    @staticmethod
    def clear(cv: tk.Canvas, tag: str) -> None:
        try:
            cv.delete(tag)
        except Exception:
            pass

    # ── prostokąt ────────────────────────────────────────────────────────────
    @staticmethod
    def draw_rect(
        cv: tk.Canvas,
        x0: float, y0: float, x1: float, y1: float,
        *, tag: str = "tool_rect",
        zoom: float = 1.0,
        style: Optional[OverlayStyle] = None,
        with_cross: bool = True,
        with_ticks: bool = True,
    ) -> Dict[str, int]:
        st = style or DEFAULT_STYLE
        px = OverlayRenderer.pixel(zoom)
        x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
        if x1 < x0: x0, x1 = x1, x0
        if y1 < y0: y0, y1 = y1, y0

        ids: Dict[str, int] = {}
        # obrys
        ids["rect"] = cv.create_rectangle(
            x0, y0, x1, y1,
            outline=st.stroke, width=max(1.0, st.width_px * px), dash=st.dash if st.dash else None,
            fill=st.fill, tags=(tag,),
        )
        # przekątne (opcjonalnie)
        if with_cross:
            ids["diag1"] = cv.create_line(x0, y0, x1, y1, fill=st.stroke_minor,
                                          width=max(1.0, (st.width_px * 0.75) * px), tags=(tag,))
            ids["diag2"] = cv.create_line(x0, y1, x1, y0, fill=st.stroke_minor,
                                          width=max(1.0, (st.width_px * 0.75) * px), tags=(tag,))
        # ticki (opcjonalnie)
        if with_ticks:
            t = 6 * px
            ids["tick_top"]    = cv.create_line(x0, y0, x0 + t, y0, fill=st.stroke_minor, tags=(tag,))
            ids["tick_left"]   = cv.create_line(x0, y0, x0, y0 + t, fill=st.stroke_minor, tags=(tag,))
            ids["tick_bottom"] = cv.create_line(x0, y1, x0 + t, y1, fill=st.stroke_minor, tags=(tag,))
            ids["tick_right"]  = cv.create_line(x1, y0, x1, y0 + t, fill=st.stroke_minor, tags=(tag,))
        return ids

    # ── elipsa ────────────────────────────────────────────────────────────────
    @staticmethod
    def draw_ellipse(
        cv: tk.Canvas,
        x0: float, y0: float, x1: float, y1: float,
        *, tag: str = "tool_ellipse",
        zoom: float = 1.0,
        style: Optional[OverlayStyle] = None,
    ) -> Dict[str, int]:
        st = style or DEFAULT_STYLE
        px = OverlayRenderer.pixel(zoom)
        x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
        if x1 < x0: x0, x1 = x1, x0
        if y1 < y0: y0, y1 = y1, y0

        ids: Dict[str, int] = {}
        ids["oval"] = cv.create_oval(
            x0, y0, x1, y1,
            outline=st.stroke, width=max(1.0, st.width_px * px), dash=st.dash if st.dash else None,
            fill=st.fill, tags=(tag,),
        )
        # osie
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        ids["axis_h"] = cv.create_line(x0, cy, x1, cy, fill=st.stroke_minor,
                                       width=max(1.0, (st.width_px * 0.75) * px), tags=(tag,))
        ids["axis_v"] = cv.create_line(cx, y0, cx, y1, fill=st.stroke_minor,
                                       width=max(1.0, (st.width_px * 0.75) * px), tags=(tag,))
        return ids

    # ── linia + etykieta (np. measure) ───────────────────────────────────────
    @staticmethod
    def draw_line_with_label(
        cv: tk.Canvas,
        x0: float, y0: float, x1: float, y1: float,
        label: str,
        *, tag: str = "measure",
        zoom: float = 1.0,
        style: Optional[OverlayStyle] = None,
    ) -> Dict[str, int]:
        st = style or DEFAULT_STYLE
        px = OverlayRenderer.pixel(zoom)

        ids: Dict[str, int] = {}
        ids["line"] = cv.create_line(
            x0, y0, x1, y1,
            fill=st.stroke, width=max(1.0, st.width_px * px), tags=(tag,)
        )
        # małe „główki” na końcach
        hl = 5 * px
        ids["cap0h"] = cv.create_line(x0 - hl, y0, x0 + hl, y0, fill=st.stroke_minor, tags=(tag,))
        ids["cap0v"] = cv.create_line(x0, y0 - hl, x0, y0 + hl, fill=st.stroke_minor, tags=(tag,))
        ids["cap1h"] = cv.create_line(x1 - hl, y1, x1 + hl, y1, fill=st.stroke_minor, tags=(tag,))
        ids["cap1v"] = cv.create_line(x1, y1 - hl, x1, y1 + hl, fill=st.stroke_minor, tags=(tag,))

        # etykieta w połowie
        mx = (x0 + x1) * 0.5
        my = (y0 + y1) * 0.5
        ids.update(OverlayRenderer._label(cv, mx, my, label, tag=tag, zoom=zoom, style=st))
        return ids

    # ── uchwyty (drag handles) ───────────────────────────────────────────────
    @staticmethod
    def draw_handles(
        cv: tk.Canvas,
        bbox: Tuple[float, float, float, float],
        *, tag: str = "handles",
        zoom: float = 1.0,
        style: Optional[OverlayStyle] = None,
        corners_only: bool = False,
    ) -> List[Tuple[str, int, int, int]]:
        """
        Rysuje uchwyty (kwadraty) i zwraca listę: [(name, x, y, id), ...]
        Nazwy: 'nw','n','ne','w','e','sw','s','se' (lub tylko rogi, jeśli corners_only=True)
        """
        st = style or DEFAULT_STYLE
        px = OverlayRenderer.pixel(zoom)
        hs = max(4.0, st.handle_size_px * px)
        x0, y0, x1, y1 = bbox
        if x1 < x0: x0, x1 = x1, x0
        if y1 < y0: y0, y1 = y1, y0
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5

        pts = [
            ("nw", x0, y0), ("n", cx, y0), ("ne", x1, y0),
            ("w", x0, cy),                 ("e", x1, cy),
            ("sw", x0, y1), ("s", cx, y1), ("se", x1, y1),
        ]
        if corners_only:
            pts = [p for p in pts if p[0] in ("nw", "ne", "sw", "se")]

        out: List[Tuple[str, int, int, int]] = []
        for name, x, y in pts:
            idr = cv.create_rectangle(
                x - hs, y - hs, x + hs, y + hs,
                outline=st.handle_stroke, fill=st.handle_fill, width=max(1.0, 1.0 * px), tags=(tag,)
            )
            out.append((name, int(x), int(y), idr))
        return out

    @staticmethod
    def hit_test_handles(
        x: int, y: int,
        handles: List[Tuple[str, int, int, int]],
        tol: int = 6
    ) -> Optional[str]:
        """Zwraca nazwę uchwytu pod kursorem (albo None)."""
        t2 = max(2, int(tol)) ** 2
        for name, hx, hy, _id in handles:
            dx = hx - int(x)
            dy = hy - int(y)
            if (dx * dx + dy * dy) <= t2:
                return name
        return None

    # ── prywatne: etykieta z tłem ───────────────────────────────────────────
    @staticmethod
    def _label(
        cv: tk.Canvas,
        x: float, y: float, text: str,
        *, tag: str, zoom: float, style: OverlayStyle
    ) -> Dict[str, int]:
        px = OverlayRenderer.pixel(zoom)
        pad = max(2.0, style.label_pad_px * px)
        # najpierw tekst, żeby poznać jego bbox, potem tło pod spód
        tid = cv.create_text(x, y, text=str(text), fill=style.label_fg,
                             font=style.font, anchor="center", tags=(tag,))
        try:
            bb = cv.bbox(tid)  # (x0,y0,x1,y1)
        except Exception:
            bb = None
        if bb:
            x0, y0, x1, y1 = bb
            bid = cv.create_rectangle(x0 - pad, y0 - pad, x1 + pad, y1 + pad,
                                      fill=style.label_bg, outline="", tags=(tag,))
            # przepchnij tło pod spód
            cv.tag_lower(bid, tid)
            return {"label_bg": bid, "label": tid}
        return {"label": tid}
