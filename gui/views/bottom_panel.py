# glitchlab/gui/views/bottom_panel.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Optional, Dict

# ─────────────────────────────────────────────────────────────────────────────
# Zależności opcjonalne (łagodne importy)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from glitchlab.gui.views.hud import HUDView as Hud
except Exception:  # pragma: no cover
    Hud = None  # type: ignore

try:
    from glitchlab.gui.widgets.graph_view import GraphView
except Exception:  # pragma: no cover
    GraphView = None  # type: ignore

try:
    from glitchlab.gui.widgets.mosaic_view import MosaicMini
except Exception:  # pragma: no cover
    MosaicMini = None  # type: ignore

try:
    from glitchlab.gui.widgets.diag_console import DiagConsole
except Exception:  # pragma: no cover
    DiagConsole = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Pomocniczy adapter: dict -> obiekt z atrybutem .cache
# ─────────────────────────────────────────────────────────────────────────────
class _DictCtx:
    __slots__ = ("cache",)
    def __init__(self, cache: Dict[str, Any]) -> None:
        self.cache = cache


# ─────────────────────────────────────────────────────────────────────────────
# BottomPanel
# ─────────────────────────────────────────────────────────────────────────────
class BottomPanel(ttk.Frame):
    """
    Zintegrowany dolny panel (zakładki HUD/Graph/Mosaic/Diagnostics).

    • Stała minimalna wysokość zawartości (nie „zapada się”).
    • Radiobuttony segmentowe do przełączania kart.
    • Tło zgodne z motywem (nie hardcodowane na czarno).
    """

    VIEWS = ("hud", "graph", "mosaic", "diag")
    PANEL_MIN_HEIGHT = 260  # px

    def __init__(self, master: tk.Misc, *, bus: Optional[Any] = None, default: str = "hud"):
        super().__init__(master, style="BottomArea.TFrame")

        self.bus = bus
        self._current = tk.StringVar(value=(default if default in self.VIEWS else "hud"))

        # ── Pasek przycisków ────────────────────────────────────────────────
        bar = ttk.Frame(self, style="BottomArea.TFrame")
        bar.pack(side="top", fill="x", padx=6, pady=(6, 4))

        def seg(text: str, val: str):
            rb = ttk.Radiobutton(
                bar, text=text, value=val, variable=self._current,
                style="Toolbutton", command=lambda v=val: self._on_select(v)
            )
            rb.pack(side="left", padx=(0, 4))
            return rb

        self._btn_hud    = seg("HUD", "hud")
        self._btn_graph  = seg("Graph", "graph")
        self._btn_mosaic = seg("Mosaic", "mosaic")
        self._btn_diag   = seg("Diagnostics", "diag")

        ttk.Separator(self).pack(fill="x")

        # ── Kontener widoków (Frame, nie Canvas z czarnym bg) ───────────────
        self._container = ttk.Frame(self, style="BottomArea.TFrame")
        self._container.pack(fill="both", expand=True)

        # ── Widoki wewnętrzne ───────────────────────────────────────────────
        self._views: dict[str, tk.Widget] = {
            "hud":    self._make_hud(self._container),
            "graph":  self._make_graph(self._container),
            "mosaic": self._make_mosaic(self._container),
            "diag":   self._make_diag(self._container),
        }

        # Startowy widok
        self._show_only(self._current.get())

    # ───────────────────── Tworzenie pod-widoków ─────────────────────────────
    def _make_hud(self, parent):
        if Hud is not None:
            w = Hud(parent)
            w.pack_forget()
            return w
        f = ttk.Frame(parent)
        ttk.Label(f, text="(HUD unavailable)").pack(padx=8, pady=8, anchor="w")
        f.pack_forget()
        return f

    def _make_graph(self, parent):
        if GraphView is not None:
            w = GraphView(parent)
            w.pack_forget()
            return w
        f = ttk.Frame(parent)
        ttk.Label(f, text="(Graph unavailable)").pack(padx=8, pady=8, anchor="w")
        f.pack_forget()
        return f

    def _make_mosaic(self, parent):
        if MosaicMini is not None:
            w = MosaicMini(parent)
            w.pack_forget()
            return w
        f = ttk.Frame(parent)
        ttk.Label(f, text="(Mosaic unavailable)").pack(padx=8, pady=8, anchor="w")
        f.pack_forget()
        return f

    def _make_diag(self, parent):
        if DiagConsole is not None:
            w = DiagConsole(parent)
            if self.bus is not None:
                try:
                    w.attach_bus(self.bus)
                except Exception:
                    pass
            w.pack_forget()
            return w
        f = ttk.Frame(parent)
        ttk.Label(f, text="(Diagnostics unavailable)").pack(padx=8, pady=8, anchor="w")
        f.pack_forget()
        return f

    # ───────────────────────────── API publiczne ─────────────────────────────
    def select(self, view_name: str) -> None:
        if view_name in self.VIEWS:
            self._current.set(view_name)
            self._show_only(view_name)
            self._publish("ui.bottom.select", {"view": view_name})

    def set_ctx(self, ctx_or_cache: Any) -> None:
        if isinstance(ctx_or_cache, dict):
            cache: Optional[Dict[str, Any]] = ctx_or_cache
            ctx_for_hud: Any = _DictCtx(cache)
        else:
            cache = getattr(ctx_or_cache, "cache", None)
            ctx_for_hud = ctx_or_cache

        if not isinstance(cache, dict):
            return

        try:
            hv = self._views.get("hud")
            if hv and hasattr(hv, "render_from_cache"):
                hv.render_from_cache(ctx_for_hud)
        except Exception:
            pass

        try:
            mv = self._views.get("mosaic")
            if mv and hasattr(mv, "render_from_cache"):
                mv.render_from_cache(cache)
        except Exception:
            pass

    def log(self, msg: str) -> None:
        try:
            diag = self._views.get("diag")
            if diag and hasattr(diag, "log"):
                diag.log("DEBUG", msg)
        except Exception:
            pass

    def set_visible(self, flag: bool) -> None:
        if flag:
            self.pack(fill="both", expand=True, side="top")
        else:
            self.pack_forget()

    # ───────────────────────────── Pomocnicze ────────────────────────────────
    def _on_select(self, view_name: str) -> None:
        self._show_only(view_name)
        self._publish("ui.bottom.select", {"view": view_name})

    def _show_only(self, key: str) -> None:
        for _, widget in self._views.items():
            try:
                widget.pack_forget()
            except Exception:
                pass

        w = self._views.get(key)
        if w:
            w.pack(fill="both", expand=True, padx=6, pady=6, anchor="center")
        self.update_idletasks()

    def _publish(self, topic: str, payload: dict) -> None:
        if self.bus is not None and hasattr(self.bus, "publish"):
            try:
                self.bus.publish(topic, dict(payload))
            except Exception:
                pass
