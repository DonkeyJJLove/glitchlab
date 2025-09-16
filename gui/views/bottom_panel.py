# glitchlab/gui/views/bottom_panel.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Optional

try:
    from glitchlab.gui.views.hud import HUDView as Hud
except Exception:
    Hud = None  # type: ignore

try:
    from glitchlab.gui.widgets.graph_view import GraphView
except Exception:
    GraphView = None  # type: ignore

try:
    from glitchlab.gui.widgets.mosaic_view import MosaicMini
except Exception:
    MosaicMini = None  # type: ignore

try:
    from glitchlab.gui.widgets.diag_console import DiagConsole
except Exception:
    DiagConsole = None  # type: ignore


class BottomPanel(ttk.Frame):
    """
    Zintegrowany dolny panel (zakładki HUD/Graph/Mosaic/Diagnostics).
    • Stała wysokość (scrollable content).
    • Sterowany radiobuttonami segmentowymi.
    API:
      - select(view_name)
      - set_ctx(ctx_or_cache)
      - log(msg)
      - set_visible(flag)
    """

    VIEWS = ("hud", "graph", "mosaic", "diag")
    PANEL_HEIGHT = 260  # stała wysokość panelu (px)

    def __init__(self, master: tk.Misc, *, bus: Optional[Any] = None, default: str = "hud"):
        super().__init__(master)
        self.bus = bus
        self._current = tk.StringVar(value=(default if default in self.VIEWS else "hud"))

        # Pasek przycisków
        bar = ttk.Frame(self)
        bar.pack(side="top", fill="x", padx=6, pady=(6, 4))

        def seg(text: str, val: str):
            rb = ttk.Radiobutton(
                bar, text=text, value=val, variable=self._current,
                style="Toolbutton", command=lambda v=val: self._on_select(v)
            )
            rb.pack(side="left", padx=(0, 4))
            return rb

        self._btn_hud = seg("HUD", "hud")
        self._btn_graph = seg("Graph", "graph")
        self._btn_mosaic = seg("Mosaic", "mosaic")
        self._btn_diag = seg("Diagnostics", "diag")

        ttk.Separator(self).pack(fill="x")

        # Obszar treści – kontener z przewijaniem
        self._scroll_canvas = tk.Canvas(self, height=self.PANEL_HEIGHT, highlightthickness=0, bg="#1a1a1a")
        self._scroll_canvas.pack(fill="x", expand=False, side="top")

        self._scroll_frame = ttk.Frame(self._scroll_canvas)
        self._scroll_win = self._scroll_canvas.create_window(0, 0, window=self._scroll_frame, anchor="nw")

        sb = ttk.Scrollbar(self, orient="vertical", command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        self._scroll_frame.bind(
            "<Configure>",
            lambda e: self._scroll_canvas.configure(
                scrollregion=self._scroll_canvas.bbox("all")
            )
        )

        # widoki wewnętrzne
        self._views: dict[str, tk.Widget] = {
            "hud": self._make_hud(self._scroll_frame),
            "graph": self._make_graph(self._scroll_frame),
            "mosaic": self._make_mosaic(self._scroll_frame),
            "diag": self._make_diag(self._scroll_frame),
        }

        self._show_only(self._current.get())

    # ---------- Tworzenie pod-widoków ----------

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

    # ---------- API ----------

    def select(self, view_name: str) -> None:
        if view_name in self.VIEWS:
            self._current.set(view_name)
            self._show_only(view_name)
            self._publish("ui.bottom.select", {"view": view_name})

    def set_ctx(self, ctx_or_cache: Any) -> None:
        """Przekaż ctx lub cache, aby odświeżyć HUD i Mosaic."""
        cache = None
        if isinstance(ctx_or_cache, dict):
            cache = ctx_or_cache
        else:
            cache = getattr(ctx_or_cache, "cache", None)

        if not isinstance(cache, dict):
            return

        try:
            hv = self._views.get("hud")
            if hv and hasattr(hv, "render_from_cache"):
                hv.render_from_cache(cache)
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
            self.pack(fill="x", expand=False, side="bottom")
        else:
            self.pack_forget()

    # ---------- Wewnętrzne ----------

    def _on_select(self, view_name: str) -> None:
        self._show_only(view_name)
        self._publish("ui.bottom.select", {"view": view_name})

    def _show_only(self, key: str) -> None:
        for name, widget in self._views.items():
            try:
                widget.pack_forget()
            except Exception:
                pass
        w = self._views.get(key)
        if w:
            w.pack(fill="both", expand=True)

    def _publish(self, topic: str, payload: dict) -> None:
        if self.bus is not None and hasattr(self.bus, "publish"):
            try:
                self.bus.publish(topic, dict(payload))
            except Exception:
                pass
