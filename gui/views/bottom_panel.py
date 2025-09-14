"""
---
version: 4
kind: module
id: "view-bottom-panel"
name: "glitchlab.gui.views.bottom_panel"
author: "GlitchLab v4"
role: "Dolny panel: status/progress/logi + przyciski run/cancel"
description: >
  Kontener znajdujący się na dole głównego okna. Pokazuje status pracy,
  pasek postępu, ostatnie komunikaty oraz podstawowe akcje uruchomieniowe.
inputs:
  master: {type: "tk.Misc"}
  bus: {type: "EventBus-like", desc: "publish(topic, payload)"}
  runner: {type: "PipelineRunner", optional: true}
  state: {type: "UiState-like", optional: true}
outputs:
  events:
    - "ui.run.apply"     # {}
    - "ui.run.cancel"    # {}
    - "ui.log.show"      # {level,msg}
    - "ui.status.set"    # {text}
    - "ui.progress.set"  # {value:0..1}
interfaces:
  exports:
    - "BottomPanel(master, bus, runner=None, state=None)"
    - "BottomPanel.set_status(text: str) -> None"
    - "BottomPanel.set_progress(value: float) -> None"
    - "BottomPanel.append_log(level: str, msg: str) -> None"
depends_on: ["tkinter", "tkinter.ttk", "typing"]
used_by: ["glitchlab.gui.app"]
policy:
  deterministic: true
  ui_thread_only: true
constraints:
  - "Żadnej logiki pipeline – tylko delegacja przez Bus/Runner"
license: "Proprietary"
---
"""
# glitchlab/gui/views/bottom_panel.py

# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Optional

try:
    from glitchlab.gui.widgets.hud import Hud
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


class BottomPanel(ttk.Frame):
    """
    Jeden, zintegrowany panel na dole okna.
    Zawiera przełączane widoki: HUD | Graph | Mosaic | Tech (log).
    Nie używa zakładek – przełączanie odbywa się przez przyciski segmentowe.
    Publiczne API (wycinek):
      - select(view_name) -> None                 # "hud"|"graph"|"mosaic"|"tech"
      - set_ctx(ctx) -> None                      # odśwież HUD/Mosaic z cache
      - log(msg: str) -> None                     # dopisz do logu
      - set_visible(flag: bool) -> None
    """

    VIEWS = ("hud", "graph", "mosaic", "tech")

    def __init__(self, master: tk.Misc, *, bus: Optional[Any] = None, default: str = "hud"):
        super().__init__(master)
        self.bus = bus
        self._current = tk.StringVar(value=(default if default in self.VIEWS else "hud"))

        # ---- Pasek przycisków (segmenty) ----
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
        self._btn_tech = seg("Tech", "tech")

        ttk.Separator(self).pack(fill="x")

        # ---- Obszar treści ----
        self.content = ttk.Frame(self)
        self.content.pack(fill="both", expand=True)

        self._views = {
            "hud": self._make_hud(self.content),
            "graph": self._make_graph(self.content),
            "mosaic": self._make_mosaic(self.content),
            "tech": self._make_tech(self.content),
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

    def _make_tech(self, parent):
        f = ttk.Frame(parent)
        self._log_text = tk.Text(
            f, height=8, bg="#141414", fg="#e6e6e6", insertbackground="#e6e6e6"
        )
        self._log_text.pack(fill="both", expand=True)
        f.pack_forget()
        return f

    # ---------- API ----------

    def select(self, view_name: str) -> None:
        if view_name in self.VIEWS:
            self._current.set(view_name)
            self._show_only(view_name)
            self._publish("ui.bottom.select", {"view": view_name})

    def set_ctx(self, ctx: Any) -> None:
        """Podaj kontekst run (z .cache), żeby odświeżyć HUD/Mosaic."""
        try:
            hv = self._views.get("hud")
            if hv and hasattr(hv, "render_from_cache"):
                hv.render_from_cache(ctx)
        except Exception:
            pass
        try:
            mv = self._views.get("mosaic")
            if mv and hasattr(mv, "render_from_cache"):
                mv.render_from_cache(ctx)
        except Exception:
            pass

    def log(self, msg: str) -> None:
        try:
            t = self._log_text
            t.insert("end", str(msg) + "\n")
            t.see("end")
        except Exception:
            pass

    def set_visible(self, flag: bool) -> None:
        if flag:
            self.pack(fill="both", expand=False, side="bottom")
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
