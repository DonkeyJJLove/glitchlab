# glitchlab/app/widgets/toolbox.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional


class Toolbox(ttk.Frame):
    """
    Poziomy toolbox do sterowania interakcją widoku i overlayami.
    Bezpieczny pod względem re-entrancy (set_* nie emituje callbacków).

    API (nowe i legacy):
      - on_tool_changed(tool: str)
      - on_toggle_crosshair(state: bool)
      - on_toggle_rulers(state: bool)
      - legacy: on_mode_change ≡ on_tool_changed
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_tool_changed: Optional[Callable[[str], None]] = None,
        on_toggle_crosshair: Optional[Callable[[bool], None]] = None,
        on_toggle_rulers: Optional[Callable[[bool], None]] = None,
        # legacy alias
        on_mode_change: Optional[Callable[[str], None]] = None,
        **_,
    ) -> None:
        super().__init__(parent)
        # normalizacja aliasów
        if on_tool_changed is None and on_mode_change is not None:
            on_tool_changed = on_mode_change

        self.on_tool_changed = on_tool_changed
        self.on_toggle_crosshair = on_toggle_crosshair
        self.on_toggle_rulers = on_toggle_rulers

        # stan
        self._suspend_emit = False
        self.tool_var = tk.StringVar(value="hand")
        self.cross_var = tk.BooleanVar(value=True)
        self.rulers_var = tk.BooleanVar(value=True)

        self._build_ui()

    # ---- UI ----
    def _build_ui(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(fill="x")

        # Ikony: używamy krótkich etykiet; styl Toolbutton, poziomo
        def rb(value: str, text: str, tip: str) -> ttk.Radiobutton:
            b = ttk.Radiobutton(bar, value=value, text=text, variable=self.tool_var, style="Toolbutton",
                                command=self._on_tool_change)
            b.pack(side="left", padx=2)
            return b

        def chk(var: tk.BooleanVar, text: str, cmd) -> ttk.Checkbutton:
            b = ttk.Checkbutton(bar, text=text, variable=var, style="Toolbutton", command=cmd)
            b.pack(side="left", padx=2)
            return b

        self._rb_hand = rb("hand", "🖐", "Pan/drag")
        self._rb_zoom = rb("zoom", "🔍", "Zoom")
        self._rb_pick = rb("pick", "🎯", "Pick")
        self._rb_meas = rb("measure", "📏", "Measure")

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=4)
        self._cb_cross = chk(self.cross_var, "Crosshair", self._on_cross_toggle)
        self._cb_rulers = chk(self.rulers_var, "Rulers", self._on_rulers_toggle)

    # ---- Handlers ----
    def _on_tool_change(self) -> None:
        if self._suspend_emit:
            return
        cb = self.on_tool_changed
        if callable(cb):
            try:
                cb(self.tool_var.get())
            except Exception:
                pass

    def _on_cross_toggle(self) -> None:
        if self._suspend_emit:
            return
        cb = self.on_toggle_crosshair
        if callable(cb):
            try:
                cb(bool(self.cross_var.get()))
            except Exception:
                pass

    def _on_rulers_toggle(self) -> None:
        if self._suspend_emit:
            return
        cb = self.on_toggle_rulers
        if callable(cb):
            try:
                cb(bool(self.rulers_var.get()))
            except Exception:
                pass

    # ---- Programmatic sync (bez emisji) ----
    def set_tool(self, tool: str) -> None:
        with self._mute():
            if tool in {"hand", "zoom", "pick", "measure"}:
                self.tool_var.set(tool)

    def set_cross(self, state: bool) -> None:
        with self._mute():
            self.cross_var.set(bool(state))

    def set_rulers(self, state: bool) -> None:
        with self._mute():
            self.rulers_var.set(bool(state))

    # ---- context manager ----
    from contextlib import contextmanager
    @contextmanager
    def _mute(self):
        prev = self._suspend_emit
        self._suspend_emit = True
        try:
            yield
        finally:
            self._suspend_emit = prev
