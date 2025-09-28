# glitchlab/gui/views/bottom_area.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Optional

try:
    from glitchlab.gui.views.statusbar import StatusBar
except Exception:
    StatusBar = None  # type: ignore

try:
    from glitchlab.gui.views.bottom_panel import BottomPanel
except Exception:
    BottomPanel = None  # type: ignore


class BottomArea(ttk.Frame):
    """
    Kontener dolny: łączy BottomPanel i StatusBar w jednym miejscu.
    Styl: BottomArea.TFrame (kolor tła zgodny z motywem TFrame).
    """

    def __init__(self, master: tk.Misc, *, bus: Optional[Any] = None, default: str = "hud") -> None:
        super().__init__(master, style="BottomArea.TFrame")

        self.bus = bus

        # Panel główny (HUD/logi/diag)
        if BottomPanel:
            self.panel = BottomPanel(self, bus=bus, default=default)
        else:
            self.panel = ttk.Frame(self, style="BottomArea.TFrame")
        self.panel.pack(fill="both", expand=True)

        # Pasek statusu (zawsze na dole)
        if StatusBar:
            self.status = StatusBar(self, show_progress=True)
        else:
            self.status = ttk.Frame(self, height=24, style="BottomArea.TFrame")
        self.status.pack(fill="x", side="bottom")

        # Powiązanie z EventBus (opcjonalnie)
        if self.bus and StatusBar and hasattr(self.status, "bind_bus"):
            try:
                self.status.bind_bus(self.bus)
            except Exception:
                pass


def init_styles(root: tk.Tk) -> None:
    """
    Inicjalizuje styl BottomArea.TFrame tak, aby używał tego samego tła
    co zwykłe ramki TFrame. Należy wywołać raz przy starcie aplikacji.
    """
    style = ttk.Style(root)
    base_bg = style.lookup("TFrame", "background")
    if not base_bg:
        base_bg = "#f0f0f0"  # neutralny fallback
    style.configure("BottomArea.TFrame", background=base_bg)
