# gui/panels/rgb_offset.py
from __future__ import annotations
from tkinter import ttk
from .base import PanelBase
from .registry import register_panel

class RgbOffsetPanel(PanelBase):
    def __init__(self, master, on_apply=None):
        super().__init__(master, on_apply)
        ttk.Label(self, text="rgb_offset", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0,4))

        # Offsets kanałów jako (dx, dy) — wpisywane jako tekst "x,y"
        self.add_text("R (dx,dy)", "r", "12,0")
        self.add_text("G (dx,dy)", "g", "-10,0")
        self.add_text("B (dx,dy)", "b", "20,0")

        self.btn_apply("rgb_offset")

    def to_params(self):
        p = super().to_params()
        def _pair(s: str):
            try:
                x, y = (w.strip() for w in str(s).split(","))
                return [int(x), int(y)]
            except Exception:
                return [0, 0]
        return {
            "r": _pair(p.get("r","0,0")),
            "g": _pair(p.get("g","0,0")),
            "b": _pair(p.get("b","0,0")),
        }

register_panel("rgb_offset", RgbOffsetPanel)
