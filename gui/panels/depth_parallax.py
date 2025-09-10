# gui/panels/depth_parallax.py
from __future__ import annotations
from tkinter import ttk
from .base import PanelBase
from .registry import register_panel

class DepthParallaxPanel(PanelBase):
    def __init__(self, master, on_apply=None):
        super().__init__(master, on_apply)
        ttk.Label(self, text="depth_parallax", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0,4))

        self.add_num("scale", "scale", 70.0)
        self.add_num("freq", "freq", 100.0)
        self.add_int("octaves", "octaves", 5)
        self.add_bool("shading", "shading", True)
        self.add_bool("stereo", "stereo", True)

        self.btn_apply("depth_parallax")

register_panel("depth_parallax", DepthParallaxPanel)
