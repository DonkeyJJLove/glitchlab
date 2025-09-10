# gui/panels/depth_displace.py
from __future__ import annotations
from tkinter import ttk
from .base import PanelBase
from .registry import register_panel

class DepthDisplacePanel(PanelBase):
    def __init__(self, master, on_apply=None):
        super().__init__(master, on_apply)
        ttk.Label(self, text="depth_displace", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0,4))

        self.add_text("depth_map", "depth_map", "noise_fractal")
        self.add_num("scale", "scale", 56.0)
        self.add_num("freq", "freq", 110.0)
        self.add_int("octaves", "octaves", 5)
        self.add_num("vertical", "vertical", 0.15)
        self.add_bool("stereo", "stereo", True)
        self.add_int("stereo_px", "stereo_px", 2)
        self.add_bool("shading", "shading", True)
        self.add_num("shade_gain", "shade_gain", 0.25)
        self.add_text("mask_key", "mask_key", "")

        self.btn_apply("depth_displace")

register_panel("depth_displace", DepthDisplacePanel)
