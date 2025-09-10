# gui/panels/anisotropic_contour_warp.py
from __future__ import annotations
from tkinter import ttk
from .base import PanelBase
from .registry import register_panel

class AnisotropicContourWarpPanel(PanelBase):
    def __init__(self, master, on_apply=None):
        super().__init__(master, on_apply)
        ttk.Label(self, text="anisotropic_contour_warp", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0,4))

        self.add_num("strength", "strength", 1.5)
        self.add_int("ksize", "ksize", 3)
        self.add_int("iters", "iters", 1)
        self.add_num("smooth", "smooth", 0.0)
        self.add_num("edge_bias", "edge_bias", 0.0)
        self.add_text("mask_key", "mask_key", "")
        self.add_num("use_amp", "use_amp", 1.0)
        self.add_bool("clamp", "clamp", True)

        self.btn_apply("anisotropic_contour_warp")

register_panel("anisotropic_contour_warp", AnisotropicContourWarpPanel)
