# glitchlab/gui/panels/anisotropic_contour_warp_panel.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from glitchlab.gui.panel_base import FilterPanel, register_panel, coerce_value
from glitchlab.gui.controls import labeled_entry, labeled_checkbox, labeled_combo, section, BG

class AnisotropicContourWarpPanel(FilterPanel):
    FILTER_NAME = "anisotropic_contour_warp"

    def __init__(self):
        super().__init__()
        self.amp_px = tk.StringVar(value="24.0")
        self.cycles = tk.StringVar(value="10.0")
        self.phase_deg = tk.StringVar(value="0.0")
        self.border = tk.StringVar(value="wrap")
        self.use_ctx_amp = tk.BooleanVar(value=True)
        self.amp_gamma = tk.StringVar(value="1.2")
        self.mask_key = tk.StringVar(value="")

    def build(self, parent: tk.Widget) -> tk.Frame:
        root = tk.Frame(parent, bg=BG)

        sec = section(root, "Falowanie po konturach")
        sec.pack(fill="x", padx=2, pady=2)
        labeled_entry(sec, "amp_px", self.amp_px, on_change=self._changed)
        labeled_entry(sec, "cycles", self.cycles, on_change=self._changed)
        labeled_entry(sec, "phase_deg", self.phase_deg, on_change=self._changed)
        labeled_combo(sec, "border", self.border, ["wrap","clamp"], on_change=self._changed)

        sec2 = section(root, "Amplituda i maska")
        sec2.pack(fill="x", padx=2, pady=2)
        labeled_checkbox(sec2, "use_ctx_amp", self.use_ctx_amp, on_change=self._changed)
        labeled_entry(sec2, "amp_gamma", self.amp_gamma, on_change=self._changed)
        mask_keys = self.get_context().mask_keys or []
        labeled_combo(sec2, "mask_key", self.mask_key, [""] + mask_keys,
                      on_change=self._changed, readonly=False)

        return root

    def _changed(self):
        if self.on_change: self.on_change()

    def get_params(self) -> dict:
        mk = self.mask_key.get().strip()
        return {
            "amp_px": coerce_value(self.amp_px.get(), "float", 0.0, 240.0),
            "cycles": coerce_value(self.cycles.get(), "float", 1.0, 128.0),
            "phase_deg": coerce_value(self.phase_deg.get(), "float", 0.0, 360.0),
            "border": self.border.get(),
            "use_ctx_amp": bool(self.use_ctx_amp.get()),
            "amp_gamma": coerce_value(self.amp_gamma.get(), "float", 0.5, 3.0),
            "mask_key": mk if mk else None,
        }

register_panel(AnisotropicContourWarpPanel)
