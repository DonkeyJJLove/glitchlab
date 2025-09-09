# glitchlab/gui/panels/phase_glitch_panel.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from glitchlab.gui.panel_base import FilterPanel, register_panel, coerce_value
from glitchlab.gui.controls import labeled_entry, labeled_checkbox, labeled_combo, section, BG

class PhaseGlitchPanel(FilterPanel):
    FILTER_NAME = "phase_glitch"

    def __init__(self):
        super().__init__()
        self.low = tk.StringVar(value="0.20")
        self.high = tk.StringVar(value="0.60")
        self.strength = tk.StringVar(value="0.70")
        self.preserve_dc = tk.BooleanVar(value=True)
        self.blend = tk.StringVar(value="0.15")
        self.mask_key = tk.StringVar(value="")

    def build(self, parent: tk.Widget) -> tk.Frame:
        root = tk.Frame(parent, bg=BG)

        sec = section(root, "Pasmo i siła")
        sec.pack(fill="x", padx=2, pady=2)
        labeled_entry(sec, "low", self.low, on_change=self._changed)
        labeled_entry(sec, "high", self.high, on_change=self._changed)
        labeled_entry(sec, "strength", self.strength, on_change=self._changed)
        labeled_checkbox(sec, "preserve_dc", self.preserve_dc, on_change=self._changed)

        sec2 = section(root, "Miks i maska")
        sec2.pack(fill="x", padx=2, pady=2)
        labeled_entry(sec2, "blend", self.blend, on_change=self._changed)
        mask_keys = self.get_context().mask_keys or []
        labeled_combo(sec2, "mask_key", self.mask_key, [""] + mask_keys,
                      on_change=self._changed, readonly=False)

        return root

    def _changed(self):
        if self.on_change: self.on_change()

    def get_params(self) -> dict:
        mk = self.mask_key.get().strip()
        return {
            "low": coerce_value(self.low.get(), "float", 0.0, 1.0),
            "high": coerce_value(self.high.get(), "float", 0.0, 1.0),
            "strength": coerce_value(self.strength.get(), "float", 0.0, 1.0),
            "preserve_dc": bool(self.preserve_dc.get()),
            "blend": coerce_value(self.blend.get(), "float", 0.0, 1.0),
            "mask_key": mk if mk else None,
        }

register_panel(PhaseGlitchPanel)
