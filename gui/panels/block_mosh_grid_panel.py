# glitchlab/gui/panels/block_mosh_grid_panel.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from glitchlab.gui.panel_base import FilterPanel, register_panel, coerce_value
from glitchlab.gui.controls import labeled_entry, labeled_checkbox, labeled_combo, section, BG

class BlockMoshGridPanel(FilterPanel):
    FILTER_NAME = "block_mosh_grid"

    def __init__(self):
        super().__init__()
        self.block = tk.StringVar(value="12")
        self.max_shift = tk.StringVar(value="64")
        self.prob = tk.StringVar(value="0.85")
        self.shuffle = tk.BooleanVar(value=True)
        self.wrap = tk.BooleanVar(value=True)
        self.amp_scale = tk.StringVar(value="1.40")
        self.mask_key = tk.StringVar(value="")

    def build(self, parent: tk.Widget) -> tk.Frame:
        root = tk.Frame(parent, bg=BG)

        sec_a = section(root, "Parametry siatki")
        sec_a.pack(fill="x", padx=2, pady=2)
        labeled_entry(sec_a, "block", self.block, on_change=self._changed)
        labeled_entry(sec_a, "max_shift", self.max_shift, on_change=self._changed)
        labeled_entry(sec_a, "prob", self.prob, on_change=self._changed)
        labeled_checkbox(sec_a, "shuffle", self.shuffle, on_change=self._changed)
        labeled_checkbox(sec_a, "wrap", self.wrap, on_change=self._changed)

        sec_b = section(root, "Amplituda i maska")
        sec_b.pack(fill="x", padx=2, pady=2)
        labeled_entry(sec_b, "amp_scale", self.amp_scale, on_change=self._changed)
        mask_keys = self.get_context().mask_keys or []
        labeled_combo(sec_b, "mask_key", self.mask_key, [""] + mask_keys,
                      on_change=self._changed, readonly=False)

        return root

    def _changed(self):
        if self.on_change: self.on_change()

    def get_params(self) -> dict:
        mk = self.mask_key.get().strip()
        return {
            "block": coerce_value(self.block.get(), "int", 2, 512),
            "max_shift": coerce_value(self.max_shift.get(), "int", 0, 1024),
            "prob": coerce_value(self.prob.get(), "float", 0.0, 1.0),
            "shuffle": bool(self.shuffle.get()),
            "wrap": bool(self.wrap.get()),
            "amp_scale": coerce_value(self.amp_scale.get(), "float", 0.0, 8.0),
            "mask_key": mk if mk else None,
        }

register_panel(BlockMoshGridPanel)
