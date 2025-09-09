# glitchlab/gui/panels/pixel_sort_adaptive_panel.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from glitchlab.gui.panel_base import FilterPanel, register_panel, coerce_value
from glitchlab.gui.controls import labeled_entry, labeled_combo, labeled_checkbox, section, BG

class PixelSortAdaptivePanel(FilterPanel):
    FILTER_NAME = "pixel_sort_adaptive"

    def __init__(self):
        super().__init__()
        self.direction = tk.StringVar(value="vertical")
        self.trigger = tk.StringVar(value="edges")
        self.threshold = tk.StringVar(value="0.35")
        self.mask_key = tk.StringVar(value="")
        self.length_px = tk.StringVar(value="160")
        self.length_gain = tk.StringVar(value="1.20")
        self.prob = tk.StringVar(value="0.90")
        self.key = tk.StringVar(value="luma")
        self.reverse = tk.BooleanVar(value=False)

    def build(self, parent: tk.Widget) -> tk.Frame:
        root = tk.Frame(parent, bg=BG)

        sec_a = section(root, "Kierunek i wyzwalanie")
        sec_a.pack(fill="x", padx=2, pady=2)
        labeled_combo(sec_a, "direction", self.direction, ["vertical","horizontal"], on_change=self._changed)
        labeled_combo(sec_a, "trigger", self.trigger, ["edges","luma","mask"], on_change=self._changed)
        labeled_entry(sec_a, "threshold", self.threshold, on_change=self._changed)

        sec_mask = section(root, "Maska (dla trigger='mask')")
        sec_mask.pack(fill="x", padx=2, pady=2)
        mask_keys = self.get_context().mask_keys or []
        labeled_combo(sec_mask, "mask_key", self.mask_key, [""] + mask_keys,
                      on_change=self._changed, readonly=False)

        sec_len = section(root, "Segmenty i prawdopodobieństwo")
        sec_len.pack(fill="x", padx=2, pady=2)
        labeled_entry(sec_len, "length_px", self.length_px, on_change=self._changed)
        labeled_entry(sec_len, "length_gain", self.length_gain, on_change=self._changed)
        labeled_entry(sec_len, "prob", self.prob, on_change=self._changed)

        sec_key = section(root, "Klucz sortowania")
        sec_key.pack(fill="x", padx=2, pady=2)
        labeled_combo(sec_key, "key", self.key, ["luma","r","g","b","sat","hue"], on_change=self._changed)
        labeled_checkbox(sec_key, "reverse", self.reverse, on_change=self._changed)

        return root

    def _changed(self):
        if self.on_change: self.on_change()

    def get_params(self) -> dict:
        mk = self.mask_key.get().strip()
        return {
            "direction": self.direction.get(),
            "trigger": self.trigger.get(),
            "threshold": coerce_value(self.threshold.get(), "float", 0.0, 1.0),
            "mask_key": mk if mk else None,
            "length_px": coerce_value(self.length_px.get(), "int", 1, 4096),
            "length_gain": coerce_value(self.length_gain.get(), "float", 0.0, 4.0),
            "prob": coerce_value(self.prob.get(), "float", 0.0, 1.0),
            "key": self.key.get(),
            "reverse": bool(self.reverse.get()),
        }

register_panel(PixelSortAdaptivePanel)
