# glitchlab/gui/panels/spectral_shaper_panel.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from glitchlab.gui.panel_base import FilterPanel, register_panel, coerce_value
from glitchlab.gui.controls import labeled_entry, labeled_combo, section, BG

class SpectralShaperPanel(FilterPanel):
    FILTER_NAME = "spectral_shaper"

    def __init__(self):
        super().__init__()
        self.mode = tk.StringVar(value="ring")
        self.low = tk.StringVar(value="0.12")
        self.high = tk.StringVar(value="0.32")
        self.angle_deg = tk.StringVar(value="0.0")
        self.ang_width = tk.StringVar(value="20.0")
        self.boost = tk.StringVar(value="1.2")
        self.soft = tk.StringVar(value="0.06")
        self.blend = tk.StringVar(value="0.10")
        self.mask_key = tk.StringVar(value="")

    def build(self, parent: tk.Widget) -> tk.Frame:
        root = tk.Frame(parent, bg=BG)

        # Band (radial)
        sec_band = section(root, "Pasmo radialne")
        sec_band.pack(fill="x", padx=2, pady=2)
        labeled_combo(sec_band, "mode", self.mode, ["ring","bandpass","bandstop","direction"],
                      on_change=self._changed)
        labeled_entry(sec_band, "low", self.low, on_change=self._changed)
        labeled_entry(sec_band, "high", self.high, on_change=self._changed)

        # Angular (direction)
        sec_ang = section(root, "Kierunek (dla 'direction')")
        sec_ang.pack(fill="x", padx=2, pady=2)
        labeled_entry(sec_ang, "angle_deg", self.angle_deg, on_change=self._changed)
        labeled_entry(sec_ang, "ang_width", self.ang_width, on_change=self._changed)

        # Gain / Mix
        sec_mix = section(root, "Wzmocnienie i miks")
        sec_mix.pack(fill="x", padx=2, pady=2)
        labeled_entry(sec_mix, "boost", self.boost, on_change=self._changed)
        labeled_entry(sec_mix, "soft", self.soft, on_change=self._changed)
        labeled_entry(sec_mix, "blend", self.blend, on_change=self._changed)

        # Mask
        sec_mask = section(root, "Maska (opcjonalnie)")
        sec_mask.pack(fill="x", padx=2, pady=2)
        # wypełnij mask_keys z kontekstu
        self.mask_key.set("")
        mask_keys = self.get_context().mask_keys or []
        labeled_combo(sec_mask, "mask_key", self.mask_key, [""] + mask_keys,
                      on_change=self._changed, readonly=False)

        return root

    def _changed(self):
        if self.on_change: self.on_change()

    def get_params(self) -> dict:
        mk = self.mask_key.get().strip()
        return {
            "mode": self.mode.get(),
            "low": coerce_value(self.low.get(), "float", 0.0, 1.2),
            "high": coerce_value(self.high.get(), "float", 0.0, 1.2),
            "angle_deg": coerce_value(self.angle_deg.get(), "float", 0.0, 360.0),
            "ang_width": coerce_value(self.ang_width.get(), "float", 1.0, 180.0),
            "boost": coerce_value(self.boost.get(), "float", -1.0, 5.0),
            "soft": coerce_value(self.soft.get(), "float", 0.0, 1.0),
            "blend": coerce_value(self.blend.get(), "float", 0.0, 1.0),
            "mask_key": mk if mk else None,
        }

register_panel(SpectralShaperPanel)
