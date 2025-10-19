# glitchlab/app/panels/panel_phase_glitch.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional, List

try:
    from glitchlab.gui.panel_base import PanelContext  # type: ignore
except Exception:
    class PanelContext:
        def __init__(self, **kw): self.__dict__.update(kw)

PLACEHOLDER_NONE = "<none>"

class PhaseGlitchPanel(ttk.Frame):
    """Panel sterowania dla filtra 'phase_glitch' (kanoniczne nazwy parametrów)."""

    def __init__(self, parent, ctx: Optional[PanelContext] = None):
        super().__init__(parent)
        self.ctx: PanelContext = ctx or PanelContext()

        self.var_low         = tk.DoubleVar(value=0.15)
        self.var_high        = tk.DoubleVar(value=0.65)
        self.var_strength    = tk.DoubleVar(value=0.70)
        self.var_preserve_dc = tk.BooleanVar(value=True)
        self.var_blend       = tk.DoubleVar(value=0.10)
        self.var_mask_key    = tk.StringVar(value=PLACEHOLDER_NONE)
        self.var_use_amp     = tk.DoubleVar(value=1.00)
        self.var_clamp       = tk.BooleanVar(value=True)

        self._build_ui()
        self._refresh_mask_list()
        self._emit_change()

    def _build_ui(self):
        pad = dict(padx=6, pady=4)

        box1 = ttk.LabelFrame(self, text="Frequency band"); box1.pack(fill="x", **pad)
        self._row_float(box1, "Low (0..1)",  self.var_low,  0.0, 1.0)
        self._row_float(box1, "High (0..1)", self.var_high, 0.0, 1.0)

        box2 = ttk.LabelFrame(self, text="Amount"); box2.pack(fill="x", **pad)
        self._row_float(box2, "Strength", self.var_strength, 0.0, 5.0)
        self._row_float(box2, "Blend",    self.var_blend,    0.0, 1.0)
        ttk.Checkbutton(box2, text="Preserve DC (keep global brightness/color)",
                        variable=self.var_preserve_dc, command=self._emit_change).pack(anchor="w", **pad)

        box3 = ttk.LabelFrame(self, text="Mask & utils"); box3.pack(fill="x", **pad)
        row = ttk.Frame(box3); row.pack(fill="x", **pad)
        ttk.Label(row, text="Mask:").pack(side="left")
        self.cmb_mask = ttk.Combobox(row, textvariable=self.var_mask_key, state="readonly",
                                     values=(PLACEHOLDER_NONE,), postcommand=self._refresh_mask_list, width=24)
        self.cmb_mask.pack(side="left", fill="x", expand=True, padx=(6,0))
        self.cmb_mask.bind("<<ComboboxSelected>>", lambda _e: self._emit_change())
        ttk.Button(row, text="Refresh", command=self._refresh_mask_list).pack(side="left", padx=(6,0))

        self._row_float(box3, "Use amplitude", self.var_use_amp, 0.0, 5.0)
        ttk.Checkbutton(box3, text="Clamp output to [0..255]",
                        variable=self.var_clamp, command=self._emit_change).pack(anchor="w", **pad)

        box4 = ttk.LabelFrame(self, text="Quick presets"); box4.pack(fill="x", **pad)
        btns = ttk.Frame(box4); btns.pack(fill="x", **pad)
        ttk.Button(btns, text="Subtle (low mids)", command=lambda: self._apply_preset(low=0.05, high=0.20, strength=0.6, preserve_dc=True, blend=0.0)).pack(side="left")
        ttk.Button(btns, text="Mid band", command=lambda: self._apply_preset(low=0.18, high=0.60, strength=0.7, preserve_dc=True, blend=0.10)).pack(side="left", padx=(6,0))
        ttk.Button(btns, text="High band", command=lambda: self._apply_preset(low=0.55, high=0.90, strength=0.8, preserve_dc=True, blend=0.15)).pack(side="left", padx=(6,0))

        self.bind("<Visibility>", lambda _e: self._refresh_mask_list())

    def _row_float(self, parent, label, var: tk.DoubleVar, mn: float, mx: float):
        row = ttk.Frame(parent); row.pack(fill="x", padx=6, pady=2)
        ttk.Label(row, text=label, width=18).pack(side="left")
        ent = ttk.Entry(row, textvariable=var, width=8); ent.pack(side="left", padx=(6,6))
        sld = ttk.Scale(row, from_=mn, to=mx, variable=var, command=lambda _=None: self._emit_change()); sld.pack(side="left", fill="x", expand=True)
        ent.bind("<Return>",   lambda _e: self._emit_change())
        ent.bind("<FocusOut>", lambda _e: self._emit_change())

    def _mask_source_keys(self) -> List[str]:
        try:
            f = getattr(self.ctx, "get_mask_keys", None)
            if callable(f):
                keys = list(f() or [])
                return [k for k in keys if isinstance(k, str)]
        except Exception:
            pass
        try:
            cache = getattr(self.ctx, "cache_ref", {}) or {}
            keys = list(cache.get("cfg/masks/keys", []))
            return [k for k in keys if isinstance(k, str)]
        except Exception:
            return []

    def _refresh_mask_list(self):
        keys = self._mask_source_keys()
        values = [PLACEHOLDER_NONE] + sorted(keys)
        try:
            self.cmb_mask["values"] = values
        except Exception:
            pass
        cur = (self.var_mask_key.get() or PLACEHOLDER_NONE)
        if cur not in values:
            cur = PLACEHOLDER_NONE
        try:
            self.cmb_mask.set(cur)
        except Exception:
            self.var_mask_key.set(cur)
        self._emit_change()

    def get_params(self) -> Dict[str, Any]:
        low = float(self.var_low.get())
        high = float(self.var_high.get())
        if high < low:
            low, high = high, low
            self.var_low.set(low); self.var_high.set(high)

        def clamp01(x: float) -> float:
            return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

        mk = (self.var_mask_key.get() or "").strip()
        return {
            "low":          clamp01(low),
            "high":         clamp01(high),
            "strength":     max(0.0, float(self.var_strength.get())),
            "preserve_dc":  bool(self.var_preserve_dc.get()),
            "blend":        clamp01(float(self.var_blend.get())),
            "mask_key":     (None if mk in ("", PLACEHOLDER_NONE) else mk),
            "use_amp":      max(0.0, float(self.var_use_amp.get())),
            "clamp":        bool(self.var_clamp.get()),
        }

    def _emit_change(self):
        cb = getattr(self.ctx, "on_change", None)
        if callable(cb):
            try:
                cb(self.get_params())
            except Exception:
                pass

    def _apply_preset(self, **kw):
        for k, v in kw.items():
            try:
                var = getattr(self, f"var_{k}")
                if isinstance(var, tk.BooleanVar): var.set(bool(v))
                else: var.set(float(v))
            except Exception:
                pass
        self._emit_change()

Panel = PhaseGlitchPanel
