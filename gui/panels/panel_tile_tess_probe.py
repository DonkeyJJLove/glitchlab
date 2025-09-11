# glitchlab/gui/panels/panel_tile_tess_probe.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Callable, Optional

MODES = ("overlay_grid", "phase_paint", "avg_tile", "quilt")
METHODS = ("acf", "fft")

class TileTessProbePanel(ttk.Frame):
    def __init__(self, master, ctx=None, on_change: Optional[Callable[[Dict[str, Any]], None]] = None):
        super().__init__(master)
        self.on_change = on_change or (lambda d: None)

        self.var_mode = tk.StringVar(value="overlay_grid")
        self.var_minp = tk.IntVar(value=4)
        self.var_maxp = tk.IntVar(value=256)
        self.var_method = tk.StringVar(value="acf")
        self.var_alpha = tk.DoubleVar(value=0.5)
        self.var_thick = tk.IntVar(value=1)
        self.var_jitter = tk.IntVar(value=2)
        self.var_use_amp = tk.DoubleVar(value=1.0)
        self.var_clamp = tk.BooleanVar(value=True)

        row = ttk.Frame(self, padding=6); row.pack(fill="x")
        ttk.Label(row, text="Mode").pack(side="left")
        ttk.Combobox(row, values=MODES, state="readonly", textvariable=self.var_mode, width=14).pack(side="left", padx=6)
        ttk.Label(row, text="Method").pack(side="left", padx=(12,0))
        ttk.Combobox(row, values=METHODS, state="readonly", textvariable=self.var_method, width=8).pack(side="left", padx=6)

        g1 = ttk.LabelFrame(self, text="Detection", padding=6); g1.pack(fill="x", padx=6, pady=6)
        ttk.Label(g1, text="min_period").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(g1, from_=2, to=2048, textvariable=self.var_minp, width=8).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(g1, text="max_period").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(g1, from_=4, to=4096, textvariable=self.var_maxp, width=8).grid(row=1, column=1, sticky="w", padx=6)

        g2 = ttk.LabelFrame(self, text="Visual", padding=6); g2.pack(fill="x", padx=6, pady=6)
        ttk.Label(g2, text="alpha").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(g2, from_=0.0, to=1.0, increment=0.05, textvariable=self.var_alpha, width=8).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(g2, text="grid_thickness").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(g2, from_=1, to=8, textvariable=self.var_thick, width=8).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(g2, text="quilt_jitter").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(g2, from_=0, to=64, textvariable=self.var_jitter, width=8).grid(row=2, column=1, sticky="w", padx=6)

        g3 = ttk.LabelFrame(self, text="Other", padding=6); g3.pack(fill="x", padx=6, pady=(0,6))
        ttk.Label(g3, text="use_amp").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(g3, from_=0.0, to=2.0, increment=0.1, textvariable=self.var_use_amp, width=8).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Checkbutton(g3, text="clamp u8", variable=self.var_clamp).grid(row=1, column=0, columnspan=2, sticky="w")

        btns = ttk.Frame(self, padding=6); btns.pack(fill="x")
        ttk.Button(btns, text="Apply changes", command=self._emit).pack(side="right")

        for var in (self.var_mode, self.var_minp, self.var_maxp, self.var_method,
                    self.var_alpha, self.var_thick, self.var_jitter,
                    self.var_use_amp, self.var_clamp):
            var.trace_add("write", lambda *args: self._emit())

    def _emit(self):
        d = {
            "mode": self.var_mode.get(),
            "min_period": int(self.var_minp.get()),
            "max_period": int(self.var_maxp.get()),
            "method": self.var_method.get(),
            "alpha": float(self.var_alpha.get()),
            "grid_thickness": int(self.var_thick.get()),
            "quilt_jitter": int(self.var_jitter.get()),
            "use_amp": float(self.var_use_amp.get()),
            "clamp": bool(self.var_clamp.get()),
        }
        self.on_change(d)
