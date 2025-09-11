from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from ..panel_base import PanelContext

class Panel(ttk.Frame):
    def __init__(self, master, ctx: PanelContext):
        super().__init__(master)
        self.ctx = ctx
        p = dict(ctx.defaults); p.update(ctx.params or {})
        self.var_gamma = tk.DoubleVar(value=float(p.get("gamma", 1.0)))
        self.var_mask = tk.StringVar(value=str(p.get("mask_key","") or ""))
        self.var_amp = tk.DoubleVar(value=float(p.get("use_amp", 1.0)))
        self.var_clamp = tk.BooleanVar(value=bool(p.get("clamp", True)))

        row = ttk.Frame(self, padding=8); row.pack(fill="x")
        ttk.Label(row, text="Gamma").pack(side="left")
        ttk.Spinbox(row, from_=0.1, to=5.0, increment=0.1,
                    textvariable=self.var_gamma, width=8).pack(side="left", padx=6)
        ttk.Button(row, text="Apply", command=self._emit).pack(side="right")

        g = ttk.Frame(self, padding=8); g.pack(fill="x")
        ttk.Label(g, text="mask_key").grid(row=0, column=0, sticky="w")
        ttk.Entry(g, textvariable=self.var_mask, width=14).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(g, text="use_amp").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(g, from_=0.0, to=2.0, increment=0.1,
                    textvariable=self.var_amp, width=8).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Checkbutton(g, text="clamp", variable=self.var_clamp).grid(row=2, column=0, columnspan=2, sticky="w")

        for v in (self.var_gamma, self.var_mask, self.var_amp, self.var_clamp):
            v.trace_add("write", lambda *a: self._emit())

    def _emit(self):
        params = {
            "gamma": float(self.var_gamma.get()),
            "mask_key": (self.var_mask.get().strip() or None),
            "use_amp": float(self.var_amp.get()),
            "clamp": bool(self.var_clamp.get()),
        }
        if self.ctx.on_change:
            self.ctx.on_change(params)
