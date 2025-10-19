# glitchlab/app/generic_form_panel.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict

from .panel_base import PanelBase as BasicPanel, PanelContext


class GenericFormPanel(BasicPanel):
    """Uniwersalny panel: tworzy pola z ctx.defaults/ctx.params; maski jako combobox."""

    def __init__(self, master, ctx: PanelContext | None = None, **kw):
        super().__init__(master, ctx=ctx, **kw)
        self.vars: Dict[str, tk.Variable] = {}
        self._build()

    def _build(self):
        ttk.Label(self, text=self.ctx.filter_name or "Filter",
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=8, pady=(6, 2))
        inner = ttk.Frame(self); inner.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        params = dict(self.ctx.defaults or {})
        if self.ctx and self.ctx.params:
            params.update(self.ctx.params)

        for key, val in params.items():
            row = ttk.Frame(inner); row.grid_columnconfigure(1, weight=1); row.pack(fill="x", pady=2)
            lbl = "Mask" if "mask" in key.lower() else key
            ttk.Label(row, text=lbl).grid(row=0, column=0, sticky="w")

            if "mask" in key.lower():
                names = (self.ctx.mask_keys() if self.ctx else []) or ["(none)", "full"]
                var = tk.StringVar(value=val if isinstance(val, str) else "(none)")
                cb = ttk.Combobox(row, values=names, state="readonly", textvariable=var)
                cb.grid(row=0, column=1, sticky="ew", padx=6)
                self.vars[key] = var
            elif isinstance(val, bool):
                var = tk.BooleanVar(value=bool(val)); ttk.Checkbutton(row, variable=var).grid(row=0, column=1, sticky="w", padx=6); self.vars[key] = var
            elif isinstance(val, int):
                var = tk.IntVar(value=int(val)); ttk.Entry(row, textvariable=var).grid(row=0, column=1, sticky="ew", padx=6); self.vars[key] = var
            elif isinstance(val, float):
                var = tk.DoubleVar(value=float(val)); ttk.Entry(row, textvariable=var).grid(row=0, column=1, sticky="ew", padx=6); self.vars[key] = var
            else:
                var = tk.StringVar(value=str(val)); ttk.Entry(row, textvariable=var).grid(row=0, column=1, sticky="ew", padx=6); self.vars[key] = var

        btns = ttk.Frame(self); btns.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(btns, text="Apply", command=self._apply).pack(side="right")

    def _apply(self):
        out: Dict[str, Any] = {k: v.get() for k, v in self.vars.items()}
        for k in list(out.keys()):
            if "mask" in k.lower():
                if out[k] == "(none)":
                    out[k] = None
        self.ctx.emit(out)
