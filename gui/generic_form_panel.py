from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict
from panel_base import PanelBase

class GenericFormPanel(PanelBase):
    def build(self) -> None:
        self.vars: Dict[str, tk.Variable] = {}
        defaults = (self._ctx.defaults if self._ctx else {})
        params = (self._ctx.params if self._ctx else {})
        row = 0
        for k, v in defaults.items():
            lab = ttk.Label(self, text=k); lab.grid(row=row, column=0, sticky="w", padx=4, pady=2)
            var = None
            if isinstance(v, bool):
                var = tk.BooleanVar(value=bool(params.get(k, v)))
                cb = ttk.Checkbutton(self, variable=var, command=self._emit)
                cb.grid(row=row, column=1, sticky="w")
            elif isinstance(v, (int, float)):
                if isinstance(v, int):
                    var = tk.IntVar(value=int(params.get(k, v)))
                else:
                    var = tk.DoubleVar(value=float(params.get(k, v)))
                ent = ttk.Entry(self, textvariable=var, width=12)
                ent.grid(row=row, column=1, sticky="w")
                ent.bind("<KeyRelease>", lambda e: self._emit())
            else:
                var = tk.StringVar(value=str(params.get(k, v)))
                ent = ttk.Entry(self, textvariable=var, width=16)
                ent.grid(row=row, column=1, sticky="w")
                ent.bind("<KeyRelease>", lambda e: self._emit())
            self.vars[k] = var
            row += 1

    def _emit(self) -> None:
        if not self._ctx: return
        out: Dict[str, Any] = {}
        for k, var in self.vars.items():
            out[k] = var.get()
        self._ctx.params = out
        if self.on_change:
            self.on_change(out)

    def get_params(self) -> Dict[str, Any]:
        if not self._ctx:
            return {}
        return dict(self._ctx.params)
