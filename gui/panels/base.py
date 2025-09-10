# gui/panels/base.py
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional, Callable

class PanelBase(ttk.Frame):
    """
    Bazowa klasa panelu parametryzującego filtr.
    - buduje kontrolki
    - zwraca dict z parametrami (to_params)
    - posiada callback apply_cb(img, ctx, fname, params) do natychmiastowego podglądu
    """
    def __init__(self, master, on_apply: Optional[Callable[[str, Dict[str, Any]], None]]=None):
        super().__init__(master)
        self.on_apply = on_apply
        self._vars: Dict[str, tk.Variable] = {}

    def add_num(self, label: str, key: str, value: float, step: float = 1.0):
        row = ttk.Frame(self); row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=18).pack(side="left")
        var = tk.DoubleVar(self, value=value)
        ttk.Entry(row, textvariable=var, width=10).pack(side="left")
        self._vars[key] = var

    def add_int(self, label: str, key: str, value: int):
        row = ttk.Frame(self); row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=18).pack(side="left")
        var = tk.IntVar(self, value=value)
        ttk.Entry(row, textvariable=var, width=10).pack(side="left")
        self._vars[key] = var

    def add_bool(self, label: str, key: str, value: bool):
        row = ttk.Frame(self); row.pack(fill="x", pady=2)
        var = tk.BooleanVar(self, value=value)
        ttk.Checkbutton(row, text=label, variable=var).pack(side="left")
        self._vars[key] = var

    def add_text(self, label: str, key: str, value: str):
        row = ttk.Frame(self); row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=18).pack(side="left")
        var = tk.StringVar(self, value=value)
        ttk.Entry(row, textvariable=var, width=20).pack(side="left", fill="x", expand=True)
        self._vars[key] = var

    def to_params(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self._vars.items():
            out[k] = v.get()
        return out

    def btn_apply(self, filter_name: str, text: str = "Apply"):
        def _act():
            if callable(self.on_apply):
                self.on_apply(filter_name, self.to_params())
        ttk.Button(self, text=text, command=_act).pack(pady=6, anchor="e")
