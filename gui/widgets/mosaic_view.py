
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from tkinter import ttk

class MosaicMini(ttk.Frame):
    """Placeholder: pokazuje liczby węzłów/krawędzi z ctx.cache['ast/json'] (jeśli są)."""
    def __init__(self, master):
        super().__init__(master, padding=6)
        self.lbl = ttk.Label(self, text="Mosaic: (n/a)")
        self.lbl.pack(anchor="w")

    def set_ast_json(self, data):
        try:
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8", errors="ignore")
            if isinstance(data, str):
                obj = json.loads(data)
            else:
                obj = data or {}
            n = len(obj.get("nodes", [])); e = len(obj.get("edges", []))
            self.lbl.config(text=f"Mosaic AST: {n} nodes / {e} edges")
        except Exception:
            self.lbl.config(text="Mosaic: (n/a)")
