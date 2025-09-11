# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Optional
import json

class GraphView(ttk.Frame):
    """Placeholder widoku AST/graph. Oczekuje ctx.cache['ast/json'] (string lub dict)."""
    def __init__(self, master):
        super().__init__(master)
        ttk.Label(self, text="Graph & Diff").pack(anchor="w", padx=6, pady=(6,2))
        self.txt = tk.Text(self, height=10, wrap="none")
        self.txt.pack(fill="both", expand=True, padx=6, pady=6)
        self.txt.insert("1.0", "AST/graph: n/a")

    def set_ast_json(self, data: Optional[object]) -> None:
        self.txt.delete("1.0", "end")
        if data is None:
            self.txt.insert("1.0", "AST/graph: n/a")
            return
        try:
            if isinstance(data, (dict, list)):
                txt = json.dumps(data, ensure_ascii=False, indent=2)
            else:
                txt = str(data)
            self.txt.insert("1.0", txt)
        except Exception as e:
            self.txt.insert("1.0", f"(error) {e}")
