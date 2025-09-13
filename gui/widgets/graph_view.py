

# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk

class GraphView(ttk.Frame):
    """Placeholder: pokazuje komunikat o grafie/pipeline."""
    def __init__(self, master):
        super().__init__(master, padding=6)
        self.txt = tk.Text(self, height=6)
        self.txt.pack(fill="both", expand=True)
        self.set_ast_json(None)

    def set_ast_json(self, data):
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", "Graph diagnostics here. (ast/json n/a)")
        self.txt.configure(state="disabled")
