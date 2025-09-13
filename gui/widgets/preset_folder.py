# glitchlab/gui/widgets/preset_folder.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json
from typing import Callable, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # optional

from ..paths import get_default_preset_dir


class PresetFolder(ttk.Frame):
    """
    Lista presetów z folderu (YAML/JSON) + szybkie wczytanie.
    Integracja: podać callback load_cfg(dict).
    """
    def __init__(self, parent, load_cfg: Callable[[dict], None], initial_dir: Optional[str] = None):
        super().__init__(parent)
        self.load_cfg = load_cfg
        self.dir = initial_dir or get_default_preset_dir()

        bar = ttk.Frame(self); bar.pack(fill="x", pady=(2, 4))
        ttk.Label(bar, text="Preset folder:").pack(side="left")
        self.var_dir = tk.StringVar(value=self.dir)
        ent = ttk.Entry(bar, textvariable=self.var_dir, width=44)
        ent.pack(side="left", padx=(6,0))
        ttk.Button(bar, text="Browse…", command=self._pick_dir).pack(side="left", padx=(6,0))
        ttk.Button(bar, text="Refresh", command=self._refresh).pack(side="left", padx=(6,0))
        ttk.Button(bar, text="Make examples", command=self._make_examples).pack(side="left", padx=(6,0))

        self.lst = tk.Listbox(self, width=56, height=16, exportselection=False)
        self.lst.pack(fill="both", expand=True)
        self.lst.bind("<Double-Button-1>", lambda e: self._load_selected())

        row = ttk.Frame(self); row.pack(fill="x", pady=(4,0))
        ttk.Button(row, text="Load", command=self._load_selected).pack(side="left")
        ttk.Button(row, text="Open folder", command=self._open_folder).pack(side="left", padx=(6,0))

        self._refresh()

    def _pick_dir(self):
        d = filedialog.askdirectory(title="Choose preset folder", initialdir=self.dir)
        if not d: return
        self.dir = d; self.var_dir.set(d); self._refresh()

    def _refresh(self):
        d = self.var_dir.get().strip() or get_default_preset_dir()
        if not os.path.isdir(d):
            self.lst.delete(0, "end"); self.lst.insert("end", "(no such folder)")
            return
        files = [f for f in os.listdir(d) if f.lower().endswith((".yaml",".yml",".json"))]
        files.sort()
        self.lst.delete(0, "end")
        if not files: self.lst.insert("end", "(no preset files)")
        for f in files:
            self.lst.insert("end", f)

    def _open_folder(self):
        d = self.var_dir.get().strip()
        if not os.path.isdir(d):
            messagebox.showinfo("Open folder", "Folder does not exist."); return
        try:
            os.startfile(d)  # Windows
        except Exception:
            try:
                import subprocess
                subprocess.Popen(["open", d])  # macOS
            except Exception:
                import webbrowser
                webbrowser.open(f"file://{d}")

    def _load_selected(self):
        d = self.var_dir.get().strip()
        if not os.path.isdir(d): return
        sel = self.lst.curselection()
        if not sel: return
        name = self.lst.get(sel[0])
        if not name.lower().endswith((".yaml",".yml",".json")):
            return
        path = os.path.join(d, name)
        try:
            text = open(path, "r", encoding="utf-8").read()
            if path.lower().endswith((".yaml",".yml")) and yaml is not None:
                data = yaml.safe_load(text)
            else:
                data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("Preset must be a mapping (dict).")
            self.load_cfg(data)
        except Exception as e:
            messagebox.showerror("Load preset", f"{name}\n\n{e}")

    def _make_examples(self):
        d = self.var_dir.get().strip() or get_default_preset_dir()
        os.makedirs(d, exist_ok=True)
        ex1 = {
            "version": 2,
            "name": "Example: ACW + BMG",
            "seed": 7,
            "amplitude": {"kind": "linear_y", "strength": 1.0},
            "edge_mask": {"thresh": 60, "dilate": 0, "ksize": 3},
            "steps": [
                {"name": "anisotropic_contour_warp",
                 "params": {"strength": 1.2, "iters": 2, "edge_bias": 0.4, "use_amp": 1.0, "clamp": True}},
                {"name": "block_mosh_grid",
                 "params": {"size": 24, "p": 0.45, "max_shift": 16, "mix": 0.9}}
            ]
        }
        ex2 = {
            "version": 2,
            "name": "Example: Spectral ring",
            "seed": 7,
            "amplitude": {"kind": "none", "strength": 1.0},
            "edge_mask": {"thresh": 60, "dilate": 0, "ksize": 3},
            "steps": [
                {"name": "spectral_shaper",
                 "params": {"mode": "ring", "low": 0.12, "high": 0.42, "boost": 0.8, "soft": 0.08, "blend": 0.0}}
            ]
        }
        for (fname, data) in (("example_acw_bmg.yaml", ex1), ("example_spectral.yaml", ex2)):
            path = os.path.join(d, fname)
            try:
                if yaml is not None:
                    open(path, "w", encoding="utf-8").write(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
                else:
                    import json
                    open(path, "w", encoding="utf-8").write(json.dumps(data, indent=2, ensure_ascii=False))
            except Exception as e:
                messagebox.showerror("Make examples", f"{fname}\n\n{e}")
        self._refresh()
