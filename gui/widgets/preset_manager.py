# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, glob
from typing import Callable, Dict, Any, List, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # optional

try:
    from glitchlab.core.pipeline import normalize_preset
except Exception:
    from core.pipeline import normalize_preset  # type: ignore

class PresetManager(ttk.Frame):
    """Lekki menedżer presetów: przeglądanie, ładowanie i zapis. 
       Integruje się z App przez przekazane callbacki.
    """
    def __init__(self, master,
                 on_load: Callable[[Dict[str, Any]], None],
                 on_apply: Callable[[], None],
                 on_save: Optional[Callable[[Dict[str, Any]], None]] = None,
                 get_current_cfg: Optional[Callable[[], Dict[str, Any]]] = None,
                 on_add_step: Optional[Callable[[], None]] = None,
                 ):
        super().__init__(master)
        self.on_load = on_load
        self.on_apply = on_apply
        self.on_save = on_save
        self.get_current_cfg = get_current_cfg or (lambda: {})
        self.on_add_step = on_add_step

        self.root_dir = os.path.abspath("presets")
        self.user_dir = os.path.expanduser("~/.glitchlab/presets")
        os.makedirs(self.user_dir, exist_ok=True)

        # UI
        top = ttk.Frame(self); top.pack(fill="x", padx=6, pady=6)
        ttk.Label(top, text="Preset folders:").pack(side="left")
        self.var_folder = tk.StringVar(value=self.root_dir if os.path.isdir(self.root_dir) else self.user_dir)
        ttk.Combobox(top, values=[self.root_dir, self.user_dir], textvariable=self.var_folder, state="readonly", width=42).pack(side="left", padx=6)

        mid = ttk.Frame(self); mid.pack(fill="both", expand=True, padx=6, pady=6)
        self.list = tk.Listbox(mid, height=10)
        self.list.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=self.list.yview); sb.pack(side="left", fill="y")
        self.list.configure(yscrollcommand=sb.set)
        self.list.bind("<Double-1>", lambda e: self._load_selected())

        btns = ttk.Frame(self); btns.pack(fill="x", padx=6, pady=(0,6))
        ttk.Button(btns, text="Refresh", command=self._refresh).pack(side="left")
        ttk.Button(btns, text="Load", command=self._load_selected).pack(side="left", padx=4)
        ttk.Button(btns, text="Apply steps", command=self.on_apply).pack(side="left", padx=4)
        ttk.Button(btns, text="Save preset as…", command=self._save_as).pack(side="left", padx=4)
        if self.on_add_step:
            ttk.Button(btns, text="Add current step", command=self.on_add_step).pack(side="right")

        self._refresh()

    def _refresh(self):
        folder = self.var_folder.get()
        files: List[str] = []
        for ext in ("*.json","*.yml","*.yaml"):
            files.extend(sorted(glob.glob(os.path.join(folder, ext))))
        self.list.delete(0, "end")
        for f in files:
            self.list.insert("end", os.path.basename(f))
        if not files:
            self.list.insert("end", "(brak plików presetów)")

    def _selected_path(self) -> Optional[str]:
        sel = self.list.curselection()
        if not sel: return None
        name = self.list.get(sel[0])
        path = os.path.join(self.var_folder.get(), name)
        return path

    def _load_selected(self):
        path = self._selected_path()
        if not path or not os.path.isfile(path): return
        try:
            text = open(path, "r", encoding="utf-8").read()
            if path.lower().endswith((".yaml",".yml")) and yaml is not None:
                raw = yaml.safe_load(text)
            else:
                raw = json.loads(text)
            cfg = normalize_preset(raw)
            self.on_load(cfg)
        except Exception as e:
            messagebox.showerror("Preset", str(e))

    def _save_as(self):
        cfg = self.get_current_cfg() or {}
        path = filedialog.asksaveasfilename(
            title="Save preset as…",
            defaultextension=".json",
            filetypes=[("JSON","*.json"),("YAML","*.yaml;*.yml"),("All files","*.*")]
        )
        if not path:
            return
        try:
            if path.lower().endswith((".yaml",".yml")) and yaml is not None:
                with open(path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Preset", f"Saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Preset", str(e))
