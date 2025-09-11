
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Dict, List
import json, os

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

class PresetManager(ttk.Frame):
    """
    Menedżer presetów 'in-memory' + pliki w folderze.
    Pokazuje aktualny cfg jako YAML/JSON, pozwala: Add current step,
    Remove, Move up/down, Load/Save, Apply steps.
    Oczekuje callbacków z App:
      - get_cfg() -> Dict
      - set_cfg(cfg: Dict) -> None
      - apply_preset_steps() -> None
      - get_available_filters() -> List[str]  (do walidacji)
      - add_current_step() -> None (opcjonalnie; inaczej używa get_cfg/active)
    """
    def __init__(self, master, get_cfg, set_cfg, apply_preset_steps, get_available_filters):
        super().__init__(master)
        self._get_cfg = get_cfg
        self._set_cfg = set_cfg
        self._apply = apply_preset_steps
        self._get_filters = get_available_filters
        self._build()
        self.refresh_preview()

    def _build(self):
        self.columnconfigure(0, weight=1)
        # toolbar
        tb = ttk.Frame(self); tb.grid(row=0, column=0, sticky="ew", pady=(2,4))
        for (txt, cmd) in [
            ("Refresh", self.refresh_preview),
            ("Load", self.load_file),
            ("Apply steps", self.apply_steps),
            ("Save preset as…", self.save_file),
        ]:
            ttk.Button(tb, text=txt, command=cmd).pack(side="left", padx=(0,6))

        # steps list
        mid = ttk.Frame(self); mid.grid(row=1, column=0, sticky="nsew")
        mid.columnconfigure(1, weight=1)
        ttk.Label(mid, text="Steps:").grid(row=0, column=0, sticky="w")
        self.lst = tk.Listbox(mid, height=7, exportselection=False)
        self.lst.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(2,4))
        btns = ttk.Frame(mid); btns.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0,4))
        ttk.Button(btns, text="Add current step", command=self.add_current_step).pack(side="left")
        ttk.Button(btns, text="Remove", command=self.remove_step).pack(side="left", padx=6)
        ttk.Button(btns, text="Up", command=lambda: self.move_sel(-1)).pack(side="left")
        ttk.Button(btns, text="Down", command=lambda: self.move_sel(+1)).pack(side="left", padx=(6,0))

        # preview
        ttk.Label(self, text="Current preset (live):").grid(row=2, column=0, sticky="w")
        self.txt = tk.Text(self, height=14, wrap="none")
        self.txt.grid(row=3, column=0, sticky="nsew")
        self.rowconfigure(3, weight=1)
        # scrollbars
        sx = ttk.Scrollbar(self, orient="horizontal", command=self.txt.xview)
        sy = ttk.Scrollbar(self, orient="vertical", command=self.txt.yview)
        self.txt.configure(xscrollcommand=sx.set, yscrollcommand=sy.set)
        sy.grid(row=3, column=1, sticky="ns")
        sx.grid(row=4, column=0, sticky="ew")

    # utils
    def _dump_cfg(self, cfg: Dict[str, Any]) -> str:
        if yaml is not None:
            try:
                return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
            except Exception:
                pass
        return json.dumps(cfg, ensure_ascii=False, indent=2)

    def refresh_preview(self):
        cfg = self._get_cfg()
        # list steps
        self.lst.delete(0, "end")
        for s in (cfg.get("steps") or []):
            nm = s.get("name", "?")
            pr = s.get("params", {})
            self.lst.insert("end", f"{nm}  {pr}")
        # text
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", self._dump_cfg(cfg))

    def _read_preview_text(self) -> Dict[str, Any]:
        raw = self.txt.get("1.0", "end").strip()
        if not raw:
            return {}
        try:
            if yaml is not None:
                return yaml.safe_load(raw)  # type: ignore
        except Exception:
            pass
        try:
            return json.loads(raw)
        except Exception as e:
            messagebox.showerror("Preset parse", f"Cannot parse: {e}")
            return {}

    def load_file(self):
        path = filedialog.askopenfilename(title="Load preset (YAML/JSON)",
                    filetypes=[("YAML","*.yaml;*.yml"), ("JSON","*.json"), ("All","*.*")])
        if not path: return
        try:
            raw = open(path, "r", encoding="utf-8").read()
            if path.lower().endswith((".yaml",".yml")) and yaml is not None:
                cfg = yaml.safe_load(raw)  # type: ignore
            else:
                cfg = json.loads(raw)
            if not isinstance(cfg, dict):
                raise ValueError("Invalid preset format")
            self._set_cfg(cfg)
            self.refresh_preview()
        except Exception as e:
            messagebox.showerror("Load preset", str(e))

    def save_file(self):
        cfg = self._read_preview_text()
        if not cfg:
            cfg = self._get_cfg()
        path = filedialog.asksaveasfilename(title="Save preset as…",
                    defaultextension=".yaml",
                    filetypes=[("YAML","*.yaml;*.yml"), ("JSON","*.json"), ("All","*.*")])
        if not path: return
        try:
            if path.lower().endswith((".yaml",".yml")) and yaml is not None:
                txt = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)  # type: ignore
            else:
                txt = json.dumps(cfg, ensure_ascii=False, indent=2)
            open(path, "w", encoding="utf-8").write(txt)
            messagebox.showinfo("Save preset", f"Saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save preset", str(e))

    def apply_steps(self):
        self._set_cfg(self._read_preview_text() or self._get_cfg())
        self._apply()

    def add_current_step(self):
        cfg = self._get_cfg()
        steps = list(cfg.get("steps") or [])
        # heurystyka: wstaw aktywny filtr/parametry jeśli App je udostępnia
        # (czyli oczekujemy, że set_cfg/get_cfg zarządza active_filter/params).
        # W przeciwnym razie tylko wpisujemy pusty obiekt z nazwą (jeśli dostępna).
        name = cfg.get("__active_name") or ""
        params = cfg.get("__active_params") or {}
        if not name:
            # fallback: pierwszy dostępny filtr (jeśli jest)
            av = list(self._get_filters() or [])
            if av:
                name = av[0]
        if not name:
            messagebox.showinfo("Add step", "Brak aktywnego filtra.")
            return
        steps.append({"name": name, "params": dict(params)})
        cfg["steps"] = steps
        self._set_cfg(cfg)
        self.refresh_preview()

    def remove_step(self):
        sel = self.lst.curselection()
        if not sel:
            return
        idx = sel[0]
        cfg = self._get_cfg()
        steps = list(cfg.get("steps") or [])
        if 0 <= idx < len(steps):
            steps.pop(idx)
            cfg["steps"] = steps
            self._set_cfg(cfg)
            self.refresh_preview()

    def move_sel(self, delta: int):
        sel = self.lst.curselection()
        if not sel: return
        idx = sel[0]
        cfg = self._get_cfg()
        steps = list(cfg.get("steps") or [])
        j = idx + delta
        if 0 <= idx < len(steps) and 0 <= j < len(steps):
            steps[idx], steps[j] = steps[j], steps[idx]
            cfg["steps"] = steps
            self._set_cfg(cfg)
            self.refresh_preview()
            self.lst.selection_set(j)
