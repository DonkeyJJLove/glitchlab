# glitchlab/gui/preset_manager.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
from typing import Callable, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # optional


class PresetManager(ttk.Frame):
    """
    Lekki menedżer presetów „w pamięci” z podglądem YAML/JSON.
    Integracja: przekaż get_cfg, set_cfg, apply_preset_steps, get_available_filters.
    Opcjonalnie: get_preset_dir() -> str  (do wyświetlenia bieżącego katalogu presetów)
    """

    def __init__(self, parent,
                 get_cfg: Callable[[], dict],
                 set_cfg: Callable[[dict], None],
                 apply_preset_steps: Callable[[], None],
                 get_available_filters: Callable[[], list[str]],
                 get_preset_dir: Optional[Callable[[], str]] = None):
        super().__init__(parent)

        # --- callbacks ---
        self.get_cfg = get_cfg
        self.set_cfg = set_cfg
        self.apply_preset_steps = apply_preset_steps
        self.get_available_filters = get_available_filters
        self.get_preset_dir = get_preset_dir

        # --- top header with folder ---
        hdr = ttk.Frame(self)
        hdr.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(hdr, text="Preset folder:", foreground="#555").pack(side="left")
        self.lbl_dir = ttk.Label(hdr, text=self._safe_dir_text(), font=("", 9, "italic"))
        self.lbl_dir.pack(side="left", padx=(6, 0))

        # Toolbar
        tb = ttk.Frame(self); tb.pack(fill="x", pady=(4, 6), padx=6)
        ttk.Button(tb, text="Refresh", command=self._refresh_view).pack(side="left")
        ttk.Button(tb, text="Load…", command=self._load).pack(side="left", padx=(6, 0))
        ttk.Button(tb, text="Save as…", command=self._save).pack(side="left", padx=(6, 0))
        ttk.Button(tb, text="Apply steps", command=self._apply).pack(side="left", padx=(6, 0))
        ttk.Button(tb, text="Add current step", command=self._add_cur_step).pack(side="left", padx=(6, 0))

        # Editor view
        tabs = ttk.Notebook(self); tabs.pack(fill="both", expand=True, padx=6, pady=(0,6))
        self.txt_yaml = tk.Text(tabs, wrap="none", undo=True, height=12)
        self.txt_json = tk.Text(tabs, wrap="none", undo=True, height=12)
        tabs.add(self.txt_yaml, text="YAML")
        tabs.add(self.txt_json, text="JSON")

        self._refresh_view()

    # ----- helpers -----
    def _safe_dir_text(self) -> str:
        try:
            if self.get_preset_dir:
                d = self.get_preset_dir() or ""
                return d
        except Exception:
            pass
        return "(n/a)"

    def _refresh_dir_label(self):
        self.lbl_dir.configure(text=self._safe_dir_text())

    # ----- actions -----
    def _refresh_view(self):
        self._refresh_dir_label()
        cfg = self.get_cfg() or {}
        # YAML
        ytxt = ""
        if yaml is not None:
            try:
                ytxt = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
            except Exception:
                ytxt = ""
        if not ytxt:
            # awaryjnie konwersja przez JSON
            ytxt = self._json_to_yaml_fallback(cfg)
        self._set_text(self.txt_yaml, ytxt)

        # JSON
        jtxt = json.dumps(cfg, indent=2, ensure_ascii=False)
        self._set_text(self.txt_json, jtxt)

    def _json_to_yaml_fallback(self, data):
        # Bardzo prosta substytucja (nieidealna, ale czytelna)
        return json.dumps(data, indent=2, ensure_ascii=False).replace("{", "").replace("}", "")

    def _set_text(self, widget: tk.Text, text: str):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _load(self):
        path = filedialog.askopenfilename(title="Load preset (YAML/JSON)",
                                          filetypes=[("YAML", "*.yaml;*.yml"), ("JSON", "*.json"),
                                                     ("All files", "*.*")])
        if not path: return
        text = open(path, "r", encoding="utf-8").read()
        data = None
        if path.lower().endswith((".yaml", ".yml")) and yaml is not None:
            try:
                data = yaml.safe_load(text)
            except Exception as e:
                messagebox.showerror("Load preset", f"YAML parse error: {e}")
                return
        if data is None:
            try:
                data = json.loads(text)
            except Exception as e:
                messagebox.showerror("Load preset", f"JSON parse error: {e}")
                return
        try:
            self.set_cfg(data)
            self._refresh_view()
        except Exception as e:
            messagebox.showerror("Load preset", str(e))

    def _save(self):
        path = filedialog.asksaveasfilename(title="Save preset as", defaultextension=".yaml",
                                            filetypes=[("YAML", "*.yaml;*.yml"), ("JSON", "*.json")])
        if not path: return
        cfg = self.get_cfg() or {}
        try:
            if path.lower().endswith((".yaml", ".yml")) and yaml is not None:
                text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
            else:
                text = json.dumps(cfg, indent=2, ensure_ascii=False)
            open(path, "w", encoding="utf-8").write(text)
        except Exception as e:
            messagebox.showerror("Save preset", str(e))

    def _apply(self):
        try:
            self.apply_preset_steps()
        except Exception as e:
            messagebox.showerror("Apply preset", str(e))

    def _add_cur_step(self):
        cfg = dict(self.get_cfg() or {})
        name = cfg.get("__active_name") or ""
        params = cfg.get("__active_params") or {}
        if not name:
            messagebox.showinfo("Add current step", "Brak aktywnego filtra.")
            return
        step = {"name": name, "params": dict(params)}
        steps = list(cfg.get("steps", []) or [])
        steps.append(step)
        cfg["steps"] = steps
        # Wyczyść meta pola:
        cfg.pop("__active_name", None); cfg.pop("__active_params", None)
        try:
            self.set_cfg(cfg); self._refresh_view()
        except Exception as e:
            messagebox.showerror("Preset", str(e))
