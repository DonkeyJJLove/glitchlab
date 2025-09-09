# glitchlab/gui/generic_form_panel.py
# -*- coding: utf-8 -*-
"""
GenericFormPanel — fallback, gdy brak panelu dedykowanego.
Buduje formularz na podstawie:
- registry.meta(schema, defaults)
- sygnatury filtra (nazwy parametrów i domyślne wartości)

Obsługa typów: int/float/bool/str + enum.
Pole mask_key (jeśli występuje) wypełniane z PanelContext.mask_keys.
"""

from __future__ import annotations
from typing import Dict, Any
import tkinter as tk
from tkinter import ttk
import inspect

from glitchlab.gui.panel_base import FilterPanel, coerce_value
from glitchlab.gui.controls import labeled_entry, labeled_checkbox, labeled_combo, BG, FG
from glitchlab.core import registry as reg


class GenericFormPanel(FilterPanel):
    def __init__(self, filter_name: str):
        super().__init__()
        self.FILTER_NAME = filter_name
        self._vars: dict[str, tk.Variable] = {}
        self._schema: dict[str, Any] = {}
        self._defaults: dict[str, Any] = {}

    def build(self, parent: tk.Widget) -> tk.Frame:
        root = tk.Frame(parent, bg=BG)
        self._root = root

        # metadata
        try:
            m = reg.meta(self.FILTER_NAME)
            self._schema = m.get("schema") or {}
            self._defaults = m.get("defaults") or {}
        except Exception:
            self._schema = {}
            self._defaults = {}

        # sygnatura filtra
        try:
            fn = reg.get(self.FILTER_NAME)
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())[2:]  # pomiń (img, ctx)
        except Exception:
            params = []

        # budowa pól
        for p in params:
            name = p.name
            default = self._defaults.get(name, p.default if p.default is not inspect._empty else "")

            sch = self._schema.get(name, {})
            ptype = sch.get("type")  # 'int'|'float'|'bool'|'str'
            penum = sch.get("enum")  # lista wartości do combo

            # mask_key → dropdown z kontekstu
            if name == "mask_key":
                var = tk.StringVar(value=str(default) if default not in (None, "") else "")
                self._vars[name] = var
                # dynamicznie wypełnimy po pierwszym wyświetleniu
                fr = tk.Frame(root, bg=BG)
                fr.pack(fill="x", pady=2)
                tk.Label(fr, text="mask_key", width=16, anchor="w", bg=BG, fg=FG).pack(side=tk.LEFT)
                self._mask_combo = ttk.Combobox(fr, textvariable=var, values=[], state="normal", width=16)
                self._mask_combo.pack(side=tk.LEFT, fill="x", expand=True)
                var.trace_add("write", lambda *a: self.on_change and self.on_change())
                # podmień listę przy budowie
                ctx = self.get_context()
                self._mask_combo.configure(values=ctx.mask_keys)
                continue

            # enum → combobox
            if penum and isinstance(penum, (list, tuple)):
                var = tk.StringVar(value=str(default) if default not in (None, "") else str(penum[0]))
                self._vars[name] = var
                labeled_combo(root, name, var, values=penum, on_change=(self.on_change or (lambda: None))).pack(fill="x", pady=2)
                continue

            # bool → checkbox
            if (ptype == "bool") or isinstance(default, bool):
                var = tk.BooleanVar(value=bool(default))
                self._vars[name] = var
                labeled_checkbox(root, name, var, on_change=self.on_change).pack(fill="x", pady=2)
                continue

            # liczby → entry
            if (ptype in ("int", "float")) or isinstance(default, (int, float)):
                # uwaga: trzymamy jako StringVar, koercja w get_params
                var = tk.StringVar(value=str(default))
                self._vars[name] = var
                labeled_entry(root, name, var, on_change=self.on_change).pack(fill="x", pady=2)
                continue

            # fallback: string
            var = tk.StringVar(value="" if default is None else str(default))
            self._vars[name] = var
            labeled_entry(root, name, var, on_change=self.on_change).pack(fill="x", pady=2)

        return root

    def get_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        # z metadanych schema weź typ/min/max
        sch = self._schema or {}
        for name, var in self._vars.items():
            val = var.get()
            sdef = sch.get(name) or {}
            ptype = sdef.get("type")
            vmin = sdef.get("min")
            vmax = sdef.get("max")
            # spróbuj JSON dla złożonych stringów
            if isinstance(val, str):
                v = val.strip()
                if v.startswith("{") or v.startswith("[") or v.startswith('"'):
                    import json
                    try:
                        params[name] = json.loads(v)
                        continue
                    except Exception:
                        pass
            params[name] = coerce_value(val, ty=ptype, vmin=vmin, vmax=vmax)
        # puste mask_key → None
        if "mask_key" in params and (params["mask_key"] == "" or params["mask_key"] is None):
            params["mask_key"] = None
        return params

    def set_params(self, params: Dict[str, Any]) -> None:
        for k, v in (params or {}).items():
            if k in self._vars:
                self._vars[k].set("" if v is None else str(v))

    def validate(self) -> None:
        # prosta walidacja numeryczna
        sch = self._schema or {}
        for name, var in self._vars.items():
            s = var.get()
            spec = sch.get(name) or {}
            ptype = spec.get("type")
            vmin = spec.get("min")
            vmax = spec.get("max")
            try:
                coerce_value(s, ty=ptype, vmin=vmin, vmax=vmax)
            except Exception as e:
                raise ValueError(f"Param '{name}': {e}")
