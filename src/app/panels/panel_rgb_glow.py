# glitchlab/app/panels/panel_rgb_glow.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, List
from ..panel_base import PanelContext

PLACEHOLDER_NONE = "<none>"

class Panel(ttk.Frame):
    def __init__(self, master, ctx: PanelContext):
        super().__init__(master)
        self.ctx = ctx
        p = dict(getattr(ctx, "defaults", {}) or {})
        p.update(getattr(ctx, "params", {}) or {})

        # --- zmienne stanu ---
        self.var_lift  = tk.DoubleVar(value=float(p.get("lift", 0.15)))
        self.var_sat   = tk.DoubleVar(value=float(p.get("sat", 0.2)))
        # mask_key jako combobox z placeholderem "<none>"
        init_mask = str(p.get("mask_key", "") or PLACEHOLDER_NONE)
        self.var_mask = tk.StringVar(value=init_mask)
        self.var_amp  = tk.DoubleVar(value=float(p.get("use_amp", 1.0)))
        self.var_clamp= tk.BooleanVar(value=bool(p.get("clamp", True)))

        # --- UI ---
        row = ttk.LabelFrame(self, text="Lift & Saturation", padding=8); row.pack(fill="x", padx=8, pady=(8,4))
        ttk.Label(row, text="lift").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(row, from_=0.0, to=1.0, increment=0.05,
                    textvariable=self.var_lift, width=7).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(row, text="sat").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(row, from_=-1.0, to=1.0, increment=0.05,
                    textvariable=self.var_sat, width=7).grid(row=0, column=3, sticky="w", padx=6)
        row.columnconfigure(4, weight=1)

        g = ttk.LabelFrame(self, text="Mask & Amplitude", padding=8); g.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(g, text="mask_key").grid(row=0, column=0, sticky="w")
        mask_row = ttk.Frame(g); mask_row.grid(row=0, column=1, columnspan=2, sticky="ew", padx=6)
        self.cmb_mask = ttk.Combobox(mask_row, state="readonly", width=24,
                                     textvariable=self.var_mask,
                                     postcommand=self._refresh_masks)
        self.cmb_mask.pack(side="left", fill="x", expand=True)
        self.cmb_mask.bind("<<ComboboxSelected>>", lambda _e: self._emit())
        ttk.Button(mask_row, text="edge",  command=lambda: self._set_mask("edge")).pack(side="left", padx=(6,2))
        ttk.Button(mask_row, text="clear", command=lambda: self._set_mask(PLACEHOLDER_NONE)).pack(side="left")

        ttk.Label(g, text="use_amp").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Spinbox(g, from_=0.0, to=2.0, increment=0.1,
                    textvariable=self.var_amp, width=8).grid(row=1, column=1, sticky="w", padx=6, pady=(6,0))
        ttk.Checkbutton(g, text="clamp", variable=self.var_clamp)\
            .grid(row=2, column=0, columnspan=2, sticky="w", pady=(6,0))
        g.columnconfigure(1, weight=1)

        # przycisk Apply (opcjonalnie — szybki trigger)
        bar = ttk.Frame(self, padding=(8,0,8,8)); bar.pack(fill="x")
        ttk.Button(bar, text="Apply", command=self._emit).pack(side="right")

        # --- powiązania ---
        for v in (self.var_lift, self.var_sat, self.var_mask, self.var_amp, self.var_clamp):
            v.trace_add("write", lambda *_: self._emit())
        # auto-refresh listy masek przy pokazaniu panelu
        self.bind("<Visibility>", lambda _e: self._refresh_masks())

        # start
        self._refresh_masks()
        self._emit()

    # ---------- maski ----------
    def _mask_source_keys(self) -> List[str]:
        """Zaciągnij live listę masek z Global albo z cache_ref."""
        try:
            f = getattr(self.ctx, "get_mask_keys", None)
            if callable(f):
                keys = list(f() or [])
                return [k for k in keys if isinstance(k, str)]
        except Exception:
            pass
        try:
            cache = getattr(self.ctx, "cache_ref", {}) or {}
            keys = list(cache.get("cfg/masks/keys", []))
            return [k for k in keys if isinstance(k, str)]
        except Exception:
            return []

    def _refresh_masks(self) -> None:
        keys = self._mask_source_keys()
        values = [PLACEHOLDER_NONE] + sorted(keys)
        try:
            self.cmb_mask["values"] = values
        except Exception:
            pass
        cur = (self.var_mask.get() or PLACEHOLDER_NONE)
        if cur not in values:
            cur = PLACEHOLDER_NONE
        try:
            self.cmb_mask.set(cur)
        except Exception:
            self.var_mask.set(cur)

    def _set_mask(self, key: str) -> None:
        self.var_mask.set(key if key else PLACEHOLDER_NONE)
        self._emit()

    # ---------- emit ----------
    def _emit(self):
        mk = (self.var_mask.get() or "").strip()
        params = {
            "lift": float(self.var_lift.get()),
            "sat":  float(self.var_sat.get()),
            "mask_key": (None if mk in ("", PLACEHOLDER_NONE) else mk),
            "use_amp": float(self.var_amp.get()),
            "clamp": bool(self.var_clamp.get()),
        }
        if getattr(self.ctx, "on_change", None):
            try:
                self.ctx.on_change(params)
            except Exception:
                pass
