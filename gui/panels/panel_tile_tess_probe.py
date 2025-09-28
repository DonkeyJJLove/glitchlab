# glitchlab/gui/panels/panel_tile_tess_probe.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional, List

try:
    from glitchlab.gui.panel_base import PanelContext  # type: ignore
except Exception:  # pragma: no cover
    class PanelContext:
        def __init__(self, **kw): self.__dict__.update(kw)

PLACEHOLDER_NONE = "<none>"

DEFAULTS: Dict[str, Any] = {
    "mode": "overlay_grid",
    "min_period": 4,
    "max_period": 256,
    "method": "acf",
    "alpha": 0.5,
    "grid_thickness": 1,
    "quilt_jitter": 2,
    "mask_key": None,
    "use_amp": 1.0,
    "clamp": True,
}

class TileTessProbePanel(ttk.Frame):
    """
    Panel GUI dla filtra 'tile_tess_probe' — integracja z biblioteką masek (Global),
    placeholder <none>, auto-refresh listy, presety i sanity clamping.
    """
    def __init__(self, master: tk.Misc, ctx: Optional[PanelContext] = None, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.ctx = ctx or PanelContext(filter_name="tile_tess_probe", defaults=DEFAULTS, params={}, on_change=None, cache_ref={})
        p = {**DEFAULTS, **(getattr(self.ctx, "defaults", {}) or {}), **(getattr(self.ctx, "params", {}) or {})}

        # zmienne UI
        self.var_mode   = tk.StringVar(value=str(p.get("mode", "overlay_grid")))
        self.var_minp   = tk.IntVar(   value=int(p.get("min_period", 4)))
        self.var_maxp   = tk.IntVar(   value=int(p.get("max_period", 256)))
        self.var_method = tk.StringVar(value=str(p.get("method", "acf")))
        self.var_alpha  = tk.DoubleVar(value=float(p.get("alpha", 0.5)))
        self.var_thick  = tk.IntVar(   value=int(p.get("grid_thickness", 1)))
        self.var_jitter = tk.IntVar(   value=int(p.get("quilt_jitter", 2)))
        self.var_mask   = tk.StringVar(value=str(p.get("mask_key", "") or PLACEHOLDER_NONE))
        self.var_amp    = tk.DoubleVar(value=float(p.get("use_amp", 1.0)))
        self.var_clamp  = tk.BooleanVar(value=bool(p.get("clamp", True)))

        self._build_ui()
        self._bind_all()
        self._refresh_masks()
        self._emit()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=(8,8,8,4)); top.pack(fill="x")
        ttk.Label(top, text="Tessellation Probe", font=("", 10, "bold")).pack(side="left")

        r0 = ttk.LabelFrame(self, text="Tryb i metoda", padding=8); r0.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(r0, text="mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(r0, values=("overlay_grid","phase_paint","avg_tile","quilt"),
                     textvariable=self.var_mode, state="readonly", width=14)\
            .grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(r0, text="method").grid(row=0, column=2, sticky="w")
        ttk.Combobox(r0, values=("acf","fft"), textvariable=self.var_method, state="readonly", width=6)\
            .grid(row=0, column=3, sticky="w", padx=6)

        r1 = ttk.LabelFrame(self, text="Zakres okresu [px]", padding=8); r1.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(r1, text="min_period").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(r1, from_=2, to=4096, textvariable=self.var_minp, width=8).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(r1, text="max_period").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(r1, from_=3, to=8192, textvariable=self.var_maxp, width=8).grid(row=0, column=3, sticky="w", padx=6)

        r2 = ttk.LabelFrame(self, text="Miks / overlay / quilt", padding=8); r2.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(r2, text="alpha").grid(row=0, column=0, sticky="w")
        ttk.Scale(r2, from_=0.0, to=1.0, variable=self.var_alpha).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(r2, textvariable=self.var_alpha, width=6).grid(row=0, column=2, sticky="w")
        ttk.Label(r2, text="grid_thickness").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Spinbox(r2, from_=1, to=16, textvariable=self.var_thick, width=6).grid(row=1, column=1, sticky="w", padx=6, pady=(6,0))
        ttk.Label(r2, text="quilt_jitter").grid(row=1, column=2, sticky="w", pady=(6,0))
        ttk.Spinbox(r2, from_=0, to=64, textvariable=self.var_jitter, width=6).grid(row=1, column=3, sticky="w", padx=6, pady=(6,0))
        r2.columnconfigure(1, weight=1)

        r3 = ttk.LabelFrame(self, text="Mask & Amplitude", padding=8); r3.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(r3, text="mask_key").grid(row=0, column=0, sticky="w")
        row = ttk.Frame(r3); row.grid(row=0, column=1, columnspan=3, sticky="ew", padx=6)
        self.cmb_mask = ttk.Combobox(row, state="readonly", width=24, textvariable=self.var_mask, values=[])
        self.cmb_mask.pack(side="left", fill="x", expand=True)
        self.cmb_mask.bind("<Button-1>", lambda _e: self._refresh_masks())
        ttk.Button(row, text="Refresh", command=self._refresh_masks).pack(side="left", padx=(6,0))
        ttk.Button(row, text="edge", command=lambda: self._set_mask("edge")).pack(side="left", padx=(6,2))
        ttk.Button(row, text="clear", command=lambda: self._set_mask("")).pack(side="left")
        ttk.Label(r3, text="use_amp").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Scale(r3, from_=0.0, to=2.0, variable=self.var_amp).grid(row=1, column=1, sticky="ew", padx=6, pady=(6,0))
        ttk.Entry(r3, textvariable=self.var_amp, width=6).grid(row=1, column=2, sticky="w", pady=(6,0))
        ttk.Checkbutton(r3, text="clamp", variable=self.var_clamp).grid(row=1, column=3, sticky="w", pady=(6,0))
        r3.columnconfigure(1, weight=1)

        pr = ttk.Frame(self, padding=(8,2,8,8)); pr.pack(fill="x")
        ttk.Label(pr, text="Presets:").pack(side="left")
        ttk.Button(pr, text="Grid (auto)", command=lambda: self._apply_preset(mode="overlay_grid", alpha=0.5, grid_thickness=1))\
            .pack(side="left", padx=2)
        ttk.Button(pr, text="Phase paint", command=lambda: self._apply_preset(mode="phase_paint", alpha=0.7))\
            .pack(side="left", padx=2)
        ttk.Button(pr, text="Avg tile strong", command=lambda: self._apply_preset(mode="avg_tile", alpha=0.8))\
            .pack(side="left", padx=2)
        ttk.Button(pr, text="Quilt gentle", command=lambda: self._apply_preset(mode="quilt", alpha=0.4, quilt_jitter=2))\
            .pack(side="left", padx=2)

    # ---------- bindings / masks ----------
    def _bind_all(self) -> None:
        for v in (self.var_mode, self.var_minp, self.var_maxp, self.var_method,
                  self.var_alpha, self.var_thick, self.var_jitter,
                  self.var_mask, self.var_amp, self.var_clamp):
            v.trace_add("write", lambda *_: self._emit())
        self.bind("<Visibility>", lambda _e: self._refresh_masks())
        try:
            self.cmb_mask.bind("<<ComboboxSelected>>", lambda _e: self._emit())
        except Exception:
            pass

    def _mask_source_keys(self) -> List[str]:
        try:
            f = getattr(self.ctx, "get_mask_keys", None)
            if callable(f):
                keys = list(f())
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
        cur = self.var_mask.get() or PLACEHOLDER_NONE
        if cur not in values:
            cur = PLACEHOLDER_NONE
        try:
            self.cmb_mask["values"] = values
        except Exception:
            pass
        self.var_mask.set(cur)

    # ---------- helpers ----------
    def _set_mask(self, key: str) -> None:
        self.var_mask.set(key if key else PLACEHOLDER_NONE)
        self._emit()

    def _apply_preset(self, **kw: Any) -> None:
        for k, v in kw.items():
            if k in ("mode","method"):
                getattr(self, f"var_{k}").set(str(v))
            elif k in ("min_period","max_period","grid_thickness","quilt_jitter"):
                getattr(self, f"var_{k if k!='min_period' else 'minp' if k!='max_period' else 'maxp'}").set(int(v))
            elif k in ("alpha","use_amp"):
                getattr(self, f"var_{'amp' if k=='use_amp' else k}").set(float(v))
        self._emit()

    def _emit(self) -> None:
        # sanity
        mn = int(max(2, self.var_minp.get()))
        mx = int(max(mn+1, self.var_maxp.get()))
        if self.var_maxp.get() != mx: self.var_maxp.set(mx)
        mk = (self.var_mask.get() or "").strip()

        params = {
            "mode":         self.var_mode.get().strip(),
            "min_period":   mn,
            "max_period":   mx,
            "method":       self.var_method.get().strip(),
            "alpha":        float(max(0.0, min(1.0, self.var_alpha.get()))),
            "grid_thickness": int(max(1, self.var_thick.get())),
            "quilt_jitter": int(max(0, self.var_jitter.get())),
            "mask_key":     (None if mk in ("", PLACEHOLDER_NONE) else mk),
            "use_amp":      float(max(0.0, self.var_amp.get())),
            "clamp":        bool(self.var_clamp.get()),
        }
        cb = getattr(self.ctx, "on_change", None)
        if callable(cb):
            try: cb(params)
            except Exception: pass

# Loader hook
Panel = TileTessProbePanel
