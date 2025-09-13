# glitchlab/gui/panels/panel_spectral_shaper.py
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
    "mode": "ring",         # ring|bandpass|bandstop|direction
    "low": 0.15,
    "high": 0.45,
    "angle_deg": 0.0,
    "ang_width": 20.0,
    "boost": 0.8,
    "soft": 0.08,
    "blend": 0.0,
    "mask_key": None,
    # wspólne:
    "use_amp": 1.0,
    "clamp": True,
}

class SpectralShaperPanel(ttk.Frame):
    """
    Panel sterujący dla 'spectral_shaper'.
    Tryby: ring/bandpass/bandstop/direction, pełne parametry + presety.
    Zintegrowany wybór maski (wspólna biblioteka), placeholder <none>, auto-odświeżanie.
    """
    def __init__(self, master: tk.Misc, ctx: Optional[PanelContext] = None, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.ctx = ctx or PanelContext(filter_name="spectral_shaper", defaults=DEFAULTS, params={}, on_change=None, cache_ref={})
        p = {**DEFAULTS, **(getattr(self.ctx, "defaults", {}) or {}), **(getattr(self.ctx, "params", {}) or {})}

        # zmienne
        self.var_mode      = tk.StringVar(value=str(p.get("mode", "ring")))
        self.var_low       = tk.DoubleVar(value=float(p.get("low", 0.15)))
        self.var_high      = tk.DoubleVar(value=float(p.get("high", 0.45)))
        self.var_angle     = tk.DoubleVar(value=float(p.get("angle_deg", 0.0)))
        self.var_angw      = tk.DoubleVar(value=float(p.get("ang_width", 20.0)))
        self.var_boost     = tk.DoubleVar(value=float(p.get("boost", 0.8)))
        self.var_soft      = tk.DoubleVar(value=float(p.get("soft", 0.08)))
        self.var_blend     = tk.DoubleVar(value=float(p.get("blend", 0.0)))
        self.var_mask_key  = tk.StringVar(value=str(p.get("mask_key", "") or PLACEHOLDER_NONE))
        self.var_use_amp   = tk.DoubleVar(value=float(p.get("use_amp", 1.0)))
        self.var_clamp     = tk.BooleanVar(value=bool(p.get("clamp", True)))

        self._build_ui()
        self._bind_all()
        self._refresh_masks()  # start: lista masek
        self._emit()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=(8,8,8,4)); top.pack(fill="x")
        ttk.Label(top, text="Spectral Shaper", font=("", 10, "bold")).pack(side="left")

        row0 = ttk.LabelFrame(self, text="Tryb", padding=8); row0.pack(fill="x", padx=8, pady=4)
        ttk.Label(row0, text="mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(row0, textvariable=self.var_mode, state="readonly",
                     values=["ring", "bandpass", "bandstop", "direction"], width=12)\
            .grid(row=0, column=1, sticky="w", padx=6)

        row1 = ttk.LabelFrame(self, text="Pasmo radialne", padding=8); row1.pack(fill="x", padx=8, pady=4)
        ttk.Label(row1, text="low").grid(row=0, column=0, sticky="w")
        ttk.Scale(row1, from_=0.0, to=1.0, variable=self.var_low, command=lambda *_: self._clamp_low_high())\
            .grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(row1, textvariable=self.var_low, width=6).grid(row=0, column=2, sticky="w")
        ttk.Label(row1, text="high").grid(row=0, column=3, sticky="w")
        ttk.Scale(row1, from_=0.0, to=1.0, variable=self.var_high, command=lambda *_: self._clamp_low_high())\
            .grid(row=0, column=4, sticky="ew", padx=6)
        ttk.Entry(row1, textvariable=self.var_high, width=6).grid(row=0, column=5, sticky="w")
        row1.columnconfigure(1, weight=1)
        row1.columnconfigure(4, weight=1)

        row2 = ttk.LabelFrame(self, text="Kierunek (dla mode=direction)", padding=8); row2.pack(fill="x", padx=8, pady=4)
        ttk.Label(row2, text="angle_deg").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(row2, from_=-180.0, to=180.0, increment=1.0, textvariable=self.var_angle, width=8)\
            .grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(row2, text="ang_width").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(row2, from_=1.0, to=180.0, increment=1.0, textvariable=self.var_angw, width=8)\
            .grid(row=0, column=3, sticky="w", padx=6)

        row3 = ttk.LabelFrame(self, text="Modyfikacja i miks", padding=8); row3.pack(fill="x", padx=8, pady=4)
        ttk.Label(row3, text="boost").grid(row=0, column=0, sticky="w")
        ttk.Scale(row3, from_=-1.0, to=3.0, variable=self.var_boost).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(row3, textvariable=self.var_boost, width=6).grid(row=0, column=2, sticky="w")
        ttk.Label(row3, text="soft").grid(row=0, column=3, sticky="w")
        ttk.Scale(row3, from_=0.0, to=1.0, variable=self.var_soft).grid(row=0, column=4, sticky="ew", padx=6)
        ttk.Entry(row3, textvariable=self.var_soft, width=6).grid(row=0, column=5, sticky="w")
        ttk.Label(row3, text="blend").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Scale(row3, from_=0.0, to=1.0, variable=self.var_blend).grid(row=1, column=1, sticky="ew", padx=6, pady=(6,0))
        ttk.Entry(row3, textvariable=self.var_blend, width=6).grid(row=1, column=2, sticky="w", pady=(6,0))
        row3.columnconfigure(1, weight=1)
        row3.columnconfigure(4, weight=1)

        # Maski / amplitude / clamp (wspólna biblioteka masek)
        row4 = ttk.LabelFrame(self, text="Maska & Amplitude", padding=8); row4.pack(fill="x", padx=8, pady=4)
        ttk.Label(row4, text="mask_key").grid(row=0, column=0, sticky="w")
        mask_row = ttk.Frame(row4); mask_row.grid(row=0, column=1, columnspan=3, sticky="ew", padx=6)
        self.cmb_mask = ttk.Combobox(mask_row, state="readonly", width=24, textvariable=self.var_mask_key, values=[])
        self.cmb_mask.pack(side="left", fill="x", expand=True)
        self.cmb_mask.bind("<Button-1>", lambda _e: self._refresh_masks())  # auto-refresh po kliknięciu
        ttk.Button(mask_row, text="Refresh", command=self._refresh_masks).pack(side="left", padx=(6, 0))
        ttk.Button(mask_row, text="edge", command=lambda: self._set_mask("edge")).pack(side="left", padx=(6, 2))
        ttk.Button(mask_row, text="clear", command=lambda: self._set_mask("")).pack(side="left")

        ttk.Label(row4, text="use_amp").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Scale(row4, from_=0.0, to=2.0, variable=self.var_use_amp, orient="horizontal")\
            .grid(row=1, column=1, sticky="ew", padx=6, pady=(6,0))
        ttk.Entry(row4, textvariable=self.var_use_amp, width=6).grid(row=1, column=2, sticky="w", pady=(6,0))
        ttk.Checkbutton(row4, text="clamp", variable=self.var_clamp).grid(row=1, column=3, sticky="w", pady=(6,0))
        row4.columnconfigure(1, weight=1)

        pres = ttk.Frame(self, padding=(8,2,8,8)); pres.pack(fill="x")
        ttk.Label(pres, text="Presets:").pack(side="left")
        ttk.Button(pres, text="Ring boost",
                   command=lambda: self._apply_preset(mode="ring", low=0.18, high=0.5, boost=1.2, soft=0.1, blend=0.0))\
            .pack(side="left", padx=2)
        ttk.Button(pres, text="Bandstop low",
                   command=lambda: self._apply_preset(mode="bandstop", low=0.0, high=0.12, boost=0.7, soft=0.08, blend=0.15))\
            .pack(side="left", padx=2)
        ttk.Button(pres, text="Directional 45°",
                   command=lambda: self._apply_preset(mode="direction", low=0.12, high=0.7, angle_deg=45.0, ang_width=22.0, boost=1.0, soft=0.06))\
            .pack(side="left", padx=2)

    # ---------- binding / masks ----------
    def _bind_all(self) -> None:
        for v in (self.var_mode, self.var_low, self.var_high, self.var_angle, self.var_angw,
                  self.var_boost, self.var_soft, self.var_blend,
                  self.var_mask_key, self.var_use_amp, self.var_clamp):
            v.trace_add("write", lambda *_: self._emit())
        # odśwież listę, gdy panel staje się widoczny
        self.bind("<Visibility>", lambda _e: self._refresh_masks())
        # natychmiast po wyborze z listy
        try:
            self.cmb_mask.bind("<<ComboboxSelected>>", lambda _e: self._emit())
        except Exception:
            pass

    def _mask_source_keys(self) -> List[str]:
        """Lista masek z Global: ctx.get_mask_keys() lub cache_ref['cfg/masks/keys']."""
        # 1) przez provider
        try:
            f = getattr(self.ctx, "get_mask_keys", None)
            if callable(f):
                keys = list(f())
                return [k for k in keys if isinstance(k, str)]
        except Exception:
            pass
        # 2) przez cache_ref
        try:
            cache = getattr(self.ctx, "cache_ref", {}) or {}
            keys = list(cache.get("cfg/masks/keys", []))
            return [k for k in keys if isinstance(k, str)]
        except Exception:
            return []

    def _refresh_masks(self) -> None:
        keys = self._mask_source_keys()
        values = [PLACEHOLDER_NONE] + sorted(keys)
        cur = self.var_mask_key.get() or PLACEHOLDER_NONE
        if cur not in values:
            cur = PLACEHOLDER_NONE
        try:
            self.cmb_mask["values"] = values
        except Exception:
            pass
        self.var_mask_key.set(cur)

    # ---------- helpers ----------
    def _clamp_low_high(self) -> None:
        lo = float(self.var_low.get()); hi = float(self.var_high.get())
        if hi < lo:
            self.var_high.set(lo)

    def _set_mask(self, key: str) -> None:
        self.var_mask_key.set(key if key else PLACEHOLDER_NONE)
        self._emit()

    def _apply_preset(self, **kw: Any) -> None:
        for k, v in kw.items():
            if k in ("mode",):
                getattr(self, f"var_{k}").set(str(v))
            elif k in ("low", "high", "boost", "soft", "blend"):
                getattr(self, f"var_{k}").set(float(v))
            elif k in ("angle_deg",):
                self.var_angle.set(float(v))
            elif k in ("ang_width",):
                self.var_angw.set(float(v))
            elif k == "mask_key":
                self.var_mask_key.set(str(v) if v else PLACEHOLDER_NONE)
        self._emit()

    def _emit(self) -> None:
        # sanity i ograniczenia
        lo = float(self.var_low.get()); hi = float(self.var_high.get())
        if hi < lo:
            hi = lo
            self.var_high.set(hi)

        mk = (self.var_mask_key.get() or "").strip()
        params = {
            "mode":      self.var_mode.get().strip(),
            "low":       float(max(0.0, min(1.5, lo))),
            "high":      float(max(0.0, min(1.5, hi))),
            "angle_deg": float(self.var_angle.get()),
            "ang_width": float(max(1.0, min(180.0, self.var_angw.get()))),
            "boost":     float(self.var_boost.get()),
            "soft":      float(max(0.0, min(1.0, self.var_soft.get()))),
            "blend":     float(max(0.0, min(1.0, self.var_blend.get()))),
            "mask_key":  (None if mk in ("", PLACEHOLDER_NONE) else mk),
            # wspólne:
            "use_amp":   float(max(0.0, self.var_use_amp.get())),
            "clamp":     bool(self.var_clamp.get()),
        }
        cb = getattr(self.ctx, "on_change", None)
        if callable(cb):
            try:
                cb(params)
            except Exception:
                pass

# Loader hook
Panel = SpectralShaperPanel
