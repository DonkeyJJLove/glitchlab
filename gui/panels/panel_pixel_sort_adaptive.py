# glitchlab/gui/panels/panel_pixel_sort_adaptive.py
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
    "direction": "vertical",
    "trigger": "edges",
    "threshold": 0.35,
    "mask_key": "",
    "length_px": 160,
    "length_gain": 1.0,
    "prob": 1.0,
    "key": "luma",
    "reverse": False,
}

class PixelSortAdaptivePanel(ttk.Frame):
    """
    Panel sterowania dla 'pixel_sort_adaptive'.
    Spójny z resztą GUI wybór maski: combobox z placeholderem '<none>' i auto-refresh.
    """
    def __init__(self, master: tk.Misc, ctx: Optional[PanelContext] = None, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.ctx = ctx or PanelContext(filter_name="pixel_sort_adaptive", defaults=DEFAULTS, params={}, on_change=None, cache_ref={})

        p0 = {**DEFAULTS, **(getattr(self.ctx, "defaults", {}) or {}), **(getattr(self.ctx, "params", {}) or {})}

        self.var_direction   = tk.StringVar(value=str(p0.get("direction", "vertical")))
        self.var_trigger     = tk.StringVar(value=str(p0.get("trigger", "edges")))
        self.var_threshold   = tk.DoubleVar(value=float(p0.get("threshold", 0.35)))
        # mask_key jako combobox z placeholderem
        init_mask = str(p0.get("mask_key", "") or PLACEHOLDER_NONE)
        self.var_mask_key    = tk.StringVar(value=init_mask)
        self.var_length_px   = tk.IntVar(value=int(p0.get("length_px", 160)))
        self.var_length_gain = tk.DoubleVar(value=float(p0.get("length_gain", 1.0)))
        self.var_prob        = tk.DoubleVar(value=float(p0.get("prob", 1.0)))
        self.var_key         = tk.StringVar(value=str(p0.get("key", "luma")))
        self.var_reverse     = tk.BooleanVar(value=bool(p0.get("reverse", False)))

        self._build_ui()
        self._bind_all()
        self._refresh_masks()
        self._emit()

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=(8,8,8,4)); top.pack(fill="x")
        ttk.Label(top, text="Pixel Sort Adaptive", font=("", 10, "bold")).pack(side="left")

        row1 = ttk.LabelFrame(self, text="Mode", padding=8); row1.pack(fill="x", padx=8, pady=4)
        ttk.Label(row1, text="direction").grid(row=0, column=0, sticky="w")
        ttk.Combobox(row1, textvariable=self.var_direction, state="readonly",
                     values=["vertical", "horizontal"], width=12).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(row1, text="key").grid(row=0, column=2, sticky="w")
        ttk.Combobox(row1, textvariable=self.var_key, state="readonly",
                     values=["luma","r","g","b","sat","hue"], width=10).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Checkbutton(row1, text="reverse", variable=self.var_reverse).grid(row=0, column=4, sticky="w", padx=6)
        row1.columnconfigure(5, weight=1)

        row2 = ttk.LabelFrame(self, text="Trigger", padding=8); row2.pack(fill="x", padx=8, pady=4)
        ttk.Label(row2, text="trigger").grid(row=0, column=0, sticky="w")
        ttk.Combobox(row2, textvariable=self.var_trigger, state="readonly",
                     values=["edges","luma","mask"], width=12).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(row2, text="threshold").grid(row=0, column=2, sticky="w")
        ttk.Scale(row2, from_=0.0, to=1.0, variable=self.var_threshold).grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Entry(row2, textvariable=self.var_threshold, width=6).grid(row=0, column=4, sticky="w")

        ttk.Label(row2, text="mask_key").grid(row=1, column=0, sticky="w", pady=(6,0))
        mask_row = ttk.Frame(row2); mask_row.grid(row=1, column=1, columnspan=3, sticky="ew", padx=6, pady=(6,0))
        self.cmb_mask = ttk.Combobox(mask_row, state="readonly", width=24,
                                     textvariable=self.var_mask_key,
                                     postcommand=self._refresh_masks)
        self.cmb_mask.pack(side="left", fill="x", expand=True)
        self.cmb_mask.bind("<<ComboboxSelected>>", lambda _e: self._emit())
        ttk.Button(mask_row, text="edge",  command=lambda: self._set_mask("edge")).pack(side="left", padx=(6,2))
        ttk.Button(mask_row, text="clear", command=lambda: self._set_mask(PLACEHOLDER_NONE)).pack(side="left")
        row2.columnconfigure(3, weight=1)

        row3 = ttk.LabelFrame(self, text="Segments", padding=8); row3.pack(fill="x", padx=8, pady=4)
        ttk.Label(row3, text="length_px").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(row3, from_=4, to=4096, textvariable=self.var_length_px, width=8).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(row3, text="length_gain").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(row3, from_=-4.0, to=8.0, increment=0.1, textvariable=self.var_length_gain, width=8).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Label(row3, text="prob").grid(row=0, column=4, sticky="w")
        ttk.Scale(row3, from_=0.0, to=1.0, variable=self.var_prob).grid(row=0, column=5, sticky="ew", padx=6)
        ttk.Entry(row3, textvariable=self.var_prob, width=6).grid(row=0, column=6, sticky="w")
        row3.columnconfigure(5, weight=1)

        pres = ttk.Frame(self, padding=(8,2,8,8)); pres.pack(fill="x")
        ttk.Label(pres, text="Presets:").pack(side="left")
        ttk.Button(pres, text="Edges strong (V)",
                   command=lambda: self._apply_preset(direction="vertical", trigger="edges", threshold=0.3,
                                                      length_px=200, length_gain=1.2, prob=1.0, key="luma", reverse=False))\
            .pack(side="left", padx=2)
        ttk.Button(pres, text="Luma artistic (H)",
                   command=lambda: self._apply_preset(direction="horizontal", trigger="luma", threshold=0.6,
                                                      length_px=140, length_gain=0.8, prob=0.8, key="hue", reverse=True))\
            .pack(side="left", padx=2)
        ttk.Button(pres, text="Masked gentle",
                   command=lambda: self._apply_preset(direction="horizontal", trigger="mask", threshold=0.5,
                                                      length_px=80, length_gain=0.6, prob=0.6, key="luma", reverse=False, mask_key="edge"))\
            .pack(side="left", padx=2)

        # auto-refresh listy masek przy pokazaniu panelu
        self.bind("<Visibility>", lambda _e: self._refresh_masks())

    # ---------- maski ----------
    def _mask_source_keys(self) -> List[str]:
        """Zaciągnij żywą listę masek (Global) lub fallback z cache."""
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
        cur = (self.var_mask_key.get() or PLACEHOLDER_NONE)
        if cur not in values:
            cur = PLACEHOLDER_NONE
        try:
            self.cmb_mask.set(cur)
        except Exception:
            self.var_mask_key.set(cur)

    # ---------- bindings / presets ----------
    def _bind_all(self) -> None:
        vars_ = (self.var_direction, self.var_trigger, self.var_threshold, self.var_mask_key,
                 self.var_length_px, self.var_length_gain, self.var_prob, self.var_key, self.var_reverse)
        for v in vars_:
            v.trace_add("write", lambda *_: self._emit())

    def _apply_preset(self, **kw: Any) -> None:
        for k, v in kw.items():
            if k == "reverse":
                self.var_reverse.set(bool(v))
            elif k in ("direction","trigger","key"):
                getattr(self, f"var_{k}").set(str(v))
            elif k == "mask_key":
                self.var_mask_key.set(str(v) if v else PLACEHOLDER_NONE)
            else:
                getattr(self, f"var_{k}").set(float(v))
        self._emit()

    def _set_mask(self, key: str) -> None:
        self.var_mask_key.set(key if key else PLACEHOLDER_NONE)
        self._emit()

    # ---------- emit ----------
    def _emit(self) -> None:
        mk = (self.var_mask_key.get() or "").strip()
        params = {
            "direction":   self.var_direction.get().strip(),
            "trigger":     self.var_trigger.get().strip(),
            "threshold":   float(max(0.0, min(1.0, self.var_threshold.get()))),
            "mask_key":    (None if mk in ("", PLACEHOLDER_NONE) else mk),
            "length_px":   int(max(4, int(self.var_length_px.get()))),
            "length_gain": float(self.var_length_gain.get()),
            "prob":        float(max(0.0, min(1.0, self.var_prob.get()))),
            "key":         self.var_key.get().strip(),
            "reverse":     bool(self.var_reverse.get()),
        }
        cb = getattr(self.ctx, "on_change", None)
        if callable(cb):
            try: cb(params)
            except Exception: pass

# Loader hook:
Panel = PixelSortAdaptivePanel
