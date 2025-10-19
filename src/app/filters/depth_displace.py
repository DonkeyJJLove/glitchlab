# glitchlab/filters/depth_displace.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional, List

try:
    from glitchlab.app.panel_base import PanelContext  # type: ignore
except Exception:  # pragma: no cover
    class PanelContext:
        def __init__(self, **kw): self.__dict__.update(kw)

PLACEHOLDER_NONE = "<none>"


class DepthDisplacePanel(ttk.Frame):
    """
    Panel sterowania dla filtra 'depth_displace'.

    Parametry: depth_map, scale, freq, octaves, vertical,
               stereo, stereo_px, shading, shade_gain,
               mask_key, use_amp, clamp.

    Integracja z biblioteką masek:
      - wspólny Combobox (lista z ctx.get_mask_keys() / ctx.cache_ref['cfg/masks/keys'])
      - przyciski Refresh oraz auto-refresh przy wejściu/rozwinięciu
    """
    MAPS = ("noise_fractal", "perlin", "sine")

    def __init__(self, master: tk.Misc, ctx: Optional[PanelContext] = None, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.ctx = ctx or PanelContext(filter_name="depth_displace", defaults={}, params={}, on_change=None, cache_ref={})
        dflt: Dict[str, Any] = dict(getattr(self.ctx, "defaults", {}) or {})
        p0:   Dict[str, Any] = dict(getattr(self.ctx, "params", {}) or {})

        def V(k: str, fb: Any) -> Any: return p0.get(k, dflt.get(k, fb))

        self.var_depth_map  = tk.StringVar(value=str(V("depth_map", "noise_fractal")))
        self.var_scale      = tk.DoubleVar(value=float(V("scale", 56.0)))
        self.var_freq       = tk.DoubleVar(value=float(V("freq", 110.0)))
        self.var_octaves    = tk.IntVar(   value=int(  V("octaves", 5)))
        self.var_vertical   = tk.DoubleVar(value=float(V("vertical", 0.15)))
        self.var_stereo     = tk.BooleanVar(value=bool(V("stereo", True)))
        self.var_stereo_px  = tk.IntVar(   value=int(  V("stereo_px", 2)))
        self.var_shading    = tk.BooleanVar(value=bool(V("shading", True)))
        self.var_shade_gain = tk.DoubleVar(value=float(V("shade_gain", 0.25)))

        # mask_key – zintegrowany z biblioteką masek (Combobox)
        mk = str(V("mask_key", "") or "")
        self.var_mask_key   = tk.StringVar(value=(mk if mk else PLACEHOLDER_NONE))

        self.var_use_amp    = tk.DoubleVar(value=float(V("use_amp", 1.0)))
        self.var_clamp      = tk.BooleanVar(value=bool(V("clamp", True)))

        self._mask_keys: List[str] = []   # cache nazw masek do comboboxa

        self._build_ui()
        self._bind_all()

        # start: wypełnij listę masek i wyemituj parametry
        self._refresh_masks()
        self._emit()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        title = ttk.Frame(self, padding=(8, 8, 8, 4)); title.pack(fill="x")
        ttk.Label(title, text="Depth Displace", font=("", 10, "bold")).pack(side="left")
        ttk.Label(title, text=" — parallax z mapy głębi, stereo i cieniowaniem", foreground="#888").pack(side="left")

        # Depth field
        g1 = ttk.LabelFrame(self, text="Depth Field", padding=8); g1.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(g1, text="depth_map").grid(row=0, column=0, sticky="w")
        ttk.Combobox(g1, values=list(self.MAPS), state="readonly",
                     textvariable=self.var_depth_map, width=16)\
            .grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(g1, text="freq").grid(row=0, column=2, sticky="w")
        ttk.Scale(g1, from_=8.0, to=512.0, variable=self.var_freq)\
            .grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Entry(g1, textvariable=self.var_freq, width=7).grid(row=0, column=4, sticky="w")
        ttk.Label(g1, text="octaves").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(g1, from_=1, to=8, textvariable=self.var_octaves, width=6)\
            .grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))
        g1.columnconfigure(3, weight=1)

        # Parallax
        g2 = ttk.LabelFrame(self, text="Parallax", padding=8); g2.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(g2, text="scale (px)").grid(row=0, column=0, sticky="w")
        ttk.Scale(g2, from_=0.0, to=256.0, variable=self.var_scale)\
            .grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(g2, textvariable=self.var_scale, width=7).grid(row=0, column=2, sticky="w")
        ttk.Label(g2, text="vertical").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Scale(g2, from_=0.0, to=1.0, variable=self.var_vertical)\
            .grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        ttk.Entry(g2, textvariable=self.var_vertical, width=7)\
            .grid(row=1, column=2, sticky="w", pady=(6, 0))
        g2.columnconfigure(1, weight=1)

        # Stereo & Shading
        g3 = ttk.LabelFrame(self, text="Stereo & Shading", padding=8); g3.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Checkbutton(g3, text="stereo (anaglyph R/B)", variable=self.var_stereo)\
            .grid(row=0, column=0, sticky="w")
        ttk.Label(g3, text="stereo_px").grid(row=0, column=1, sticky="w")
        ttk.Spinbox(g3, from_=0, to=32, textvariable=self.var_stereo_px, width=6)\
            .grid(row=0, column=2, sticky="w", padx=6)
        ttk.Checkbutton(g3, text="shading", variable=self.var_shading)\
            .grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(g3, text="shade_gain").grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Scale(g3, from_=0.0, to=1.0, variable=self.var_shade_gain)\
            .grid(row=1, column=2, sticky="ew", padx=6, pady=(6, 0))
        ttk.Entry(g3, textvariable=self.var_shade_gain, width=6)\
            .grid(row=1, column=3, sticky="w", pady=(6, 0))
        g3.columnconfigure(2, weight=1)

        # Mask & Amplitude (zintegrowany wybór maski)
        g4 = ttk.LabelFrame(self, text="Mask & Amplitude", padding=8); g4.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(g4, text="mask_key").grid(row=0, column=0, sticky="w")

        mask_row = ttk.Frame(g4); mask_row.grid(row=0, column=1, columnspan=2, sticky="ew", padx=6)
        self.cmb_mask = ttk.Combobox(mask_row, state="readonly", width=20, textvariable=self.var_mask_key)
        self.cmb_mask.pack(side="left", fill="x", expand=True)
        # auto-refresh listy masek przy kliknięciu i wejściu
        self.cmb_mask.bind("<Button-1>", lambda _e: self._refresh_masks())
        ttk.Button(mask_row, text="Refresh", command=self._refresh_masks).pack(side="left", padx=(6, 0))

        ttk.Label(g4, text="use_amp").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Scale(g4, from_=0.0, to=2.0, variable=self.var_use_amp)\
            .grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        ttk.Entry(g4, textvariable=self.var_use_amp, width=6)\
            .grid(row=1, column=2, sticky="w", pady=(6, 0))
        g4.columnconfigure(1, weight=1)

        # Output
        g5 = ttk.LabelFrame(self, text="Output", padding=8); g5.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Checkbutton(g5, text="clamp (final clip to u8)", variable=self.var_clamp)\
            .grid(row=0, column=0, sticky="w")

        # Presets
        pr = ttk.Frame(self, padding=(8, 4, 8, 8)); pr.pack(fill="x")
        ttk.Label(pr, text="Presets:").pack(side="left")
        ttk.Button(
            pr, text="Stereo Pop",
            command=lambda: self._apply_preset(depth_map="noise_fractal", scale=56, freq=110, octaves=5,
                                               vertical=0.1, stereo=True, stereo_px=2,
                                               shading=True, shade_gain=0.25, use_amp=1.0)
        ).pack(side="left", padx=2)
        ttk.Button(
            pr, text="Vertical Parallax",
            command=lambda: self._apply_preset(depth_map="sine", scale=42, freq=64, octaves=1,
                                               vertical=0.4, stereo=False,
                                               shading=True, shade_gain=0.35, use_amp=0.8)
        ).pack(side="left", padx=2)
        ttk.Button(
            pr, text="Soft Depth",
            command=lambda: self._apply_preset(depth_map="perlin", scale=28, freq=160, octaves=4,
                                               vertical=0.0, stereo=False,
                                               shading=True, shade_gain=0.15, use_amp=0.6)
        ).pack(side="left", padx=2)
        ttk.Button(
            pr, text="Flat (no shade)",
            command=lambda: self._apply_preset(depth_map="sine", scale=40, freq=96, octaves=1,
                                               vertical=0.15, stereo=True, stereo_px=1,
                                               shading=False, shade_gain=0.0, use_amp=1.0)
        ).pack(side="left", padx=2)

    # ---------- masks & bindings ----------
    def _bind_all(self) -> None:
        vars_ = (
            self.var_depth_map, self.var_scale, self.var_freq, self.var_octaves,
            self.var_vertical, self.var_stereo, self.var_stereo_px,
            self.var_shading, self.var_shade_gain,
            self.var_mask_key, self.var_use_amp, self.var_clamp
        )
        for v in vars_:
            v.trace_add("write", lambda *_: self._emit())

        # odśwież listę masek, gdy panel staje się widoczny
        self.bind("<Visibility>", lambda _e: self._refresh_masks())
        self.cmb_mask.bind("<<ComboboxSelected>>", lambda _e: self._emit())

    def _mask_source_keys(self) -> List[str]:
        """Zbiera nazwy masek z Global: ctx.get_mask_keys() albo cache_ref['cfg/masks/keys']."""
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
        cur = self.var_mask_key.get() or PLACEHOLDER_NONE
        if cur not in values:
            cur = PLACEHOLDER_NONE
        self.cmb_mask["values"] = values
        self.var_mask_key.set(cur)

    # ---------- helpers ----------
    def _apply_preset(self, **kw: Any) -> None:
        for k, v in kw.items():
            if k in ("octaves", "stereo_px"):
                getattr(self, f"var_{k}").set(int(v))
            elif k in ("stereo", "shading", "clamp"):
                getattr(self, f"var_{k}").set(bool(v))
            elif k in ("depth_map",):
                getattr(self, f"var_{k}").set(str(v))
            else:
                getattr(self, f"var_{k}").set(float(v))
        self._emit()

    def _emit(self) -> None:
        mk = (self.var_mask_key.get().strip() or PLACEHOLDER_NONE)
        params = {
            "depth_map":  self.var_depth_map.get().strip() or "noise_fractal",
            "scale":      float(self.var_scale.get()),
            "freq":       float(self.var_freq.get()),
            "octaves":    int(max(1, int(self.var_octaves.get()))),
            "vertical":   float(max(0.0, self.var_vertical.get())),
            "stereo":     bool(self.var_stereo.get()),
            "stereo_px":  int(max(0, int(self.var_stereo_px.get()))),
            "shading":    bool(self.var_shading.get()),
            "shade_gain": float(max(0.0, min(1.0, self.var_shade_gain.get()))),
            "mask_key":   (None if mk in ("", PLACEHOLDER_NONE) else mk),
            "use_amp":    float(max(0.0, self.var_use_amp.get())),
            "clamp":      bool(self.var_clamp.get()),
        }
        cb = getattr(self.ctx, "on_change", None)
        if callable(cb):
            try:
                cb(params)
            except Exception:
                pass


# Loader hook
Panel = DepthDisplacePanel
