# glitchlab/gui/panels/panel_gamma_gain.py
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

class GammaGainPanel(ttk.Frame):
    """
    Panel sterowania dla filtra 'gamma_gain'.
    Parametry: gamma, mask_key, use_amp, clamp.
    Integracja mask_key ze wspólną biblioteką masek (Global).
    """
    def __init__(self, master: tk.Misc, ctx: Optional[PanelContext] = None, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.ctx = ctx or PanelContext(filter_name="gamma_gain",
                                       defaults={}, params={}, on_change=None, cache_ref={})
        dflt: Dict[str, Any] = dict(getattr(self.ctx, "defaults", {}) or {})
        p0:   Dict[str, Any] = dict(getattr(self.ctx, "params", {}) or {})

        def V(k: str, fb: Any) -> Any: return p0.get(k, dflt.get(k, fb))

        self.var_gamma    = tk.DoubleVar(value=float(V("gamma", 1.0)))
        self.var_mask_key = tk.StringVar(value=str(V("mask_key", "") or PLACEHOLDER_NONE))
        self.var_use_amp  = tk.DoubleVar(value=float(V("use_amp", 1.0)))
        self.var_clamp    = tk.BooleanVar(value=bool(V("clamp", True)))

        self._build_ui()
        self._bind_all()
        self._refresh_masks()  # start: zsynchronizuj listę z Global
        self._emit()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        title = ttk.Frame(self, padding=(8,8,8,4)); title.pack(fill="x")
        ttk.Label(title, text="Gamma / Gain", font=("", 10, "bold")).pack(side="left")
        ttk.Label(title, text=" — prosty korektor gamma", foreground="#888").pack(side="left")

        g = ttk.LabelFrame(self, text="Gamma", padding=8); g.pack(fill="x", padx=8, pady=(0,6))
        ttk.Scale(g, from_=0.10, to=5.00, variable=self.var_gamma, orient="horizontal")\
            .grid(row=0, column=0, sticky="ew", padx=(0,6))
        ttk.Entry(g, textvariable=self.var_gamma, width=7).grid(row=0, column=1, sticky="w")
        # presety
        pb = ttk.Frame(g); pb.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6,0))
        ttk.Label(pb, text="Presets:").pack(side="left")
        ttk.Button(pb, text="Soft (0.85)",   command=lambda: self._apply_preset(gamma=0.85)).pack(side="left", padx=2)
        ttk.Button(pb, text="Neutral (1.0)", command=lambda: self._apply_preset(gamma=1.0)).pack(side="left", padx=2)
        ttk.Button(pb, text="Contrast (1.15)", command=lambda: self._apply_preset(gamma=1.15)).pack(side="left", padx=2)
        ttk.Button(pb, text="Strong (1.6)",  command=lambda: self._apply_preset(gamma=1.6)).pack(side="left", padx=2)
        g.columnconfigure(0, weight=1)

        ma = ttk.LabelFrame(self, text="Mask & Amplitude", padding=8); ma.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(ma, text="mask_key").grid(row=0, column=0, sticky="w")
        row = ttk.Frame(ma); row.grid(row=0, column=1, columnspan=2, sticky="ew", padx=6)
        self.cmb_mask = ttk.Combobox(row, state="readonly", width=24, textvariable=self.var_mask_key)
        self.cmb_mask.pack(side="left", fill="x", expand=True)
        self.cmb_mask.bind("<Button-1>", lambda _e: self._refresh_masks())  # auto-refresh przy rozwinięciu
        ttk.Button(row, text="edge",  command=lambda: self._set_mask("edge")).pack(side="left", padx=(6, 2))
        ttk.Button(row, text="clear", command=lambda: self._set_mask("")).pack(side="left")

        ttk.Label(ma, text="use_amp").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Scale(ma, from_=0.0, to=2.0, variable=self.var_use_amp, orient="horizontal")\
            .grid(row=1, column=1, sticky="ew", padx=6, pady=(6,0))
        ttk.Entry(ma, textvariable=self.var_use_amp, width=7).grid(row=1, column=2, sticky="w", pady=(6,0))
        ma.columnconfigure(1, weight=1)

        out = ttk.LabelFrame(self, text="Output", padding=8); out.pack(fill="x", padx=8, pady=(0,6))
        ttk.Checkbutton(out, text="clamp (final clip to u8)", variable=self.var_clamp).grid(row=0, column=0, sticky="w")

    # ---------- bind / masks ----------
    def _bind_all(self) -> None:
        for v in (self.var_gamma, self.var_mask_key, self.var_use_amp, self.var_clamp):
            v.trace_add("write", lambda *_: self._emit())
        # gdy panel pojawia się — odśwież listę
        self.bind("<Visibility>", lambda _e: self._refresh_masks())
        # natychmiast po wyborze z listy
        try:
            self.cmb_mask.bind("<<ComboboxSelected>>", lambda _e: self._emit())
        except Exception:
            pass

    def _mask_source_keys(self) -> List[str]:
        """Lista masek z Global: ctx.get_mask_keys() lub cache_ref['cfg/masks/keys']."""
        # preferuj funkcję dostarczoną przez App (bieżący stan)
        try:
            f = getattr(self.ctx, "get_mask_keys", None)
            if callable(f):
                keys = list(f() or [])
                return [k for k in keys if isinstance(k, str)]
        except Exception:
            pass
        # fallback do cache (ostatni znany zestaw)
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
    def _set_mask(self, key: str) -> None:
        self.var_mask_key.set(key if key else PLACEHOLDER_NONE)
        self._emit()

    def _apply_preset(self, **kw: Any) -> None:
        if "gamma" in kw: self.var_gamma.set(float(kw["gamma"]))
        self._emit()

    def _emit(self) -> None:
        mk = (self.var_mask_key.get() or "").strip()
        params = {
            "gamma":   float(max(0.05, self.var_gamma.get())),
            "mask_key": None if mk in ("", PLACEHOLDER_NONE) else mk,
            "use_amp": float(max(0.0, self.var_use_amp.get())),
            "clamp":   bool(self.var_clamp.get()),
        }
        cb = getattr(self.ctx, "on_change", None)
        if callable(cb):
            try: cb(params)
            except Exception: pass

# Loader hook dla panel_loader
Panel = GammaGainPanel
