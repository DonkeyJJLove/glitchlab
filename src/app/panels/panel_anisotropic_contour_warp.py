# glitchlab.app.panels.panel_anisotropic_contour_warp.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional, List


# Kontekst panelu – miękki fallback gdy panel_base nie jest dostępny
try:
    from glitchlab.gui.panel_base import PanelContext  # type: ignore
except Exception:  # pragma: no cover
    class PanelContext:
        def __init__(self, **kw): self.__dict__.update(kw)


PLACEHOLDER_NONE = "<none>"


class ACWPanel(ttk.Frame):
    """
    Panel sterowania dla filtra 'anisotropic_contour_warp'.
    - Maski: rozwijana lista spięta z biblioteką masek (Global).
    - Każda zmiana parametrów emituje ctx.on_change(params).
    """
    def __init__(self, master: tk.Misc, ctx: Optional[PanelContext] = None, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.ctx = ctx or PanelContext(
            filter_name="anisotropic_contour_warp",
            defaults={}, params={}, on_change=None, cache_ref={}
        )

        # --- wartości początkowe (defaults ⨝ params) ---
        dflt: Dict[str, Any] = dict(getattr(self.ctx, "defaults", {}) or {})
        p0: Dict[str, Any] = dict(getattr(self.ctx, "params", {}) or {})
        def V(key: str, fb: Any) -> Any: return p0.get(key, dflt.get(key, fb))

        self.var_strength  = tk.DoubleVar(value=float(V("strength", 1.5)))
        self.var_ksize     = tk.StringVar(value=str(int(V("ksize", 3))))
        self.var_iters     = tk.IntVar(value=int(V("iters", 1)))
        self.var_smooth    = tk.DoubleVar(value=float(V("smooth", 0.0)))
        self.var_edge_bias = tk.DoubleVar(value=float(V("edge_bias", 0.0)))
        self.var_mask_key  = tk.StringVar(value=str(V("mask_key", "") or PLACEHOLDER_NONE))
        self.var_use_amp   = tk.DoubleVar(value=float(V("use_amp", 1.0)))
        self.var_clamp     = tk.BooleanVar(value=bool(V("clamp", True)))

        # lokalny cache listy masek
        self._mask_keys: List[str] = []

        # UI
        self._build_ui()
        self._bind_all()

        # pierwszy refresh listy masek + emit
        self._refresh_masks()
        self._emit()

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8); top.pack(fill="x")
        ttk.Label(top, text="Anisotropic Contour Warp", font=("", 10, "bold")).pack(side="left")

        # Strength & Iterations
        r1 = ttk.LabelFrame(self, text="Strength & Iterations", padding=8); r1.pack(fill="x", padx=8, pady=(4,6))
        ttk.Label(r1, text="strength").grid(row=0, column=0, sticky="w")
        ttk.Scale(r1, from_=0.0, to=5.0, variable=self.var_strength)\
            .grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(r1, textvariable=self.var_strength, width=6).grid(row=0, column=2, sticky="w")
        ttk.Label(r1, text="iters").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Spinbox(r1, from_=1, to=12, textvariable=self.var_iters, width=6)\
            .grid(row=1, column=1, sticky="w", padx=6, pady=(6,0))
        r1.columnconfigure(1, weight=1)

        # Gradients
        r2 = ttk.LabelFrame(self, text="Gradients", padding=8); r2.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(r2, text="ksize").grid(row=0, column=0, sticky="w")
        ttk.Combobox(r2, values=("3", "5"), textvariable=self.var_ksize,
                     state="readonly", width=6)\
            .grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(r2, text="smooth (σ)").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Scale(r2, from_=0.0, to=3.0, variable=self.var_smooth, orient="horizontal")\
            .grid(row=1, column=1, sticky="ew", padx=6, pady=(6,0))
        ttk.Entry(r2, textvariable=self.var_smooth, width=6).grid(row=1, column=2, sticky="w", pady=(6,0))
        r2.columnconfigure(1, weight=1)

        # Bias & ROI (maski!)
        r3 = ttk.LabelFrame(self, text="Bias & ROI", padding=8); r3.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(r3, text="edge_bias").grid(row=0, column=0, sticky="w")
        ttk.Scale(r3, from_=-2.0, to=2.0, variable=self.var_edge_bias)\
            .grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(r3, textvariable=self.var_edge_bias, width=6).grid(row=0, column=2, sticky="w")

        ttk.Label(r3, text="mask_key").grid(row=1, column=0, sticky="w", pady=(6,0))
        mask_row = ttk.Frame(r3); mask_row.grid(row=1, column=1, columnspan=2, sticky="ew", padx=6, pady=(6,0))
        self.cmb_mask = ttk.Combobox(mask_row, state="readonly", width=28, textvariable=self.var_mask_key)
        self.cmb_mask.pack(side="left", fill="x", expand=True)
        # odśwież przy rozwinięciu listy
        self.cmb_mask.bind("<Button-1>", lambda _e: self._refresh_masks())
        ttk.Button(mask_row, text="Refresh", command=self._refresh_masks).pack(side="left", padx=(6,0))
        r3.columnconfigure(1, weight=1)

        # Amplitude & Boundary
        r4 = ttk.LabelFrame(self, text="Amplitude & Boundary", padding=8); r4.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(r4, text="use_amp").grid(row=0, column=0, sticky="w")
        ttk.Scale(r4, from_=0.0, to=2.0, variable=self.var_use_amp)\
            .grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(r4, textvariable=self.var_use_amp, width=6).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(r4, text="clamp", variable=self.var_clamp)\
            .grid(row=1, column=0, sticky="w", pady=(6,0))

        # Quick presets
        qp = ttk.Frame(self, padding=(8,0)); qp.pack(fill="x", pady=(0,6))
        ttk.Button(qp, text="Soft flow",
                   command=lambda: self._apply_preset(0.6, 3, 1, 0.0, 0.0)).pack(side="left", padx=2)
        ttk.Button(qp, text="Edge flow",
                   command=lambda: self._apply_preset(1.5, 3, 2, 0.0, 0.8)).pack(side="left", padx=2)
        ttk.Button(qp, text="Smoothed",
                   command=lambda: self._apply_preset(1.0, 5, 2, 1.0, 0.3)).pack(side="left", padx=2)

    # ---------------- BINDINGS ----------------
    def _bind_all(self) -> None:
        vars_to_watch = (
            self.var_strength, self.var_ksize, self.var_iters, self.var_smooth,
            self.var_edge_bias, self.var_mask_key, self.var_use_amp, self.var_clamp
        )
        for v in vars_to_watch:
            v.trace_add("write", lambda *_: self._emit())

        # Autoodświeżenie listy masek po pokazaniu panelu
        self.bind("<Visibility>", lambda _e: self._refresh_masks())

    # ---------------- MASKI ----------------
    def _mask_source_keys(self) -> List[str]:
        """Spróbuj pobrać listę masek z ctx.get_mask_keys() albo z cache."""
        # 1) provider z kontekstu
        try:
            f = getattr(self.ctx, "get_mask_keys", None)
            if callable(f):
                keys = list(f())
                return [k for k in keys if isinstance(k, str)]
        except Exception:
            pass
        # 2) cache: "cfg/masks/keys" (ustawiane przez App po wczytaniu/edycji masek)
        try:
            cache = getattr(self.ctx, "cache_ref", {}) or {}
            keys = list(cache.get("cfg/masks/keys", []))
            return [k for k in keys if isinstance(k, str)]
        except Exception:
            return []

    def _refresh_masks(self) -> None:
        keys = self._mask_source_keys()
        keys_ui = [PLACEHOLDER_NONE] + sorted(keys)
        self._mask_keys = keys  # zapamiętaj ostatnią listę

        # zachowaj wybór jeśli istnieje
        current = self.var_mask_key.get() or PLACEHOLDER_NONE
        if current not in keys_ui:
            current = PLACEHOLDER_NONE

        self.cmb_mask["values"] = keys_ui
        self.var_mask_key.set(current)

    # ---------------- PRESETY SZYBKIE ----------------
    def _apply_preset(self, strength: float, ksize: int, iters: int, smooth: float, edge_bias: float) -> None:
        self.var_strength.set(strength)
        self.var_ksize.set(str(ksize))
        self.var_iters.set(iters)
        self.var_smooth.set(smooth)
        self.var_edge_bias.set(edge_bias)
        self._emit()

    # ---------------- EMIT ----------------
    def _emit(self) -> None:
        mk = self.var_mask_key.get().strip()
        params = {
            "strength": float(self.var_strength.get()),
            "ksize": int(self.var_ksize.get()),
            "iters": int(self.var_iters.get()),
            "smooth": float(self.var_smooth.get()),
            "edge_bias": float(self.var_edge_bias.get()),
            "mask_key": (None if mk in ("", PLACEHOLDER_NONE) else mk),
            "use_amp": float(self.var_use_amp.get()),
            "clamp": bool(self.var_clamp.get()),
        }
        cb = getattr(self.ctx, "on_change", None)
        if callable(cb):
            try:
                cb(params)
            except Exception:
                pass


# Loader hook dla App._mount_panel_for(...)
Panel = ACWPanel
