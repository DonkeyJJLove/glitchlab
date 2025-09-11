# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional

try:
    from glitchlab.gui.panel_base import PanelContext  # type: ignore
except Exception:  # pragma: no cover
    class PanelContext:
        def __init__(self, **kw): self.__dict__.update(kw)

class PhaseGlitchPanel(ttk.Frame):
    """
    Panel sterowania dla filtra 'phase_glitch'.
    Kontroluje pasmo (low/high), siłę (strength), zachowanie DC, miks i maskę.
    Z presetami pasm: Low Ring / Mid Ring / High Ring.
    """

    def __init__(self, master: tk.Misc, ctx: Optional[PanelContext] = None, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.ctx = ctx or PanelContext(filter_name="phase_glitch", defaults={}, params={}, on_change=None, cache_ref={})
        dflt: Dict[str, Any] = dict(self.ctx.defaults or {})
        p0: Dict[str, Any] = dict(self.ctx.params or {})

        def V(k: str, fb: Any) -> Any: return p0.get(k, dflt.get(k, fb))

        self.var_low        = tk.DoubleVar(value=float(V("low", 0.18)))
        self.var_high       = tk.DoubleVar(value=float(V("high", 0.60)))
        self.var_strength   = tk.DoubleVar(value=float(V("strength", 0.60)))
        self.var_preserve   = tk.BooleanVar(value=bool(V("preserve_dc", True)))
        self.var_blend      = tk.DoubleVar(value=float(V("blend", 0.00)))
        self.var_mask_key   = tk.StringVar(value=str(V("mask_key", "") or ""))

        self._build_ui()
        self._bind_all()
        self._emit()

    def _build_ui(self) -> None:
        title = ttk.Frame(self, padding=(8,8,8,4)); title.pack(fill="x")
        ttk.Label(title, text="Phase Glitch", font=("", 10, "bold")).pack(side="left")
        ttk.Label(title, text=" — randomizacja fazy w paśmie", foreground="#888").pack(side="left")

        g1 = ttk.LabelFrame(self, text="Pasmo (promień, 0..1)", padding=8); g1.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(g1, text="low").grid(row=0, column=0, sticky="w")
        ttk.Scale(g1, from_=0.0, to=1.0, variable=self.var_low).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(g1, textvariable=self.var_low, width=6).grid(row=0, column=2, sticky="w")
        ttk.Label(g1, text="high").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Scale(g1, from_=0.0, to=1.0, variable=self.var_high).grid(row=1, column=1, sticky="ew", padx=6, pady=(6,0))
        ttk.Entry(g1, textvariable=self.var_high, width=6).grid(row=1, column=2, sticky="w", pady=(6,0))
        g1.columnconfigure(1, weight=1)

        g2 = ttk.LabelFrame(self, text="Siła / Miks", padding=8); g2.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(g2, text="strength").grid(row=0, column=0, sticky="w")
        ttk.Scale(g2, from_=0.0, to=1.0, variable=self.var_strength).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(g2, textvariable=self.var_strength, width=6).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(g2, text="preserve DC", variable=self.var_preserve).grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Label(g2, text="blend").grid(row=1, column=1, sticky="w", pady=(6,0))
        ttk.Scale(g2, from_=0.0, to=1.0, variable=self.var_blend).grid(row=1, column=2, sticky="ew", padx=6, pady=(6,0))
        ttk.Entry(g2, textvariable=self.var_blend, width=6).grid(row=1, column=3, sticky="w", pady=(6,0))
        g2.columnconfigure(2, weight=1)

        g3 = ttk.LabelFrame(self, text="Maska (miks przestrzenny)", padding=8); g3.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(g3, text="mask_key").grid(row=0, column=0, sticky="w")
        ttk.Entry(g3, textvariable=self.var_mask_key, width=16).grid(row=0, column=1, sticky="w", padx=6)
        btns = ttk.Frame(g3); btns.grid(row=0, column=2, sticky="w")
        ttk.Button(btns, text="edge", command=lambda: self._set_mask("edge")).pack(side="left", padx=(0,4))
        ttk.Button(btns, text="clear", command=lambda: self._set_mask("")).pack(side="left")

        pr = ttk.Frame(self, padding=(8,4,8,8)); pr.pack(fill="x")
        ttk.Label(pr, text="Presets:").pack(side="left")
        ttk.Button(pr, text="Low Ring",
                   command=lambda: self._apply_preset(low=0.05, high=0.20, strength=0.6, preserve_dc=True, blend=0.0))\
            .pack(side="left", padx=2)
        ttk.Button(pr, text="Mid Ring",
                   command=lambda: self._apply_preset(low=0.18, high=0.60, strength=0.7, preserve_dc=True, blend=0.1))\
            .pack(side="left", padx=2)
        ttk.Button(pr, text="High Ring",
                   command=lambda: self._apply_preset(low=0.55, high=0.90, strength=0.8, preserve_dc=True, blend=0.15))\
            .pack(side="left", padx=2)

    def _bind_all(self) -> None:
        for v in (self.var_low, self.var_high, self.var_strength, self.var_preserve, self.var_blend, self.var_mask_key):
            v.trace_add("write", lambda *_: self._emit())

        # wzajemne ograniczenie low ≤ high
        self.var_low.trace_add("write", lambda *_: self._clamp_band())
        self.var_high.trace_add("write", lambda *_: self._clamp_band())

    def _clamp_band(self) -> None:
        try:
            lo = float(self.var_low.get()); hi = float(self.var_high.get())
        except Exception:
            return
        lo = max(0.0, min(1.0, lo)); hi = max(0.0, min(1.0, hi))
        if hi < lo:
            # prosta polityka: dosuwamy hi do lo
            hi = lo
            self.var_high.set(hi)

    def _set_mask(self, key: str) -> None:
        self.var_mask_key.set(key); self._emit()

    def _apply_preset(self, **kw: Any) -> None:
        for k, v in kw.items():
            if k in ("preserve_dc",):
                getattr(self, f"var_{k}").set(bool(v))
            elif k in ("mask_key",):
                getattr(self, f"var_{k}").set(str(v))
            else:
                getattr(self, f"var_{k}").set(float(v))
        self._emit()

    def _emit(self) -> None:
        params = {
            "low":        float(max(0.0, min(1.0, self.var_low.get()))),
            "high":       float(max(0.0, min(1.0, self.var_high.get()))),
            "strength":   float(max(0.0, min(1.0, self.var_strength.get()))),
            "preserve_dc": bool(self.var_preserve.get()),
            "blend":      float(max(0.0, min(1.0, self.var_blend.get()))),
            "mask_key":   (self.var_mask_key.get().strip() or None),
        }
        if callable(getattr(self.ctx, "on_change", None)):
            try: self.ctx.on_change(params)
            except Exception: pass

# Loader hook dla panel_loadera:
Panel = PhaseGlitchPanel
