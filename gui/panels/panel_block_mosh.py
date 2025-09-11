# glitchlab/gui/panels/panel_block_mosh.py
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

class BlockMoshPanel(ttk.Frame):
    """
    Panel sterowania dla filtra 'block_mosh' (simple).
    Parametry: size, p, max_shift, per_channel, wrap, mix, mask_key, use_amp, amp_influence, clamp.
    Presety: Subtle / Classic / Heavy.
    """
    def __init__(self, master: tk.Misc, ctx: Optional[PanelContext] = None, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.ctx = ctx or PanelContext(filter_name="block_mosh", defaults={}, params={}, on_change=None, cache_ref={})
        dflt: Dict[str, Any] = dict(self.ctx.defaults or {})
        p0: Dict[str, Any] = dict(self.ctx.params or {})

        def V(k: str, fb: Any) -> Any: return p0.get(k, dflt.get(k, fb))

        self.var_size        = tk.IntVar(   value=int(V("size", 24)))
        self.var_p           = tk.DoubleVar(value=float(V("p", 0.33)))
        self.var_max_shift   = tk.IntVar(   value=int(V("max_shift", 8)))
        self.var_per_channel = tk.BooleanVar(value=bool(V("per_channel", False)))
        self.var_wrap        = tk.BooleanVar(value=bool(V("wrap", True)))
        self.var_mix         = tk.DoubleVar(value=float(V("mix", 1.0)))
        self.var_mask_key    = tk.StringVar(value=str(V("mask_key", "") or ""))
        self.var_use_amp     = tk.DoubleVar(value=float(V("use_amp", 1.0)))
        self.var_amp_infl    = tk.DoubleVar(value=float(V("amp_influence", 1.0)))
        self.var_clamp       = tk.BooleanVar(value=bool(V("clamp", True)))

        self._build_ui()
        self._bind_all()
        self._emit()

    def _build_ui(self) -> None:
        title = ttk.Frame(self, padding=(8,8,8,4)); title.pack(fill="x")
        ttk.Label(title, text="Block Mosh (simple)", font=("", 10, "bold")).pack(side="left")
        ttk.Label(title, text=" — roll bloków; maska+amplitude; diagnostyka HUD", foreground="#888").pack(side="left")

        g = ttk.LabelFrame(self, text="Grid & Force", padding=8); g.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(g, text="size (px)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(g, from_=4, to=256, textvariable=self.var_size, width=8)\
            .grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(g, text="p (block prob.)").grid(row=0, column=2, sticky="w")
        ttk.Scale(g, from_=0.0, to=1.0, variable=self.var_p).grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Entry(g, textvariable=self.var_p, width=6).grid(row=0, column=4, sticky="w", padx=(4,0))
        ttk.Label(g, text="max_shift (px)").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Spinbox(g, from_=0, to=128, textvariable=self.var_max_shift, width=8)\
            .grid(row=1, column=1, sticky="w", padx=6, pady=(6,0))
        g.columnconfigure(3, weight=1)

        op = ttk.LabelFrame(self, text="Operation", padding=8); op.pack(fill="x", padx=8, pady=(0,6))
        ttk.Checkbutton(op, text="per_channel (independent RGB shifts)", variable=self.var_per_channel)\
            .grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(op, text="wrap (roll) else clamp", variable=self.var_wrap)\
            .grid(row=0, column=1, sticky="w", padx=12)
        ttk.Label(op, text="mix").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Scale(op, from_=0.0, to=1.0, variable=self.var_mix)\
            .grid(row=1, column=1, sticky="ew", padx=6, pady=(6,0))
        ttk.Entry(op, textvariable=self.var_mix, width=6).grid(row=1, column=2, sticky="w", pady=(6,0))
        op.columnconfigure(1, weight=1)

        ma = ttk.LabelFrame(self, text="Mask & Amplitude", padding=8); ma.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(ma, text="mask_key").grid(row=0, column=0, sticky="w")
        ttk.Entry(ma, textvariable=self.var_mask_key, width=16).grid(row=0, column=1, sticky="w", padx=6)
        btns = ttk.Frame(ma); btns.grid(row=0, column=2, sticky="w")
        ttk.Button(btns, text="edge", command=lambda: self._set_mask("edge")).pack(side="left", padx=(0,4))
        ttk.Button(btns, text="clear", command=lambda: self._set_mask("")).pack(side="left")
        ttk.Label(ma, text="use_amp").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Scale(ma, from_=0.0, to=2.0, variable=self.var_use_amp)\
            .grid(row=1, column=1, sticky="ew", padx=6, pady=(6,0))
        ttk.Entry(ma, textvariable=self.var_use_amp, width=6)\
            .grid(row=1, column=2, sticky="w", pady=(6,0))
        ttk.Label(ma, text="amp_influence").grid(row=2, column=0, sticky="w", pady=(6,0))
        ttk.Scale(ma, from_=0.0, to=2.0, variable=self.var_amp_infl)\
            .grid(row=2, column=1, sticky="ew", padx=6, pady=(6,0))
        ttk.Entry(ma, textvariable=self.var_amp_infl, width=6)\
            .grid(row=2, column=2, sticky="w", pady=(6,0))
        ma.columnconfigure(1, weight=1)

        mb = ttk.LabelFrame(self, text="Output", padding=8); mb.pack(fill="x", padx=8, pady=(0,6))
        ttk.Checkbutton(mb, text="clamp (final clip to u8)", variable=self.var_clamp)\
            .grid(row=0, column=0, sticky="w")

        pr = ttk.Frame(self, padding=(8,4,8,8)); pr.pack(fill="x")
        ttk.Label(pr, text="Presets:").pack(side="left")
        ttk.Button(pr, text="Subtle",
                   command=lambda: self._apply_preset(size=32, p=0.2, max_shift=6, mix=0.6, use_amp=0.8, amp_influence=0.8, per_channel=False))\
            .pack(side="left", padx=2)
        ttk.Button(pr, text="Classic",
                   command=lambda: self._apply_preset(size=24, p=0.33, max_shift=8, mix=0.9, use_amp=1.0, amp_influence=1.0, per_channel=False))\
            .pack(side="left", padx=2)
        ttk.Button(pr, text="Heavy",
                   command=lambda: self._apply_preset(size=16, p=0.6, max_shift=16, mix=1.0, use_amp=1.2, amp_influence=1.2, per_channel=True))\
            .pack(side="left", padx=2)

    def _bind_all(self) -> None:
        vars_ = (
            self.var_size, self.var_p, self.var_max_shift, self.var_per_channel, self.var_wrap,
            self.var_mix, self.var_mask_key, self.var_use_amp, self.var_amp_infl, self.var_clamp
        )
        for v in vars_:
            v.trace_add("write", lambda *_: self._emit())

    def _set_mask(self, key: str) -> None:
        self.var_mask_key.set(key); self._emit()

    def _apply_preset(self, **kw: Any) -> None:
        for k, v in kw.items():
            if k in ("size","max_shift"): getattr(self, f"var_{k}").set(int(v))
            elif k in ("per_channel","wrap","clamp"): getattr(self, f"var_{k}").set(bool(v))
            else: getattr(self, f"var_{k}").set(float(v))
        self._emit()

    def _emit(self) -> None:
        params = {
            "size":          max(4, int(self.var_size.get())),
            "p":             float(min(max(self.var_p.get(), 0.0), 1.0)),
            "max_shift":     max(0, int(self.var_max_shift.get())),
            "per_channel":   bool(self.var_per_channel.get()),
            "wrap":          bool(self.var_wrap.get()),
            "mix":           float(min(max(self.var_mix.get(), 0.0), 1.0)),
            "mask_key":      (self.var_mask_key.get().strip() or None),
            "use_amp":       float(max(0.0, self.var_use_amp.get())),
            "amp_influence": float(max(0.0, self.var_amp_infl.get())),
            "clamp":         bool(self.var_clamp.get()),
        }
        if callable(getattr(self.ctx, "on_change", None)):
            try: self.ctx.on_change(params)
            except Exception: pass

# Loader hook
Panel = BlockMoshPanel
