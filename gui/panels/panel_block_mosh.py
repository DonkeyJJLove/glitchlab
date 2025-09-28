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

class BlockMoshPanel(ttk.Frame):
    """
    Panel sterowania dla filtra 'block_mosh' (simple).
    Parametry: size, p, max_shift, per_channel, wrap, mix, mask_key,
               use_amp, amp_influence, clamp.
    Presety: Subtle / Classic / Heavy.
    """
    def __init__(self, master: tk.Misc, ctx: Optional[PanelContext] = None, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.ctx = ctx or PanelContext(filter_name="block_mosh", defaults={}, params={}, on_change=None, cache_ref={})
        dflt: Dict[str, Any] = dict(self.ctx.defaults or {})
        p0:   Dict[str, Any] = dict(self.ctx.params or {})

        def V(k: str, fb: Any) -> Any: return p0.get(k, dflt.get(k, fb))

        self.var_size        = tk.IntVar(   value=int(V("size", 24)))
        self.var_p           = tk.DoubleVar(value=float(V("p", 0.33)))
        self.var_max_shift   = tk.IntVar(   value=int(V("max_shift", 8)))
        self.var_per_channel = tk.BooleanVar(value=bool(V("per_channel", False)))
        self.var_wrap        = tk.BooleanVar(value=bool(V("wrap", True)))
        self.var_mix         = tk.DoubleVar(value=float(V("mix", 1.0)))
        # ⬇️ ujednolicony placeholder dla braku maski
        self.var_mask_key    = tk.StringVar(value=str(V("mask_key", "") or PLACEHOLDER_NONE))
        self.var_use_amp     = tk.DoubleVar(value=float(V("use_amp", 1.0)))
        # UI ma var_amp_infl, API: amp_influence
        self.var_amp_infl    = tk.DoubleVar(value=float(V("amp_influence", 1.0)))
        self.var_clamp       = tk.BooleanVar(value=bool(V("clamp", True)))

        self._build_ui()
        self._bind_all()
        self._refresh_masks()
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

        # ⬇️ Maska & amplitude — COMBO zamiast Entry, z Refresh i skrótami
        ma = ttk.LabelFrame(self, text="Mask & Amplitude", padding=8); ma.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(ma, text="mask_key").grid(row=0, column=0, sticky="w")
        row = ttk.Frame(ma); row.grid(row=0, column=1, columnspan=2, sticky="ew", padx=6)
        self.cmb_mask = ttk.Combobox(row, state="readonly", width=24, textvariable=self.var_mask_key, values=[PLACEHOLDER_NONE])
        self.cmb_mask.pack(side="left", fill="x", expand=True)
        self.cmb_mask.bind("<Button-1>", lambda _e: self._refresh_masks())
        ttk.Button(row, text="Refresh", command=self._refresh_masks).pack(side="left", padx=(6, 2))
        ttk.Button(row, text="edge", command=lambda: self._set_mask("edge")).pack(side="left", padx=(0, 2))
        ttk.Button(row, text="clear", command=lambda: self._set_mask("")).pack(side="left")
        ma.columnconfigure(1, weight=1)

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
        # odśwież przy pojawieniu się i reaguj na zmianę w comboboxie
        self.bind("<Visibility>", lambda _e: self._refresh_masks())
        try:
            self.cmb_mask.bind("<<ComboboxSelected>>", lambda _e: self._emit())
        except Exception:
            pass

    # ---- maski (wspólna biblioteka) ----
    def _mask_source_keys(self) -> List[str]:
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
        cur = self.var_mask_key.get() or PLACEHOLDER_NONE
        if cur not in values:
            cur = PLACEHOLDER_NONE
        try:
            self.cmb_mask["values"] = values
        except Exception:
            pass
        self.var_mask_key.set(cur)

    # ---- reszta ----
    def _set_mask(self, key: str) -> None:
        self.var_mask_key.set(key if key else PLACEHOLDER_NONE)
        self._emit()

    def _apply_preset(self, **kw: Any) -> None:
        keymap = {"amp_influence": "amp_infl"}
        for k, v in kw.items():
            tgt = keymap.get(k, k)
            try:
                if tgt in ("size","max_shift"):
                    getattr(self, f"var_{tgt}").set(int(v))
                elif tgt in ("per_channel","wrap","clamp"):
                    getattr(self, f"var_{tgt}").set(bool(v))
                elif tgt == "mask_key":
                    self._set_mask(str(v))
                else:
                    getattr(self, f"var_{tgt}").set(float(v))
            except Exception:
                pass
        self._emit()

    def _emit(self) -> None:
        def fget(var: tk.Variable, fallback: float = 0.0) -> float:
            try: return float(var.get())
            except Exception: return float(fallback)

        mk = (self.var_mask_key.get() or "").strip()
        params = {
            "size":          max(4, int(self.var_size.get())),
            "p":             float(min(max(fget(self.var_p), 0.0), 1.0)),
            "max_shift":     max(0, int(self.var_max_shift.get())),
            "per_channel":   bool(self.var_per_channel.get()),
            "wrap":          bool(self.var_wrap.get()),
            "mix":           float(min(max(fget(self.var_mix), 0.0), 1.0)),
            "mask_key":      (None if mk in ("", PLACEHOLDER_NONE) else mk),
            "use_amp":       float(max(0.0, fget(self.var_use_amp, 1.0))),
            "amp_influence": float(max(0.0, fget(self.var_amp_infl, 1.0))),
            "clamp":         bool(self.var_clamp.get()),
        }
        if callable(getattr(self.ctx, "on_change", None)):
            try: self.ctx.on_change(params)
            except Exception: pass

# Loader hook
Panel = BlockMoshPanel
