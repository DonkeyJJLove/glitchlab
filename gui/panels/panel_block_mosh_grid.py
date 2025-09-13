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


class BlockMoshGridPanel(ttk.Frame):
    """
    Panel sterowania dla filtra 'block_mosh_grid'.

    Najważniejsze:
    - mask_key to teraz wspólny Combobox z biblioteką masek (Global) + przycisk Refresh.
    - Lista masek pobierana z ctx.get_mask_keys() lub cache_ref["cfg/masks/keys"].
    - Każda zmiana emituje ctx.on_change(params).
    """
    def __init__(self, master: tk.Misc, ctx: Optional[PanelContext] = None, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.ctx = ctx or PanelContext(
            filter_name="block_mosh_grid",
            defaults={}, params={}, on_change=None, cache_ref={}
        )

        dflt: Dict[str, Any] = dict(getattr(self.ctx, "defaults", {}) or {})
        p0:   Dict[str, Any] = dict(getattr(self.ctx, "params", {}) or {})

        def V(key: str, fb: Any) -> Any: return p0.get(key, dflt.get(key, fb))

        # --- zmienne UI ---
        self.var_size          = tk.IntVar(   value=int(V("size", 24)))
        self.var_p             = tk.DoubleVar(value=float(V("p", 0.35)))
        self.var_max_shift     = tk.IntVar(   value=int(V("max_shift", 16)))
        self.var_mode          = tk.StringVar(value=str(V("mode", "shift")))
        self.var_swap_radius   = tk.IntVar(   value=int(V("swap_radius", 2)))
        self.var_rot_p         = tk.DoubleVar(value=float(V("rot_p", 0.0)))
        self.var_wrap          = tk.BooleanVar(value=bool(V("wrap", True)))
        self.var_channel_jit   = tk.DoubleVar(value=float(V("channel_jitter", 0.0)))
        self.var_poster_bits   = tk.IntVar(   value=int(V("posterize_bits", 0)))
        self.var_mix           = tk.DoubleVar(value=float(V("mix", 1.0)))
        # maski – teraz combobox
        self.var_mask_key      = tk.StringVar(value=str(V("mask_key", "") or PLACEHOLDER_NONE))
        self.var_use_amp       = tk.DoubleVar(value=float(V("use_amp", 1.0)))
        self.var_amp_infl      = tk.DoubleVar(value=float(V("amp_influence", 1.0)))
        self.var_clamp         = tk.BooleanVar(value=bool(V("clamp", True)))

        self._mask_keys: List[str] = []  # lokalny cache listy masek

        self._build_ui()
        self._bind_all()
        self._refresh_masks()
        self._emit()

    # ---------------- UI layout ----------------
    def _build_ui(self) -> None:
        title = ttk.Frame(self, padding=(8,8,8,4)); title.pack(fill="x")
        ttk.Label(title, text="Block Mosh (grid)", font=("", 10, "bold")).pack(side="left")
        ttk.Label(title, text="  — losowe przestawianie bloków: shift/swap/rot/posterize",
                  foreground="#888").pack(side="left")

        # GRID
        g = ttk.LabelFrame(self, text="Grid", padding=8); g.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(g, text="size (px)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(g, from_=4, to=256, increment=1, textvariable=self.var_size, width=8)\
            .grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(g, text="max_shift (px)").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(g, from_=0, to=128, increment=1, textvariable=self.var_max_shift, width=8)\
            .grid(row=0, column=3, sticky="w", padx=6)
        g.columnconfigure(4, weight=1)

        # SELECTION & FORCE
        sf = ttk.LabelFrame(self, text="Selection & Force", padding=8); sf.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(sf, text="p (block prob.)").grid(row=0, column=0, sticky="w")
        ttk.Scale(sf, from_=0.0, to=1.0, variable=self.var_p).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(sf, textvariable=self.var_p, width=6).grid(row=0, column=2, sticky="w")
        ttk.Label(sf, text="amp_influence").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Scale(sf, from_=0.0, to=2.0, variable=self.var_amp_infl).grid(row=1, column=1, sticky="ew", padx=6, pady=(6,0))
        ttk.Entry(sf, textvariable=self.var_amp_infl, width=6).grid(row=1, column=2, sticky="w", pady=(6,0))
        sf.columnconfigure(1, weight=1)

        # OPERATIONS
        op = ttk.LabelFrame(self, text="Operations", padding=8); op.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(op, text="mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(op, values=("shift", "swap", "shift+swap"),
                     textvariable=self.var_mode, state="readonly", width=12)\
            .grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(op, text="swap_radius (blocks)").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(op, from_=0, to=12, textvariable=self.var_swap_radius, width=8)\
            .grid(row=0, column=3, sticky="w", padx=6)
        ttk.Label(op, text="rot_p").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Scale(op, from_=0.0, to=1.0, variable=self.var_rot_p)\
            .grid(row=1, column=1, sticky="ew", padx=6, pady=(6,0))
        ttk.Entry(op, textvariable=self.var_rot_p, width=6).grid(row=1, column=2, sticky="w", pady=(6,0))
        ttk.Checkbutton(op, text="wrap (roll inside patch)", variable=self.var_wrap)\
            .grid(row=1, column=3, sticky="w", padx=6, pady=(6,0))
        op.columnconfigure(1, weight=1)

        # COLOR & POST
        cp = ttk.LabelFrame(self, text="Color & Post", padding=8); cp.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(cp, text="channel_jitter (px)").grid(row=0, column=0, sticky="w")
        ttk.Scale(cp, from_=0.0, to=24.0, variable=self.var_channel_jit)\
            .grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(cp, textvariable=self.var_channel_jit, width=6)\
            .grid(row=0, column=2, sticky="w")
        ttk.Label(cp, text="posterize_bits").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Spinbox(cp, from_=0, to=7, textvariable=self.var_poster_bits, width=8)\
            .grid(row=1, column=1, sticky="w", padx=6, pady=(6,0))
        cp.columnconfigure(1, weight=1)

        # MASK & AMP — wspólny Combobox masek
        ma = ttk.LabelFrame(self, text="Mask & Amplitude", padding=8); ma.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(ma, text="mask_key").grid(row=0, column=0, sticky="w")
        row = ttk.Frame(ma); row.grid(row=0, column=1, columnspan=2, sticky="ew", padx=6)
        self.cmb_mask = ttk.Combobox(row, state="readonly", width=24, textvariable=self.var_mask_key)
        self.cmb_mask.pack(side="left", fill="x", expand=True)
        self.cmb_mask.bind("<Button-1>", lambda _e: self._refresh_masks())  # auto-refresh przy rozwinięciu
        ttk.Button(row, text="Refresh", command=self._refresh_masks).pack(side="left", padx=(6,0))
        ma.columnconfigure(1, weight=1)

        ttk.Label(ma, text="use_amp").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Scale(ma, from_=0.0, to=2.0, variable=self.var_use_amp)\
            .grid(row=1, column=1, sticky="ew", padx=6, pady=(6,0))
        ttk.Entry(ma, textvariable=self.var_use_amp, width=6)\
            .grid(row=1, column=2, sticky="w", pady=(6,0))

        # MIX & BOUNDARIES
        mb = ttk.LabelFrame(self, text="Mix & Boundaries", padding=8); mb.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(mb, text="mix").grid(row=0, column=0, sticky="w")
        ttk.Scale(mb, from_=0.0, to=1.0, variable=self.var_mix)\
            .grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(mb, textvariable=self.var_mix, width=6).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(mb, text="clamp (final clip to u8)", variable=self.var_clamp)\
            .grid(row=1, column=0, sticky="w", pady=(6,0))
        mb.columnconfigure(1, weight=1)

        # PRESETS
        pr = ttk.Frame(self, padding=(8,4,8,8)); pr.pack(fill="x")
        ttk.Label(pr, text="Presets:").pack(side="left")
        ttk.Button(pr, text="Subtle (grid 32)",
                   command=lambda: self._apply_preset(
                       size=32, p=0.25, max_shift=8, mode="shift", swap_radius=0,
                       rot_p=0.0, wrap=True, channel_jitter=0.0, posterize_bits=0,
                       mix=0.6, use_amp=0.8, amp_influence=0.8
                   )).pack(side="left", padx=2)
        ttk.Button(pr, text="Aggressive swap",
                   command=lambda: self._apply_preset(
                       size=24, p=0.6, max_shift=24, mode="shift+swap", swap_radius=2,
                       rot_p=0.2, wrap=True, channel_jitter=2.0, posterize_bits=1,
                       mix=0.95, use_amp=1.2, amp_influence=1.2
                   )).pack(side="left", padx=2)
        ttk.Button(pr, text="JPEG-ish blocks",
                   command=lambda: self._apply_preset(
                       size=16, p=0.45, max_shift=12, mode="shift",
                       swap_radius=0, rot_p=0.0, wrap=True, channel_jitter=0.0,
                       posterize_bits=0, mix=0.9, use_amp=1.0, amp_influence=1.0
                   )).pack(side="left", padx=2)

    # ------------- maski -------------
    def _mask_source_keys(self) -> List[str]:
        """Pobierz listę masek z ctx.get_mask_keys() lub z cache_ref['cfg/masks/keys']."""
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
        current = self.var_mask_key.get() or PLACEHOLDER_NONE
        if current not in values:
            current = PLACEHOLDER_NONE
        self.cmb_mask["values"] = values
        self.var_mask_key.set(current)

    # ------------- helpers/bindings -------------
    def _bind_all(self) -> None:
        vars_ = (
            self.var_size, self.var_p, self.var_max_shift, self.var_mode, self.var_swap_radius,
            self.var_rot_p, self.var_wrap, self.var_channel_jit, self.var_poster_bits,
            self.var_mix, self.var_mask_key, self.var_use_amp, self.var_amp_infl, self.var_clamp
        )
        for v in vars_:
            v.trace_add("write", lambda *_: self._emit())

        # odśwież listę masek gdy panel pojawi się/otrzyma focus
        self.bind("<Visibility>", lambda _e: self._refresh_masks())
        self.cmb_mask.bind("<<ComboboxSelected>>", lambda _e: self._emit())

    def _apply_preset(self, **kw: Any) -> None:
        # bezpieczne ustawienia typów
        if "size" in kw:            self.var_size.set(int(kw["size"]))
        if "p" in kw:               self.var_p.set(float(kw["p"]))
        if "max_shift" in kw:       self.var_max_shift.set(int(kw["max_shift"]))
        if "mode" in kw:            self.var_mode.set(str(kw["mode"]))
        if "swap_radius" in kw:     self.var_swap_radius.set(int(kw["swap_radius"]))
        if "rot_p" in kw:           self.var_rot_p.set(float(kw["rot_p"]))
        if "wrap" in kw:            self.var_wrap.set(bool(kw["wrap"]))
        if "channel_jitter" in kw:  self.var_channel_jit.set(float(kw["channel_jitter"]))
        if "posterize_bits" in kw:  self.var_poster_bits.set(int(kw["posterize_bits"]))
        if "mix" in kw:             self.var_mix.set(float(kw["mix"]))
        if "use_amp" in kw:         self.var_use_amp.set(float(kw["use_amp"]))
        if "amp_influence" in kw:   self.var_amp_infl.set(float(kw["amp_influence"]))
        self._emit()

    # ------------- emit -------------
    def _emit(self) -> None:
        mk = (self.var_mask_key.get().strip())
        params = {
            "size":          max(4, int(self.var_size.get())),
            "p":             float(min(max(self.var_p.get(), 0.0), 1.0)),
            "max_shift":     max(0, int(self.var_max_shift.get())),
            "mode":          str(self.var_mode.get()).lower(),
            "swap_radius":   max(0, int(self.var_swap_radius.get())),
            "rot_p":         float(min(max(self.var_rot_p.get(), 0.0), 1.0)),
            "wrap":          bool(self.var_wrap.get()),
            "channel_jitter":float(max(0.0, self.var_channel_jit.get())),
            "posterize_bits":max(0, int(self.var_poster_bits.get())),
            "mix":           float(min(max(self.var_mix.get(), 0.0), 1.0)),
            "mask_key":      (None if mk in ("", PLACEHOLDER_NONE) else mk),
            "use_amp":       float(max(0.0, self.var_use_amp.get())),
            "amp_influence": float(max(0.0, self.var_amp_infl.get())),
            "clamp":         bool(self.var_clamp.get()),
        }
        cb = getattr(self.ctx, "on_change", None)
        if callable(cb):
            try: cb(params)
            except Exception: pass


# Loader hook for panel_loader
Panel = BlockMoshGridPanel
