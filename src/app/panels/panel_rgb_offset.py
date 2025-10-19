# glitchlab/app/panels/panel_rgb_offset.py
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
PANEL_FILTER = "rgb_offset"


class RgbOffsetPanel(ttk.Frame):
    """
    Panel do 'rgb_offset' z:
      - subpikselowym dx/dy per-kanał + global dx/dy (sync/copy),
      - mix (0..1), wrap(edge) boundary, clamp,
      - use_amp, mask_key z globalnej listy (Combobox + <none>),
      - presety: Classic/Halo/Green/Reset.
    """

    def __init__(self, master: tk.Misc, ctx: Optional[PanelContext] = None, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.ctx = ctx or PanelContext(filter_name=PANEL_FILTER, defaults={}, params={}, on_change=None, cache_ref={})
        dflt: Dict[str, Any] = dict(getattr(self.ctx, "defaults", {}) or {})
        p0: Dict[str, Any] = dict(getattr(self.ctx, "params", {}) or {})

        def V(k: str, fb: Any) -> Any: return p0.get(k, dflt.get(k, fb))

        # mix / edges / clamp
        self.var_mix = tk.DoubleVar(value=float(V("mix", 1.0)))
        self.var_wrap = tk.BooleanVar(value=bool(V("wrap", False)))  # True=wrap, False=edge clamp
        self.var_clamp = tk.BooleanVar(value=bool(V("clamp", True)))

        # amp / mask (global combobox)
        self.var_use_amp = tk.DoubleVar(value=float(V("use_amp", 1.0)))
        self.var_mask_key = tk.StringVar(value=str(V("mask_key", "") or PLACEHOLDER_NONE))
        self._mask_keys: List[str] = []

        # global dx/dy (float)
        self.var_dx = tk.DoubleVar(value=float(V("dx", 0.0)))
        self.var_dy = tk.DoubleVar(value=float(V("dy", 0.0)))

        # per-channel (float)
        self.var_dx_r = tk.DoubleVar(value=float(V("dx_r", 2.0)))
        self.var_dy_r = tk.DoubleVar(value=float(V("dy_r", 0.0)))
        self.var_dx_g = tk.DoubleVar(value=float(V("dx_g", 0.0)))
        self.var_dy_g = tk.DoubleVar(value=float(V("dy_g", 0.0)))
        self.var_dx_b = tk.DoubleVar(value=float(V("dx_b", -2.0)))
        self.var_dy_b = tk.DoubleVar(value=float(V("dy_b", 0.0)))

        # sync flags
        self.var_sync_r = tk.BooleanVar(value=False)
        self.var_sync_g = tk.BooleanVar(value=False)
        self.var_sync_b = tk.BooleanVar(value=False)

        self._build_ui()
        self._bind_all()
        self._refresh_masks()
        self._emit()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=(8, 8, 8, 4));
        top.pack(fill="x")
        ttk.Label(top, text="RGB Offset", font=("", 10, "bold")).pack(side="left")

        # Presets
        pf = ttk.Frame(self, padding=(8, 0, 8, 8));
        pf.pack(fill="x")
        ttk.Label(pf, text="Presets:").pack(side="left")
        ttk.Button(pf, text="Classic split", command=self._preset_classic).pack(side="left", padx=2)
        ttk.Button(pf, text="Halo", command=self._preset_halo).pack(side="left", padx=2)
        ttk.Button(pf, text="G push", command=self._preset_green).pack(side="left", padx=2)
        ttk.Button(pf, text="Reset", command=self._preset_reset).pack(side="left", padx=2)

        # Mix & edges
        me = ttk.LabelFrame(self, text="Mix & Edges", padding=8);
        me.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(me, text="mix").grid(row=0, column=0, sticky="w")
        ttk.Scale(me, from_=0.0, to=1.0, variable=self.var_mix).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(me, textvariable=self.var_mix, width=6).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(me, text="wrap (else: edge clamp)", variable=self.var_wrap).grid(row=0, column=3, sticky="w",
                                                                                         padx=8)
        ttk.Checkbutton(me, text="clamp output", variable=self.var_clamp).grid(row=0, column=4, sticky="w", padx=8)
        me.columnconfigure(1, weight=1)

        # Amplitude & mask
        am = ttk.LabelFrame(self, text="Amplitude & Mask", padding=8);
        am.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(am, text="use_amp").grid(row=0, column=0, sticky="w")
        ttk.Scale(am, from_=0.0, to=2.0, variable=self.var_use_amp).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(am, textvariable=self.var_use_amp, width=6).grid(row=0, column=2, sticky="w")
        ttk.Label(am, text="mask_key").grid(row=1, column=0, sticky="w", pady=(6, 0))
        row = ttk.Frame(am);
        row.grid(row=1, column=1, columnspan=2, sticky="ew", padx=6, pady=(6, 0))
        self.cmb_mask = ttk.Combobox(row, state="readonly", width=24, textvariable=self.var_mask_key, values=[])
        self.cmb_mask.pack(side="left", fill="x", expand=True)
        self.cmb_mask.bind("<Button-1>", lambda _e: self._refresh_masks())
        ttk.Button(row, text="Refresh", command=self._refresh_masks).pack(side="left", padx=(6, 0))
        am.columnconfigure(1, weight=1)

        # Global offset
        gf = ttk.LabelFrame(self, text="Global Offset (px, float)", padding=8);
        gf.pack(fill="x", padx=8, pady=(0, 6))
        self._mk_float(gf, "dx", self.var_dx).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self._mk_float(gf, "dy", self.var_dy).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Button(gf, text="Copy → R/G/B", command=self._copy_global_to_all).grid(row=0, column=2, sticky="w", padx=8)

        # Per-channel
        cf = ttk.LabelFrame(self, text="Per-channel Offsets (px, float)", padding=8);
        cf.pack(fill="x", padx=8, pady=(0, 8))
        self._mk_chan(cf, "R", self.var_dx_r, self.var_dy_r, self.var_sync_r, 0)
        self._mk_chan(cf, "G", self.var_dx_g, self.var_dy_g, self.var_sync_g, 1)
        self._mk_chan(cf, "B", self.var_dx_b, self.var_dy_b, self.var_sync_b, 2)
        tools = ttk.Frame(cf);
        tools.grid(row=3, column=0, columnspan=5, sticky="w", pady=(6, 0))
        ttk.Button(tools, text="±1 jitter", command=self._jitter_1px).pack(side="left", padx=4)
        ttk.Button(tools, text="Sync enabled now", command=self._sync_enabled_now).pack(side="left", padx=4)

    def _mk_float(self, parent: tk.Misc, label: str, var: tk.DoubleVar) -> ttk.Frame:
        fr = ttk.Frame(parent)
        ttk.Label(fr, text=label).pack(side="left")
        e = ttk.Entry(fr, textvariable=var, width=7);
        e.pack(side="left", padx=4)
        s = ttk.Scale(fr, from_=-64.0, to=64.0, variable=var);
        s.pack(side="left", fill="x", expand=True)
        return fr

    def _mk_chan(self, parent: tk.Misc, name: str,
                 vdx: tk.DoubleVar, vdy: tk.DoubleVar, vsync: tk.BooleanVar, row: int) -> None:
        ttk.Label(parent, text=f"{name}:").grid(row=row, column=0, sticky="w")
        self._mk_float(parent, "dx", vdx).grid(row=row, column=1, sticky="w", padx=4, pady=2)
        self._mk_float(parent, "dy", vdy).grid(row=row, column=2, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(parent, text="sync", variable=vsync, command=self._maybe_sync_once) \
            .grid(row=row, column=3, sticky="w", padx=6)

    # ---------- bindings ----------
    def _bind_all(self) -> None:
        vars_ = (
            self.var_mix, self.var_wrap, self.var_clamp,
            self.var_use_amp, self.var_mask_key,
            self.var_dx, self.var_dy,
            self.var_dx_r, self.var_dy_r, self.var_dx_g, self.var_dy_g, self.var_dx_b, self.var_dy_b,
            self.var_sync_r, self.var_sync_g, self.var_sync_b,
        )
        for v in vars_:
            v.trace_add("write", lambda *_: self._on_any_change())
        # odśwież listę masek przy pokazaniu panelu
        self.bind("<Visibility>", lambda _e: self._refresh_masks())
        try:
            self.cmb_mask.bind("<<ComboboxSelected>>", lambda _e: self._emit())
        except Exception:
            pass

    # ---------- mask list ----------
    def _mask_source_keys(self) -> List[str]:
        try:
            f = getattr(self.ctx, "get_mask_keys", None)
            if callable(f):
                ks = list(f())
                return [k for k in ks if isinstance(k, str)]
        except Exception:
            pass
        try:
            cache = getattr(self.ctx, "cache_ref", {}) or {}
            ks = list(cache.get("cfg/masks/keys", []))
            return [k for k in ks if isinstance(k, str)]
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
    def _preset_classic(self) -> None:
        self.var_dx.set(0.0);
        self.var_dy.set(0.0)
        self.var_dx_r.set(4.0);
        self.var_dy_r.set(0.0)
        self.var_dx_g.set(0.0);
        self.var_dy_g.set(0.0)
        self.var_dx_b.set(-4.0);
        self.var_dy_b.set(0.0)
        self.var_mix.set(1.0);
        self._emit()

    def _preset_halo(self) -> None:
        self.var_dx.set(0.0);
        self.var_dy.set(0.0)
        self.var_dx_r.set(2.0);
        self.var_dy_r.set(1.0)
        self.var_dx_g.set(0.0);
        self.var_dy_g.set(0.0)
        self.var_dx_b.set(-2.0);
        self.var_dy_b.set(-1.0)
        self.var_mix.set(0.6);
        self._emit()

    def _preset_green(self) -> None:
        self.var_dx.set(0.0);
        self.var_dy.set(0.0)
        self.var_dx_r.set(0.0);
        self.var_dy_r.set(0.0)
        self.var_dx_g.set(3.0);
        self.var_dy_g.set(0.0)
        self.var_dx_b.set(0.0);
        self.var_dy_b.set(0.0)
        self.var_mix.set(0.8);
        self._emit()

    def _preset_reset(self) -> None:
        self.var_dx.set(0.0);
        self.var_dy.set(0.0)
        self.var_dx_r.set(2.0);
        self.var_dy_r.set(0.0)
        self.var_dx_g.set(0.0);
        self.var_dy_g.set(0.0)
        self.var_dx_b.set(-2.0);
        self.var_dy_b.set(0.0)
        self.var_mix.set(1.0);
        self.var_wrap.set(False);
        self.var_clamp.set(True)
        self.var_use_amp.set(1.0);
        self.var_mask_key.set(PLACEHOLDER_NONE)
        self.var_sync_r.set(False);
        self.var_sync_g.set(False);
        self.var_sync_b.set(False)
        self._emit()

    def _jitter_1px(self) -> None:
        import random
        for v in (self.var_dx_r, self.var_dy_r, self.var_dx_g, self.var_dy_g, self.var_dx_b, self.var_dy_b):
            v.set(v.get() + float(random.choice((-1, 1))))
        self._emit()

    def _copy_global_to_all(self) -> None:
        dx, dy = float(self.var_dx.get()), float(self.var_dy.get())
        self.var_dx_r.set(dx);
        self.var_dy_r.set(dy)
        self.var_dx_g.set(dx);
        self.var_dy_g.set(dy)
        self.var_dx_b.set(dx);
        self.var_dy_b.set(dy)
        self._emit()

    def _maybe_sync_once(self) -> None:
        dx, dy = float(self.var_dx.get()), float(self.var_dy.get())
        if self.var_sync_r.get(): self.var_dx_r.set(dx); self.var_dy_r.set(dy)
        if self.var_sync_g.get(): self.var_dx_g.set(dx); self.var_dy_g.set(dy)
        if self.var_sync_b.get(): self.var_dx_b.set(dx); self.var_dy_b.set(dy)
        self._emit()

    def _sync_enabled_now(self) -> None:
        self._maybe_sync_once()

    def _on_any_change(self) -> None:
        # jeżeli sync aktywny, trzymaj kanały przy globalu
        self._maybe_sync_once() if any(
            (self.var_sync_r.get(), self.var_sync_g.get(), self.var_sync_b.get())) else self._emit()

    def _emit(self) -> None:
        mk = (self.var_mask_key.get() or "").strip()
        params = {
            "mix": float(max(0.0, min(1.0, self.var_mix.get()))),
            "wrap": bool(self.var_wrap.get()),
            "clamp": bool(self.var_clamp.get()),
            "use_amp": float(max(0.0, self.var_use_amp.get())),
            "mask_key": (None if mk in ("", PLACEHOLDER_NONE) else mk),

            "dx": float(self.var_dx.get()),
            "dy": float(self.var_dy.get()),
            "dx_r": float(self.var_dx_r.get()),
            "dy_r": float(self.var_dy_r.get()),
            "dx_g": float(self.var_dx_g.get()),
            "dy_g": float(self.var_dy_g.get()),
            "dx_b": float(self.var_dx_b.get()),
            "dy_b": float(self.var_dy_b.get()),
        }
        cb = getattr(self.ctx, "on_change", None)
        if callable(cb):
            try:
                cb(params)
            except Exception:
                pass


# Loader hook (część loaderów szuka 'Panel')
Panel = RgbOffsetPanel
