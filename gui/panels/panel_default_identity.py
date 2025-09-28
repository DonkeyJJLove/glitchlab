# glitchlab/gui/panels/panel_default_identity.py
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


class DefaultIdentityPanel(ttk.Frame):
    MODES = ("identity", "gray", "edges", "edges_overlay",
             "mask_overlay", "amp_overlay", "r", "g", "b")

    def __init__(self, master: tk.Misc, ctx: Optional[PanelContext] = None, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.ctx = ctx or PanelContext(
            filter_name="default_identity", defaults={}, params={}, on_change=None, cache_ref={}
        )
        dflt: Dict[str, Any] = dict(getattr(self.ctx, "defaults", {}) or {})
        p0:   Dict[str, Any] = dict(getattr(self.ctx, "params", {}) or {})

        def _v(k: str, fb: Any) -> Any: return p0.get(k, dflt.get(k, fb))

        self.var_mode       = tk.StringVar(value=str(_v("mode", "identity")).lower())
        self.var_strength   = tk.DoubleVar(value=float(_v("strength", 1.0)))
        # ksize jako StringVar (Combobox „3/5”), konwersja do int przy emit
        self.var_edge_ksize = tk.StringVar(value=str(int(_v("edge_ksize", 3))))
        self.var_use_amp    = tk.DoubleVar(value=float(_v("use_amp", 1.0)))
        # maska – wspólny combobox
        self.var_mask_key   = tk.StringVar(value=str(_v("mask_key", "") or PLACEHOLDER_NONE))
        self.var_clamp      = tk.BooleanVar(value=bool(_v("clamp", True)))

        self._mask_keys: List[str] = []

        self._build_ui()
        self._bind_all()

        # start: wypełnij listę masek i wyemituj stan
        self._refresh_masks()
        self._emit()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8); top.pack(fill="x")
        ttk.Label(top, text="Mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(top, values=self.MODES, textvariable=self.var_mode,
                     state="readonly", width=18)\
            .grid(row=0, column=1, sticky="w", padx=6)

        # strength + clamp
        row1 = ttk.Frame(self, padding=(8, 2)); row1.pack(fill="x")
        ttk.Label(row1, text="strength").pack(side="left")
        ttk.Scale(row1, from_=0.0, to=2.0, variable=self.var_strength)\
            .pack(side="left", fill="x", expand=True, padx=6)
        ttk.Entry(row1, textvariable=self.var_strength, width=6).pack(side="left")
        ttk.Checkbutton(row1, text="clamp", variable=self.var_clamp).pack(side="left", padx=6)

        # edges params
        eg = ttk.LabelFrame(self, text="Edges", padding=8); eg.pack(fill="x", padx=8, pady=(2, 6))
        ttk.Label(eg, text="edge_ksize").grid(row=0, column=0, sticky="w")
        ttk.Combobox(eg, values=("3", "5"), textvariable=self.var_edge_ksize,
                     state="readonly", width=6)\
            .grid(row=0, column=1, sticky="w", padx=6)

        # amp & mask (wspólny wybór maski)
        am = ttk.LabelFrame(self, text="Amplitude & Mask", padding=8); am.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(am, text="use_amp").grid(row=0, column=0, sticky="w")
        ttk.Scale(am, from_=0.0, to=2.0, variable=self.var_use_amp)\
            .grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Entry(am, textvariable=self.var_use_amp, width=6).grid(row=0, column=2, sticky="w")

        ttk.Label(am, text="mask_key").grid(row=1, column=0, sticky="w", pady=(6, 0))
        row = ttk.Frame(am); row.grid(row=1, column=1, columnspan=2, sticky="ew", padx=6, pady=(6, 0))
        self.cmb_mask = ttk.Combobox(row, state="readonly", width=24, textvariable=self.var_mask_key)
        self.cmb_mask.pack(side="left", fill="x", expand=True)
        # auto-refresh przy kliknięciu w rozwijane
        self.cmb_mask.bind("<Button-1>", lambda _e: self._refresh_masks())
        ttk.Button(row, text="Refresh", command=self._refresh_masks).pack(side="left", padx=(6, 0))
        am.columnconfigure(1, weight=1)

        # quick buttons
        qb = ttk.Frame(self, padding=(8, 0)); qb.pack(fill="x")
        ttk.Button(qb, text="Edges overlay", command=lambda: self._quick("edges_overlay")).pack(side="left", padx=2)
        ttk.Button(qb, text="Mask overlay",  command=lambda: self._quick("mask_overlay")).pack(side="left", padx=2)
        ttk.Button(qb, text="Amp overlay",   command=lambda: self._quick("amp_overlay")).pack(side="left", padx=2)
        ttk.Button(qb, text="Gray",          command=lambda: self._quick("gray")).pack(side="left", padx=2)

    # ---------- binding / masks ----------
    def _bind_all(self) -> None:
        for v in (self.var_mode, self.var_strength, self.var_edge_ksize,
                  self.var_use_amp, self.var_mask_key, self.var_clamp):
            v.trace_add("write", lambda *_: self._emit())
        # odśwież listę, gdy panel staje się widoczny
        self.bind("<Visibility>", lambda _e: self._refresh_masks())
        self.cmb_mask.bind("<<ComboboxSelected>>", lambda _e: self._emit())

    def _mask_source_keys(self) -> List[str]:
        """Lista masek z Global: ctx.get_mask_keys() lub cache_ref['cfg/masks/keys']."""
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
    def _quick(self, mode: str) -> None:
        self.var_mode.set(mode)
        if mode.endswith("overlay"):
            self.var_strength.set(1.0)
        self._emit()

    def _emit(self) -> None:
        mk = self.var_mask_key.get().strip()
        params = {
            "mode":        self.var_mode.get().strip().lower(),
            "strength":    float(self.var_strength.get()),
            "edge_ksize":  int(self.var_edge_ksize.get()),
            "use_amp":     float(self.var_use_amp.get()),
            "mask_key":    (None if mk in ("", PLACEHOLDER_NONE) else mk),
            "clamp":       bool(self.var_clamp.get()),
        }
        cb = getattr(self.ctx, "on_change", None)
        if callable(cb):
            try:
                cb(params)
            except Exception:
                pass


# loader: większość loaderów szuka klasy 'Panel'
Panel = DefaultIdentityPanel
