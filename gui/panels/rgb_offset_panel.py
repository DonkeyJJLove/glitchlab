# glitchlab/gui/panels/rgb_offset_panel.py
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional

try:
    # preferowana ścieżka: PanelContext z Twojej bazy
    from glitchlab.gui.panel_base import PanelContext  # type: ignore
except Exception:  # pragma: no cover
    class PanelContext:  # minimalny fallback dla type-checków
        def __init__(self, **kw): self.__dict__.update(kw)

# Próba rejestracji w centralnym loaderze paneli (jeśli jest)
try:  # pragma: no cover
    from glitchlab.gui.panels.base import register_panel  # type: ignore
except Exception:  # pragma: no cover
    register_panel = None  # zostawiamy loaderowi get_panel_class() inne heurystyki

PANEL_FILTER = "rgb_offset"  # nazwa filtra dla tego panelu


class RgbOffsetPanel(ttk.Frame):
    """
    Zaawansowany panel do filtra 'rgb_offset':
    - globalne dx/dy,
    - per-channel dx/dy (R/G/B) z opcją synchronizacji do globalnego,
    - blend (mix), mode (wrap/black),
    - amplitude (use_amp), mask_key, clamp,
    - presety i drobne narzędzia (±1 jitter, reset).
    """

    def __init__(self, master: tk.Misc, ctx: Optional[PanelContext] = None, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.ctx = ctx or PanelContext(filter_name=PANEL_FILTER, defaults={}, params={}, on_change=None, cache_ref={})

        # --- Vars (tk) ---
        dflt: Dict[str, Any] = dict(self.ctx.defaults or {})
        params0: Dict[str, Any] = dict(self.ctx.params or {})

        # podstawowe wartości (params ⟵ defaults)
        def _v(key: str, fallback: Any) -> Any:
            return params0.get(key, dflt.get(key, fallback))

        # Blend / mode
        self.var_mix = tk.DoubleVar(value=float(_v("mix", 1.0)))
        self.var_mode = tk.StringVar(value=str(_v("mode", "wrap")).lower())
        self.var_clamp = tk.BooleanVar(value=bool(_v("clamp", True)))

        # Amp / mask
        self.var_use_amp = tk.DoubleVar(value=float(_v("use_amp", 1.0)))  # float|bool — traktujemy jako float
        self.var_mask_key = tk.StringVar(value=str(_v("mask_key", "") or ""))

        # Global offsets
        self.var_dx = tk.IntVar(value=int(_v("dx", 0)))
        self.var_dy = tk.IntVar(value=int(_v("dy", 0)))

        # Per-channel offsets (absolutne; nie delty)
        self.var_dx_r = tk.IntVar(value=int(_v("dx_r", 2)))
        self.var_dy_r = tk.IntVar(value=int(_v("dy_r", 0)))
        self.var_dx_g = tk.IntVar(value=int(_v("dx_g", 0)))
        self.var_dy_g = tk.IntVar(value=int(_v("dy_g", 0)))
        self.var_dx_b = tk.IntVar(value=int(_v("dx_b", -2)))
        self.var_dy_b = tk.IntVar(value=int(_v("dy_b", 0)))

        # Sync kanałów do globalnego?
        self.var_sync_r = tk.BooleanVar(value=False)
        self.var_sync_g = tk.BooleanVar(value=False)
        self.var_sync_b = tk.BooleanVar(value=False)

        self._build_ui()
        self._bind_var_changes()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        # Presety
        pf = ttk.LabelFrame(self, text="Presets", padding=8)
        pf.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Button(pf, text="Classic split (R↔B)", command=self._preset_classic).pack(side="left", padx=2)
        ttk.Button(pf, text="Halo (soft)", command=self._preset_halo).pack(side="left", padx=2)
        ttk.Button(pf, text="G→Y offset", command=self._preset_green).pack(side="left", padx=2)
        ttk.Button(pf, text="Reset", command=self._preset_reset).pack(side="right", padx=2)

        # Blend / Mode
        bf = ttk.LabelFrame(self, text="Blend & Mode", padding=8)
        bf.pack(fill="x", padx=8, pady=4)
        self._mk_labeled_scale(bf, "mix", self.var_mix, 0.0, 1.0, 0.01).grid(row=0, column=0, sticky="ew", padx=4, pady=2)
        ttk.Label(bf, text="mode").grid(row=0, column=1, sticky="w")
        ttk.Combobox(bf, textvariable=self.var_mode, values=("wrap", "black"), state="readonly", width=8)\
            .grid(row=0, column=2, sticky="w", padx=4)
        ttk.Checkbutton(bf, text="clamp", variable=self.var_clamp).grid(row=0, column=3, sticky="w", padx=4)
        bf.columnconfigure(0, weight=1)

        # Amplitude / Mask
        af = ttk.LabelFrame(self, text="Amplitude & Mask", padding=8)
        af.pack(fill="x", padx=8, pady=4)
        self._mk_labeled_scale(af, "use_amp", self.var_use_amp, 0.0, 2.0, 0.05).grid(row=0, column=0, sticky="ew", padx=4, pady=2)
        ttk.Label(af, text="mask_key").grid(row=1, column=0, sticky="w", padx=4)
        mk = ttk.Entry(af, textvariable=self.var_mask_key, width=24)
        mk.grid(row=1, column=0, sticky="ew", padx=4, pady=2)
        af.columnconfigure(0, weight=1)

        # Global offset
        gf = ttk.LabelFrame(self, text="Global Offset (px)", padding=8)
        gf.pack(fill="x", padx=8, pady=4)
        self._mk_spin(gf, "dx", self.var_dx).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self._mk_spin(gf, "dy", self.var_dy).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Button(gf, text="Copy → R/G/B", command=self._copy_global_to_all).grid(row=0, column=2, sticky="w", padx=8)

        # Per-channel offsets
        cf = ttk.LabelFrame(self, text="Per-channel Offsets (px)", padding=8)
        cf.pack(fill="x", padx=8, pady=(4, 8))

        # R
        self._mk_channel_row(cf, "R", self.var_dx_r, self.var_dy_r, self.var_sync_r, row=0)
        # G
        self._mk_channel_row(cf, "G", self.var_dx_g, self.var_dy_g, self.var_sync_g, row=1)
        # B
        self._mk_channel_row(cf, "B", self.var_dx_b, self.var_dy_b, self.var_sync_b, row=2)

        # Tools
        tf = ttk.Frame(cf)
        tf.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Button(tf, text="±1 jitter", command=self._jitter_1px).pack(side="left", padx=4)
        ttk.Button(tf, text="Sync enabled with Global", command=self._sync_enabled_now).pack(side="left", padx=4)

    def _mk_labeled_scale(self, parent: tk.Misc, label: str, var: tk.DoubleVar,
                          vmin: float, vmax: float, step: float) -> ttk.Frame:
        fr = ttk.Frame(parent)
        ttk.Label(fr, text=label).pack(side="left")
        s = ttk.Scale(fr, from_=vmin, to=vmax, variable=var)
        s.pack(side="left", fill="x", expand=True, padx=6)
        ent = ttk.Entry(fr, textvariable=var, width=6)
        ent.pack(side="left")
        return fr

    def _mk_spin(self, parent: tk.Misc, label: str, var: tk.IntVar, rng: tuple[int, int] = (-64, 64)) -> ttk.Frame:
        fr = ttk.Frame(parent)
        ttk.Label(fr, text=label).pack(side="left")
        sp = ttk.Spinbox(fr, from_=rng[0], to=rng[1], textvariable=var, width=6)
        sp.pack(side="left")
        return fr

    def _mk_channel_row(self, parent: tk.Misc, name: str, vdx: tk.IntVar, vdy: tk.IntVar,
                        vsync: tk.BooleanVar, row: int) -> None:
        col0 = ttk.Label(parent, text=f"{name}:")
        col0.grid(row=row, column=0, sticky="w")
        wdx = self._mk_spin(parent, "dx", vdx); wdx.grid(row=row, column=1, sticky="w", padx=4, pady=2)
        wdy = self._mk_spin(parent, "dy", vdy); wdy.grid(row=row, column=2, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(parent, text="sync", variable=vsync, command=self._maybe_sync_once)\
            .grid(row=row, column=3, sticky="w", padx=4)

    # ---------- Presety ----------
    def _preset_classic(self) -> None:
        # R→ +4, B→ -4, G 0; mix 1.0
        self.var_dx.set(0); self.var_dy.set(0)
        self.var_dx_r.set(4); self.var_dy_r.set(0)
        self.var_dx_g.set(0); self.var_dy_g.set(0)
        self.var_dx_b.set(-4); self.var_dy_b.set(0)
        self.var_mix.set(1.0)
        self._emit()

    def _preset_halo(self) -> None:
        # delikatne, symetryczne
        self.var_dx.set(0); self.var_dy.set(0)
        self.var_dx_r.set(2); self.var_dy_r.set(1)
        self.var_dx_g.set(0); self.var_dy_g.set(0)
        self.var_dx_b.set(-2); self.var_dy_b.set(-1)
        self.var_mix.set(0.6)
        self._emit()

    def _preset_green(self) -> None:
        self.var_dx.set(0); self.var_dy.set(0)
        self.var_dx_r.set(0); self.var_dy_r.set(0)
        self.var_dx_g.set(3); self.var_dy_g.set(0)
        self.var_dx_b.set(0); self.var_dy_b.set(0)
        self.var_mix.set(0.8)
        self._emit()

    def _preset_reset(self) -> None:
        # zgodne z DEFAULTS w filtrze
        self.var_dx.set(0); self.var_dy.set(0)
        self.var_dx_r.set(2); self.var_dy_r.set(0)
        self.var_dx_g.set(0); self.var_dy_g.set(0)
        self.var_dx_b.set(-2); self.var_dy_b.set(0)
        self.var_mix.set(1.0)
        self.var_mode.set("wrap")
        self.var_use_amp.set(1.0)
        self.var_mask_key.set("")
        self.var_clamp.set(True)
        self.var_sync_r.set(False); self.var_sync_g.set(False); self.var_sync_b.set(False)
        self._emit()

    # ---------- Tools ----------
    def _jitter_1px(self) -> None:
        import random
        for v in (self.var_dx_r, self.var_dy_r, self.var_dx_g, self.var_dy_g, self.var_dx_b, self.var_dy_b):
            v.set(v.get() + random.choice((-1, 1)))
        self._emit()

    def _copy_global_to_all(self) -> None:
        dx, dy = self.var_dx.get(), self.var_dy.get()
        self.var_dx_r.set(dx); self.var_dy_r.set(dy)
        self.var_dx_g.set(dx); self.var_dy_g.set(dy)
        self.var_dx_b.set(dx); self.var_dy_b.set(dy)
        self._emit()

    def _maybe_sync_once(self) -> None:
        # jeśli kanał ma zaznaczone "sync", ustaw jego wartości = global
        dx, dy = self.var_dx.get(), self.var_dy.get()
        if self.var_sync_r.get():
            self.var_dx_r.set(dx); self.var_dy_r.set(dy)
        if self.var_sync_g.get():
            self.var_dx_g.set(dx); self.var_dy_g.set(dy)
        if self.var_sync_b.get():
            self.var_dx_b.set(dx); self.var_dy_b.set(dy)
        self._emit()

    def _sync_enabled_now(self) -> None:
        # jednorazowo: zastosuj global do wszystkich z aktywnym "sync"
        self._maybe_sync_once()

    # ---------- Emisja parametrów ----------
    def _bind_var_changes(self) -> None:
        # proste bindingi (debounce można dodać później)
        vars_ = (
            self.var_mix, self.var_mode, self.var_clamp,
            self.var_use_amp, self.var_mask_key,
            self.var_dx, self.var_dy,
            self.var_dx_r, self.var_dy_r, self.var_dx_g, self.var_dy_g, self.var_dx_b, self.var_dy_b,
            self.var_sync_r, self.var_sync_g, self.var_sync_b,
        )
        for v in vars_:
            v.trace_add("write", lambda *_: self._on_any_change())

    def _on_any_change(self) -> None:
        # jeśli sync aktywny – ściągaj per-channel do globalu
        dx, dy = self.var_dx.get(), self.var_dy.get()
        if self.var_sync_r.get():
            self.var_dx_r.set(dx); self.var_dy_r.set(dy)
        if self.var_sync_g.get():
            self.var_dx_g.set(dx); self.var_dy_g.set(dy)
        if self.var_sync_b.get():
            self.var_dx_b.set(dx); self.var_dy_b.set(dy)
        self._emit()

    def _emit(self) -> None:
        params = {
            "mix": float(self.var_mix.get()),
            "mode": self.var_mode.get().strip().lower(),
            "clamp": bool(self.var_clamp.get()),
            "use_amp": float(self.var_use_amp.get()),
            "mask_key": self._mk_mask_key(self.var_mask_key.get()),

            "dx": int(self.var_dx.get()),
            "dy": int(self.var_dy.get()),
            "dx_r": int(self.var_dx_r.get()),
            "dy_r": int(self.var_dy_r.get()),
            "dx_g": int(self.var_dx_g.get()),
            "dy_g": int(self.var_dy_g.get()),
            "dx_b": int(self.var_dx_b.get()),
            "dy_b": int(self.var_dy_b.get()),
        }
        # powiadom aplikację
        if callable(getattr(self.ctx, "on_change", None)):
            try:
                self.ctx.on_change(params)  # App przejmie i zaktualizuje active_params
            except Exception:
                pass

    @staticmethod
    def _mk_mask_key(s: str) -> Optional[str]:
        s = (s or "").strip()
        return s if s else None

    # --- API opcjonalne, wspierane przez App ---
    def load_defaults(self, defaults: Dict[str, Any]) -> None:
        # łagodnie ustaw (nie nadpisuj jawnie ustawionych już pól)
        if "mix" in defaults: self.var_mix.set(float(defaults["mix"]))
        if "mode" in defaults: self.var_mode.set(str(defaults["mode"]).lower())
        if "clamp" in defaults: self.var_clamp.set(bool(defaults["clamp"]))
        if "use_amp" in defaults: self.var_use_amp.set(float(defaults["use_amp"]))
        if "mask_key" in defaults: self.var_mask_key.set(str(defaults["mask_key"] or ""))

        if "dx" in defaults: self.var_dx.set(int(defaults["dx"]))
        if "dy" in defaults: self.var_dy.set(int(defaults["dy"]))
        for k, v in (("dx_r", self.var_dx_r), ("dy_r", self.var_dy_r),
                     ("dx_g", self.var_dx_g), ("dy_g", self.var_dy_g),
                     ("dx_b", self.var_dx_b), ("dy_b", self.var_dy_b)):
            if k in defaults:
                v.set(int(defaults[k]))
        self._emit()


# Rejestracja panelu w centralnym rejestrze paneli (jeśli dostępny)
if register_panel is not None:  # pragma: no cover
    try:
        register_panel(PANEL_FILTER, RgbOffsetPanel)
    except Exception:
        pass

# Część loaderów szuka klasy o nazwie 'Panel' w module:
Panel = RgbOffsetPanel
