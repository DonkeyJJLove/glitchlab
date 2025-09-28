"""
---
version: 3
kind: module
id: "gui-views-notebook"
created_at: "2025-09-13"
name: "glitchlab.gui.views.notebook"
author: "GlitchLab v3"
role: "Prawy notatnik: zakładki Global / Filter / Presets /(Algorithms*)"
description: >
  Komponent zbierający podstawowe akcje użytkownika (seed, maski, amplitude,
  wybór filtra i jego parametrów, operacje na presetach). Nie wykonuje I/O ani
  obliczeń – publikuje zdarzenia na EventBus i renderuje dostarczone dane.

inputs:
  bus: {type: "EventBus-like", note: "subscribe/publish; wszystkie callbacki on_ui"}
  filters.available: {topic: "filters.available", payload: {names: list[str], select?: str}}
  masks.list:        {topic: "masks.list",        payload: {names: list[str]}}
  preset.loaded:     {topic: "preset.loaded",     payload: {text: str}}
  preset.status:     {topic: "preset.status",     payload: {message: str}}

outputs:
  ui.global.seed_changed:      {payload: {seed: int}}
  ui.masks.add_request:        {payload: {}}
  ui.masks.clear_request:      {payload: {}}
  ui.global.amplitude_changed: {payload: {kind: str}}
  ui.filter.select:            {payload: {name: str}}
  ui.filter.apply:             {payload: {name: str, params: dict}}
  ui.presets.open_request:     {payload: {}}
  ui.presets.save_request:     {payload: {text: str}}
  ui.presets.apply:            {payload: {text: str}}

interfaces:
  exports:
    - "RightNotebook(master, bus=None, show_algorithms: bool=False)"
    - "set_bus(bus)"
    - "set_seed(seed:int)"
    - "set_filter_list(names:list[str], select:str|None=None)"
    - "set_mask_names(names:list[str])"
    - "current_filter() -> str|None"
    - "current_filter_params() -> dict"
    - "set_preset_text(text:str) / get_preset_text() -> str"
    - "set_preset_status(msg:str)"

depends_on: ["tkinter/ttk","typing","glitchlab.gui.panel_loader?"]
used_by: ["glitchlab.gui.app_shell","glitchlab.gui.app (legacy)","glitchlab.gui.views.menu"]
policy:
  deterministic: true
  ui_thread_only: true
constraints:
  - "brak bezpośredniego I/O; wszystkie efekty przez EventBus"
  - "panele dedykowane ładowane dynamicznie; fallback formularza wspólnych parametrów"
license: "Proprietary"
---
"""
# glitchlab/gui/views/notebook.py
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Callable, Dict, List, Optional

try:
    # prefer the package loader if available
    from glitchlab.gui.panel_loader import get_panel_class  # type: ignore
except Exception:
    # fallback: simple generic panel
    def get_panel_class(_name: str):
        return None  # Notebook zrobi panel generowany ad-hoc


# Typ alias (unikamy twardych zależności na EventBus w celu prostych testów)
EventBusLike = Any  # obiekt z metodami: subscribe(pattern, cb, on_ui=True), publish(topic, payload)


# -----------------------------
# Global tab
# -----------------------------

class _GlobalTab(ttk.Frame):
    def __init__(self, master, bus: Optional[EventBusLike] = None):
        super().__init__(master)
        self._bus = bus

        # UI
        self.columnconfigure(1, weight=1)

        ttk.Label(self, text="Seed:").grid(row=0, column=0, sticky="w", padx=6, pady=(8, 2))
        self.var_seed = tk.IntVar(value=7)
        self.spn_seed = ttk.Spinbox(self, from_=0, to=2**31 - 1, textvariable=self.var_seed, width=12)
        self.spn_seed.grid(row=0, column=1, sticky="w", padx=6, pady=(8, 2))

        btn_row = ttk.Frame(self)
        btn_row.grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=6)
        ttk.Button(btn_row, text="Ustaw seed", command=self._emit_seed).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Dodaj maskę…", command=self._emit_add_mask).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Wyczyść maski", command=self._emit_clear_masks).pack(side="left")

        # amplitude (prosty wybór – szczegóły w serwisie)
        ttk.Separator(self, orient="horizontal").grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        ttk.Label(self, text="Amplitude preset:").grid(row=3, column=0, sticky="w", padx=6, pady=2)
        self.var_amp = tk.StringVar(value="none")
        self.cmb_amp = ttk.Combobox(self, textvariable=self.var_amp, width=18,
                                    values=["none", "linear_x", "linear_y", "radial", "perlin", "mask"])
        self.cmb_amp.grid(row=3, column=1, sticky="w", padx=6, pady=2)
        ttk.Button(self, text="Zastosuj amplitude", command=self._emit_amp).grid(row=4, column=1, sticky="w", padx=6, pady=2)

    # ---- events ----

    def _emit_seed(self) -> None:
        if self._bus:
            self._bus.publish("ui.global.seed_changed", {"seed": int(self.var_seed.get())})

    def _emit_add_mask(self) -> None:
        if self._bus:
            self._bus.publish("ui.masks.add_request", {})  # File dialog handled by MaskService

    def _emit_clear_masks(self) -> None:
        if self._bus:
            self._bus.publish("ui.masks.clear_request", {})

    def _emit_amp(self) -> None:
        if self._bus:
            self._bus.publish("ui.global.amplitude_changed", {"kind": self.var_amp.get()})

    # ---- external API ----

    def set_seed(self, seed: int) -> None:
        self.var_seed.set(int(seed))

    def set_bus(self, bus: EventBusLike) -> None:
        self._bus = bus


# -----------------------------
# Filter tab
# -----------------------------

class _FilterTab(ttk.Frame):
    """
    Zakładka wyboru filtra i edycji jego parametrów.
    - dynamiczny panel specyficzny dla filtra (jeśli dostępny),
    - fallback: formularz generowany (etykieta z informacją).
    """
    def __init__(self, master, bus: Optional[EventBusLike] = None):
        super().__init__(master)
        self._bus = bus
        self._mask_names: List[str] = []
        self._current_filter: Optional[str] = None
        self._panel_area = ttk.Frame(self)
        self._panel_area.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        # Header: filter selector + Apply
        hdr = ttk.Frame(self)
        hdr.grid(row=0, column=0, sticky="ew", padx=6, pady=(8, 2))
        hdr.columnconfigure(1, weight=1)

        ttk.Label(hdr, text="Filter:").grid(row=0, column=0, sticky="w")
        self.var_filter = tk.StringVar()
        self.cmb_filter = ttk.Combobox(hdr, textvariable=self.var_filter, state="readonly", width=40)
        self.cmb_filter.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        self.cmb_filter.bind("<<ComboboxSelected>>", self._on_filter_changed)

        ttk.Button(hdr, text="Apply", command=self._emit_apply).grid(row=0, column=2, sticky="e")

        # panel placeholder
        self._panel_widget: Optional[ttk.Frame] = None
        self._panel_values_getter: Optional[Callable[[], Dict[str, Any]]] = None

    # ---- external API ----

    def set_bus(self, bus: EventBusLike) -> None:
        self._bus = bus

    def set_filter_list(self, names: List[str], *, select: Optional[str] = None) -> None:
        self.cmb_filter["values"] = names
        if select and select in names:
            self.var_filter.set(select)
            self._mount_panel(select)
        elif names:
            self.var_filter.set(names[0])
            self._mount_panel(names[0])

    def set_mask_names(self, mask_names: List[str]) -> None:
        self._mask_names = list(mask_names)
        # Jeżeli panel ma metodę odświeżania masek – wywołaj (łagodnie)
        try:
            if self._panel_widget and hasattr(self._panel_widget, "set_mask_names"):
                getattr(self._panel_widget, "set_mask_names")(self._mask_names)
        except Exception:
            pass

    def current_filter(self) -> Optional[str]:
        return self._current_filter

    def current_params(self) -> Dict[str, Any]:
        if self._panel_values_getter:
            try:
                return dict(self._panel_values_getter())
            except Exception:
                return {}
        return {}

    # ---- internals ----

    def _on_filter_changed(self, _evt=None) -> None:
        name = self.var_filter.get()
        if name:
            self._mount_panel(name)
            if self._bus:
                self._bus.publish("ui.filter.select", {"name": name})

    def _emit_apply(self) -> None:
        if not self._bus:
            return
        name = self._current_filter
        params = self.current_params()
        if not name:
            messagebox.showwarning("GlitchLab", "Wybierz filtr.")
            return
        self._bus.publish("ui.filter.apply", {"name": name, "params": params})

    def _mount_panel(self, name: str) -> None:
        # usuwanie starego panelu
        for w in self._panel_area.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        self._panel_widget = None
        self._panel_values_getter = None
        self._current_filter = name

        # Załaduj dedykowany panel, jeśli istnieje
        PanelClass = None
        try:
            PanelClass = get_panel_class(name)
        except Exception:
            PanelClass = None

        if PanelClass is None:
            # Fallback – prosty placeholder
            frm = ttk.Frame(self._panel_area)
            ttk.Label(frm, text=f"(Brak dedykowanego panelu dla '{name}'. Użyj domyślnych parametrów.)",
                      foreground="#666").pack(anchor="w", padx=2, pady=2)

            # Minimalny formularz dla wspólnych parametrów
            common = ttk.LabelFrame(frm, text="Wspólne parametry")
            common.pack(fill="x", expand=False, padx=2, pady=(6, 2))
            self._var_mask = tk.StringVar(value="")
            ttk.Label(common, text="mask_key:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
            self._cmb_mask = ttk.Combobox(common, textvariable=self._var_mask, values=self._mask_names, width=24)
            self._cmb_mask.grid(row=0, column=1, sticky="w", padx=4, pady=2)

            self._var_use_amp = tk.DoubleVar(value=1.0)
            ttk.Label(common, text="use_amp:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
            ttk.Spinbox(common, from_=0.0, to=2.0, increment=0.1, textvariable=self._var_use_amp, width=8)\
                .grid(row=1, column=1, sticky="w", padx=4, pady=2)

            self._var_clamp = tk.BooleanVar(value=True)
            ttk.Checkbutton(common, text="clamp", variable=self._var_clamp).grid(row=2, column=1, sticky="w", padx=4, pady=2)

            def _values() -> Dict[str, Any]:
                vals = {
                    "mask_key": self._var_mask.get().strip() or None,
                    "use_amp": float(self._var_use_amp.get()),
                    "clamp": bool(self._var_clamp.get()),
                }
                return vals

            self._panel_values_getter = _values
            frm.pack(fill="both", expand=True)
            self._panel_widget = frm
            return

        # Panel dedykowany
        panel = PanelClass(self._panel_area)  # type: ignore[call-arg]
        # Jeżeli panel przewiduje mount(ctx) – podajemy minimalny kontekst
        try:
            if hasattr(panel, "mount"):
                ctx = _MinimalPanelContext(mask_names=self._mask_names)
                panel.mount(ctx)  # type: ignore[attr-defined]
        except Exception:
            pass

        # Próbujemy odczytać wartości
        def _values() -> Dict[str, Any]:
            try:
                if hasattr(panel, "values"):
                    return dict(panel.values())  # type: ignore[attr-defined]
            except Exception:
                pass
            return {}
        self._panel_values_getter = _values
        panel.pack(fill="both", expand=True)
        self._panel_widget = panel


class _MinimalPanelContext:
    """Lekki kontekst przekazywany do mount(ctx) paneli dedykowanych."""
    def __init__(self, mask_names: List[str]):
        self._mask_names = list(mask_names)

    def mask_names(self) -> List[str]:
        return list(self._mask_names)

    def emit(self, _changes: Dict[str, Any]) -> None:
        # noop – Notebook zbiera wartości na Apply
        pass


# -----------------------------
# Presets tab
# -----------------------------

class _PresetsTab(ttk.Frame):
    def __init__(self, master, bus: Optional[EventBusLike] = None):
        super().__init__(master)
        self._bus = bus

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Toolbar
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew", padx=6, pady=(8, 2))
        ttk.Button(bar, text="Otwórz…", command=self._emit_open).pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="Zapisz…", command=self._emit_save).pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="Zastosuj preset", command=self._emit_apply).pack(side="left")

        # Editor (prosty Text – YAML jako tekst, serwis zajmuje się walidacją)
        self.txt = tk.Text(self, height=20, wrap="none", undo=True)
        self.txt.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

        # status
        self._status = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._status, foreground="#666").grid(row=2, column=0, sticky="w", padx=6, pady=(0, 6))

    def set_bus(self, bus: EventBusLike) -> None:
        self._bus = bus

    def set_text(self, text: str) -> None:
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", text or "")

    def get_text(self) -> str:
        return self.txt.get("1.0", "end-1c")

    def set_status(self, msg: str) -> None:
        self._status.set(msg or "")

    # ---- events ----

    def _emit_open(self) -> None:
        if self._bus:
            self._bus.publish("ui.presets.open_request", {})

    def _emit_save(self) -> None:
        if self._bus:
            self._bus.publish("ui.presets.save_request", {"text": self.get_text()})

    def _emit_apply(self) -> None:
        if self._bus:
            self._bus.publish("ui.presets.apply", {"text": self.get_text()})


# -----------------------------
# RightNotebook (public)
# -----------------------------

class RightNotebook(ttk.Notebook):
    """
    Główna zakładka po prawej: Global / Filter / Presets / (opcjonalnie) Algorithms.
    Nie trzyma „prawdy” stanu – tylko publikuje zdarzenia i renderuje dostarczone dane.
    """
    def __init__(self, master, *, bus: Optional[EventBusLike] = None, show_algorithms: bool = False):
        super().__init__(master)
        self._bus = bus

        self._tab_global = _GlobalTab(self, bus=self._bus)
        self._tab_filter = _FilterTab(self, bus=self._bus)
        self._tab_presets = _PresetsTab(self, bus=self._bus)

        self.add(self._tab_global, text="Global")
        self.add(self._tab_filter, text="Filter")
        self.add(self._tab_presets, text="Presets")

        if show_algorithms:
            frm_alg = ttk.Frame(self)
            ttk.Label(frm_alg, text="Algorithms — w przygotowaniu").pack(anchor="w", padx=8, pady=8)
            self.add(frm_alg, text="Algorithms")

        # subskrypcje bus (opcjonalnie)
        if self._bus:
            try:
                self._bus.subscribe("preset.loaded", self._on_preset_loaded, on_ui=True)
                self._bus.subscribe("preset.status", self._on_preset_status, on_ui=True)
                self._bus.subscribe("masks.list", self._on_masks_list, on_ui=True)
                self._bus.subscribe("filters.available", self._on_filters_available, on_ui=True)
            except Exception:
                pass

    # ---- wiring ----

    def set_bus(self, bus: EventBusLike) -> None:
        self._bus = bus
        self._tab_global.set_bus(bus)
        self._tab_filter.set_bus(bus)
        self._tab_presets.set_bus(bus)

    # ---- Global API passthrough ----

    def set_seed(self, seed: int) -> None:
        self._tab_global.set_seed(seed)

    # ---- Filter API ----

    def set_filter_list(self, names: List[str], *, select: Optional[str] = None) -> None:
        self._tab_filter.set_filter_list(names, select=select)

    def set_mask_names(self, names: List[str]) -> None:
        self._tab_filter.set_mask_names(names)

    def current_filter(self) -> Optional[str]:
        return self._tab_filter.current_filter()

    def current_filter_params(self) -> Dict[str, Any]:
        return self._tab_filter.current_params()

    # ---- Presets API ----

    def set_preset_text(self, text: str) -> None:
        self._tab_presets.set_text(text)

    def get_preset_text(self) -> str:
        return self._tab_presets.get_text()

    def set_preset_status(self, msg: str) -> None:
        self._tab_presets.set_status(msg)

    # ---- Bus handlers ----

    def _on_preset_loaded(self, _topic: str, data: Dict[str, Any]) -> None:
        txt = data.get("text")
        if isinstance(txt, str):
            self._tab_presets.set_text(txt)

    def _on_preset_status(self, _topic: str, data: Dict[str, Any]) -> None:
        msg = data.get("message") or ""
        self._tab_presets.set_status(str(msg))

    def _on_masks_list(self, _topic: str, data: Dict[str, Any]) -> None:
        names = data.get("names") or []
        if isinstance(names, list):
            self._tab_filter.set_mask_names([str(n) for n in names])

    def _on_filters_available(self, _topic: str, data: Dict[str, Any]) -> None:
        names = data.get("names") or []
        pick = data.get("select")
        if isinstance(names, list):
            self._tab_filter.set_filter_list([str(n) for n in names], select=str(pick) if pick else None)
