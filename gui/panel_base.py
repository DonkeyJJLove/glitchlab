# glitchlab/gui/panels/base.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Tuple, Optional, Type
import tkinter as tk
from tkinter import ttk

__all__ = [
    "PanelContext",
    "PanelBase",
    "BasicPanel",
    "register_panel",
    "get_panel",
    "list_registered",
]

# ---------------------------------------------------------------------
# Panel registry (lekki, opcjonalny)
# Panele mogą: from .base import register_panel; register_panel("filter", PanelCls)
# a loader może: get_panel("filter") → PanelCls | None
# ---------------------------------------------------------------------

_PANEL_REG: Dict[str, Type[ttk.Frame]] = {}


def register_panel(filter_name: str, panel_cls: Type[ttk.Frame]) -> None:
    """Zarejestruj klasę panelu dla danego filtra."""
    if not isinstance(filter_name, str) or not filter_name:
        raise ValueError("register_panel: filter_name must be non-empty str")
    if not isinstance(panel_cls, type) or not issubclass(panel_cls, ttk.Frame):
        raise TypeError("register_panel: panel_cls must be a ttk.Frame subclass")
    _PANEL_REG[filter_name] = panel_cls


def get_panel(filter_name: str) -> Optional[Type[ttk.Frame]]:
    """Zwróć klasę panelu z rejestru, lub None jeśli nie ma."""
    return _PANEL_REG.get(filter_name)


def list_registered() -> List[str]:
    return sorted(_PANEL_REG.keys())


# ---------------------------------------------------------------------
# Kontekst i baza paneli (jak dotąd)
# ---------------------------------------------------------------------

@dataclass
class PanelContext:
    """
    Lekki kontekst przekazywany do paneli filtrów.

    Pola opcjonalne:
      - on_change: callable(params: dict) – panel wywołuje przy zmianie;
      - cache_ref: referencja do ctx.cache (jeśli panel potrzebuje zajrzeć do HUD);
      - get_mask_keys: callable() -> list[str] – dla dropdownów z maskami.
    """
    filter_name: str = ""
    defaults: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    on_change: Optional[Callable[[Dict[str, Any]], None]] = None
    cache_ref: Optional[Dict[str, Any]] = None
    get_mask_keys: Optional[Callable[[], List[str]]] = None

    # Pomocnicze – ułatwia panelom dostęp do listy masek
    def mask_keys(self) -> List[str]:
        if callable(self.get_mask_keys):
            try:
                return list(self.get_mask_keys())
            except Exception:
                pass
        if isinstance(self.cache_ref, dict):
            keys = self.cache_ref.get("cfg/masks/keys")
            if isinstance(keys, list):
                return list(keys)
        return []

    # Szybkie wywołanie eventu zmiany
    def emit(self, params: Optional[Dict[str, Any]] = None) -> None:
        if callable(self.on_change):
            try:
                self.on_change(dict(params or {}))
            except Exception:
                pass


class PanelBase(ttk.Frame):
    """
    Minimalna baza paneli filtrów:
      - dziedziczy po ttk.Frame,
      - śledzi zmienne (tk.Variable) podpięte przez track_var(),
      - zwraca parametry przez get_params(),
      - emituje on_change przy każdej zmianie (ctx.emit).
    """

    def __init__(self, parent: tk.Widget, ctx: PanelContext | None = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.ctx: PanelContext = ctx or PanelContext()
        # lista: (name, var, conv)
        self._tracked: List[Tuple[str, tk.Variable, Optional[Callable[[Any], Any]]]] = []
        # domyślnie wypełnij tracked wartościami z ctx.params (jeśli panel tego użyje)
        self._initial_params: Dict[str, Any] = dict(self.ctx.params or {})

    # -------- śledzenie zmiennych --------
    def track_var(
            self,
            name: str,
            var: tk.Variable,
            conv: Optional[Callable[[Any], Any]] = None,
            emit_immediately: bool = True,
    ) -> None:
        """Zarejestruj zmienną formularza. conv – opcjonalna funkcja konwersji."""
        self._tracked.append((name, var, conv))
        try:
            var.trace_add("write", lambda *_: self._on_any_change())
        except Exception:
            # starsze Tk
            var.trace("w", lambda *_: self._on_any_change())
        if emit_immediately:
            self._on_any_change()

    def _on_any_change(self) -> None:
        self.ctx.emit(self.get_params())

    # -------- parametry --------
    def get_params(self) -> Dict[str, Any]:
        """Domyślna implementacja: zbiera wartości z track_var()."""
        params: Dict[str, Any] = dict(self._initial_params)
        for name, var, conv in self._tracked:
            try:
                val = var.get()
                if conv is not None:
                    try:
                        val = conv(val)
                    except Exception:
                        pass
                params[name] = val
            except Exception:
                pass
        return params

    # -------- helpers --------
    def add_labeled_entry(
            self, parent: tk.Widget, label: str, name: str, width: int = 8,
            init: Any = "", conv: Optional[Callable[[str], Any]] = None
    ) -> ttk.Frame:
        frm = ttk.Frame(parent)
        ttk.Label(frm, text=label).pack(side="left", padx=(0, 6))
        var = tk.StringVar(value=str(init))
        ent = ttk.Entry(frm, textvariable=var, width=width)
        ent.pack(side="left", fill="x", expand=True)
        self.track_var(name, var, conv=conv)
        return frm

    def add_check(self, parent: tk.Widget, label: str, name: str, init: bool = False) -> ttk.Checkbutton:
        var = tk.BooleanVar(value=bool(init))
        chk = ttk.Checkbutton(parent, text=label, variable=var)
        self.track_var(name, var, conv=lambda v: bool(v))
        return chk

    def add_combo(
            self, parent: tk.Widget, label: str, name: str, values: List[str], init: str = ""
    ) -> ttk.Frame:
        frm = ttk.Frame(parent)
        ttk.Label(frm, text=label).pack(side="left", padx=(0, 6))
        if not values:
            values = [""]
        init_val = init if init in values else values[0]
        var = tk.StringVar(value=init_val)
        cmb = ttk.Combobox(frm, state="readonly", values=values, textvariable=var)
        cmb.pack(side="left", fill="x", expand=True)
        self.track_var(name, var, conv=lambda v: str(v))
        return frm

    def add_slider(
            self, parent: tk.Widget, label: str, name: str,
            from_: float, to: float, init: float, resolution: float = 0.01
    ) -> ttk.Frame:
        frm = ttk.Frame(parent)
        ttk.Label(frm, text=label).pack(side="left", padx=(0, 6))
        var = tk.DoubleVar(value=float(init))
        scl = ttk.Scale(frm, from_=from_, to=to, variable=var)
        scl.pack(side="left", fill="x", expand=True)
        self.track_var(name, var, conv=lambda v: float(v))
        return frm


# Alias dla starszych importów:
BasicPanel = PanelBase
