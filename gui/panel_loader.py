# glitchlab/gui/panel_loader.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Panel loader (odporny) z fallbackiem do GenericFormPanel.

Publiczne API (nowe):
  - PanelLoadError
  - PanelSpec
  - list_available_panels() -> list[PanelSpec]
  - load_panel(key, parent, bus, on_apply) -> tk.Widget

Zachowana kompatybilność (stare):
  - get_panel_class(filter_name) -> Optional[type]
  - instantiate_panel(parent, filter_name, ctx=None) -> Optional[tk.Widget]

Zasady:
  • Żaden błąd importu panelu nie blokuje GUI – zawsze próbujemy fallback.
  • Fallback nie wymaga PIL ani schematów parametrów – zawsze tworzy się bez wyjątku.
"""

import importlib
import inspect
import logging
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    ttk = None  # type: ignore

logger = logging.getLogger("glitchlab.gui.panel_loader")


# ──────────────────────────────────────────────────────────────────────────────
# Typy / dataclasses
# ──────────────────────────────────────────────────────────────────────────────
class PanelLoadError(Exception):
    """Nie udało się załadować panelu oraz fallbacku."""


@dataclass
class PanelSpec:
    key: str
    title: str
    module: str
    class_name: str


# ──────────────────────────────────────────────────────────────────────────────
# Pomocnicze
# ──────────────────────────────────────────────────────────────────────────────
def _guess_panel_class_name(key: str) -> str:
    # "rgb_offset" -> "PanelRgbOffset"
    parts = [p for p in str(key).replace("-", "_").split("_") if p]
    return "Panel" + "".join(p[:1].upper() + p[1:] for p in parts)


def _is_panel_class(obj: Any) -> bool:
    """Czy to wygląda na klasę panelu Tk? (subklasa ttk.Frame)."""
    try:
        if inspect.isclass(obj):
            if ttk is not None and issubclass(obj, getattr(ttk, "Frame")):
                return True
    except Exception:
        pass
    return False


def _probe_panels_registry() -> Dict[str, Dict[str, str]]:
    """
    Próbuje odczytać rejestr paneli z glitchlab.gui.panels.__init__ (jeśli dostępny).
    Dozwolony format:
        PANELS = {
            "rgb_offset": {"title": "RGB Offset"},
            ...
        }
    Zwraca {} jeśli brak.
    """
    try:
        mod = importlib.import_module("glitchlab.gui.panels")
        meta = getattr(mod, "PANELS", None)
        if isinstance(meta, dict):
            # normalizacja wartości 'title'
            out: Dict[str, Dict[str, str]] = {}
            for k, v in meta.items():
                if isinstance(v, dict):
                    t = str(v.get("title", k))
                else:
                    t = str(v)  # dopuszczamy krótszą formę
                out[str(k)] = {"title": t}
            return out
    except Exception:
        pass
    return {}


def _discover_panel_keys_fallback() -> Dict[str, Dict[str, str]]:
    """
    Gdy brak rejestru, wykryj moduły paneli po nazwie pliku:
        glitchlab.gui.panels.panel_<key>
    """
    out: Dict[str, Dict[str, str]] = {}
    try:
        pkg = importlib.import_module("glitchlab.gui.panels")
        for m in pkgutil.iter_modules(pkg.__path__):  # type: ignore[attr-defined]
            name = m.name  # np. "panel_rgb_offset"
            if name.startswith("panel_") and len(name) > 6:
                key = name[6:]
                # prosty tytuł
                title = " ".join(part.capitalize() for part in key.replace("_", " ").split())
                out[key] = {"title": title}
    except Exception:
        pass
    return out


def list_available_panels() -> list[PanelSpec]:
    """
    Zwraca listę PanelSpec na podstawie rejestru PANELS lub autodetekcji.
    """
    reg = _probe_panels_registry()
    if not reg:
        reg = _discover_panel_keys_fallback()

    specs: list[PanelSpec] = []
    for key, meta in sorted(reg.items()):
        title = str(meta.get("title", key))
        module = f"glitchlab.gui.panels.panel_{key}"
        cls = _guess_panel_class_name(key)
        specs.append(PanelSpec(key=key, title=title, module=module, class_name=cls))
    return specs


def _import_panel_class(spec: PanelSpec) -> Optional[Type]:
    """
    Import klasy panelu wg specyfikacji. Obsługuje też starą konwencję modułu:
      glitchlab.gui.panels.<key>_panel
    Zwraca None przy błędzie.
    """
    candidates = [
        (spec.module, spec.class_name),                                   # nowa konwencja
        (f"glitchlab.gui.panels.{spec.key}_panel", spec.class_name),      # stara konwencja
    ]

    for mod_name, class_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        # 1) spróbuj klasę wg nazwy
        cand = getattr(mod, class_name, None)
        if cand and _is_panel_class(cand):
            return cand

        # 2) symbol 'Panel'
        cand = getattr(mod, "Panel", None)
        if cand and _is_panel_class(cand):
            return cand

        # 3) pierwsza klasa kończąca się na 'Panel'
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ == mod.__name__ and obj.__name__.lower().endswith("panel") and _is_panel_class(obj):
                return obj

    return None


def _construct_fallback(parent: tk.Misc, bus: Any, on_apply: Callable[[dict], None], key: str):
    """
    Zbuduj GenericFormPanel różnymi możliwymi sygnaturami (dla kompatybilności).
    """
    # Import na żądanie – nie utrzymujemy globalnej referencji
    Generic = None
    err: Optional[Exception] = None
    try:
        mod = importlib.import_module("glitchlab.gui.generic_form_panel")
        Generic = getattr(mod, "GenericFormPanel", None)
    except Exception as e:
        err = e

    if Generic is None:
        logger.error("Fallback GenericFormPanel unavailable: %s", err)
        raise PanelLoadError("GenericFormPanel not available")

    # Próbujemy kilku wariantów sygnatur:
    # 1) (parent, bus, on_apply, filter_key, param_schema=None)
    try:
        return Generic(parent, bus, on_apply, key, None)
    except Exception:
        pass

    # 2) (parent, bus=..., on_apply=..., filter_key=...)
    try:
        return Generic(parent, bus=bus, on_apply=on_apply, filter_key=key)
    except Exception:
        pass

    # 3) Minimalny: (parent,)
    try:
        return Generic(parent)
    except Exception as e:
        logger.exception("All GenericFormPanel constructor variants failed for key=%s", key)
        raise PanelLoadError(str(e))


# ──────────────────────────────────────────────────────────────────────────────
# NOWE GŁÓWNE API
# ──────────────────────────────────────────────────────────────────────────────
def load_panel(key: str, parent: tk.Misc, bus: Any, on_apply: Callable[[dict], None]) -> tk.Widget:
    """
    Ładuje panel filtra 'key' jako widżet. W razie problemów – fallback do GenericFormPanel.
    Może podnieść PanelLoadError tylko wtedy, gdy fallback też zawiedzie.
    """
    # Znajdź spec po kluczu
    spec = None
    for s in list_available_panels():
        if s.key == key:
            spec = s
            break
    if spec is None:
        logger.warning("Unknown panel key=%s; using fallback.", key)
        return _construct_fallback(parent, bus, on_apply, key)

    # Spróbuj zaimportować klasę panelu
    Cls = _import_panel_class(spec)
    if Cls is None:
        logger.exception("Panel import failed for key=%s (module=%s)", key, spec.module)
        return _construct_fallback(parent, bus, on_apply, key)

    # Zbuduj panel – wspieramy kilka wariantów konstruktorów
    try:
        try:
            # Najnowszy kontrakt
            return Cls(parent, bus=bus, on_apply=on_apply)
        except TypeError:
            # Starszy: (parent, ctx) lub (parent)
            return Cls(parent)
    except Exception:
        logger.exception("Panel construction failed for key=%s; falling back.", key)
        return _construct_fallback(parent, bus, on_apply, key)


# ──────────────────────────────────────────────────────────────────────────────
# KOMPATYBILNOŚĆ WSTECZNA
# ──────────────────────────────────────────────────────────────────────────────
def get_panel_class(filter_name: str) -> Optional[Type]:
    """
    STARE API: tylko zwraca klasę (albo None).
    Używane potencjalnie przez starsze miejsca kodu.
    """
    spec = PanelSpec(
        key=filter_name,
        title=filter_name,
        module=f"glitchlab.gui.panels.panel_{filter_name}",
        class_name=_guess_panel_class_name(filter_name),
    )
    return _import_panel_class(spec)


def instantiate_panel(parent, filter_name: str, ctx: Optional[Any] = None):
    """
    STARE API: tworzy instancję panelu lub None.
    Nie posiada bus/on_apply – przeznaczone do starszych wersji TabFilter.
    """
    Cls = get_panel_class(filter_name)
    if Cls is None:
        # bezpiecznie zwracamy None – stare ścieżki zwykle same wpadały w fallback
        return None
    try:
        try:
            # Jeżeli nowy panel akceptuje ctx – przekaż
            return Cls(parent, ctx=ctx)
        except TypeError:
            return Cls(parent)
    except Exception:
        logger.exception("instantiate_panel() failed for filter=%s", filter_name)
        return None
