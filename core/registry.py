# glitchlab/core/registry.py
# -*- coding: utf-8 -*-
"""
Rejestr filtrów dla glitchlab.

Użycie:
    from glitchlab.core.registry import register, get, available, meta

    @register("rgb_offset", aliases=["chroma_split"])
    def rgb_offset(img, ctx, r=(5,0), g=(0,0), b=(-5,0)):
        ...

Kontrakt filtra:
    - sygnatura: fn(img, ctx, **params)
    - zwraca: numpy.ndarray (H,W,3) uint8 (GUI/pipeline rzutuje defensywnie)
"""

from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional


@dataclass
class FilterMeta:
    name: str
    fn: Callable[..., Any]
    module: str
    doc: str
    defaults: Dict[str, Any]
    schema: Optional[Dict[str, Any]] = None
    alias_of: Optional[str] = None  # jeśli to alias → wskazuje nazwę oryginału


_registry: Dict[str, FilterMeta] = {}
_lock = threading.RLock()


def _defaults_from_signature(sig: inspect.Signature) -> Dict[str, Any]:
    """Wyciąga domyślne wartości parametrów z sygnatury filtra (po img, ctx)."""
    out: Dict[str, Any] = {}
    params = list(sig.parameters.values())
    # pomiń pierwsze dwa: (img, ctx)
    for p in params[2:]:
        if p.default is not inspect._empty:
            out[p.name] = p.default
    return out


def register(name: str, aliases: Optional[List[str]] = None, schema: Optional[Dict[str, Any]] = None):
    """
    Dekorator rejestrujący filtr pod nazwą `name`.
    - `aliases`: lista aliasów wskazujących na ten sam obiekt funkcji.
    - `schema`: opcjonalny deskryptor parametrów (np. zakresy do GUI).

    Przykład:
        @register("wave_distort", aliases=["wave"])
        def wave_distort(img, ctx, axis="x", amplitude=12, frequency=0.05):
            ...
    """
    if not isinstance(name, str) or not name:
        raise ValueError("register(): name must be non-empty string")
    aliases = aliases or []

    def _decorator(fn: Callable[..., Any]):
        # Walidacja sygnatury
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if len(params) < 2:
            raise TypeError(
                f"Filter '{name}' must accept at least two positional parameters: (img, ctx, ...)"
            )
        # opcjonalnie stricte nazwy pierwszych argumentów:
        # if params[0].name != "img" or params[1].name != "ctx": ...

        meta = FilterMeta(
            name=name,
            fn=fn,
            module=getattr(fn, "__module__", "<unknown>"),
            doc=getattr(fn, "__doc__", "") or "",
            defaults=_defaults_from_signature(sig),
            schema=schema,
            alias_of=None,
        )

        with _lock:
            if name in _registry:
                raise KeyError(f"Filter '{name}' already registered (module={_registry[name].module})")
            _registry[name] = meta

            # Aliasowanie
            for alias in aliases:
                if not isinstance(alias, str) or not alias:
                    raise ValueError("Alias must be non-empty string")
                if alias in _registry:
                    raise KeyError(f"Alias '{alias}' already registered")
                _registry[alias] = FilterMeta(
                    name=alias,
                    fn=fn,
                    module=meta.module,
                    doc=meta.doc,
                    defaults=meta.defaults.copy(),
                    schema=meta.schema,
                    alias_of=name,
                )
        return fn

    return _decorator


def get(name: str) -> Callable[..., Any]:
    """Zwraca funkcję filtra (lub podnosi KeyError z czytelnym komunikatem)."""
    if name not in _registry:
        # podpowiedz najbliższe nazwy (prosta heurystyka)
        suggestions = [n for n in _registry.keys() if n.startswith(name[:3])]
        hint = f" Did you mean: {', '.join(sorted(suggestions)[:5])}?" if suggestions else ""
        raise KeyError(f"Unknown filter '{name}'.{hint}")
    return _registry[name].fn


def available() -> List[str]:
    """Lista zarejestrowanych nazw (włącznie z aliasami), posortowana alfabetycznie."""
    return sorted(_registry.keys())


def canonical(name: str) -> str:
    """Zwraca nazwę kanoniczną (jeśli to alias, zwróci nazwę oryginału)."""
    if name not in _registry:
        raise KeyError(f"Unknown filter '{name}'")
    meta = _registry[name]
    return meta.alias_of or meta.name


def meta(name: str) -> Dict[str, Any]:
    """Metadane filtra (doc, defaults, module, schema, alias_of)."""
    if name not in _registry:
        raise KeyError(f"Unknown filter '{name}'")
    return asdict(_registry[name])


def describe(name: str) -> str:
    """Ładny opis filtra dla debug/CLI/GUI."""
    m = meta(name)
    lines = [
        f"[{m['name']}]  module={m['module']}",
        f"alias_of={m['alias_of']}",
        "defaults=" + (", ".join(f"{k}={v!r}" for k, v in (m['defaults'] or {}).items()) or "—"),
        "schema=" + (", ".join(f"{k}" for k in (m['schema'] or {}).keys()) or "—"),
        "",
        (m["doc"] or "").strip(),
    ]
    return "\n".join(lines)
