# glitchlab/core/registry.py
# -*- coding: utf-8 -*-
"""
Lekki rejestr filtrów:
- @register("nazwa") – dekorator do rejestracji filtrów
- get(name)         – pobierz funkcję filtra (uwzględnia aliasy)
- available()       – listuje wszystkie nazwy (łącznie z aliasami)
- canonical(name)   – rozwija alias do nazwy bazowej
- meta(name)        – metadane filtra
- register_alias(dst, src) – rejestracja aliasu (druga nazwa tego samego filtra)

Filtr ma sygnaturę:  f(img: np.ndarray, ctx: Ctx, **params) -> np.ndarray
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any

@dataclass
class _Entry:
    fn: Callable
    module: str
    alias_of: Optional[str] = None
    defaults: Dict[str, Any] = None
    doc: str = ""

_REGISTRY: Dict[str, _Entry] = {}

def _extract_defaults(fn: Callable) -> Dict[str, Any]:
    import inspect
    sig = getattr(fn, "__signature__", None) or inspect.signature(fn)
    out: Dict[str, Any] = {}
    params = list(sig.parameters.values())
    for p in params[2:]:  # pomiń img, ctx
        if p.default is not inspect._empty:
            out[p.name] = p.default
    return out

def register(name: Optional[str] = None):
    """Dekorator rejestrujący filtr pod wskazaną nazwą."""
    def deco(fn: Callable):
        nonlocal name
        if name is None:
            name = fn.__name__
        _REGISTRY[name] = _Entry(
            fn=fn,
            module=getattr(fn, "__module__", ""),
            alias_of=None,
            defaults=_extract_defaults(fn),
            doc=(fn.__doc__ or "").strip(),
        )
        return fn
    return deco

def register_alias(dst: str, src: str) -> None:
    """Zarejestruj alias 'dst' wskazujący na istniejący filtr 'src'."""
    base = _REGISTRY.get(src)
    if base is None:
        return
    _REGISTRY[dst] = _Entry(
        fn=base.fn, module=base.module, alias_of=src,
        defaults=base.defaults, doc=base.doc
    )

def get(name: str) -> Callable:
    ent = _REGISTRY.get(name)
    if ent is None:
        raise KeyError(f"Unknown filter '{name}'")
    return ent.fn

def canonical(name: str) -> str:
    seen = set()
    cur = name
    while True:
        if cur in seen:
            return cur
        seen.add(cur)
        ent = _REGISTRY.get(cur)
        if not ent or not ent.alias_of:
            return cur
        cur = ent.alias_of

def available() -> list[str]:
    return sorted(_REGISTRY.keys(), key=str.casefold)

def meta(name: str) -> Dict[str, Any]:
    ent = _REGISTRY.get(name)
    if ent is None:
        raise KeyError(name)
    return {
        "module": ent.module,
        "alias_of": ent.alias_of,
        "defaults": ent.defaults,
        "doc": ent.doc,
    }

# Debug (opcjonalnie)
def _dump_registry() -> Dict[str, Dict[str, Any]]:
    return {k: {"module": v.module, "alias_of": v.alias_of} for k, v in _REGISTRY.items()}
