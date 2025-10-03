"""
---
version: 2
kind: module
id: "mosaic-router"
created_at: "2025-09-13"
name: "glitchlab.gui.mosaic.router"
author: "GlitchLab v2"
role: "HUD/Mosaic Routing Resolver"
description: >
  Resolver kluczy HUD/mozaiki: dopasowuje wzorce z spec (slot1..slot3, overlay)
  do dostępnych kluczy w ctx.cache, przyznaje punktację i wybiera stabilnie
  najlepsze źródła. Wspiera preferencje/wykluczenia, historię wyborów (LRU),
  deterministyczną rozstrzygalność remisów i diagnostyczne „explain”.

inputs:
  spec:        {type: "Mapping[str, Any]", desc: "znormalizowana specyfikacja routingu"}
  keys:        {type: "Iterable[str]",     desc: "lista dostępnych kluczy (np. ctx.cache.keys())"}
  history:     {type: "Mapping[str,str]",  desc: "ostatnie wybory per slot, opcjonalnie"}
  prefer:      {type: "Sequence[str]",     desc: "dodatkowe wzorce preferowane globalnie, opcjonalnie"}
  deny:        {type: "Sequence[str]",     desc: "wzorce wykluczeń (blacklist), opcjonalnie"}

outputs:
  selection:
    slot1:     {type: "str|None"}
    slot2:     {type: "str|None"}
    slot3:     {type: "str|None"}
    overlay:   {type: "str|None"}
  ranking:
    type: "Dict[str, list[tuple[str,int]]]"
    desc: "kandydaci per slot (klucz, score) w kolejności malejącej"
  explain:
    type: "Dict[str, Any]"
    desc: "zapis decyzji (dopasowania, konflikty, tiebreak)"

interfaces:
  exports:
    - "match_patterns"
    - "score_keys"
    - "resolve_selection"
    - "apply_history_bias"
    - "explain_decision"
  depends_on: ["re","fnmatch","typing","itertools","collections"]
  used_by:
    - "glitchlab.mosaic.spec"
    - "glitchlab.gui.widgets.hud"
    - "glitchlab.gui.widgets.image_canvas"
    - "glitchlab.gui.app"

policy:
  deterministic: true
  side_effects: false
constraints:
  - "pure Python; bez I/O i frameworków"
  - "wildcards: fnmatch + kotwica na pełny klucz"
  - "stabilny tiebreak: (score desc, prefer idx, alnum sort)"

hud:
  influence:
    - "Zapewnia spójny wybór źródeł dla slotów HUD"
    - "Pozwala wymusić/wykluczyć kanały przez prefer/deny"
license: "Proprietary"
---
"""
# glitchlab/mosaic/router.py
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "match_patterns",
    "score_keys",
    "apply_history_bias",
    "resolve_selection",
    "explain_decision",
]


# -----------------------
# Helpers & data types
# -----------------------

_WILDCARDS = ("*", "?", "[", "]")


@dataclass(frozen=True)
class PatternSpec:
    pattern: str
    weight: int = 0  # optional manual weight (adds to base)
    idx: int = 0     # order in the slot list (lower = stronger)


def _normalize_patterns(seq: Sequence[Any]) -> List[PatternSpec]:
    out: List[PatternSpec] = []
    for i, it in enumerate(seq or ()):
        if isinstance(it, str):
            out.append(PatternSpec(it, 0, i))
        elif isinstance(it, Mapping):
            pat = str(it.get("pattern", ""))
            w = int(it.get("weight", 0))
            out.append(PatternSpec(pat, w, i))
        else:
            # tuple like (pattern, weight)
            try:
                pat = str(it[0])  # type: ignore[index]
                w = int(it[1])    # type: ignore[index]
                out.append(PatternSpec(pat, w, i))
            except Exception:
                continue
    return out


def _prefer_rank(key: str, prefer: Optional[Sequence[str]]) -> int:
    if not prefer:
        return 10_000  # "no preference"
    for i, p in enumerate(prefer):
        if fnmatch.fnmatchcase(key, p):
            return i
    return 10_000


def _is_exact_pattern(p: str) -> bool:
    return not any(ch in p for ch in _WILDCARDS)


def _matches_any(key: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(key, p) for p in patterns)


# -----------------------
# Public API
# -----------------------

def match_patterns(
    patterns: Sequence[str | Mapping[str, Any] | Tuple[str, int]],
    keys: Iterable[str],
    *,
    deny: Optional[Sequence[str]] = None,
) -> Dict[str, PatternSpec]:
    """
    Zwraca mapowanie: key -> najlepiej dopasowany PatternSpec (najmniejszy idx, potem większy weight).
    Klucze dopasowane przez jakikolwiek wzorzec w `deny` są wykluczane.
    """
    specs = _normalize_patterns(patterns)
    best: Dict[str, PatternSpec] = {}
    for k in keys:
        if deny and _matches_any(k, deny):
            continue
        chosen: Optional[PatternSpec] = None
        for ps in specs:
            if fnmatch.fnmatchcase(k, ps.pattern):
                if chosen is None:
                    chosen = ps
                else:
                    # prefer lower idx (earlier), then higher manual weight
                    if (ps.idx < chosen.idx) or (ps.idx == chosen.idx and ps.weight > chosen.weight):
                        chosen = ps
        if chosen:
            best[k] = chosen
    return best


def score_keys(
    slot_patterns: Sequence[str | Mapping[str, Any] | Tuple[str, int]],
    keys: Iterable[str],
    *,
    prefer: Optional[Sequence[str]] = None,
    deny: Optional[Sequence[str]] = None,
) -> List[Tuple[str, int, Dict[str, Any]]]:
    """
    Oblicza ranking (key, score, details) dla jednego slotu.
    Scoring:
      + (1000 - pattern_idx)               # im wcześniej w liście, tym wyżej
      + pattern_weight                     # ręczne ważenie
      + 120 jeśli exact pattern == key     # silna preferencja za pełne dopasowanie
      + max(0, 80 - prefer_rank*5)         # preferencje globalne (im wcześniej, tym lepiej)
    Tiebreak (deterministycznie): score desc, prefer_rank asc, key alnum asc.
    """
    matched = match_patterns(slot_patterns, keys, deny=deny)
    out: List[Tuple[str, int, Dict[str, Any]]] = []
    for k, ps in matched.items():
        base = 1000 - ps.idx
        manual = ps.weight
        exact = 120 if (_is_exact_pattern(ps.pattern) and k == ps.pattern) else 0
        pr = _prefer_rank(k, prefer)
        pref = max(0, 80 - pr * 5) if pr < 10_000 else 0
        score = base + manual + exact + pref
        out.append(
            (k, score, {
                "pattern": ps.pattern,
                "pattern_idx": ps.idx,
                "pattern_weight": ps.weight,
                "exact_bonus": exact,
                "prefer_rank": pr if pr < 10_000 else None,
                "prefer_bonus": pref,
                "base": base,
            })
        )
    # sort: score desc, prefer_rank asc (None -> large), key asc
    def _sort_key(it: Tuple[str, int, Dict[str, Any]]):
        k, sc, det = it
        pr = det.get("prefer_rank")
        prn = pr if pr is not None else 10_000
        return (-sc, prn, k)
    out.sort(key=_sort_key)
    return out


def apply_history_bias(
    ranking: List[Tuple[str, int, Dict[str, Any]]],
    last_key: Optional[str],
    *,
    bonus: int = 25,
) -> List[Tuple[str, int, Dict[str, Any]]]:
    """
    Dodaje bonus do pozycji odpowiadającej `last_key` (jeśli nadal dostępna).
    Zwraca nową, przeliczoną listę rankingową (posortowaną).
    """
    if not ranking or not last_key:
        return ranking
    biased: List[Tuple[str, int, Dict[str, Any]]] = []
    for k, sc, det in ranking:
        if k == last_key:
            sc2 = sc + bonus
            det = {**det, "history_bonus": bonus}
            biased.append((k, sc2, det))
        else:
            biased.append((k, sc, det))

    # resort deterministycznie
    def _sort_key(it: Tuple[str, int, Dict[str, Any]]):
        k, sc, det = it
        pr = det.get("prefer_rank")
        prn = pr if pr is not None else 10_000
        return (-sc, prn, k)
    biased.sort(key=_sort_key)
    return biased


def resolve_selection(
    spec: Mapping[str, Any],
    keys: Iterable[str],
    *,
    history: Optional[Mapping[str, str]] = None,
    prefer: Optional[Sequence[str]] = None,
    deny: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Optional[str]], Dict[str, List[Tuple[str, int]]], Dict[str, Any]]:
    """
    Główny resolver wyboru źródeł dla slotów HUD i overlay.
    Zwraca: (selection, ranking, explain)
      selection = {"slot1": key|None, "slot2": ..., "slot3": ..., "overlay": ...}
      ranking   = per-slot lista (key, score) po zsumowaniu biasów
      explain   = szczegóły decyzji, w tym wkład prefer/history/idx/weight
    """
    # Spec może mieć formę:
    # {"slots": {"slot1":[...],"slot2":[...],"slot3":[...]}, "overlay":[...]}
    # Lub bez "slots": bezpośrednio sloty jako klucze.
    slots_spec = spec.get("slots") if isinstance(spec, Mapping) else None
    if not isinstance(slots_spec, Mapping):
        # bezpośrednio na top-level
        slots_spec = {
            "slot1": spec.get("slot1", []),
            "slot2": spec.get("slot2", []),
            "slot3": spec.get("slot3", []),
        }
    overlay_spec = spec.get("overlay", [])

    sel: Dict[str, Optional[str]] = {"slot1": None, "slot2": None, "slot3": None, "overlay": None}
    ranking: Dict[str, List[Tuple[str, int]]] = {}
    detail: Dict[str, Any] = {"slots": {}, "overlay": {}}

    all_keys = list(keys)

    # Per slot resolution
    for slot in ("slot1", "slot2", "slot3"):
        pats = slots_spec.get(slot, []) if isinstance(slots_spec, Mapping) else []
        rnk = score_keys(pats, all_keys, prefer=prefer, deny=deny)
        if history:
            rnk = apply_history_bias(rnk, history.get(slot))
        ranking[slot] = [(k, sc) for (k, sc, _det) in rnk]
        chosen = rnk[0] if rnk else None
        sel[slot] = chosen[0] if chosen else None
        detail["slots"][slot] = {
            "candidates": [
                {"key": k, "score": sc, **det} for (k, sc, det) in rnk
            ],
            "selected": chosen[0] if chosen else None,
        }

    # Overlay (nie ma historii per se, ale można użyć prefer/deny)
    rnk_ov = score_keys(overlay_spec, all_keys, prefer=prefer, deny=deny)
    ranking["overlay"] = [(k, sc) for (k, sc, _det) in rnk_ov]
    chosen_ov = rnk_ov[0] if rnk_ov else None
    sel["overlay"] = chosen_ov[0] if chosen_ov else None
    detail["overlay"] = {
        "candidates": [{"key": k, "score": sc, **det} for (k, sc, det) in rnk_ov],
        "selected": chosen_ov[0] if chosen_ov else None,
    }

    # Dodatkowe informacje
    detail["policy"] = {
        "tiebreak": ["score desc", "prefer_rank asc", "key alnum asc"],
        "bonuses": {
            "exact": 120,
            "history": 25,
            "prefer": "max(0, 80 - 5*rank)",
        },
    }
    detail["inputs"] = {
        "prefer": list(prefer) if prefer else None,
        "deny": list(deny) if deny else None,
        "has_history": bool(history),
    }

    return sel, ranking, detail


def explain_decision(explain: Mapping[str, Any]) -> str:
    """
    Buduje zwięzły opis tekstowy decyzji resolvera (dla logów/diagnostyki).
    """
    lines: List[str] = []
    pol = explain.get("policy", {})
    lines.append("Routing policy:")
    lines.append(f"  tiebreak: {', '.join(pol.get('tiebreak', []))}")
    lines.append("  bonuses: " + ", ".join(f"{k}={v}" for k, v in pol.get("bonuses", {}).items()))
    slots = explain.get("slots", {})
    for slot in ("slot1", "slot2", "slot3"):
        s = slots.get(slot, {})
        sel = s.get("selected")
        lines.append(f"{slot}: {sel if sel else '-'}")
        cand = s.get("candidates", [])[:5]
        for i, c in enumerate(cand):
            parts = [f"#{i+1} {c.get('key')}", f"score={c.get('score')}"]
            if c.get("pattern"):
                parts.append(f"pat='{c.get('pattern')}'@{c.get('pattern_idx')}")
            if c.get("exact_bonus"):
                parts.append("exact+")
            if c.get("history_bonus"):
                parts.append("hist+")
            if c.get("prefer_rank") is not None:
                parts.append(f"pref@{c.get('prefer_rank')}")
            lines.append("    " + " ".join(parts))
    ov = explain.get("overlay", {})
    lines.append(f"overlay: {ov.get('selected') if ov.get('selected') else '-'}")
    return "\n".join(lines)
