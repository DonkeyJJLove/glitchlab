# glitchlab/delta/features.py
# -*- coding: utf-8 -*-
"""
Cechy (features) oparte o histogram Δ-tokenów (VOCAB v1).

Założenia:
- Wejściem jest histogram z kluczami zgodnymi z VOCAB v1 (zob. delta.tokens.Vocabulary),
  tj. m.in. „T:<TYPE>”, „PATH:<top>”, „LANG:<py|other>”, „KIND:<fn|class>”.
- Liczba tokenów = suma wartości tylko dla kluczy „T:*” (bo 1 token → wiele kubełków).
- Brak stanów globalnych. Zwracane wartości są liczbami zmiennoprzecinkowymi (float).
- Brak NaN/Inf: wszystkie dzielenia są „bezpieczne”.

Publiczne API:
- FEATURES_VERSION = "v1"
- build_features(hist: Dict[str, int]) -> Dict[str, float]
- features_from_tokens(tokens: List[Token]) -> Dict[str, float]  # zgodność wsteczna

Python: 3.9
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .tokens import Token, tokens_to_hist, Vocabulary

FEATURES_VERSION = "v1"


# ──────────────────────────────────────────────────────────────────────────────
# Pomocnicze
# ──────────────────────────────────────────────────────────────────────────────

def _safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0

def _entropy_from_counts(counts: List[float]) -> float:
    """Shannon H w log2, dla rozkładu typów tokenów."""
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0.0:
            p = c / total
            h -= p * math.log(p, 2.0)
    return h

def _gini_from_counts(counts: List[float]) -> float:
    """Gini impurity = 1 - sum(p_i^2) dla rozkładu typów tokenów."""
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    s2 = 0.0
    for c in counts:
        if c > 0.0:
            p = c / total
            s2 += p * p
    return 1.0 - s2


# ──────────────────────────────────────────────────────────────────────────────
# Cechy z histogramu VOCAB v1
# ──────────────────────────────────────────────────────────────────────────────

def build_features(hist: Dict[str, int]) -> Dict[str, float]:
    """
    Buduje wektor cech z histogramu VOCAB v1 (delta.tokens.tokens_to_hist).

    Zwracane m.in.:
      - tokens_total, distinct_types
      - shares: add/del/rename/modify/imports
      - file_ops, density (tokens_total / max(1, file_ops)), churn (+ per_file)
      - entropia i gini dla rozkładu typów T:
      - top_type_share (dominanta typów)
      - path_share.* (app/core/analysis/mosaic/delta/docs/tests/bench/io/spec/other)
      - lang_share.py / .other
      - kind_share.fn / .class
    """
    # 1) rozdzielenie kubełków
    t_counts: Dict[str, int] = {}     # typ tokena: COUNT  (z kluczy "T:*")
    path_counts: Dict[str, int] = {}  # "PATH:*" → COUNT
    lang_counts: Dict[str, int] = {}  # "LANG:*" → COUNT
    kind_counts: Dict[str, int] = {}  # "KIND:*" → COUNT

    for k, v in hist.items():
        if k.startswith("T:"):
            t_counts[k[2:]] = t_counts.get(k[2:], 0) + int(v)
        elif k.startswith("PATH:"):
            path_counts[k[5:]] = path_counts.get(k[5:], 0) + int(v)
        elif k.startswith("LANG:"):
            lang_counts[k[5:]] = lang_counts.get(k[5:], 0) + int(v)
        elif k.startswith("KIND:"):
            kind_counts[k[5:]] = kind_counts.get(k[5:], 0) + int(v)

    tokens_total = float(sum(t_counts.values()))
    distinct_types = float(len(t_counts))

    # 2) agregaty semantyczne po typach T:
    def _sum_types(pred) -> int:
        s = 0
        for t, c in t_counts.items():
            if pred(t):
                s += int(c)
        return s

    adds = _sum_types(lambda t: t.startswith("ADD_") or t == "ΔIMPORT+")
    dels = _sum_types(lambda t: t.startswith("DEL_") or t == "ΔIMPORT-")
    renames = _sum_types(lambda t: t.startswith("RENAME_"))
    mod_sig = t_counts.get("MODIFY_SIG_FN", 0)
    mod_body = t_counts.get("MODIFY_BODY_FN", 0) + t_counts.get("MODIFY_BODY_CLASS", 0)
    imports = _sum_types(lambda t: t.startswith("ΔIMPORT"))
    file_ops = (
        t_counts.get("ADD_FILE", 0)
        + t_counts.get("DEL_FILE", 0)
        + t_counts.get("MOVE_FILE", 0)
        + t_counts.get("MOD_FILE", 0)
    )

    modify = mod_sig + mod_body + t_counts.get("MOD_FILE", 0)
    churn = adds + dels + renames + modify

    # 3) miary rozkładu typów
    type_counts_list = [float(c) for c in t_counts.values()]
    type_entropy = _entropy_from_counts(type_counts_list)
    type_gini = _gini_from_counts(type_counts_list)
    top_type_share = _safe_div(float(max(type_counts_list) if type_counts_list else 0.0), tokens_total)

    # 4) udziały PATH/LANG/KIND (te kubełki są 1:1 z tokenami – PATH i LANG,
    #    KIND może być < tokens_total)
    def _share_in(d: Dict[str, int], key: str) -> float:
        return _safe_div(float(d.get(key, 0)), float(sum(d.values())))

    # PATH shares — zawsze w oparciu o sumę path_counts (czytelniej niż tokens_total)
    path_total = float(sum(path_counts.values()))
    path_share = {p: _safe_div(float(path_counts.get(p, 0)), path_total) for p in (
        "app", "core", "analysis", "mosaic", "delta", "docs", "tests", "bench", "io", "spec", "other"
    )}

    # LANG shares
    lang_total = float(sum(lang_counts.values()))
    lang_share_py = _safe_div(float(lang_counts.get("py", 0)), lang_total)
    lang_share_other = _safe_div(float(lang_counts.get("other", 0)), lang_total)

    # KIND shares (odnosimy do tokens_total, bo KIND nie zawsze jest obecny)
    kind_share_fn = _safe_div(float(kind_counts.get("fn", 0)), tokens_total)
    kind_share_class = _safe_div(float(kind_counts.get("class", 0)), tokens_total)

    # 5) gęstość i churn per file
    density = _safe_div(tokens_total, float(max(1, file_ops)))
    churn_per_file = _safe_div(float(churn), float(max(1, file_ops)))

    # 6) miary normalizowane (udziały)
    share_add = _safe_div(float(adds), tokens_total)
    share_del = _safe_div(float(dels), tokens_total)
    share_rename = _safe_div(float(renames), tokens_total)
    share_modify = _safe_div(float(modify), tokens_total)
    share_imports = _safe_div(float(imports), tokens_total)

    # 7) Złożenie wyniku — tylko floaty, bez NaN/Inf
    out: Dict[str, float] = {
        # rozmiary
        "tokens_total": float(tokens_total),
        "distinct_types": float(distinct_types),

        # udziały kategorii zmian
        "share_add": share_add,
        "share_del": share_del,
        "share_rename": share_rename,
        "share_modify": share_modify,
        "share_imports": share_imports,

        # operacje plikowe i gęstość
        "file_ops": float(file_ops),
        "density": density,           # tokens_total / file_ops
        "churn": float(churn),        # adds + dels + renames + modify
        "churn_per_file": churn_per_file,

        # rozkład typów
        "type_entropy": type_entropy,
        "type_gini": type_gini,
        "top_type_share": top_type_share,

        # LANG
        "lang_share.py": lang_share_py,
        "lang_share.other": lang_share_other,

        # KIND (odniesione do tokens_total)
        "kind_share.fn": kind_share_fn,
        "kind_share.class": kind_share_class,
    }

    # PATH shares: spłaszczamy do prefiksów przyjaznych dla narzędzi metrycznych
    for p, v in path_share.items():
        out[f"path_share.{p}"] = float(v)

    # Sanity: wymuszenie liczb skończonych
    for k, v in list(out.items()):
        if not (isinstance(v, float) or isinstance(v, int)):
            out[k] = float(v) if v is not None else 0.0
        if math.isnan(float(out[k])) or math.isinf(float(out[k])):  # pragma: no cover (ochrona)
            out[k] = 0.0

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Zgodność wsteczna: cechy z listy tokenów (przez histogram)
# ──────────────────────────────────────────────────────────────────────────────

def features_from_tokens(tokens: List[Token]) -> Dict[str, float]:
    """
    Zgodność wsteczna (DEPRECATED w warstwie API):
    Agreguje cechy tak jak wcześniej, ale bazując na histogramie VOCAB v1.
    """
    hist = tokens_to_hist(tokens)
    return build_features(hist)
