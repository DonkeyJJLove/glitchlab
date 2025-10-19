# glitchlab/analysis/metrics.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict, Any, Iterable, Tuple


def _collect_histogram(features: Dict[str, float]) -> Tuple[float, list[float]]:
    """Zwraca sum(hist) i listę wartości bucketów w kolejności hist_00, hist_01..."""
    buckets = []
    i = 0
    while True:
        key = f"hist_{i:02d}"
        if key in features:
            buckets.append(float(features.get(key, 0.0)))
            i += 1
            continue
        # zakończ gdy nie ma kolejnego klucza
        break
    total = float(sum(buckets))
    return total, buckets


def _compute_entropy_and_gini_from_buckets(buckets: Iterable[float]) -> Tuple[float, float]:
    """
    Entropy normalized (0..1) and gini-like measure such that:
    - one-hot histogram -> gini == 1.0 (so (1-gini) == 0)
    - uniform histogram -> gini small
    We will use gini = sum(p_i^2) in [0,1], which gives gini==1 for one-hot.
    """
    buckets = list(buckets)
    s = sum(buckets)
    if s <= 0:
        return 0.0, 0.0
    ps = [b / s for b in buckets]
    # entropy: use normalized Shannon: H / log2(n) (if n>1), else 0
    n = len(ps)
    if n <= 1:
        entropy_norm = 0.0
    else:
        H = 0.0
        for p in ps:
            if p > 0:
                H -= p * math.log2(p)
        denom = math.log2(n) if n > 1 else 1.0
        entropy_norm = 0.0 if denom == 0 else (H / denom)
    # gini-like: sum(p_i^2) (1 for one-hot)
    gini = float(sum(p * p for p in ps))
    return entropy_norm, gini


def _compute_churn_norm(features: Dict[str, float]) -> float:
    """
    churn_norm is either provided, or tokens_churn / max(8, tokens_total)
    """
    if "churn_norm" in features and features["churn_norm"] is not None:
        try:
            return float(features["churn_norm"])
        except Exception:
            pass
    tokens_churn = float(features.get("tokens_churn", 0.0))
    tokens_total = float(features.get("tokens_total", 0.0))
    denom = max(8.0, tokens_total if tokens_total > 0 else 8.0)
    return tokens_churn / denom if denom > 0 else 0.0


def _compute_dens(features: Dict[str, float]) -> float:
    """
    dens is either provided, or files_changed / 16 (16 is arbitrary small repo scale)
    """
    if "dens" in features and features["dens"] is not None:
        try:
            return float(features["dens"])
        except Exception:
            pass
    files_changed = float(features.get("files_changed", 0.0))
    return files_changed / 16.0


def evaluate_features(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Zwraca dict z kluczami: score (float 0..1), gate (I0..I4) oraz użytymi cechami.
    Firma wag i progi zgodna z testami:
      score = 0.45*churn_norm + 0.2*entropy_norm + 0.2*(1-gini) + 0.15*dens
    """
    # normalizacja / fallbacks
    churn_norm = _compute_churn_norm(features)
    entropy_norm = float(features.get("entropy_norm", float("nan")))
    gini = float(features.get("gini", float("nan")))
    dens = float(features.get("dens", float("nan")))

    # jeśli brakuje entropy/gini, policz z histogramu gdy dostępne
    if (math.isnan(entropy_norm) or math.isnan(gini)):
        _, buckets = _collect_histogram(features)
        if buckets:
            e, g = _compute_entropy_and_gini_from_buckets(buckets)
            if math.isnan(entropy_norm):
                entropy_norm = e
            if math.isnan(gini):
                gini = g

    # jeśli dalej NaN, ustaw bezpieczne wartości
    if math.isnan(entropy_norm):
        entropy_norm = 0.0
    if math.isnan(gini):
        # jeżeli brak danych traktujemy jako niski gini (więcej różnych tokenów) -> gini ~ 0
        gini = 0.0
    if math.isnan(dens):
        dens = _compute_dens(features)

    # clamp to [0,1]
    def _clamp01(x: float) -> float:
        if x != x:  # NaN
            return 0.0
        return max(0.0, min(1.0, float(x)))

    churn_norm = _clamp01(churn_norm)
    entropy_norm = _clamp01(entropy_norm)
    gini = _clamp01(gini)
    dens = _clamp01(dens)

    score = 0.45 * churn_norm + 0.2 * entropy_norm + 0.2 * (1.0 - gini) + 0.15 * dens
    # clamp again
    score = float(max(0.0, min(1.0, score)))

    tokens_modify_sig = int(features.get("tokens_modify_sig", 0))

    gate = classify_gate(score, tokens_modify_sig=tokens_modify_sig)

    return {
        "score": score,
        "gate": gate,
        "churn_norm": churn_norm,
        "entropy_norm": entropy_norm,
        "gini": gini,
        "dens": dens,
    }


def classify_gate(score: float, tokens_modify_sig: int | None = None) -> str:
    """
    Proste progi bramek (dostosowane do testów).
    Rozmiary progów:
      score < 0.2 -> I1
      score < 0.5 -> I2
      score < 0.8 -> I3
      else -> I4
    Dodatkowo eskalacja gdy tokens_modify_sig >= 3 -> +1 poziom (np. I2 -> I3).
    (I0 nie jest używane w testach, ale zostawiamy I0 dla kompletności.)
    """
    # podstawowe przypisanie
    if score < 0.05:
        base = "I0"
    elif score < 0.2:
        base = "I1"
    elif score < 0.5:
        base = "I2"
    elif score < 0.8:
        base = "I3"
    else:
        base = "I4"

    # eskalacja na sygnatury
    level_map = {"I0": 0, "I1": 1, "I2": 2, "I3": 3, "I4": 4}
    inv_map = {v: k for k, v in level_map.items()}
    lvl = level_map.get(base, 2)
    if tokens_modify_sig is not None and int(tokens_modify_sig) >= 3:
        lvl = min(4, lvl + 1)
    return inv_map.get(lvl, base)


def evaluate_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Przyjmuje struktury jak w testach:
      { "features": {...}, "meta": {...} }
    Zwraca rozszerzony raport z "score", "gate" i zachowaną meta.
    """
    features = report.get("features", {})
    evaluated = evaluate_features(features)
    out = {
        "score": evaluated["score"],
        "gate": evaluated["gate"],
        "features": features,
        "meta": report.get("meta", {}),
    }
    # ensure meta.range exists when tests expect it
    if "meta" not in out:
        out["meta"] = {}
    return out


# build_delta_report — placeholder, testy podstawiają stub/gmock
def build_delta_report(base: str, head: str) -> Dict[str, Any]:
    raise NotImplementedError("build_delta_report should be provided by delta module or tests")


def evaluate_range(base: str, head: str) -> Dict[str, Any]:
    """
    Wywołuje build_delta_report(base, head), a następnie evaluate_report.
    Testy zamieniają build_delta_report() więc ta funkcja musi korzystać z globalnego symbolu.
    """
    report = build_delta_report(base, head)
    # Some build_delta_report implementations return features dict directly — handle both
    if isinstance(report, dict) and "features" not in report and all(k.startswith("hist_") or k in ("tokens_total", "tokens_churn") for k in report.keys()):
        # treat as features
        report = {"features": report, "meta": {"range": {"base": base, "head": head}}}
    # ensure meta.range if missing
    if "meta" not in report:
        report["meta"] = {"range": {"base": base, "head": head}}
    out = evaluate_report(report)
    # copy meta.range through (tests assert on it)
    out["meta"] = report.get("meta", {})
    return out
