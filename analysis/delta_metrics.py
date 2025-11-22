# glitchlab/analysis/delta_metrics.py
# Δ-code metrics: score∈[0,1] + bramki I1–I4 na bazie cech z glitchlab.delta
# Python 3.9+

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Zależności opcjonalne (delikatne importy – moduł działa też na gołych cechach)
# ──────────────────────────────────────────────────────────────────────────────
try:
    # fingerprint: raport Δ (zawiera już cechy) – preferowane źródło
    from glitchlab.delta.fingerprint import build_delta_report  # type: ignore
except Exception:  # pragma: no cover
    build_delta_report = None  # type: ignore

# Wprost cechy z tokenów (fallback przy własnym zasilaniu)
try:
    from glitchlab.delta.features import features_from_tokens  # type: ignore
except Exception:  # pragma: no cover
    def features_from_tokens(*a, **k):  # type: ignore
        return {}

__all__ = [
    "DeltaEval",
    "compute_delta_score",
    "gates_from_score",
    "evaluate_features",
    "evaluate_report",
    "evaluate_range",
    "to_jsonable",
]

# ──────────────────────────────────────────────────────────────────────────────
# Model wyniku
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DeltaEval:
    features_version: str
    score: float                 # [0,1] – wyższe = większy wpływ/ryzyko
    gate: str                    # "I1" | "I2" | "I3" | "I4"
    gates: Dict[str, bool]       # {"I1":..,"I2":..,"I3":..,"I4":..}
    details: Dict[str, float]    # znormalizowane składowe
    raw: Dict[str, float]        # surowe (jak w raporcie cech)
    meta: Dict[str, str]         # np. {"base":"abc1234","head":"def5678"}

# ──────────────────────────────────────────────────────────────────────────────
# Pomocnicze – bezpieczna normalizacja i statystyki
# ──────────────────────────────────────────────────────────────────────────────

def _clamp01(x: float) -> float:
    try:
        if x != x or x == float("inf") or x == float("-inf"):
            return 0.0
    except Exception:
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        if b == 0 or b != b:
            return default
        v = a / b
        return v if v == v and v != float("inf") and v != float("-inf") else default
    except Exception:
        return default

def _gini_from_hist(hist: List[float]) -> float:
    """
    Gini ∈ [0,1] – 0: równomiernie, 1: całkowita koncentracja.
    Wejście dowolnie skalowane; normalizujemy wewnętrznie.
    """
    if not hist:
        return 0.0
    s = float(sum(max(0.0, h) for h in hist))
    if s <= 0.0:
        return 0.0
    xs = sorted((max(0.0, h) / s for h in hist))
    n = len(xs)
    # klasyczna formuła: G = 1 - 2 * sum_{i=1..n} ( (n+1-i)/n * x_i )
    acc = 0.0
    for i, x in enumerate(xs, start=1):
        acc += (i / n) * x
    # alternatywna to 2*acc/n - (n+1)/n; po przekształceniu:
    g = 1.0 - 2.0 * ( (n + 1) / (2.0 * n) - acc )
    return _clamp01(g)

def _entropy_from_hist(hist: List[float]) -> float:
    """
    Shannon H z histogramu (normalizacja do [0,1] względem log2(k)).
    """
    import math
    if not hist:
        return 0.0
    s = float(sum(max(0.0, h) for h in hist))
    if s <= 0.0:
        return 0.0
    ps = [max(0.0, h) / s for h in hist if h > 0.0]
    if not ps:
        return 0.0
    H = -sum(p * math.log2(p) for p in ps)
    Hmax = math.log2(len(hist)) if len(hist) > 1 else 1.0
    return _clamp01(H / Hmax)

def _extract_hist(features: Dict[str, float]) -> List[float]:
    """
    Akceptujemy jeden z formatów:
      - klucze 'hist_00','hist_01',... (posortowane po nazwie)
      - lub 'histogram' jako lista/licznik w JSON-raporcie (pomijamy tutaj; to warstwa delta/*)
    """
    # poszukaj kluczy hist_*
    buckets = [(k, v) for k, v in features.items() if k.startswith("hist_")]
    if not buckets:
        return []
    buckets.sort(key=lambda kv: kv[0])
    return [float(v) for _, v in buckets]

# ──────────────────────────────────────────────────────────────────────────────
# Rdzeń: score + bramki
# ──────────────────────────────────────────────────────────────────────────────

def compute_delta_score(features: Dict[str, float], *, weights: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
    """
    Liczy wynik ∈[0,1] na podstawie cech „v1”.
    Oczekiwane cechy (jeśli dostępne):
      - churn ∈[0,1]          – intensywność zmian
      - dens  ∈[0,1]          – „gęstość” zmian (np. na plik/patch)
      - entropy_norm ∈[0,1]   – rozproszenie zmian po typach
      - gini ∈[0,1]           – koncentracja (1=skupione)
    Fallbacki:
      - gdy brak entropy_norm/gini: policz z histogramu hist_**
      - gdy brak churn: użyj tokens_churn / (tokens_total+K)
      - gdy brak dens: użyj min(1, files_changed/16) jeśli w raw są liczniki plików
    """
    w = dict(churn=0.45, entropy=0.20, gini_inv=0.20, dens=0.15)
    if weights:
        w.update(weights)

    # Churn
    churn = float(features.get("churn", 0.0))
    if churn <= 0.0:
        # spróbuj z tokenów
        ch = float(features.get("tokens_churn", 0.0))
        tot = float(features.get("tokens_total", 0.0))
        churn = _clamp01(_safe_div(ch, max(8.0, tot)))  # miękki próg dla małych patchy

    # Dens
    dens = float(features.get("dens", 0.0))
    if dens <= 0.0:
        # przybliż: liczba plików/rozsądny limit
        files = float(features.get("files_changed", 0.0) or features.get("tokens_file_ops", 0.0))
        dens = _clamp01(files / 16.0)

    # Entropia & Gini
    entropy = float(features.get("entropy_norm", features.get("entropy", 0.0)))
    gini = float(features.get("gini", 0.0))
    if (entropy <= 0.0 and gini <= 0.0) or not (0.0 <= entropy <= 1.0):
        hist = _extract_hist(features)
        if hist:
            entropy = _entropy_from_hist(hist)
            gini = _gini_from_hist(hist)
        else:
            # awaryjnie: estymuj z rozkładu typów tokenów (jeśli są)
            # brak – ustaw neutralne
            entropy = 0.5 if entropy == 0.0 else entropy
            gini = 0.5 if gini == 0.0 else gini

    # Gini „odwrócony”: większa rozproszenie = większy wpływ
    gini_inv = _clamp01(1.0 - _clamp01(gini))

    # Znormalizowane składowe
    parts = {
        "churn": _clamp01(churn),
        "entropy": _clamp01(entropy),
        "gini_inv": _clamp01(gini_inv),
        "dens": _clamp01(dens),
    }

    # Score (ważona suma)
    score_raw = (w["churn"] * parts["churn"] +
                 w["entropy"] * parts["entropy"] +
                 w["gini_inv"] * parts["gini_inv"] +
                 w["dens"] * parts["dens"])
    score = _clamp01(score_raw)

    # Eskalacje heurystyczne (bezpieczne „clipy” w górę)
    # - modyfikacje sygnatur zwykle podnoszą ryzyko
    sig_mod = float(features.get("tokens_modify_sig", 0.0))
    body_mod = float(features.get("tokens_modify_body", 0.0))
    renames = float(features.get("tokens_rename", 0.0))
    if sig_mod >= 1:
        score = _clamp01(max(score, 0.70))
    if body_mod >= 5 or renames >= 3:
        score = _clamp01(max(score, 0.55))

    return score, parts

def gates_from_score(score: float, features: Optional[Dict[str, float]] = None) -> Tuple[str, Dict[str, bool]]:
    """
    Bramki I1–I4 (rosnąca „ważność/ryzyko”):
      I1: score < 0.25
      I2: 0.25–0.50
      I3: 0.50–0.75
      I4: >= 0.75
    Z eskalacją „twardą” gdy obecne zmiany sygnatur/duże rename.
    """
    s = _clamp01(score)
    # domyślna bramka po score
    if s < 0.25:
        gate = "I1"
    elif s < 0.50:
        gate = "I2"
    elif s < 0.75:
        gate = "I3"
    else:
        gate = "I4"

    # twarde eskalacje
    if features:
        if float(features.get("tokens_modify_sig", 0.0)) >= 1:
            gate = max(gate, "I4")  # typowo I4
        elif float(features.get("tokens_rename", 0.0)) >= 5:
            gate = max(gate, "I3")

    flags = {"I1": False, "I2": False, "I3": False, "I4": False}
    flags[gate] = True
    return gate, flags

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_features(features: Dict[str, float], *, features_version: str = "v1",
                      meta: Optional[Dict[str, str]] = None) -> DeltaEval:
    """
    Wejście: cechy Δ (np. z delta.fingerprint.build_delta_report()['features']).
    Wyjście: score + bramki + szczegóły składowych.
    """
    score, parts = compute_delta_score(features)
    gate, flags = gates_from_score(score, features)
    return DeltaEval(
        features_version=features_version,
        score=score,
        gate=gate,
        gates=flags,
        details=parts,
        raw={k: float(v) for k, v in features.items()},
        meta=dict(meta or {}),
    )

def evaluate_report(report: Dict) -> DeltaEval:
    """
    Przyjmuje payload zgodny ze spec/schemas/delta_report.json.
    Preferuje report['features']; meta wyciąga z report['range'].
    """
    feats = dict(report.get("features", {}))
    meta = {}
    rng = report.get("range") or {}
    if isinstance(rng, dict):
        meta = {
            "base": str(rng.get("base", "")),
            "head": str(rng.get("head", "")),
        }
    fv = str(report.get("features_version") or "v1")
    return evaluate_features(feats, features_version=fv, meta=meta)

def evaluate_range(base: str, head: str = "HEAD") -> Optional[DeltaEval]:
    """
    Buduje raport Δ (jeśli dostępny moduł fingerprint) i liczy wynik.
    Gdy fingerprint nieosiągalny – zwraca None.
    """
    if build_delta_report is None:
        return None
    rep = build_delta_report(base, head)  # oczekiwany zgodny ze schema
    return evaluate_report(rep)

def to_jsonable(ev: DeltaEval) -> Dict:
    d = asdict(ev)
    # porządek i zaokrąglenia
    d["score"] = round(float(d["score"]), 6)
    for k in list(d.get("details", {}).keys()):
        d["details"][k] = round(float(d["details"][k]), 6)
    return d

# ──────────────────────────────────────────────────────────────────────────────
# Mini-CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cli(argv: Optional[List[str]] = None) -> None:  # pragma: no cover
    import argparse, json
    p = argparse.ArgumentParser(prog="delta-metrics", description="Δ-code score + gates I1–I4")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("range", help="policz wynik dla zakresu base..head")
    r.add_argument("base")
    r.add_argument("head", nargs="?", default="HEAD")

    f = sub.add_parser("features", help="policz wynik z cech (JSON na stdin)")

    args = p.parse_args(argv)
    if args.cmd == "range":
        ev = evaluate_range(args.base, args.head)
        if ev is None:
            print(json.dumps({"ok": False, "error": "fingerprint module unavailable"}, indent=2))
            return
        print(json.dumps(to_jsonable(ev), ensure_ascii=False, indent=2))
    else:
        raw = json.loads(input())
        ev = evaluate_features(raw)
        print(json.dumps(to_jsonable(ev), ensure_ascii=False, indent=2))

if __name__ == "__main__":  # pragma: no cover
    _cli()
