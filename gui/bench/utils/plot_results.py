# glitchlab/gui/bench/stats.py
from __future__ import annotations
import statistics
import math
from typing import Dict, Any, List


def _safe_mean(xs: List[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def _aggregate(agent_results: Dict[str, Any]) -> Dict[str, Any]:
    """Agreguje wyniki pojedynczego agenta (A1, A2, B)."""
    pass_counts, totals, times = [], [], []
    aligns, j_phi2s, cr_tos, cr_asts = [], [], [], []

    for _, tres in agent_results.items():
        if not isinstance(tres, dict):
            continue

        if "pass_at_1" in tres:
            pass_counts.append(tres["pass_at_1"])
        if "total" in tres:
            totals.append(tres["total"])
        if "time_s" in tres:
            times.append(tres["time_s"])

        # Obsługa obu wariantów kluczy (małe/wielkie litery)
        aligns.extend([tres[k]] for k in ("align", "Align") if k in tres and tres[k] is not None)
        j_phi2s.extend([tres[k]] for k in ("j_phi2", "J_phi2") if k in tres and tres[k] is not None)
        cr_tos.extend([tres[k]] for k in ("cr_to", "CR_TO") if k in tres and tres[k] is not None)
        cr_asts.extend([tres[k]] for k in ("cr_ast", "CR_AST") if k in tres and tres[k] is not None)

    return {
        "pass_at_1": sum(pass_counts),
        "total": sum(totals),
        "mean_time": _safe_mean(times),
        "mean_align": _safe_mean(aligns),
        "mean_j_phi2": _safe_mean(j_phi2s),
        "mean_cr_to": _safe_mean(cr_tos),
        "mean_cr_ast": _safe_mean(cr_asts),
    }


def cliffs_delta(xs: List[float], ys: List[float]) -> float:
    """Cliff’s delta effect size."""
    nx, ny = len(xs), len(ys)
    if nx == 0 or ny == 0:
        return 0.0
    gt = sum(1 for x in xs for y in ys if x > y)
    lt = sum(1 for x in xs for y in ys if x < y)
    return (gt - lt) / (nx * ny)


def binomial_sign_test_p(wins: int, losses: int) -> float:
    """Dwustronny test znaku (p-value)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)


def summarize(A1: Dict[str, Any], A2: Dict[str, Any], B: Dict[str, Any]) -> Dict[str, Any]:
    """Zbiorczy raport z wyników wszystkich agentów."""
    agg_A1 = _aggregate(A1)
    agg_A2 = _aggregate(A2)
    agg_B = _aggregate(B)

    return {
        "A1": agg_A1,
        "A2": agg_A2,
        "B": agg_B,
        # Rozszerzona analiza porównawcza
        "A2_vs_A1": _compare(agg_A2, agg_A1),
        "A2_vs_B": _compare(agg_A2, agg_B),
    }


def _compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Porównanie dwóch agentów na podstawie pass@1 (sign test + effect size)."""
    wins = 1 if a["pass_at_1"] > b["pass_at_1"] else 0
    losses = 1 if a["pass_at_1"] < b["pass_at_1"] else 0
    ties = 1 if a["pass_at_1"] == b["pass_at_1"] else 0

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "p_sign": binomial_sign_test_p(wins, losses),
        "cliffs_delta": cliffs_delta([a["pass_at_1"]], [b["pass_at_1"]]),
    }
