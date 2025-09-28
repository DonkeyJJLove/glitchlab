# glitchlab/gui/bench/stats.py
import statistics
import math
from typing import Dict, Any, List

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

        # Obsługa kluczy w obu wariantach (małe/wielkie litery)
        if "align" in tres and tres["align"] is not None:
            aligns.append(tres["align"])
        elif "Align" in tres and tres["Align"] is not None:
            aligns.append(tres["Align"])

        if "j_phi2" in tres and tres["j_phi2"] is not None:
            j_phi2s.append(tres["j_phi2"])
        elif "J_phi2" in tres and tres["J_phi2"] is not None:
            j_phi2s.append(tres["J_phi2"])

        if "cr_to" in tres and tres["cr_to"] is not None:
            cr_tos.append(tres["cr_to"])
        elif "CR_TO" in tres and tres["CR_TO"] is not None:
            cr_tos.append(tres["CR_TO"])

        if "cr_ast" in tres and tres["cr_ast"] is not None:
            cr_asts.append(tres["cr_ast"])
        elif "CR_AST" in tres and tres["CR_AST"] is not None:
            cr_asts.append(tres["CR_AST"])

    return {
        "pass_at_1": sum(pass_counts),
        "total": sum(totals),
        "mean_time": statistics.mean(times) if times else None,
        "mean_align": statistics.mean(aligns) if aligns else None,
        "mean_j_phi2": statistics.mean(j_phi2s) if j_phi2s else None,
        "mean_cr_to": statistics.mean(cr_tos) if cr_tos else None,
        "mean_cr_ast": statistics.mean(cr_asts) if cr_asts else None,
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
    return {
        "A1": _aggregate(A1),
        "A2": _aggregate(A2),
        "B": _aggregate(B),
    }
