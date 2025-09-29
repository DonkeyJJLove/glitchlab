# glitchlab/analysis/policy_phi_psi.py
# Polityki/latawce Φ/Ψ nad RepoMosaic ↔ AST:
# - konwersja RepoMosaic → Mosaic (grid)
# - selektory Φ (balanced/entropy)
# - Ψ-feedback + sprzężenie (α,β)
# - zestaw polityk (edge-preserve, roi-stability, diff-budget)
# - generator raportu commit-note (mosaic/AST)
#
# Python 3.9+ (deps: numpy; stdlib only poza importami z naszego projektu)

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ── lokalne moduły (1.1 i 1.2) ────────────────────────────────────────────────
from glitchlab.gui.mosaic.hybrid_ast_mosaic import (
    Mosaic,
    build_mosaic,                # nieużywane bezpośrednio; zostawione dla spójności API
    region_ids,
    phi_region_for_balanced,
    phi_region_for_entropy,
    phi_cost,
    psi_feedback,
    couple_alpha_beta,
    distance_ast_mosaic,
    ast_deltas,
    compress_ast,
    EXAMPLE_SRC,
    EDGE_THR_DEFAULT,
    W_DEFAULT,
)

from glitchlab.analysis.git_io import (
    RepoMosaic,
    RepoInfo,
)


# ──────────────────────────────────────────────────────────────────────────────
# 0) Pomocnicze: RepoMosaic → Mosaic (grid)
# ──────────────────────────────────────────────────────────────────────────────

def repo_mosaic_to_mosaic(R: RepoMosaic) -> Mosaic:
    """
    Rzutuje RepoMosaic (kafel = plik) na Mosaic grid używany przez algorytm Φ/Ψ.
    - edge: z RepoMosaic.edge
    - roi:  z RepoMosaic.roi
    - ssim: placeholder (1.0)
    """
    rows = max(1, int(R.layout_rows))
    cols = max(1, int(R.layout_cols))
    N = rows * cols
    # jeżeli liczba plików < rows*cols, dopełniamy zerami (stabilne)
    edge = np.zeros(N, dtype=float)
    roi = np.zeros(N, dtype=float)
    ssim = np.ones(N, dtype=float)
    for i, e in enumerate(R.edge.tolist()):
        if i >= N:
            break
        edge[i] = float(e)
        roi[i] = float(R.roi[i]) if i < len(R.roi) else 0.0
    return Mosaic(rows=rows, cols=cols, edge=edge, ssim=ssim, roi=roi, kind="grid")


# ──────────────────────────────────────────────────────────────────────────────
# 1) Definicje polityk (latawce)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyResult:
    name: str
    score: float
    hints: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)   # deklaratywne sugestie


@dataclass
class Policy:
    name: str
    fn: Callable[..., PolicyResult]


def policy_edge_preserve(M: Mosaic, thr: float = EDGE_THR_DEFAULT) -> PolicyResult:
    """Edge-preserve: IO/Call/Expr → edges, filtry destruktywne ograniczaj do ~edges."""
    n_edges = len(region_ids(M, "edges", thr))
    n_all = M.rows * M.cols
    p = n_edges / max(1, n_all)
    # im więcej 'edges', tym ważniejsze rozdzielenie efektów ubocznych (IO) od ROI
    score = float(min(1.0, 0.5 + 0.5 * p))
    hints = [
        f"edges tiles: {n_edges}/{n_all} (p≈{p:.2f})",
        "Preferuj Call/Expr w warstwie edges; unikaj side-effects w ROI."
    ]
    actions = [
        "Oznacz wywołania IO jako region='edges'.",
        "Jeśli filtr modyfikuje szeroko, ogranicz region do ~edges lub zastosuj feathering na granicach."
    ]
    return PolicyResult("edge_preserve", score, hints, actions)


def policy_roi_stability(M: Mosaic, thr: float = EDGE_THR_DEFAULT) -> PolicyResult:
    """ROI-stability: Assign/Def → roi oraz ~edges; pilnuj stabilizacji stanu."""
    roi_ids = region_ids(M, "roi", thr)
    n_roi = len(roi_ids)
    n_all = M.rows * M.cols
    p = n_roi / max(1, n_all)
    # jeżeli ROI jest niewielkie – podbij stabilizację tam i w ~edges
    score = float(min(1.0, 0.6 + 0.4 * (1.0 - p)))
    hints = [
        f"roi tiles: {n_roi}/{n_all} (p≈{p:.2f})",
        "Stabilizuj stan (Assign/Def) w ROI i ~edges, ogranicz efekt w edges."
    ]
    actions = [
        "Wstaw inicjalizacje stanu (Assign) przed gałęziami decyzyjnymi.",
        "Reguła: w ROI nie emituj side-effects – preferuj 'return' nad 'print'."
    ]
    return PolicyResult("roi_stability", score, hints, actions)


def policy_diff_budget(M: Mosaic, thr: float = EDGE_THR_DEFAULT) -> PolicyResult:
    """
    Diff-budget: kontrola globalnej zmiany poza ROI.
    Tu heurystycznie: im większy udział ~edges, tym ciaśniejszy budżet.
    """
    n_ne = len(region_ids(M, "~edges", thr))
    n_all = M.rows * M.cols
    p = n_ne / max(1, n_all)
    # im większa strefa spokojna, tym większy nacisk na ograniczenie zmian
    score = float(min(1.0, 0.4 + 0.6 * p))
    B = max(0.02, 0.15 - 0.10 * p)  # proponowany budżet MSE poza ROI
    hints = [
        f"~edges tiles: {n_ne}/{n_all} (p≈{p:.2f})",
        f"Proponowany budżet MSE poza ROI: ≤ {B:.3f}",
    ]
    actions = [
        f"Wymuś 'diff-budget' poza ROI ≤ {B:.3f}.",
        "Jeśli przekroczony → wstaw węzeł kompensujący (Repair) albo rollback."
    ]
    return PolicyResult("diff_budget", score, hints, actions)


POLICIES: List[Policy] = [
    Policy("edge_preserve", policy_edge_preserve),
    Policy("roi_stability", policy_roi_stability),
    Policy("diff_budget", policy_diff_budget),
]


# ──────────────────────────────────────────────────────────────────────────────
# 2) Analiza Φ/Ψ nad repo: metryki + sugestie
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisConfig:
    edge_thr: float = EDGE_THR_DEFAULT
    lmbd: float = 0.60         # λ-kompresja AST
    delta: float = 0.25        # siła Ψ-feedback
    kappa_ab: float = 0.35     # sprzężenie (α,β)
    use_entropy_selector: bool = False  # Φ: entropy vs balanced


@dataclass
class AnalysisResult:
    align: float
    j_phi: float
    alpha: float
    beta: float
    S: int
    H: int
    Z: int
    policies: List[PolicyResult]
    commit_note: str
    details: Dict[str, float]


Selector = Callable[[str, Mosaic, float], str]


def analyze_repo_ast(
    repo: RepoMosaic,
    src_text: Optional[str] = None,
    cfg: Optional[AnalysisConfig] = None,
) -> AnalysisResult:
    """
    Główna pętla: RepoMosaic → Mosaic → Φ/Ψ → metryki + commit-note.
    - src_text: jeżeli None, używa EXAMPLE_SRC (placeholder do czasu wpięcia rzeczywistego AST).
    """
    cfg = cfg or AnalysisConfig()
    selector: Selector = (phi_region_for_entropy if cfg.use_entropy_selector else phi_region_for_balanced)
    M = repo_mosaic_to_mosaic(repo)

    # AST: źródło (na dziś: placeholder lub w przyszłości scalone moduły)
    src = src_text if src_text is not None else EXAMPLE_SRC
    ast0 = ast_deltas(src)
    ast_l = compress_ast(ast0, cfg.lmbd)

    # Koszt Φ (przy wybranym selektorze)
    J_phi, _ = phi_cost(ast_l, M, cfg.edge_thr, selector=selector)

    # Ψ + sprzężenie (α,β)
    ast_after = psi_feedback(ast_l, M, cfg.delta, cfg.edge_thr)
    ast_cpl = couple_alpha_beta(ast_after, M, cfg.edge_thr, delta=cfg.delta, kappa_ab=cfg.kappa_ab)

    align = 1.0 - min(1.0, distance_ast_mosaic(ast_cpl, M, cfg.edge_thr, w=W_DEFAULT))

    # Polityki
    pol_res: List[PolicyResult] = [p.fn(M, cfg.edge_thr) for p in POLICIES]
    pol_res.sort(key=lambda pr: pr.score, reverse=True)

    # Commit-note (syntetyczny, pod nasze standardy)
    note = _render_commit_note(repo, align, J_phi, ast_cpl, pol_res)

    details = dict(
        Align=align,
        J_phi=J_phi,
        alpha=ast_cpl.alpha,
        beta=ast_cpl.beta,
        S=float(ast_cpl.S),
        H=float(ast_cpl.H),
        Z=float(ast_cpl.Z),
    )

    return AnalysisResult(
        align=align,
        j_phi=J_phi,
        alpha=ast_cpl.alpha,
        beta=ast_cpl.beta,
        S=ast_cpl.S,
        H=ast_cpl.H,
        Z=ast_cpl.Z,
        policies=pol_res,
        commit_note=note,
        details=details,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3) Render commit-note (zgodnie z naszym wzorcem)
# ──────────────────────────────────────────────────────────────────────────────

def _render_commit_note(repo: RepoMosaic, align: float, j_phi: float,
                        ast_cpl, pols: List[PolicyResult]) -> str:
    files_n = len([f for f in repo.files if f])
    top_pols = pols[:3]
    pol_lines = []
    for pr in top_pols:
        pol_lines.append(f"- {pr.name}: score={pr.score:.2f}")
        if pr.hints:
            pol_lines += [f"  • {h}" for h in pr.hints[:2]]
        if pr.actions:
            pol_lines += [f"  → {a}" for a in pr.actions[:2]]

    return (
        f"[Δ] Zakres\n"
        f"- files: {files_n}\n"
        f"- typ: analiza Φ/Ψ + polityki (repo-mosaic)\n\n"
        f"[Φ/Ψ] Mozaika/AST\n"
        f"- Align(mean): {align:.3f}\n"
        f"- J_phi(balanced): {j_phi:.4f}\n"
        f"- α/β: {ast_cpl.alpha:.2f}/{ast_cpl.beta:.2f}\n"
        f"- AST(S/H/Z): {ast_cpl.S}/{ast_cpl.H}/{ast_cpl.Z}\n\n"
        f"[Polityki] (top)\n" + "\n".join(pol_lines) + "\n\n"
        f"[Dokumentacja]\n"
        f"- decyzja: auto-note (no-op dla docs; tylko raport analityczny)\n\n"
        f"[Testy / Ryzyko]\n"
        f"- smoke: analiza off-line (repo snapshot)\n"
        f"- ryzyko: niskie (bez zmian kodu)\n\n"
        f"Meta\n"
        f"- Generated-by: policy_phi_psi (Φ/Ψ + latawce)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4) Proste CLI do lokalnego uruchomienia (np. w hookach)
# ──────────────────────────────────────────────────────────────────────────────

def cli_main(argv: Optional[List[str]] = None) -> None:
    import argparse
    from glitchlab.analysis.git_io import build_repo_mosaic

    p = argparse.ArgumentParser(prog="policy_phi_psi", description="Repo Φ/Ψ + polityki (commit-note)")
    p.add_argument("--edge-thr", type=float, default=EDGE_THR_DEFAULT)
    p.add_argument("--lmbd", type=float, default=0.60)
    p.add_argument("--delta", type=float, default=0.25)
    p.add_argument("--kappa-ab", type=float, default=0.35)
    p.add_argument("--entropy", action="store_true", help="użyj selektora Φ(entropy) zamiast balanced")
    p.add_argument("--export", action="store_true", help="wypisz JSON wyniku")
    args = p.parse_args(argv)

    info, repo_m, churn = build_repo_mosaic()
    cfg = AnalysisConfig(
        edge_thr=args.edge_thr,
        lmbd=args.lmbd,
        delta=args.delta,
        kappa_ab=args.kappa_ab,
        use_entropy_selector=args.entropy,
    )
    res = analyze_repo_ast(repo_m, src_text=None, cfg=cfg)

    print(res.commit_note)
    if args.export:
        payload = dict(
            details=res.details,
            policies=[dict(name=p.name, score=p.score, hints=p.hints, actions=p.actions) for p in res.policies],
            note=res.commit_note,
        )
        print("\n[JSON]\n" + json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli_main()
