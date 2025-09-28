# glitchlab/mosaic/hybrid_ast_mosaic.py
# Hybrydowy algorytm AST ⇄ Mozaika (Φ/Ψ), ΔS/ΔH/ΔZ, λ-kompresja,
# warianty Φ, Ψ-feedback, metryki, inwarianty, sweep λ×Δ, CLI.
# Python 3.9+  (deps: numpy; stdlib: ast, math, json, argparse, itertools, hashlib)

from __future__ import annotations
import ast
import math
import json
import argparse
import itertools
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 0) PARAMS / PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────
EDGE_THR_DEFAULT: float = 0.55
W_DEFAULT = dict(wS=1.0, wH=1.0, wZ=0.4)  # wagi Align

# soft-labels dla krawędzi
SOFT_LABELS_DEFAULT: bool = True
SOFT_LABELS_TAU: float = 0.08  # temperatura sigmoidy dla soft-etykiety

# sprzężenie Ψ→(α,β) (wpływ Δ na Align)
KAPPA_AB_DEFAULT: float = 0.35  # siła mieszania α/β z profilem mozaiki

__all__ = [
    # dataclasses
    "AstNode", "AstSummary", "Mosaic",
    # AST
    "ast_deltas", "compress_ast",
    # Mosaic
    "build_mosaic", "build_mosaic_grid", "build_mosaic_hex",
    "region_ids", "D_M",
    # Phi / Psi
    "phi_region_for", "phi_region_for_balanced", "phi_region_for_entropy",
    "phi_cost", "psi_feedback",
    # Coupling
    "couple_alpha_beta",
    # Metrics / Tests
    "mosaic_profile", "distance_ast_mosaic", "invariants_check",
    # Runs
    "run_once", "sweep", "sign_test_phi2_better",
    # CLI
    "build_cli", "main",
    # Example & params
    "EXAMPLE_SRC", "EDGE_THR_DEFAULT", "W_DEFAULT",
    "SOFT_LABELS_DEFAULT", "SOFT_LABELS_TAU", "KAPPA_AB_DEFAULT",
]

EXAMPLE_SRC = r"""
import os, sys
from pathlib import Path

def hello(msg: str) -> str:
    x = msg.upper()
    print(x)
    return x

def main():
    p = Path('.')
    s = hello("Hi")
    if p.exists():
        print(s)
    return 0

if __name__ == "__main__":
    main()
"""


# ──────────────────────────────────────────────────────────────────────────────
# 1) AST → ΔS/ΔH/ΔZ  + λ-kompresja
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AstNode:
    """Węzeł AST z meta-wektorem [L,S,Sel,Stab,Cau,H]."""
    id: int
    label: str
    depth: int
    parent: Optional[int]
    children: List[int] = field(default_factory=list)
    meta: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))


@dataclass
class AstSummary:
    """Zbiorcze statystyki AST + indeks węzłów."""
    S: int
    H: int
    Z: int
    maxZ: int
    alpha: float
    beta: float
    nodes: Dict[int, AstNode]
    labels: List[str]


def _rng_for_meta(label: str, depth: int) -> np.random.Generator:
    """Deterministyczny RNG niezależny od PYTHONHASHSEED (md5(label|depth))."""
    key = f"{label}|{depth}".encode("utf-8")
    h = hashlib.md5(key).digest()
    seed = int.from_bytes(h[:8], "little", signed=False)
    return np.random.default_rng(seed)


def _meta_for(label: str, depth: int) -> np.ndarray:
    rng = _rng_for_meta(label, depth)
    L, S, Sel, Stab, Cau, H = rng.uniform(0.35, 0.85, size=6)
    if label in ("FunctionDef", "ClassDef"):
        Stab = max(Stab, 0.8);
        Sel = max(Sel, 0.6)
    if label in ("If", "While", "For", "With", "Try"):
        Sel = max(Sel, 0.75);
        Cau = max(Cau, 0.7)
    if label in ("Call", "Expr"):
        L = max(L, 0.6)
    if label in ("Assign",):
        Stab = max(Stab, 0.7)
    return np.array([L, S, Sel, Stab, Cau, H], dtype=float)


def ast_deltas(src: str) -> AstSummary:
    """Parsuje źródło Pythona i liczy przybliżone ΔS/ΔH/ΔZ wg prostych reguł."""
    tree = ast.parse(src)
    nodes: Dict[int, AstNode] = {}
    S = H = Z = 0
    maxZ = 0
    nid = 0

    def add(a: ast.AST, depth: int, parent: Optional[int]) -> int:
        nonlocal nid, S, H, Z, maxZ
        i = nid;
        nid += 1
        lab = a.__class__.__name__
        n = AstNode(i, lab, depth, parent)
        n.meta = _meta_for(lab, depth)
        nodes[i] = n
        if parent is not None:
            nodes[parent].children.append(i)

        # Δ-reguły (skrót)
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            S += 1;
            H += 1;
            Z += 1
        elif isinstance(a, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            S += 1;
            Z += 1
        elif isinstance(a, ast.Assign):
            S += 1;
            H += 1
        elif isinstance(a, ast.Call):
            S += 1;
            H += 2
        elif isinstance(a, (ast.Import, ast.ImportFrom)):
            S += 1;
            H += len(a.names)
        elif isinstance(a, ast.Name):
            H += 1

        maxZ = max(maxZ, depth)

        for ch in ast.iter_child_nodes(a):
            add(ch, depth + 1, i)

        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                          ast.If, ast.For, ast.While, ast.With, ast.Try)):
            Z -= 1
        return i

    add(tree, 0, None)
    S = int(S);
    H = int(H);
    Z = int(max(Z, 0))
    tot = max(1, S + H)
    return AstSummary(S, H, Z, maxZ=maxZ, alpha=S / tot, beta=H / tot,
                      nodes=nodes, labels=[n.label for n in nodes.values()])


def compress_ast(summary: AstSummary, lam: float) -> AstSummary:
    """λ-kompresja: redukcja udziału liściastych wkładów S/H oraz spłaszczenie Z."""
    lam = float(min(1.0, max(0.0, lam)))
    labels = summary.labels
    leaf_ratio = (labels.count("Name") + labels.count("Constant")) / max(1, len(labels))
    S_leaf = int(round(summary.S * 0.35 * leaf_ratio))
    H_leaf = int(round(summary.H * 0.35 * leaf_ratio))
    S2 = max(0, summary.S - int(round(lam * S_leaf)))
    H2 = max(0, summary.H - int(round(lam * H_leaf)))
    targetZ = int(math.ceil(max(1, summary.maxZ) / 2))
    Z2 = int(round((1 - lam) * summary.Z + lam * targetZ))
    tot = max(1, S2 + H2)
    return AstSummary(S2, H2, Z2, summary.maxZ, S2 / tot, H2 / tot, summary.nodes, summary.labels)


# ──────────────────────────────────────────────────────────────────────────────
# 2) Mozaika: grid/hex, regiony, metryka D_M
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Mosaic:
    rows: int
    cols: int
    edge: np.ndarray  # (N,) [0..1]
    ssim: np.ndarray  # (N,)
    roi: np.ndarray  # (N,)
    kind: str = "grid"  # "grid" | "hex"
    hex_centers: Optional[List[Tuple[float, float]]] = None
    hex_R: Optional[float] = None


def build_mosaic_grid(rows: int, cols: int, seed: int = 7) -> Mosaic:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols]
    diag = 1.0 - np.abs(xx - yy) / max(rows, cols)
    edge = np.clip(0.45 + 0.5 * diag + 0.06 * rng.standard_normal((rows, cols)), 0, 1).reshape(-1)
    ssim = np.ones(rows * cols, dtype=float)
    roi = np.zeros(rows * cols, dtype=float)
    r0, r1 = int(0.30 * rows), int(0.62 * rows)
    c0, c1 = int(0.30 * cols), int(0.62 * cols)
    for r in range(r0, r1):
        for c in range(c0, c1):
            roi[r * cols + c] = 1.0
    return Mosaic(rows, cols, edge, ssim, roi, kind="grid")


def build_mosaic_hex(rows: int, cols: int, seed: int = 7, R: float = 1.0) -> Mosaic:
    """Hex (odd-r offset) – heksy stykają się bokami."""
    M = build_mosaic_grid(rows, cols, seed)
    centers: List[Tuple[float, float]] = []
    w = math.sqrt(3) * R
    for r in range(rows):
        for c in range(cols):
            x = c * w + (r % 2) * (w / 2.0)
            y = r * 1.5 * R
            centers.append((x, y))
    M.kind = "hex"
    M.hex_centers = centers
    M.hex_R = R
    return M


def build_mosaic(rows: int, cols: int, seed: int = 7,
                 kind: str = "grid", edge_thr: float = EDGE_THR_DEFAULT) -> Mosaic:
    """Ujednolicony builder (edge_thr tylko dla spójności API)."""
    return build_mosaic_hex(rows, cols, seed) if kind == "hex" else build_mosaic_grid(rows, cols, seed)


def region_ids(M: Mosaic, kind: str, thr: float) -> List[int]:
    if kind == "edges":   return [i for i, v in enumerate(M.edge) if v > thr]
    if kind == "~edges":  return [i for i, v in enumerate(M.edge) if v <= thr]
    if kind == "roi":     return [i for i, v in enumerate(M.roi) if v > 0.5]
    if kind == "all":     return list(range(M.rows * M.cols))
    return []


def _xy_of_idx(i: int, M: Mosaic) -> Tuple[float, float]:
    if M.kind == "grid" or not M.hex_centers:
        return (float(i % M.cols), float(i // M.cols))
    else:
        return M.hex_centers[i]


def _sigmoid(x: float) -> float:
    # stabilna sigmoida
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _soft_label(edge_val: float, thr: float, tau: float) -> float:
    """P(edge|val) – miękka etykieta krawędzi."""
    return _sigmoid((edge_val - thr) / max(1e-6, tau))


def tile_dist(i: int, j: int, M: Mosaic, thr: float,
              alpha=1.0, beta=0.7, gamma=0.5,
              use_soft_labels: bool = SOFT_LABELS_DEFAULT,
              tau: float = SOFT_LABELS_TAU) -> float:
    x1, y1 = _xy_of_idx(i, M)
    x2, y2 = _xy_of_idx(j, M)
    geo = math.hypot(x1 - x2, y1 - y2)
    feat = abs(float(M.edge[i]) - float(M.edge[j]))
    if use_soft_labels:
        p1 = _soft_label(float(M.edge[i]), thr, tau)
        p2 = _soft_label(float(M.edge[j]), thr, tau)
        label_pen = abs(p1 - p2)  # [0..1]
    else:
        label_pen = 1.0 if ((M.edge[i] > thr) != (M.edge[j] > thr)) else 0.0
    return alpha * geo + beta * feat + gamma * label_pen


def _dm_greedy_oneway(S: List[int], T: List[int], M: Mosaic, thr: float) -> float:
    """Zachłanne dopasowanie O(k^2) – pojedynczy kierunek (S→T)."""
    if not S and not T: return 0.0
    if not S or not T:  return 0.0  # kara długości dodamy wyżej
    S2, T2 = S[:], T[:]
    cost = 0.0
    while S2 and T2:
        i = S2.pop()
        j_best = min(T2, key=lambda j: tile_dist(i, j, M, thr))
        cost += tile_dist(i, j_best, M, thr)
        T2.remove(j_best)
    return cost


def _length_penalty(len_diff: int, M: Mosaic, thr: float) -> float:
    """Kara za różnicę rozmiarów (stabilizuje przypadki brzegowe bez ∞)."""
    if len_diff <= 0: return 0.0
    # oszacuj typowy dystans przez próbkowanie
    N = M.rows * M.cols
    if N < 2: return float(len_diff)
    rng = np.random.default_rng(12345)
    K = min(32, max(1, N // 4))
    idx = rng.choice(N, size=2 * K, replace=False)
    sample = [tile_dist(int(idx[2 * t]), int(idx[2 * t + 1]), M, thr) for t in range(K)]
    kappa = float(np.mean(sample)) if sample else 1.0
    return kappa * float(len_diff)


def _pair_cost(S: List[int], T: List[int], M: Mosaic, thr: float, max_match: int) -> float:
    """Koszt dopasowania w jednym kierunku + kara długości."""
    if not S and not T:
        return 0.0
    k = min(len(S), len(T), max_match)
    len_diff = abs(len(S) - len(T))
    S2, T2 = S[:k], T[:k]
    if k <= 8:
        # dokładne
        if k == 0:
            base = 0.0
        else:
            best = float('inf')
            for perm in itertools.permutations(T2):
                cost = 0.0
                for i, j in zip(S2, perm):
                    cost += tile_dist(i, j, M, thr)
                if cost < best: best = cost
                if best == 0.0: break
            base = best
    else:
        base = _dm_greedy_oneway(S2, T2, M, thr)
    return base + _length_penalty(len_diff, M, thr)


def D_M(S: List[int], T: List[int], M: Mosaic, thr: float, max_match: int = 12) -> float:
    """
    Symetryczna wersja kosztu dopasowania (Earth-Mover-lite).
    - k<=8: dokładne (permutacje)
    - k>8: zachłanne O(k^2)
    - brak inf: puste zbiory i różne długości → skończona kara
    """
    c1 = _pair_cost(S, T, M, thr, max_match)
    c2 = _pair_cost(T, S, M, thr, max_match)
    return 0.5 * (c1 + c2)


# ──────────────────────────────────────────────────────────────────────────────
# 3) Φ / Ψ
# ──────────────────────────────────────────────────────────────────────────────

def phi_region_for(label: str, M: Mosaic, thr: float) -> str:
    if label in ("Call", "Expr"):                        return "edges"
    if label in ("Assign",):                             return "~edges"
    if label in ("If", "For", "While", "With", "Return"): return "all"
    if label in ("FunctionDef", "ClassDef"):             return "roi"
    return "~edges"


def phi_region_for_balanced(label: str, M: Mosaic, thr: float) -> str:
    """Wersja stabilna: używa kwantyli Q25/Q75 zamiast jednego p_edge."""
    q25, q75 = np.quantile(M.edge, [0.25, 0.75])
    if label in ("Call", "Expr"):
        return "edges" if q75 >= thr else "~edges"
    if label in ("Assign",):
        return "~edges" if q25 <= thr else "edges"
    if label in ("FunctionDef", "ClassDef"):
        return "roi"
    return "all"


def phi_region_for_entropy(label: str, M: Mosaic, thr: float) -> str:
    def near_thr(x): return abs(x - thr) <= 0.05

    fuzzy = float(np.mean([near_thr(v) for v in M.edge])) > 0.25
    if fuzzy: return "all"
    return phi_region_for(label, M, thr)


Selector = Callable[[str, Mosaic, float], str]


def centroid(ids: List[int], M: Mosaic) -> Tuple[float, float]:
    if not ids:
        return (M.cols * 0.5, M.rows * 0.5)
    xs = np.array([_xy_of_idx(i, M)[0] for i in ids], float)
    ys = np.array([_xy_of_idx(i, M)[1] for i in ids], float)
    return float(xs.mean()), float(ys.mean())


def phi_cost(ast: AstSummary, M: Mosaic, thr: float, selector: Selector) -> Tuple[float, Dict[int, Dict]]:
    """Średni koszt rozdziału regionów Φ (niżej = lepiej)."""
    details: Dict[int, Dict] = {}
    total = 0.0
    for n in ast.nodes.values():
        kind = selector(n.label, M, thr)
        ids = region_ids(M, kind, thr)
        if not ids:
            details[n.id] = dict(kind=kind, cost=0.0, centroid=(None, None))
            continue
        alt_kind = "edges" if kind == "~edges" else "~edges"
        alt = region_ids(M, alt_kind, thr)
        if not alt: alt = ids[:]  # unik
        alt = alt[:len(ids)] if len(alt) >= len(ids) else (alt + ids[:len(ids) - len(alt)])
        cost = max(0.0, D_M(ids, alt, M, thr))
        details[n.id] = dict(kind=kind, cost=cost, centroid=centroid(ids, M))
        total += cost
    N = max(1, len(ast.nodes))
    return total / float(N), details


def psi_feedback(ast: AstSummary, M: Mosaic, delta: float, thr: float) -> AstSummary:
    """Miękki update meta-wektorów węzłów na bazie cech regionu."""
    if delta <= 1e-9: return ast
    nodes = ast.nodes
    for n in nodes.values():
        kind = phi_region_for(n.label, M, thr)
        ids = region_ids(M, kind, thr)
        if not ids: continue
        ed = np.array([M.edge[i] for i in ids], float)
        psi = np.array([
            float(1.0 - ed.mean()),  # L
            float(0.5 + 0.5 * ed.std()),  # S
            float(min(1.0, 0.5 + ed.mean())),  # Sel
            float(1.0 - ed.std()),  # Stab
            float(min(1.0, 0.3 + 0.7 * ed.mean())),  # Cau
            float(0.4 + 0.5 * ed.std())  # H
        ], dtype=float)
        n.meta = (1.0 - delta) * n.meta + delta * psi
    S, H, Z = ast.S, ast.H, ast.Z
    tot = max(1, S + H)
    return AstSummary(S, H, Z, ast.maxZ, S / tot, H / tot, nodes, ast.labels)


# ──────────────────────────────────────────────────────────────────────────────
# 3b) Sprzężenie Ψ→(α,β) – wpływ Δ na Align bez naruszania I1
# ──────────────────────────────────────────────────────────────────────────────

def couple_alpha_beta(ast: AstSummary, M: Mosaic, thr: float,
                      delta: float, kappa_ab: float = KAPPA_AB_DEFAULT) -> AstSummary:
    """
    Sprzężenie α/β z profilem mozaiki: blend α/β z (aM,bM) proporcjonalnie do delta i globalnej
    „entropii krawędzi”. Utrzymuje I1 (α+β=1). S/H/Z pozostają bez zmian.
    """
    if delta <= 1e-9 or kappa_ab <= 1e-9:
        return ast
    # profil mozaiki
    _, _, _, aM, bM = mosaic_profile(M, thr)
    # globalna niepewność (0..1): użyj znormalizowanego std krawędzi
    ed = np.asarray(M.edge, dtype=float)
    uncert = float(min(1.0, max(0.0, ed.std())))  # ~[0,0.3..0.5]; ograniczamy do [0,1]
    w = float(min(1.0, max(0.0, kappa_ab * delta * (0.5 + 0.5 * uncert))))
    alpha_new = (1 - w) * ast.alpha + w * aM
    beta_new = 1.0 - alpha_new
    return AstSummary(ast.S, ast.H, ast.Z, ast.maxZ, alpha_new, beta_new, ast.nodes, ast.labels)


# ──────────────────────────────────────────────────────────────────────────────
# 4) Metryki, Align, inwarianty
# ──────────────────────────────────────────────────────────────────────────────

def mosaic_profile(M: Mosaic, thr: float) -> Tuple[int, int, int, float, float]:
    S = M.rows + M.cols
    H = int(np.sum(M.edge > thr))
    Z = 1
    tot = max(1, S + H)
    return S, H, Z, S / tot, H / tot


def distance_ast_mosaic(ast: AstSummary, M: Mosaic, thr: float, w=W_DEFAULT) -> float:
    _, _, _, aM, bM = mosaic_profile(M, thr)
    return (w['wS'] * abs(ast.alpha - aM) +
            w['wH'] * abs(ast.beta - bM) +
            w['wZ'] * abs(ast.Z / max(1, ast.maxZ) - 0.0))


def invariants_check(astA: AstSummary, astB: AstSummary, M: Mosaic, thr: float) -> Dict[str, bool]:
    """I1: α+β=1; I2: własności D_M; I3: kompresja nie zwiększa S+H+Z."""
    I1A = abs(astA.alpha + astA.beta - 1.0) < 1e-9
    I1B = abs(astB.alpha + astB.beta - 1.0) < 1e-9
    roi = region_ids(M, "roi", thr)
    top = region_ids(M, "all", thr)
    I2a = D_M(roi, roi, M, thr) == 0.0
    I2b = D_M(roi, top[:len(roi)], M, thr) >= 0.0
    I2c = abs(D_M(roi, top[:len(roi)], M, thr) - D_M(top[:len(roi)], roi, M, thr)) < 1e-9
    sumA = astA.S + astA.H + max(1, astA.Z)
    sumB = astB.S + astB.H + max(1, astB.Z)
    I3 = sumB <= sumA
    return {
        "I1_alpha_plus_beta_eq_1": (I1A and I1B),
        "I2_metric_nonneg_sym_identity": (I2a and I2b and I2c),
        "I3_compression_monotone": I3
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5) Pojedynczy przebieg, sweep, test znaku
# ──────────────────────────────────────────────────────────────────────────────

def run_once(lam: float, delta: float, rows: int, cols: int, thr: float,
             mosaic_kind: str = "grid",
             kappa_ab: float = KAPPA_AB_DEFAULT) -> Dict[str, float]:
    ast_raw = ast_deltas(EXAMPLE_SRC)
    ast_l = compress_ast(ast_raw, lam)
    M = build_mosaic(rows, cols, seed=7, kind=mosaic_kind, edge_thr=thr)
    # Φ – trzy warianty
    J1, _ = phi_cost(ast_l, M, thr, selector=phi_region_for)
    J2, _ = phi_cost(ast_l, M, thr, selector=phi_region_for_balanced)
    J3, _ = phi_cost(ast_l, M, thr, selector=phi_region_for_entropy)
    # Ψ i sprzężenie z (α,β)
    ast_after = psi_feedback(ast_l, M, delta, thr)
    ast_cpl = couple_alpha_beta(ast_after, M, thr, delta=delta, kappa_ab=kappa_ab)
    Align = 1.0 - min(1.0, distance_ast_mosaic(ast_cpl, M, thr))
    p_edge = float(np.mean(M.edge > thr))
    CR_AST = (ast_raw.S + ast_raw.H + max(1, ast_raw.Z)) / max(1, ast_l.S + ast_l.H + max(1, ast_l.Z))
    CR_TO = (1.0 / max(1e-6, min(p_edge, 1 - p_edge))) - 1.0
    return dict(J_phi1=J1, J_phi2=J2, J_phi3=J3, Align=Align, CR_AST=CR_AST, CR_TO=CR_TO,
                S=ast_l.S, H=ast_l.H, Z=ast_l.Z, alpha=ast_cpl.alpha, beta=ast_cpl.beta)


def sweep(rows: int, cols: int, thr: float, mosaic_kind: str = "grid",
          kappa_ab: float = KAPPA_AB_DEFAULT) -> List[Dict[str, float]]:
    lams = [0.0, 0.25, 0.5, 0.75]
    dels = [0.0, 0.25, 0.5]
    out = []
    for lam in lams:
        for de in dels:
            r = run_once(lam, de, rows, cols, thr, mosaic_kind, kappa_ab=kappa_ab)
            r.update(dict(lambda_=lam, delta_=de))
            out.append(r)
    return out


def sign_test_phi2_better(n_runs: int, rows: int, cols: int, thr: float,
                          lam: float, mosaic_kind: str = "grid") -> Dict[str, float]:
    wins = losses = ties = 0
    diffs: List[float] = []
    for seed in range(n_runs):
        M = build_mosaic(rows, cols, seed=seed, kind=mosaic_kind, edge_thr=thr)
        aL = compress_ast(ast_deltas(EXAMPLE_SRC), lam)
        J1, _ = phi_cost(aL, M, thr, selector=phi_region_for)
        J2, _ = phi_cost(aL, M, thr, selector=phi_region_for_balanced)
        d = J1 - J2
        diffs.append(d)
        if d > 0:
            wins += 1
        elif d < 0:
            losses += 1
        else:
            ties += 1
    n_eff = wins + losses
    if n_eff == 0:
        p_sign = 1.0
    else:
        from math import comb
        k = max(wins, losses)
        p_sign = sum(comb(n_eff, t) for t in range(k, n_eff + 1)) / (2 ** n_eff)
    return dict(wins=wins, losses=losses, ties=ties,
                mean_diff=float(np.mean(diffs)), median_diff=float(np.median(diffs)),
                p_sign=float(p_sign))


# ──────────────────────────────────────────────────────────────────────────────
# 6) CLI
# ──────────────────────────────────────────────────────────────────────────────

def _pretty_table(rows: List[Dict[str, float]]) -> str:
    header = ["λ", "Δ", "Align", "J_phi1", "J_phi2", "J_phi3", "CR_AST", "CR_TO", "α", "β", "S", "H", "Z"]
    widths = [4, 4, 7, 8, 8, 8, 8, 8, 5, 5, 4, 4, 3]

    def rowfmt(r):
        return [
            f"{r['lambda_']:.2f}", f"{r['delta_']:.2f}",
            f"{r['Align']:.3f}",
            f"{r['J_phi1']:.4f}", f"{r['J_phi2']:.4f}", f"{r['J_phi3']:.4f}",
            f"{r['CR_AST']:.3f}", f"{r['CR_TO']:.3f}",
            f"{r['alpha']:.2f}", f"{r['beta']:.2f}",
            int(r['S']), int(r['H']), int(r['Z'])
        ]

    def line(cols): return " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))

    s = []
    s.append(line(header))
    s.append("-" * (sum(widths) + len(widths) - 1))
    for r in rows: s.append(line(rowfmt(r)))
    return "\n".join(s)


def cmd_run(args):
    lam = args.lmbd
    de = args.delta
    r = args.rows
    c = args.cols
    thr = args.edge_thr
    kind = args.mosaic
    res = run_once(lam, de, r, c, thr, mosaic_kind=kind, kappa_ab=args.kappa_ab)
    print(json.dumps({
        "lambda": lam,
        "delta": de,
        "rows": r,
        "cols": c,
        "kind": kind,
        "edge_thr": thr,
        **res
    }, indent=2))


def cmd_sweep(args):
    rows = sweep(args.rows, args.cols, args.edge_thr, mosaic_kind=args.mosaic, kappa_ab=args.kappa_ab)
    print(_pretty_table(rows))
    if args.json:
        print("\n[JSON]")
        print(json.dumps(rows, indent=2))


def cmd_test(args):
    inv_astA = ast_deltas(EXAMPLE_SRC)
    inv_astB = compress_ast(inv_astA, args.lmbd)
    M = build_mosaic(args.rows, args.cols, seed=7, kind=args.mosaic, edge_thr=args.edge_thr)
    inv = invariants_check(inv_astA, inv_astB, M, args.edge_thr)
    print("[INVARIANTS]")
    for k, v in inv.items(): print(f"  - {k}: {'PASS' if v else 'FAIL'}")
    print("\n[SIGN TEST Φ2 > Φ1]")
    sign = sign_test_phi2_better(args.runs, args.rows, args.cols, args.edge_thr,
                                 lam=args.lmbd, mosaic_kind=args.mosaic)
    print(json.dumps(sign, indent=2))


def build_cli():
    p = argparse.ArgumentParser(prog="hybrid_ast_mosaic",
                                description="Hybryda AST ⇄ Mozaika (Φ/Ψ) – metryki, sweep, inwarianty")
    p.add_argument("--mosaic", choices=["grid", "hex"], default="grid", help="rodzaj mozaiki")
    p.add_argument("--rows", type=int, default=6, help="liczba wierszy mozaiki")
    p.add_argument("--cols", type=int, default=6, help="liczba kolumn mozaiki")
    p.add_argument("--edge-thr", type=float, default=EDGE_THR_DEFAULT, help="próg edge dla regionów")
    p.add_argument("--kappa-ab", type=float, default=KAPPA_AB_DEFAULT, help="siła sprzężenia Ψ→(α,β)")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("run", help="pojedynczy przebieg (λ, Δ)")
    q.add_argument("--lmbd", type=float, default=0.60)
    q.add_argument("--delta", type=float, default=0.25)
    q.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="siatka λ×Δ i tabela metryk")
    s.add_argument("--json", action="store_true", help="dodatkowo wypisz JSON")
    s.set_defaults(func=cmd_sweep)

    t = sub.add_parser("test", help="inwarianty + test znaku Φ2 vs Φ1 (wiele seedów)")
    t.add_argument("--lmbd", type=float, default=0.60)
    t.add_argument("--runs", type=int, default=100)
    t.set_defaults(func=cmd_test)
    return p


def main():
    args = build_cli().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
