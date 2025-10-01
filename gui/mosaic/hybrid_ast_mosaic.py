# glitchlab/mosaic/hybrid_ast_mosaic.py
# Hybrydowy algorytm AST ⇄ Mozaika (Φ/Ψ), ΔS/ΔH/ΔZ, λ-kompresja,
# warianty Φ, Ψ-feedback, metryki, inwarianty, sweep λ×Δ, CLI, + from-git.
# Python 3.9+  (deps: numpy; stdlib: ast, math, json, argparse, itertools, hashlib)

from __future__ import annotations
import ast
import math
import json
import argparse
import itertools
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Callable, Iterable
import numpy as np
import os
from pathlib import Path

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
    "make_selector_policy_aware", "load_policy_json",
    "phi_cost", "psi_feedback",
    # Coupling
    "couple_alpha_beta",
    # Metrics / Tests
    "mosaic_profile", "distance_ast_mosaic", "invariants_check",
    # Runs
    "run_once", "sweep", "sign_test_phi2_better",
    # FROM-GIT
    "run_from_git",
    # CLI
    "build_cli", "main",
    # Example & params
    "EXAMPLE_SRC", "EDGE_THR_DEFAULT", "W_DEFAULT",
    "SOFT_LABELS_DEFAULT", "SOFT_LABELS_TAU", "KAPPA_AB_DEFAULT",
]

# ──────────────────────────────────────────────────────────────────────────────
# Importy warstwy analizy (mogą nie istnieć na wczesnym etapie – fallbacki)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from glitchlab.analysis.git_io import (
        repo_root, git_merge_base, changed_py_files, show_file_at_rev
    )
except Exception:
    repo_root = None
    git_merge_base = None
    changed_py_files = None
    show_file_at_rev = None

try:
    # zakładamy zgodny interfejs: ast_summary_of_source(src, file_path) -> AstSummary-kompatybilny
    from glitchlab.analysis.ast_index import ast_summary_of_source as idx_ast_summary_of_source
except Exception:
    idx_ast_summary_of_source = None

try:
    # AstDelta(dS,dH,dZ, per_label, mapy) – traktujemy jako struktura/dict kompatybilna
    from glitchlab.analysis.ast_delta import ast_delta as idx_ast_delta
except Exception:
    idx_ast_delta = None

try:
    from glitchlab.analysis.impact import impact_zone
except Exception:
    impact_zone = None

# artefakty raportowe (opcjonalnie — miękki fallback)
try:
    from glitchlab.analysis.reporting import emit_artifacts as _emit_artifacts
except Exception:
    _emit_artifacts = None  # pragma: no cover

# ──────────────────────────────────────────────────────────────────────────────
# PRZYKŁADOWE ŹRÓDŁO (dla trybów demo/run/sweep/test)
# ──────────────────────────────────────────────────────────────────────────────

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
# 1) AST → ΔS/ΔH/ΔZ  + λ-kompresja   (lokalny model + fallback)
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
    """
    Deterministyczna meta oparta o etykietę i głębokość (bez losowości czasu wykonania).
    Uwaga: w trybie FROM-GIT preferujemy ast_index.ast_summary_of_source, a to jest fallback.
    """
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
    """Parsuje źródło Pythona i liczy przybliżone ΔS/ΔH/ΔZ wg prostych reguł (fallback)."""
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

# Ujednolicone API selektora (rozszerzalne)
# Selector = Callable[[AstNode, Dict, Mosaic, float], str]
# Na razie korzystamy z etykiety – kompatybilnie z wersjami label-only.
Selector = Callable[[str, Mosaic, float], str]


def phi_region_for(label: str, M: Mosaic, thr: float) -> str:
    if label in ("Call", "Expr"):                        return "edges"
    if label in ("Assign",):                             return "~edges"
    if label in ("If", "For", "While", "With", "Return"): return "all"
    if label in ("FunctionDef", "ClassDef"):             return "roi"
    return "~edges"


def phi_region_for_balanced(label: str, M: Mosaic, thr: float) -> str:
    """
    Wersja stabilna: używa kwantyli Q25/Q75 zamiast pojedynczego p_edge.
    Odporna na puste/niepoprawne dane i NaN-y w M.edge.
    """
    # Bezpieczne pobranie i sanity danych krawędzi
    try:
        edge = np.asarray(getattr(M, "edge", []), dtype=float).reshape(-1)
    except Exception:
        edge = np.asarray([], dtype=float)

    if edge.size == 0:
        # Brak informacji o mozaice → fallback wg etykiety, ale bez biasu na edges/~edges
        if label in ("FunctionDef", "ClassDef"):
            return "roi"
        if label in ("If", "For", "While", "With", "Return"):
            return "all"
        return "~edges"

    # Wyczyść NaN/inf i ogranicz do [0,1] (profil edge jest w tym zakresie)
    edge = np.nan_to_num(edge, nan=0.0, posinf=1.0, neginf=0.0)
    edge = np.clip(edge, 0.0, 1.0)

    # Kwantyle odporne na rozkład
    q25, q75 = np.quantile(edge, [0.25, 0.75])

    # Stabilizacja progu
    thr = float(np.clip(thr, 0.0, 1.0))

    if label in ("Call", "Expr"):
        # aktywne węzły trafiają na "edges" tylko gdy górny kwartyl faktycznie przekracza próg
        return "edges" if q75 >= thr else "~edges"
    if label in ("Assign",):
        # stabilizujące węzły preferują "~edges", jeśli dolny kwartyl nie przekracza progu
        return "~edges" if q25 <= thr else "edges"
    if label in ("FunctionDef", "ClassDef"):
        return "roi"
    if label in ("If", "For", "While", "With", "Return"):
        return "all"
    return "~edges"


def phi_region_for_entropy(label: str, M: Mosaic, thr: float) -> str:
    def near_thr(x): return abs(x - thr) <= 0.05

    fuzzy = float(np.mean([near_thr(v) for v in M.edge])) > 0.25
    if fuzzy: return "all"
    return phi_region_for(label, M, thr)


# ──────────────────────────────────────────────────────────────────────────────
# (NEW) Policy-aware Φ
# ──────────────────────────────────────────────────────────────────────────────

def load_policy_json(path: Optional[str]) -> Dict:
    """
    Ładuje polityki z JSON (lekka walidacja). Brak/None → pusty słownik.
    Przykład kluczy (opcjonalne):
      - avoid_roi_side_effects: bool (default True)
      - prefer_edges: bool
      - prefer_stability: bool
      - roi_labels: list[str]
      - edge_thr_bias: float (dodawane do thr przy decyzji)
    """
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        # sanity podstawowych typów
        out: Dict = {}
        if "avoid_roi_side_effects" in data:
            out["avoid_roi_side_effects"] = bool(data["avoid_roi_side_effects"])
        if "prefer_edges" in data:
            out["prefer_edges"] = bool(data["prefer_edges"])
        if "prefer_stability" in data:
            out["prefer_stability"] = bool(data["prefer_stability"])
        if "roi_labels" in data and isinstance(data["roi_labels"], list):
            out["roi_labels"] = [str(x) for x in data["roi_labels"]]
        if "edge_thr_bias" in data:
            try:
                out["edge_thr_bias"] = float(data["edge_thr_bias"])
            except Exception:
                pass
        return out
    except Exception:
        return {}


def make_selector_policy_aware(policy: Optional[Dict] = None) -> Selector:
    """
    Buduje selektor Φ zależny od prostych polityk.
    Pipeline decyzyjny:
      1) hard override: jeśli label w roi_labels → "roi"
      2) bazowa decyzja: phi_region_for_balanced przy progu (thr + edge_thr_bias)
      3) miękkie preferencje:
         - prefer_edges      → Call/Expr skłaniają do "edges"
         - prefer_stability  → Assign/Return skłaniają do "~edges"
      4) avoid_roi_side_effects (domyślnie True): Call/Expr nigdy nie trafiają do "roi"
    """
    P = policy or {}
    roi_labels = set(P.get("roi_labels", []) or [])
    prefer_edges = bool(P.get("prefer_edges", False))
    prefer_stability = bool(P.get("prefer_stability", False))
    avoid_roi_side_effects = P.get("avoid_roi_side_effects", True)
    thr_bias = float(P.get("edge_thr_bias", 0.0) or 0.0)

    def _sel(label: str, M: Mosaic, thr: float) -> str:
        # 1) hard override
        if label in roi_labels:
            return "roi"

        # 2) bazowa decyzja (z biasem progu)
        base_kind = phi_region_for_balanced(label, M, float(thr + thr_bias))

        # 3) miękkie preferencje
        if prefer_edges and label in ("Call", "Expr"):
            base_kind = "edges"
        if prefer_stability and label in ("Assign", "Return"):
            base_kind = "~edges"

        # 4) unikaj I/O w ROI (bezpiecznik; balanced i tak nie przypisuje Call do ROI)
        if avoid_roi_side_effects and label in ("Call", "Expr") and base_kind == "roi":
            base_kind = "edges"

        return base_kind

    return _sel


# ──────────────────────────────────────────────────────────────────────────────
# Centroid & koszt Φ
# ──────────────────────────────────────────────────────────────────────────────

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
# (NEW) Adiacencja kafelków & komponenty spójne – grid/hex (odd-r)
# ──────────────────────────────────────────────────────────────────────────────

def _neighbors_grid(i: int, rows: int, cols: int) -> List[int]:
    r, c = divmod(i, cols)
    nbrs = []
    if r > 0:         nbrs.append((r - 1) * cols + c)  # up
    if r + 1 < rows:  nbrs.append((r + 1) * cols + c)  # down
    if c > 0:         nbrs.append(r * cols + (c - 1))  # left
    if c + 1 < cols:  nbrs.append(r * cols + (c + 1))  # right
    return nbrs


def _neighbors_hex_oddr(i: int, rows: int, cols: int) -> List[int]:
    # heksy w układzie odd-r offset (tak jak w build_mosaic_hex)
    r, c = divmod(i, cols)
    if r % 2 == 0:
        candidates = [(r - 1, c - 1), (r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c - 1), (r + 1, c)]
    else:
        candidates = [(r - 1, c), (r - 1, c + 1), (r, c - 1), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
    nbrs = []
    for rr, cc in candidates:
        if 0 <= rr < rows and 0 <= cc < cols:
            nbrs.append(rr * cols + cc)
    return nbrs


def _adjacency_indices(M: "Mosaic") -> List[List[int]]:
    """Zwraca listę sąsiadów dla każdego kafla zgodnie z topologią mozaiki."""
    rows, cols = int(M.rows), int(M.cols)
    N = rows * cols
    if N <= 0:
        return []
    adj: List[List[int]] = [[] for _ in range(N)]
    nbr_fn = _neighbors_hex_oddr if getattr(M, "kind", "grid") == "hex" else _neighbors_grid
    for i in range(N):
        adj[i] = nbr_fn(i, rows, cols)
    return adj


def _connected_components(mask: np.ndarray, M: "Mosaic") -> int:
    """
    Liczba spójnych komponentów dla kafli, gdzie mask[i]==True.
    Spójność wg adiacencji siatki (grid/hex).
    """
    rows, cols = int(M.rows), int(M.cols)
    N = rows * cols
    if N == 0 or mask.size == 0:
        return 0
    mask = mask.astype(bool).reshape(N)
    if not np.any(mask):
        return 0
    adj = _adjacency_indices(M)
    seen = np.zeros(N, dtype=bool)
    comps = 0
    for i in range(N):
        if not mask[i] or seen[i]:
            continue
        comps += 1
        q = [i]
        seen[i] = True
        while q:
            u = q.pop()
            for v in adj[u]:
                if mask[v] and not seen[v]:
                    seen[v] = True
                    q.append(v)
    return comps


# ──────────────────────────────────────────────────────────────────────────────
# (REPLACED) Profil mozaiki – wersja produkcyjna
# ──────────────────────────────────────────────────────────────────────────────

def mosaic_profile(M: "Mosaic", thr: float) -> Tuple[int, int, int, float, float]:
    """
    Produkcyjny profil mozaiki:
      S  – skala/rozmiar struktury (rows + cols), kompatybilna skalą z H,
      H  – liczba kafli o edge > thr,
      Z  – liczba spójnych komponentów w masce (edge > thr) wg topologii mozaiki,
      α  – S / (S + H)   (waga „logiki/struktury”),
      β  – H / (S + H)   (waga „materii/energii”).
    Zwraca wartości deterministyczne, odporne na skraje i NaN.
    """
    rows = int(getattr(M, "rows", 0))
    cols = int(getattr(M, "cols", 0))
    if rows < 0: rows = 0
    if cols < 0: cols = 0
    N = rows * cols

    edge = np.asarray(getattr(M, "edge", np.empty(0)), dtype=float).reshape(-1)
    if edge.size != N:
        try:
            edge = edge.reshape(N)
        except Exception:
            edge = np.zeros(N, dtype=float)

    if np.any(~np.isfinite(edge)):
        edge = np.nan_to_num(edge, nan=0.0, posinf=1.0, neginf=0.0)

    S = rows + cols
    H = int(np.count_nonzero(edge > float(thr))) if N > 0 else 0
    mask = (edge > float(thr)) if N > 0 else np.zeros(0, dtype=bool)
    Z = int(_connected_components(mask, M)) if N > 0 else 0

    tot = max(1, S + H)
    alpha = float(S) / float(tot)
    beta = float(H) / float(tot)
    return int(S), int(H), int(Z), alpha, beta


def couple_alpha_beta(ast: AstSummary, M: Mosaic, thr: float,
                      delta: float, kappa_ab: float = KAPPA_AB_DEFAULT) -> AstSummary:
    """
    Sprzężenie α/β z profilem mozaiki: blend α/β z (aM,bM) proporcjonalnie do delta i globalnej
    „entropii krawędzi”. Utrzymuje I1 (α+β=1). S/H/Z pozostają bez zmian.
    """
    if delta <= 1e-9 or kappa_ab <= 1e-9:
        return ast
    _, _, _, aM, bM = mosaic_profile(M, thr)
    ed = np.asarray(M.edge, dtype=float)
    uncert = float(min(1.0, max(0.0, ed.std())))
    w = float(min(1.0, max(0.0, kappa_ab * delta * (0.5 + 0.5 * uncert))))
    alpha_new = (1 - w) * ast.alpha + w * aM
    beta_new = 1.0 - alpha_new
    return AstSummary(ast.S, ast.H, ast.Z, ast.maxZ, alpha_new, beta_new, ast.nodes, ast.labels)


# ──────────────────────────────────────────────────────────────────────────────
# 4) Metryki, Align, inwarianty
# ──────────────────────────────────────────────────────────────────────────────

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
# 5) Pojedynczy przebieg, sweep, test znaku (demo na EXAMPLE_SRC)
# ──────────────────────────────────────────────────────────────────────────────

def _selector_from_args(phi_name: str, policy_file: Optional[str]) -> Tuple[str, Selector, Optional[Dict]]:
    phi_name = (phi_name or "balanced").lower()
    policy = load_policy_json(policy_file) if phi_name == "policy" else None
    if phi_name == "basic":
        return "basic", phi_region_for, None
    if phi_name == "entropy":
        return "entropy", phi_region_for_entropy, None
    if phi_name == "policy":
        return "policy", make_selector_policy_aware(policy), policy or {}
    # default
    return "balanced", phi_region_for_balanced, None


def run_once(lam: float, delta: float, rows: int, cols: int, thr: float,
             mosaic_kind: str = "grid",
             kappa_ab: float = KAPPA_AB_DEFAULT,
             phi_name: str = "balanced",
             policy_file: Optional[str] = None) -> Dict[str, float]:
    ast_raw = ast_deltas(EXAMPLE_SRC)
    ast_l = compress_ast(ast_raw, lam)
    M = build_mosaic(rows, cols, seed=7, kind=mosaic_kind, edge_thr=thr)
    # Φ – bazowe warianty
    J1, _ = phi_cost(ast_l, M, thr, selector=phi_region_for)
    J2, _ = phi_cost(ast_l, M, thr, selector=phi_region_for_balanced)
    J3, _ = phi_cost(ast_l, M, thr, selector=phi_region_for_entropy)

    # Φ – policy (opcjonalnie)
    sel_label, selector, policy = _selector_from_args(phi_name, policy_file)
    Jp = None
    if sel_label == "policy":
        Jp, _ = phi_cost(ast_l, M, thr, selector=selector)

    # Ψ i sprzężenie z (α,β)
    ast_after = psi_feedback(ast_l, M, delta, thr)
    ast_cpl = couple_alpha_beta(ast_after, M, thr, delta=delta, kappa_ab=kappa_ab)
    Align = 1.0 - min(1.0, distance_ast_mosaic(ast_cpl, M, thr))
    p_edge = float(np.mean(M.edge > thr))
    CR_AST = (ast_raw.S + ast_raw.H + max(1, ast_raw.Z)) / max(1, ast_l.S + ast_l.H + max(1, ast_l.Z))
    CR_TO = (1.0 / max(1e-6, min(p_edge, 1 - p_edge))) - 1.0

    res = dict(J_phi1=J1, J_phi2=J2, J_phi3=J3, Align=Align, CR_AST=CR_AST, CR_TO=CR_TO,
               S=ast_l.S, H=ast_l.H, Z=ast_l.Z, alpha=ast_cpl.alpha, beta=ast_cpl.beta)
    if Jp is not None:
        res["J_phiP"] = Jp
    return res


def sweep(rows: int, cols: int, thr: float, mosaic_kind: str = "grid",
          kappa_ab: float = KAPPA_AB_DEFAULT,
          phi_name: str = "balanced",
          policy_file: Optional[str] = None) -> List[Dict[str, float]]:
    lams = [0.0, 0.25, 0.5, 0.75]
    dels = [0.0, 0.25, 0.5]
    out = []
    for lam in lams:
        for de in dels:
            r = run_once(lam, de, rows, cols, thr, mosaic_kind,
                         kappa_ab=kappa_ab, phi_name=phi_name, policy_file=policy_file)
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
# 5b) FROM-GIT: realne BASE..HEAD, Δ, Φ/Ψ, wynik JSON (stdout)
# ──────────────────────────────────────────────────────────────────────────────

def _ast_summary_for_source_det(src: str, file_path: str) -> AstSummary:
    """
    Deterministyczny AstSummary z preferencją na glitchlab.analysis.ast_index, z fallbackiem
    do lokalnego ast_deltas (z deterministyczną metą).
    """
    if idx_ast_summary_of_source is not None:
        try:
            s = idx_ast_summary_of_source(src, file_path=file_path)
            if isinstance(s, AstSummary):
                return s
            if isinstance(s, dict):
                return AstSummary(
                    S=int(s.get("S", 0)),
                    H=int(s.get("H", 0)),
                    Z=int(s.get("Z", 0)),
                    maxZ=int(s.get("maxZ", max(1, int(s.get("Z", 0))))),
                    alpha=float(s.get("alpha", 0.5)),
                    beta=float(s.get("beta", 0.5)),
                    nodes=s.get("nodes", {}) or {},
                    labels=s.get("labels", []) or []
                )
        except Exception:
            pass
    return ast_deltas(src)


def _file_selector_filter(paths_like: Optional[List[str]], rel_path: str) -> bool:
    if not paths_like:
        return True
    rel = rel_path.replace("\\", "/")
    for p in paths_like:
        if p in rel or rel.endswith(p):
            return True
    return False


def run_from_git(
        base: str,
        head: str,
        rows: int = 6,
        cols: int = 6,
        thr: float = EDGE_THR_DEFAULT,
        mosaic_kind: str = "grid",
        delta: float = 0.25,
        kappa_ab: float = KAPPA_AB_DEFAULT,
        paths: Optional[List[str]] = None,
        phi_name: str = "balanced",
        policy_file: Optional[str] = None,
) -> Dict:
    """
    Spina: Git diff (BASE..HEAD) → pliki .py → AstSummary(base/head) → Δ, Φ/Ψ → JSON (stdout).
    Nie zapisuje artefaktów na dysk (to zrobimy w module reporting).
    """
    if changed_py_files is None or show_file_at_rev is None:
        raise RuntimeError("analysis.git_io nie jest dostępny – zainstaluj/udostępnij moduły analizy.")

    root = None
    try:
        root = repo_root() if repo_root else None
    except Exception:
        root = None

    changed = changed_py_files(base, head)
    changed = [p for p in changed if _file_selector_filter(paths, p)]

    # Budujemy jedną mozaikę na commit (lekko, domyślnie 6x6)
    M = build_mosaic(rows, cols, seed=7, kind=mosaic_kind, edge_thr=thr)

    # Selector wg CLI
    sel_label, selector, policy = _selector_from_args(phi_name, policy_file)

    files_out: List[Dict] = []
    totals = dict(dS=0, dH=0, dZ=0, align_gain=0.0)

    for rel_path in changed:
        base_src = show_file_at_rev(rel_path, base) or ""
        head_src = show_file_at_rev(rel_path, head) or ""

        a_base = _ast_summary_for_source_det(base_src, rel_path)
        a_head = _ast_summary_for_source_det(head_src, rel_path)

        # Δ (best-effort)
        if idx_ast_delta is not None:
            try:
                d = idx_ast_delta(a_base, a_head)
                if isinstance(d, dict):
                    dS = int(d.get("dS", 0));
                    dH = int(d.get("dH", 0));
                    dZ = int(d.get("dZ", 0))
                elif isinstance(d, (tuple, list)) and len(d) >= 3:
                    dS, dH, dZ = map(int, d[:3])
                else:
                    dS = int(getattr(d, "dS", 0));
                    dH = int(getattr(d, "dH", 0));
                    dZ = int(getattr(d, "dZ", 0))
            except Exception:
                dS = a_head.S - a_base.S;
                dH = a_head.H - a_base.H;
                dZ = a_head.Z - a_base.Z
        else:
            dS = a_head.S - a_base.S;
            dH = a_head.H - a_base.H;
            dZ = a_head.Z - a_base.Z

        # Φ – bazowe warianty
        J1, _ = phi_cost(a_head, M, thr, selector=phi_region_for)
        J2, _ = phi_cost(a_head, M, thr, selector=phi_region_for_balanced)
        J3, _ = phi_cost(a_head, M, thr, selector=phi_region_for_entropy)

        # Φ – policy (opcjonalnie)
        Jp = None
        if sel_label == "policy":
            Jp, _ = phi_cost(a_head, M, thr, selector=selector)

        # Align przed/po Ψ
        Align_before = 1.0 - min(1.0, distance_ast_mosaic(a_head, M, thr))
        a_after = psi_feedback(a_head, M, delta, thr)
        a_cpl = couple_alpha_beta(a_after, M, thr, delta=delta, kappa_ab=kappa_ab)
        Align_after = 1.0 - min(1.0, distance_ast_mosaic(a_cpl, M, thr))

        # Impact-Zone (opcjonalnie)
        impact = None
        if impact_zone is not None:
            try:
                fake_delta = type("D", (), dict(dS=dS, dH=dH, dZ=dZ))
                impact = impact_zone(fake_delta, base_src, head_src, file_path=rel_path)
            except Exception:
                impact = None

        entry = {
            "path": rel_path,
            "delta": {"dS": dS, "dH": dH, "dZ": dZ},
            "phi": {"J1": J1, "J2": J2, "J3": J3},
            "align": {"before": round(Align_before, 6), "after": round(Align_after, 6)},
            "mosaic": {"kind": mosaic_kind, "thr": thr, "grid": {"rows": rows, "cols": cols}},
            "impact": impact or {},
        }
        if Jp is not None:
            entry["phi"]["J_policy"] = Jp
            entry["phi"]["policy_used"] = True
        files_out.append(entry)

        totals["dS"] += dS
        totals["dH"] += dH
        totals["dZ"] += dZ
        totals["align_gain"] += (Align_after - Align_before)

    out = {
        "base": base,
        "head": head,
        "branch": None,
        "repo_root": root,
        "mosaic": {"kind": mosaic_kind, "grid": {"rows": rows, "cols": cols}, "thr": thr},
        "delta": {"files": len(files_out)},
        "phi_selector": sel_label,
        "policy_file": policy_file if policy_file else None,
        "files": files_out,
        "totals": totals,
    }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 6) CLI
# ──────────────────────────────────────────────────────────────────────────────

def _pretty_table(rows: List[Dict[str, float]]) -> str:
    header = ["λ", "Δ", "Align", "J_phi1", "J_phi2", "J_phi3", "J_phiP", "CR_AST", "CR_TO", "α", "β", "S", "H", "Z"]
    widths = [4, 4, 7, 8, 8, 8, 8, 8, 8, 5, 5, 4, 4, 3]

    def rowfmt(r):
        return [
            f"{r['lambda_']:.2f}", f"{r['delta_']:.2f}",
            f"{r['Align']:.3f}",
            f"{r['J_phi1']:.4f}", f"{r['J_phi2']:.4f}", f"{r['J_phi3']:.4f}",
            f"{r.get('J_phiP', '-') if 'J_phiP' in r else '-'}",
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
    res = run_once(
        args.lmbd, args.delta, args.rows, args.cols, args.edge_thr,
        mosaic_kind=args.mosaic, kappa_ab=args.kappa_ab,
        phi_name=args.phi, policy_file=args.policy_file
    )
    out = {
        "lambda": args.lmbd,
        "delta": args.delta,
        "rows": args.rows,
        "cols": args.cols,
        "kind": args.mosaic,
        "edge_thr": args.edge_thr,
        "phi": args.phi,
        "policy_file": args.policy_file,
        **res
    }
    print(json.dumps(out, indent=2))


def cmd_sweep(args):
    rows = sweep(
        args.rows, args.cols, args.edge_thr,
        mosaic_kind=args.mosaic, kappa_ab=args.kappa_ab,
        phi_name=args.phi, policy_file=args.policy_file
    )
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


def _normpath_under_project(pth: str, project_root: Path) -> Path:
    p = Path(pth)
    return (project_root / p).resolve() if not p.is_absolute() else p.resolve()

def cmd_from_git_dump(args):
    res = run_from_git(
        base=args.base,
        head=args.head,
        rows=args.rows,
        cols=args.cols,
        thr=args.edge_thr,
        mosaic_kind=args.mosaic,
        delta=args.delta,
        kappa_ab=args.kappa_ab,
        paths=args.paths or None,
        phi_name=args.phi,
        policy_file=args.policy_file,
    )

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    errors: list[str] = []
    try:
        if _emit_artifacts is not None:
            try:
                artifacts = _emit_artifacts(res, outdir=outdir) or {}
            except Exception as e:
                errors.append(f"emit_artifacts failed: {e.__class__.__name__}: {e}")
                artifacts = {}
        else:
            errors.append("reporting.emit_artifacts not available")
    except Exception as e:
        errors.append(f"unexpected error: {e.__class__.__name__}: {e}")

    # sprawdź, czy zwrócone ścieżki faktycznie istnieją
    existing = {}
    missing = {}
    for k, v in (artifacts or {}).items():
        if not v:
            missing[k] = v
            continue
        p = _normpath_under_project(v, Path.cwd())
        if p.exists():
            existing[k] = str(p)
        else:
            missing[k] = str(p)

    ok = bool(existing)
    diagnostics = {
        "base": res.get("base"),
        "head": res.get("head"),
        "files_analyzed": res.get("delta", {}).get("files", 0),
        "missing_artifacts": missing,
        "notes": errors,
    }

    # Jeżeli nic nie powstało: wypisz ostrzeżenie na STDERR, przekaż ok:false w STDOUT
    if not ok:
        msg = "[GLX] WARN: no artifacts produced by from-git-dump; check --base/--head and reporting module."
        if diagnostics["files_analyzed"] == 0:
            msg += " (No changed .py files between BASE..HEAD.)"
        print(msg, file=sys.stderr)

        out_json = {"ok": False, "artifacts": {}, "diagnostics": diagnostics}
        print(json.dumps(out_json, ensure_ascii=False, indent=2))
        # RC zależnie od trybu
        strict_env = os.getenv("GLX_STRICT_ARTIFACTS", "0").strip() in ("1", "true", "yes")
        if getattr(args, "strict_artifacts", False) or strict_env:
            sys.exit(1)
        sys.exit(0)

    # Sukces: zwracamy tylko istniejące artefakty (reszta w diagnostyce)
    out_json = {"ok": True, "artifacts": existing, "diagnostics": diagnostics}
    print(json.dumps(out_json, ensure_ascii=False, indent=2))


def cmd_from_git(args):
    res = run_from_git(
        base=args.base,
        head=args.head,
        rows=args.rows,
        cols=args.cols,
        thr=args.edge_thr,
        mosaic_kind=args.mosaic,
        delta=args.delta,
        kappa_ab=args.kappa_ab,
        paths=args.paths or None,
        phi_name=args.phi,
        policy_file=args.policy_file,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


def build_cli():
    p = argparse.ArgumentParser(prog="hybrid_ast_mosaic",
                                description="Hybryda AST ⇄ Mozaika (Φ/Ψ) – metryki, sweep, inwarianty, from-git")
    p.add_argument("--mosaic", choices=["grid", "hex"], default="grid", help="rodzaj mozaiki")
    p.add_argument("--rows", type=int, default=6, help="liczba wierszy mozaiki")
    p.add_argument("--cols", type=int, default=6, help="liczba kolumn mozaiki")
    p.add_argument("--edge-thr", type=float, default=EDGE_THR_DEFAULT, help="próg edge dla regionów")
    p.add_argument("--kappa-ab", type=float, default=KAPPA_AB_DEFAULT, help="siła sprzężenia Ψ→(α,β)")
    # NOWE: wybór selektora Φ i pliku polityk
    p.add_argument("--phi", choices=["basic", "balanced", "entropy", "policy"], default="balanced",
                   help="wariant selektora Φ")
    p.add_argument("--policy-file", default=None, help="JSON z politykami dla --phi policy")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("run", help="pojedynczy przebieg (λ, Δ) – demo na EXAMPLE_SRC")
    q.add_argument("--lmbd", type=float, default=0.60)
    q.add_argument("--delta", type=float, default=0.25)
    q.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="siatka λ×Δ i tabela metryk (demo)")
    s.add_argument("--json", action="store_true", help="dodatkowo wypisz JSON")
    s.set_defaults(func=cmd_sweep)

    t = sub.add_parser("test", help="inwarianty + test znaku Φ2 vs Φ1 (wiele seedów, demo)")
    t.add_argument("--lmbd", type=float, default=0.60)
    t.add_argument("--runs", type=int, default=100)
    t.set_defaults(func=cmd_test)

    g = sub.add_parser("from-git", help="Analiza realnego diffu BASE..HEAD z repo")
    g.add_argument("--base", required=True, help="BASE sha/refs (np. merge-base, .glx/state.json)")
    g.add_argument("--head", default="HEAD", help="HEAD sha/ref (domyślnie HEAD)")
    g.add_argument("--delta", type=float, default=0.25, help="siła Ψ-feedback (0..1)")
    g.add_argument("--paths", nargs="*", help="opcjonalny filtr ścieżek (substring/endswith)")
    g.set_defaults(func=cmd_from_git)

    gd = sub.add_parser("from-git-dump",
                        help="Analiza BASE..HEAD i zapis artefaktów (report.json, summary.md, mosaic_map.json)")
    gd.add_argument("--base", required=True, help="BASE sha/refs (np. merge-base, .glx/state.json)")
    gd.add_argument("--head", default="HEAD", help="HEAD sha/ref (domyślnie HEAD)")
    gd.add_argument("--rows", type=int, default=6)
    gd.add_argument("--cols", type=int, default=6)
    gd.add_argument("--edge-thr", type=float, default=EDGE_THR_DEFAULT)
    gd.add_argument("--mosaic", choices=["grid", "hex"], default="grid")
    gd.add_argument("--delta", type=float, default=0.25)
    gd.add_argument("--kappa-ab", type=float, default=KAPPA_AB_DEFAULT)
    gd.add_argument("--paths", nargs="*")
    gd.add_argument("--out", default="analysis/last", help="katalog wyjściowy na artefakty")
    # współdzielone z top-level: --phi, --policy-file już są w parserze głównym
    # tryb surowy – jeśli artefakty nie powstały, zakończ rc = 1
    gd.add_argument("--strict-artifacts", action="store_true",
                    help="jeżeli artefakty nie powstaną, zakończ błędem (rc=1)")
    gd.set_defaults(func=cmd_from_git_dump)

    return p


def main():
    args = build_cli().parse_args()
    args.func(args)


if __name__ == "__main__":
    pass
#   main()
