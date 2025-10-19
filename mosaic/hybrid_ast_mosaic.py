# glitchlab/app/mosaic/hybrid_ast_mosaic.py
# Hybrydowy algorytm AST ⇄ Mozaika (Φ/Ψ), ΔS/ΔH/ΔZ, λ-kompresja,
# warianty Φ, Ψ-feedback, metryki, inwarianty, sweep λ×Δ, CLI, + from-git z obsługą .env.
# Python 3.9+  (deps: numpy; stdlib: ast, math, json, argparse, itertools, hashlib, os)

# ──[GLX:SECTION:1/3:CORE+ENV+ALIAS]────────────────────────────────────────────

from __future__ import annotations
import ast
import math
import json
import argparse
import itertools
import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Iterable, Any
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 0) STAŁE / PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

EDGE_THR_DEFAULT: float = 0.55
W_DEFAULT = dict(wS=1.0, wH=1.0, wZ=0.4)  # wagi Align
SOFT_LABELS_DEFAULT: bool = True
SOFT_LABELS_TAU: float = 0.08
KAPPA_AB_DEFAULT: float = 0.35

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
    "run_from_git", "cmd_from_git", "cmd_from_git_dump",
    # CLI
    "build_cli", "main",
    # Example & params
    "EXAMPLE_SRC", "EDGE_THR_DEFAULT", "W_DEFAULT",
    "SOFT_LABELS_DEFAULT", "SOFT_LABELS_TAU", "KAPPA_AB_DEFAULT",
    # Test hooks (muszą istnieć jako globalne nazwy dla monkeypatch)
    "collect_changed_files", "ana_show_file_at_rev", "ana_repo_root", "ana_changed_py_files",
]

# ──────────────────────────────────────────────────────────────────────────────
# 0a) Minimalne ładowanie .env (konwencja GLX_*)
# ──────────────────────────────────────────────────────────────────────────────

def _load_env(start: Path) -> None:
    """
    Ładuje .env (UTF-8) i ustawia os.environ tylko dla brakujących kluczy.
    Szuka:  <start>/.env  oraz  <start>/glitchlab/.env
    Usuwa komentarze inline: "KEY=VAL  # komentarz"
    """
    candidates = [start / ".env", start / "glitchlab" / ".env"]
    for env_path in candidates:
        try:
            if not env_path.exists():
                continue
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # komentarz inline (gdy bez cudzysłowów)
                if not ((val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'"))):
                    if "#" in val:
                        val = val.split("#", 1)[0].rstrip()
                # zdejmij otaczające cudzysłowy
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = val
        except Exception:
            # best-effort: brak .env nie powinien psuć działania
            pass


def _env_get(key: str, default: Any = None, cast: Callable = str) -> Any:
    """
    Pobiera GLX_* z os.environ z rzutowaniem i bezpiecznym fallbackiem.
    """
    val = os.environ.get(key, None)
    if val is None or str(val).strip() == "":
        return default
    try:
        return cast(val)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env_get(name, default, str)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_get(name, default, str))
    except Exception:
        return default


# ──────────────────────────────────────────────────────────────────────────────
# 0b) Importy z warstwy analizy – *bezpieczne*, z aliasami do testów
# ──────────────────────────────────────────────────────────────────────────────
# Utrzymujemy DWIE nazwy: „wewnętrzną” i „alias testowy”, np.:
#   _show_file_at_rev_ana  → (alias) ana_show_file_at_rev
#   changed_py_files       → (alias) ana_changed_py_files
# dzięki temu test może monkeypatchować aliasy, a nasz kod może wołać bezpiecznie.

# git_io
try:
    from backup.analysis.git_io import (
        repo_root as _repo_root_ana,
        git_merge_base,
        changed_py_files as _changed_py_files,
        show_file_at_rev as _show_file_at_rev_ana,
        EMPTY_TREE_SHA,
    )
except Exception:
    _repo_root_ana = None
    git_merge_base = None
    _changed_py_files = None
    _show_file_at_rev_ana = None
    EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # stały pusty tree w Git

# ast_index
try:
    from backup.analysis.ast_index import ast_summary_of_source as idx_ast_summary_of_source
except Exception:
    idx_ast_summary_of_source = None

# ast_delta
try:
    from backup.analysis import ast_delta as idx_ast_delta
except Exception:
    idx_ast_delta = None

# impact (opcjonalnie)
try:
    from backup.analysis.impact import impact_zone
except Exception:
    impact_zone = None

# reporting (opcjonalnie – pełne artefakty)
try:
    from backup.analysis import emit_artifacts as _emit_artifacts
except Exception:
    _emit_artifacts = None  # pragma: no cover

# Zbiórka zmienionych plików (publiczny helper z GUI) – to *musi* mieć globalny alias
# bo test monkeypatchuje: hma.collect_changed_files
try:
    from glitchlab.app.mosaic.git_delta import collect_changed_files as _collect_changed_files
except Exception:
    _collect_changed_files = None

# ── Aliasowanie NAZW, które testy spodziewają się zobaczyć w module ───────────
# (niech zawsze istnieją; None jeśli brak implementacji)
ana_repo_root = _repo_root_ana                 # callable | None
ana_changed_py_files = _changed_py_files       # callable | None
ana_show_file_at_rev = _show_file_at_rev_ana   # callable | None
collect_changed_files = _collect_changed_files # callable | None

# ──────────────────────────────────────────────────────────────────────────────
# 0c) Repo-top fallback (nie wymaga Gita; szuka najbliższego .git w górę)
# ──────────────────────────────────────────────────────────────────────────────

def _repo_top_fallback(start: Path) -> Path:
    """
    Idź w górę katalogów od `start` aż znajdziesz `.git`. Jeśli nie znajdziesz – zwróć `start`.
    Jeżeli w .env podane GLX_ROOT – użyj go w pierwszej kolejności (jeśli istnieje).
    """
    # 1) honoruj GLX_ROOT
    env_root = _env_get("GLX_ROOT", None, str)
    if env_root:
        try:
            p = Path(env_root).resolve()
            if (p / ".git").exists():
                return p
        except Exception:
            pass

    # 2) idź do góry aż do korzenia dysku
    p = start.resolve()
    try:
        if (p / ".git").exists():
            return p
    except Exception:
        pass
    for parent in [*p.parents]:
        try:
            if (parent / ".git").exists():
                return parent
        except Exception:
            continue

    # 3) dodatkowo sprawdź GLX_PKG w ramach GLX_ROOT (jeśli jest)
    if env_root:
        pkg = _env_get("GLX_PKG", None, str)
        if pkg:
            cand = Path(env_root) / pkg
            try:
                if (cand / ".git").exists():
                    return cand.resolve()
            except Exception:
                pass

    return p

# ──[GLX:SECTION:1/3:CORE+ENV+ALIAS::END]──────────────────────────────────────
# ──[GLX:SECTION:2/3:AST+MOSAIC+PHI_PSI]───────────────────────────────────────

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
    key = f"{label}|{depth}".encode("utf-8")
    h = hashlib.md5(key).digest()
    seed = int.from_bytes(h[:8], "little", signed=False)
    return np.random.default_rng(seed)


def _meta_for(label: str, depth: int) -> np.ndarray:
    rng = _rng_for_meta(label, depth)
    L, S, Sel, Stab, Cau, H = rng.uniform(0.35, 0.85, size=6)
    if label in ("FunctionDef", "ClassDef"):
        Stab = max(Stab, 0.8); Sel = max(Sel, 0.6)
    if label in ("If", "While", "For", "With", "Try"):
        Sel = max(Sel, 0.75); Cau = max(Cau, 0.7)
    if label in ("Call", "Expr"):
        L = max(L, 0.6)
    if label in ("Assign",):
        Stab = max(Stab, 0.7)
    return np.array([L, S, Sel, Stab, Cau, H], dtype=float)


def ast_deltas(src: str) -> AstSummary:
    tree = ast.parse(src)
    nodes: Dict[int, AstNode] = {}
    S = H = Z = 0
    maxZ = 0
    nid = 0

    def add(a: ast.AST, depth: int, parent: Optional[int]) -> int:
        nonlocal nid, S, H, Z, maxZ
        i = nid; nid += 1
        lab = a.__class__.__name__
        n = AstNode(i, lab, depth, parent)
        n.meta = _meta_for(lab, depth)
        nodes[i] = n
        if parent is not None:
            nodes[parent].children.append(i)

        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            S += 1; H += 1; Z += 1
        elif isinstance(a, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            S += 1; Z += 1
        elif isinstance(a, ast.Assign):
            S += 1; H += 1
        elif isinstance(a, ast.Call):
            S += 1; H += 2
        elif isinstance(a, (ast.Import, ast.ImportFrom)):
            S += 1; H += len(a.names)
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
    S = int(S); H = int(H); Z = int(max(Z, 0))
    tot = max(1, S + H)
    return AstSummary(S, H, Z, maxZ=maxZ, alpha=S / tot, beta=H / tot,
                      nodes=nodes, labels=[n.label for n in nodes.values()])


def compress_ast(summary: AstSummary, lam: float) -> AstSummary:
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
    edge: np.ndarray
    ssim: np.ndarray
    roi: np.ndarray
    kind: str = "grid"
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
    M = build_mosaic_grid(rows, cols, seed)
    centers: List[Tuple[float, float]] = []
    w = math.sqrt(3) * R
    for r in range(rows):
        for c in range(cols):
            x = c * w + (r % 2) * (w / 2.0)
            y = r * 1.5 * R
            centers.append((x, y))
    M.kind = "hex"; M.hex_centers = centers; M.hex_R = R
    return M


def build_mosaic(rows: int, cols: int, seed: int = 7,
                 kind: str = "grid", edge_thr: float = EDGE_THR_DEFAULT) -> Mosaic:
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
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _soft_label(edge_val: float, thr: float, tau: float) -> float:
    return _sigmoid((edge_val - thr) / max(1e-6, tau))


def _only_py(files: List[str]) -> List[str]:
    return [f for f in files if f.lower().endswith(".py")]

# pomocniczo – przyda się w innych sekcjach
def _abs(repo_root: Path, rel: str) -> Path:
    return (repo_root / rel).resolve()


def tile_dist(i: int, j: int, M: Mosaic, thr: float,
              alpha=1.0, beta=0.7, gamma=0.5,
              use_soft_labels: bool = SOFT_LABELS_DEFAULT,
              tau: float = SOFT_LABELS_TAU) -> float:
    x1, y1 = _xy_of_idx(i, M); x2, y2 = _xy_of_idx(j, M)
    geo = math.hypot(x1 - x2, y1 - y2)
    feat = abs(float(M.edge[i]) - float(M.edge[j]))
    if use_soft_labels:
        p1 = _soft_label(float(M.edge[i]), thr, tau)
        p2 = _soft_label(float(M.edge[j]), thr, tau)
        label_pen = abs(p1 - p2)
    else:
        label_pen = 1.0 if ((M.edge[i] > thr) != (M.edge[j] > thr)) else 0.0
    return alpha * geo + beta * feat + gamma * label_pen


def _dm_greedy_oneway(S: List[int], T: List[int], M: Mosaic, thr: float) -> float:
    if not S and not T: return 0.0
    if not S or not T:  return 0.0
    S2, T2 = S[:], T[:]
    cost = 0.0
    while S2 and T2:
        i = S2.pop()
        j_best = min(T2, key=lambda j: tile_dist(i, j, M, thr))
        cost += tile_dist(i, j_best, M, thr)
        T2.remove(j_best)
    return cost


def _length_penalty(len_diff: int, M: Mosaic, thr: float) -> float:
    if len_diff <= 0: return 0.0
    N = M.rows * M.cols
    if N < 2: return float(len_diff)
    rng = np.random.default_rng(12345)
    K = min(32, max(1, N // 4))
    idx = rng.choice(N, size=2 * K, replace=False)
    sample = [tile_dist(int(idx[2 * t]), int(idx[2 * t + 1]), M, thr) for t in range(K)]
    kappa = float(np.mean(sample)) if sample else 1.0
    return kappa * float(len_diff)


def _pair_cost(S: List[int], T: List[int], M: Mosaic, thr: float, max_match: int) -> float:
    if not S and not T:
        return 0.0
    k = min(len(S), len(T), max_match)
    len_diff = abs(len(S) - len(T))
    S2, T2 = S[:k], T[:k]
    if k <= 8:
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
    c1 = _pair_cost(S, T, M, thr, max_match)
    c2 = _pair_cost(T, S, M, thr, max_match)
    return 0.5 * (c1 + c2)

# ──────────────────────────────────────────────────────────────────────────────
# 3) Φ / Ψ
# ──────────────────────────────────────────────────────────────────────────────

Selector = Callable[[str, Mosaic, float], str]


def phi_region_for(label: str, M: Mosaic, thr: float) -> str:
    if label in ("Call", "Expr"):                         return "edges"
    if label in ("Assign",):                              return "~edges"
    if label in ("If", "For", "While", "With", "Return"): return "all"
    if label in ("FunctionDef", "ClassDef"):              return "roi"
    return "~edges"


def phi_region_for_balanced(label: str, M: Mosaic, thr: float) -> str:
    try:
        edge = np.asarray(getattr(M, "edge", []), dtype=float).reshape(-1)
    except Exception:
        edge = np.asarray([], dtype=float)
    if edge.size == 0:
        if label in ("FunctionDef", "ClassDef"): return "roi"
        if label in ("If", "For", "While", "With", "Return"): return "all"
        return "~edges"
    edge = np.nan_to_num(edge, nan=0.0, posinf=1.0, neginf=0.0)
    edge = np.clip(edge, 0.0, 1.0)
    q25, q75 = np.quantile(edge, [0.25, 0.75])
    thr = float(np.clip(thr, 0.0, 1.0))
    if label in ("Call", "Expr"): return "edges" if q75 >= thr else "~edges"
    if label in ("Assign",):      return "~edges" if q25 <= thr else "edges"
    if label in ("FunctionDef", "ClassDef"): return "roi"
    if label in ("If", "For", "While", "With", "Return"): return "all"
    return "~edges"


def phi_region_for_entropy(label: str, M: Mosaic, thr: float) -> str:
    def near_thr(x): return abs(x - thr) <= 0.05
    fuzzy = float(np.mean([near_thr(v) for v in M.edge])) > 0.25
    if fuzzy: return "all"
    return phi_region_for(label, M, thr)

# ──────────────────────────────────────────────────────────────────────────────
# Policy-aware Φ
# ──────────────────────────────────────────────────────────────────────────────

def load_policy_json(path: Optional[str]) -> Dict:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        out: Dict[str, Any] = {}
        if "avoid_roi_side_effects" in data: out["avoid_roi_side_effects"] = bool(data["avoid_roi_side_effects"])
        if "prefer_edges" in data:           out["prefer_edges"] = bool(data["prefer_edges"])
        if "prefer_stability" in data:       out["prefer_stability"] = bool(data["prefer_stability"])
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
    P = policy or {}
    roi_labels = set(P.get("roi_labels", []) or [])
    prefer_edges = bool(P.get("prefer_edges", False))
    prefer_stability = bool(P.get("prefer_stability", False))
    avoid_roi_side_effects = P.get("avoid_roi_side_effects", True)
    thr_bias = float(P.get("edge_thr_bias", 0.0) or 0.0)

    def _sel(label: str, M: Mosaic, thr: float) -> str:
        if label in roi_labels:
            return "roi"
        base_kind = phi_region_for_balanced(label, M, float(thr + thr_bias))
        if prefer_edges and label in ("Call", "Expr"):
            base_kind = "edges"
        if prefer_stability and label in ("Assign", "Return"):
            base_kind = "~edges"
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
    details: Dict[int, Dict[str, Any]] = {}
    total = 0.0
    for n in ast.nodes.values():
        kind = selector(n.label, M, thr)
        ids = region_ids(M, kind, thr)
        if not ids:
            details[n.id] = dict(kind=kind, cost=0.0, centroid=(None, None))
            continue
        alt_kind = "edges" if kind == "~edges" else "~edges"
        alt = region_ids(M, alt_kind, thr)
        if not alt: alt = ids[:]
        alt = alt[:len(ids)] if len(alt) >= len(ids) else (alt + ids[:len(ids) - len(alt)])
        cost = max(0.0, D_M(ids, alt, M, thr))
        details[n.id] = dict(kind=kind, cost=cost, centroid=centroid(ids, M))
        total += cost
    N = max(1, len(ast.nodes))
    return total / float(N), details


def psi_feedback(ast: AstSummary, M: Mosaic, delta: float, thr: float) -> AstSummary:
    if delta <= 1e-9: return ast
    nodes = ast.nodes
    for n in nodes.values():
        kind = phi_region_for(n.label, M, thr)
        ids = region_ids(M, kind, thr)
        if not ids: continue
        ed = np.array([M.edge[i] for i in ids], float)
        psi = np.array([
            float(1.0 - ed.mean()),
            float(0.5 + 0.5 * ed.std()),
            float(min(1.0, 0.5 + ed.mean())),
            float(1.0 - ed.std()),
            float(min(1.0, 0.3 + 0.7 * ed.mean())),
            float(0.4 + 0.5 * ed.std())
        ], dtype=float)
        n.meta = (1.0 - delta) * n.meta + delta * psi
    S, H, Z = ast.S, ast.H, ast.Z
    tot = max(1, S + H)
    return AstSummary(S, H, Z, ast.maxZ, S / tot, H / tot, nodes, ast.labels)

# ──────────────────────────────────────────────────────────────────────────────
# Adiacencja / komponenty spójne
# ──────────────────────────────────────────────────────────────────────────────

def _neighbors_grid(i: int, rows: int, cols: int) -> List[int]:
    r, c = divmod(i, cols)
    nbrs = []
    if r > 0:         nbrs.append((r - 1) * cols + c)
    if r + 1 < rows:  nbrs.append((r + 1) * cols + c)
    if c > 0:         nbrs.append(r * cols + (c - 1))
    if c + 1 < cols:  nbrs.append(r * cols + (c + 1))
    return nbrs


def _neighbors_hex_oddr(i: int, rows: int, cols: int) -> List[int]:
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
# Profil mozaiki, sprzężenie, metryki/inwarianty
# ──────────────────────────────────────────────────────────────────────────────

def mosaic_profile(M: "Mosaic", thr: float) -> Tuple[int, int, int, float, float]:
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
    if delta <= 1e-9 or kappa_ab <= 1e-9:
        return ast
    _, _, _, aM, bM = mosaic_profile(M, thr)
    ed = np.asarray(M.edge, dtype=float)
    uncert = float(min(1.0, max(0.0, ed.std())))
    w = float(min(1.0, max(0.0, kappa_ab * delta * (0.5 + 0.5 * uncert))))
    alpha_new = (1 - w) * ast.alpha + w * aM
    beta_new = 1.0 - alpha_new
    return AstSummary(ast.S, ast.H, ast.Z, ast.maxZ, alpha_new, beta_new, ast.nodes, ast.labels)


def distance_ast_mosaic(ast: AstSummary, M: Mosaic, thr: float, w=W_DEFAULT) -> float:
    _, _, _, aM, bM = mosaic_profile(M, thr)
    return (w['wS'] * abs(ast.alpha - aM) +
            w['wH'] * abs(ast.beta - bM) +
            w['wZ'] * abs(ast.Z / max(1, ast.maxZ) - 0.0))


def invariants_check(astA: AstSummary, astB: AstSummary, M: Mosaic, thr: float) -> Dict[str, bool]:
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
    return "balanced", phi_region_for_balanced, None


def run_once(lam: float, delta: float, rows: int, cols: int, thr: float,
             mosaic_kind: str = "grid",
             kappa_ab: float = KAPPA_AB_DEFAULT,
             phi_name: str = "balanced",
             policy_file: Optional[str] = None) -> Dict[str, float]:
    ast_raw = ast_deltas(EXAMPLE_SRC)
    ast_l = compress_ast(ast_raw, lam)
    M = build_mosaic(rows, cols, seed=7, kind=mosaic_kind, edge_thr=thr)
    J1, _ = phi_cost(ast_l, M, thr, selector=phi_region_for)
    J2, _ = phi_cost(ast_l, M, thr, selector=phi_region_for_balanced)
    J3, _ = phi_cost(ast_l, M, thr, selector=phi_region_for_entropy)
    sel_label, selector, _policy = _selector_from_args(phi_name, policy_file)
    Jp = None
    if sel_label == "policy":
        Jp, _ = phi_cost(ast_l, M, thr, selector=selector)
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
    out: List[Dict[str, float]] = []
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

# ──[GLX:SECTION:2/3:AST+MOSAIC+PHI_PSI::END]──────────────────────────────────
# ──[GLX:SECTION:3/3:FROM_GIT+ARTIFACTS+CLI]───────────────────────────────────
# Wszystko co dotyczy:
# • detekcji repo / diffu (BASE..HEAD),
# • fallbacków GIT,
# • emisji artefaktów (report.json, mosaic_map.json, summary.md),
# • komend CLI,
# • bezpiecznych aliasów analysis.git_io (żeby testy/CI nie łapały NameError).

# ──────────────────────────────────────────────────────────────────────────────
# Publiczne aliasy i bezpieczne wrappery na warstwę analysis.git_io
# ──────────────────────────────────────────────────────────────────────────────

def ana_repo_root() -> Optional[Path]:
    """Bezpieczny alias na analysis.git_io.repo_root (może nie istnieć)."""
    if _repo_root_ana is None:
        return None
    try:
        return _repo_root_ana()
    except Exception:
        return None


def ana_changed_py_files(base: str, head: str) -> Optional[List[str]]:
    """Bezpieczny alias na analysis.git_io.changed_py_files (może nie istnieć)."""
    if changed_py_files is None:
        return None
    try:
        return changed_py_files(base, head)
    except Exception:
        return None


def ana_show_file_at_rev(rel_path: str, rev: str) -> Optional[str]:
    """Bezpieczny alias na analysis.git_io.show_file_at_rev (może nie istnieć)."""
    if _show_file_at_rev_ana is None:
        return None
    try:
        return _show_file_at_rev_ana(rel_path, rev)
    except Exception:
        return None


# Publiczna funkcja (łatwa do monkeypatchowania w testach).
# Jeśli nie ma wersji z app/mosaic/git_delta – użyj fallbacku lokalnego.
def collect_changed_files(repo_root: Path, base: str, head: str) -> List[str]:
    if '_collect_changed_files' in globals() and callable(_collect_changed_files):
        try:
            return list(_collect_changed_files(repo_root, base, head))  # type: ignore[misc]
        except Exception:
            pass
    return _fallback_collect_changed_files(repo_root, base, head)


# ──────────────────────────────────────────────────────────────────────────────
# Pomocnicze: filtr ścieżek i repo-root z .env
# ──────────────────────────────────────────────────────────────────────────────

def _file_selector_filter(paths_like: Optional[List[str]], rel_path: str) -> bool:
    if not paths_like:
        return True
    rel = rel_path.replace("\\", "/")
    for p in paths_like:
        if p in rel or rel.endswith(p):
            return True
    return False


def _env_repo_root() -> Optional[Path]:
    """GLX_ROOT[/GLX_PKG] → Path, jeśli .git istnieje; w przeciwnym razie None."""
    r = _env_get("GLX_ROOT", None, str)
    if not r:
        return None
    base = Path(r).resolve()
    if (base / ".git").exists():
        return base
    pkg = _env_get("GLX_PKG", None, str)
    if pkg:
        cand = (base / pkg).resolve()
        if (cand / ".git").exists():
            return cand
    # skan podkatalogów (poziom 1)
    try:
        for child in base.iterdir():
            if child.is_dir() and (child / ".git").exists():
                return child.resolve()
    except Exception:
        pass
    return base


def _repo_top_from_env() -> Optional[Path]:
    """Wersja ‘raw’: jeśli GLX_ROOT da się zresolvować – zwróć Path, inaczej None."""
    root = os.environ.get("GLX_ROOT")
    if root:
        try:
            return Path(root).resolve()
        except Exception:
            return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# GIT helpers (z fallbackami)
# ──────────────────────────────────────────────────────────────────────────────

def _git_run(repo_root: Path, args: List[str]) -> Tuple[int, str, str]:
    """Uruchamia `git ...` w katalogu repo_root. Zwraca (rc, stdout, stderr)."""
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except Exception as e:
        return 1, "", f"{e}"


def _git_show_file(repo_root: Path, rev: str, rel_path: str) -> str:
    rc, out, _ = _git_run(repo_root, ["show", f"{rev}:{rel_path}"])
    return out if rc == 0 else ""


def _git_top(start: Path) -> Optional[Path]:
    try:
        rc, out, _ = _git_run(start, ["rev-parse", "--show-toplevel"])
        if rc == 0 and out.strip():
            return Path(out.strip()).resolve()
    except Exception:
        pass
    return None


def _guess_repo_top(start: Path) -> Path:
    p = _git_top(start)
    if isinstance(p, Path):
        return p
    # spróbuj analysis.git_io.repo_root
    try:
        q = ana_repo_root()
        if q and (q / ".git").exists():
            return q
    except Exception:
        pass
    # heurystyka: podkatalogi ze `.git`
    try:
        for child in start.iterdir():
            if child.is_dir() and (child / ".git").exists():
                return child.resolve()
    except Exception:
        pass
    return start.resolve()


def _git_ls_files_py(repo_root: Path) -> List[str]:
    """`git ls-files` (jeśli się da), w przeciwnym razie rekurencja po FS."""
    rc, out, _ = _git_run(repo_root, ["ls-files"])
    if rc == 0 and out:
        items = [ln.strip() for ln in out.splitlines() if ln.strip()]
        return [p for p in items if p.lower().endswith(".py")]

    # fallback FS
    skip = {".git", ".venv", "venv", "__pycache__", ".idea", ".pytest_cache"}
    py: List[str] = []
    root_abs = repo_root.resolve()
    for root, dirs, files in os.walk(root_abs):
        dirs[:] = [d for d in dirs if d not in skip]
        for fn in files:
            if not fn.lower().endswith(".py"):
                continue
            abs_p = Path(root) / fn
            try:
                rel = abs_p.resolve().relative_to(root_abs)
            except Exception:
                rel = abs_p
            py.append(str(rel).replace("\\", "/"))
    return py


def _fallback_collect_changed_files(repo_root: Path, base: str, head: str) -> List[str]:
    rc, out, _ = _git_run(repo_root, ["diff", "--name-only", f"{base}..{head}"])
    if rc != 0:
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _repo_top_fallback(start: Path) -> Path:
    """Zgadywanie repo topu gdy .env / analysis nie pomogły."""
    return _guess_repo_top(start)


# ──────────────────────────────────────────────────────────────────────────────
# Solidny kolektor zmienionych plików .py (kilka strategii + ścieżki)
# ──────────────────────────────────────────────────────────────────────────────

def _collect_changed_py_robust(repo_top: Path,
                               base: str,
                               head: str,
                               paths: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    """
    Zwraca (changed_py, attempts).
    Próby (w tej kolejności):
      1) collect_changed_files(repo_top, base, head)    [publiczny alias / test-friendly]
      2) analysis.git_io.changed_py_files(base, head)   [jeśli jest]
      3) git diff --name-only base..head
      4) git diff-tree --no-commit-id --name-only -r head
      5) gdy base == EMPTY_TREE_SHA → git ls-files (*.py)
      6) fallback: git ls-files (*.py)
    """
    attempts: List[str] = []

    # 1) publiczny helper
    try:
        attempts.append("collect_changed_files")
        changed = collect_changed_files(repo_top, base, head)
        changed_py = _only_py([p for p in changed if _file_selector_filter(paths, p)])
        if changed_py:
            return changed_py, attempts
    except Exception:
        pass

    # 2) analysis.git_io.changed_py_files
    if ana_changed_py_files(None, None) is not None:  # type: ignore[arg-type]
        try:
            attempts.append("analysis.changed_py_files")
            changed2 = ana_changed_py_files(base, head) or []
            changed_py = _only_py([p for p in changed2 if _file_selector_filter(paths, p)])
            if changed_py:
                return changed_py, attempts
        except Exception:
            pass

    # 3) git diff --name-only
    attempts.append("git diff --name-only")
    rc, out, _ = _git_run(repo_top, ["diff", "--name-only", f"{base}..{head}"])
    if out.strip():
        ch = [ln.strip() for ln in out.splitlines() if ln.strip()]
        changed_py = _only_py([p for p in ch if _file_selector_filter(paths, p)])
        if changed_py:
            return changed_py, attempts

    # 4) git diff-tree -r
    attempts.append("git diff-tree -r")
    rc, out, _ = _git_run(repo_top, ["diff-tree", "--no-commit-id", "--name-only", "-r", head])
    if out.strip():
        ch = [ln.strip() for ln in out.splitlines() if ln.strip()]
        changed_py = _only_py([p for p in ch if _file_selector_filter(paths, p)])
        if changed_py:
            return changed_py, attempts

    # 5) pierwszy commit
    attempts.append("git ls-files (.py) [first-commit-mode]")
    try:
        is_first = (EMPTY_TREE_SHA is not None) and (base == EMPTY_TREE_SHA)
    except Exception:
        is_first = False
    if is_first:
        changed_py = _git_ls_files_py(repo_top)
        changed_py = [p for p in changed_py if _file_selector_filter(paths, p)]
        if changed_py:
            return changed_py, attempts

    # 6) ostateczny fallback
    changed_py = _git_ls_files_py(repo_top)
    changed_py = [p for p in changed_py if _file_selector_filter(paths, p)]
    return changed_py, attempts


# ──────────────────────────────────────────────────────────────────────────────
# FROM-GIT (JSON) i DUMP (artefakty)
# ──────────────────────────────────────────────────────────────────────────────

def _ast_summary_for_source_det(src: str, file_path: str) -> AstSummary:
    """Deterministyczny wybór AstSummary: preferuj analysis.ast_index, inaczej lokalny model."""
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
        repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Git diff (BASE..HEAD) → .py → AstSummary(base/head) → Δ, Φ/Ψ → JSON.
    repo_root: jawny root repo; jeśli None → .env → analysis → zgadywanie → CWD.
    """
    # sanity wejścia (.env może mieć inline-komentarze/cudzysłowy)
    def _clean(s: Optional[str], choices: Optional[Iterable[str]] = None) -> Optional[str]:
        if s is None:
            return None
        v = str(s).strip()
        if "#" in v:
            v = v.split("#", 1)[0].rstrip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1].strip()
        if choices:
            opts = list(choices)
            if v not in opts:
                v = opts[0]
        return v or None

    base = _clean(base) or base
    head = _clean(head) or head
    mosaic_kind = _clean(mosaic_kind, choices=("grid", "hex")) or "grid"
    phi_name = _clean(phi_name, choices=("basic", "balanced", "entropy", "policy")) or "balanced"
    policy_file = _clean(policy_file)

    # repo_root
    root = repo_root or _env_repo_root() or ana_repo_root() or _guess_repo_top(Path.cwd())

    # policy_file względne → względem root
    if policy_file:
        p = Path(policy_file)
        if not p.is_absolute():
            p = (root / p).resolve()
        policy_file = str(p) if p.exists() else None

    # 1) lista plików
    changed = collect_changed_files(root, base, head)
    changed = _only_py([p for p in changed if _file_selector_filter(paths, p)])

    # 2) mozaika + selektor
    M = build_mosaic(rows, cols, seed=7, kind=mosaic_kind, edge_thr=thr)
    sel_label, selector, _policy = _selector_from_args(phi_name, policy_file)

    # 3) pobieranie treści (prefer analysis.show_file_at_rev gdy ma dobry root)
    def _show(rel: str, rev: str) -> str:
        try:
            ana_root = ana_repo_root()
        except Exception:
            ana_root = None
        if isinstance(ana_root, Path) and (ana_root / ".git").exists():
            try:
                s = ana_show_file_at_rev(rel, rev)
                if isinstance(s, str):
                    return s
            except Exception:
                pass
        return _git_show_file(root, rev, rel)

    files_out: List[Dict[str, Any]] = []
    totals = dict(dS=0, dH=0, dZ=0, align_gain=0.0)

    for rel_path in changed:
        base_src = _show(rel_path, base)
        head_src = _show(rel_path, head)

        a_base = _ast_summary_for_source_det(base_src, rel_path)
        a_head = _ast_summary_for_source_det(head_src, rel_path)

        # Δ (prefer analysis.ast_delta)
        if idx_ast_delta is not None:
            try:
                d = idx_ast_delta(a_base, a_head)
                if isinstance(d, dict):
                    dS = int(d.get("dS", 0)); dH = int(d.get("dH", 0)); dZ = int(d.get("dZ", 0))
                elif isinstance(d, (tuple, list)) and len(d) >= 3:
                    dS, dH, dZ = map(int, d[:3])
                else:
                    dS = int(getattr(d, "dS", 0)); dH = int(getattr(d, "dH", 0)); dZ = int(getattr(d, "dZ", 0))
            except Exception:
                dS = a_head.S - a_base.S; dH = a_head.H - a_base.H; dZ = a_head.Z - a_base.Z
        else:
            dS = a_head.S - a_base.S; dH = a_head.H - a_base.H; dZ = a_head.Z - a_base.Z

        # Φ
        J1, _ = phi_cost(a_head, M, thr, selector=phi_region_for)
        J2, _ = phi_cost(a_head, M, thr, selector=phi_region_for_balanced)
        J3, _ = phi_cost(a_head, M, thr, selector=phi_region_for_entropy)
        Jp = None
        if sel_label == "policy":
            Jp, _ = phi_cost(a_head, M, thr, selector=selector)

        # Align
        Align_before = 1.0 - min(1.0, distance_ast_mosaic(a_head, M, thr))
        a_after = psi_feedback(a_head, M, delta, thr)
        a_cpl = couple_alpha_beta(a_after, M, thr, delta=delta, kappa_ab=kappa_ab)
        Align_after = 1.0 - min(1.0, distance_ast_mosaic(a_cpl, M, thr))

        files_out.append({
            "path": rel_path,
            "delta": {"dS": dS, "dH": dH, "dZ": dZ},
            "phi": {"J1": J1, "J2": J2, "J3": J3, **({"J_policy": Jp, "policy_used": True} if Jp is not None else {})},
            "align": {"before": round(Align_before, 6), "after": round(Align_after, 6)},
            "mosaic": {"kind": mosaic_kind, "thr": thr, "grid": {"rows": rows, "cols": cols}},
        })

        totals["dS"] += dS
        totals["dH"] += dH
        totals["dZ"] += dZ
        totals["align_gain"] += (Align_after - Align_before)

    out: Dict[str, Any] = {
        "base": base, "head": head,
        "branch": _env_get("GLX_BRANCH", None, str),
        "repo_root": str(root) if root else None,
        "mosaic": {"kind": mosaic_kind, "grid": {"rows": rows, "cols": cols}, "thr": thr},
        "delta": {"files": len(files_out)},
        "phi_selector": sel_label,
        "policy_file": policy_file if policy_file else None,
        "files": files_out,
        "totals": totals,
    }
    if not files_out:
        out["note"] = "git-delta-empty: próbowano diff/diff-tree/show; żadnych .py do analizy."
    return out


def _emit_fallback_artifacts(res: Dict[str, Any], outdir: Path) -> Dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    report_p = (outdir / "report.json").resolve()
    map_p = (outdir / "mosaic_map.json").resolve()
    summary_p = (outdir / "summary.md").resolve()

    report_p.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    rows = int(res.get("mosaic", {}).get("grid", {}).get("rows", 6))
    cols = int(res.get("mosaic", {}).get("grid", {}).get("cols", 6))
    thr = float(res.get("mosaic", {}).get("thr", EDGE_THR_DEFAULT))
    kind = str(res.get("mosaic", {}).get("kind", "grid"))
    M = build_mosaic(rows, cols, seed=7, kind=kind, edge_thr=thr)

    mosaic_map = {
        "schema": "glx.mosaic.map.v1",
        "grid": {"rows": rows, "cols": cols},
        "kind": kind,
        "edges": [float(x) for x in np.asarray(M.edge, float).reshape(-1).tolist()],
        "meta": {"base": res.get("base"), "head": res.get("head"), "thr": thr},
    }
    map_p.write_text(json.dumps(mosaic_map, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = [
        "# GLX Mosaic — hybrid AST map",
        f"- base: `{res.get('base')}`",
        f"- head: `{res.get('head')}`",
        f"- mosaic: `{kind}`  rows={rows} cols={cols} thr={thr}",
        f"- files analyzed: {res.get('delta', {}).get('files', 0)}",
        "",
    ]
    summary_p.write_text("\n".join(summary), encoding="utf-8")

    return {"report": str(report_p), "mosaic_map": str(map_p), "summary": str(summary_p)}


# ──────────────────────────────────────────────────────────────────────────────
# Wariacja: from-git-dump → zapis artefaktów
# ──────────────────────────────────────────────────────────────────────────────

def _env_defaults() -> Dict[str, Any]:
    return dict(
        mosaic=_env_get("GLX_MOSAIC", "grid", str),
        rows=_env_int("GLX_ROWS", 6),
        cols=_env_int("GLX_COLS", 6),
        edge_thr=_env_float("GLX_EDGE_THR", EDGE_THR_DEFAULT),
        delta=_env_float("GLX_DELTA", 0.25),
        kappa_ab=_env_float("GLX_KAPPA", KAPPA_AB_DEFAULT),
        phi=_env_get("GLX_PHI", "balanced", str),
        policy_file=_env_get("GLX_POLICY", None, str),
    )


def cmd_from_git_dump(args) -> Dict[str, Any]:
    """
    Analiza BASE..HEAD, zapis artefaktów (report.json, mosaic_map.json, summary.md).
    Relatywne --out liczymy względem repo-top. Wspiera GLX_OUT z .env.
    Priorytety repo-root:  --repo-root > .env(GLX_ROOT) > analysis.git_io.repo_root() > fallback (szukanie .git).
    """
    # 0) .env (best-effort; uzupełnia tylko brakujące os.environ)
    _load_env(Path.cwd())

    # 1) repo-root: preferuj jawny --repo-root, potem .env, potem analysis.git_io, na koniec heurystyka
    repo_top: Optional[Path] = None
    if getattr(args, "repo_root", None):
        try:
            repo_top = Path(args.repo_root).resolve()
        except Exception:
            repo_top = None
    if not isinstance(repo_top, Path) or not repo_top:
        try:
            repo_top = _repo_top_from_env() or (_repo_root_ana() if _repo_root_ana else None)
        except Exception:
            repo_top = None
    if not isinstance(repo_top, Path) or not repo_top:
        repo_top = _repo_top_fallback(Path.cwd())

    # 2) Parametry: wartości z .env NIE są nadpisywane przez CLI (CLI to tylko default)
    base        = str(getattr(args, "base", "") or "HEAD~1")
    head        = str(getattr(args, "head", "") or "HEAD")
    rows        = int(_env_get("GLX_ROWS",       getattr(args, "rows", 6),          int))
    cols        = int(_env_get("GLX_COLS",       getattr(args, "cols", 6),          int))
    thr         = float(_env_get("GLX_EDGE_THR", getattr(args, "edge_thr", 0.55),   float))
    mosaic_kind = str(_env_get("GLX_MOSAIC",     getattr(args, "mosaic", "grid"),   str))
    delta       = float(_env_get("GLX_DELTA",    getattr(args, "delta", 0.25),      float))
    kappa       = float(_env_get("GLX_KAPPA",    getattr(args, "kappa_ab", 0.35),   float))
    phi_name    = str(_env_get("GLX_PHI",        getattr(args, "phi", "balanced"),  str))
    policy_file = _env_get("GLX_POLICY",         getattr(args, "policy_file", None), str)
    paths       = getattr(args, "paths", None) or None

    # Jeżeli podano policy_file względnie → przelicz względem repo_top i sprawdź istnienie
    if policy_file:
        pol = Path(policy_file)
        if not pol.is_absolute():
            pol = (repo_top / pol).resolve()
        policy_file = str(pol) if pol.exists() else None

    # 3) Lista zmienionych .py (robust chain: alias → analysis → git diff → git diff-tree → ls-files)
    changed_py, attempts = _collect_changed_py_robust(repo_top, base, head, paths)

    # 4) Mozaika + selektor Φ (uwzględnia wariant policy)
    M = build_mosaic(rows, cols, seed=7, kind=mosaic_kind, edge_thr=thr)
    sel_label, selector, _policy = _selector_from_args(phi_name, policy_file)

    # 5) Helper do pobierania treści z GIT: preferuj analysis.show_file_at_rev TYLKO gdy jego repo jest prawidłowe
    def _show(rel_path: str, rev: str) -> str:
        use_local = True
        if _show_file_at_rev_ana is not None and _repo_root_ana is not None:
            try:
                ana_root = _repo_root_ana()
            except Exception:
                ana_root = None
            if isinstance(ana_root, Path) and (ana_root / ".git").exists():
                use_local = False
        if use_local:
            return _git_show_file(repo_top, rev, rel_path)
        try:
            return _show_file_at_rev_ana(rel_path, rev) or ""
        except Exception:
            return _git_show_file(repo_top, rev, rel_path)

    files_out: List[Dict[str, Any]] = []
    totals = dict(dS=0, dH=0, dZ=0, align_gain=0.0)

    # 6) Pętla po plikach: ΔS/ΔH/ΔZ, koszty Φ, Align przed/po Ψ + sprzężeniu κ
    for rel_path in changed_py:
        base_src = _show(rel_path, base)
        head_src = _show(rel_path, head)

        a_base = _ast_summary_for_source_det(base_src, rel_path)
        a_head = _ast_summary_for_source_det(head_src, rel_path)

        # Δ (preferencja idx_ast_delta, fallback różnice surowe)
        if idx_ast_delta is not None:
            try:
                d = idx_ast_delta(a_base, a_head)
                if isinstance(d, dict):
                    dS = int(d.get("dS", 0)); dH = int(d.get("dH", 0)); dZ = int(d.get("dZ", 0))
                elif isinstance(d, (tuple, list)) and len(d) >= 3:
                    dS, dH, dZ = map(int, d[:3])
                else:
                    dS = int(getattr(d, "dS", 0)); dH = int(getattr(d, "dH", 0)); dZ = int(getattr(d, "dZ", 0))
            except Exception:
                dS = a_head.S - a_base.S; dH = a_head.H - a_base.H; dZ = a_head.Z - a_base.Z
        else:
            dS = a_head.S - a_base.S; dH = a_head.H - a_base.H; dZ = a_head.Z - a_base.Z

        # Φ (trzy warianty + opcjonalny policy)
        J1, _ = phi_cost(a_head, M, thr, selector=phi_region_for)
        J2, _ = phi_cost(a_head, M, thr, selector=phi_region_for_balanced)
        J3, _ = phi_cost(a_head, M, thr, selector=phi_region_for_entropy)
        Jp = None
        if sel_label == "policy":
            Jp, _ = phi_cost(a_head, M, thr, selector=selector)

        # Align przed/po Ψ i sprzężeniu (κ)
        Align_before = 1.0 - min(1.0, distance_ast_mosaic(a_head, M, thr))
        a_after = psi_feedback(a_head, M, delta, thr)
        a_cpl = couple_alpha_beta(a_after, M, thr, delta=delta, kappa_ab=kappa)
        Align_after = 1.0 - min(1.0, distance_ast_mosaic(a_cpl, M, thr))

        entry = {
            "path": rel_path,
            "delta": {"dS": dS, "dH": dH, "dZ": dZ},
            "phi": {"J1": J1, "J2": J2, "J3": J3},
            "align": {"before": round(Align_before, 6), "after": round(Align_after, 6)},
            "mosaic": {"kind": mosaic_kind, "thr": thr, "grid": {"rows": rows, "cols": cols}},
        }
        if Jp is not None:
            entry["phi"]["J_policy"] = Jp
            entry["phi"]["policy_used"] = True

        files_out.append(entry)
        totals["dS"] += dS; totals["dH"] += dH; totals["dZ"] += dZ
        totals["align_gain"] += (Align_after - Align_before)

    # 7) Raport JSON (stdout) + notatka, jeśli brak plików
    out: Dict[str, Any] = {
        "base": base, "head": head, "branch": _env_get("GLX_BRANCH", None, str),
        "repo_root": str(repo_top),
        "mosaic": {"kind": mosaic_kind, "grid": {"rows": rows, "cols": cols}, "thr": thr},
        "delta": {"files": len(files_out)},
        "phi_selector": sel_label,
        "policy_file": policy_file if policy_file else None,
        "files": files_out,
        "totals": totals,
    }
    if not changed_py:
        out["note"] = f"git-delta-empty: próbowano {', '.join(attempts)}; żadnych .py do analizy."

    # 8) Artefakty: --out > GLX_OUT > 'analysis/last' (relatywny → względem repo_top)
    out_dir_str = getattr(args, "out", None) or _env_get("GLX_OUT", "analysis/last", str)
    out_dir = Path(out_dir_str)
    if not out_dir.is_absolute():
        out_dir = (repo_top / out_dir).resolve()

    try:
        if _emit_artifacts is not None:
            _emit_artifacts(out, out_dir)  # jeśli dostępna warstwa reporting
    except Exception:
        pass

    artifacts = _emit_fallback_artifacts(out, out_dir)

    # 9) Tryb ścisły artefaktów (CI-friendly)
    if getattr(args, "strict_artifacts", False):
        missing = [k for k, v in artifacts.items() if not Path(v).exists()]
        if missing:
            raise SystemExit(1)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out

# ──────────────────────────────────────────────────────────────────────────────
# CLI (run/sweep/test/from-git/from-git-dump)
# ──────────────────────────────────────────────────────────────────────────────

def _pretty_table(rows: List[Dict[str, float]]) -> str:
    header = ["λ", "Δ", "Align", "J_phi1", "J_phi2", "J_phi3", "J_phiP", "CR_AST", "CR_TO", "α", "β", "S", "H", "Z"]
    widths = [4, 4, 7, 8, 8, 8, 8, 8, 8, 5, 5, 4, 4, 3]

    def rowfmt(r):
        return [
            f"{r['lambda_']:.2f}", f"{r['delta_']:.2f}", f"{r['Align']:.3f}",
            f"{r['J_phi1']:.4f}", f"{r['J_phi2']:.4f}", f"{r['J_phi3']:.4f}",
            f"{r.get('J_phiP', '-') if 'J_phiP' in r else '-'}",
            f"{r['CR_AST']:.3f}", f"{r['CR_TO']:.3f}",
            f"{r['alpha']:.2f}", f"{r['beta']:.2f}",
            int(r['S']), int(r['H']), int(r['Z'])
        ]

    def line(cols): return " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))

    s = [line(header), "-" * (sum(widths) + len(widths) - 1)]
    for r in rows: s.append(line(rowfmt(r)))
    return "\n".join(s)


def cmd_run(args):
    res = run_once(
        args.lmbd, args.delta, args.rows, args.cols, args.edge_thr,
        mosaic_kind=args.mosaic, kappa_ab=args.kappa_ab,
        phi_name=args.phi, policy_file=args.policy_file
    )
    out = {
        "lambda": args.lmbd, "delta": args.delta,
        "rows": args.rows, "cols": args.cols, "kind": args.mosaic,
        "edge_thr": args.edge_thr, "phi": args.phi, "policy_file": args.policy_file,
        **res
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_sweep(args):
    rows = sweep(
        args.rows, args.cols, args.edge_thr,
        mosaic_kind=args.mosaic, kappa_ab=args.kappa_ab,
        phi_name=args.phi, policy_file=args.policy_file
    )
    print(_pretty_table(rows))
    if args.json:
        print("\n[JSON]")
        print(json.dumps(rows, indent=2, ensure_ascii=False))


def cmd_test(args):
    inv_astA = ast_deltas(EXAMPLE_SRC)
    inv_astB = compress_ast(inv_astA, args.lmbd)
    M = build_mosaic(args.rows, args.cols, seed=7, kind=args.mosaic, edge_thr=args.edge_thr)
    inv = invariants_check(inv_astA, inv_astB, M, args.edge_thr)
    print("[INVARIANTS]")
    for k, v in inv.items():
        print(f"  - {k}: {'PASS' if v else 'FAIL'}")
    print("\n[SIGN TEST Φ2 > Φ1]")
    sign = sign_test_phi2_better(args.runs, args.rows, args.cols, args.edge_thr,
                                 lam=args.lmbd, mosaic_kind=args.mosaic)
    print(json.dumps(sign, indent=2, ensure_ascii=False))


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
        repo_root=Path(args.repo_root).resolve() if getattr(args, "repo_root", None) else None,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


def build_cli():
    envd = _env_defaults()
    p = argparse.ArgumentParser(
        prog="hybrid_ast_mosaic",
        description="Hybryda AST ⇄ Mozaika (Φ/Ψ) – metryki, sweep, inwarianty, from-git (.env-aware)"
    )
    p.add_argument("--mosaic", choices=["grid", "hex"], default=envd["mosaic"], help="rodzaj mozaiki")
    p.add_argument("--rows", type=int, default=envd["rows"], help="liczba wierszy mozaiki")
    p.add_argument("--cols", type=int, default=envd["cols"], help="liczba kolumn mozaiki")
    p.add_argument("--edge-thr", type=float, default=envd["edge_thr"], help="próg edge dla regionów")
    p.add_argument("--kappa-ab", type=float, default=envd["kappa_ab"], help="siła sprzężenia Ψ→(α,β)")
    p.add_argument("--phi", choices=["basic", "balanced", "entropy", "policy"], default=envd["phi"],
                   help="wariant selektora Φ")
    p.add_argument("--policy-file", default=envd["policy_file"], help="JSON z politykami dla --phi policy")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("run", help="pojedynczy przebieg (λ, Δ) – demo na EXAMPLE_SRC")
    q.add_argument("--lmbd", type=float, default=0.60)
    q.add_argument("--delta", type=float, default=envd["delta"])
    q.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="siatka λ×Δ i tabela metryk (demo)")
    s.add_argument("--json", action="store_true", help="dodatkowo wypisz JSON")
    s.set_defaults(func=cmd_sweep)

    t = sub.add_parser("test", help="inwarianty + test znaku Φ2 vs Φ1 (wiele seedów, demo)")
    t.add_argument("--lmbd", type=float, default=0.60)
    t.add_argument("--runs", type=int, default=100)
    t.set_defaults(func=cmd_test)

    g = sub.add_parser("from-git", help="Analiza realnego diffu BASE..HEAD z repo (stdout JSON)")
    g.add_argument("--base", required=True, help="BASE sha/ref (np. merge-base, .glx/state.json)")
    g.add_argument("--head", default="HEAD", help="HEAD sha/ref (domyślnie HEAD)")
    g.add_argument("--rows", type=int, default=envd["rows"])
    g.add_argument("--cols", type=int, default=envd["cols"])
    g.add_argument("--edge-thr", type=float, default=envd["edge_thr"])
    g.add_argument("--mosaic", choices=["grid", "hex"], default=envd["mosaic"])
    g.add_argument("--delta", type=float, default=envd["delta"])
    g.add_argument("--kappa-ab", type=float, default=envd["kappa_ab"])
    g.add_argument("--paths", nargs="*")
    g.add_argument("--phi", choices=["basic", "balanced", "entropy", "policy"], default=envd["phi"])
    g.add_argument("--policy-file", default=envd["policy_file"])
    g.add_argument("--repo-root", default=_env_get("GLX_ROOT", None, str),
                   help="jawne wskazanie katalogu repo (.git)")
    g.set_defaults(func=cmd_from_git)

    gd = sub.add_parser("from-git-dump",
                        help="Analiza BASE..HEAD i zapis artefaktów (report.json, summary.md, mosaic_map.json)")
    gd.add_argument("--base", required=True, help="BASE sha/ref (np. merge-base, .glx/state.json)")
    gd.add_argument("--head", default="HEAD", help="HEAD sha/ref (domyślnie HEAD)")
    gd.add_argument("--rows", type=int, default=envd["rows"])
    gd.add_argument("--cols", type=int, default=envd["cols"])
    gd.add_argument("--edge-thr", type=float, default=envd["edge_thr"])
    gd.add_argument("--mosaic", choices=["grid", "hex"], default=envd["mosaic"])
    gd.add_argument("--delta", type=float, default=envd["delta"])
    gd.add_argument("--kappa-ab", type=float, default=envd["kappa_ab"])
    gd.add_argument("--paths", nargs="*")
    gd.add_argument("--phi", choices=["basic", "balanced", "entropy", "policy"], default=envd["phi"])
    gd.add_argument("--policy-file", default=envd["policy_file"])
    gd.add_argument("--out", default=_env_get("GLX_OUT", "analysis/last", str),
                    help="katalog wyjściowy na artefakty (RELATYWNY → repo-root)")
    gd.add_argument("--strict-artifacts", action="store_true",
                    help="jeżeli artefakty nie powstaną, zakończ błędem (rc=1)")
    gd.add_argument("--repo-root", default=_env_get("GLX_ROOT", None, str),
                    help="jawne wskazanie katalogu repo (.git)")
    gd.set_defaults(func=cmd_from_git_dump)

    return p


def cli_entry() -> int:
    """Wejście CLI gdy skrypt uruchamiany bezpośrednio (nie podczas importu/testów)."""
    parser = build_cli()
    args = parser.parse_args()
    args.func(args)
    return 0


def main():
    # Świadomie NIE wołamy nic przy imporcie. Pozostawiamy entrypoint opcjonalny.
    return cli_entry()


# Nie uruchamiaj w trakcie importu (używane także przez testy).
# if __name__ == "__main__":
#     raise SystemExit(cli_entry())

# ──[GLX:SECTION:3/3:FROM_GIT+ARTIFACTS+CLI::END]──────────────────────────────
