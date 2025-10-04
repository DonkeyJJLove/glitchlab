# glitchlab/analysis/ast_index.py
# Deterministyczny indeks AST → AstSummary (bez RNG)
# Python 3.9+ (stdlib + lokalny import git_io)

from __future__ import annotations

import ast
import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Lokalny I/O git
try:
    from .git_io import show_file_at_rev
except Exception:  # pozwól na uruchomienie modułu samodzielnie
    def show_file_at_rev(path: str, rev: str = "HEAD") -> Optional[str]:
        return None


__all__ = [
    "AstNodeLite",
    "AstSummary",
    "ast_summary_of_source",
    "ast_summary_of_file",
    "ast_summary_of_rev",
    "summarize_labels",
]


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AstNodeLite:
    """
    Minimalny węzeł: etykieta + pozycja + głębokość + meta-wektor.
    meta = [L, S, Sel, Stab, Cau, H]  ∈ [0..1]^6 (deterministycznie z cech)
    """
    id: int
    label: str
    depth: int
    parent: Optional[int]
    lineno: Optional[int]
    col: Optional[int]
    children: List[int] = field(default_factory=list)
    meta: Tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)


@dataclass
class AstSummary:
    """
    Zbiorcze statystyki AST – kompatybilne polami z warstwą mozaiki.
    S, H, Z deterministyczne z cech; alpha=S/(S+H), beta=H/(S+H).
    """
    file_path: str
    S: int
    H: int
    Z: int
    maxZ: int
    alpha: float
    beta: float
    nodes: Dict[int, AstNodeLite]
    labels: List[str]
    per_label: Dict[str, int]


# ──────────────────────────────────────────────────────────────────────────────
# Heurystyki/metryki deterministyczne
# ──────────────────────────────────────────────────────────────────────────────

_CONTROL_NODES = (
    ast.If, ast.For, ast.While, ast.With, ast.Try, ast.Match
)
_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_CALL_NODES = (ast.Call,)
_DATA_NODES = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Yield, ast.YieldFrom)
_IO_NODES = (ast.Raise, ast.Assert, ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom, ast.Expr)
_LEAF_NODES = (ast.Name, ast.Attribute, ast.Constant)

# Wagi wkładu do S/H z typów – wszystkie deterministyczne
_W_S = {
    ast.Module: 1,
    ast.FunctionDef: 3, ast.AsyncFunctionDef: 3, ast.ClassDef: 4,
    ast.If: 2, ast.For: 3, ast.While: 3, ast.With: 2, ast.Try: 3, ast.Match: 3,
    ast.Assign: 2, ast.AnnAssign: 2, ast.AugAssign: 2, ast.Return: 1,
    ast.Call: 1, ast.Import: 1, ast.ImportFrom: 1,
}
_W_H = {
    ast.Name: 1, ast.Attribute: 1, ast.Constant: 1,
    ast.Call: 3, ast.Assign: 2, ast.AnnAssign: 2, ast.AugAssign: 2,
    ast.Import: 2, ast.ImportFrom: 2,
}


def _clamp01(x: float) -> float:
    return 0.0 if x <= 0 else (1.0 if x >= 1.0 else float(x))


def _entropy(proportions: List[float]) -> float:
    """Shannon H na proporcjach etykiet (0..1 z normalizacją)."""
    ps = [p for p in proportions if p > 0]
    if not ps:
        return 0.0
    H = -sum(p * math.log2(p) for p in ps)
    Hmax = math.log2(len(ps)) if len(ps) > 1 else 1.0
    return _clamp01(H / Hmax)


def _label_props(labels: List[str]) -> Dict[str, float]:
    n = max(1, len(labels))
    counts: Dict[str, int] = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return {k: v / n for k, v in counts.items()}


def _node_meta(a: ast.AST, depth: int, siblings: int, label_props_global: Dict[str, float]) -> Tuple[float, float, float, float, float, float]:
    """
    Deterministyczne meta=[L,S,Sel,Stab,Cau,H]:
    - L: "liniowość" ~ maleje przy kontrolach/pętlach; rośnie przy liściach
    - S: udział w strukturze (z wag)
    - Sel: selektywność ~ duża przy warunkach/wyborach, średnia przy wywołaniach
    - Stab: stabilność ~ wyższa dla Assign/definicji/stałych; niższa dla Call/Import
    - Cau: przyczynowość/efekt ~ Call/Return/Assign/Import wysokie
    - H: zróżnicowanie ~ entropia lokalna (po typie) i głębokość
    Wszystko skaluje się do [0..1].
    """
    # typy uproszczone
    is_ctrl = isinstance(a, _CONTROL_NODES)
    is_def = isinstance(a, _DEF_NODES)
    is_call = isinstance(a, _CALL_NODES)
    is_data = isinstance(a, _DATA_NODES)
    is_io = isinstance(a, _IO_NODES)
    is_leaf = isinstance(a, _LEAF_NODES)

    # L
    L = 0.75 if is_leaf else (0.35 if is_ctrl else 0.55)
    # S (normalizowana z wag)
    s_w = 0
    for t, w in _W_S.items():
        if isinstance(a, t):
            s_w = max(s_w, w)
    S = _clamp01(s_w / 4.0)

    # Sel
    if is_ctrl:
        Sel = 0.8
    elif is_call:
        Sel = 0.7
    elif is_def:
        Sel = 0.55
    else:
        Sel = 0.5

    # Stab
    if isinstance(a, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Constant)):
        Stab = 0.8
    elif is_def:
        Stab = 0.75
    elif is_call or is_import(a):
        Stab = 0.45
    else:
        Stab = 0.6

    # Cau
    if is_call or isinstance(a, (ast.Return, ast.Raise, ast.Yield, ast.YieldFrom)):
        Cau = 0.75
    elif is_ctrl:
        Cau = 0.65
    elif is_def:
        Cau = 0.6
    else:
        Cau = 0.5

    # H – entropia globalna etykiet + korekta głębokością i rozgałęzieniem
    lbl = a.__class__.__name__
    p_lbl = label_props_global.get(lbl, 0.0)
    Hglob = _entropy(list(label_props_global.values()))
    dep_term = _clamp01(depth / 12.0)
    sib_term = _clamp01(siblings / 6.0)
    H = _clamp01(0.6 * Hglob + 0.25 * dep_term + 0.15 * sib_term + 0.05 * (1.0 - p_lbl))

    return (_clamp01(L), _clamp01(Sel), _clamp01(S), _clamp01(Stab), _clamp01(Cau), _clamp01(H))


def is_import(a: ast.AST) -> bool:
    return isinstance(a, (ast.Import, ast.ImportFrom))


# ──────────────────────────────────────────────────────────────────────────────
# Główne funkcje indeksujące
# ──────────────────────────────────────────────────────────────────────────────

def ast_summary_of_source(src: str, file_path: str = "<memory>") -> AstSummary:
    """
    Parsuje źródło i tworzy deterministyczny AstSummary:
    - S/H – suma ważona z typów
    - Z – maksymalna zagnieżdżoność węzłów kontrolnych/def
    - alpha/beta – normalizacja S/H
    - nodes – mapa węzłów z meta=[L,S,Sel,Stab,Cau,H]
    """
    tree = ast.parse(src, filename=file_path)

    nodes: Dict[int, AstNodeLite] = {}
    labels: List[str] = []
    S = H = 0
    maxZ = 0
    nid = 0

    # prepass: policz globalne proporcje etykiet (po prostu z surowego DFS)
    tmp_labels: List[str] = []
    for _n in ast.walk(tree):
        tmp_labels.append(_n.__class__.__name__)
    props = _label_props(tmp_labels)

    def add(a: ast.AST, depth: int, parent: Optional[int]) -> int:
        nonlocal nid, S, H, maxZ
        i = nid; nid += 1
        lbl = a.__class__.__name__
        # ile dzieci?
        chs = list(ast.iter_child_nodes(a))
        siblings = len(chs)
        # meta
        L, Sel, Smeta, Stab, Cau, Hmeta = _node_meta(a, depth, siblings, props)

        # akumulacja S/H (z wag deterministycznych)
        for t, w in _W_S.items():
            if isinstance(a, t):
                S += w
                break
        for t, w in _W_H.items():
            if isinstance(a, t):
                H += w
                break

        node = AstNodeLite(
            id=i,
            label=lbl,
            depth=depth,
            parent=parent,
            lineno=getattr(a, "lineno", None),
            col=getattr(a, "col_offset", None),
            meta=(L, Smeta, Sel, Stab, Cau, Hmeta),
        )
        nodes[i] = node
        labels.append(lbl)
        if parent is not None:
            nodes[parent].children.append(i)

        # Z – maks. głębokość dla węzłów sterujących/def
        if isinstance(a, _CONTROL_NODES + _DEF_NODES):
            maxZ = max(maxZ, depth)

        for ch in chs:
            add(ch, depth + 1, i)
        return i

    add(tree, 0, None)

    tot = max(1, S + H)
    alpha = S / tot
    beta = H / tot
    Z = max(1, maxZ)
    return AstSummary(
        file_path=file_path,
        S=int(S),
        H=int(H),
        Z=int(Z),
        maxZ=int(maxZ),
        alpha=float(alpha),
        beta=float(beta),
        nodes=nodes,
        labels=labels,
        per_label=summarize_labels(labels),
    )


def ast_summary_of_file(path: str | Path) -> Optional[AstSummary]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        src = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        src = p.read_text(encoding="latin-1", errors="replace")
    return ast_summary_of_source(src, str(p))


def ast_summary_of_rev(path: str, rev: str = "HEAD") -> Optional[AstSummary]:
    """
    Pobiera plik z rev:path i zwraca jego AstSummary; gdy brak pliku w rev – None.
    """
    src = show_file_at_rev(path, rev)
    if src is None:
        return None
    return ast_summary_of_source(src, f"{rev}:{path}")


def summarize_labels(labels: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for l in labels:
        out[l] = out.get(l, 0) + 1
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Mini-CLI do szybkiego uruchomienia lokalnie
# ──────────────────────────────────────────────────────────────────────────────

def _cli(argv: Optional[List[str]] = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="ast_index", description="Deterministyczny indeks AST → AstSummary")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("file", help="indeks z lokalnego pliku")
    q.add_argument("path")

    r = sub.add_parser("rev", help="indeks z gita (rev:path)")
    r.add_argument("rev")
    r.add_argument("path")

    args = p.parse_args(argv)
    if args.cmd == "file":
        summ = ast_summary_of_file(args.path)
    else:
        summ = ast_summary_of_rev(args.path, args.rev)

    if summ is None:
        print(json.dumps({"ok": False, "error": "file not found in given source"}, indent=2))
        return

    # Zwięzły JSON (bez pełnych nodes, żeby nie zalewać konsoli)
    payload = dict(
        file=summ.file_path,
        S=summ.S, H=summ.H, Z=summ.Z, maxZ=summ.maxZ,
        alpha=round(summ.alpha, 4), beta=round(summ.beta, 4),
        n_nodes=len(summ.nodes),
        top_labels=sorted(summ.per_label.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
