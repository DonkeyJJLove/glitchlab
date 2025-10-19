#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glitchlab.analysis.scope_meta — MetaSoczewka (silnik)
Python 3.9 • stdlib only

Cel
----
Elastyczny mechanizm wycinania „okna widoku” (metasoczewki) nad globalnym
grafem projektu z filtracją po „polach operacyjnych” i prostym pan/zoom.
Zapisuje artefakty do .glx/graphs/meta_<level>_<name>.{json,dot}.

We
---
- ProjectGraph (analysis.project_graph)
- resolve_field(...) (analysis.fields) — pola operacyjne
- io.artifacts.GlxArtifacts — zapis artefaktów (z fallbackiem)

Wy
---
- ScopeSpec           — specyfikacja widoku
- ScopeResult         — wynik (podgraf, anchory, metryki)
- build_meta_view(...)        — buduje widok
- export_meta_view(...)       — zapisuje JSON/DOT (Graphviz)
"""

from __future__ import annotations

import json
import fnmatch
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Iterable, Callable

# ── Artefakty (fallback, jeśli moduł nieobecny) ───────────────────────────────
try:
    from glitchlab.io.artifacts import GlxArtifacts  # type: ignore
except Exception:  # pragma: no cover
    class GlxArtifacts:  # type: ignore
        def __init__(self) -> None:
            self.repo_root = Path(__file__).resolve().parents[3]
            self.base = self.repo_root / "glitchlab" / ".glx"
            self.base.mkdir(parents=True, exist_ok=True)
        def write_text(self, rel_name: str, text: str) -> Path:
            p = (self.base / rel_name).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            return p
        def write_json(self, rel_name: str, obj) -> Path:
            return self.write_text(rel_name, json.dumps(obj, ensure_ascii=False, indent=2))
        def read_json(self, rel_name: str) -> dict:
            p = (self.base / rel_name).resolve()
            return json.loads(p.read_text(encoding="utf-8"))
        def path(self, rel_name: str) -> Path:
            return (self.base / rel_name).resolve()

af = GlxArtifacts()

# ── Projektowy graf i DOT ────────────────────────────────────────────────────
from glitchlab.analysis.project_graph import (  # type: ignore
    ProjectGraph,
    Node,
    Edge,
    build_project_graph,
    to_dot as project_to_dot,
)

# ── Pola operacyjne (fields) ────────────────────────────────────────────────
from glitchlab.analysis.fields import resolve_field, FieldSpec  # type: ignore

__all__ = [
    "FieldFilter",
    "PanSpec",
    "ScopeSpec",
    "ScopeResult",
    "build_meta_view",
    "export_meta_view",
]

# =============================================================================
# Specyfikacje i wynik
# =============================================================================

@dataclass
class FieldFilter:
    """Filtr pola operacyjnego w metasoczewce."""
    name: str = "degree"         # nazwa pola z analysis.fields (np. pagerank, betweenness_approx, churn, itp.)
    top_k: int = 0               # jeżeli >0: wybierz top-K względem wartości pola
    threshold: float = 0.0       # próg względny (do max), w [0,1]; 0.4 → zostaw >= 40% max
    normalize: bool = True       # czy normalizować pole (dla filtracji zwykle True)


@dataclass
class PanSpec:
    """Parametry „panowania” — przesunięcia soczewki po grafie."""
    mode: str = "neighbor"       # "neighbor" (po sąsiadach)
    steps: int = 0               # liczba kroków panningu (0 = brak)
    field: str = "pagerank"      # pole sterujące wyborem sąsiadów (np. pagerank/degree/churn)
    keep_anchors: bool = False   # True: dokładaj nowych anchorów zamiast zastępować


@dataclass
class ScopeSpec:
    """
    Spec metasoczewki:
      - level: semantyczny poziom (project/module/file/func/bus/custom) → domyślny zestaw allowed_kinds
      - center: lista wzorców (glob/regex) dopasowywanych do id/label/path
      - depth: promień BFS (liczba hopów)
      - allowed_edges/kinds: ograniczenia typów krawędzi/węzłów (można nadpisać dla 'custom')
      - max_nodes: budżet wielkości okna
      - field: filtr pola (top-K / próg)
      - pan: parametry panowania
      - fallback: kolejność alternatywnych poziomów, gdy brak wyników
    """
    level: str = "module"                              # project|module|file|func|bus|custom
    center: List[str] = field(default_factory=list)    # wzorce id/label/path
    depth: int = 1
    allowed_edges: Set[str] = field(default_factory=lambda: {"import", "define", "call", "link", "use", "rpc"})
    allowed_kinds: Set[str] = field(default_factory=lambda: {"module", "file", "func", "topic"})
    max_nodes: int = 400
    field: Optional[FieldFilter] = field(default_factory=FieldFilter)
    pan: Optional[PanSpec] = field(default_factory=PanSpec)
    fallback: List[str] = field(default_factory=lambda: ["module", "file", "func"])


@dataclass
class ScopeResult:
    """Wynik metasoczewki."""
    graph_sub: ProjectGraph
    anchors: List[str]
    metrics: Dict[str, object]  # nodes_total, edges_total, by_kind, AST_S/H/Z (raw), itp.


# =============================================================================
# Pomocnicze: dopasowania, sąsiedzi, BFS, filtry
# =============================================================================

def _glob_or_regex(pattern: str, text: str) -> bool:
    if not pattern:
        return False
    return fnmatch.fnmatch(text, pattern) or bool(re.search(pattern, text))


def _match_node(n: Node, patterns: List[str]) -> bool:
    if not patterns:
        return False
    hay = [n.id, n.label, str(n.meta.get("path", ""))]
    return any(any(_glob_or_regex(p, h) for h in hay) for p in patterns)


def _neighbors(G: ProjectGraph, nid: str, allowed_edges: Set[str]) -> List[str]:
    out: List[str] = []
    for e in G.edges:
        if e.kind not in allowed_edges:
            continue
        if e.src == nid:
            out.append(e.dst)
        elif e.dst == nid:
            out.append(e.src)
    return out


def _degree(G: ProjectGraph, nid: str, allowed_edges: Set[str]) -> int:
    d = 0
    for e in G.edges:
        if e.kind not in allowed_edges:
            continue
        if e.src == nid or e.dst == nid:
            d += 1
    return d


def _default_allowed_kinds_for_level(level: str) -> Set[str]:
    lv = (level or "module").lower()
    if lv == "project":
        return {"project", "module", "topic"}
    if lv == "module":
        return {"module", "topic"}
    if lv == "file":
        return {"file", "module", "topic"}
    if lv == "func":
        return {"func", "topic"}
    if lv == "bus":
        return {"topic", "module", "file"}
    return {"module", "file", "func", "topic"}  # custom/domyslnie


def _pick_anchors(G: ProjectGraph, spec: ScopeSpec) -> List[str]:
    """Wybierz anchory: dopasowanie wzorców lub fallback: top-degree w preferowanych kinds."""
    # 1) dopasowanie wzorców
    cand = [nid for nid, n in G.nodes.items() if n.kind in spec.allowed_kinds and _match_node(n, spec.center)]
    if cand:
        # deterministycznie (po id) — stabilność artefaktów
        return sorted(cand)

    # 2) fallback: najwyższy stopień (po allowed_edges), preferuj module/file/func/topic
    prefer = ["module", "file", "func", "topic", "project"]
    for k in prefer:
        nodes = [nid for nid, n in G.nodes.items() if n.kind == k]
        if not nodes:
            continue
        nodes.sort(key=lambda x: _degree(G, x, spec.allowed_edges), reverse=True)
        if nodes:
            return [nodes[0]]
    # 3) skrajny fallback — pierwszy węzeł, jeśli graf pustawy
    return list(G.nodes.keys())[:1]


def _bfs_window(G: ProjectGraph, starts: List[str], spec: ScopeSpec) -> Set[str]:
    """Ograniczony BFS z filtracją typów, budżetem i promieniem."""
    if not starts:
        return set()
    visited: Set[str] = set()
    frontier = list(starts)
    depth = 0
    while frontier and len(visited) < spec.max_nodes and depth <= spec.depth:
        nxt: List[str] = []
        for nid in frontier:
            if nid in visited:
                continue
            node = G.nodes.get(nid)
            if not node or node.kind not in spec.allowed_kinds:
                continue
            visited.add(nid)
            for m in _neighbors(G, nid, spec.allowed_edges):
                if m not in visited:
                    nxt.append(m)
            if len(visited) >= spec.max_nodes:
                break
        frontier = nxt
        depth += 1
    return visited


def _apply_field_filter(
    G: ProjectGraph,
    keep: Set[str],
    spec: ScopeSpec,
    resolve: Callable[[FieldSpec], Dict[str, float]],
) -> Set[str]:
    """Filtr pola operacyjnego (top-K / próg)."""
    if not spec.field or not keep:
        return keep

    fs = FieldSpec(name=spec.field.name, normalize=spec.field.normalize)
    values = resolve(fs)  # node_id -> float (zwykle [0,1])
    # ogranicz do keep
    vals = {nid: float(values.get(nid, 0.0)) for nid in keep}

    # próg względem max
    if spec.field.threshold > 0.0 and vals:
        vmax = max(vals.values())
        if vmax > 0.0:
            vals = {nid: v for nid, v in vals.items() if (v / vmax) >= spec.field.threshold}
        else:
            vals = {}

    # top-K
    if spec.field.top_k > 0 and len(vals) > spec.field.top_k:
        order = sorted(vals.items(), key=lambda t: t[1], reverse=True)
        cutoff_ids = {nid for nid, _ in order[: spec.field.top_k]}
        vals = {nid: vals[nid] for nid in cutoff_ids}

    return set(vals.keys())


def _pan_anchors(
    G: ProjectGraph,
    anchors: List[str],
    pan: PanSpec,
    allowed_edges: Set[str],
    resolve: Callable[[FieldSpec], Dict[str, float]],
) -> List[str]:
    """Przesuń anchory po grafie po sąsiadach maksymalizujących wskazane pole."""
    if not anchors or pan.steps <= 0:
        return anchors

    field_map = resolve(FieldSpec(name=pan.field, normalize=True))
    cur: Set[str] = set(anchors)
    for _ in range(max(0, pan.steps)):
        nxt: Set[str] = set(cur) if pan.keep_anchors else set()
        for a in list(cur):
            nbrs = _neighbors(G, a, allowed_edges)
            if not nbrs:
                if pan.keep_anchors:
                    nxt.add(a)
                continue
            # wybierz sąsiada o najwyższym polu
            best = max(nbrs, key=lambda x: float(field_map.get(x, 0.0)))
            nxt.add(best)
        cur = nxt
    # deterministycznie
    return sorted(cur)


# =============================================================================
# Podgraf i metryki
# =============================================================================

def _subgraph_from_ids(G: ProjectGraph, ids: Set[str]) -> ProjectGraph:
    out = ProjectGraph()
    for nid in ids:
        if nid in G.nodes:
            n = G.nodes[nid]
            out.add_node(n.id, n.kind, n.label, **dict(n.meta))
    for e in G.edges:
        if e.src in ids and e.dst in ids:
            out.add_edge(e.src, e.dst, e.kind, weight=e.weight)
    return out


def _aggregate_counts_by_kind(G: ProjectGraph) -> Dict[str, int]:
    by_kind: Dict[str, int] = {}
    for n in G.nodes.values():
        by_kind[n.kind] = by_kind.get(n.kind, 0) + 1
    return by_kind


def _sum_field_on_subgraph(
    sub: ProjectGraph,
    field_name: str,
    resolve: Callable[[FieldSpec], Dict[str, float]],
) -> float:
    """Suma *nienormalizowanych* wartości danego pola na podgrafie."""
    vals = resolve(FieldSpec(name=field_name, normalize=False))
    total = 0.0
    for nid in sub.nodes.keys():
        total += float(vals.get(nid, 0.0))
    return float(total)


# =============================================================================
# API główne
# =============================================================================

def build_meta_view(
    spec: ScopeSpec,
    graph: Optional[ProjectGraph] = None,
    *,
    resolve: Callable[[FieldSpec], Dict[str, float]] = None,
) -> ScopeResult:
    """
    Buduje widok metasoczewki na globalnym grafie:
      1) ustalenie allowed_kinds z level (jeśli nie 'custom'),
      2) wybór anchorów (center → wzorce; fallback: top-degree),
      3) opcjonalny pan po sąsiadach wg wskazanego pola,
      4) BFS okno (depth/max_nodes/allowed_*),
      5) filtracja pola (threshold/top-K),
      6) podgraf + metryki agregowane (by_kind, AST_S/H/Z).
    """
    G = graph or _load_or_build_graph()
    if resolve is None:
        # opakowanie analysis.fields.resolve_field na sygnaturę: FieldSpec -> dict
        def _default_resolver(fs: FieldSpec) -> Dict[str, float]:
            return resolve_field(fs)
        resolve = _default_resolver

    # 1) allowed_kinds dla level (o ile spec nie jest 'custom' — i nie nadpisano ręcznie)
    if spec.level != "custom" and not spec.allowed_kinds:
        spec.allowed_kinds = _default_allowed_kinds_for_level(spec.level)
    else:
        # nawet dla custom — zadbaj o niepusty zbiór
        spec.allowed_kinds = spec.allowed_kinds or _default_allowed_kinds_for_level("custom")

    # 2) anchory
    anchors = _pick_anchors(G, spec)

    # 3) pan
    if spec.pan and spec.pan.steps > 0:
        anchors = _pan_anchors(G, anchors, spec.pan, spec.allowed_edges, resolve)

    # 4) BFS okno
    keep_ids = _bfs_window(G, anchors, spec)

    # 5) filtr pola
    keep_ids = _apply_field_filter(G, keep_ids, spec, resolve)

    # Fallback na innych levelach, gdy nadal pusto
    if not keep_ids and spec.fallback:
        for lvl in spec.fallback:
            s2 = ScopeSpec(
                level=lvl,
                center=spec.center,
                depth=spec.depth,
                allowed_edges=set(spec.allowed_edges),
                allowed_kinds=_default_allowed_kinds_for_level(lvl),
                max_nodes=spec.max_nodes,
                field=spec.field,
                pan=spec.pan,
                fallback=[],
            )
            anchors = _pick_anchors(G, s2)
            if s2.pan and s2.pan.steps > 0:
                anchors = _pan_anchors(G, anchors, s2.pan, s2.allowed_edges, resolve)
            keep_ids = _bfs_window(G, anchors, s2)
            keep_ids = _apply_field_filter(G, keep_ids, s2, resolve)
            if keep_ids:
                break

    # 6) Podgraf i metryki
    sub = _subgraph_from_ids(G, keep_ids)
    metrics = {
        "nodes_total": len(sub.nodes),
        "edges_total": len(sub.edges),
        "by_kind": _aggregate_counts_by_kind(sub),
        # AST: sumy „surowe” (nie znormalizowane)
        "AST_S": _sum_field_on_subgraph(sub, "ast_S", resolve),
        "AST_H": _sum_field_on_subgraph(sub, "ast_H", resolve),
        "AST_Z": _sum_field_on_subgraph(sub, "ast_Z", resolve),
    }

    return ScopeResult(graph_sub=sub, anchors=anchors, metrics=metrics)


def export_meta_view(
    result: ScopeResult,
    *,
    spec: ScopeSpec,
    name_hint: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Zapis JSON/DOT do .glx/graphs/meta_<level>_<name>.{json,dot}.
    Zwraca (json_path, dot_path).
    """
    level = (spec.level or "custom").lower()
    name = (name_hint or (spec.center[0] if spec.center else "auto")).strip() or "auto"
    safe = (
        name.replace("/", "_")
        .replace(":", "_")
        .replace("*", "star")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    # JSON (graf + meta)
    payload = {
        "nodes": [
            {"id": n.id, "kind": n.kind, "label": n.label, "meta": n.meta}
            for n in result.graph_sub.nodes.values()
        ],
        "edges": [
            {"src": e.src, "dst": e.dst, "kind": e.kind, **({"weight": e.weight} if e.weight is not None else {})}
            for e in result.graph_sub.edges
        ],
        "_meta": {
            "spec": _spec_to_json(spec),
            "anchors": result.anchors,
            "metrics": result.metrics,
        },
    }
    json_path = af.write_json(f"graphs/meta_{level}_{safe}.json", payload)

    # DOT
    title = f"Meta Lens: {level} [{safe}]"
    dot = project_to_dot(result.graph_sub, title=title)
    dot_path = af.write_text(f"graphs/meta_{level}_{safe}.dot", dot)

    return json_path, dot_path


# =============================================================================
# Narzędzia wewnętrzne
# =============================================================================

def _load_or_build_graph() -> ProjectGraph:
    """
    Spróbuj wczytać .glx/graphs/project_graph.json; jeżeli brak — zbuduj.
    """
    try:
        data = af.read_json("graphs/project_graph.json")
        return ProjectGraph.from_json_obj(Path("."), data)  # repo_root nie jest tu krytyczny
    except Exception:
        return build_project_graph()


def _spec_to_json(spec: ScopeSpec) -> dict:
    d = asdict(spec)
    # FieldFilter / PanSpec jako słowniki (już są przez asdict)
    # Zamień sety na listy — JSON-friendly
    d["allowed_edges"] = sorted(list(d.get("allowed_edges", [])))
    d["allowed_kinds"] = sorted(list(d.get("allowed_kinds", [])))
    return d


# =============================================================================
# (Opcjonalnie) mini-CLI podglądowe — pełny CLI jest w scope_viz.py
# =============================================================================

if __name__ == "__main__":  # pragma: no cover
    import argparse, sys
    p = argparse.ArgumentParser(prog="scope_meta", description="MetaLens (quick preview/emit)")
    p.add_argument("--level", default="module", choices=["project","module","file","func","bus","custom"])
    p.add_argument("--center", action="append", help="Wzorzec (glob/regex) — można wielokrotnie")
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--max-nodes", type=int, default=400)
    p.add_argument("--field", default="degree", help="Nazwa pola do filtracji (np. pagerank, churn, ...)")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--pan-steps", type=int, default=0)
    p.add_argument("--pan-field", default="pagerank")
    p.add_argument("--keep-anchors", action="store_true")
    args = p.parse_args()

    spec = ScopeSpec(
        level=args.level,
        center=args.center or [],
        depth=args.depth,
        max_nodes=args.max_nodes,
        field=FieldFilter(name=args.field, top_k=args.top_k, threshold=args.threshold, normalize=True),
        pan=PanSpec(mode="neighbor", steps=args.pan_steps, field=args.pan_field, keep_anchors=args.keep_anchors),
    )

    res = build_meta_view(spec)
    j, d = export_meta_view(res, spec=spec)
    print(str(j))
    print(str(d))
    sys.exit(0)
