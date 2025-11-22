#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glitchlab.analysis.meta_lens — metasoczewka (elastyczne okno widoku)
Python 3.9 • stdlib only

Działa na kanonicznym grafie z analysis.project_graph.ProjectGraph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import math
import re as _re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

from glitchlab.analysis.project_graph import ProjectGraph, Node  # kanoniczny graf

# (opcjonalnie) metryki AST – łagodny import
try:
    from glitchlab.analysis.ast_index import ast_summary_of_file  # type: ignore
except Exception:  # pragma: no cover
    ast_summary_of_file = None  # type: ignore


NodeKind = str  # 'project'|'module'|'file'|'func'|'topic'|...


@dataclass
class ScopeSpec:
    level: NodeKind = "module"  # project|module|file|func|bus|custom
    center: list[str] = field(default_factory=list)  # glob/regex (id/label/path)
    depth: int = 1  # hopy BFS
    allowed_edges: set[str] = field(default_factory=lambda: {"import", "define", "call", "link", "use", "rpc"})
    allowed_kinds: set[str] = field(default_factory=lambda: {"module", "file", "func", "topic"})
    max_nodes: int = 400
    prefer_kinds: list[str] = field(default_factory=lambda: ["module", "file", "func", "topic"])
    fallback: list[str] = field(default_factory=lambda: ["module", "file", "func"])
    field: dict | None = None  # {"name":"degree|edge_density", "top_k":120, "threshold":0.4}
    pan: dict | None = None    # future: {"mode":"neighbor|gradient","steps":1}


@dataclass
class ScopeResult:
    graph: ProjectGraph
    anchors: list[str]
    metrics: dict


def _glob_or_regex(pattern: str, text: str) -> bool:
    if not pattern:
        return False
    return fnmatch.fnmatch(text, pattern) or bool(_re.search(pattern, text))


def _match_node(node: Node, patterns: list[str]) -> bool:
    if not patterns:
        return False
    hay = [node.id, node.label, str(node.meta.get("path", ""))]
    return any(any(_glob_or_regex(p, h) for h in hay) for p in patterns)


def _neighbors(g: ProjectGraph, nid: str, allowed: set[str]) -> list[str]:
    out: list[str] = []
    for e in g.edges:
        if e.kind not in allowed:
            continue
        if e.src == nid:
            out.append(e.dst)
        elif e.dst == nid:
            out.append(e.src)
    return out


def _degree(g: ProjectGraph, nid: str, allowed: set[str]) -> int:
    d = 0
    for e in g.edges:
        if e.kind not in allowed:
            continue
        if e.src == nid or e.dst == nid:
            d += 1
    return d


def _field_score(name: str, g: ProjectGraph, nid: str, allowed: set[str]) -> float:
    if name == "degree":
        return float(_degree(g, nid, allowed))
    if name == "edge_density":
        N = max(1, len(g.nodes))
        return float(_degree(g, nid, allowed)) / max(1.0, math.log(N + 1.0))
    return 0.0


def _pick_anchors(g: ProjectGraph, spec: ScopeSpec) -> list[str]:
    cand = [nid for nid, n in g.nodes.items() if n.kind in spec.allowed_kinds and _match_node(n, spec.center)]
    if cand:
        return sorted(cand)
    for k in spec.prefer_kinds:
        nodes = [nid for nid, n in g.nodes.items() if n.kind == k]
        if nodes:
            # deterministycznie – po stopniu, tie-break po id
            scored = sorted(nodes, key=lambda x: (_degree(g, x, spec.allowed_edges), x), reverse=True)
            return [scored[0]]
    return []


def _bfs_window(g: ProjectGraph, starts: list[str], spec: ScopeSpec) -> set[str]:
    if not starts:
        return set()
    visited: set[str] = set()
    frontier = list(starts)
    depth = 0
    while frontier and len(visited) < spec.max_nodes and depth <= spec.depth:
        nxt: list[str] = []
        for nid in frontier:
            if nid in visited:
                continue
            node = g.nodes.get(nid)
            if not node or node.kind not in spec.allowed_kinds:
                continue
            visited.add(nid)
            for m in _neighbors(g, nid, spec.allowed_edges):
                if m not in visited:
                    nxt.append(m)
            if len(visited) >= spec.max_nodes:
                break
        frontier = nxt
        depth += 1
    return visited


def _apply_field_filter(g: ProjectGraph, keep: set[str], spec: ScopeSpec) -> set[str]:
    if not spec.field or not keep:
        return keep
    name = str(spec.field.get("name", "") or "degree")
    top_k = int(spec.field.get("top_k") or 0)
    thr = float(spec.field.get("threshold") or 0.0)
    scored = [(nid, _field_score(name, g, nid, spec.allowed_edges)) for nid in keep]
    if thr > 0.0:
        mx = max((s for _, s in scored), default=1.0)
        keep = {nid for nid, s in scored if (mx and s / mx >= thr)}
    if top_k > 0 and len(keep) > top_k:
        scored.sort(key=lambda t: (t[1], t[0]), reverse=True)
        keep = {nid for nid, _ in scored[:top_k]}
    return keep


def _subgraph_from_ids(g: ProjectGraph, ids: set[str]) -> ProjectGraph:
    sub = ProjectGraph(repo_root=g.repo_root)
    for nid in ids:
        n = g.nodes.get(nid)
        if n:
            sub.add_node(n.id, n.kind, n.label, **n.meta)
    for e in g.edges:
        if e.src in ids and e.dst in ids:
            sub.add_edge(e.src, e.dst, e.kind, e.weight)
    sub.meta = {**g.meta, "lens_nodes": len(sub.nodes), "lens_edges": len(sub.edges)}
    return sub


def _aggregate_ast_metrics(g: ProjectGraph) -> dict:
    if ast_summary_of_file is None:
        return {}
    S = H = Z = 0
    files = [n.meta.get("path") for n in g.nodes.values() if n.kind == "file" and n.meta.get("path")]
    for p in files:
        try:
            s = ast_summary_of_file(Path(p))  # type: ignore
            if s:
                S += int(s.S)
                H += int(s.H)
                Z += int(s.Z)
        except Exception:
            pass
    return {"AST_S": S, "AST_H": H, "AST_Z": Z}


def build_meta_lens(global_graph: ProjectGraph, spec: ScopeSpec) -> ScopeResult:
    # domyślne allowed_kinds wg level (o ile nie custom)
    if spec.level != "custom":
        if spec.level == "project":
            spec.allowed_kinds = {"module", "topic", "project"}
        elif spec.level == "module":
            spec.allowed_kinds = {"module", "topic"}
        elif spec.level == "file":
            spec.allowed_kinds = {"file", "module", "topic"}
        elif spec.level == "func":
            spec.allowed_kinds = {"func", "topic"}
        elif spec.level == "bus":
            spec.allowed_kinds = {"topic", "module"}

    anchors = _pick_anchors(global_graph, spec)
    keep = _bfs_window(global_graph, anchors, spec)
    keep = _apply_field_filter(global_graph, keep, spec)

    if not keep:
        for lvl in spec.fallback:
            s2 = ScopeSpec(**{**spec.__dict__, "level": lvl})
            if lvl == "project":
                s2.allowed_kinds = {"module", "topic", "project"}
            elif lvl == "module":
                s2.allowed_kinds = {"module", "topic"}
            elif lvl == "file":
                s2.allowed_kinds = {"file", "module", "topic"}
            elif lvl == "func":
                s2.allowed_kinds = {"func", "topic"}
            elif lvl == "bus":
                s2.allowed_kinds = {"topic", "module"}
            anchors = _pick_anchors(global_graph, s2)
            keep = _bfs_window(global_graph, anchors, s2)
            keep = _apply_field_filter(global_graph, keep, s2)
            if keep:
                break

    sub = _subgraph_from_ids(global_graph, keep)

    by_kind: Dict[str, int] = {}
    for n in sub.nodes.values():
        by_kind[n.kind] = by_kind.get(n.kind, 0) + 1
    metrics = {"nodes_total": len(sub.nodes), "edges_total": len(sub.edges), "by_kind": by_kind}
    metrics.update(_aggregate_ast_metrics(sub))

    return ScopeResult(graph=sub, anchors=anchors, metrics=metrics)
