#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glitchlab.analysis.scope_viz — CLI do wizualizacji widoków (soczewka legacy + metasoczewka)
Python 3.9 • stdlib only

Podrozkazy
----------
- lens  : tryb legacy (module|file|func|bus) — deleguje do scope_meta z domyślną konfiguracją
- meta  : tryb metasoczewki — pełna spec (level, center, depth, allowed_*, max_nodes, field, pan)

Artefakty
---------
Wszystkie pliki zapisujemy do: .glx/graphs/

- lens_<kind>_<name>.{json,dot}
- meta_<level>_<name>.{json,dot}

Zależności
----------
- analysis.project_graph (ProjectGraph, build_project_graph, to_dot)
- analysis.scope_meta    (ScopeSpec, FieldFilter, PanSpec, build_meta_view)
- io.artifacts.GlxArtifacts (z fallbackiem, jeśli niedostępny)

Uwaga
-----
Ten moduł jest tylko CLI — bez własnej logiki budowy grafu ani metryk.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

af = GlxArtifacts()

# ── Biblioteki projektowe ─────────────────────────────────────────────────────
from glitchlab.analysis.project_graph import (  # type: ignore
    ProjectGraph,
    build_project_graph,
    to_dot as graph_to_dot,
)
from glitchlab.analysis.scope_meta import (  # type: ignore
    ScopeSpec,
    FieldFilter,
    PanSpec,
    build_meta_view,
)

# ── Utils ─────────────────────────────────────────────────────────────────────

def _safe_name(s: str) -> str:
    s = (s or "auto").strip() or "auto"
    return (
        s.replace("/", "_")
         .replace("\\", "_")
         .replace(":", "_")
         .replace("*", "star")
         .replace(" ", "_")
    )

def _parse_edges(s: Optional[str]) -> Set[str]:
    if not s:
        return {"import", "define", "call", "link", "use", "rpc"}
    return {e.strip() for e in s.split(",") if e.strip()}

def _level_to_allowed_kinds(level: str) -> Set[str]:
    lv = (level or "module").lower()
    if lv == "project": return {"project", "module", "topic"}
    if lv == "module":  return {"module", "topic"}
    if lv == "file":    return {"file", "module", "topic"}
    if lv == "func":    return {"func", "topic"}
    if lv == "bus":     return {"topic", "module", "file"}
    return {"module", "file", "func", "topic"}  # custom/default

def _emit_json_dot(prefix: str, name: str, subgraph: ProjectGraph, meta: Dict) -> Tuple[Path, Path]:
    """
    Zapisuje JSON (graf + _meta) i DOT do .glx/graphs/<prefix>_<name>.{json,dot}.
    """
    fname = f"{prefix}_{_safe_name(name)}"
    payload = {
        "nodes": [
            {"id": n.id, "kind": n.kind, "label": n.label, "meta": n.meta}
            for n in subgraph.nodes.values()
        ],
        "edges": [
            {"src": e.src, "dst": e.dst, "kind": e.kind, **({"weight": e.weight} if e.weight is not None else {})}
            for e in subgraph.edges
        ],
        "_meta": meta,
    }
    j = af.write_json(f"graphs/{fname}.json", payload)
    dot = graph_to_dot(subgraph, title=f"{prefix.upper()} :: {name}")
    d = af.write_text(f"graphs/{fname}.dot", dot)
    return j, d

# ── LENS (legacy) → deleguje do metasoczewki ──────────────────────────────────

def run_lens(kind: str, target: Optional[str], *, depth: int, max_nodes: int, edges: Set[str]) -> int:
    """
    Legacy soczewka: module|file|func|bus — używa metasoczewki z domyślną konfiguracją.
    """
    # heurystycznie ustaw allowed_kinds z level
    level = kind.lower()
    allowed_kinds = _level_to_allowed_kinds(level)

    # Centrum: akceptujemy zarówno „glitchlab/core” jak i „glitchlab.core”
    center: List[str] = []
    if target:
        t = target.strip()
        center = [t, t.replace("/", "."), t.replace(".", "/")]

    spec = ScopeSpec(
        level=level,
        center=center,
        depth=max(0, depth),
        allowed_edges=edges,
        allowed_kinds=allowed_kinds,
        max_nodes=max(1, max_nodes),
        # filtr pola — neutralny (brak odcięcia)
        field=FieldFilter(name="degree", top_k=0, threshold=0.0, normalize=True),
        pan=PanSpec(mode="neighbor", steps=0, field="pagerank", keep_anchors=False),
        fallback=[],
    )

    result = build_meta_view(spec)
    meta = {
        "spec": _jsonify_spec(spec),
        "anchors": result.anchors,
        "metrics": result.metrics,
        "mode": "lens",
    }
    name = f"{level}_{(target or 'auto')}"
    j, d = _emit_json_dot("lens", name, result.graph_sub, meta)
    print(str(j))
    print(str(d))
    return 0

# ── META (pełna metasoczewka) ────────────────────────────────────────────────

def run_meta(
    spec_file: Optional[str],
    *,
    level: Optional[str],
    centers: Optional[List[str]],
    depth: Optional[int],
    max_nodes: Optional[int],
    edges: Optional[str],
    field_name: Optional[str],
    top_k: Optional[int],
    threshold: Optional[float],
    pan_steps: Optional[int],
    pan_field: Optional[str],
    keep_anchors: bool,
) -> int:
    # 1) Bazowa spec z pliku (jeśli podano)
    spec = ScopeSpec()
    if spec_file:
        data = json.loads(Path(spec_file).read_text(encoding="utf-8"))
        # Uwaga: ScopeSpec zawiera zbiory; dopuszczamy listy w JSON
        if "allowed_edges" in data and isinstance(data["allowed_edges"], list):
            data["allowed_edges"] = set(data["allowed_edges"])
        if "allowed_kinds" in data and isinstance(data["allowed_kinds"], list):
            data["allowed_kinds"] = set(data["allowed_kinds"])
        # Field/Pan mogą być dict-ami — zmerguj delikatnie
        spec = ScopeSpec(**{**asdict(spec), **data})

    # 2) Nadpisania z CLI
    if level is not None:
        spec.level = level
        # jeśli w JSON nie wymuszono allowed_kinds, ustaw domyślne z level
        if not spec.allowed_kinds or spec.level != "custom":
            spec.allowed_kinds = _level_to_allowed_kinds(spec.level)
    if centers:
        spec.center = centers
    if depth is not None:
        spec.depth = max(0, depth)
    if max_nodes is not None:
        spec.max_nodes = max(1, max_nodes)
    if edges is not None:
        spec.allowed_edges = _parse_edges(edges)
    # filtr pola
    if field_name is not None or top_k is not None or threshold is not None:
        spec.field = spec.field or FieldFilter()
        if field_name is not None:
            spec.field.name = field_name
        if top_k is not None:
            spec.field.top_k = max(0, int(top_k))
        if threshold is not None:
            spec.field.threshold = max(0.0, min(1.0, float(threshold)))
        # normalize zostaje True (najczęściej oczekiwane)
    # pan
    if pan_steps is not None or pan_field is not None or keep_anchors:
        spec.pan = spec.pan or PanSpec()
        if pan_steps is not None:
            spec.pan.steps = max(0, int(pan_steps))
        if pan_field is not None:
            spec.pan.field = pan_field
        spec.pan.keep_anchors = bool(keep_anchors)

    # 3) Budowa widoku
    result = build_meta_view(spec)
    meta = {
        "spec": _jsonify_spec(spec),
        "anchors": result.anchors,
        "metrics": result.metrics,
        "mode": "meta",
    }

    # Nazwa wynikowa
    name = (spec.center[0] if spec.center else "auto")
    j, d = _emit_json_dot(f"meta_{spec.level}", name, result.graph_sub, meta)
    print(str(j))
    print(str(d))
    return 0

# ── Pomocnicze ────────────────────────────────────────────────────────────────

def _jsonify_spec(spec: ScopeSpec) -> Dict:
    d = asdict(spec)
    # zbiory → listy
    d["allowed_edges"] = sorted(list(d.get("allowed_edges", [])))
    d["allowed_kinds"] = sorted(list(d.get("allowed_kinds", [])))
    return d

# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="scope_viz", description="GLX: soczewki legacy + metasoczewki (CLI)")
    sp = p.add_subparsers(dest="cmd", required=True)

    # lens (legacy)
    sp_lens = sp.add_parser("lens", help="Legacy soczewka (module|file|func|bus)")
    sp_lens.add_argument("kind", choices=["module", "file", "func", "bus"])
    sp_lens.add_argument("target", nargs="?", help="np. glitchlab/core lub ścieżka pliku lub file.py:func")
    sp_lens.add_argument("--depth", type=int, default=1)
    sp_lens.add_argument("--max-nodes", type=int, default=400)
    sp_lens.add_argument("--edges", help="CSV typów krawędzi (import,define,call,link,use,rpc)")

    # meta (pełna spec)
    sp_meta = sp.add_parser("meta", help="Metasoczewka (pełna spec)")
    sp_meta.add_argument("--spec", help="Ścieżka do JSON specyfikacji")
    sp_meta.add_argument("--level", choices=["project", "module", "file", "func", "bus", "custom"])
    sp_meta.add_argument("--center", action="append", help="Wzorzec (glob/regex). Można powtarzać.")
    sp_meta.add_argument("--depth", type=int)
    sp_meta.add_argument("--max-nodes", type=int)
    sp_meta.add_argument("--edges", help="CSV typów krawędzi (import,define,call,link,use,rpc)")
    sp_meta.add_argument("--field", dest="field_name", help="Nazwa pola do filtracji (np. pagerank, degree, churn)")
    sp_meta.add_argument("--top-k", type=int, help="Top-K wg pola")
    sp_meta.add_argument("--threshold", type=float, help="Próg względny pola [0..1]")
    sp_meta.add_argument("--pan-steps", type=int, help="Kroki 'panowania' po sąsiadach wg pola")
    sp_meta.add_argument("--pan-field", help="Pole sterujące 'pan' (np. pagerank, degree)")
    sp_meta.add_argument("--keep-anchors", action="store_true", help="Zachowaj poprzednie anchory przy 'pan'")

    args = p.parse_args(argv)

    if args.cmd == "lens":
        return run_lens(
            kind=args.kind,
            target=args.target,
            depth=int(args.depth or 0),
            max_nodes=int(args.max_nodes or 400),
            edges=_parse_edges(args.edges),
        )

    if args.cmd == "meta":
        return run_meta(
            spec_file=args.spec,
            level=args.level,
            centers=args.center,
            depth=args.depth,
            max_nodes=args.max_nodes,
            edges=args.edges,
            field_name=args.field_name,
            top_k=args.top_k,
            threshold=args.threshold,
            pan_steps=args.pan_steps,
            pan_field=args.pan_field,
            keep_anchors=bool(args.keep_anchors),
        )

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
