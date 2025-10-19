# glitchlab/analysis/project_graph.py
# -*- coding: utf-8 -*-
"""
ProjectGraph — globalny graf projektu (module/file/func + import/call/define).
Python 3.9, stdlib only (opcjonalnie używa glitchlab.analysis.ast_index jeżeli dostępny).

Użycie:
  # z katalogu, który JEST rootem repo (tu: glitchlab/)
  python -m glitchlab.analysis.project_graph build --repo-root . --write

  # alias --repo działa tak samo
  python -m glitchlab.analysis.project_graph build --repo .

Efekt:
  .glx/graphs/project_graph.json  (wypełniony nodes/edges)
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Mapping, Any, Set

# ──────────────────────────────────────────────────────────────────────────────
# Opcjonalna integracja z ast_index (jeśli masz nasz moduł)
# ──────────────────────────────────────────────────────────────────────────────
try:
    # Unikamy zależności twardej – jeśli jest, skorzystamy
    from glitchlab.analysis.ast_index import ast_index_of_file  # type: ignore
except Exception:  # pragma: no cover
    ast_index_of_file = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Modele
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PGNode:
    id: str
    kind: str        # module|file|func|topic
    label: str
    meta: Dict[str, Any]

@dataclass
class PGEdge:
    src: str
    dst: str
    kind: str        # import|call|define|use|link|rpc

@dataclass
class ProjectGraph:
    version: str
    meta: Dict[str, Any]
    nodes: Dict[str, PGNode]
    edges: List[PGEdge]


# ──────────────────────────────────────────────────────────────────────────────
# Skany plików i filtry
# ──────────────────────────────────────────────────────────────────────────────

_IGNORE_FRAGMENTS = {
    "/.git/", "/.glx/", "/__pycache__/", "/.venv/", "/venv/",
    "/node_modules/", "/build/", "/dist/", "/.idea/", "/.pytest_cache/",
}

def _should_skip_file(p: Path) -> bool:
    s = "/" + p.as_posix() + "/"
    for frag in _IGNORE_FRAGMENTS:
        if frag in s:
            return True
    return False

def iter_py_files(repo_root: Path) -> List[Path]:
    files: List[Path] = []
    for p in repo_root.rglob("*.py"):
        # pomiń wszystko „ignorable”
        if _should_skip_file(p.parent):
            continue
        files.append(p)
    # deterministycznie
    files.sort(key=lambda x: x.as_posix())
    return files


# ──────────────────────────────────────────────────────────────────────────────
# Parsowanie AST (fallback lekki, gdy brak analysis.ast_index)
# ──────────────────────────────────────────────────────────────────────────────

class _LiteVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.funcs: Dict[str, Tuple[int, int]] = {}
        self.calls: List[Tuple[str, str, int]] = []  # caller, callee, line
        self.imports: Set[str] = set()
        self.scope: List[str] = []

    def _scope(self) -> str:
        return ".".join(self.scope) if self.scope else "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        q = ".".join([*self.scope, node.name]) if self.scope else node.name
        self.funcs[q] = (node.lineno, node.col_offset)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = ""
        f = node.func
        if isinstance(f, ast.Name):
            callee = f.id
        elif isinstance(f, ast.Attribute):
            # a.b.c -> weźmy końcówkę
            while isinstance(f, ast.Attribute):
                last = f.attr
                f = f.value
            callee = last
        caller = self._scope()
        self.calls.append((caller, callee, getattr(node, "lineno", -1)))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for n in node.names:
            self.imports.add(n.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module.split(".")[0])


def _parse_file_lite(path: Path) -> Tuple[Dict[str, Tuple[int, int]], List[Tuple[str, str, int]], Set[str]]:
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src, filename=str(path))
    except Exception:
        return {}, [], set()
    v = _LiteVisitor()
    v.visit(tree)
    return v.funcs, v.calls, v.imports


# ──────────────────────────────────────────────────────────────────────────────
# Identyfikatory węzłów
# ──────────────────────────────────────────────────────────────────────────────

def _module_id_from_path(repo_root: Path, p: Path) -> str:
    """
    Relatywnie do repo_root:  a/b/c.py -> a.b.c
    Jeżeli plik leży poza rootem (edge-case) – „best effort” z nazwy pliku.
    """
    try:
        rel = p.relative_to(repo_root).as_posix()
    except Exception:
        rel = p.name
    if rel.endswith(".py"):
        rel = rel[:-3]
    # __init__.py → moduł katalogu
    if rel.endswith("/__init__"):
        rel = rel[: -len("/__init__")]
    return rel.replace("/", ".").strip(".")


# ──────────────────────────────────────────────────────────────────────────────
# Budowa grafu
# ──────────────────────────────────────────────────────────────────────────────

def _sha256(obj: Any) -> str:
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def build_project_graph(repo_root: Path) -> ProjectGraph:
    repo_root = Path(repo_root).resolve()
    files = iter_py_files(repo_root)

    nodes: Dict[str, PGNode] = {}
    edges: List[PGEdge] = []

    def add_node(id_: str, kind: str, label: str, **meta) -> None:
        if id_ not in nodes:
            nodes[id_] = PGNode(id=id_, kind=kind, label=label, meta=dict(meta))

    def add_edge(src: str, dst: str, kind: str) -> None:
        edges.append(PGEdge(src=src, dst=dst, kind=kind))

    for fp in files:
        mod = _module_id_from_path(repo_root, fp)
        mod_id = f"module:{mod}"
        file_id = f"file:{fp.as_posix()}"

        add_node(mod_id, "module", mod, path=fp.as_posix())
        add_node(file_id, "file", fp.name, path=fp.as_posix())
        add_edge(mod_id, file_id, "define")

        if ast_index_of_file is not None:
            # bogatszy tryb (jeżeli nasz index jest zainstalowany)
            try:
                idx = ast_index_of_file(str(fp))  # type: ignore
            except Exception:
                idx = None
            if idx:
                # funkcje
                for q, d in (idx.defs or {}).items():  # type: ignore[attr-defined]
                    fn_id = f"func:{mod}:{q}"
                    add_node(fn_id, "func", f"{q}()", path=fp.as_posix(), line=getattr(d, "lineno", None))
                    add_edge(file_id, fn_id, "define")
                # wywołania
                for c in (idx.calls or []):  # type: ignore[attr-defined]
                    caller = c.scope if getattr(c, "scope", "") else "<module>"
                    src_id = f"func:{mod}:{caller}" if caller != "<module>" else file_id
                    dst_id = f"name:{getattr(c, 'func', '')}"
                    add_node(dst_id, "func", getattr(c, "func", ""))
                    add_edge(src_id, dst_id, "call")
                # importy
                for im in (idx.imports or []):  # type: ignore[attr-defined]
                    base = getattr(im, "what", "").split(".")[0]
                    if base:
                        tgt = f"module:{base}"
                        add_node(tgt, "module", base)
                        add_edge(mod_id, tgt, "import")
                continue  # gotowe dla tego pliku

        # fallback lekki (bez ast_index)
        funcs, calls, imports = _parse_file_lite(fp)
        for q, (lin, _col) in funcs.items():
            fn_id = f"func:{mod}:{q}"
            add_node(fn_id, "func", f"{q}()", path=fp.as_posix(), line=lin)
            add_edge(file_id, fn_id, "define")
        for caller, callee, _line in calls:
            src_id = f"func:{mod}:{caller}" if caller != "<module>" else file_id
            dst_id = f"name:{callee}"
            add_node(dst_id, "func", callee)
            add_edge(src_id, dst_id, "call")
        for imp in sorted(imports):
            tgt = f"module:{imp}"
            add_node(tgt, "module", imp)
            add_edge(mod_id, tgt, "import")

    meta = {
        "roots": [repo_root.as_posix()],
        "generated_by": "analysis.project_graph",
        "graph_version": "v1",
        "repo_root": repo_root.as_posix(),
        "nodes_count": len(nodes),
        "edges_count": len(edges),
    }

    # hash deterministyczny (z serializacji transportowej)
    graph_for_hash = {
        "version": "v1",
        "meta": meta,
        "nodes": [
            {"id": n.id, "kind": n.kind, "label": n.label, "meta": n.meta}
            for n in sorted(nodes.values(), key=lambda x: x.id)
        ],
        "edges": [
            {"src": e.src, "dst": e.dst, "kind": e.kind}
            for e in sorted(edges, key=lambda e: (e.src, e.dst, e.kind))
        ],
    }
    meta["graph_hash"] = _sha256(graph_for_hash)

    return ProjectGraph(version="v1", meta=meta, nodes=nodes, edges=edges)


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def _glx_dir(repo_root: Path) -> Path:
    p = Path(repo_root) / ".glx" / "graphs"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_json(g: ProjectGraph) -> Dict[str, Any]:
    return {
        "version": g.version,
        "meta": g.meta,
        "nodes": [
            {"id": n.id, "kind": n.kind, "label": n.label, "meta": n.meta}
            for n in sorted(g.nodes.values(), key=lambda x: x.id)
        ],
        "edges": [{"src": e.src, "dst": e.dst, "kind": e.kind} for e in g.edges],
    }

def save_project_graph(g: ProjectGraph, repo_root: Path) -> Path:
    out = _glx_dir(repo_root) / "project_graph.json"
    payload = _to_json(g)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="project_graph", description="Budowa globalnego grafu projektu")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Zbuduj graf projektu")
    b.add_argument("--repo-root", "--repo", dest="repo_root", default=".", help="Katalog będący rootem repo (ten z .git/)")
    b.add_argument("--write", action="store_true", help="Zapisz .glx/graphs/project_graph.json")
    b.add_argument("--stdout", action="store_true", help="Wypisz JSON na stdout")

    args = p.parse_args(argv)

    if args.cmd == "build":
        rr = Path(args.repo_root).resolve()
        g = build_project_graph(rr)
        if args.write:
            out = save_project_graph(g, rr)
            print(str(out))
        if args.stdout or not args.write:
            print(json.dumps(_to_json(g), ensure_ascii=False, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(_cli())
