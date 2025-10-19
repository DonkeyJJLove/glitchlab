#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glitchlab — agregator CLI
Użycie (przykłady):
  # 1) budowa grafu projektu i zapis artefaktu
  python -m glitchlab project-graph build --repo-root . --write

  # 2) metasoczewka (deleguje do analysis.scope_viz)
  python -m glitchlab scope meta --repo-root . --level module --center glitchlab.core --depth 2 --write

  # 3) 3D (jeśli masz już project_graph.json)
  python -m glitchlab tools graph3d --input ./.glx/graphs/project_graph.json --output ./.glx/graphs/project_graph_3d.html

  # 4) przeliczenie globalnych metryk i odświeżenie cache
  python -m glitchlab recompute fields --repo-root .
  python -m glitchlab recompute metrics --repo-root .
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Pomocnicze: import z fallbackami
# ──────────────────────────────────────────────────────────────────────────────

def _try_import(*mod_names: str):
    last_err = None
    for name in mod_names:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
    raise last_err or ImportError(f"cannot import any of: {mod_names!r}")

def _ensure_repo_root(repo_root: Optional[str]) -> Path:
    p = Path(repo_root or ".").resolve()
    return p

def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def _print_err(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Komendy
# ──────────────────────────────────────────────────────────────────────────────

def cmd_project_graph(args: argparse.Namespace) -> int:
    # import modułu budującego graf (oba warianty)
    mod = _try_import("glitchlab.analysis.project_graph", "analysis.project_graph")
    rr = _ensure_repo_root(args.repo_root)

    if args.subcmd == "build":
        # używamy API jeśli jest, inaczej wołamy ich mini-CLI
        build = getattr(mod, "build_project_graph", None)
        save = getattr(mod, "save_project_graph", None)
        to_json = getattr(mod, "_to_json", None)
        if build and save and to_json:
            g = build(rr)
            if args.write:
                out = save(g, rr if rr.name != "glitchlab" else rr)
                print(str(out))
            if args.stdout or not args.write:
                print(json.dumps(to_json(g), ensure_ascii=False, indent=2))
            return 0
        # fallback CLI wewnątrz modułu
        cli = getattr(mod, "_cli", None)
        if cli:
            argv = ["build", "--repo-root", str(rr)]
            if args.write: argv.append("--write")
            if args.stdout: argv.append("--stdout")
            return int(cli(argv))
        _print_err("project_graph: brak API i CLI w module")
        return 2
    return 2


def cmd_scope(args: argparse.Namespace) -> int:
    # delegujemy do analysis.scope_viz (odchudzony CLI)
    mod = _try_import("glitchlab.analysis.scope_viz", "analysis.scope_viz")
    cli = getattr(mod, "_cli", None)
    if not cli:
        _print_err("scope_viz: nie znaleziono funkcji _cli")
        return 2
    # zbuduj argv dla podkomendy
    argv = [args.subcmd]
    if args.subcmd == "lens":
        argv += [args.kind]
        if args.target:
            argv += [args.target]
        if args.repo_root:
            argv += ["--repo-root", str(args.repo_root)]
        if args.write:
            argv += ["--write"]
        if args.stdout:
            argv += ["--stdout"]
    elif args.subcmd == "meta":
        if args.repo_root:
            argv += ["--repo-root", str(args.repo_root)]
        if args.spec:
            argv += ["--spec", str(args.spec)]
        if args.level:
            argv += ["--level", args.level]
        for c in (args.center or []):
            argv += ["--center", c]
        if args.depth is not None:
            argv += ["--depth", str(args.depth)]
        if args.max_nodes is not None:
            argv += ["--max-nodes", str(args.max_nodes)]
        if args.edges:
            argv += ["--edges", args.edges]
        if args.field:
            argv += ["--field", args.field]
        if args.write:
            argv += ["--write"]
        if args.stdout:
            argv += ["--stdout"]
    else:
        _print_err(f"scope: nieznana subkomenda {args.subcmd}")
        return 2
    return int(cli(argv))


def cmd_tools(args: argparse.Namespace) -> int:
    if args.tool == "graph3d":
        # glx.tools.project_graph_3d
        mod = _try_import("glitchlab.glx.tools.project_graph_3d", "glx.tools.project_graph_3d")
        main = getattr(mod, "main", None)
        if not main:
            _print_err("graph3d: nie znaleziono funkcji main")
            return 2
        argv = []
        if args.input:
            argv += ["--input", str(Path(args.input).resolve())]
        if args.repo_root:
            argv += ["--repo-root", str(Path(args.repo_root).resolve())]
        if args.output:
            argv += ["--output", str(Path(args.output).resolve())]
        if args.limit_nodes is not None:
            argv += ["--limit-nodes", str(int(args.limit_nodes))]
        return int(main(argv))
    _print_err(f"tools: nieznane narzędzie {args.tool}")
    return 2


def cmd_recompute(args: argparse.Namespace) -> int:
    rr = _ensure_repo_root(args.repo_root)
    if args.what == "fields":
        mod = _try_import("glitchlab.glx.tools.recompute_scope_fields", "glx.tools.recompute_scope_fields")
        main = getattr(mod, "main", None)
        if not main:
            _print_err("recompute fields: nie znaleziono funkcji main w glx.tools.recompute_scope_fields")
            return 2
        return int(main(["--repo-root", str(rr)]))
    if args.what == "metrics":
        # bezpośrednio wywołujemy analysis.graph_metrics CLI
        mod = _try_import("glitchlab.analysis.graph_metrics", "analysis.graph_metrics")
        cli = getattr(mod, "main", None) or getattr(mod, "_cli", None)
        if not cli:
            _print_err("recompute metrics: nie znaleziono CLI w analysis.graph_metrics")
            return 2
        return int(cli(["--repo-root", str(rr), "--write"]))
    _print_err(f"recompute: nieznany cel {args.what}")
    return 2


# ──────────────────────────────────────────────────────────────────────────────
# Parser główny
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="glitchlab", description="GlitchLab CLI (agregator)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # project-graph
    pg = sub.add_parser("project-graph", help="Operacje na globalnym grafie projektu")
    pg_sub = pg.add_subparsers(dest="subcmd", required=True)
    pg_build = pg_sub.add_parser("build", help="Zbuduj graf projektu")
    pg_build.add_argument("--repo-root", default=".", help="Root repo (katalog 'glitchlab/' lub jego rodzic)")
    pg_build.add_argument("--write", action="store_true", help="Zapisz artefakt .glx/graphs/project_graph.json")
    pg_build.add_argument("--stdout", action="store_true", help="Wypisz JSON na stdout")
    pg_build.set_defaults(run=cmd_project_graph)

    # scope (lens/meta)
    sc = sub.add_parser("scope", help="Soczewki (legacy lens) i MetaLens")
    sc_sub = sc.add_subparsers(dest="subcmd", required=True)

    sc_lens = sc_sub.add_parser("lens", help="Legacy: module|file|func|bus")
    sc_lens.add_argument("kind", choices=["module", "file", "func", "bus"])
    sc_lens.add_argument("target", nargs="?", help="np. glitchlab/core lub ścieżka pliku lub file.py:func")
    sc_lens.add_argument("--repo-root", default=".", help="Root repo")
    sc_lens.add_argument("--write", action="store_true")
    sc_lens.add_argument("--stdout", action="store_true")
    sc_lens.set_defaults(run=cmd_scope)

    sc_meta = sc_sub.add_parser("meta", help="Metasoczewka (MetaLens)")
    sc_meta.add_argument("--repo-root", default=".", help="Root repo")
    sc_meta.add_argument("--spec", help="Ścieżka do spec JSON")
    sc_meta.add_argument("--level", choices=["project", "module", "file", "func", "bus", "custom"])
    sc_meta.add_argument("--center", action="append", help="Wzorzec (glob/regex). Można powtarzać.")
    sc_meta.add_argument("--depth", type=int)
    sc_meta.add_argument("--max-nodes", type=int)
    sc_meta.add_argument("--edges", help="Lista krawędzi rozdzielona przecinkiem")
    sc_meta.add_argument("--field", help='JSON np. {"name":"degree","top_k":120,"threshold":0.4}')
    sc_meta.add_argument("--write", action="store_true")
    sc_meta.add_argument("--stdout", action="store_true")
    sc_meta.set_defaults(run=cmd_scope)

    # tools
    tl = sub.add_parser("tools", help="Narzędzia dodatkowe")
    tl_sub = tl.add_subparsers(dest="tool", required=True)

    t_g3d = tl_sub.add_parser("graph3d", help="Wizualizacja grafu w 3D (Three.js/Plotly)")
    t_g3d.add_argument("--input", help="Ścieżka do project_graph.json (jeśli brak, spróbujemy zbudować)")
    t_g3d.add_argument("--repo-root", default=".", help="Root repo (do zbudowania grafu gdy brak --input)")
    t_g3d.add_argument("--output", default="./.glx/graphs/project_graph_3d.html", help="Ścieżka wyjściowego HTML")
    t_g3d.add_argument("--limit-nodes", type=int, default=None, help="Limit liczby węzłów do wizualizacji")
    t_g3d.set_defaults(run=cmd_tools)

    # recompute
    rc = sub.add_parser("recompute", help="Batch: przelicz artefakty scope/metryk")
    rc_sub = rc.add_subparsers(dest="what", required=True)
    r_fields = rc_sub.add_parser("fields", help="Przelicz field-cache (globalne metryki + cache)")
    r_fields.add_argument("--repo-root", default=".", help="Root repo")
    r_fields.set_defaults(run=cmd_recompute)
    r_metrics = rc_sub.add_parser("metrics", help="Przelicz globalne metryki grafu")
    r_metrics.add_argument("--repo-root", default=".", help="Root repo")
    r_metrics.set_defaults(run=cmd_recompute)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    # Jeśli odpalasz bezpośrednio z katalogu 'glitchlab/', importy 'analysis.*' i 'glx.*' zadziałają,
    # a importy 'glitchlab.*' też, bo jesteśmy wewnątrz pakietu. Nie modyfikujemy sys.path.
    parser = build_parser()
    args = parser.parse_args(argv)
    run = getattr(args, "run", None)
    if not run:
        parser.print_help()
        return 2
    return int(run(args))


if __name__ == "__main__":
    sys.exit(main())
