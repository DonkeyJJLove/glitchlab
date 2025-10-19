#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glx.tools.scope_meta_cli — wygodny wrapper CLI dla metasoczewek

Cel:
- Alias do: `python -m glitchlab.analysis.scope_viz meta`
- Presety (makra) specyfikacji metasoczewek: arch-overview, hotspots, bus,
  module-focus, file-neighborhood, func-neighborhood.
- Opcje nadpisujące preset (depth, max-nodes, field top_k/threshold, edges).

Zależności:
- glitchlab.analysis.scope_viz (odchudzone CLI delegujące do project_graph/scope_meta)
- stdlib only

Przykłady:
  # alias (przekierowanie surowych argumentów do scope_viz meta)
  glx-scope meta --level module --center glitchlab.core --depth 2

  # preset: przegląd architektury (import/define + lekki call)
  glx-scope preset arch-overview

  # preset: hotspots (PageRank; top-200; próg 0.4)
  glx-scope preset hotspots --top-k 200 --threshold 0.4

  # preset: bus (tematy + producenci/konsumenci)
  glx-scope preset bus

  # preset: fokus na moduł
  glx-scope preset module-focus --target glitchlab/core --depth 2

  # tylko zapisz spec do pliku (bez uruchamiania)
  glx-scope preset module-focus --target glitchlab/core --write-spec .glx/graphs/spec_module_focus.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Delegacja do analysis.scope_viz
# ──────────────────────────────────────────────────────────────────────────────

def _import_scope_viz():
    """
    Importuje moduł scope_viz i zwraca funkcję run_cli(argv) lub None.
    W razie braku funkcji run_cli, zwraca sam moduł (fallback na subprocess -m).
    """
    try:
        import glitchlab.analysis.scope_viz as scope_viz  # type: ignore
        run_cli = getattr(scope_viz, "run_cli", None)
        return run_cli or scope_viz
    except Exception:
        return None


def _dispatch_scope_viz_meta(argv_like: List[str]) -> int:
    """
    Przekazuje wywołanie do scope_viz meta. Preferuje wywołanie funkcji run_cli.
    W ostateczności uruchamia `python -m glitchlab.analysis.scope_viz` przez subprocess.
    """
    sv = _import_scope_viz()
    if sv is None:
        # Fallback: subprocess
        import subprocess
        cmd = [sys.executable, "-m", "glitchlab.analysis.scope_viz"] + argv_like
        return subprocess.call(cmd)

    # Jeśli mamy funkcję run_cli, wywołaj bezpośrednio
    if callable(sv):
        try:
            return int(sv(argv_like))  # type: ignore[call-arg]
        except SystemExit as e:
            return int(getattr(e, "code", 0) or 0)

    # W innym wypadku spróbuj wywołać modułowy _cli lub main
    try:
        fn = getattr(sv, "run_cli", None) or getattr(sv, "_cli", None) or getattr(sv, "main", None)
        if fn:
            return int(fn(argv_like))  # type: ignore[misc]
    except SystemExit as e:
        return int(getattr(e, "code", 0) or 0)
    except Exception:
        pass

    # Ostateczny fallback: subprocess
    import subprocess
    cmd = [sys.executable, "-m", "glitchlab.analysis.scope_viz"] + argv_like
    return subprocess.call(cmd)


# ──────────────────────────────────────────────────────────────────────────────
# Presety (makra) specyfikacji
# ──────────────────────────────────────────────────────────────────────────────

def _preset_arch_overview(args) -> Dict[str, Any]:
    # Widok architektury: moduły + słabe call, import/define
    return {
        "level": "project",
        "center": args.center or ([] if not args.target else [str(args.target).replace("/", ".")]),
        "depth": args.depth if args.depth is not None else 2,
        "allowed_edges": args.edges or ["import", "define", "call"],
        "allowed_kinds": ["project", "module", "topic"],
        "max_nodes": args.max_nodes if args.max_nodes is not None else 900,
        "field": {
            "name": args.field_name or "degree",
            "top_k": args.top_k if args.top_k is not None else 0,
            "threshold": args.threshold if args.threshold is not None else 0.0,
        },
    }

def _preset_hotspots(args) -> Dict[str, Any]:
    # Hotspots: PageRank / betweenness_approx; edge rodzaje szerzej
    field_name = args.field_name or "pagerank"
    return {
        "level": "module",
        "center": args.center or ([] if not args.target else [str(args.target)]),
        "depth": args.depth if args.depth is not None else 2,
        "allowed_edges": args.edges or ["import", "define", "call", "link", "use"],
        "allowed_kinds": ["module", "file", "topic"],
        "max_nodes": args.max_nodes if args.max_nodes is not None else 800,
        "field": {
            "name": field_name,
            "top_k": args.top_k if args.top_k is not None else 200,
            "threshold": args.threshold if args.threshold is not None else 0.35,
        },
    }

def _preset_bus(args) -> Dict[str, Any]:
    # BUS: tematy + producenci/konsumenci
    return {
        "level": "bus",
        "center": args.center or ([] if not args.target else [str(args.target)]),
        "depth": args.depth if args.depth is not None else 2,
        "allowed_edges": args.edges or ["define", "use", "link", "rpc"],
        "allowed_kinds": ["topic", "module", "file"],
        "max_nodes": args.max_nodes if args.max_nodes is not None else 1200,
        "field": {
            "name": args.field_name or "degree",
            "top_k": args.top_k if args.top_k is not None else 0,
            "threshold": args.threshold if args.threshold is not None else 0.0,
        },
    }

def _preset_module_focus(args) -> Dict[str, Any]:
    # Fokus na moduł: obowiązkowy target (prefiks lub exact), call+import+define
    target = (args.target or "").replace("/", ".")
    return {
        "level": "module",
        "center": args.center or ([target] if target else []),
        "depth": args.depth if args.depth is not None else 2,
        "allowed_edges": args.edges or ["import", "define", "call"],
        "allowed_kinds": ["module", "file", "func", "topic"],
        "max_nodes": args.max_nodes if args.max_nodes is not None else 600,
        "field": {
            "name": args.field_name or "degree",
            "top_k": args.top_k if args.top_k is not None else 0,
            "threshold": args.threshold if args.threshold is not None else 0.0,
        },
    }

def _preset_file_neighborhood(args) -> Dict[str, Any]:
    # Sąsiedztwo pliku
    return {
        "level": "file",
        "center": args.center or ([str(args.target)] if args.target else []),
        "depth": args.depth if args.depth is not None else 2,
        "allowed_edges": args.edges or ["import", "define", "call"],
        "allowed_kinds": ["file", "module", "func", "topic"],
        "max_nodes": args.max_nodes if args.max_nodes is not None else 500,
        "field": {
            "name": args.field_name or "degree",
            "top_k": args.top_k if args.top_k is not None else 0,
            "threshold": args.threshold if args.threshold is not None else 0.0,
        },
    }

def _preset_func_neighborhood(args) -> Dict[str, Any]:
    # Sąsiedztwo funkcji: target może być wzorcem nazwy lub file.py:func
    return {
        "level": "func",
        "center": args.center or ([str(args.target)] if args.target else []),
        "depth": args.depth if args.depth is not None else 2,
        "allowed_edges": args.edges or ["call", "define", "import"],
        "allowed_kinds": ["func", "file", "module", "topic"],
        "max_nodes": args.max_nodes if args.max_nodes is not None else 400,
        "field": {
            "name": args.field_name or "degree",
            "top_k": args.top_k if args.top_k is not None else 0,
            "threshold": args.threshold if args.threshold is not None else 0.0,
        },
    }

_PRESETS = {
    "arch-overview": _preset_arch_overview,
    "hotspots": _preset_hotspots,
    "bus": _preset_bus,
    "module-focus": _preset_module_focus,
    "file-neighborhood": _preset_file_neighborhood,
    "func-neighborhood": _preset_func_neighborhood,
}

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="glx-scope", description="Presety metasoczewek i alias do scope_viz meta")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Alias: przekieruj args do scope_viz meta bez zmian
    meta = sub.add_parser("meta", help="Alias do scope_viz meta (przekazuje argumenty 1:1)")
    meta.add_argument("rest", nargs=argparse.REMAINDER, help="Argumenty po 'meta' zostaną przekazane do scope_viz")

    # Presety
    preset = sub.add_parser("preset", help="Uruchom metasoczewkę z predefiniowanym presetem")
    preset.add_argument("name", choices=sorted(_PRESETS.keys()))
    preset.add_argument("--target", help="Cel/centrum (moduł/file/func/topic pattern)")
    preset.add_argument("--center", action="append", help="Dodatkowe wzorce centrum (glob/regex); można powtarzać")
    preset.add_argument("--depth", type=int, help="Głębokość BFS (hopy)")
    preset.add_argument("--max-nodes", type=int, help="Limit węzłów w oknie")
    preset.add_argument("--edges", help="Lista krawędzi rozdzielona przecinkiem (np. import,define,call)")
    preset.add_argument("--field-name", help="Nazwa pola operacyjnego (np. degree, pagerank, betweenness_approx)")
    preset.add_argument("--top-k", type=int, help="Top-K wg pola")
    preset.add_argument("--threshold", type=float, help="Próg (0..1) relatywny do maksimum pola")
    preset.add_argument("--write-spec", help="Zapisz wygenerowaną spec do wskazanego pliku JSON (opcjonalnie)")
    preset.add_argument("--dry-run", action="store_true", help="Tylko wygeneruj i ewentualnie zapisz spec; nie uruchamiaj metasoczewki")

    args = p.parse_args(argv)

    if args.cmd == "meta":
        # przekaż 1:1 do scope_viz
        forward = ["meta"] + list(args.rest or [])
        return _dispatch_scope_viz_meta(forward)

    # Preset
    build_fn = _PRESETS[args.name]
    # edges jako lista
    if getattr(args, "edges", None):
        args.edges = [e.strip() for e in str(args.edges).split(",") if e.strip()]
    spec = build_fn(args)

    # opcjonalny zapis spec
    if args.write_spec:
        out_p = Path(args.write_spec).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(out_p))

    if args.dry_run:
        # tylko pokaż spec na stdout
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        return 0

    # uruchom scope_viz meta z przygotowaną spec
    # składamy argv pod aliasem:
    forward: List[str] = ["meta"]
    if "level" in spec and spec["level"]:
        forward += ["--level", str(spec["level"])]
    for c in spec.get("center", []) or []:
        forward += ["--center", str(c)]
    if "depth" in spec and spec["depth"] is not None:
        forward += ["--depth", str(int(spec["depth"]))]
    if "max_nodes" in spec and spec["max_nodes"] is not None:
        forward += ["--max-nodes", str(int(spec["max_nodes"]))]
    if "allowed_edges" in spec and spec["allowed_edges"]:
        forward += ["--edges", ",".join(spec["allowed_edges"])]
    # field jako JSON
    if "field" in spec and spec["field"]:
        forward += ["--field", json.dumps(spec["field"], ensure_ascii=False)]

    return _dispatch_scope_viz_meta(forward)


def main():
    return _cli()


if __name__ == "__main__":
    sys.exit(main())

