#!/usr/bin/env python
# glitchlab/.githooks/pre-diff.py
# -*- coding: utf-8 -*-
# Python 3.9+
"""
PRE-DIFF (hook pomocniczy uruchamiany przez pre-commit):
- zbiera staged pliki (git diff --cached --name-status)
- analizuje .py (AST/Mozaika) + metawarstwa
- buduje .glx/commit_snippet.txt oraz .glx/commit_analysis.json
- nie modyfikuje treści commita (to robi prepare-commit-msg)

Zasady .env:
- Ścieżki względne licz wględem GLX_ROOT (prefer. absolutny).
- .env.local nadpisuje .env; szukamy od root dysku → repo_top.
"""
from __future__ import annotations

import ast
import datetime
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

# ── import wspólnych utili ────────────────────────────────────────────────────
HERE = Path(__file__).resolve()
HOOKS = HERE.parent
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

from _common import (  # type: ignore
    log,
    fail,
    git,
    repo_top_from_here,
    load_env,
    norm_paths_in_env,
    ensure_import_roots,
)


# ── próba użycia "prawdziwego" analizatora; fallback do lekkiej heurystyki ────
def _try_import_real_analyzer():
    mods = (
        "glitchlab.backup.scripts.ast_mosaic_analyzer",
        "backup.scripts.ast_mosaic_analyzer",
    )
    for m in mods:
        try:
            analyze_py = __import__(m, fromlist=["analyze_py"]).analyze_py  # type: ignore[attr-defined]
            aggregate = __import__(m, fromlist=["aggregate"]).aggregate  # type: ignore[attr-defined]
            return analyze_py, aggregate
        except Exception:
            continue
    return None, None


def _heur_analyze_py(path: Path) -> Dict[str, Any]:
    """
    Minimalna, szybka analiza AST gdy brak właściwego modułu:
    - S: liczba węzłów ast.stmt
    - H: liczba def/class
    - Z: liczba importów
    - align: heurystyka 0.35..0.85 w zależności od udziału def/class
    """
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
        nodes = list(ast.walk(tree))
        S = sum(isinstance(n, ast.stmt) for n in nodes)
        H = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in nodes)
        Z = sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in nodes)
        dens = (H / max(1, S)) if S else 0.0
        align = max(0.35, min(0.85, 0.35 + 0.9 * dens))
    except Exception:
        S = H = Z = 0
        align = 0.5
    return {"path": str(path), "S": S, "H": H, "Z": Z, "align": round(align, 4)}


def _heur_aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"files": 0, "avg_align": 0.0, "S": 0, "H": 0, "Z": 0}
    return {
        "files": len(results),
        "avg_align": round(mean(r.get("align", 0.0) for r in results), 4),
        "S": int(sum(r.get("S", 0) for r in results)),
        "H": int(sum(r.get("H", 0) for r in results)),
        "Z": int(sum(r.get("Z", 0) for r in results)),
    }


REAL_ANALYZE_PY, REAL_AGGREGATE = _try_import_real_analyzer()


# ── git helpers ───────────────────────────────────────────────────────────────
def _sh(*a: str, text: bool = True, cwd: Path | None = None) -> str:
    return subprocess.check_output(a, text=text, cwd=str(cwd) if cwd else None).strip()


def staged_name_status(repo: Path) -> List[Tuple[str, str]]:
    """
    Zwraca [(status, path), ...] z indeksu. Przy Rxxx/Cxxx bierzemy nową ścieżkę.
    """
    r = git(["diff", "--cached", "--name-status"], repo)
    if r.returncode != 0:
        fail("Nie mogę pobrać listy staged plików (git diff --cached --name-status).", 1)

    rows: List[Tuple[str, str]] = []
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        st = parts[0].strip()
        p = parts[-1].strip()
        rows.append((st, p))
    return rows


# ── core ─────────────────────────────────────────────────────────────────────
def _decide_doc(agg: Dict[str, Any]) -> Tuple[str, str]:
    """
    Heurystyka decyzji dokumentacyjnej spójna z wcześniejszym szkicem.
    """
    files = int(agg.get("files", 0))
    avg_align = float(agg.get("avg_align", 0.0))
    if avg_align >= 0.2 and files >= 2:
        return "REVIEW", "lokalne dopasowanie; sprawdź sekcje dotkniętych modułów"
    return "NO-OP", "brak wpływu na strukturę sąsiednich kafli"


def main() -> int:
    # 0) repo_top + .env
    repo_top = repo_top_from_here(HERE)
    env = load_env(repo_top)
    env = norm_paths_in_env(env, repo_top)

    # 1) Import roots projektu (żeby działały importy glitchlab.* itp.)
    ensure_import_roots(env)

    # 2) .glx dir (pod GLX_ROOT)
    glx_root = Path(env["GLX_ROOT"]).resolve()
    glx_dir = glx_root / ".glx"
    glx_dir.mkdir(parents=True, exist_ok=True)

    # 3) Zbierz staged pliki i wybierz .py
    ns = staged_name_status(repo_top)
    py_files = [Path(p) for (st, p) in ns if p.endswith(".py") and st[:1] in {"A", "M", "R", "C"}]

    # 4) Analiza .py
    results: List[Dict[str, Any]] = []
    if REAL_ANALYZE_PY and REAL_AGGREGATE:
        for p in py_files:
            try:
                results.append(REAL_ANALYZE_PY((repo_top / p).resolve()))  # type: ignore[operator]
            except Exception:
                # fallback per-file jeśli realny analizator padnie
                results.append(_heur_analyze_py((repo_top / p).resolve()))
        agg = REAL_AGGREGATE(results)  # type: ignore[call-arg]
    else:
        results = [_heur_analyze_py((repo_top / p).resolve()) for p in py_files]
        agg = _heur_aggregate(results)

    # 5) Snippet + raport
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    files_list = ", ".join([Path(p).name for (_, p) in ns][:6]) or "(brak)"
    if len(ns) > 6:
        files_list += f" (+{len(ns) - 6})"

    doc_decision, hint = _decide_doc(agg)

    snippet = (
        f"{len(ns)} file(s) staged: [Δ] Zakres\n"
        f"- files: {len(ns)} ({files_list})\n"
        f"- typ: auto (pre-diff)\n\n"
        f"[Φ/Ψ] Mozaika (semantyka kodu)\n"
        f"- Align(mean .py): {agg.get('avg_align', 0.0):.2f}\n"
        f"- Hint: {hint}\n\n"
        f"[AST] Deltas (staged .py)\n"
        f"- S: {agg.get('S', 0)}  H: {agg.get('H', 0)}\n"
        f"- uwagi: wartości przybliżone (heurystyki)\n\n"
        f"[Dokumentacja]\n"
        f"- decyzja: {doc_decision}\n\n"
        f"Meta\n"
        f"- Generated-by: pre-diff/AST-mosaic @ {now}\n"
    )

    # 6) Zapisy
    (glx_dir / "commit_snippet.txt").write_text(snippet, encoding="utf-8")
    report = dict(
        when=now,
        files=ns,
        py_results=results,
        aggregate=agg,
        doc_decision=doc_decision,
    )
    (glx_dir / "commit_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log(f"[pre-diff] analysis generated: {(glx_dir / 'commit_analysis.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
