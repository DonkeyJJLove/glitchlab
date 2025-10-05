#!/usr/bin/env python
# glitchlab/.githooks/pre-commit.py
# -*- coding: utf-8 -*-
# Python 3.9+
"""
PRE-COMMIT:
- Ładuje i normalizuje .env (łańcuch przodków; .env.local nadpisuje .env),
- Weryfikuje kluczowe GLX_* i sanity parametrów mozaiki,
- Szybkie kontrole staged: rozmiary plików, kompilacja .py,
- **Wywołuje** pre-diff.py (generuje .glx/commit_snippet.txt przed prepare-commit-msg),
  ale tylko jeśli są jakiekolwiek pliki w indeksie.

Zgodność:
- Ścieżki względne w .env liczone względem GLX_ROOT (po normalizacji).
- Jeśli GLX_PHI=policy → wymagany, istniejący plik GLX_POLICY (po normalizacji).
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List
from _common import (
    log,
    fail,
    git,
    repo_top_from_here,
    load_env,
    norm_paths_in_env,
    req,
    as_int,
    as_float01,
    staged_files,
)
# Wymuś UTF-8 (zwł. Windows)
os.environ.setdefault("PYTHONUTF8", "1")

# ── Wspólne utilsy ───────────────────────────────────────────────────────────
HERE = Path(__file__).resolve()
HOOKS = HERE.parent
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))


# ── Lokalne, lekkie sanity ───────────────────────────────────────────────────
def _py_compile(paths: List[Path]) -> None:
    """Szybka kompilacja plików .py z indeksu. Blokuje commit przy błędach."""
    if not paths:
        return
    errors: List[str] = []
    for p in paths:
        if p.suffix.lower() != ".py":
            continue
        r = subprocess.run([sys.executable, "-m", "py_compile", str(p)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            errors.append(f"- {p}: {r.stderr.strip() or 'compile error'}")
    if errors:
        fail("Błędy kompilacji Pythona:\n" + "\n".join(errors))


def _reject_big_files(paths: List[Path], threshold_mb: int = 10) -> None:
    """Odrzuca zbyt duże pliki w staged (domyślnie >10MB)."""
    too_big = [
        p for p in paths
        if p.exists() and p.is_file() and p.stat().st_size > threshold_mb * 1024 * 1024
    ]
    if too_big:
        listing = "\n".join(
            f"- {p} ({p.stat().st_size / 1024 / 1024:.1f} MB)" for p in too_big
        )
        fail(f"Zbyt duże pliki w commitcie (> {threshold_mb} MB):\n{listing}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    # 0) Repo top + .env
    repo_top = repo_top_from_here(HERE)
    if not (repo_top / ".env").exists() and not (repo_top / ".env.local").exists():
        fail(f"Brak .env/.env.local w {repo_top}")

    env: Dict[str, str] = load_env(repo_top)
    env = norm_paths_in_env(env, repo_top)

    # 1) Kluczowe GLX_* oraz sanity mozaiki
    glx_root = Path(req(env, "GLX_ROOT"))
    if not glx_root.exists():
        fail(f"GLX_ROOT wskazuje na nieistniejącą ścieżkę: {glx_root}")

    req(env, "GLX_PKG")
    req(env, "GLX_OUT")
    req(env, "GLX_AUTONOMY_OUT")

    req(env, "GLX_PHI")
    as_int(env, "GLX_ROWS", 1, 2048)
    as_int(env, "GLX_COLS", 1, 2048)
    as_float01(env, "GLX_EDGE_THR")
    as_float01(env, "GLX_DELTA")
    as_float01(env, "GLX_KAPPA")

    if env.get("GLX_PHI", "").strip().lower() == "policy":
        pol = Path(env.get("GLX_POLICY", ""))
        if not pol.exists():
            fail(f"GLX_PHI=policy, ale brak pliku polityki: {pol}")

    # 2) Staged sanity
    files = staged_files(repo_top)
    _reject_big_files(files, threshold_mb=10)
    _py_compile(files)

    # 3) Pre-diff tylko gdy coś jest w indeksie (żeby nie brudzić .glx/)
    if files:
        pre_diff_new = HOOKS / "pre-diff.py"
        pre_diff_old = repo_top / "scripts" / "pre_diff.py"  # kompatybilność wstecz
        pre_diff = pre_diff_new if pre_diff_new.exists() else pre_diff_old

        if pre_diff.exists():
            log(f"[pre-commit] run pre-diff: {pre_diff.relative_to(repo_top)}")
            r = subprocess.run([sys.executable, str(pre_diff)], cwd=str(repo_top))
            if r.returncode != 0:
                fail(f"pre-diff zakończył się kodem {r.returncode}")
        else:
            log("[pre-commit] pre-diff not found; skipping snippet generation")
    else:
        log("[pre-commit] no staged files; skipping pre-diff")

    log("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
