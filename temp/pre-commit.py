#!/usr/bin/env bash
# -*- coding: utf-8 -*-

"""
.githooks/pre-commit.py — lokalna bramka jakości GLX (Python 3.9)

Zakres:
- Higiena kodu: ruff, black --check
- Typy: mypy (miękko; twarde egzekwowanie w CI)
- Dokumentacja: glx.tools.doclint (twardo)
- Walidacje lekkie: zbyt duże pliki, kompilowalność .py

Wymagania:
- Python 3.9+
- Zainstalowane dev-extras: `pip install -e .[dev]`
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Upewniamy się, że możemy zaimportować helpery z .githooks/_common.py
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import _common as H  # noqa: E402


def _require_bin(name: str) -> None:
    if shutil.which(name) is None:
        H.fail(
            f"Brak narzędzia '{name}'. Zainstaluj dev-extras: `pip install -e .[dev]`"
        )


def main() -> int:
    repo = H.git_root(THIS_DIR)
    glx = H.ensure_glx_dir(repo)

    H.log("Pre-commit: walidacje lokalne (ruff/black/mypy/doclint)…")

    # 1) Walidacje lekkie na staged plikach
    staged = H.staged_paths(repo)
    H.reject_big_files(staged, threshold_mb=10)
    H.check_python_compiles(staged)

    # 2) Higiena kodu
    _require_bin("ruff")
    rc = H.run_cmd(repo, ["ruff", ".", "--force-exclude"])
    if rc != 0:
        H.fail("Ruff: błędy lintowania")

    rc = H.run_cmd(repo, [sys.executable, "-m", "black", "--check", "."])
    if rc != 0:
        H.fail("Black: wymagane formatowanie (uruchom `black .`)")

    # 3) Typy (miękko lokalnie, twardo w CI) — można wymusić twardo przez ENV
    _require_bin("mypy")
    rc = H.run_cmd(repo, ["mypy", "glitchlab"])
    if rc != 0:
        if os.environ.get("GLX_PRECOMMIT_STRICT_MYPY", "0") in ("1", "true", "yes"):
            H.fail("Mypy: błędy typów (tryb STRICT)")
        else:
            H.warn("Mypy: błędy typów (zignorowano lokalnie; CI egzekwuje twardo)")

    # 4) Dokumentacja (twardo)
    # uruchamiamy moduł: glx.tools.doclint (albo fallback po ścieżce)
    rc = H.run_glx_module(
        "doclint",
        repo,
        "glx.tools.doclint",
        "glitchlab.glx.tools.doclint",  # ewentualny alias, jeśli katalog glx/ leży w glitchlab/
    )
    if rc != 0:
        H.fail("doclint: niespójna dokumentacja (patrz log powyżej)")

    H.log("Pre-commit: OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        H.fail("Przerwano przez użytkownika")
