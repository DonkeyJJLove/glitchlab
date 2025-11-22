#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.githooks/pre-merge-commit.py — bramka jakości przed wykonaniem merge (Python 3.9)

Cel:
- Zatrzymać merge, jeśli wprowadzane zmiany łamią inwarianty I1–I4 (score > β / > Z).
- Wspierać oba tryby: twardy (domyślnie) oraz miękki (tylko ostrzeżenia).

Zachowanie:
- Wykrywa zakres do oceny jako `merge-base(HEAD, MERGE_HEAD)..MERGE_HEAD`.
- Uruchamia:
    * glx.tools.delta_fingerprint  → aktualizuje .glx/delta_report.json
    * glx.tools.invariants_check   → ocenia score (α/β/Z), zapisuje .glx/commit_analysis.json
- Domyślnie **blokuje** merge przy `score > β` (kod wyjścia 1) i przy `score > Z` (kod 2).
- Tryb miękki (nie blokuje, tylko ostrzega): ustaw `GLX_PREMERGE_SOFT=1`.

Uwagi:
- Hook nie modyfikuje drzewa ani indexu; działa tylko diagnostycznie.
- Zakłada obecność .githooks/_common.py oraz narzędzi w glx/tools/*.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Helpery wspólne
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
import _common as H  # noqa: E402


def _merge_head(repo: Path) -> str:
    sha = H.rev_parse(repo, "MERGE_HEAD")
    if not sha:
        H.fail("pre-merge-commit: brak MERGE_HEAD (nie wygląda na trwający merge).")
    return sha


def _merge_base(repo: Path, other: str) -> str:
    base = None
    try:
        # git merge-base HEAD <MERGE_HEAD>
        import subprocess

        r = subprocess.run(
            ["git", "merge-base", "HEAD", other],
            cwd=str(repo),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        base = (r.stdout or "").strip()
    except Exception:
        base = None
    if not base:
        # Fallback: HEAD^ (jeśli dostępne), inaczej pierwszy rodzic HEAD
        base = H.rev_parse(repo, "HEAD^") or H.rev_parse(repo, "HEAD")
    if not base:
        H.fail("pre-merge-commit: nie można wyznaczyć bazy porównania.")
    return base


def main() -> int:
    repo = H.git_root(THIS_DIR)
    H.ensure_glx_dir(repo)

    soft = os.environ.get("GLX_PREMERGE_SOFT", "0") in ("1", "true", "yes")

    other = _merge_head(repo)
    base = _merge_base(repo, other)
    rng = f"{base}..{other}"
    H.log(f"pre-merge-commit: zakres analizy = {rng} (soft={soft})")

    # 1) Δ-tokens + fingerprint
    rc_df = H.run_glx_module(
        "delta_fingerprint",
        repo,
        "glx.tools.delta_fingerprint",
        "glitchlab.glx.tools.delta_fingerprint",
        args=["--range", rng],
    )
    if rc_df != 0:
        # W trybie miękkim — tylko ostrzegamy
        if soft:
            H.warn("delta_fingerprint: błąd (tryb miękki — kontynuuję).")
        else:
            H.fail("delta_fingerprint: błąd analizy Δ-tokens/fingerprint")

    # 2) Inwarianty I1–I4 (gating wg α/β/Z)
    rc_inv = H.run_glx_module(
        "invariants_check",
        repo,
        "glx.tools.invariants_check",
        "glitchlab.glx.tools.invariants_check",
        args=["--range", rng],
    )

    if rc_inv != 0:
        # rc 1 → >β, rc 2 → >Z
        if soft:
            H.warn("invariants_check: naruszenia wykryte (tryb miękki — nie blokuję merge).")
            return 0
        # Tryb twardy — blokujemy
        if rc_inv == 2:
            H.fail("invariants_check: score > Z — merge wstrzymany (hard-block).", code=2)
        H.fail("invariants_check: score > β — merge wstrzymany.", code=1)

    H.log("pre-merge-commit: OK — inwarianty spełnione.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        H.fail("Przerwano przez użytkownika")
