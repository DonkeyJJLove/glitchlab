#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.githooks/pre-push.py — lokalna bramka Δ + inwarianty I1–I4 (Python 3.9)

Zadania:
- Wyznacza zakres commitów dla push (na podstawie STDIN z Gita lub parametru --range).
- Uruchamia:
    1) glx.tools.delta_fingerprint  → generuje .glx/delta_report.json (Δ-tokens, fingerprint)
    2) glx.tools.invariants_check   → ocenia score vs progi α/β/Z, zapisuje .glx/commit_analysis.json
- Zwraca kod ≠0 i blokuje push, gdy score > β (lub > Z dla hard-block).
- Opcjonalnie przygotowuje snippet do commita (.glx/commit_snippet.txt) dla dalszych hooków.

Użycie ręczne (debug):
    .githooks/pre-push.py --range HEAD~1..HEAD
"""
from __future__ import annotations

import sys
from pathlib import Path

# Załaduj helpery
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
import _common as H  # noqa: E402


def _read_git_stdin_last_range() -> str:
    """
    Czyta linie z STDIN (format Git pre-push: local_ref local_sha remote_ref remote_sha)
    i zwraca zakres commitów w postaci:
      - 'REMOTE_SHA..LOCAL_SHA' dla aktualizacji istniejącej gałęzi
      - 'LOCAL_SHA' gdy push nowej gałęzi (remote_sha = zera)
    Jeżeli STDIN jest pusty (np. uruchomienie ręczne), zwraca domyślnie 'HEAD~1..HEAD'.
    """
    if sys.stdin.isatty():
        return "HEAD~1..HEAD"

    last_local = None
    last_remote = None
    any_line = False

    for raw in sys.stdin.read().splitlines():
        any_line = True
        parts = raw.strip().split()
        if len(parts) < 4:
            # Nietypowa linia — ignorujemy
            continue
        local_ref, local_sha, remote_ref, remote_sha = parts[:4]
        last_local = local_sha
        last_remote = remote_sha

    if not any_line or not last_local:
        return "HEAD~1..HEAD"

    zeros = "0000000000000000000000000000000000000000"
    if last_remote == zeros:
        # nowa gałąź — pojedynczy commit/zakres od zera
        return str(last_local)
    # standardowy przypadek
    return f"{last_remote}..{last_local}"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    repo = H.git_root(THIS_DIR)
    H.ensure_glx_dir(repo)

    # Parametr ręczny --range (przydaje się w CI / debug)
    rng = None
    if "--range" in argv:
        idx = argv.index("--range")
        try:
            rng = argv[idx + 1]
        except IndexError:
            H.fail("Brak wartości po --range (oczekiwano np. 'A..B' lub 'HEAD~1..HEAD')")
    if rng is None:
        rng = _read_git_stdin_last_range()

    H.log(f"pre-push: używany zakres DIFF = {rng}")

    # 1) Δ-tokens + fingerprint
    rc = H.run_glx_module(
        "delta_fingerprint",
        repo,
        "glx.tools.delta_fingerprint",
        "glitchlab.glx.tools.delta_fingerprint",  # ewentualny alias
        args=["--range", rng],
    )
    if rc != 0:
        H.fail("delta_fingerprint: błąd analizy Δ-tokens/fingerprint")

    # 2) Inwarianty I1–I4 (gating wg α/β/Z)
    rc = H.run_glx_module(
        "invariants_check",
        repo,
        "glx.tools.invariants_check",
        "glitchlab.glx.tools.invariants_check",
        args=["--range", rng],
    )
    if rc != 0:
        # invariants_check sam zadecydował o poziomie blokady (β / Z)
        H.fail("invariants_check: naruszenia inwariantów — push zablokowany", code=rc)

    # (opcjonalnie) przygotuj snippet dla kolejnych commitów
    try:
        out = H.write_commit_snippet(repo)
        if out:
            H.log(f"Zapisano snippet: {out}")
    except Exception:
        # nie przerywamy push — to tylko wygoda
        H.warn("Nie udało się przygotować commit_snippet.txt (pomijam)")

    H.log("pre-push: OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        H.fail("Przerwano przez użytkownika")
