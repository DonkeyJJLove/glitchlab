#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.githooks/commit-msg.py — egzekwowanie Conventional Commits + wstawka GLX snippet (Python 3.9)

Funkcje:
- Walidacja nagłówka wg Conventional Commits (typ, opcjonalny scope, dwukropek i opis).
- Podstawowe zasady formatowania (długość nagłówka ≤ 72, pusta linia przed body).
- Wyjątki: merge/revert, fixup!/squash! — przepuszczane.
- Opcjonalne dołączenie wstawki GLX z `.glx/commit_snippet.txt` (Δ-tokens, fingerprint).

Wymagania:
- Python 3.9+
- Współdzielone helpery: .githooks/_common.py
"""
from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path

# Upewniamy się, że możemy zaimportować helpery z .githooks/_common.py
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import _common as H  # noqa: E402

# Dozwolone typy Conventional Commits (rozszerzalne)
TYPES = (
    "feat",
    "fix",
    "perf",
    "refactor",
    "docs",
    "ci",
    "build",
    "test",
    "chore",
)

# Wzorzec CC: type(scope)?: description
CC_RE = re.compile(
    r"^(?P<type>{types})(\([a-z0-9_\-\/\.]+\))?:\s+\S.*$".format(types="|".join(TYPES))
)

MAX_HEADER = 72  # znaki


def _is_exempt(msg_header: str) -> bool:
    """Wiadomości zwolnione z CC (merge, revert, fixup/squash)."""
    h = msg_header.strip()
    if not h:
        return False
    lowered = h.lower()
    return (
        lowered.startswith("merge ")
        or lowered.startswith("revert ")
        or lowered.startswith("fixup!")
        or lowered.startswith("squash!")
    )


def _append_snippet_if_available(repo: Path, msg_path: Path, msg_text: str) -> str:
    """
    Jeśli istnieje .glx/commit_snippet.txt i nie ma go jeszcze w wiadomości,
    dodajemy go na końcu (po jednej pustej linii).
    """
    snippet_path = H.glx_dir(repo) / "commit_snippet.txt"
    if not snippet_path.exists():
        return msg_text

    try:
        snippet = snippet_path.read_text(encoding="utf-8").strip()
    except Exception:
        return msg_text

    if not snippet:
        return msg_text

    if snippet in msg_text:
        return msg_text  # już wstawione

    # Dodajemy separator (pusta linia) + snippet
    if not msg_text.endswith("\n"):
        msg_text += "\n"
    if not msg_text.endswith("\n\n"):
        msg_text += "\n"
    msg_text += snippet + "\n"
    return msg_text


def main() -> int:
    if len(sys.argv) < 2:
        H.fail("Brak ścieżki do pliku wiadomości (COMMIT_EDITMSG)")

    msg_file = Path(sys.argv[1]).resolve()
    repo = H.git_root(THIS_DIR)

    if not msg_file.exists():
        H.fail(f"Nie znaleziono pliku wiadomości: {msg_file}")

    raw = msg_file.read_text(encoding="utf-8", errors="replace")

    # Pierwsza niepusta linia to nagłówek
    header = ""
    for line in raw.splitlines():
        if line.strip() and not line.strip().startswith("#"):
            header = line.strip()
            break

    if not header:
        H.fail(
            "Pusta wiadomość commita. Użyj formatu Conventional Commits, np.:\n"
            "  feat(app): panel Delta Inspector"
        )

    # Wyjątki (merge/revert/fixup/squash)
    if _is_exempt(header):
        H.log("commit-msg: wyjątek (merge/revert/fixup/squash) — przepuszczam.")
        return 0

    # Walidacja wzorca CC
    if not CC_RE.match(header):
        allowed = "|".join(TYPES)
        H.fail(
            "commit-msg: niepoprawny nagłówek.\n"
            f"Dozwolone typy: {allowed}\n"
            "Wzorzec: type(scope)?: opis\n"
            "Przykłady:\n"
            "  feat(app): panel Delta Inspector\n"
            "  fix(core): obsługa brzegowego przypadku w Δ-SSIM\n"
        )

    # Długość nagłówka
    if len(header) > MAX_HEADER:
        H.fail(
            f"commit-msg: nagłówek zbyt długi ({len(header)} > {MAX_HEADER}). "
            "Skróć opis do sedna, szczegóły przenieś do body."
        )

    # Pusta linia przed body (jeśli body istnieje)
    lines = raw.splitlines()
    if len(lines) >= 2:
        # znajdź pozycję nagłówka
        try:
            idx = lines.index(header)
        except ValueError:
            idx = 0
        # jeśli istnieją kolejne linie i druga nie jest pusta → wstaw pustą
        if idx + 1 < len(lines) and lines[idx + 1].strip() and not lines[idx + 1].startswith("#"):
            # wstawimy pustą i zapiszemy
            lines.insert(idx + 1, "")

    new_text = "\n".join(lines) + ("\n" if not raw.endswith("\n") else "")

    # Opcjonalna wstawka GLX (Δ-tokens/fingerprint) z artefaktu
    # (zwykle generowana przez pre-push lub post-commit)
    new_text2 = _append_snippet_if_available(repo, msg_file, new_text)

    if new_text2 != raw:
        # Nadpisujemy plik wiadomości
        msg_file.write_text(new_text2, encoding="utf-8")

    H.log("commit-msg: OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        H.fail("Przerwano przez użytkownika")
