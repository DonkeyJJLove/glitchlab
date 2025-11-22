#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.githooks/commit-msg.py â€” egzekwowanie Conventional Commits + wstawka GLX snippet (Python 3.9)

Funkcje:
- Walidacja nagÅ‚Ã³wka wg Conventional Commits (typ, opcjonalny scope, dwukropek i opis).
- Podstawowe zasady formatowania (dÅ‚ugoÅ›Ä‡ nagÅ‚Ã³wka â‰¤ 72, pusta linia przed body).
- WyjÄ…tki: merge/revert, fixup!/squash! â€” przepuszczane.
- Opcjonalne doÅ‚Ä…czenie wstawki GLX z `.glx/commit_snippet.txt` (Î”-tokens, fingerprint).

Wymagania:
- Python 3.9+
- WspÃ³Å‚dzielone helpery: .githooks/_common.py
"""
from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path

# Upewniamy siÄ™, Å¼e moÅ¼emy zaimportowaÄ‡ helpery z .githooks/_common.py
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
    """WiadomoÅ›ci zwolnione z CC (merge, revert, fixup/squash)."""
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
    JeÅ›li istnieje .glx/commit_snippet.txt i nie ma go jeszcze w wiadomoÅ›ci,
    dodajemy go na koÅ„cu (po jednej pustej linii).
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
        return msg_text  # juÅ¼ wstawione

    # Dodajemy separator (pusta linia) + snippet
    if not msg_text.endswith("\n"):
        msg_text += "\n"
    if not msg_text.endswith("\n\n"):
        msg_text += "\n"
    msg_text += snippet + "\n"
    return msg_text


def main() -> int:
    if len(sys.argv) < 2:
        H.fail("Brak Å›cieÅ¼ki do pliku wiadomoÅ›ci (COMMIT_EDITMSG)")

    msg_file = Path(sys.argv[1]).resolve()
    repo = H.git_root(THIS_DIR)

    if not msg_file.exists():
        H.fail(f"Nie znaleziono pliku wiadomoÅ›ci: {msg_file}")

    raw = msg_file.read_text(encoding="utf-8", errors="replace")

    # Pierwsza niepusta linia to nagÅ‚Ã³wek
    header = ""
    for line in raw.splitlines():
        if line.strip() and not line.strip().startswith("#"):
            header = line.strip()
            break

    if not header:
        H.fail(
            "Pusta wiadomoÅ›Ä‡ commita. UÅ¼yj formatu Conventional Commits, np.:\n"
            "  feat(app): panel Delta Inspector"
        )

    # WyjÄ…tki (merge/revert/fixup/squash)
    if _is_exempt(header):
        H.log("commit-msg: wyjÄ…tek (merge/revert/fixup/squash) â€” przepuszczam.")
        return 0

    # Walidacja wzorca CC
    if not CC_RE.match(header):
        allowed = "|".join(TYPES)
        H.fail(
            "commit-msg: niepoprawny nagÅ‚Ã³wek.\n"
            f"Dozwolone typy: {allowed}\n"
            "Wzorzec: type(scope)?: opis\n"
            "PrzykÅ‚ady:\n"
            "  feat(app): panel Delta Inspector\n"
            "  fix(core): obsÅ‚uga brzegowego przypadku w Î”-SSIM\n"
        )

    # DÅ‚ugoÅ›Ä‡ nagÅ‚Ã³wka
    if len(header) > MAX_HEADER:
        H.fail(
            f"commit-msg: nagÅ‚Ã³wek zbyt dÅ‚ugi ({len(header)} > {MAX_HEADER}). "
            "SkrÃ³Ä‡ opis do sedna, szczegÃ³Å‚y przenieÅ› do body."
        )

    # Pusta linia przed body (jeÅ›li body istnieje)
    lines = raw.splitlines()
    if len(lines) >= 2:
        # znajdÅº pozycjÄ™ nagÅ‚Ã³wka
        try:
            idx = lines.index(header)
        except ValueError:
            idx = 0
        # jeÅ›li istniejÄ… kolejne linie i druga nie jest pusta â†’ wstaw pustÄ…
        if idx + 1 < len(lines) and lines[idx + 1].strip() and not lines[idx + 1].startswith("#"):
            # wstawimy pustÄ… i zapiszemy
            lines.insert(idx + 1, "")

    new_text = "\n".join(lines) + ("\n" if not raw.endswith("\n") else "")

    # Opcjonalna wstawka GLX (Î”-tokens/fingerprint) z artefaktu
    # (zwykle generowana przez pre-push lub post-commit)
    new_text2 = _append_snippet_if_available(repo, msg_file, new_text)

    if new_text2 != raw:
        # Nadpisujemy plik wiadomoÅ›ci
        msg_file.write_text(new_text2, encoding="utf-8")

    H.log("commit-msg: OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        H.fail("Przerwano przez uÅ¼ytkownika")
