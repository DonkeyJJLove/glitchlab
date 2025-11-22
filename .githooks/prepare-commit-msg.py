#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.githooks/prepare-commit-msg.py â€” podpowiedÅº nagÅ‚Ã³wka CC + wskazÃ³wki GLX (Python 3.9)

Zadania:
- JeÅ›li nagÅ‚Ã³wek jest pusty / brak treÅ›ci: wstaw szablon Conventional Commit z dobranym scope.
- Dla istniejÄ…cej treÅ›ci: NIE modyfikuj nagÅ‚Ã³wka; dopisz (w komentarzu) wskazÃ³wki GLX (Î”-tokens, fingerprint).
- Nie dziaÅ‚a na commitach MERGE/REVERT/FIXUP/SQUASH/TAG (zachowuje oryginaÅ‚).

Sygnatura wywoÅ‚ania (Git):
    prepare-commit-msg <path> [source] [sha]

Å¹rÃ³dÅ‚a (source):
- message | template | commit | merge | squash | tag | revert | (puste)
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Helpery wspÃ³lne
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
import _common as H  # noqa: E402

CC_TYPES = ("feat", "fix", "perf", "refactor", "docs", "ci", "build", "test", "chore")
CC_RE = re.compile(r"^(%s)(\([a-z0-9_\-\/\.]+\))?:\s+\S.*$" % "|".join(CC_TYPES))

EXEMPT_SOURCES = {"merge", "squash", "tag"}
EXEMPT_PREFIXES = ("merge ", "revert ", "fixup!", "squash!")


def _deduce_scope(repo: Path) -> str:
    """
    Heurystyka: wyznacz scope na podstawie staged plikÃ³w.
    Priorytet: app/core/analysis/mosaic/delta/docs â†’ odpowiedni scope; inaczej 'repo'.
    """
    staged = H.staged_paths(repo)
    roots = [str(p.relative_to(repo)).replace("\\", "/").split("/", 2)[0] for p in staged]
    # mapuj znane korzenie
    if any(r == "glitchlab" for r in roots):
        # sprawdÅº drugi poziom
        second = []
        for p in staged:
            rel = str(p.relative_to(repo)).replace("\\", "/")
            parts = rel.split("/")
            if parts and parts[0] == "glitchlab" and len(parts) >= 2:
                second.append(parts[1])
        # hierarchia preferencji
        prefs = ["app", "core", "analysis", "mosaic", "delta", "docs"]
        for pref in prefs:
            if pref in second:
                return pref
        if second:
            return second[0]
    # alternatywa: top-level katalog
    if roots:
        return roots[0]
    return "repo"


def _has_header(lines: list[str]) -> bool:
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        return True
    return False


def _first_header(lines: list[str]) -> str:
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        return s
    return ""


def _is_exempt_header(header: str) -> bool:
    h = header.strip().lower()
    return any(h.startswith(pfx) for pfx in EXEMPT_PREFIXES)


def _append_glx_hint(repo: Path, text: str) -> str:
    """
    Dopisuje w komentarzu (# â€¦) skrÃ³t GLX (Î”-tokens top-k oraz fingerprint),
    Å¼eby autor miaÅ‚ kontekst przy redagowaniu wiadomoÅ›ci.
    commit-msg.py (pÃ³Åºniejszy hook) i tak doÅ‚Ä…czy realny snippet, jeÅ¼eli dostÄ™pny.
    """
    hint = H.build_commit_snippet(repo)
    if not hint:
        return text
    # ZamieÅ„ wiersze na komentarze
    commented = "\n".join("# " + ln for ln in hint.splitlines() if ln.strip())
    if not text.endswith("\n"):
        text += "\n"
    if not text.endswith("\n\n"):
        text += "\n"
    text += commented + "\n"
    return text


def main() -> int:
    if len(sys.argv) < 2:
        H.fail("Brak Å›cieÅ¼ki do pliku wiadomoÅ›ci (COMMIT_EDITMSG)")

    msg_path = Path(sys.argv[1]).resolve()
    source = sys.argv[2] if len(sys.argv) >= 3 else ""

    repo = H.git_root(THIS_DIR)

    # WyjÄ…tki wg ÅºrÃ³dÅ‚a
    if source in EXEMPT_SOURCES:
        H.log(f"prepare-commit-msg: source={source} (exempt) â€” bez zmian.")
        return 0

    raw = msg_path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    # JeÅ›li nagÅ‚Ã³wek istnieje i nie jest exempt â€” nie dotykamy nagÅ‚Ã³wka
    header_present = _has_header(lines)
    if header_present:
        header = _first_header(lines)
        if _is_exempt_header(header):
            H.log("prepare-commit-msg: exempt header â€” bez zmian.")
            return 0
        # tylko dopisz wskazÃ³wkÄ™ GLX (komentarze)
        new_text = _append_glx_hint(repo, raw)
        if new_text != raw:
            msg_path.write_text(new_text, encoding="utf-8")
        H.log("prepare-commit-msg: dopisano wskazÃ³wkÄ™ GLX.")
        return 0

    # Brak nagÅ‚Ã³wka â†’ wstaw szablon CC z heurystycznym scope
    scope = _deduce_scope(repo)
    template = f"feat({scope}): "  # autor uzupeÅ‚ni opis
    out_lines = [template, "", "# UÅ¼yj Conventional Commits. PrzykÅ‚ad:", "# fix(core): obsÅ‚uga brzegowego przypadku w Î”-SSIM"]
    text = "\n".join(out_lines) + ("\n" if not raw.endswith("\n") else "")
    text = _append_glx_hint(repo, text)
    msg_path.write_text(text, encoding="utf-8")
    H.log(f"prepare-commit-msg: wstawiono szablon CC z scope='{scope}'.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        H.fail("Przerwano przez uÅ¼ytkownika")
