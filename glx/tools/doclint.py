#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""glx.tools.doclint — fail-closed consistency checks for the current GLX docs corpus.

The historical linter expected YAML front matter and a ``99_refactor_plan.md``
file that are not part of the current repository documentation convention. That
made the check permanently red and led to it being configured as non-blocking.
This module validates properties that are actually represented by the current
repository instead of maintaining a fictional schema.

Hard checks:
- the current canonical documentation paths exist;
- active core documents are non-empty and UTF-8 readable;
- the glossary contains the GLX semantic primitives S/H/Z, Δ, Φ, Ψ and I1–I4;
- the retired ``src/gui`` or top-level ``gui`` implementation tree is not
  reintroduced (the current application tree is ``src/app``).

Exit status: 0 = consistent, 1 = inconsistency found.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List


def _docs_dir() -> Path:
    legacy = Path("glitchlab") / "docs"
    if legacy.exists():
        return legacy
    return Path("docs")


# Current checked-in documentation contract. Some files are intentionally kept
# as placeholders, so existence is required while non-emptiness is enforced only
# for the active documents below.
REQUIRED_FILES = {
    "00_overview.md",
    "10_architecture.md",
    "11_spec_glossary.md",
    "12_invariants.md",
    "13_delta_algebra.md",
    "14_mosaic.md",
    "20_bus.md",
    "21_egdb.md",
    "22_analytics.md",
    "30_sast_bridge.md",
    "40_gui_app.md",
    "41_pipelines.md",
    "50_ci_ops.md",
    "60_security.md",
    "70_observability.md",
    "82_release_and_channels.md",
    "92_playbooks.md",
}

CORE_NONEMPTY_FILES = {
    "00_overview.md",
    "10_architecture.md",
    "11_spec_glossary.md",
    "12_invariants.md",
    "13_delta_algebra.md",
    "21_egdb.md",
    "22_analytics.md",
    "30_sast_bridge.md",
    "50_ci_ops.md",
    "60_security.md",
    "70_observability.md",
    "92_playbooks.md",
}

GLOSSARY_TOKENS = ("S", "H", "Z", "Δ", "Φ", "Ψ")


def _read_utf8(path: Path) -> tuple[str, List[str]]:
    try:
        return path.read_text(encoding="utf-8"), []
    except Exception as exc:
        return "", [f"{path.name}: nie można odczytać jako UTF-8: {exc}"]


def _check_file(path: Path) -> List[str]:
    text, errors = _read_utf8(path)
    if errors:
        return errors

    if path.name in CORE_NONEMPTY_FILES and not text.strip():
        errors.append(f"{path.name}: aktywny dokument jest pusty")

    return errors


def _check_glossary(path: Path) -> List[str]:
    text, errors = _read_utf8(path)
    if errors:
        return errors

    for token in GLOSSARY_TOKENS:
        if token not in text:
            errors.append(f"{path.name}: brak kanonicznego tokenu {token!r}")

    if "I1" not in text or "I4" not in text:
        errors.append(f"{path.name}: brak zakresu inwariantów I1–I4")
    return errors


def _check_source_topology() -> List[str]:
    errors: List[str] = []
    for retired in (Path("src/gui"), Path("gui")):
        if retired.exists():
            errors.append(
                f"wykryto przywróconą przestarzałą ścieżkę implementacji {retired.as_posix()!r}; użyj 'src/app'"
            )
    return errors


def main() -> int:
    docs_dir = _docs_dir()
    if not docs_dir.exists():
        print("[doclint] brak katalogu docs (ani glitchlab/docs)", file=sys.stderr)
        return 1

    missing = sorted(name for name in REQUIRED_FILES if not (docs_dir / name).is_file())
    if missing:
        print(f"[doclint] brak plików: {missing}", file=sys.stderr)
        return 1

    errors: List[str] = []
    for name in sorted(REQUIRED_FILES):
        errors.extend(_check_file(docs_dir / name))
    errors.extend(_check_glossary(docs_dir / "11_spec_glossary.md"))
    errors.extend(_check_source_topology())

    if errors:
        for error in errors:
            print(f"[doclint] {error}", file=sys.stderr)
        return 1

    print("[doclint] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
