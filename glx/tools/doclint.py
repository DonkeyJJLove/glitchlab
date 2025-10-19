#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glx.tools.doclint — szybki lint zestawu dokumentacji GLX (Python 3.9)

Sprawdzenia (twarde):
- Istnieją wszystkie wymagane pliki w glitchlab/docs/.
- Każdy plik ma prawidłowy front-matter YAML na początku (--- ... ---).
- Front-matter zawiera: title, version==v1.0, doc-id==./<nazwa>, status==final,
  spec zawiera co najmniej: S,H,Z, Δ, Φ, Ψ, I1–I4 (akceptuje 'I1–I4' lub 'I1-I4'),
  ownership==GLX-Core, sekcję links z wpisem (rel: glossary, href: ./11_spec_glossary.md).
- Treść nie zawiera przestarzałych ścieżek 'gui/' (po refaktorze powinno być 'app/').

Zakończenie:
- 0 → OK
- 1 → wykryto błędy (wypisane na stderr)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

DOCS = Path("glitchlab") / "docs"

REQUIRED_FILES = {
    "00_overview.md", "10_architecture.md", "11_spec_glossary.md", "12_invariants.md",
    "13_delta_algebra.md", "14_mosaic.md", "20_bus.md", "21_egdb.md", "22_analytics.md",
    "30_sast_bridge.md", "40_gui_app.md", "41_pipelines.md", "50_ci_ops.md", "60_security.md",
    "70_observability.md", "82_release_and_channels.md", "92_playbooks.md", "99_refactor_plan.md",
}

REQ_VERSION = "v1.0"
REQ_STATUS = "final"
REQ_OWNERSHIP = "GLX-Core"
REQ_GLOSSARY_HREF = "./11_spec_glossary.md"

# Delimitery front-matter na początku pliku
FM_BEGIN = re.compile(r"^---\s*\n", re.DOTALL)
FM_END = re.compile(r"\n---\s*\n", re.DOTALL)

# Prosty parser klucz: wartość (w obrębie front-matter)
KV_RE = re.compile(r"^([a-zA-Z0-9_\-]+)\s*:\s*(.+?)\s*$")

# Dopuszczamy zarówno en-dash (–) jak i minus (-) w zapisie I1–I4
def _spec_tokens_ok(spec_raw: str) -> bool:
    # Wyciągnij elementy ze środka nawiasów kwadratowych, np. [S,H,Z, Δ, Φ, Ψ, I1–I4]
    # Usuwamy nawiasy, rozbijamy po przecinkach/whitespace
    s = spec_raw.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip() for p in re.split(r"[,\s]+", s) if p.strip()]
    items = set(parts)
    # Normalizuj zapis I1–I4
    i14_ok = ("I1–I4" in items) or ("I1-I4" in items)
    must = {"S", "H", "Z", "Δ", "Φ", "Ψ"}
    return i14_ok and must.issubset(items)

def _extract_front_matter(text: str) -> Tuple[str, str]:
    """
    Zwraca (blok_front_matter, reszta_treści). Gdy brak poprawnego FM, zwraca ("","").
    """
    if not FM_BEGIN.match(text):
        return "", ""
    # znajdź koniec
    m = FM_END.search(text, 4)  # po '---\n'
    if not m:
        return "", ""
    start = 4  # po pierwszym '---\n'
    end = m.start()
    fm = text[start:end]
    rest = text[m.end():]
    return fm, rest

def _parse_front_matter(fm: str) -> Dict[str, str]:
    """
    Bardzo prosty parser klucz: wartość dla płaskich pól. Sekcje zagnieżdżone
    (links) weryfikujemy regexami kontekstowymi poniżej.
    """
    data: Dict[str, str] = {}
    for line in fm.splitlines():
        line = line.rstrip()
        if not line or line.strip().startswith("#"):
            continue
        m = KV_RE.match(line)
        if m:
            k, v = m.group(1), m.group(2)
            data[k] = v
    return data

def _has_glossary_link(fm: str) -> bool:
    """
    Sprawdza, czy front-matter zawiera blok:
      links:
        - rel: glossary
          href: ./11_spec_glossary.md
    Dopuszczamy dowolne odstępy i dodatkowe wpisy w links.
    """
    # szukamy 'links:' a dalej co najmniej jeden wpis z rel: glossary i właściwym href
    links_block = re.search(r"(?ms)^\s*links\s*:\s*\n(.+?)(?:^\S| \Z)", fm + "\nX")  # do następnego klucza lub końca
    if not links_block:
        return False
    block = links_block.group(1)
    has_rel = re.search(r"(?m)^\s*-\s*rel\s*:\s*glossary\s*$", block) is not None
    has_href = re.search(r"(?m)^\s*href\s*:\s*\.\/11_spec_glossary\.md\s*$", block) is not None
    return bool(has_rel and has_href)

def _check_file(md_path: Path) -> List[str]:
    errors: List[str] = []
    try:
        text = md_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return [f"{md_path.name}: nie można odczytać pliku: {e}"]

    # 1) Front-matter
    fm, body = _extract_front_matter(text)
    if not fm:
        return [f"{md_path.name}: brak lub niekompletny front-matter na początku pliku"]

    meta = _parse_front_matter(fm)
    # title
    if "title" not in meta or not meta["title"].strip():
        errors.append(f"{md_path.name}: brak 'title' w front-matter")
    # version
    if meta.get("version") != REQ_VERSION:
        errors.append(f"{md_path.name}: 'version' musi być '{REQ_VERSION}'")
    # doc-id
    expected_docid = f"./{md_path.name}"
    if meta.get("doc-id") != expected_docid:
        errors.append(f"{md_path.name}: 'doc-id' musi być '{expected_docid}'")
    # status
    if meta.get("status") != REQ_STATUS:
        errors.append(f"{md_path.name}: 'status' musi być '{REQ_STATUS}'")
    # ownership
    if meta.get("ownership") != REQ_OWNERSHIP:
        errors.append(f"{md_path.name}: 'ownership' musi być '{REQ_OWNERSHIP}'")
    # spec
    spec_val = meta.get("spec")
    if spec_val is None or not _spec_tokens_ok(spec_val):
        errors.append(f"{md_path.name}: pole 'spec' musi zawierać S,H,Z, Δ, Φ, Ψ, I1–I4")
    # links → glossary
    if not _has_glossary_link(fm):
        errors.append(f"{md_path.name}: sekcja 'links' musi zawierać wpis (rel: glossary, href: {REQ_GLOSSARY_HREF})")

    # 2) Migracja GUI→APP
    if "gui/" in text:
        errors.append(f"{md_path.name}: wykryto przestarzałe ścieżki 'gui/' — zamień na 'app/'")

    return errors

def main() -> int:
    if not DOCS.exists():
        print("[doclint] brak katalogu glitchlab/docs", file=sys.stderr)
        return 1

    # 0) kompletność zestawu
    missing = [f for f in REQUIRED_FILES if not (DOCS / f).exists()]
    if missing:
        print(f"[doclint] brak plików: {sorted(missing)}", file=sys.stderr)
        return 1

    # 1) walidacja każdego pliku
    any_errors = False
    for md in sorted(DOCS.glob("*.md")):
        errs = _check_file(md)
        if errs:
            any_errors = True
            for e in errs:
                print(f"[doclint] {e}", file=sys.stderr)

    if any_errors:
        return 1

    print("[doclint] OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
