#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glx.tools.delta_fingerprint — ekstrakcja histogramu Δ-tokenów i fingerprint (Python 3.9)

Funkcje:
- Analiza zmian w zakresie commitów (A..B lub pojedynczy commit) z użyciem `git diff`.
- Ekstrakcja prostych Δ-tokenów dla plików .py:
    * ΔIMPORT      — dodane importy
    * ADD_FN       — dodane funkcje
    * DEL_FN       — usunięte funkcje
    * MODIFY_SIG   — zmienione sygnatury funkcji
- Zapis artefaktu: glitchlab/.glx/delta_report.json
- Fingerprint: SHA-256 histogramu (pierwsze 16 hex znaków)

Uwaga:
- Moduł używa wyłącznie stdlib.
- Testy mogą stubować `_git_changed_files`, `_git_show`, `_get_parent_sha`.

CLI:
    python -m glx.tools.delta_fingerprint --range A..B
    python -m glx.tools.delta_fingerprint            # domyślnie HEAD~1..HEAD
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# ──────────────────────────────────────────────────────────────────────────────
# Konfiguracja ścieżek artefaktów
# ──────────────────────────────────────────────────────────────────────────────
GLX_DIR = Path("glitchlab/.glx")
DELTA_REPORT = GLX_DIR / "delta_report.json"


# ──────────────────────────────────────────────────────────────────────────────
# GIT helpers — nadpisywane w testach monkeypatch'em
# ──────────────────────────────────────────────────────────────────────────────
def _git_changed_files(commit_range: str) -> List[str]:
    """
    Zwraca listę zmienionych plików w zakresie (tylko ścieżki).
    Obsługuje formę A..B lub pojedynczy commit (B^!..).
    """
    args = ["git", "diff", "--name-only"]
    if ".." in commit_range:
        args.append(commit_range)
    else:
        args += [f"{commit_range}^!", "--"]
    out = subprocess.check_output(args, text=True)
    return [p for p in out.splitlines() if p.strip()]


def _get_parent_sha(rev: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", f"{rev}^"], text=True).strip()
    except subprocess.CalledProcessError:
        return ""


def _git_show(path: str, rev: str) -> str:
    """
    Zwraca zawartość pliku `path` na rewizji `rev`. Gdy nie istnieje — pusty string.
    """
    try:
        return subprocess.check_output(["git", "show", f"{rev}:{path}"], text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Ekstrakcja Δ-tokenów (dla plików .py)
# ──────────────────────────────────────────────────────────────────────────────
PY_DEF_RE = re.compile(r'^\s*def\s+([a-zA-Z_]\w*)\s*\((.*?)\)\s*:', re.M)
IMPORT_RE = re.compile(r'^\s*(?:from\s+[^\s]+|import\s+.+)$', re.M)


def _py_tokens(old: str, new: str) -> Dict[str, int]:
    """
    Porównuje dwie wersje pliku .py i zwraca histogram Δ-tokenów.
    """
    tokens: Dict[str, int] = {}

    def add(tok: str, n: int = 1) -> None:
        tokens[tok] = tokens.get(tok, 0) + n

    # Importy
    old_imps = set(IMPORT_RE.findall(old or ""))
    new_imps = set(IMPORT_RE.findall(new or ""))
    imp_added = len(new_imps - old_imps)
    if imp_added > 0:
        add("ΔIMPORT", imp_added)

    # Funkcje i ich sygnatury
    old_defs = {m.group(1): m.group(2) for m in PY_DEF_RE.finditer(old or "")}
    new_defs = {m.group(1): m.group(2) for m in PY_DEF_RE.finditer(new or "")}

    # Dodane/usunięte
    for fn in new_defs.keys() - old_defs.keys():
        add("ADD_FN")
    for fn in old_defs.keys() - new_defs.keys():
        add("DEL_FN")

    # Zmiana sygnatury
    for fn in (new_defs.keys() & old_defs.keys()):
        if new_defs[fn] != old_defs[fn]:
            add("MODIFY_SIG")

    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Fingerprint histogramu
# ──────────────────────────────────────────────────────────────────────────────
def _fingerprint(hist: Dict[str, int]) -> str:
    items = sorted(hist.items())
    payload = ";".join(f"{k}={v}" for k, v in items).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────────
# Główna ścieżka
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_dirs() -> None:
    GLX_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_parent_head(commit_range: str) -> (str, str):
    """
    Zwraca (parent, head) dla zakresu. Dla pojedynczego commita używa parent=`rev^`.
    """
    if ".." in commit_range:
        a, b = commit_range.split("..", 1)
        return a, b
    return _get_parent_sha(commit_range), commit_range


def run(commit_range: str) -> Dict[str, object]:
    """
    Uruchamia ekstrakcję Δ-tokenów dla podanego zakresu.
    Zwraca strukturę delta_report (hist, hash, range).
    """
    parent, head = _resolve_parent_head(commit_range)
    hist: Dict[str, int] = {}

    for path in _git_changed_files(commit_range):
        if not path.endswith(".py"):
            continue
        old = _git_show(path, parent) if parent else ""
        new = _git_show(path, head) if head else ""
        if not (old or new):
            continue
        dt = _py_tokens(old, new)
        for k, v in dt.items():
            hist[k] = hist.get(k, 0) + int(v)

    rep = {"range": commit_range, "hist": hist, "hash": _fingerprint(hist)}
    return rep


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", default="HEAD~1..HEAD", help="Zakres diff (A..B) lub pojedynczy commit")
    args = ap.parse_args(argv)

    _ensure_dirs()
    try:
        rep = run(args.range)
    except Exception as e:
        # bezpieczny fallback — nie generujemy artefaktu, ale sygnalizujemy błąd
        print(f"[delta_fingerprint] ERROR: {e}", file=sys.stderr)
        return 1

    # Zapis artefaktu
    try:
        DELTA_REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[delta_fingerprint] cannot write {DELTA_REPORT}: {e}", file=sys.stderr)
        return 1

    # Log na stdout (użyteczne w CI)
    print(json.dumps(rep, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
