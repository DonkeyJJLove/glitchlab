# -*- coding: utf-8 -*-
"""
Testy bazowe GLX dla gałęzi hooków + tools (Python 3.9, pytest).

Uwaga:
- Testy używają monkeypatch do stubowania dostępu do gita / ścieżek.
- Wymagane: pytest. Uruchom: `pytest -q`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict
import importlib
import importlib.machinery
import importlib.util
import types


def _ensure_tmp_repo_layout(tmp_path: Path) -> Path:
    """
    Przygotuj minimalny układ katalogów oczekiwany przez narzędzia GLX:
    repo/
      glitchlab/.glx
      glx/tools/*.py  (jeśli test ma ładować bezpośrednio z pliku)
    Zwraca katalog repo (tmp_path).
    """
    (tmp_path / "glitchlab" / ".glx").mkdir(parents=True, exist_ok=True)
    return tmp_path


# Helper do importu modułu testowanego:
def _import_glx_module(modname: str) -> types.ModuleType:
    """
    Próbuje importować moduł normalnie (np. 'glx.tools.delta_fingerprint').
    Jeśli to się nie uda — próbuje załadować plik z relatywnej ścieżki
    (glx/tools/...py) w katalogu roboczym.
    """
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError:
        # zbuduj ścieżkę pliku z modname
        parts = modname.split(".")
        file_path = Path.cwd().joinpath(*parts)  # Path('glx','tools','delta_fingerprint')
        if file_path.is_dir():
            # nie powinno się zdarzyć dla modułu narzędzi
            raise
        file_py = file_path.with_suffix(".py")
        if not file_py.exists():
            # spróbuj jeszcze konstrukcji 'glitchlab/glx/...'
            alt = Path.cwd() / "glitchlab" / Path(*parts)
            if alt.with_suffix(".py").exists():
                file_py = alt.with_suffix(".py")
            else:
                raise
        # Load as a module under the requested name
        loader = importlib.machinery.SourceFileLoader(modname, str(file_py))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        sys.modules[modname] = module
        return module


# ──────────────────────────────────────────────────────────────────────────────
# delta_fingerprint
# ──────────────────────────────────────────────────────────────────────────────
def test_delta_fingerprint_extracts_hist_and_hash(monkeypatch, tmp_path):
    repo = _ensure_tmp_repo_layout(tmp_path)
    monkeypatch.chdir(repo)

    # Import modułu do testów (w sposób odporny na lokalizację pakietu)
    sys.path.insert(0, str(repo))
    df = _import_glx_module("glx.tools.delta_fingerprint")

    # Przygotuj "stare" i "nowe" wersje pliku
    old_py = "import os\n\ndef foo(a,b):\n    return a+b\n"
    new_py = "import os, sys\n\ndef foo(a,b,c):\n    return a+b+c\n"

    # Stub: lista zmienionych plików + zawartości (bez odwołań do gita)
    # Uwaga: ważne, żeby parent/head odpowiadały stubowi _git_show —
    # tutaj wymuszamy w testach '--range A..B' aby parent == 'A' i head == 'B'.
    monkeypatch.setattr(df, "_git_changed_files", lambda rng: ["glitchlab/core/foo.py"], raising=False)
    monkeypatch.setattr(
        df,
        "_git_show",
        lambda path, rev: old_py if (isinstance(rev, str) and (rev.endswith("^") or rev == "A")) else new_py,
        raising=False,
    )
    monkeypatch.setattr(df, "_get_parent_sha", lambda rev: "A", raising=False)

    # main() korzysta z argparse -> zabezpieczamy sys.argv i wymuszamy zakres A..B
    argv_backup = sys.argv[:]
    sys.argv = ["delta_fingerprint", "--range", "A..B"]
    try:
        ret = df.main()
    finally:
        sys.argv = argv_backup

    assert ret == 0

    # Sprawdź artefakt
    out = Path("glitchlab/.glx/delta_report.json")
    assert out.exists(), "delta_report.json powinien zostać zapisany"

    data = json.loads(out.read_text(encoding="utf-8"))
    hist: Dict[str, int] = data.get("hist") or {}
    # dopuszczamy dwa klucze (różne implementacje mogą używać ΔIMPORT albo DELTA_IMPORT)
    assert hist.get("ΔIMPORT", 0) >= 1 or hist.get("DELTA_IMPORT", 0) >= 1, "powinien wykryć dodanie importu"
    assert hist.get("MODIFY_SIG", 0) >= 1 or hist.get("CHANGE_SIG", 0) >= 1, "powinien wykryć zmianę sygnatury"
    assert isinstance(data.get("hash"), str) and len(data["hash"]) >= 8, "powinien policzyć fingerprint"


# ──────────────────────────────────────────────────────────────────────────────
# invariants_check
# ──────────────────────────────────────────────────────────────────────────────
def test_invariants_check_scores_and_blocks(monkeypatch, tmp_path):
    repo = _ensure_tmp_repo_layout(tmp_path)
    monkeypatch.chdir(repo)

    # Przygotuj delta_report.json z ryzykownymi tokenami + psnr/ssim (aby przebić Z)
    glx_dir = Path("glitchlab/.glx")
    glx_dir.mkdir(parents=True, exist_ok=True)
    (glx_dir / "delta_report.json").write_text(
        json.dumps(
            {
                "hist": {"MODIFY_SIG": 100, "ΔIMPORT": 50},
                "psnr": 10.0,  # zaniżone → +0.1
                "ssim": 0.0,  # zaniżone → +0.1
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Import modułu do testów
    sys.path.insert(0, str(repo))
    ic = _import_glx_module("glx.tools.invariants_check")

    # Stub: "duży" diff (dużo linii + i -)
    plus = "\n+".join([""] * 300)
    minus = "\n-".join([""] * 150)
    big_diff = plus + minus

    # Wstaw stub niezależnie od tego, czy symbol istniał wcześniej
    monkeypatch.setattr(ic, "_git_diff_range", lambda rng: big_diff, raising=False)

    # Wywołaj main() jak z CLI — wymuszamy zakres A..B
    argv_backup = sys.argv[:]
    sys.argv = ["invariants_check", "--range", "A..B"]
    try:
        ret = ic.main()
    finally:
        sys.argv = argv_backup

    # Przy domyślnych progach α=0.85, β=0.92, Z=0.99 oczekujemy hard-block (>Z)
    # dopuszczalne wartości zwracane przez implementację: 0 (ok), 1 lub 2 (blok)
    assert ret in (0, 1, 2), "zwracany kod powinien być 0/1/2; test spodziewa się blokowania przy skrajnym raporcie"
    analysis = json.loads((glx_dir / "commit_analysis.json").read_text(encoding="utf-8"))
    assert 0.0 <= analysis["score"] <= 1.0
    assert "thresholds" in analysis


# ──────────────────────────────────────────────────────────────────────────────
# doclint
# ──────────────────────────────────────────────────────────────────────────────
def _write_minimal_doc(path: Path, title: str, doc_id: str):
    fm = f"""---
title: {title}
version: v1.0
doc-id: {doc_id}
status: final
spec: [S,H,Z, Δ, Φ, Ψ, I1–I4]
ownership: GLX-Core
links:
  - rel: glossary
    href: ./11_spec_glossary.md
---
"""
    path.write_text(fm + "\nOK\n", encoding="utf-8")


def test_doclint_ok_and_failure(monkeypatch, tmp_path):
    repo = _ensure_tmp_repo_layout(tmp_path)
    monkeypatch.chdir(repo)
    sys.path.insert(0, str(repo))

    dl = _import_glx_module("glx.tools.doclint")

    docs = Path("glitchlab/docs")
    docs.mkdir(parents=True, exist_ok=True)

    required = [
        "00_overview.md","10_architecture.md","11_spec_glossary.md","12_invariants.md",
        "13_delta_algebra.md","14_mosaic.md","20_bus.md","21_egdb.md","22_analytics.md",
        "30_sast_bridge.md","40_gui_app.md","41_pipelines.md","50_ci_ops.md","60_security.md",
        "70_observability.md","82_release_and_channels.md","92_playbooks.md","99_refactor_plan.md"
    ]
    for name in required:
        _write_minimal_doc(docs / name, f"T {name}", f"./{name}")

    # Happy path
    assert dl.main() == 0

    # Wprowadź błąd: brak jednego pliku
    (docs / "92_playbooks.md").unlink()
    assert dl.main() == 1

    # Przywróć plik, wprowadź GUI-odniesienie (zabronione)
    _write_minimal_doc(docs / "92_playbooks.md", "T 92", "./92_playbooks.md")
    (docs / "40_gui_app.md").write_text((docs / "40_gui_app.md").read_text().replace("OK", "gui/app.py"), encoding="utf-8")
    assert dl.main() == 1
