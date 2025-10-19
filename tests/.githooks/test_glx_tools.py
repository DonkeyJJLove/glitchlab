# -*- coding: utf-8 -*-
"""
Lżejszy zestaw testów integracyjnych dla glx.tools.*
- odporny na różne odmiany importu (glx.* lub glitchlab.glx.*)
- nie oczekuje konkretnych wartości progów/blokad, tylko poprawnych artefaktów i typów
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import pytest


def _ensure_tmp_repo_layout(tmp_path: Path) -> Path:
    (tmp_path / "glitchlab" / ".glx").mkdir(parents=True, exist_ok=True)
    (tmp_path / "glitchlab" / "docs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _import_glx_module(name: str):
    """
    Próbuje zaimportować modulę nazwą 'glx.tools.xxx'. Jeśli nie uda się,
    próbuje 'glitchlab.glx.tools.xxx'. Podnosi ImportError jeśli obie próby zawiodą.
    """
    try:
        return importlib.import_module(name)
    except Exception:
        alt = name.replace("glx.", "glitchlab.glx.")
        return importlib.import_module(alt)


# ---------------------------
# delta_fingerprint (minimal)
# ---------------------------
def test_delta_fingerprint_minimal(monkeypatch, tmp_path):
    repo = _ensure_tmp_repo_layout(tmp_path)
    monkeypatch.chdir(repo)

    # Umożliwiamy import z katalogu tymczasowego
    sys.path.insert(0, str(repo))

    df = _import_glx_module("glx.tools.delta_fingerprint")

    # Przygotuj proste stuby, bez zależności od gita.
    # Nie zakładamy istnienia symboli — ustawiamy raising=False, by dodać je jeśli brak.
    monkeypatch.setattr(df, "_git_changed_files", lambda rng: ["glitchlab/core/foo.py"], raising=False)
    old_py = "import os\n\ndef foo(a,b):\n    return a+b\n"
    new_py = "import os, sys\n\ndef foo(a,b,c):\n    return a+b+c\n"
    monkeypatch.setattr(
        df, "_git_show",
        lambda path, rev: old_py if (isinstance(rev, str) and (rev.endswith("^") or rev == "A")) else new_py,
        raising=False,
    )
    monkeypatch.setattr(df, "_get_parent_sha", lambda rev: "A", raising=False)

    # main() najpewniej korzysta z argparse -> zabezpieczamy sys.argv
    argv_orig = sys.argv[:]
    sys.argv = ["delta_fingerprint"]
    try:
        ret = df.main()
    finally:
        sys.argv = argv_orig

    # Oczekujemy kodu całkowitego i artefaktu delta_report.json
    assert isinstance(ret, int)
    out = Path("glitchlab/.glx/delta_report.json")
    assert out.exists(), "delta_report.json powinien zostać utworzony"

    # Sprawdź strukturę pliku (hist jako dict, hash jako string)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "hist" in data and isinstance(data["hist"], dict)
    assert "hash" in data and isinstance(data["hash"], str) and len(data["hash"]) > 0


# ---------------------------
# invariants_check (minimal)
# ---------------------------
def test_invariants_check_minimal(monkeypatch, tmp_path):
    repo = _ensure_tmp_repo_layout(tmp_path)
    monkeypatch.chdir(repo)

    # Stwórz delta_report.json z zawartością, która powinna dać sensowne score
    glx_dir = Path("glitchlab/.glx")
    glx_dir.mkdir(parents=True, exist_ok=True)
    (glx_dir / "delta_report.json").write_text(
        json.dumps({"hist": {"MODIFY_SIG": 10, "ΔIMPORT": 5}, "psnr": 10.0, "ssim": 0.0}, ensure_ascii=False),
        encoding="utf-8",
    )

    sys.path.insert(0, str(repo))
    ic = _import_glx_module("glx.tools.invariants_check")

    # Stubuj pobieranie diffu. Używamy raising=False, bo symbol może mieć inną nazwę.
    big_diff = "\n".join(["+line"] * 200 + ["-line"] * 100)
    # Invariants module provides _git_diff_text — jeśli nie, dopisujemy
    monkeypatch.setattr(ic, "_git_diff_text", lambda rng: big_diff, raising=False)

    # Zabezpiecz argv i wywołaj main
    argv_orig = sys.argv[:]
    sys.argv = ["invariants_check", "--range", "A..B"]
    try:
        ret = ic.main()
    finally:
        sys.argv = argv_orig

    # main powinien zwracać int, a artefakt commit_analysis.json powinien istnieć
    assert isinstance(ret, int)
    analysis_file = glx_dir / "commit_analysis.json"
    assert analysis_file.exists(), "commit_analysis.json powinien zostać zapisany"

    analysis = json.loads(analysis_file.read_text(encoding="utf-8"))
    assert isinstance(analysis, dict)
    assert "score" in analysis and isinstance(analysis["score"], (int, float))
    assert 0.0 <= float(analysis["score"]) <= 1.0
    assert "thresholds" in analysis


# ---------------------------
# doclint (minimal)
# ---------------------------
def _write_minimal_doc(path: Path, title: str):
    fm = f"""---
title: {title}
version: v1.0
doc-id: ./{path.name}
status: final
spec: [S,H,Z, Δ, Φ, Ψ, I1–I4]
ownership: GLX-Core
links:
  - rel: glossary
    href: ./11_spec_glossary.md
---
"""
    path.write_text(fm + "\nOK\n", encoding="utf-8")


def test_doclint_minimal(monkeypatch, tmp_path):
    repo = _ensure_tmp_repo_layout(tmp_path)
    monkeypatch.chdir(repo)
    sys.path.insert(0, str(repo))

    dl = _import_glx_module("glx.tools.doclint")

    docs = Path("glitchlab/docs")
    docs.mkdir(parents=True, exist_ok=True)

    # wymagane pliki (zbliżone do oryginalnego zestawu)
    required = [
        "00_overview.md","10_architecture.md","11_spec_glossary.md","12_invariants.md",
        "13_delta_algebra.md","14_mosaic.md","20_bus.md","21_egdb.md","22_analytics.md",
        "30_sast_bridge.md","40_gui_app.md","41_pipelines.md","50_ci_ops.md","60_security.md",
        "70_observability.md","82_release_and_channels.md","92_playbooks.md","99_refactor_plan.md"
    ]
    for name in required:
        _write_minimal_doc(docs / name, f"Title {name}")

    # Happy path -> powinno zwrócić 0
    assert dl.main() == 0

    # Usuń jeden plik -> powinno zwrócić błąd (1)
    (docs / "92_playbooks.md").unlink()
    assert dl.main() == 1

    # Przywróć, wprowadź 'gui/' w treści -> powinno wykryć przestarzałą ścieżkę i zwrócić 1
    _write_minimal_doc(docs / "92_playbooks.md", "Title 92")
    (docs / "40_gui_app.md").write_text((docs / "40_gui_app.md").read_text().replace("OK", "gui/app.py"), encoding="utf-8")
    assert dl.main() == 1
