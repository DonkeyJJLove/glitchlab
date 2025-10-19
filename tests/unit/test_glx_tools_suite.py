# tests/unit/test_glx_tools_suite.py
from __future__ import annotations

"""
Zestaw testów jednostkowych / integracyjnych dla modułów w glx.tools:
- delta_fingerprint: podstawowe wykrycia (dodany import, zmiana sygnatury)
- invariants_check: logika liczenia score oraz klasyfikacji wg progów
- doclint: wykrywanie braków i zabronionych odwołań (gui/)

Testy są odporne na to, czy pakiet znajduje się jako `glx` czy `glitchlab.glx`.
Stosują monkeypatch do stubowania "git" i sys.argv.
"""
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, Any

import pytest


def _ensure_tmp_repo_layout(tmp_path: Path) -> Path:
    (tmp_path / "glitchlab" / ".glx").mkdir(parents=True, exist_ok=True)
    (tmp_path / "glitchlab" / "docs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _import_glx_module(name: str):
    """Próbuje zaimportować modulę 'glx.tools.x' lub alternatywnie 'glitchlab.glx.tools.x'."""
    try:
        return importlib.import_module(name)
    except Exception:
        alt = name.replace("glx.", "glitchlab.glx.")
        return importlib.import_module(alt)


# -------------------------
# delta_fingerprint tests
# -------------------------
def test_delta_fingerprint_detects_import_and_sig_change(monkeypatch, tmp_path):
    repo = _ensure_tmp_repo_layout(tmp_path)
    monkeypatch.chdir(repo)

    sys.path.insert(0, str(repo))
    df = _import_glx_module("glx.tools.delta_fingerprint")

    # proste "stare" i "nowe" źródło: dodano import oraz zmieniono sygnaturę
    old_py = "import os\n\ndef foo(a, b):\n    return a + b\n"
    new_py = "import os, sys\n\ndef foo(a, b, c):\n    return a + b + c\n"

    # stuby git-owe (nie poleganie na git w testach)
    monkeypatch.setattr(df, "_git_changed_files", lambda rng: ["glitchlab/core/foo.py"], raising=False)
    monkeypatch.setattr(
        df,
        "_git_show",
        lambda path, rev: old_py if (isinstance(rev, str) and (rev.endswith("^") or rev == "A")) else new_py,
        raising=False,
    )
    monkeypatch.setattr(df, "_get_parent_sha", lambda rev: "A", raising=False)

    # zabezpiecz argv tak aby nie wciągnąć argumentów pytest
    argv_orig = sys.argv[:]
    sys.argv = ["delta_fingerprint"]
    try:
        ret = df.main()
    finally:
        sys.argv = argv_orig

    assert isinstance(ret, int)

    out = Path("glitchlab/.glx/delta_report.json")
    assert out.exists(), "delta_report.json powinien zostać utworzony"

    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    hist: Dict[str, int] = data.get("hist") or {}

    # akceptujemy różne nazewnictwo kluczy: dopuszczalne, by implementacja używała ΔIMPORT lub DELTA_IMPORT itp.
    found_import = any("IMPORT" in k.upper() for k in hist.keys())
    found_sig = any("SIG" in k.upper() or "SIGN" in k.upper() for k in hist.keys())

    assert found_import, f"hist: oczekiwano klucza związane z importami w {hist}"
    assert found_sig, f"hist: oczekiwano klucza związane ze zmianą sygnatury w {hist}"
    assert "hash" in data and isinstance(data["hash"], str) and len(data["hash"]) >= 4


# -------------------------
# invariants_check tests
# -------------------------
@pytest.mark.parametrize(
    "hist,psnr,ssim,expected_block",
    [
        ({"MODIFY_SIG": 0, "ΔIMPORT": 0}, 50.0, 0.9, False),
        ({"MODIFY_SIG": 100, "ΔIMPORT": 50}, 10.0, 0.0, True),
        ({"MODIFY_SIG": 10, "ΔIMPORT": 1}, 30.0, 0.5, False),
    ],
)
def test_invariants_scoring_and_thresholds(hist: Dict[str, int], psnr: float, ssim: float, expected_block: bool):
    # import modułu (funkcje pomocnicze powinny istnieć: compute_score_from_report, classify_by_thresholds)
    ic = _import_glx_module("glx.tools.invariants_check")

    # jeśli moduł dostarcza funkcję compute_score_from_report — użyjemy jej; w przeciwnym razie spróbujemy main()
    if hasattr(ic, "compute_score_from_report") and hasattr(ic, "classify_by_thresholds"):
        score = ic.compute_score_from_report({"hist": hist, "psnr": psnr, "ssim": ssim})
        assert isinstance(score, (float, int)) and 0.0 <= float(score) <= 1.0
        blocked = ic.classify_by_thresholds(score, thresholds={"alpha": 0.85, "beta": 0.92, "z": 0.99})
        # klasyfikator może zwracać boole lub int kodu blokady — normalizujemy do boola
        blocked_bool = bool(blocked)
        assert blocked_bool == expected_block
    else:
        # fallback: stwórz delta_report i wywołaj main() — test sprawdza jedynie, że main wykonuje się i zapisuje artefakt
        tmp = Path("glitchlab/.glx")
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "delta_report.json").write_text(json.dumps({"hist": hist, "psnr": psnr, "ssim": ssim}), encoding="utf-8")
        argv_orig = sys.argv[:]
        sys.argv = ["invariants_check", "--range", "A..B"]
        try:
            ret = ic.main()
        finally:
            sys.argv = argv_orig
        assert isinstance(ret, int)
        analysis = json.loads((tmp / "commit_analysis.json").read_text(encoding="utf-8"))
        assert "score" in analysis and 0.0 <= float(analysis["score"]) <= 1.0


# -------------------------
# doclint tests
# -------------------------
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


def test_doclint_missing_and_gui_reference(monkeypatch, tmp_path):
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
        _write_minimal_doc(docs / name, f"Title {name}")

    # happy path
    assert dl.main() == 0

    # usuń plik -> powinno zwrócić błąd
    (docs / "92_playbooks.md").unlink()
    assert dl.main() == 1

    # przywróć i wprowadź zakazaną referencję GUI -> powinno zwrócić 1
    _write_minimal_doc(docs / "92_playbooks.md", "Title 92")
    (docs / "40_gui_app.md").write_text((docs / "40_gui_app.md").read_text().replace("OK", "gui/app.py"), encoding="utf-8")
    assert dl.main() == 1
