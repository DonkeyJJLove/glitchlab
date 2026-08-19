# tests/unit/test_invariants_scoring.py
import importlib
import json
import subprocess

import pytest

ic = importlib.import_module("glx.tools.invariants_check")


def _paths(monkeypatch, tmp_path):
    glx = tmp_path / ".glx"
    for name, leaf in (("SPEC_STATE", "spec_state.json"), ("DELTA_REPORT", "delta_report.json"), ("COMMIT_ANALYSIS", "commit_analysis.json")):
        monkeypatch.setattr(ic, name, glx / leaf)
    monkeypatch.setattr(ic, "GLX_DIR", glx)
    glx.mkdir()


@pytest.mark.parametrize("hist,psnr,ssim,expected_block", [
    ({"MODIFY_SIG": 0, "ΔIMPORT": 0}, 50.0, 0.9, False),
    ({"MODIFY_SIG": 100, "ΔIMPORT": 50}, 10.0, 0.0, True),
    # dodaj przypadki krańcowe...
])
def test_score_and_block_logic(hist, psnr, ssim, expected_block):
    score = ic.compute_score_from_report({"hist": hist, "psnr": psnr, "ssim": ssim})
    assert 0.0 <= score <= 1.0
    blocked = ic.classify_by_thresholds(score, thresholds={"alpha": 0.85, "beta": 0.92, "z": 0.99})
    assert bool(blocked) == expected_block


@pytest.mark.parametrize("payload", [None, "{bad", '{"range":"X","hist":{}}'])
def test_required_delta_evidence_fails_closed(monkeypatch, tmp_path, payload):
    _paths(monkeypatch, tmp_path)
    if payload is not None:
        ic.DELTA_REPORT.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(ic, "_git_diff_text", lambda _: "")
    assert ic.main(["--range", "A..B"]) == 1


def test_threshold_policy_is_explicit_and_invalid_spec_denied(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    ic.DELTA_REPORT.write_text(json.dumps({"range": "A..B", "hist": {}}), encoding="utf-8")
    monkeypatch.setattr(ic, "_git_diff_text", lambda _: "")
    analysis, code = ic.run("A..B")
    assert (code, analysis["threshold_source"], ic.SPEC_STATE.exists()) == (0, "builtin", False)
    ic.SPEC_STATE.write_text("{bad", encoding="utf-8")
    assert ic.main(["--range", "A..B"]) == 1


def test_git_and_persistence_fail_closed(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    ic.DELTA_REPORT.write_text(json.dumps({"range": "A..B", "hist": {}}), encoding="utf-8")
    err = subprocess.CalledProcessError(1, "git")
    monkeypatch.setattr(ic.subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(err))
    assert ic.main(["--range", "A..B"]) == 1
    monkeypatch.setattr(ic, "_git_diff_text", lambda _: "")
    monkeypatch.setattr(ic, "COMMIT_ANALYSIS", tmp_path)
    assert ic.main(["--range", "A..B"]) == 1
