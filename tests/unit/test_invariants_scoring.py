# tests/unit/test_invariants_scoring.py
import importlib
import pytest

ic = importlib.import_module("glx.tools.invariants_check")


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
