#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glx.tools.invariants_check — bramka I1–I4 (score vs progi α/β/Z) • Python 3.9

Wejście:
- Zakres zmian z Gita: --range A..B (lub pojedynczy commit, np. HEAD).
- Artefakt delta: glitchlab/.glx/delta_report.json (histogram Δ-tokenów, opcj. psnr/ssim).

Wyjście:
- glitchlab/.glx/commit_analysis.json  → {"range","score","thresholds","violations":[...]}
- Kod procesu:
    0  → OK (score ≤ β)
    1  → BLOCK (> β)
    2  → HARD-BLOCK (> Z)

Zasada punktacji (heurystyczna, 0..1):
- 50%: ryzyko z Δ-tokenów (ważona średnia kategorii)
- 30%: intensywność zmian (churn: linie +/− w diff)
- 20%: opcjonalne wskaźniki jakości (psnr↓, ssim↓)

Progi (domyślne, gdy brak spec_state.json):
- α=0.85, β=0.92, Z=0.99 (repo-level). Plik baseline tworzy się automatycznie.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Ścieżki artefaktów
# ──────────────────────────────────────────────────────────────────────────────
GLX_DIR = Path("glitchlab/.glx")
SPEC_STATE = GLX_DIR / "spec_state.json"
DELTA_REPORT = GLX_DIR / "delta_report.json"
COMMIT_ANALYSIS = GLX_DIR / "commit_analysis.json"


# ──────────────────────────────────────────────────────────────────────────────
# Narzędzia Gita
# ──────────────────────────────────────────────────────────────────────────────
def _git_diff_text(commit_range: str) -> str:
    """Zwraca pełny diff (unified=0) dla zakresu; przy pojedynczym commicie używa ^!."""
    try:
        if ".." in commit_range:
            args = ["git", "diff", "--unified=0", commit_range]
        else:
            args = ["git", "diff", "--unified=0", f"{commit_range}^!"]
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return ""


def _churn_from_diff(diff_text: str) -> Tuple[int, int]:
    """
    Zlicza linie dodane/uszunięte w stylu unified, ignorując nagłówki '+++', '---'.
    """
    added = 0
    deleted = 0
    for line in diff_text.splitlines():
        if not line:
            continue
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            deleted += 1
    return added, deleted


# ──────────────────────────────────────────────────────────────────────────────
# Spec / progi
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULT_THRESHOLDS = {"alpha": 0.85, "beta": 0.92, "z": 0.99}


def _load_thresholds() -> Dict[str, float]:
    """
    Ładuje progi z spec_state.json, jeśli istnieją.
    Struktura oczekiwana:
        {"thresholds": {"repo": {"alpha":..,"beta":..,"z":..}}, ...}
    W przeciwnym razie tworzy baseline i zwraca domyślne wartości.
    """
    GLX_DIR.mkdir(parents=True, exist_ok=True)
    if SPEC_STATE.exists():
        try:
            obj = json.loads(SPEC_STATE.read_text(encoding="utf-8"))
            t = (obj.get("thresholds") or {}).get("repo") or {}
            alpha = float(t.get("alpha", _DEFAULT_THRESHOLDS["alpha"]))
            beta = float(t.get("beta", _DEFAULT_THRESHOLDS["beta"]))
            z = float(t.get("z", _DEFAULT_THRESHOLDS["z"]))
            return {"alpha": alpha, "beta": beta, "z": z}
        except Exception:
            pass

    # baseline
    baseline = {
        "ts": time.time(),
        "note": "baseline (auto)",
        "thresholds": {"repo": dict(_DEFAULT_THRESHOLDS)},
    }
    try:
        SPEC_STATE.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return dict(_DEFAULT_THRESHOLDS)


# ──────────────────────────────────────────────────────────────────────────────
# Delta / punktacja
# ──────────────────────────────────────────────────────────────────────────────
_WEIGHTS = {
    "MODIFY_SIG": 3.0,
    "EXTRACT_FN": 2.5,
    "MOVE_BLOCK": 2.0,
    "RENAME": 1.5,
    "ΔIMPORT": 1.8,
    "ΔTYPE_HINTS": 0.8,
    "ΔTESTS": 2.2,
    "ADD_FN": 1.0,
    "DEL_FN": 2.0,
    "ΔCC": 2.0,
}


def _load_delta_report() -> Dict[str, Any]:
    if not DELTA_REPORT.exists():
        return {}
    try:
        return json.loads(DELTA_REPORT.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _score_from(rep: Dict[str, Any], diff_text: str) -> float:
    """
    Skala [0,1]. Składniki:
      - token_risk  (0..1) * 0.5
      - churn_ratio (0..1) * 0.3
      - quality     (0..1) * 0.2
    """
    # 1) Ryzyko z Δ-tokenów
    hist = rep.get("hist") or {}
    total = sum(int(v) for v in hist.values()) or 1
    token_risk = 0.0
    for k, v in hist.items():
        w = _WEIGHTS.get(k, 1.0)
        token_risk += w * (float(v) / float(total))
    token_risk = min(1.0, token_risk / 2.5)  # normalizacja
    part_tokens = 0.5 * token_risk

    # 2) Churn z diff
    add, dele = _churn_from_diff(diff_text)
    churn = add + dele
    churn_ratio = min(1.0, churn / 200.0)
    part_churn = 0.3 * churn_ratio

    # 3) Wskaźniki jakości (opcjonalne)
    psnr = rep.get("psnr")
    ssim = rep.get("ssim")
    part_quality = 0.0
    if isinstance(psnr, (int, float)):
        part_quality += 0.1 * (1.0 if psnr < 20 else 0.0)  # niski psnr → podnosi ryzyko
    if isinstance(ssim, (int, float)):
        # im mniejsze ssim, tym wyższe ryzyko
        s = max(0.0, min(1.0, float(ssim)))
        part_quality += 0.1 * (1.0 - s)

    score = part_tokens + part_churn + part_quality
    return max(0.0, min(1.0, score))


def compute_score_from_report(report: dict) -> float:
    """
    Compute normalized risk score in [0, 1] from delta report.
    Expected keys:
    - hist: dict token -> count
    - psnr: float, optional
    - ssim: float, optional
    """
    hist = report.get("hist") or {}
    modify = float(hist.get("MODIFY_SIG", 0) or hist.get("CHANGE_SIG", 0) or 0)
    imp = float(hist.get("ΔIMPORT", 0) or hist.get("DELTA_IMPORT", 0) or 0)
    token_pressure = min(1.0, (modify + imp) / 100.0)

    psnr_raw = report.get("psnr", 50.0)
    ssim_raw = report.get("ssim", 1.0)
    psnr = float(50.0 if psnr_raw is None else psnr_raw)
    ssim = float(1.0 if ssim_raw is None else ssim_raw)

    psnr_pressure = 0.0 if psnr >= 40.0 else (1.0 if psnr <= 20.0 else (40.0 - psnr) / 20.0)
    ssim_pressure = min(1.0, max(0.0, 1.0 - ssim))

    score = 0.6 * token_pressure + 0.2 * psnr_pressure + 0.2 * ssim_pressure
    return max(0.0, min(1.0, float(score)))


def classify_by_thresholds(score: float, thresholds: dict | None = None):
    """
    Return bool block decision for compatibility with tests.
    Uses z threshold if present, otherwise beta/alpha fallback.
    """
    thresholds = thresholds or {}
    z = float(thresholds.get("z", thresholds.get("beta", thresholds.get("alpha", 0.85))))
    return float(score) >= z


# ──────────────────────────────────────────────────────────────────────────────
# Główna procedura
# ──────────────────────────────────────────────────────────────────────────────
def run(commit_range: str) -> Dict[str, Any]:
    thresholds = _load_thresholds()
    diff_text = _git_diff_text(commit_range)
    rep = _load_delta_report()
    score = _score_from(rep, diff_text)

    analysis = {
        "range": commit_range,
        "score": score,
        "thresholds": thresholds,
        "violations": [],
    }

    a = float(thresholds["alpha"])
    b = float(thresholds["beta"])
    z = float(thresholds["z"])

    # Gating — uwzględniamy tylko progi β/Z do kodów wyjścia
    exit_code = 0
    if score > b:
        analysis["violations"].append(
            {"invariant": "I*", "severity": "block", "details": "score > beta"}
        )
        exit_code = 1
    if score > z:
        analysis["violations"].append(
            {"invariant": "I*", "severity": "hard-block", "details": "score > Z"}
        )
        exit_code = 2

    # Zapis artefaktu
    try:
        GLX_DIR.mkdir(parents=True, exist_ok=True)
        COMMIT_ANALYSIS.write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        print(f"[invariants_check] cannot write {COMMIT_ANALYSIS}: {e}", file=sys.stderr)

    # Log JSON (przydatne w CI)
    print(json.dumps(analysis, ensure_ascii=False))
    return analysis, exit_code


def main(argv: list = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--range",
        default="HEAD~1..HEAD",
        help="Zakres commitów do oceny (A..B) lub pojedynczy commit (np. HEAD)",
    )
    args = ap.parse_args(argv)

    try:
        _, code = run(args.range)
        return int(code)
    except Exception as e:
        print(f"[invariants_check] ERROR: {e}", file=sys.stderr)
        # Ostrożność: jeśli nie można ocenić, w lokalnym środowisku zwróć błąd,
        # aby użytkownik miał sygnał, że bramka nie zadziałała poprawnie.
        return 1


if __name__ == "__main__":
    sys.exit(main())
