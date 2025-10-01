# -*- coding: utf-8 -*-
# Python: 3.9  (ten test uruchamia się tylko na 3.9.x)

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# ŚCIEŻKI (ten plik leży w: glitchlab/tests/test_post_commit_e2e.py)
# ──────────────────────────────────────────────────────────────────────────────
THIS = Path(__file__).resolve()
PKG_ROOT = THIS.parents[2]  # .../glitchlab
PROJECT_ROOT = PKG_ROOT.parent  # katalog-rodzic pakietu (to MA być na sys.path)

IS_PY39 = (3, 9) <= sys.version_info[:2] < (3, 10)


# ──────────────────────────────────────────────────────────────────────────────
# GIT UTILS
# ──────────────────────────────────────────────────────────────────────────────
def _git_top(cwd: Path) -> Optional[Path]:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if p.returncode == 0 and p.stdout.strip():
            return Path(p.stdout.strip())
    except Exception:
        pass
    return None


GIT_ROOT = _git_top(PROJECT_ROOT) or _git_top(PROJECT_ROOT.parent) or PROJECT_ROOT


def _have_git_worktree() -> bool:
    return shutil.which("git") is not None and GIT_ROOT is not None


def _rev(ref: str) -> str:
    p = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=str(GIT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else ref


# ──────────────────────────────────────────────────────────────────────────────
# ENV/CWD do subprocessów
# ──────────────────────────────────────────────────────────────────────────────
def _subenv() -> dict:
    """Środowisko dla subprocessów z poprawnym PYTHONPATH."""
    env = os.environ.copy()
    parts = [str(PROJECT_ROOT)]
    if GIT_ROOT and GIT_ROOT != PROJECT_ROOT:
        parts.append(str(GIT_ROOT))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


# ──────────────────────────────────────────────────────────────────────────────
# POMOCNICZE
# ──────────────────────────────────────────────────────────────────────────────
def _existing(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def _list_files(base: Path) -> List[Path]:
    if not base.exists():
        return []
    return [p for p in base.rglob("*") if p.is_file()]


# ──────────────────────────────────────────────────────────────────────────────
# TEST 1: Autonomy Gateway build
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not IS_PY39, reason="tests require Python 3.9.x")
@pytest.mark.skipif(not _have_git_worktree(), reason="no git worktree detected")
def test_autonomy_gateway_build():
    out_dir = PKG_ROOT / "analysis" / "last" / "autonomy"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m",
        "glitchlab.analysis.autonomy.gateway",
        "build", "--out", str(out_dir),
    ]
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=_subenv())
    assert r.returncode == 0, f"gateway build failed with rc={r.returncode}"
    assert _existing(out_dir / "pack.json"), "pack.json not created"
    assert _existing(out_dir / "prompt.json"), "prompt.json not created"
    assert _existing(out_dir / "pack.md"), "pack.md not created"


# ──────────────────────────────────────────────────────────────────────────────
# TEST 2: Mosaic from-git-dump (obsługuje oba import-pathy modułu)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not IS_PY39, reason="tests require Python 3.9.x")
@pytest.mark.skipif(not _have_git_worktree(), reason="no git worktree detected")
def test_hybrid_mosaic_from_git_dump_and_optional_html():
    # wyznacz BASE..HEAD (origin/master → origin/main → HEAD~1)
    p = subprocess.run(
        ["git", "merge-base", "HEAD", "origin/master"],
        cwd=str(GIT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    base = p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else ""
    if not base:
        p2 = subprocess.run(
            ["git", "merge-base", "HEAD", "origin/main"],
            cwd=str(GIT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        base = p2.stdout.strip() if p2.returncode == 0 and p2.stdout.strip() else ""
    if not base:
        base = _rev("HEAD~1")
    head = _rev("HEAD")

    out_dir = PKG_ROOT / "analysis" / "last"
    out_dir.mkdir(parents=True, exist_ok=True)

    module_candidates = [
        "glitchlab.gui.mosaic.hybrid_ast_mosaic",
        "glitchlab.mosaic.hybrid_ast_mosaic",
    ]

    rc = 1
    last_out = ""
    last_err = ""
    stdout_json = None

    for mod in module_candidates:
        cmd = [
            sys.executable, "-m", mod,
            "--mosaic", os.getenv("GLX_MOSAIC", "grid"),
            "--rows", os.getenv("GLX_ROWS", "6"),
            "--cols", os.getenv("GLX_COLS", "6"),
            "--edge-thr", os.getenv("GLX_EDGE_THR", "0.55"),
            "--kappa-ab", os.getenv("GLX_KAPPA", "0.35"),
            "--phi", os.getenv("GLX_PHI", "balanced"),
            "from-git-dump",
            "--base", base, "--head", head,
            "--delta", os.getenv("GLX_DELTA", "0.25"),
            "--out", str(out_dir),
        ]
        pol_rel = os.getenv("GLX_POLICY", "analysis/policy.json")
        pol_path = (PKG_ROOT / pol_rel).resolve()
        if os.getenv("GLX_PHI", "balanced").lower() == "policy" and pol_path.exists():
            cmd += ["--policy-file", str(pol_path)]

        r = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), env=_subenv(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        rc = r.returncode
        last_out, last_err = r.stdout or "", r.stderr or ""

        # spróbuj sparsować stdout jako JSON {"ok":..., "artifacts": {...}}
        try:
            stdout_json = json.loads(last_out.strip()) if last_out.strip().startswith("{") else None
        except Exception:
            stdout_json = None

        if rc == 0:
            break

    assert rc == 0, f"from-git-dump failed (both module paths)\nSTDOUT:\n{last_out}\nSTDERR:\n{last_err}"

    # 1) Jeśli CLI zwrócił listę artefaktów – sprawdź, że istnieją
    if isinstance(stdout_json, dict) and isinstance(stdout_json.get("artifacts"), dict):
        missing = []
        for _, pth in stdout_json["artifacts"].items():
            if not pth:
                continue
            p = Path(pth)
            if not p.is_absolute():
                p = (PROJECT_ROOT / p).resolve()
            if not p.exists():
                missing.append(str(p))
        assert not missing, f"Some artifacts listed by CLI do not exist:\n" + "\n".join(missing)
    else:
        # 2) Fallback: szukamy sensownych plików w analysis/last/**
        found = list(out_dir.rglob("report.json")) \
                + list(out_dir.rglob("mosaic_map.json")) \
                + list(out_dir.rglob("summary.md"))
        if not found:
            created = [str(p.relative_to(PKG_ROOT)) for p in _list_files(out_dir)]
            msg = "No mosaic artifacts found under analysis/last.\n" \
                  "STDOUT:\n" + last_out + "\nSTDERR:\n" + last_err + "\n" \
                                                                      "Files under analysis/last:\n" + "\n".join(
                f" - {c}" for c in sorted(created))
            pytest.fail(msg)

    # (opcjonalnie) jeśli istnieje generator HTML – zbuduj
    html_script = PKG_ROOT / "scripts" / "report_html.py"
    if html_script.exists():
        r2 = subprocess.run([sys.executable, str(html_script)], cwd=str(PROJECT_ROOT), env=_subenv())
        assert r2.returncode == 0, f"report_html failed with rc={r2.returncode}"
