#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python: 3.9  (ten test uruchamia się tylko na 3.9.x)
# Ścieżka: glitchlab/tests/test_post_commit_e2e.py

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# ŚCIEŻKI BAZOWE
# ──────────────────────────────────────────────────────────────────────────────
THIS = Path(__file__).resolve()
PKG_ROOT = THIS.parents[1]         # .../glitchlab
PROJECT_ROOT = PKG_ROOT.parent     # .../<repo_root_katalog_projektu>

IS_PY39 = (3, 9) <= sys.version_info[:2] < (3, 10)

# ──────────────────────────────────────────────────────────────────────────────
# .ENV — ładujemy wcześnie (dotenv → fallback)
# ──────────────────────────────────────────────────────────────────────────────
def _simple_parse_env(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    except Exception:
        pass
    return env


def _load_dotenv_into_env(dotenv_dir: Path) -> None:
    # próbujemy .env w PROJECT_ROOT oraz katalog wyżej (tak jak w starszych setupach)
    candidates = [dotenv_dir / ".env", dotenv_dir.parent / ".env"]
    dot = next((p for p in candidates if p.is_file()), None)
    if not dot:
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dot, override=False)
        return
    except Exception:
        pass
    parsed = _simple_parse_env(dot)
    for k, v in parsed.items():
        os.environ.setdefault(k, v)


def _norm_path_from_root(p: str, root: Path) -> Path:
    if not p:
        return root
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp).resolve()


# Załaduj .env zanim odczytamy wartości
_load_dotenv_into_env(PROJECT_ROOT)

# Standardowe klucze .env (fallbacki bezpieczne)
GLX_ROOT = _norm_path_from_root(os.getenv("GLX_ROOT", str(PROJECT_ROOT)), PROJECT_ROOT)
GLX_PKG = os.getenv("GLX_PKG", "glitchlab")

GLX_OUT = _norm_path_from_root(
    os.getenv("GLX_OUT", str(PKG_ROOT / "analysis" / "last")),
    GLX_ROOT
)
GLX_AUTONOMY_OUT = _norm_path_from_root(
    os.getenv("GLX_AUTONOMY_OUT", str(PKG_ROOT / "analysis" / "last" / "autonomy")),
    GLX_ROOT
)
GLX_POLICY = _norm_path_from_root(
    os.getenv("GLX_POLICY", str(PKG_ROOT / "analysis" / "policy.json")),
    GLX_ROOT
)

# Parametry mozaiki (opcjonalne)
GLX_MOSAIC = os.getenv("GLX_MOSAIC", "grid")
GLX_ROWS = os.getenv("GLX_ROWS", "6")
GLX_COLS = os.getenv("GLX_COLS", "6")
GLX_EDGE_THR = os.getenv("GLX_EDGE_THR", "0.55")
GLX_KAPPA = os.getenv("GLX_KAPPA", "0.35")
GLX_PHI = os.getenv("GLX_PHI", "balanced")
GLX_DELTA = os.getenv("GLX_DELTA", "0.25")
GLX_PYTHONPATH_APPEND = os.getenv("GLX_PYTHONPATH_APPEND", "")

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


def _git_top_any(candidates: List[Path]) -> Optional[Path]:
    """Zwróć pierwszy katalog, dla którego git-top istnieje."""
    for c in candidates:
        top = _git_top(c)
        if top:
            return top
    return None


# Kolejność prób: GLX_ROOT, PROJECT_ROOT, GLX_ROOT/GLX_PKG, PKG_ROOT
GIT_ROOT = _git_top_any([
    GLX_ROOT,
    PROJECT_ROOT,
    (GLX_ROOT / GLX_PKG),
    PKG_ROOT,
]) or (GLX_ROOT / GLX_PKG)

def _have_git_worktree() -> bool:
    return shutil.which("git") is not None and _git_top(GIT_ROOT) is not None


def _rev(ref: str) -> str:
    p = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=str(GIT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else ref

# ──────────────────────────────────────────────────────────────────────────────
# ENV/CWD do subprocessów — respektujemy .env
# ──────────────────────────────────────────────────────────────────────────────
def _subenv() -> dict:
    """
    PYTHONPATH układamy deterministycznie:
      [GLX_ROOT/GLX_PKG, GLX_ROOT, (opcjonalnie) GIT_ROOT, appendy z .env, istniejący PYTHONPATH]
    """
    env = os.environ.copy()
    path_parts: List[str] = []

    # 1) Najpierw katalog pakietu, potem root projektu
    pkg_dir = (GLX_ROOT / GLX_PKG)
    if pkg_dir.exists():
        path_parts.append(str(pkg_dir))
    path_parts.append(str(GLX_ROOT))

    # 2) Jeżeli GIT_ROOT różni się od GLX_ROOT — dodaj (na końcu rdzenia)
    if GIT_ROOT and GIT_ROOT != GLX_ROOT:
        path_parts.append(str(GIT_ROOT))

    # 3) Appendy z .env (rozdzielone os.pathsep)
    if GLX_PYTHONPATH_APPEND:
        for piece in GLX_PYTHONPATH_APPEND.split(os.pathsep):
            piece = piece.strip()
            if piece:
                path_parts.append(piece)

    # 4) Istniejący PYTHONPATH
    if env.get("PYTHONPATH"):
        path_parts.append(env["PYTHONPATH"])

    # Deduplikacja z zachowaniem kolejności
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(path_parts))
    return env


def _assert_can_import_glitchlab(cwd: Path, env: dict):
    """
    Sprawdza import 'glitchlab' z PYTHONPATH zbudowanego wg .env.
    """
    code = (
        "import sys\n"
        "import json\n"
        "res = {'ok': False, 'err': None, 'path': sys.path}\n"
        "try:\n"
        "    import glitchlab\n"
        "    res['ok'] = True\n"
        "    res['file'] = getattr(glitchlab, '__file__', None)\n"
        "except Exception as e:\n"
        "    res['err'] = repr(e)\n"
        "print(json.dumps(res))\n"
    )
    p = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        data = json.loads(p.stdout.strip() or "{}")
    except Exception:
        data = {"ok": False, "err": "malformed probe stdout", "stdout": p.stdout, "stderr": p.stderr}

    if not data.get("ok"):
        pytest.fail(
            "Nie można zaimportować 'glitchlab' z ustawieniami .env.\n"
            f"CWD={cwd}\nPYTHONPATH={env.get('PYTHONPATH', '')}\n"
            f"Probe STDOUT={p.stdout!r}\nProbe STDERR={p.stderr!r}"
        )

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


def _try_run_module(mod: str, args: List[str]):
    """
    Uruchom `python -m <mod> <args>` z **CWD=GIT_ROOT** (ważne dla komend git),
    env=_subenv(). Zwraca: (rc, stdout, stderr, used_cwd, used_env)
    """
    env = _subenv()
    # Import testujemy względem GLX_ROOT (gdzie leży kod pakietu),
    # ale sam proces modułu uruchamiamy z CWD=GIT_ROOT (żeby git widział repo).
    _assert_can_import_glitchlab(GLX_ROOT, env)
    cmd = [sys.executable, "-m", mod] + args
    p = subprocess.run(cmd, cwd=str(GIT_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout or "", p.stderr or "", GIT_ROOT, env

# ──────────────────────────────────────────────────────────────────────────────
# TEST 1: Autonomy Gateway build (z .env → GLX_AUTONOMY_OUT)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not IS_PY39, reason="tests require Python 3.9.x")
@pytest.mark.skipif(not _have_git_worktree(), reason="no git worktree detected")
def test_autonomy_gateway_build():
    out_dir = GLX_AUTONOMY_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    env = _subenv()
    # Tu import może być z GLX_ROOT, nie wymaga repo git
    _assert_can_import_glitchlab(GLX_ROOT, env)

    cmd = [
        sys.executable, "-m",
        f"{GLX_PKG}.analysis.autonomy.gateway",
        "build", "--out", str(out_dir),
    ]
    # Gateway nie potrzebuje git — CWD=GLX_ROOT jest OK
    r = subprocess.run(cmd, cwd=str(GLX_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert r.returncode == 0, (
        "gateway build failed\n"
        f"RC={r.returncode}\nCWD={GLX_ROOT}\nPYTHONPATH={env.get('PYTHONPATH', '')}\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    assert _existing(out_dir / "pack.json"), "pack.json not created"
    assert _existing(out_dir / "prompt.json"), "prompt.json not created"
    assert _existing(out_dir / "pack.md"), "pack.md not created"

# ──────────────────────────────────────────────────────────────────────────────
# TEST 2: Mosaic from-git-dump (obsługa 2 ścieżek modułu, ścieżki z .env)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not IS_PY39, reason="tests require Python 3.9.x")
@pytest.mark.skipif(not _have_git_worktree(), reason="no git worktree detected")
def test_hybrid_mosaic_from_git_dump_and_optional_html():
    # BASE..HEAD (origin/master → origin/main → HEAD~1)
    base = ""
    p = subprocess.run(["git", "merge-base", "HEAD", "origin/master"], cwd=str(GIT_ROOT),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode == 0 and p.stdout.strip():
        base = p.stdout.strip()
    if not base:
        p2 = subprocess.run(["git", "merge-base", "HEAD", "origin/main"], cwd=str(GIT_ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p2.returncode == 0 and p2.stdout.strip():
            base = p2.stdout.strip()
    if not base:
        base = _rev("HEAD~1")
    head = _rev("HEAD")

    out_dir = GLX_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    module_candidates = [
        f"{GLX_PKG}.gui.mosaic.hybrid_ast_mosaic",
        f"{GLX_PKG}.mosaic.hybrid_ast_mosaic",
    ]

    last = (1, "", "", GIT_ROOT, os.environ.copy())
    stdout_json = None

    for mod in module_candidates:
        args = [
            "--mosaic", GLX_MOSAIC,
            "--rows", GLX_ROWS,
            "--cols", GLX_COLS,
            "--edge-thr", GLX_EDGE_THR,
            "--kappa-ab", GLX_KAPPA,
            "--phi", GLX_PHI,
            "from-git-dump",
            "--base", base, "--head", head,
            "--delta", GLX_DELTA,
            "--out", str(out_dir),
        ]
        if GLX_PHI.lower() == "policy" and GLX_POLICY.exists():
            args += ["--policy-file", str(GLX_POLICY)]

        rc, so, se, used_cwd, used_env = _try_run_module(mod, args)
        last = (rc, so, se, used_cwd, used_env)

        try:
            if so.strip().startswith("{"):
                stdout_json = json.loads(so.strip())
        except Exception:
            stdout_json = None

        if rc == 0:
            break

    rc, so, se, used_cwd, used_env = last
    if rc != 0:
        pytest.fail(
            "from-git-dump failed (both module paths)\n"
            f"RC={rc}\nCWD={used_cwd}\nPYTHONPATH={used_env.get('PYTHONPATH', '')}\n"
            f"STDOUT:\n{so or '(empty)'}\nSTDERR:\n{se or '(empty)'}"
        )

    # 1) Jeśli CLI zwróciło artefakty — zweryfikuj istnienie
    if isinstance(stdout_json, dict) and isinstance(stdout_json.get("artifacts"), dict):
        missing = []
        for _, pth in stdout_json["artifacts"].items():
            if not pth:
                continue
            pth = Path(pth)
            if not pth.is_absolute():
                pth = (GLX_ROOT / pth).resolve()
            if not pth.exists():
                missing.append(str(pth))
        assert not missing, "Some artifacts listed by CLI do not exist:\n" + "\n".join(missing)
    else:
        # 2) Fallback: sensowne pliki w analysis/last/**
        found = list(out_dir.rglob("report.json")) \
                + list(out_dir.rglob("mosaic_map.json")) \
                + list(out_dir.rglob("summary.md"))
        if not found:
            created = [str(p.relative_to(PKG_ROOT)) for p in _list_files(out_dir)]
            pytest.fail(
                "No mosaic artifacts found under analysis/last.\n"
                f"CWD={used_cwd}\nPYTHONPATH={used_env.get('PYTHONPATH', '')}\n"
                "STDOUT:\n" + (so or "(empty)") + "\n"
                "STDERR:\n" + (se or "(empty)") + "\n"
                "Files under analysis/last:\n" + "\n".join(f" - {c}" for c in sorted(created))
            )

    # (opcjonalnie) generator HTML
    html_script = PKG_ROOT / "scripts" / "report_html.py"
    if html_script.exists():
        report_path = out_dir / "report.json"
        if not report_path.exists():
            cands = sorted(out_dir.rglob("report.json"))
            report_path = cands[0] if cands else None
        assert report_path and report_path.exists(), "Brak report.json dla report_html.py"

        env = _subenv()
        out_html = report_path.with_suffix(".html")
        r2 = subprocess.run(
            [sys.executable, str(html_script),
             "--report", str(report_path),
             "--out", str(out_html),
             "--title", "Mosaic Report"],
            cwd=str(GLX_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        assert r2.returncode == 0, (
            f"report_html failed rc={r2.returncode}\nSTDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}"
        )
