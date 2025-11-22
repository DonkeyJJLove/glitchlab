#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.githooks/_common.py â€” wspÃ³lne helpery dla hookÃ³w GLX (GitLab/GlitchLab)

Wymagania:
- Python 3.9+
- Tylko stdlib (brak zewnÄ™trznych zaleÅ¼noÅ›ci)

Funkcje:
- git_root()                â†’ wykrycie katalogu gÅ‚Ã³wnego repo
- log()/warn()/fail()       â†’ spÃ³jne logowanie
- staged_paths()            â†’ lista plikÃ³w w indeksie (staged)
- changed_files()           â†’ lista plikÃ³w w zakresie commitÃ³w
- glx_dir()/ensure_glx_dir()
- write_json_atomic()/append_jsonline()
- run_cmd()/run_glx_module()/py_env()
- build_commit_snippet()/write_commit_snippet()
- reject_big_files()/check_python_compiles()
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Logowanie
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def log(msg: str) -> None:
    print(f"[githooks] {msg}", file=sys.stdout, flush=True)


def warn(msg: str) -> None:
    print(f"[githooks][WARN] {msg}", file=sys.stderr, flush=True)


def fail(msg: str, code: int = 1) -> None:
    print(f"[githooks][FAIL] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Repo / Git utils
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def git_root(start: Path | str) -> Path:
    """
    Zwraca katalog gÅ‚Ã³wny repo (git rev-parse --show-toplevel), z bezpiecznym fallbackiem.
    """
    here = Path(start).resolve()
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(here),
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip()).resolve()
    except Exception:
        pass
    # Fallback: standardowa lokalizacja .githooks/ w root â†’ dwa poziomy w gÃ³rÄ™
    parents = list(here.parents)
    return (parents[2] if len(parents) >= 3 else here).resolve()


def staged_paths(root: Path) -> List[Path]:
    """
    Zwraca listÄ™ staged plikÃ³w (dodane/zmienione/renamed) w repo.
    """
    r = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(root),
        check=False,
    )
    paths: List[Path] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        p = (root / line).resolve()
        # ignorujemy usuniÄ™te/przeniesione poza repo
        if p.exists():
            paths.append(p)
    return paths


def changed_files(root: Path, commit_range: str) -> List[str]:
    """
    Zwraca listÄ™ plikÃ³w w zakresie commitÃ³w (A..B lub pojedynczy).
    """
    args = ["git", "diff", "--name-only"]
    if ".." in commit_range:
        args.append(commit_range)
    else:
        args += [f"{commit_range}^!", "--"]
    r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(root))
    return [p for p in r.stdout.splitlines() if p.strip()]


def rev_parse(root: Path, ref: str) -> Optional[str]:
    r = subprocess.run(["git", "rev-parse", ref], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(root))
    return r.stdout.strip() if r.returncode == 0 else None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# IO / Artefakty .glx/*
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def glx_dir(root: Path) -> Path:
    """Lokalizacja katalogu artefaktÃ³w GLX."""
    return (root / "glitchlab" / ".glx").resolve()


def ensure_glx_dir(root: Path) -> Path:
    d = glx_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json_atomic(path: Path, obj: Dict, *, indent: int = 2) -> None:
    """Zapis JSON w sposÃ³b atomowy (tmp â†’ rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        json.dump(obj, tmp, ensure_ascii=False, indent=indent)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def append_jsonline(path: Path, line: Dict) -> None:
    """Dopisuje liniÄ™ JSON (JSON Lines) do pliku."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Uruchamianie procesÃ³w / moduÅ‚Ã³w
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_cmd(root: Path, cmd: List[str], *, env: Optional[Dict[str, str]] = None) -> int:
    log(" ".join(cmd))
    return subprocess.run(cmd, cwd=str(root), env=env, check=False).returncode


def py_env(glx_root: Path, repo_root: Path) -> Dict[str, str]:
    """Minimalny ENV dla narzÄ™dzi GLX (moÅ¼e byÄ‡ rozszerzany)."""
    env = os.environ.copy()
    env.setdefault("GLX_ROOT", str(glx_root))
    env.setdefault("GIT_ROOT", str(repo_root))
    return env


def run_glx_module(what: str, repo_root: Path, *mod_candidates: str, args: Optional[List[str]] = None) -> int:
    """
    Uruchamia moduÅ‚ Pythona po nazwie (np. 'glx.tools.delta_fingerprint').
    Szuka zarÃ³wno w sys.path, jak i po Å›cieÅ¼ce ÅºrÃ³dÅ‚owej w repo (fallback).
    """
    if args is None:
        args = []
    glx_root = repo_root  # przyjmujemy, Å¼e narzÄ™dzia leÅ¼Ä… w repo (glx/tools/*)
    for mod in mod_candidates:
        try:
            __import__(mod)  # sprawdÅº importowalnoÅ›Ä‡
            cmd = [sys.executable, "-m", mod, *args]
            log(f"{what}: {' '.join(cmd)}")
            return subprocess.run(cmd, cwd=str(repo_root), env=py_env(glx_root, repo_root)).returncode
        except Exception:
            continue
    # fallback: bezpoÅ›rednia Å›cieÅ¼ka do pliku
    for mod in mod_candidates:
        rel = Path(*mod.split("."))
        for cand in (glx_root / f"{rel}.py", glx_root / rel / "__init__.py"):
            if cand.exists():
                cmd = [sys.executable, str(cand), *args]
                log(f"{what}: {' '.join(cmd)}")
                return subprocess.run(cmd, cwd=str(repo_root), env=py_env(glx_root, repo_root)).returncode
    fail("Module not found: " + " | ".join(mod_candidates))
    return 1


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Snippet do wiadomoÅ›ci commita (z delta_report.json)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_commit_snippet(root: Path) -> str:
    """
    Buduje prefix wiadomoÅ›ci commit na podstawie artefaktu .glx/delta_report.json.
    Gdy brak artefaktu â€“ zwraca pusty string.
    """
    d = glx_dir(root)
    rep = d / "delta_report.json"
    if not rep.exists():
        return ""
    try:
        obj = json.loads(rep.read_text(encoding="utf-8"))
    except Exception:
        return ""
    hist: Dict[str, int] = obj.get("hist") or {}
    fp = obj.get("hash") or ""
    if not hist and not fp:
        return ""
    # Top-k tokenÃ³w
    top = sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    top_str = ", ".join(f"{k}Ã—{v}" for k, v in top) if top else "â€”"
    lines = [
        "[GLX] Î”-tokens: " + top_str,
        f"[GLX] Fingerprint: {fp}" if fp else "",
    ]
    return "\n".join([ln for ln in lines if ln]).strip()


def write_commit_snippet(root: Path) -> Optional[Path]:
    """
    Zapisuje .glx/commit_snippet.txt na podstawie build_commit_snippet().
    Zwraca Å›cieÅ¼kÄ™ pliku lub None (gdy nic nie zapisano).
    """
    text = build_commit_snippet(root)
    if not text:
        return None
    d = ensure_glx_dir(root)
    out = d / "commit_snippet.txt"
    out.write_text(text + "\n", encoding="utf-8")
    return out


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Walidacje lekkie (opcjonalnie wywoÅ‚ywane przez hooki)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def reject_big_files(paths: Iterable[Path], *, threshold_mb: int = 10) -> None:
    """
    ZgÅ‚asza bÅ‚Ä…d, jeÅ›li wÅ›rÃ³d Å›cieÅ¼ek sÄ… pliki wiÄ™ksze niÅ¼ threshold_mb.
    """
    too_big: List[Tuple[Path, float]] = []
    for p in paths:
        try:
            if p.exists() and p.is_file():
                sz_mb = p.stat().st_size / (1024 * 1024)
                if sz_mb > threshold_mb:
                    too_big.append((p, sz_mb))
        except Exception:
            continue
    if too_big:
        listing = "\n".join(f"- {p} ({sz:.1f} MB)" for p, sz in too_big)
        fail(f"Zbyt duÅ¼e pliki w commitcie (> {threshold_mb} MB):\n{listing}")


def check_python_compiles(paths: Iterable[Path]) -> None:
    """
    KrÃ³tkie sprawdzenie kompilowalnoÅ›ci plikÃ³w .py (py_compile).
    """
    py = [p for p in paths if p.suffix == ".py" and p.exists()]
    if not py:
        return
    errors: List[str] = []
    for p in py:
        rc = subprocess.run([sys.executable, "-m", "py_compile", str(p)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if rc.returncode != 0:
            errors.append(f"- {p}")
    if errors:
        fail("BÅ‚Ä™dy kompilacji Pythona:\n" + "\n".join(errors))
