# glitchlab/analysis/diff.py
# -*- coding: utf-8 -*-
"""
Visual-less DIFF: odczyt patcha i metryki zmian dla zakresu GIT (BASE..HEAD).
Bez tokenizacji/AST — tylko statystyki i surowy patch.

Python: 3.9 (stdlib only)

Publiczne API:
- read_unified_patch(repo_root: Path, diff_range: str, **opts) -> str
- read_numstat(repo_root: Path, diff_range: str, include=None, exclude=None) -> list[dict]
- read_name_status(repo_root: Path, diff_range: str, include=None, exclude=None) -> list[dict]
- read_shortstat(repo_root: Path, diff_range: str) -> dict
- summarize_diff(repo_root: Path, diff_range: str, include=None, exclude=None) -> dict

Uwaga:
- „Zakres” może mieć postać 'A..B', pojedynczego refa ('HEAD' / sha),
  lub być rozwiązywany z fallbackami zgodnie z analysis.git_io.parse_range_arg().
- Filtry include/exclude to listy regexów (kompilowane z re.IGNORECASE).
"""
from __future__ import annotations

import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple

from .git_io import parse_range_arg, repo_root as _repo_root

__all__ = [
    "DEFAULT_IGNORE_REGEX",
    "read_unified_patch",
    "read_numstat",
    "read_name_status",
    "read_shortstat",
    "summarize_diff",
]

# ──────────────────────────────────────────────────────────────────────────────
# Konfiguracja filtrowania (domyślna)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_IGNORE_REGEX = re.compile(
    r"""(?ix)
    ( ^|/ )(
        \.git
      | \.glx
      | __pycache__
      | \.pytest_cache
      | dist
      | build
      | backup
    )( /|$ )
    | \.(png|jpg|jpeg|gif|svg|ico|bmp|pdf|zip|tar|gz|7z|bin|exe|dll|dylib)$
    """
)

# ──────────────────────────────────────────────────────────────────────────────
# Pomocnicze
# ──────────────────────────────────────────────────────────────────────────────

def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

def _ensure_repo(path: Optional[Path]) -> Path:
    return _repo_root(path)

def _compile_filters(patterns: Optional[Iterable[str]]) -> Optional[List[re.Pattern]]:
    if not patterns:
        return None
    out: List[re.Pattern] = []
    for p in patterns:
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error:
            # potraktuj jako dosłowny fragment ścieżki
            out.append(re.compile(re.escape(p), re.IGNORECASE))
    return out

def _wanted(path: str, include: Optional[List[re.Pattern]], exclude: Optional[List[re.Pattern]]) -> bool:
    # normalizacja separatorów
    p = str(PurePosixPath(path))
    if exclude and any(rx.search(p) for rx in exclude):
        return False
    if DEFAULT_IGNORE_REGEX.search(p):
        # domyślne ignorowanie – ale include może nadpisać (preferencja dla include)
        if include and any(rx.search(p) for rx in include):
            return True
        return False
    if include:
        return any(rx.search(p) for rx in include)
    return True

def _split_range(repo: Path, diff_range: str) -> Tuple[str, str, str]:
    rng, base, head = parse_range_arg(repo, diff_range)
    return rng, base, head

# ──────────────────────────────────────────────────────────────────────────────
# Odczyt patcha (unified diff)
# ──────────────────────────────────────────────────────────────────────────────

def read_unified_patch(
    repo_root: Path,
    diff_range: str,
    *,
    unified: int = 3,
    detect_renames: bool = True,
    detect_copies: bool = True,
    ignore_ws: bool = False,
    pathspec: Optional[Iterable[str]] = None,
) -> str:
    """
    Zwraca surowy patch (unified diff) dla zakresu.
    """
    repo = _ensure_repo(repo_root)
    rng, base, head = _split_range(repo, diff_range)

    args = ["diff", f"-U{unified}"]
    if detect_renames:
        args.append("-M")
    if detect_copies:
        args.append("-C")
    # stabilna kolejność plików
    args.extend(["--diff-algorithm=histogram", "--no-color"])
    if ignore_ws:
        args.append("-w")
    args.append(rng)
    if pathspec:
        args.append("--")
        args.extend(list(pathspec))

    proc = _git(repo, *args)
    return proc.stdout if proc.returncode == 0 else ""

# ──────────────────────────────────────────────────────────────────────────────
# Numstat / Name-Status / Shortstat
# ──────────────────────────────────────────────────────────────────────────────

def read_numstat(
    repo_root: Path,
    diff_range: str,
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> List[Dict[str, object]]:
    """
    Zwraca listę rekordów: {"adds": int|None, "dels": int|None, "path": str, "binary": bool}
    (None dla wartości z '-' w numstat = plik binarny).
    """
    repo = _ensure_repo(repo_root)
    rng, base, head = _split_range(repo, diff_range)
    inc = _compile_filters(include)
    exc = _compile_filters(exclude)

    args = ["diff", "--numstat", "-M", "-C", rng]
    out = _git(repo, *args)
    rows: List[Dict[str, object]] = []
    if out.returncode != 0:
        return rows

    for ln in out.stdout.splitlines():
        if not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        a_raw, d_raw, path = parts[0], parts[1], parts[2]
        if not _wanted(path, inc, exc):
            continue
        binary = (a_raw == "-" or d_raw == "-")
        adds = None if binary else int(a_raw or 0)
        dels = None if binary else int(d_raw or 0)
        rows.append({"adds": adds, "dels": dels, "path": path, "binary": binary})
    return rows


def read_name_status(
    repo_root: Path,
    diff_range: str,
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    """
    Zwraca listę rekordów: {"status": "A|M|D|R|C|T", "path": "nowa_ścieżka", "from": "stara_ścieżka?"}
    """
    repo = _ensure_repo(repo_root)
    rng, base, head = _split_range(repo, diff_range)
    inc = _compile_filters(include)
    exc = _compile_filters(exclude)

    args = ["diff", "--name-status", "-M", "-C", rng]
    out = _git(repo, *args)
    rows: List[Dict[str, str]] = []
    if out.returncode != 0:
        return rows

    for ln in out.stdout.splitlines():
        if not ln.strip():
            continue
        parts = ln.split("\t")
        st = parts[0].strip()
        if st.startswith("R") or st.startswith("C"):
            # R### <old> <new>  |  C### <old> <new>
            if len(parts) >= 3:
                old_p, new_p = parts[1], parts[2]
                if _wanted(new_p, inc, exc):
                    rows.append({"status": st[:1], "path": new_p, "from": old_p})
        else:
            # A/M/D/T  <path>
            if len(parts) >= 2:
                p = parts[1]
                if _wanted(p, inc, exc):
                    rows.append({"status": st[:1], "path": p, "from": ""})
    return rows


def read_shortstat(repo_root: Path, diff_range: str) -> Dict[str, int]:
    """
    Zwraca: {"files": X, "insertions": Y, "deletions": Z}
    (parsowane z `git diff --shortstat`).
    """
    repo = _ensure_repo(repo_root)
    rng, base, head = _split_range(repo, diff_range)
    args = ["diff", "--shortstat", rng]
    out = _git(repo, *args)
    files = insertions = deletions = 0
    if out.returncode == 0:
        txt = out.stdout.strip()
        # Przykład: "3 files changed, 12 insertions(+), 5 deletions(-)"
        m_files = re.search(r"(\d+)\s+files?\s+changed", txt)
        m_ins = re.search(r"(\d+)\s+insertions?\(\+\)", txt)
        m_del = re.search(r"(\d+)\s+deletions?\(-\)", txt)
        files = int(m_files.group(1)) if m_files else 0
        insertions = int(m_ins.group(1)) if m_ins else 0
        deletions = int(m_del.group(1)) if m_del else 0
    return {"files": files, "insertions": insertions, "deletions": deletions}

# ──────────────────────────────────────────────────────────────────────────────
# Agregaty i metryki
# ──────────────────────────────────────────────────────────────────────────────

def _ext_of(path: str) -> str:
    p = str(PurePosixPath(path))
    dot = p.rfind(".")
    if dot <= 0:
        return ""
    return p[dot + 1 :].lower()

def _top_k(counter: Counter, k: int = 10) -> List[Tuple[str, int]]:
    return counter.most_common(k)

def summarize_diff(
    repo_root: Path,
    diff_range: str,
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """
    Zwraca syntetyczny raport z zakresu (bez tokenów):
    {
      "range": "A..B",
      "base": "<sha>",
      "head": "<sha>",
      "files_changed": int,
      "insertions": int,
      "deletions": int,
      "delta": int,               # adds + dels
      "net": int,                 # adds - dels
      "binary_files": int,
      "by_status": {"A": n, "M": n, "D": n, "R": n, "C": n, "T": n},
      "by_ext": { "py": {"files": n, "adds": a, "dels": d}, ... },
      "top_dirs": [ ["analysis", n], ... ],
      "top_files_adds": [ ["path", adds], ... ],
      "top_files_dels": [ ["path", dels], ... ],
      "hunks": int,               # liczba @@ w unified patch
      "patch_bytes": int
    }
    """
    repo = _ensure_repo(repo_root)
    rng, base, head = _split_range(repo, diff_range)

    inc = _compile_filters(include)
    exc = _compile_filters(exclude)

    # 1) shortstat
    ss = read_shortstat(repo, diff_range)

    # 2) name-status
    ns = read_name_status(repo, diff_range, include=include, exclude=exclude)
    by_status = Counter([r["status"] for r in ns])

    # 3) numstat
    num = read_numstat(repo, diff_range, include=include, exclude=exclude)

    # agregaty
    adds_total = sum((r["adds"] or 0) for r in num)
    dels_total = sum((r["dels"] or 0) for r in num)
    binary_files = sum(1 for r in num if r["binary"])

    # by_ext
    ext_files: Dict[str, set] = defaultdict(set)
    ext_adds: Dict[str, int] = defaultdict(int)
    ext_dels: Dict[str, int] = defaultdict(int)

    for r in num:
        ext = _ext_of(str(r["path"]))
        ext_files[ext].add(r["path"])
        if not r["binary"]:
            ext_adds[ext] += int(r["adds"] or 0)
            ext_dels[ext] += int(r["dels"] or 0)

    by_ext = {
        k or "": {"files": len(ext_files[k]), "adds": ext_adds[k], "dels": ext_dels[k]}
        for k in sorted(ext_files.keys())
    }

    # top dirs (po pierwszym segmentcie ścieżki)
    dir_counter: Counter = Counter()
    for r in ns:
        p = str(PurePosixPath(r["path"]))
        first = p.split("/", 1)[0] if "/" in p else p
        if not _wanted(p, inc, exc):
            continue
        dir_counter[first] += 1

    # top files by adds/dels
    adds_by_file: Dict[str, int] = defaultdict(int)
    dels_by_file: Dict[str, int] = defaultdict(int)
    for r in num:
        if r["binary"]:
            continue
        adds_by_file[str(r["path"])] += int(r["adds"] or 0)
        dels_by_file[str(r["path"])] += int(r["dels"] or 0)

    top_adds = sorted(adds_by_file.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_dels = sorted(dels_by_file.items(), key=lambda kv: kv[1], reverse=True)[:10]

    # 4) policz hunki i wielkość patcha (unified = 3, wykrywanie rename/copy)
    patch_txt = read_unified_patch(repo, diff_range, unified=3, detect_renames=True, detect_copies=True, ignore_ws=False)
    hunks = len(re.findall(r"^@@", patch_txt, flags=re.MULTILINE))
    patch_bytes = len(patch_txt.encode("utf-8"))

    return {
        "range": rng,
        "base": base,
        "head": head,
        "files_changed": int(ss.get("files", 0)),
        "insertions": int(ss.get("insertions", adds_total)),  # shortstat preferowane; fallback na numstat
        "deletions": int(ss.get("deletions", dels_total)),
        "delta": int(adds_total + dels_total),
        "net": int(adds_total - dels_total),
        "binary_files": int(binary_files),
        "by_status": dict(by_status),
        "by_ext": by_ext,
        "top_dirs": _top_k(dir_counter, 10),
        "top_files_adds": top_adds,
        "top_files_dels": top_dels,
        "hunks": hunks,
        "patch_bytes": patch_bytes,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Proste CLI (dla szybkiej inspekcji lokalnej)
# ──────────────────────────────────────────────────────────────────────────────

def _cli(argv: Optional[List[str]] = None) -> None:
    import argparse
    import json as _json

    p = argparse.ArgumentParser(prog="analysis.diff", description="GLX: patch & metrics (bez tokenizacji)")
    p.add_argument("--range", dest="diff_range", default="HEAD~1..HEAD", help="zakres A..B lub pojedynczy ref (domyślnie HEAD~1..HEAD)")
    p.add_argument("--patch", action="store_true", help="wypisz unified diff")
    p.add_argument("--json", action="store_true", help="wypisz JSON z summarize_diff()")
    p.add_argument("--include", nargs="*", default=None, help="regex-y include (nadpisują domyślne ignore)")
    p.add_argument("--exclude", nargs="*", default=None, help="regex-y exclude")
    args = p.parse_args(argv)

    repo = _ensure_repo(None)

    if args.patch:
        print(read_unified_patch(repo, args.diff_range))
        return

    rep = summarize_diff(repo, args.diff_range, include=args.include, exclude=args.exclude)
    if args.json:
        print(_json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        print(f"[range] {rep['range']}  files={rep['files_changed']}  +{rep['insertions']} -{rep['deletions']}  Δ={rep['delta']} net={rep['net']}")
        print(f"[status] {rep['by_status']}")
        print(f"[binary] {rep['binary_files']}  [hunks] {rep['hunks']}  [patch_bytes] {rep['patch_bytes']}")
        print(f"[top dirs] {rep['top_dirs'][:5]}")
        print(f"[by ext] keys={list(rep['by_ext'].keys())[:6]} ..")

if __name__ == "__main__":
    _cli()
