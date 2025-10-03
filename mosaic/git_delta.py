# glitchlab/gui/mosaic/git_delta.py
# -*- coding: utf-8 -*-
"""
Robust Git delta utilities (BASE..HEAD, staged, fallbacks).

Cel:
- Stabilne pobieranie listy zmienionych plików między BASE..HEAD (prod).
- Obsługa pierwszego commitu (HEAD vs empty-tree).
- Zgodność z Windows (bez /dev/null) — empty-tree wyliczane przez --stdin.
- Zwracanie ścieżek znormalizowanych (posix, względnie do repo).

Public API:
- collect_changed_files(repo_root: Path, base: Optional[str], head: Optional[str]) -> List[str]
- collect_staged_files(repo_root: Path) -> List[str]

Zwracane ścieżki są względem repo_root, z separatorem '/'.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


__all__ = [
    "collect_changed_files",
    "collect_staged_files",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _posix_rel(p: Path, root: Path) -> str:
    try:
        r = p.resolve().relative_to(root.resolve())
    except Exception:
        r = p
    return str(r).replace("\\", "/")


def _git_exec(args: List[str], cwd: Path, *, input_bytes: bytes = b"") -> Tuple[int, str, str]:
    """Uruchamia `git <args>` i zwraca (rc, stdout, stderr) z text=True."""
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return (p.returncode, p.stdout or "", p.stderr or "")
    except Exception as e:
        return (127, "", f"{e.__class__.__name__}: {e}")


def _is_git_repo(path: Path) -> bool:
    rc, out, _ = _git_exec(["rev-parse", "--is-inside-work-tree"], path)
    return rc == 0 and out.strip() == "true"


def _verify_ref(ref: str, cwd: Path) -> bool:
    rc, _, _ = _git_exec(["rev-parse", "--verify", ref], cwd)
    return rc == 0


def _head_ref(cwd: Path) -> Optional[str]:
    rc, out, _ = _git_exec(["rev-parse", "HEAD"], cwd)
    return out.strip() if rc == 0 and out.strip() else None


def _empty_tree_oid(cwd: Path) -> Optional[str]:
    """
    Wyznacza OID pustego drzewa repo, zgodny z aktualnym algorytmem (SHA-1 / SHA-256).
    Działa krzyżowo (Windows/Linux) przez --stdin (pusty input), bez /dev/null.
    """
    rc, out, err = _git_exec(["hash-object", "-t", "tree", "--stdin"], cwd, input_bytes=b"")
    if rc == 0 and out.strip():
        return out.strip()
    # ostateczny fallback: klasyczny SHA-1 pustego drzewa (gdy repo też jest SHA-1)
    return "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def _parse_name_status_z(payload: str) -> List[str]:
    """
    Parsuje wynik `--name-status -z` → zwraca listę ścieżek.
    Format to naprzemiennie: STATUS\0PATH\0 lub dla rename: R100\0OLD\0NEW\0.
    Interesują nas ścieżki docelowe (dla R* bierzemy NEW).
    """
    if not payload:
        return []
    parts = payload.split("\0")
    out: List[str] = []
    i = 0
    N = len(parts)
    while i < N:
        status = parts[i].strip() if parts[i] is not None else ""
        i += 1
        if not status:
            break
        if status.startswith("R"):  # rename: status, old, new
            if i + 1 >= N:
                break
            # old = parts[i]
            newp = parts[i + 1]
            if newp:
                out.append(newp)
            i += 2
        else:
            if i >= N:
                break
            path = parts[i]
            if path:
                out.append(path)
            i += 1
    # usuń puste i duplikaty z zachowaniem kolejności
    seen = set()
    uniq: List[str] = []
    for s in out:
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _diff_name_status(repo_root: Path, base: str, head: str) -> List[str]:
    """`git diff --name-status -z base..head` → listę ścieżek (lub [])."""
    rc, out, _ = _git_exec(["diff", "--name-status", "-z", f"{base}..{head}"], repo_root)
    if rc != 0 or not out:
        return []
    return _parse_name_status_z(out)


def _diff_tree_name_status(repo_root: Path, head: str) -> List[str]:
    """`git diff-tree --no-commit-id --name-status -r -z head` → lista ścieżek z jednego commit."""
    rc, out, _ = _git_exec(["diff-tree", "--no-commit-id", "--name-status", "-r", "-z", head], repo_root)
    if rc != 0 or not out:
        return []
    return _parse_name_status_z(out)


def _show_name_status(repo_root: Path, head: str) -> List[str]:
    """`git show --name-status -z --format= head` → lista ścieżek z commitu."""
    rc, out, _ = _git_exec(["show", "--name-status", "-z", "--format=", head], repo_root)
    if rc != 0 or not out:
        return []
    return _parse_name_status_z(out)


def _ls_files_head(repo_root: Path) -> List[str]:
    """
    Ostateczny fallback: wszystkie pliki śledzone w HEAD (dla pierwszego commitu, gdy BASE „puste”).
    """
    rc, out, _ = _git_exec(["ls-files", "-z"], repo_root)
    if rc != 0 or not out:
        return []
    parts = [p for p in out.split("\0") if p]
    # bez duplikatów
    seen = set()
    uniq: List[str] = []
    for s in parts:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def collect_changed_files(repo_root: Path, base: Optional[str], head: Optional[str]) -> List[str]:
    """
    Zwraca listę ścieżek (względem repo_root, posix) zmienionych między BASE..HEAD.
    Solidne fallbacki + pierwsze commit(y).
    """
    root = Path(repo_root).resolve()
    if not _is_git_repo(root):
        return []

    h = (head or "HEAD").strip() or "HEAD"
    if not _verify_ref(h, root):
        # brak HEAD? nic nie zrobimy sensownie
        return []

    b = (base or "").strip()

    # 1) Jeśli BASE niezweryfikowane — spróbuj wziąć empty-tree (pierwszy commit)
    if not b or not _verify_ref(b, root):
        et = _empty_tree_oid(root) or ""
        if et and _verify_ref(et, root):
            b = et
        else:
            # brak sensownego BASE — przechodzimy do fallbacków „pojedynczego commitu”
            b = ""

    # 2) Główna próba: diff base..head
    out = _diff_name_status(root, b, h) if b else []
    if out:
        return [_posix_rel(root / p, root) for p in out]

    # 3) Fallback: diff-tree (dla jednego HEAD)
    out = _diff_tree_name_status(root, h)
    if out:
        return [_posix_rel(root / p, root) for p in out]

    # 4) Fallback: show (też jeden commit)
    out = _show_name_status(root, h)
    if out:
        return [_posix_rel(root / p, root) for p in out]

    # 5) Ostatecznie: wszystkie śledzone pliki z HEAD (symulujemy „pierwszy commit”)
    out = _ls_files_head(root)
    return [_posix_rel(root / p, root) for p in out]


def collect_staged_files(repo_root: Path) -> List[str]:
    """
    Zwraca listę ścieżek (wzgl. repo_root, posix) plików z **indeksu** (staged changes).
    Użyteczne, gdy chcesz analizować to, co faktycznie będzie commitowane.
    """
    root = Path(repo_root).resolve()
    if not _is_git_repo(root):
        return []
    rc, out, _ = _git_exec(["diff", "--cached", "--name-only", "-z"], root)
    if rc != 0 or not out:
        return []
    parts = [p for p in out.split("\0") if p]
    return [_posix_rel(root / p, root) for p in parts]
