# glitchlab/analysis/git_io.py
# -*- coding: utf-8 -*-
"""
Git I/O + RepoMosaic (kafel=plik) dla analizy Δ (BASE..HEAD) — Python 3.9 (stdlib only)

Cele:
- Jednoznaczne rozwiązywanie zakresów (A..B) z deterministycznymi fallbackami.
- Odporne na brak base: merge-base → HEAD~1 → pusty tree.
- Integracja z .glx/state.json (odczyt/zapis base_sha), pomocnicze narzędzia Git.
- RepoMosaic: lista „kafli” (pliki), edge/roi, siatka, churn; tryb „unstaged”.
- Stabilne parsowanie `git diff --name-status` (A/M/D/R/C/T …).
- Proste CLI do inspekcji.

Publiczne API:
- resolve_range(repo_root, base=None, head=None) -> (range_str, base_sha, head_sha)
- parse_range_arg(repo_root, diff_range) -> (range_str, base_sha, head_sha)
- repo_root(start=None) -> Path
- is_git_repo(repo_root) -> bool
- rev_parse(repo_root, ref) -> Optional[str]
- merge_base(repo_root, a, b) -> Optional[str]
- empty_tree(repo_root) -> str
- current_branch(repo_root=None) -> Optional[str]
- rev_short(repo_root, rev) -> str
- list_tracked_files(repo_root=None) -> List[str]
- changed_files(base, head="HEAD", filters=None) -> List[str]
- changed_py_files(base, head="HEAD") -> List[str]
- show_file_at_rev(path, rev="HEAD") -> Optional[str]
- read_glx_state() / write_glx_state(state)
- build_repo_mosaic(base=None, head="HEAD", include_unstaged=False) -> (RepoInfo, RepoMosaic, churn)
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Stałe
# ──────────────────────────────────────────────────────────────────────────────

# SHA „pustego drzewa” (gdy repo ma 0/1 commit i HEAD~1 nie istnieje)
EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
GLX_DIR = ".glx"
GLX_STATE = "state.json"


# ──────────────────────────────────────────────────────────────────────────────
# Lokator repo i podstawowe wywołania gita
# ──────────────────────────────────────────────────────────────────────────────

def repo_root(start: Optional[Path] = None) -> Path:
    """
    Zwraca korzeń repo (git rev-parse --show-toplevel).
    Fallback: szuka w górę katalogu .git albo .glx/state.json.
    """
    start = Path(start or Path.cwd()).resolve()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start), text=True, stderr=subprocess.DEVNULL
        ).strip()
        if out:
            return Path(out)
    except Exception:
        pass

    # fallback: manualny spacer w górę
    for p in [start, *start.parents]:
        if (p / ".git").exists() or (p / GLX_DIR / "state.json").exists():
            return p
    return start


def _git(args: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Uruchamia 'git <args>' i zwraca CompletedProcess (text)."""
    cwd = cwd or repo_root()
    res = subprocess.run(["git", *args], cwd=str(cwd),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, ["git", *args], res.stdout, res.stderr)
    return res


def _git_text(args: List[str], cwd: Optional[Path] = None, check: bool = True) -> str:
    p = _git(args, cwd=cwd, check=check)
    return p.stdout.strip() if p.stdout else ""


def is_git_repo(root: Path) -> bool:
    try:
        out = _git_text(["rev-parse", "--is-inside-work-tree"], cwd=root, check=True)
        return out.lower() == "true"
    except Exception:
        return False


def rev_parse(root: Path, ref: str) -> Optional[str]:
    """Zwraca pełny SHA dla `ref` lub None, jeśli niepoprawny."""
    if not ref:
        return None
    try:
        out = _git_text(["rev-parse", ref], cwd=root, check=True)
        return out or None
    except Exception:
        return None


def merge_base(root: Path, a: str, b: str) -> Optional[str]:
    """Zwraca merge-base(a, b) lub None."""
    try:
        out = _git_text(["merge-base", a, b], cwd=root, check=True)
        return out or None
    except Exception:
        return None


def empty_tree(root: Path) -> str:
    """Hash pustego drzewa Gita (stały w ramach implementacji gita)."""
    try:
        out = _git_text(["hash-object", "-t", "tree", "/dev/null"], cwd=root, check=True)
        return out or EMPTY_TREE_SHA
    except Exception:
        return EMPTY_TREE_SHA


def current_branch(root: Optional[Path] = None) -> Optional[str]:
    """Zwraca nazwę aktualnej gałęzi (lub None np. w detached HEAD)."""
    try:
        b = _git_text(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root or repo_root(), check=True)
        return b if b and b != "HEAD" else None
    except Exception:
        return None


def rev_short(root: Path, rev: str) -> str:
    """Zwraca skrót 7 znaków dla podanego revision (bez rzucania błędów na brak)."""
    try:
        s = _git_text(["rev-parse", "--short=7", rev], cwd=root, check=True)
        return s or (rev[:7] if len(rev) >= 7 else rev)
    except Exception:
        return rev[:7] if len(rev) >= 7 else rev


def list_tracked_files(root: Optional[Path] = None) -> List[str]:
    """Wszystkie śledzone pliki (git ls-files)."""
    out = _git_text(["ls-files"], cwd=root or repo_root(), check=True)
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Walidacja / fallbacki referencji
# ──────────────────────────────────────────────────────────────────────────────

def _rev_ok(rev: str, root: Optional[Path] = None) -> bool:
    """
    Czy referencja istnieje? Akceptujemy commit/annotated tag.
    UWAGA: empty-tree nie jest zwykłym refem – ale traktujemy go jako „ok”.
    """
    root = root or repo_root()
    if not rev:
        return False
    if rev == EMPTY_TREE_SHA:
        return True
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{rev}^{{commit}}"],
            cwd=str(root), text=True, stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# ŹRÓDŁO PRAWDY DLA ZAKRESÓW: resolve_range / parse_range_arg
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_head(root: Path, head: Optional[str]) -> str:
    """HEAD jeśli brak, inaczej ref → sha (o ile możliwe)."""
    h = head or "HEAD"
    return rev_parse(root, h) or h


def _resolve_base_with_fallbacks(root: Path, head_sha: str, base: Optional[str]) -> str:
    """
    Strategia:
      1) jeśli `base` podano → zrev-parsuj (lub zostaw jak jest, jeśli nie można).
      2) spróbuj merge-base(HEAD, origin/<branch|master|main>).
      3) HEAD~1
      4) pusty tree (pierwszy commit).
    """
    if base:
        return rev_parse(root, base) or base

    # 2) merge-base z „główną” gałęzią zdalną (preferencja: bieżąca → master → main)
    br = current_branch(root) or ""
    for remote in (f"origin/{br}" if br else None, "origin/master", "origin/main"):
        if not remote:
            continue
        mb = merge_base(root, head_sha or "HEAD", remote)
        if mb:
            return mb

    # 3) HEAD~1
    prev = rev_parse(root, "HEAD~1")
    if prev:
        return prev

    # 4) pusty tree
    return empty_tree(root)


def resolve_range(repo_root_: Path, base: Optional[str] = None, head: Optional[str] = None) -> Tuple[str, str, str]:
    """
    Zwraca (range_str, base_sha, head_sha) z deterministycznymi fallbackami.
    Przykłady:
      - resolve_range(root) → („HEAD~1..HEAD”, <sha_prev|empty_tree>, <sha_head>)
      - resolve_range(root, base="A", head="B") → („A..B”, <shaA>, <shaB>)
      - resolve_range(root, head="HEAD") → („<auto_base>..HEAD”, <sha>, <shaHEAD>)
    """
    root = Path(repo_root_)
    if not is_git_repo(root):
        raise RuntimeError(f"Nieprawidłowe repozytorium git: {root}")

    head_sha = _resolve_head(root, head)
    base_sha = _resolve_base_with_fallbacks(root, head_sha, base)
    return f"{base_sha}..{head_sha}", base_sha, head_sha


def parse_range_arg(repo_root_: Path, diff_range: str) -> Tuple[str, str, str]:
    """
    Parsuje argument zakresu w formatach:
      - 'A..B'  → resolve A..B (rev-parse każdego końca)
      - 'HEAD'  → resolve_range(head='HEAD')  (base: fallbacki)
      - '<sha>' → resolve_range(head='<sha>')
    Zwraca (range_str, base_sha, head_sha).
    """
    root = Path(repo_root_)
    dr = (diff_range or "").strip()
    if ".." in dr:
        a, b = dr.split("..", 1)
        a_sha = rev_parse(root, a.strip()) or a.strip()
        b_sha = rev_parse(root, b.strip()) or b.strip()
        return f"{a_sha}..{b_sha}", a_sha, b_sha
    return resolve_range(root, base=None, head=dr or "HEAD")


# ──────────────────────────────────────────────────────────────────────────────
# Parsowanie diffu name-status z fallbackami
# ──────────────────────────────────────────────────────────────────────────────

def _parse_diff_name_status_lines(lines: List[str]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for ln in lines:
        if not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) == 2:
            st, p = parts
        elif len(parts) == 3:
            st, _, p = parts  # Rxxx<TAB>old<TAB>new → bierzemy „new”
        else:
            st, p = parts[0], parts[-1]
        st = st.strip()
        p = (p or "").strip()
        if p:
            items.append((st, p))
    return items


def _diff_name_status_safe(base: str, head: str, cwd: Optional[Path] = None) -> List[Tuple[str, str]]:
    """
    Zwraca listę (status, path) z `git diff --name-status base..head`, z fallbackami:
      1) jeśli base nie istnieje → spróbuj merge-base(HEAD, origin/<branch|master|main>)
      2) jeśli dalej nie działa → HEAD~1
      3) jeśli nadal błąd → użyj pustego drzewa jako base
      4) jeśli nadal błąd → pusta lista
    """
    root = cwd or repo_root()

    # 1) szybka ścieżka
    p = _git(["diff", "--name-status", f"{base}..{head}"], cwd=root, check=False)
    if p.returncode == 0:
        return _parse_diff_name_status_lines(p.stdout.splitlines())

    # 2) merge-base
    br = current_branch(root) or "master"
    mb = merge_base(root, head or "HEAD", f"origin/{br}") or merge_base(root, head or "HEAD", "origin/master") or merge_base(root, head or "HEAD", "origin/main")
    candidate = mb or base

    if not _rev_ok(candidate, root=root):
        # 3) HEAD~1
        if _rev_ok("HEAD~1", root=root):
            candidate = "HEAD~1"
        else:
            # 4) pusty tree
            candidate = empty_tree(root)

    p2 = _git(["diff", "--name-status", f"{candidate}..{head}"], cwd=root, check=False)
    if p2.returncode == 0:
        return _parse_diff_name_status_lines(p2.stdout.splitlines())

    return []


# ──────────────────────────────────────────────────────────────────────────────
# Pliki zmienione i snapshoty
# ──────────────────────────────────────────────────────────────────────────────

def changed_files(base: str, head: str = "HEAD", filters: Optional[Iterable[str]] = None) -> List[str]:
    """
    Lista zmienionych ścieżek (A/M/D/R/C/T) między base..head (odporna na brak base).
    filters: np. (".py", "glitchlab/") – proste startswith/endswith; None = bez filtra.
    """
    name_status = _diff_name_status_safe(base, head)
    paths: List[str] = []
    for st_code, path in name_status:
        if not path:
            continue
        if filters:
            ok = False
            for f in filters:
                if f.startswith("."):
                    ok |= path.endswith(f)
                else:
                    ok |= path.startswith(f)
            if not ok:
                continue
        paths.append(path)  # dla rename mamy już „new”
    return paths


def changed_py_files(base: str, head: str = "HEAD") -> List[str]:
    """Skrót: tylko .py z diffu base..head (odporny na brak base)."""
    return changed_files(base, head, filters=(".py",))


def show_file_at_rev(path: str, rev: str = "HEAD") -> Optional[str]:
    """
    Zwraca zawartość pliku z danej rewizji (git show rev:path). Gdy plik nie istniał – None.
    """
    p = _git(["show", f"{rev}:{path}"], check=False)
    return p.stdout if p.returncode == 0 else None


# ──────────────────────────────────────────────────────────────────────────────
# GLX state.json (źródło prawdy: base_sha, itp.)
# ──────────────────────────────────────────────────────────────────────────────

def _glx_path(root: Optional[Path] = None) -> Path:
    return (root or repo_root()) / GLX_DIR


def _glx_state_path(root: Optional[Path] = None) -> Path:
    return _glx_path(root) / GLX_STATE


def read_glx_state(root: Optional[Path] = None) -> Dict:
    """
    Czyta .glx/state.json. Gdy brak – zwraca domyślne pole z base_sha=None.
    """
    r = root or repo_root()
    p = _glx_state_path(r)
    if not p.exists():
        return {"app": "glitchlab", "base_sha": None, "last_seq": "", "branch": current_branch(r) or "master"}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"app": "glitchlab", "base_sha": None, "last_seq": "", "branch": current_branch(r) or "master"}


def write_glx_state(state: Dict, root: Optional[Path] = None) -> None:
    """
    Zapisuje .glx/state.json (pretty, utf-8). Tworzy katalog .glx jeśli trzeba.
    """
    r = root or repo_root()
    d = _glx_path(r)
    d.mkdir(parents=True, exist_ok=True)
    p = d / GLX_STATE
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# RepoMosaic = „kafelki plików” (edge/roi jako heurystyki z diff)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RepoInfo:
    root: Path
    branch: str
    base_sha: str
    head_sha: str
    base7: str
    head7: str


@dataclass
class RepoMosaic:
    """
    Prosty model mozaiki repo dla Φ/Ψ:
    - files: lista ścieżek (kafli)
    - edge: „żywość” (0..1) kafla – heurystyka na bazie statusu i rozszerzenia
    - roi:  kafle „istotne” (np. .py i/lub dotknięte w diff)
    - layout_rows/cols: propozycja siatki (kwadrat najbliższy N)
    """
    files: List[str]
    edge: List[float]
    roi: List[float]
    layout_rows: int
    layout_cols: int
    extras: Dict[str, float] = field(default_factory=dict)


def _grid_for_n(n: int) -> Tuple[int, int]:
    """Dobiera (rows, cols) ≈ kwadrat dla n kafli."""
    if n <= 0:
        return (1, 1)
    import math
    r = int(math.sqrt(n))
    c = r
    while r * c < n:
        c += 1
        if r * c < n and r < c:
            r += 1
    return (r, c)


def _edge_score_for_path(path: str, status: str) -> float:
    """
    Heurystyka 'edge' (0..1): A/M>R>D, .py > inne, głębokie ścieżki nieco wyżej.
    """
    base = {"A": 0.75, "M": 0.70, "R": 0.60, "C": 0.55, "D": 0.50}.get(status[:1], 0.60)
    if path.endswith(".py"):
        base += 0.10
    depth = max(0, path.count(os.sep) - 0)  # im głębiej, tym „bardziej kontekstowe”
    base += min(0.15, 0.02 * depth)
    return float(max(0.0, min(1.0, base)))


def _roi_flag_for_path(path: str, status: str) -> float:
    """Heurystyka 'roi' (0/1): .py oraz zmienione (A/M/R/C/D) → 1.0, inaczej 0.0"""
    return 1.0 if path.endswith(".py") and status[:1] in {"A", "M", "R", "C", "D"} else 0.0


def build_repo_mosaic(
    base: Optional[str] = None,
    head: str = "HEAD",
    include_unstaged: bool = False
) -> Tuple[RepoInfo, RepoMosaic, Dict[str, int]]:
    """
    Buduje RepoMosaic dla zakresu base..head.
    - base: gdy None → .glx/state.json:base_sha, w ostateczności merge-base(HEAD, origin/<branch>),
            a jeśli brak – empty-tree (pierwszy commit)
    - include_unstaged: jeśli True, traktuje 'git status --porcelain' jako diff vs HEAD
    Zwraca: (RepoInfo, RepoMosaic, churn_counters)
    """
    root = repo_root()
    branch = current_branch(root) or "master"

    # resolve base/head (użyjemy też „źródła prawdy”)
    if base is None:
        st = read_glx_state(root)
        base = st.get("base_sha") or None
    _, base_sha, head_sha = resolve_range(root, base=base, head=head)

    base7 = rev_short(root, base_sha)
    head7 = rev_short(root, head_sha)

    # nazwy/statusy
    if include_unstaged:
        # porównanie HEAD..(working tree) – 'git status --porcelain'
        st_out = _git_text(["status", "--porcelain"], cwd=root, check=True).splitlines()
        ns: List[Tuple[str, str]] = []
        for ln in st_out:
            ln = ln.strip()
            if not ln or len(ln) < 4:
                continue
            X, Y, rest = ln[0], ln[1], ln[3:]
            status = (Y if Y != " " else X).strip()
            ns.append((status, rest))
        name_status = ns
    else:
        name_status = _diff_name_status_safe(base_sha, head_sha, cwd=root)

    churn = {"A": 0, "M": 0, "D": 0, "R": 0, "C": 0, "other": 0}
    files: List[str] = []
    edge: List[float] = []
    roi: List[float] = []

    for st_code, path in name_status:
        if not path:
            continue
        files.append(path)
        sig = st_code[:1] if st_code else "M"
        if sig in churn:
            churn[sig] += 1
        else:
            churn["other"] += 1
        edge.append(_edge_score_for_path(path, st_code))
        roi.append(_roi_flag_for_path(path, st_code))

    # jeśli brak zmian – użyj śledzonych .py jako kafli „spokojnych”
    if not files:
        tracked = [p for p in list_tracked_files(root) if p.endswith(".py")]
        files = tracked[:36]
        edge = [0.35 for _ in files]
        roi = [0.0 for _ in files]

    rows, cols = _grid_for_n(len(files))
    repoM = RepoMosaic(files=files, edge=edge, roi=roi, layout_rows=rows, layout_cols=cols, extras={})

    info = RepoInfo(root=root, branch=branch, base_sha=base_sha, head_sha=head_sha, base7=base7, head7=head7)
    return info, repoM, churn


# ──────────────────────────────────────────────────────────────────────────────
# Proste CLI do szybkich testów lokalnych
# ──────────────────────────────────────────────────────────────────────────────

def _cli(argv: Optional[List[str]] = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="git_io", description="Git I/O + RepoMosaic")
    p.add_argument("--base", type=str, default=None, help="base sha (domyślnie z .glx/state.json lub merge-base/empty-tree)")
    p.add_argument("--head", type=str, default="HEAD", help="head rev (domyślnie HEAD)")
    p.add_argument("--unstaged", action="store_true", help="uwzględnij zmiany w working tree")
    p.add_argument("--json", action="store_true", help="wypisz JSON (skrót)")
    args = p.parse_args(argv)

    info, repoM, churn = build_repo_mosaic(base=args.base, head=args.head, include_unstaged=args.unstaged)
    print(f"[repo] {info.root}  branch={info.branch}  {info.base7}..{info.head7}")
    print(f"[files] n={len(repoM.files)}  grid={repoM.layout_rows}x{repoM.layout_cols}  churn={churn}")
    if args.json:
        payload = dict(
            repo=dict(root=str(info.root), branch=info.branch, base=info.base7, head=info.head7),
            files=repoM.files,
            edge=repoM.edge,
            roi=repoM.roi,
            grid=dict(rows=repoM.layout_rows, cols=repoM.layout_cols),
            churn=churn,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
