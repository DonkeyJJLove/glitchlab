# glitchlab/analysis/git_io.py
# Git I/O + RepoMosaic (kafel=plik) dla analizy Δ (BASE..HEAD)
# Python 3.9+ (stdlib only)

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Bezpieczne uruchomienie git (cross-platform, bez backticków)
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
        if (p / ".git").exists() or (p / ".glx" / "state.json").exists():
            return p
    return start


def _git(args: List[str], cwd: Optional[Path] = None, check: bool = True) -> str:
    """
    Uruchamia 'git <args>' i zwraca stdout (text). Przy błędzie – rzuca CalledProcessError.
    """
    cwd = cwd or repo_root()
    res = subprocess.run(["git", *args], cwd=str(cwd),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, ["git", *args], res.stdout, res.stderr)
    return res.stdout


# ──────────────────────────────────────────────────────────────────────────────
# Podstawowe operacje GIT
# ──────────────────────────────────────────────────────────────────────────────

def git_merge_base(head: str = "HEAD", other: str = "origin/master") -> Optional[str]:
    """Zwraca sha wspólnego przodka (merge-base). Gdy brak – None."""
    try:
        sha = _git(["merge-base", head, other]).strip()
        return sha or None
    except subprocess.CalledProcessError:
        return None


def current_branch() -> Optional[str]:
    """Zwraca nazwę aktualnej gałęzi (lub None np. w detached HEAD)."""
    try:
        b = _git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
        return b if b and b != "HEAD" else None
    except subprocess.CalledProcessError:
        return None


def rev_short(rev: str) -> str:
    """Zwraca skrót 7 znaków dla podanego revision (bez rzucania błędów na brak)."""
    try:
        s = _git(["rev-parse", "--short=7", rev]).strip()
        return s or (rev[:7] if len(rev) >= 7 else rev)
    except subprocess.CalledProcessError:
        return rev[:7] if len(rev) >= 7 else rev


def list_tracked_files() -> List[str]:
    """Wszystkie śledzone pliki (git ls-files)."""
    out = _git(["ls-files"]).splitlines()
    return [ln.strip() for ln in out if ln.strip()]


def changed_files(base: str, head: str = "HEAD",
                  filters: Optional[Iterable[str]] = None) -> List[str]:
    """
    Lista zmienionych ścieżek (A/M/D/R/T) między base..head (git diff --name-status).
    filters: np. (".py", "glitchlab/") – proste startswith/endswith; None = bez filtra.
    """
    args = ["diff", "--name-status", f"{base}..{head}"]
    out = _git(args).splitlines()
    paths: List[str] = []
    for ln in out:
        if not ln.strip():
            continue
        # format: "M\tpath" lub "R100\told\tnew"
        parts = ln.split("\t")
        if len(parts) == 2:
            _, p = parts
        elif len(parts) == 3:
            # renames: bierzemy nową ścieżkę
            _, _, p = parts
        else:
            p = parts[-1]
        p = p.strip()
        if not p:
            continue
        if filters:
            ok = False
            for f in filters:
                if f.startswith("."):
                    ok |= p.endswith(f)
                else:
                    ok |= p.startswith(f)
            if not ok:
                continue
        paths.append(p)
    return paths


def changed_py_files(base: str, head: str = "HEAD") -> List[str]:
    """Skrót: tylko .py z diffu base..head."""
    return changed_files(base, head, filters=(".py",))


def show_file_at_rev(path: str, rev: str = "HEAD") -> Optional[str]:
    """
    Zwraca zawartość pliku z danej rewizji (git show rev:path). Gdy plik nie istniał – None.
    """
    try:
        blob = _git(["show", f"{rev}:{path}"])
        return blob
    except subprocess.CalledProcessError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# GLX state.json (źródło prawdy: base_sha, itp.)
# ──────────────────────────────────────────────────────────────────────────────

GLX_DIR = ".glx"
GLX_STATE = "state.json"


def _glx_path() -> Path:
    return repo_root() / GLX_DIR


def _glx_state_path() -> Path:
    return _glx_path() / GLX_STATE


def read_glx_state() -> Dict:
    """
    Czyta .glx/state.json. Gdy brak – zwraca domyślne pole z base_sha=None.
    """
    p = _glx_state_path()
    if not p.exists():
        return {"app": "glitchlab", "base_sha": None, "last_seq": "", "branch": current_branch() or "master"}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # plik uszkodzony: nie blokuj – zwróć minimalny słownik
        return {"app": "glitchlab", "base_sha": None, "last_seq": "", "branch": current_branch() or "master"}


def write_glx_state(state: Dict) -> None:
    """
    Zapisuje .glx/state.json (pretty, utf-8). Tworzy katalog .glx jeśli trzeba.
    """
    d = _glx_path()
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
    """
    Heurystyka 'roi' (0/1): .py oraz zmienione (A/M/R/C/D) → 1.0, inaczej 0.0
    """
    return 1.0 if path.endswith(".py") and status[:1] in {"A", "M", "R", "C", "D"} else 0.0


def _diff_name_status(base: str, head: str) -> List[Tuple[str, str]]:
    """
    Zwraca listę (status, path) z git diff --name-status base..head
    Status przykładowo: 'A', 'M', 'D', 'R100', ...
    """
    out = _git(["diff", "--name-status", f"{base}..{head}"]).splitlines()
    items: List[Tuple[str, str]] = []
    for ln in out:
        parts = ln.split("\t")
        if len(parts) == 2:
            st, p = parts
        elif len(parts) == 3:
            st, _, p = parts
        else:
            st, p = parts[0], parts[-1]
        items.append((st.strip(), p.strip()))
    return items


def build_repo_mosaic(
    base: Optional[str] = None,
    head: str = "HEAD",
    include_unstaged: bool = False
) -> Tuple[RepoInfo, RepoMosaic, Dict[str, int]]:
    """
    Buduje RepoMosaic dla zakresu base..head.
    - base: gdy None → .glx/state.json:base_sha, w ostateczności merge-base(HEAD, origin/<branch>)
    - include_unstaged: jeśli True, traktuje 'git diff' jako HEAD + working tree (porównuje z HEAD)
    Zwraca: (RepoInfo, RepoMosaic, churn_counters)
    """
    root = repo_root()
    branch = current_branch() or "master"

    # HEAD sha
    try:
        head_sha = _git(["rev-parse", head]).strip()
    except subprocess.CalledProcessError:
        head_sha = head

    # base sha (state → merge-base)
    st = read_glx_state()
    base_sha = base or st.get("base_sha")
    if not base_sha:
        # spróbuj merge-base z origin/<branch>
        fallback = git_merge_base("HEAD", f"origin/{branch}")
        base_sha = fallback or head_sha  # w skrajnym razie (brak zdalnej gałęzi)
    base7 = rev_short(base_sha)
    head7 = rev_short(head_sha)

    # nazwy/statusy
    if include_unstaged:
        # porównanie HEAD..(working tree) – użyjemy 'git status --porcelain'
        st_out = _git(["status", "--porcelain"]).splitlines()
        # mapowanie na status 'M'/'A'/'D' jak najbliższe name-status
        ns: List[Tuple[str, str]] = []
        for ln in st_out:
            ln = ln.strip()
            if not ln:
                continue
            # format: XY path  (X=index, Y=worktree)
            # bierzemy Y (worktree) gdy różny od ' '
            if len(ln) < 4:
                continue
            X, Y, rest = ln[0], ln[1], ln[3:]
            status = (Y if Y != " " else X).strip()
            if status in {"M", "A", "D"}:
                ns.append((status, rest))
            else:
                # sprowadź np. 'R'/'C' itp. do „zmienione”
                ns.append((status, rest))
        name_status = ns
    else:
        name_status = _diff_name_status(base_sha, head_sha)

    # metryki ruchu
    churn = {"A": 0, "M": 0, "D": 0, "R": 0, "C": 0, "other": 0}
    files: List[str] = []
    edge: List[float] = []
    roi: List[float] = []

    for st_code, path in name_status:
        if not path:
            continue
        files.append(path)
        # status bazowy (pierwsza litera)
        sig = st_code[:1] if st_code else "M"
        if sig in churn:
            churn[sig] += 1
        else:
            churn["other"] += 1
        edge.append(_edge_score_for_path(path, st_code))
        roi.append(_roi_flag_for_path(path, st_code))

    # jeśli brak zmian – użyj śledzonych .py jako kafli „spokojnych”
    if not files:
        tracked = [p for p in list_tracked_files() if p.endswith(".py")]
        files = tracked[:36]  # nie przesadzaj z rozmiarem siatki
        edge = [0.35 for _ in files]
        roi = [0.0 for _ in files]

    rows, cols = _grid_for_n(len(files))
    repoM = RepoMosaic(
        files=files,
        edge=edge,
        roi=roi,
        layout_rows=rows,
        layout_cols=cols,
        extras={}
    )

    info = RepoInfo(
        root=root,
        branch=branch,
        base_sha=base_sha,
        head_sha=head_sha,
        base7=base7,
        head7=head7
    )
    return info, repoM, churn


# ──────────────────────────────────────────────────────────────────────────────
# Proste CLI do szybkich testów lokalnych
# ──────────────────────────────────────────────────────────────────────────────

def _cli(argv: Optional[List[str]] = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="git_io", description="Git I/O + RepoMosaic")
    p.add_argument("--base", type=str, default=None, help="base sha (domyślnie z .glx/state.json lub merge-base)")
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
