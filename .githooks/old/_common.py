#!/usr/bin/env python
# glitchlab/.githooks/_common.py
# -*- coding: utf-8 -*-
# Python 3.9+
"""
Wspólne utilsy dla hooków GLX (pre-/prepare-/post-commit).

Zakres:
- log/fail
- git utils (repo top, rev-parse/verify, merge-base heurystyka BASE, pusty tree OID)
- loader .env (wycina komentarze inline, obsługa cudzysłowów, ~, $VAR/%VAR%)
- normalizacja ścieżek względem GLX_ROOT
- parser GLX_RUN (A/M/E/Z + wartości liczbowo: 0x.. lub dziesiętnie) + sanity
- uruchamianie modułów (prefer -m, fallback do pliku)
- redakcja sekretów, zapis JSON, prosty ZIP audytu z deterministycznym timestampem
- staged_files (lista plików z indeksu)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# wymuś UTF-8 (Windows)
os.environ.setdefault("PYTHONUTF8", "1")

# ──────────────────────────────────────────────────────────────────────────────
# Log & fail
# ──────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[GLX][hook] {msg}")


def fail(msg: str, code: int = 1) -> "NoReturn":  # type: ignore[name-defined]
    print(f"[GLX][ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


# ──────────────────────────────────────────────────────────────────────────────
# Git utils
# ──────────────────────────────────────────────────────────────────────────────

def git(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd),
    )


def git_top(start: Path) -> Optional[Path]:
    """
    Zwraca git toplevel dla katalogu start (jeśli start to plik → użyj start.parent).
    """
    start_dir = start if start.is_dir() else start.parent
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())
    except Exception:
        pass
    return None


def _scan_for_git_dir(start_dir: Path) -> Optional[Path]:
    """
    Szuka katalogu z `.git` idąc po przodkach, włącznie ze start_dir.
    """
    for d in [start_dir, *start_dir.parents]:
        if (d / ".git").exists():
            return d
    return None


def repo_top_from_here(here: Path) -> Path:
    """
    Solidne wykrycie repo-top:
      1) git_top(here lub here.parent)
      2) skanowanie przodków w poszukiwaniu `.git`
      3) fallback: jeśli jesteśmy w `<repo>/.githooks/...` → zwróć `<repo>`
      4) w ostateczności: katalog skryptu
    """
    here = Path(here).resolve()
    start_dir = here if here.is_dir() else here.parent

    top = git_top(start_dir)
    if top:
        return top

    scanned = _scan_for_git_dir(start_dir)
    if scanned:
        return scanned

    # klasyczny układ: <repo>/.githooks/<plik>.py
    if start_dir.name == ".githooks" and start_dir.parent.exists():
        return start_dir.parent

    return start_dir


def rev_ok(ref: str, cwd: Path) -> bool:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return r.returncode == 0
    except Exception:
        return False


def rev_parse(ref: str, cwd: Path) -> str:
    r = git(["rev-parse", ref], cwd)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else ref


def empty_tree_oid(cwd: Path) -> str:
    """
    Zwraca OID pustego drzewa dla aktualnej konfiguracji repo (SHA-1/SHA-256).
    Fallback do znanego SHA-1 jeśli polecenie się nie powiedzie.
    """
    try:
        r = subprocess.run(
            ["git", "hash-object", "-t", "tree", "/dev/null"],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    # SHA-1 dla pustego drzewa:
    return "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def detect_base(cwd: Path, state_json: Path) -> str:
    """
    Heurystyka wyboru BASE:
      1) .glx/state.json.base_sha
      2) merge-base z origin/master|main
      3) HEAD~1
      4) pusty tree OID (pierwszy commit)
    """
    # 1) state.json
    if state_json.exists():
        try:
            data = json.loads(state_json.read_text(encoding="utf-8"))
            bs = str(data.get("base_sha") or "").strip()
            if bs and rev_ok(bs, cwd):
                return bs
        except Exception as e:
            log(f"state.json read warning: {e}")

    # 2) merge-base
    for br in ("origin/master", "origin/main"):
        r = git(["merge-base", "HEAD", br], cwd)
        if r.returncode == 0 and r.stdout.strip():
            cand = r.stdout.strip()
            if rev_ok(cand, cwd):
                return cand

    # 3) HEAD~1
    r = git(["rev-parse", "HEAD~1"], cwd)
    if r.returncode == 0 and r.stdout.strip() and rev_ok(r.stdout.strip(), cwd):
        return r.stdout.strip()

    # 4) empty-tree
    et = empty_tree_oid(cwd)
    log(f"fallback BASE → empty-tree {et}")
    return et


def staged_files(repo_root: Path) -> List[Path]:
    """
    Zwraca listę plików z indeksu (`git diff --cached --name-only`).
    Zawiera wyłącznie istniejące w FS (np. rename-źródła mogą nie istnieć).
    """
    r = git(["diff", "--cached", "--name-only"], repo_root)
    if r.returncode != 0:
        # spróbuj jeszcze raz z recalcem repo_top (na wypadek złej ścieżki)
        fixed_root = repo_top_from_here(repo_root)
        if fixed_root != repo_root:
            r2 = git(["diff", "--cached", "--name-only"], fixed_root)
            if r2.returncode == 0:
                return [
                    (fixed_root / ln.strip()).resolve()
                    for ln in r2.stdout.splitlines()
                    if ln.strip() and (fixed_root / ln.strip()).exists()
                ]
        err = (r.stderr or "").strip()
        fail("Nie mogę pobrać listy staged plików (git diff --cached --name-only)."
             + (f" Git stderr: {err}" if err else ""))

    files: List[Path] = []
    for ln in r.stdout.splitlines():
        p = (repo_root / ln.strip()).resolve()
        if p.exists():
            files.append(p)
    return files


# ──────────────────────────────────────────────────────────────────────────────
# Filesystem helpers
# ──────────────────────────────────────────────────────────────────────────────

def dirs_chain_up(start: Path) -> List[Path]:
    """
    Lista katalogów od korzenia dysku do `start` (włącznie); stabilna i bez duplikatów.
    """
    cur = Path(start).resolve()
    chain_fwd: List[Path] = []
    while True:
        chain_fwd.append(cur)
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    chain = list(reversed(chain_fwd))
    seen = set()
    uniq: List[Path] = []
    for p in chain:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


# ──────────────────────────────────────────────────────────────────────────────
# .env loader & path normalization
# ──────────────────────────────────────────────────────────────────────────────

def _strip_inline_comment(s: str) -> str:
    """
    Usuwa #… tylko poza cudzysłowami.
    """
    s = s or ""
    in_s = in_d = False
    out: List[str] = []
    for ch in s:
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            break
        out.append(ch)
    return "".join(out).rstrip()


def _clean_value(v: str) -> str:
    v = _strip_inline_comment(v or "").strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1]
    return v.rstrip(":").strip()


def parse_env_file(p: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not p.exists():
        return out
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = _clean_value(v)
    return out


def load_env(repo_root: Path) -> Dict[str, str]:
    """
    Ładuje .env/.env.local z przodków repo_root:
      - warstwy: root → … → repo_root
      - kolejność w warstwie: .env, potem .env.local (local nadpisuje .env)
    """
    env: Dict[str, str] = {}
    for d in dirs_chain_up(repo_root):
        base = d / ".env"
        loc = d / ".env.local"
        if base.exists():
            env.update(parse_env_file(base))
        if loc.exists():
            env.update(parse_env_file(loc))
    if not env:
        fail(f"Nie znaleziono .env ani .env.local w łańcuchu przodków od {repo_root}", 2)
    return env


def _expand_vars_and_user(s: str) -> str:
    return os.path.expanduser(os.path.expandvars(s or ""))


def resolve_path(val: str, base: Path) -> Path:
    """
    Ścieżki względne liczymy względem `base` (GLX_ROOT).
    """
    v = _expand_vars_and_user(_clean_value(val))
    p = Path(v) if v else Path("")
    if not v:
        return base.resolve()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def norm_paths_in_env(env: Dict[str, str], repo_root: Path) -> Dict[str, str]:
    """
    GLX_ROOT preferencyjnie absolutny. Jeśli brak → repo_root.
    Pozostałe ścieżki (GLX_OUT/GLX_AUTONOMY_OUT/GLX_POLICY) normowane do GLX_ROOT.
    """
    raw_root = env.get("GLX_ROOT", "")
    if not raw_root:
        env["GLX_ROOT"] = str(repo_root.resolve())
        log(f"GLX_ROOT nie podany w .env — używam repo_root={env['GLX_ROOT']}")
    glx_root_path = Path(_expand_vars_and_user(env["GLX_ROOT"]))
    if not glx_root_path.is_absolute():
        log(f"UWAGA: GLX_ROOT nie jest absolutny ('{glx_root_path}'), przeliczam względem repo_root.")
        glx_root_path = (repo_root / glx_root_path).resolve()
        env["GLX_ROOT"] = str(glx_root_path)

    for key in ("GLX_OUT", "GLX_AUTONOMY_OUT", "GLX_POLICY"):
        if key in env and str(env[key]).strip():
            env[key] = str(resolve_path(env[key], glx_root_path))
    return env


# ──────────────────────────────────────────────────────────────────────────────
# GLX_RUN
# ──────────────────────────────────────────────────────────────────────────────

FLAGS: Dict[str, int] = {"A": 0x1, "M": 0x2, "E": 0x4, "Z": 0x8}
ORDER_CANON: List[str] = ["A", "M", "E", "Z"]


def parse_glx_run(s: str) -> Tuple[int, List[str]]:
    """
    Obsługuje:
    - litery (z separatorami): 'A+M|Z', 'amez', 'MEAZ'
    - liczby: '0xF', '15'
    Zwraca (mask, order), gdzie 'order' to KOLEJNOŚĆ bez duplikatów
    (dla postaci liczbowych — kolejność kanoniczna ORDER_CANON).
    """
    s0 = _strip_inline_comment((s or "").strip())
    if not s0:
        fail("GLX_RUN jest pusty. Podaj np. 'A', 'A+M', 'MEAZ', '0xD' lub '13'.", 2)

    up = s0.upper().replace(" ", "")

    # 1) Hex / decimal
    try:
        if up.startswith("0X"):
            mask = int(up, 16)
            order = [k for k in ORDER_CANON if FLAGS[k] & mask]
            if not mask:
                fail("GLX_RUN=0x0 nie zawiera żadnych trybów.", 2)
            return mask, order
        if up.isdigit():
            mask = int(up, 10)
            order = [k for k in ORDER_CANON if FLAGS[k] & mask]
            if not mask:
                fail("GLX_RUN=0 nie zawiera żadnych trybów.", 2)
            return mask, order
    except ValueError:
        pass

    # 2) Literki z separatorami
    if re.findall(r"[^AMEZ\s,+;/|.\-]", up):
        fail("GLX_RUN zawiera niedozwolone znaki (dozwolone A,M,E,Z i separatory).", 2)

    letters = re.findall(r"[AMEZ]", up)
    if not letters:
        fail("GLX_RUN nie zawiera liter z zestawu [A M E Z].", 2)

    order, seen = [], set()
    for ch in letters:
        if ch not in seen:
            seen.add(ch)
            order.append(ch)

    mask = 0
    for ch in order:
        mask |= FLAGS[ch]
    return mask, order


def validate_glx_mask(mask: int) -> None:
    """
    Minimalna sanity: Z wymaga A lub M (ZIP musi mieć co archiwizować).
    """
    if (mask & FLAGS["Z"]) and not (mask & (FLAGS["A"] | FLAGS["M"])):
        fail("GLX_RUN zawiera Z bez A/M — ZIP audytu wymaga artefaktów z Autonomii lub Mozaiki.", 2)


# ──────────────────────────────────────────────────────────────────────────────
# Module runner
# ──────────────────────────────────────────────────────────────────────────────

def module_exists(mod: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def module_cmd_from_candidates(candidates: Sequence[str], project_root: Path) -> List[str]:
    """
    Zwraca polecenie uruchomienia modułu:
      - preferuje `python -m <mod>` gdy importowalny,
      - fallback do pliku w repo: <mod>.py albo <mod>/__init__.py
    """
    # prefer -m
    for mod in candidates:
        if module_exists(mod):
            return [sys.executable, "-m", mod]

    # fallback: plik
    for mod in candidates:
        rel = Path(*mod.split("."))
        for cand in (project_root / f"{rel}.py", project_root / rel / "__init__.py"):
            if cand.exists():
                return [sys.executable, str(cand)]

    fail("Nie znaleziono modułu ani skryptu dla: " + " | ".join(candidates), 2)
    return []  # unreachable


# ──────────────────────────────────────────────────────────────────────────────
# Secrets redaction & JSON helpers
# ──────────────────────────────────────────────────────────────────────────────

_SECRET_HINTS = ("PASS", "PASSWORD", "SECRET", "TOKEN", "KEY")


def redact_env(d: Dict[str, str]) -> Dict[str, str]:
    red: Dict[str, str] = {}
    for k, v in d.items():
        if any(h in k.upper() for h in _SECRET_HINTS):
            red[k] = "******"
        else:
            red[k] = v
    return red


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


# ──────────────────────────────────────────────────────────────────────────────
# ZIP audytu (deterministyczny)
# ──────────────────────────────────────────────────────────────────────────────

def _should_zip(p: Path) -> bool:
    rel = str(p).replace("\\", "/")
    if "/.git/" in rel or "/__pycache__/" in rel:
        return False
    return p.is_file()


def _writestr_deterministic(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    # stały timestamp (dla stabilnych hashy ZIP)
    zi = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
    zi.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(zi, data)


def make_audit_zip(repo_root: Path, roots: Iterable[Path], meta: Dict[str, Any]) -> Path:
    """
    Tworzy `backup/AUDIT_<UTCts>.zip` z plikami z podanych katalogów (rekurencyjnie),
    z pominięciem .git/__pycache__. Dodaje GLX_AUDIT_META.json na końcu archiwum.
    """
    backups = repo_root / "backup"
    backups.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    zip_path = backups / f"AUDIT_{ts}.zip"

    added: set = set()  # posix-arcname dedupe
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for root in roots:
            if not root.exists():
                continue
            files = [p for p in root.rglob("*") if _should_zip(p)]
            files.sort(key=lambda p: str(p.relative_to(repo_root)).replace("\\", "/"))
            for p in files:
                arc = str(p.relative_to(repo_root)).replace("\\", "/")
                if arc in added:
                    continue
                added.add(arc)
                with p.open("rb") as f:
                    _writestr_deterministic(zf, arc, f.read())

        meta_obj = {
            **meta,
            "created_utc": datetime.utcnow().isoformat() + "Z",
        }
        _writestr_deterministic(
            zf,
            "GLX_AUDIT_META.json",
            json.dumps(meta_obj, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    log(f"audit zip: {zip_path.name}")
    return zip_path


# ──────────────────────────────────────────────────────────────────────────────
# Walidacje pomocnicze
# ──────────────────────────────────────────────────────────────────────────────

def req(env: Dict[str, str], key: str) -> str:
    v = str(env.get(key, "")).strip()
    if not v:
        fail(f"Wymagany klucz .env brakujący: {key}", 2)
    return v


def as_int(env: Dict[str, str], key: str, lo: int, hi: int) -> int:
    v = int(req(env, key))
    if not (lo <= v <= hi):
        fail(f"{key}={v} poza zakresem [{lo},{hi}]", 2)
    return v


def as_float01(env: Dict[str, str], key: str) -> float:
    v = float(req(env, key))
    if not (0.0 <= v <= 1.0):
        fail(f"{key}={v} poza zakresem [0,1]", 2)
    return v


# ──────────────────────────────────────────────────────────────────────────────
# Pomocnicze: PYTHONPATH dla importów projektu
# ──────────────────────────────────────────────────────────────────────────────

def ensure_import_roots(env: Dict[str, str]) -> Tuple[Path, Path]:
    """
    Zwraca (import_root, glx_root) i ustawia je na sys.path (jeśli nie dodane).
    - Jeśli GLX_ROOT zawiera folder pakietu GLX_PKG → import_root = GLX_ROOT
    - Jeśli GLX_ROOT to sam folder pakietu → import_root = parent
    - Inaczej: import_root = GLX_ROOT
    """
    glx_root = Path(env["GLX_ROOT"]).resolve()
    pkg = req(env, "GLX_PKG")

    if (glx_root / pkg).is_dir():
        import_root = glx_root
    elif glx_root.name == pkg and glx_root.parent.exists():
        import_root = glx_root.parent
    else:
        import_root = glx_root

    for p in (str(import_root), str(glx_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    return import_root, glx_root
