#!/usr/bin/env python
# -*- coding: utf-8 -*-
# glitchlab/.githooks/post-commit.py
from __future__ import annotations

import os, sys, re, json, zipfile, importlib.util, subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

os.environ.setdefault("PYTHONUTF8", "1")

# ── GLX_RUN (A/M/E/Z) ────────────────────────────────────────────────────────
FLAGS = {"A": 0x1, "M": 0x2, "E": 0x4, "Z": 0x8}
ORDER_CANON = ["A", "M", "E", "Z"]


# ── log/fail ─────────────────────────────────────────────────────────────────
def _log(msg: str) -> None: print(f"[GLX][post-commit] {msg}")


def _fail(msg: str, code: int = 2) -> None:
    print(f"[GLX][ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


# ── helpery parsujące ────────────────────────────────────────────────────────
def _strip_inline_comment(s: str) -> str:
    s = s or ""
    in_s = in_d = False
    out = []
    for ch in s:
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == '#' and not in_s and not in_d:
            break
        out.append(ch)
    return "".join(out).rstrip()


def _parse_glx_run(s: str) -> Tuple[int, List[str]]:
    s0 = _strip_inline_comment((s or "").strip())
    if not s0:
        _fail("GLX_RUN jest pusty. Podaj np. 'A', 'A+M', 'MEAZ', '0xD' lub '13'.")
    up = s0.upper().replace(" ", "")
    try:
        if up.startswith("0X"):
            mask = int(up, 16)
            order = [k for k in ORDER_CANON if FLAGS[k] & mask]
            if not mask:
                _fail("GLX_RUN=0x0 nie zawiera żadnych trybów.")
            return mask, order
        if up.isdigit():
            mask = int(up, 10)
            order = [k for k in ORDER_CANON if FLAGS[k] & mask]
            if not mask:
                _fail("GLX_RUN=0 nie zawiera żadnych trybów.")
            return mask, order
    except ValueError:
        pass
    if re.findall(r"[^AMEZ\s,+;/|.\-]", up):
        _fail("GLX_RUN zawiera niedozwolone znaki (dozwolone A,M,E,Z i separatory).")
    letters = re.findall(r"[AMEZ]", up)
    if not letters:
        _fail("GLX_RUN nie zawiera liter z zestawu [A M E Z].")
    order, seen = [], set()
    for ch in letters:
        if ch not in seen:
            seen.add(ch)
            order.append(ch)
    mask = 0
    for ch in order:
        mask |= FLAGS[ch]
    return mask, order


def _validate_mask(mask: int) -> None:
    if (mask & FLAGS["Z"]) and not (mask & (FLAGS["A"] | FLAGS["M"])):
        _fail("GLX_RUN zawiera Z bez A/M — ZIP audytu wymaga artefaktów.")


# ── .env loader (łańcuch przodków; .env.local nadpisuje) ─────────────────────
def _clean_value(v: str) -> str:
    v = _strip_inline_comment(v or "").strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1]
    return v.rstrip(":").strip()


def _parse_env_file(p: Path) -> Dict[str, str]:
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


def _dirs_chain_up(start: Path) -> List[Path]:
    cur = Path(start).resolve()
    chain_fwd: List[Path] = []
    while True:
        chain_fwd.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    chain = list(reversed(chain_fwd))
    seen, uniq = set(), []
    for p in chain:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _expand_vars_and_user(s: str) -> str:
    return os.path.expanduser(os.path.expandvars(s or ""))


def _resolve_path(val: str, base: Path) -> Path:
    v = _expand_vars_and_user(_clean_value(val))
    p = Path(v) if v else Path("")
    if not v:
        return base.resolve()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def _load_env(git_root: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for d in _dirs_chain_up(git_root):
        base, loc = d / ".env", d / ".env.local"
        if base.exists():
            env.update(_parse_env_file(base))
        if loc.exists():
            env.update(_parse_env_file(loc))
    if not env:
        _fail(f"Nie znaleziono .env ani .env.local w łańcuchu przodków od {git_root}")
    return env


def _norm_paths_in_env(env: Dict[str, str], git_root: Path) -> Dict[str, str]:
    raw_root = env.get("GLX_ROOT", "")
    if not raw_root:
        env["GLX_ROOT"] = str(git_root.resolve())
        _log(f"GLX_ROOT nie podany — używam git_root={env['GLX_ROOT']}")
    glx_root = Path(_expand_vars_and_user(env["GLX_ROOT"]))
    if not glx_root.is_absolute():
        _log(f"UWAGA: GLX_ROOT nie jest absolutny ('{glx_root}'), przeliczam względem git_root.")
        glx_root = (git_root / glx_root).resolve()
        env["GLX_ROOT"] = str(glx_root)
    for key in ("GLX_OUT", "GLX_AUTONOMY_OUT", "GLX_POLICY"):
        if key in env and str(env[key]).strip():
            env[key] = str(_resolve_path(env[key], glx_root))
    return env


# ── walidacje ────────────────────────────────────────────────────────────────
def _req(env: Dict[str, str], key: str) -> str:
    v = str(env.get(key, "")).strip()
    if not v:
        _fail(f"Wymagany klucz .env brakujący: {key}")
    return v


def _as_int(env: Dict[str, str], key: str, lo: int, hi: int) -> int:
    v = int(_req(env, key))
    if not (lo <= v <= hi):
        _fail(f"{key}={v} poza zakresem [{lo},{hi}]")
    return v


def _as_float01(env: Dict[str, str], key: str) -> float:
    v = float(_req(env, key))
    if not (0.0 <= v <= 1.0):
        _fail(f"{key}={v} poza zakresem [0,1]")
    return v


# ── git helpers ──────────────────────────────────────────────────────────────
def _git(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    # Upewnij się, że cwd to katalog (na Windows NotADirectoryError gdy wskażesz plik)
    dir_cwd = cwd if cwd.is_dir() else cwd.parent
    return subprocess.run(["git", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(dir_cwd))


def _git_top(cwd: Path) -> Optional[Path]:
    r = _git(["rev-parse", "--show-toplevel"], cwd)
    # Działa zarówno gdy przekażesz katalog jak i ścieżkę do pliku
    r = _git(["rev-parse", "--show-toplevel"], cwd if cwd.is_dir() else cwd.parent)
    return Path(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None


def _rev_ok(ref: str, cwd: Path) -> bool:
    r = subprocess.run(["git", "rev-parse", "--verify", ref], cwd=str(cwd), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
    return r.returncode == 0


def _rev_parse(ref: str, cwd: Path) -> str:
    r = _git(["rev-parse", ref], cwd)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else ref


def _empty_tree_oid(cwd: Path) -> str:
    r = subprocess.run(["git", "hash-object", "-t", "tree", "/dev/null"], cwd=str(cwd), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    return "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # SHA-1 empty tree


def _detect_base(git_root: Path, state_p: Path) -> str:
    if state_p.exists():
        try:
            data = json.loads(state_p.read_text(encoding="utf-8"))
            bs = str(data.get("base_sha") or "").strip()
            if bs and _rev_ok(bs, git_root):
                return bs
        except Exception as e:
            _log(f"state.json read warning: {e}")
    for br in ("origin/master", "origin/main"):
        r = _git(["merge-base", "HEAD", br], git_root)
        if r.returncode == 0 and r.stdout.strip():
            cand = r.stdout.strip()
            if _rev_ok(cand, git_root):
                return cand
    r = _git(["rev-parse", "HEAD~1"], git_root)
    if r.returncode == 0 and r.stdout.strip() and _rev_ok(r.stdout.strip(), git_root):
        return r.stdout.strip()
    et = _empty_tree_oid(git_root)
    _log(f"fallback BASE → empty-tree {et}")
    return et


# ── moduły & uruchamianie ────────────────────────────────────────────────────
def _module_exists(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def _module_cmd_from_candidates(candidates: List[str], root_for_files: Path) -> List[str]:
    for mod in candidates:
        if _module_exists(mod):
            return [sys.executable, "-m", mod]
    for mod in candidates:
        rel = Path(*mod.split("."))
        for cand in (root_for_files / f"{rel}.py", root_for_files / rel / "__init__.py"):
            if cand.exists():
                return [sys.executable, str(cand)]
    _fail("Nie znaleziono modułu ani skryptu dla: " + " | ".join(candidates))
    return []


def _redact_env(d: Dict[str, str]) -> Dict[str, str]:
    hints = ("PASS", "PASSWORD", "SECRET", "TOKEN", "KEY")
    return {k: ("******" if any(h in k.upper() for h in hints) else v) for k, v in d.items()}


# ── kroki ────────────────────────────────────────────────────────────────────
def _run_post_diff(here: Path, git_root: Path) -> None:
    """Best-effort odpalenie sąsiedniego pre/post-diff upgradera po commit'cie."""
    script = here.parent / "post-diff.py"
    if script.exists():
        try:
            _log(f"post-diff: {script}")
            subprocess.run([sys.executable, str(script)], cwd=str(git_root), check=False)
        except Exception as e:
            _log(f"post-diff warn: {e}")


def _run_gateway(env: Dict[str, str], git_root: Path, glx_root: Path) -> None:
    pkg = _req(env, "GLX_PKG")
    out_dir = Path(_req(env, "GLX_AUTONOMY_OUT"))
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _detect_base(git_root, glx_root / ".glx" / "state.json")
    head = _rev_parse("HEAD", git_root)
    cmd = _module_cmd_from_candidates(
        [f"{pkg}.analysis.autonomy.gateway", f"{pkg}.gui.analysis.autonomy.gateway", "analysis.autonomy.gateway"],
        glx_root
    ) + ["build", "--out", str(out_dir), "--base", base, "--head", head]
    _log("gateway: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(glx_root))
    if r.returncode != 0:
        _fail(f"gateway zakończył się kodem {r.returncode}")
    snap = _redact_env(env)
    (out_dir / "env.json").write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    # weryfikacja artefaktów
    must = {"pack.json", "pack.md", "prompt.json"}
    present = {p.name for p in out_dir.glob("*")}
    if "pack.json" not in present and "packz.json" in present:
        (out_dir / "pack.json").write_text((out_dir / "packz.json").read_text(encoding="utf-8"), encoding="utf-8")
        present.add("pack.json")
    missing = [fn for fn in must if fn not in present]
    if missing:
        _fail("Brak wymaganych artefaktów autonomii: " + ", ".join(missing))


def _run_mosaic(env: Dict[str, str], git_root: Path, glx_root: Path) -> None:
    pkg = _req(env, "GLX_PKG")
    mosaic = _req(env, "GLX_MOSAIC")
    rows = _as_int(env, "GLX_ROWS", 1, 512)
    cols = _as_int(env, "GLX_COLS", 1, 512)
    edge_thr = _as_float01(env, "GLX_EDGE_THR")
    psi = _as_float01(env, "GLX_DELTA")
    kappa = _as_float01(env, "GLX_KAPPA")
    phi = _req(env, "GLX_PHI").lower()
    policy = env.get("GLX_POLICY", "analysis/policy.json")

    base = _detect_base(git_root, glx_root / ".glx" / "state.json")

    candidates = [
        f"{pkg}.mosaic.hybrid_ast_mosaic",
        f"{pkg}.gui.mosaic.hybrid_ast_mosaic",
        "mosaic.hybrid_ast_mosaic",
        "gui.mosaic.hybrid_ast_mosaic",
    ]
    cmd: List[str] = _module_cmd_from_candidates(candidates, glx_root) + [
        "--mosaic", mosaic, "--rows", str(rows), "--cols", str(cols),
        "--edge-thr", str(edge_thr), "--kappa-ab", str(kappa),
        "--phi", phi, "--repo-root", str(git_root),
        "from-git-dump", "--base", base, "--head", "HEAD",
        "--delta", str(psi),
        "--out", str(Path(_req(env, "GLX_OUT"))),
        "--strict-artifacts",
    ]
    if phi == "policy":
        pol_p = Path(policy) if Path(policy).is_absolute() else (glx_root / policy)
        if not pol_p.exists():
            _fail(f"GLX_PHI=policy ale brak pliku polityki: {pol_p}")
        cmd.insert(-2, "--policy-file")
        cmd.insert(-2, str(pol_p))

    _log("mosaic: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(glx_root))
    if r.returncode != 0:
        _fail(f"mozaika zakończyła się kodem {r.returncode}")

    out_dir = Path(_req(env, "GLX_OUT"))
    found = list(out_dir.rglob("report.json")) + list(out_dir.rglob("mosaic_map.json")) + list(
        out_dir.rglob("summary.md"))
    if not found:
        listing = "\n".join(f" - {p.relative_to(glx_root)}" for p in sorted(out_dir.rglob("*")) if p.is_file())
        _fail("Brak artefaktów mozaiki w " + str(out_dir) + "\nPliki znalezione:\n" + (listing or "(pusto)"))


def _should_zip(p: Path) -> bool:
    rel = str(p).replace("\\", "/")
    return p.is_file() and "/.git/" not in rel and "/__pycache__/" not in rel


def _writestr_deterministic(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    zi = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
    zi.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(zi, data)


def _make_zip(env: Dict[str, str], glx_root: Path) -> None:
    out_dir = Path(_req(env, "GLX_OUT"))
    auto_dir = Path(_req(env, "GLX_AUTONOMY_OUT"))
    backups = glx_root / "backup"
    backups.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    zip_path = backups / f"AUDIT_{ts}.zip"

    roots: List[Path] = []
    if out_dir.exists():
        roots.append(out_dir)
    try:
        if auto_dir.exists() and auto_dir.resolve() not in out_dir.resolve().parents and auto_dir.resolve() != out_dir.resolve():
            roots.append(auto_dir)
    except Exception:
        if auto_dir.exists():
            roots.append(auto_dir)

    added: set = set()
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for root in roots:
            files = [p for p in root.rglob("*") if _should_zip(p)]
            files.sort(key=lambda p: str(p.relative_to(glx_root)).replace("\\", "/"))
            for p in files:
                arc = str(p.relative_to(glx_root)).replace("\\", "/")
                if arc in added:
                    continue
                added.add(arc)
                with p.open("rb") as f:
                    _writestr_deterministic(zf, arc, f.read())
        meta = {
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "glx_run": _req(env, "GLX_RUN"),
            "project": env.get("GLX_PROJECT", "?"),
            "branch": env.get("GLX_BRANCH", "?"),
            "ver": env.get("GLX_VER", "?"),
        }
        _writestr_deterministic(zf, "GLX_AUDIT_META.json",
                                json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"))
    _log(f"audit zip: {zip_path.name}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> int:
    here = Path(__file__).resolve()
    git_root = _git_top(here) or here.parents[2]
    if not (git_root / ".env").exists() and not (git_root / ".env.local").exists():
        _fail(f"Nie znaleziono .env ani .env.local w {git_root}")

    # Najpierw uzupełnij raport post-diff (best-effort)
    _run_post_diff(here, git_root)

    env = _load_env(git_root)
    env = _norm_paths_in_env(env, git_root)

    mask, order = _parse_glx_run(env.get("GLX_RUN", ""))
    _validate_mask(mask)

    # bazowe wymagania
    glx_root = Path(_req(env, "GLX_ROOT")).resolve()
    if not glx_root.exists():
        _fail(f"GLX_ROOT wskazuje na nieistniejący katalog: {glx_root}")
    for k in ("GLX_PKG", "GLX_OUT", "GLX_AUTONOMY_OUT"):
        _req(env, k)

    # PYTHONPATH (importy modułów z GLX_ROOT)
    pkg = _req(env, "GLX_PKG")
    import_root = glx_root if (glx_root / pkg).is_dir() else (
        glx_root.parent if glx_root.name == pkg and glx_root.parent.exists() else glx_root)
    for p in (str(import_root), str(glx_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    _log(f"git_root={git_root}  glx_root={glx_root}  import_root={import_root}")

    # dispatcher w KOLEJNOŚCI z GLX_RUN
    DISPATCH = {
        "A": lambda: (_log("run:A"), _run_gateway(env, git_root, glx_root)),
        "M": lambda: (_log("run:M"), _run_mosaic(env, git_root, glx_root)),
        "E": lambda: (_log("run:E"), None),  # opcjonalny krok e-mail (tu puste)
        "Z": lambda: (_log("run:Z"), _make_zip(env, glx_root)),
    }
    for ch in order:
        DISPATCH[ch]()
    _log(f"OK (mask=0x{mask:X}, order={''.join(order)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
