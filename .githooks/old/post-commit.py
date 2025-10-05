#!/usr/bin/env python
# -*- coding: utf-8 -*-
# glitchlab/.githooks/post-commit.py
"""
GLX-CTX:v2
component: tools.git.hooks.post_commit
role: post-commit driver (A/M/E/Z); korzysta z .env *wyłącznie* z katalogu PROJEKTU (git_root.parent)

# MOSAIC::PROFILE (S/H/Z)
S: fs_roots,env_loader,git_helpers,validators
H: analysis.autonomy.gateway,mosaic.hybrid_ast_mosaic,zip.audit
Z: 1

# AST::SURFACE
imports: os,sys,re,json,zipfile,importlib.util,subprocess,datetime,pathlib,typing
public_api: main,_dispatch,_run_gateway,_run_mosaic,_make_zip

# CONTRACTS::EVENTS
publish:
  run.start: ctx:str
  run.progress: value:float[0,1],text:str
  run.done: artifacts:list
  run.error: error:str,ctx:str

# INVARIANTS
- env_from_project_dir_only
- glx_root_equals_project_dir
- paths_anchor_git_only
- outputs_inside_git_repo
- fundamental_env_present
- no_fallback_env

# DATA::SOURCES
env: <project_dir>/.env | .env.local
repo: <git_root>/.git
state: <glx_root>/.glx/state.json

# DATA::SINKS
out: GLX_OUT (w repo)
autonomy: GLX_AUTONOMY_OUT (w repo)
audit_zip: <git_root>/backup/AUDIT_*.zip

# GRAMMAR::HOOKS  (komentarze #glx:event=… → Δ)
enter_scope/exit_scope, define/use, link, bucket_jump, reassign, contract/expand

# TAG-SCHEMA
- # glx:ast.fn=<Name>
- # glx:mosaic.S=<csv>
- # glx:mosaic.H=<csv>
- # glx:contracts.publish=<csv>
- # glx:data.in=<csv>   | # glx:data.out=<csv>
- # glx:event=<kind[:args]>   # kind∈{enter_scope,exit_scope,define,use,link,bucket_jump,reassign,contract,expand}

# PARSER
- listy w tagach: CSV (bez spacji); wartości surowe; klucz=wartość
- kompatybilność: JSON później; aktualny parser: CSV
"""

# --- HEADER::GUIDE (krótka instrukcja metanagłówków) -------------------------
# 1) Zaczynaj od GLX-CTX/component/role, potem: MOSAIC::PROFILE, AST::SURFACE,
#    CONTRACTS::EVENTS, INVARIANTS, DATA::SOURCES/DATA::SINKS, GRAMMAR::HOOKS, TAG-SCHEMA, PARSER.
# 2) Publiczne funkcje oznaczaj:  # glx:ast.fn=... (+ ewentualne S/H, data.in/out).
# 3) Zakresy:  # glx:event=enter_scope:<name>  /  # glx:event=exit_scope

from __future__ import annotations

# glx:event=enter_scope:module_imports
import os, sys, re, json, zipfile, importlib.util, subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# glx:event=exit_scope

os.environ.setdefault("PYTHONUTF8", "1")

# ── GLX_RUN (A/M/E/Z) ────────────────────────────────────────────────────────
FLAGS = {"A": 0x1, "M": 0x2, "E": 0x4, "Z": 0x8}
ORDER_CANON = ["A", "M", "E", "Z"]


# ── log/fail ─────────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    # glx:ast.fn=_log
    print(f"[GLX][post-commit] {msg}")


def _fail(msg: str, code: int = 2) -> None:
    # glx:ast.fn=_fail
    print(f"[GLX][ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


# ── drobne utils / parsery ───────────────────────────────────────────────────
def _strip_inline_comment(s: str) -> str:
    # glx:ast.fn=_strip_inline_comment
    # glx:event=enter_scope:_strip_inline_comment
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
    # glx:event=exit_scope
    return "".join(out).rstrip()


def _parse_glx_run(s: str) -> Tuple[int, List[str]]:
    # glx:ast.fn=_parse_glx_run
    # glx:event=enter_scope:_parse_glx_run
    s0 = _strip_inline_comment((s or "").strip())
    if not s0:
        _fail("GLX_RUN jest pusty. Podaj np. 'A', 'A+M', 'MEAZ', '0xD' lub '13'.")
    up = s0.upper().replace(" ", "")
    try:
        if up.startswith("0X"):
            mask = int(up, 16)
            order = [k for k in ORDER_CANON if FLAGS[k] & mask]
        elif up.isdigit():
            mask = int(up, 10)
            order = [k for k in ORDER_CANON if FLAGS[k] & mask]
        else:
            raise ValueError
        if not mask:
            _fail("GLX_RUN=0/0x0 nie zawiera żadnych trybów.")
        # glx:event=exit_scope
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
    # glx:event=exit_scope
    return mask, order


def _validate_mask(mask: int) -> None:
    # glx:ast.fn=_validate_mask
    if (mask & FLAGS["Z"]) and not (mask & (FLAGS["A"] | FLAGS["M"])):
        _fail("GLX_RUN zawiera Z bez A/M — ZIP audytu wymaga artefaktów.")


# ── .env loader — WYŁĄCZNIE katalog PROJEKTU (git_root.parent) ──────────────
def _clean_value(v: str) -> str:
    # glx:ast.fn=_clean_value
    v = _strip_inline_comment(v or "").strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1]
    return v.rstrip(":").strip()


# glx:ast.fn=_parse_env_file
# glx:mosaic.S=file_read,line_parse
# glx:data.in=path:.env
# glx:data.out=dict[str,str]
def _parse_env_file(p: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not p.exists():
        return out
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # opcjonalne wsparcie dla 'export KEY=VALUE'
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if not k:
            continue
        out[k] = _clean_value(v)
    return out


def _expand_vars_and_user(s: str) -> str:
    # glx:ast.fn=_expand_vars_and_user
    return os.path.expanduser(os.path.expandvars(s or ""))


def _resolve_path(val: str, base: Path) -> Path:
    # glx:ast.fn=_resolve_path
    v = _expand_vars_and_user(_clean_value(val))
    p = Path(v) if v else Path("")
    if not v:
        return base.resolve()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def _load_env_project_only(git_root: Path) -> Dict[str, str]:
    """
    Ładuje .env/.env.local *wyłącznie* z katalogu PROJEKTU (git_root.parent).
    Brak → twardy błąd (odmowa działania).
    """
    # glx:ast.fn=_load_env_project_only
    # glx:event=enter_scope:_load_env_project_only
    env: Dict[str, str] = {}
    project_dir = git_root.parent
    proj_env = project_dir / ".env"
    proj_local = project_dir / ".env.local"
    if proj_env.exists():
        env.update(_parse_env_file(proj_env))
    if proj_local.exists():
        env.update(_parse_env_file(proj_local))
    if not env:
        _fail(f"ENV: oczekuję .env/.env.local w katalogu projektu: {project_dir}")
    _log(f"ENV: używam plików z katalogu projektu: {project_dir}")
    # glx:event=exit_scope
    return env


# ── walidacje / inwarianty ENV ───────────────────────────────────────────────
def _req(env: Dict[str, str], key: str) -> str:
    # glx:ast.fn=_req
    v = str(env.get(key, "")).strip()
    if not v:
        _fail(f"Wymagany klucz .env brakujący: {key}")
    return v


def _as_int(env: Dict[str, str], key: str, lo: int, hi: int) -> int:
    # glx:ast.fn=_as_int
    v = int(_req(env, key))
    if not (lo <= v <= hi):
        _fail(f"{key}={v} poza zakresem [{lo},{hi}]")
    return v


def _as_float01(env: Dict[str, str], key: str) -> float:
    # glx:ast.fn=_as_float01
    v = float(_req(env, key))
    if not (0.0 <= v <= 1.0):
        _fail(f"{key}={v} poza zakresem [0,1]")
    return v


def _is_subpath(child: Path, parent: Path) -> bool:
    # glx:ast.fn=_is_subpath
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _norm_paths_in_env(env: Dict[str, str], git_root: Path) -> Dict[str, str]:
    """
    TWARDY reżim:
      • GLX_ROOT musi wskazywać katalog PROJEKTU (git_root.parent)
      • GLX_PATH_ANCHOR == 'git' (inne wartości → błąd)
      • GLX_OUT i GLX_AUTONOMY_OUT muszą znajdować się *wewnątrz* repo (git_root)
    """
    # glx:ast.fn=_norm_paths_in_env
    # glx:event=enter_scope:_norm_paths_in_env
    project_dir = git_root.parent.resolve()

    # GLX_ROOT → musi = katalog projektu
    raw_root = env.get("GLX_ROOT", "")
    if not raw_root:
        env["GLX_ROOT"] = str(project_dir)
    glx_root = Path(_expand_vars_and_user(env["GLX_ROOT"])).resolve()
    if glx_root != project_dir:
        _fail(f"GLX_ROOT={glx_root} ≠ katalog projektu={project_dir} — przerwano.")

    # Kotwica ścieżek: tylko 'git'
    anchor = (env.get("GLX_PATH_ANCHOR") or "git").strip().lower()
    if anchor != "git":
        _fail(f"GLX_PATH_ANCHOR='{anchor}' niedozwolone. Dozwolone wyłącznie: 'git'.")

    # Rozwiąż ścieżki względem git_root i *sprawdź* położenie
    for key in ("GLX_OUT", "GLX_AUTONOMY_OUT", "GLX_POLICY"):
        if key in env and str(env[key]).strip():
            resolved = _resolve_path(env[key], git_root)
            if key in ("GLX_OUT", "GLX_AUTONOMY_OUT") and not _is_subpath(resolved, git_root):
                _fail(f"{key}={resolved} wychodzi poza repo ({git_root}) — przerwano.")
            env[key] = str(resolved)
    # glx:event=exit_scope
    return env


def _enforce_env_invariants(env: Dict[str, str]) -> None:
    """
    Fundamentalne zmienne — brak = twardy błąd:
      • GLX_RUN, GLX_PKG, GLX_OUT, GLX_AUTONOMY_OUT
    Dodatkowo sanity dla mozaiki, jeśli w GLX_RUN występuje 'M'.
    """
    # glx:ast.fn=_enforce_env_invariants
    # glx:event=enter_scope:_enforce_env_invariants
    for k in ("GLX_RUN", "GLX_PKG", "GLX_OUT", "GLX_AUTONOMY_OUT"):
        _req(env, k)
    if "M" in _parse_glx_run(env.get("GLX_RUN", ""))[1]:
        _as_int(env, "GLX_ROWS", 1, 2048)
        _as_int(env, "GLX_COLS", 1, 2048)
        _as_float01(env, "GLX_EDGE_THR")
        _as_float01(env, "GLX_DELTA")
        _as_float01(env, "GLX_KAPPA")
        _req(env, "GLX_PHI")
    # glx:event=exit_scope


# ── PY runner helpers (ważne: cwd=git_root + PYTHONPATH=glx_root,git_root) ──
def _py_env_with_path(glx_root: Path, git_root: Path) -> Dict[str, str]:
    # glx:ast.fn=_py_env_with_path
    env2 = dict(os.environ)
    parts = [str(glx_root), str(git_root)]
    if env2.get("PYTHONPATH"):
        env2["PYTHONPATH"] = os.pathsep.join(parts + [env2["PYTHONPATH"]])
    else:
        env2["PYTHONPATH"] = os.pathsep.join(parts)
    return env2


def _run_py(cmd: List[str], git_root: Path, glx_root: Path, what: str) -> int:
    # glx:ast.fn=_run_py
    _log(f"{what}: {' '.join(cmd)} [cwd={git_root}]")
    r = subprocess.run(cmd, cwd=str(git_root), env=_py_env_with_path(glx_root, git_root))
    return r.returncode


# ── git helpers ──────────────────────────────────────────────────────────────
def _git(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    # glx:ast.fn=_git
    d = cwd if cwd.is_dir() else cwd.parent
    return subprocess.run(["git", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(d))


def _rev_ok(ref: str, cwd: Path) -> bool:
    # glx:ast.fn=_rev_ok
    r = subprocess.run(["git", "rev-parse", "--verify", ref], cwd=str(cwd),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.returncode == 0


def _rev_parse(ref: str, cwd: Path) -> str:
    # glx:ast.fn=_rev_parse
    r = _git(["rev-parse", ref], cwd)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else ref


def _empty_tree_oid(cwd: Path) -> str:
    # glx:ast.fn=_empty_tree_oid
    r = subprocess.run(["git", "hash-object", "-t", "tree", "/dev/null"], cwd=str(cwd),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def _detect_base(git_root: Path, state_p: Path) -> str:
    # glx:ast.fn=_detect_base
    # glx:event=enter_scope:_detect_base
    if state_p.exists():
        try:
            data = json.loads(state_p.read_text(encoding="utf-8"))
            bs = str(data.get("base_sha") or "").strip()
            if bs and _rev_ok(bs, git_root):
                # glx:event=exit_scope
                return bs
        except Exception as e:
            _log(f"state.json read warning: {e}")
    for br in ("origin/master",):
        r = _git(["merge-base", "HEAD", br], git_root)
        if r.returncode == 0 and r.stdout.strip():
            cand = r.stdout.strip()
            if _rev_ok(cand, git_root):
                # glx:event=exit_scope
                return cand
    r = _git(["rev-parse", "HEAD~1"], git_root)
    if r.returncode == 0 and r.stdout.strip() and _rev_ok(r.stdout.strip(), git_root):
        # glx:event=exit_scope
        return r.stdout.strip()
    et = _empty_tree_oid(git_root)
    _log(f"fallback BASE → empty-tree {et}")
    # glx:event=exit_scope
    return et


# ── moduły & uruchamianie (helpery) ──────────────────────────────────────────
def _module_exists(mod: str) -> bool:
    # glx:ast.fn=_module_exists
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def _module_cmd_from_candidates(candidates: List[str], root_for_files: Path) -> List[str]:
    # glx:ast.fn=_module_cmd_from_candidates
    # glx:event=enter_scope:_module_cmd_from_candidates
    for mod in candidates:
        if _module_exists(mod):
            # glx:event=exit_scope
            return [sys.executable, "-m", mod]
    for mod in candidates:
        rel = Path(*mod.split("."))
        for cand in (root_for_files / f"{rel}.py", root_for_files / rel / "__init__.py"):
            if cand.exists():
                # glx:event=exit_scope
                return [sys.executable, str(cand)]
    _fail("Nie znaleziono modułu ani skryptu dla: " + " | ".join(candidates))
    # glx:event=exit_scope
    return []


def _redact_env(d: Dict[str, str]) -> Dict[str, str]:
    # glx:ast.fn=_redact_env
    hints = ("PASS", "PASSWORD", "SECRET", "TOKEN", "KEY")
    return {k: ("******" if any(h in k.upper() for h in hints) else v) for k, v in d.items()}


# ── kroki: post-diff / gateway / mosaic / zip ────────────────────────────────

# glx:ast.fn=_run_post_diff
# glx:mosaic.S=fs_probe
# glx:mosaic.H=.githooks/post-diff.py
# glx:data.in=here,git_root
def _run_post_diff(here: Path, git_root: Path) -> None:
    """Best-effort odpalenie sąsiedniego pre/post-diff upgradera po commit'cie."""
    script = here.parent / "post-diff.py"
    if script.exists():
        try:
            _log(f"post-diff: {script}")
            subprocess.run([sys.executable, str(script)], cwd=str(git_root), check=False)
        except Exception as e:
            _log(f"post-diff warn: {e}")


# glx:ast.fn=_run_gateway
# glx:mosaic.H=analysis.autonomy.gateway
# glx:contracts.publish=run.start,run.done,run.error
# glx:data.in=GLX_AUTONOMY_OUT
# glx:data.out=pack.json,pack.md,prompt.json,env.json
def _run_gateway(env: Dict[str, str], git_root: Path, glx_root: Path) -> None:
    pkg = _req(env, "GLX_PKG")
    out_dir = Path(_req(env, "GLX_AUTONOMY_OUT"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = _module_cmd_from_candidates(
        [f"{pkg}.analysis.autonomy.gateway",
         f"{pkg}.gui.analysis.autonomy.gateway",
         "analysis.autonomy.gateway"],
        glx_root
    ) + ["build", "--out", str(out_dir)]

    rc = _run_py(cmd, git_root, glx_root, "gateway")
    if rc != 0:
        _fail(f"gateway zakończył się kodem {rc}")

    # Artefakty + migawka ENV (zredagowana)
    snap = _redact_env(env)
    (out_dir / "env.json").write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")

    must = {"pack.json", "pack.md", "prompt.json"}
    present = {p.name for p in out_dir.glob("*")}
    if "pack.json" not in present and "packz.json" in present:
        (out_dir / "pack.json").write_text((out_dir / "packz.json").read_text(encoding="utf-8"), encoding="utf-8")
        present.add("pack.json")
    missing = [fn for fn in must if fn not in present]
    if missing:
        _fail("Brak wymaganych artefaktów autonomii: " + ", ".join(missing))


# glx:ast.fn=_run_mosaic
# glx:mosaic.H=mosaic.hybrid_ast_mosaic
# glx:contracts.publish=run.start,run.done,run.error
# glx:data.in=GLX_OUT,GLX_ROWS,GLX_COLS,GLX_EDGE_THR,GLX_DELTA,GLX_KAPPA,GLX_PHI
# glx:data.out=report.json,mosaic_map.json,summary.md
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
        "mosaic.hybrid_ast_mosaic",
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

    rc = _run_py(cmd, git_root, glx_root, "mosaic")
    if rc != 0:
        _fail(f"mozaika zakończyła się kodem {rc}")

    out_dir = Path(_req(env, "GLX_OUT"))
    found = list(out_dir.rglob("report.json")) + list(out_dir.rglob("mosaic_map.json")) + list(
        out_dir.rglob("summary.md"))
    if not found:
        listing = "\n".join(f" - {p.relative_to(git_root)}" for p in sorted(out_dir.rglob("*")) if p.is_file())
        _fail("Brak artefaktów mozaiki w " + str(out_dir) + "\nPliki znalezione:\n" + (listing or "(pusto)"))


# glx:ast.fn=_should_zip
# glx:data.in=path
# glx:data.out=bool
def _should_zip(p: Path) -> bool:
    rel = str(p).replace("\\", "/")
    return p.is_file() and "/.git/" not in rel and "/__pycache__/" not in rel


# glx:ast.fn=_writestr_deterministic
# glx:data.in=arcname,data
def _writestr_deterministic(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    zi = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
    zi.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(zi, data)


# glx:ast.fn=_make_zip
# glx:mosaic.S=zip_index
# glx:mosaic.H=out_dir,autonomy_dir
# glx:contracts.publish=run.done
# glx:data.in=GLX_OUT,GLX_AUTONOMY_OUT
# glx:data.out=backup/AUDIT_*.zip
def _make_zip(env: Dict[str, str], git_root: Path, glx_root: Path) -> None:
    out_dir = Path(_req(env, "GLX_OUT"))
    auto_dir = Path(_req(env, "GLX_AUTONOMY_OUT"))

    backups = git_root / "backup"  # ZIP w KORZENIU REPO (nie w katalogu projektu)
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
            files.sort(key=lambda p: str(p.relative_to(git_root)).replace("\\", "/"))
            for p in files:
                arc = str(p.relative_to(git_root)).replace("\\", "/")
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
            "anchors": {"env": "project_dir", "paths": "git"},
        }
        _writestr_deterministic(zf, "GLX_AUDIT_META.json",
                                json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"))
    _log(f"audit zip: {zip_path.name}")


# ── DISPATCH (kolejność wg GLX_RUN) ──────────────────────────────────────────

# glx:ast.fn=_dispatch
# glx:mosaic.H=steps:A|M|E|Z
# glx:data.in=mask,order,env,git_root,glx_root
def _dispatch(mask: int, order: List[str],
              env: Dict[str, str], git_root: Path, glx_root: Path) -> None:
    DISPATCH = {
        "A": lambda: (_log("run:A"), _run_gateway(env, git_root, glx_root)),
        "M": lambda: (_log("run:M"), _run_mosaic(env, git_root, glx_root)),
        "E": lambda: (_log("run:E"), None),  # rezerwacja (np. mail statusu)
        "Z": lambda: (_log("run:Z"), _make_zip(env, git_root, glx_root)),
    }
    for ch in order:
        fn = DISPATCH.get(ch)
        if fn:
            fn()


# ── MAIN ─────────────────────────────────────────────────────────────────────

# glx:ast.fn=main
# glx:mosaic.S=fs_roots,env_loader,git_helpers
# glx:data.in=git_root,glx_root,GLX_RUN
# glx:invariants=env_from_project_dir_only,glx_root_equals_project_dir,outputs_inside_git_repo,no_fallback_env
def main() -> int:
    here = Path(__file__).resolve()

    # Ustal git_root (preferuj rev-parse, fallback: strukturalnie 2 poziomy wyżej)
    r = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(here.parent)
    )
    if r.returncode == 0 and r.stdout.strip():
        git_root = Path(r.stdout.strip()).resolve()
    else:
        git_root = here.parents[1].resolve()

    # Best-effort: post-diff (zanim odczytamy ENV)
    _run_post_diff(here, git_root)

    # ENV: WYŁĄCZNIE projekt (git_root.parent), twarde inwarianty — brak → BŁĄD i odmowa działania
    env = _load_env_project_only(git_root)
    env = _norm_paths_in_env(env, git_root)
    _enforce_env_invariants(env)

    # GLX_RUN → maska i kolejność; walidacja Z bez A/M
    mask, order = _parse_glx_run(env.get("GLX_RUN", ""))
    _validate_mask(mask)

    # Ścieżki/import
    glx_root = Path(_req(env, "GLX_ROOT")).resolve()
    if not glx_root.exists():
        _fail(f"GLX_ROOT wskazuje na nieistniejący katalog: {glx_root}")

    # Telemetria startowa
    _log(f"git_root={git_root}")
    _log(f"glx_root={glx_root}")
    _log(f"GLX_OUT={env.get('GLX_OUT')}")
    _log(f"GLX_AUTONOMY_OUT={env.get('GLX_AUTONOMY_OUT')}")

    # Dispatcher wg GLX_RUN
    _dispatch(mask, order, env, git_root, glx_root)

    _log(f"OK (mask=0x{mask:X}, order={''.join(order)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
