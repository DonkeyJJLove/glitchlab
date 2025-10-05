# -*- coding: utf-8 -*-
# Python: 3.9  (ten kod uruchamia się tylko na 3.9.x)
# Ścieżka: glitchlab/analysis/autonomy/gateway.py
"""
GLX-CTX:v2
component: analysis.autonomy.gateway
role: Brama danych dla LLM (SelfDoc Pack + Prompt Pack)

# MOSAIC::PROFILE (S/H/Z)
S: env_loader,fs_roots,ast_scanner,git_helpers,pack_builder
H: env(.env project),repo.git,ast.parse,json.dumps
Z: 1

# AST::SURFACE
imports: ast,json,os,sys,time,traceback,dataclasses,pathlib,typing
public_api: main,build_selfdoc,write_outputs,make_prompt_pack

# CONTRACTS::EVENTS
publish:
  gateway.start: ctx:str
  gateway.done: artifacts:list
  gateway.error: error:str

# INVARIANTS
- env_from_project_dir_only
- glx_root_equals_project_dir
- outputs_anchored_to_git_root
- git_calls_from_git_root
- no_network_io
- deterministic_outputs

# DATA::SOURCES
env: <PROJECT_DIR>/.env | .env.local
repo: <GIT_ROOT>/.git
code: <GIT_ROOT>/<GLX_PKG>/**/*.py
state: <PROJECT_DIR>/.glx/state.json

# DATA::SINKS
out: <GIT_ROOT>/<GLX_AUTONOMY_OUT>/{pack.json,pack.md,prompt.json}
diag: <GIT_ROOT>/<GLX_AUTONOMY_OUT>/gateway_error.log

# GRAMMAR::HOOKS (komentarze #glx:event=… → Δ)
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
"""
from __future__ import annotations

# glx:event=enter_scope:module_imports
import ast
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
# glx:event=exit_scope

# ----------------------------- ANCHORS & ENV ---------------------------------

# glx:ast.fn=_strip_inline_comment
def _strip_inline_comment(s: str) -> str:
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


# glx:ast.fn=_clean_value
def _clean_value(v: str) -> str:
    v = _strip_inline_comment(v or "").strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1]
    return v.rstrip(":").strip()


# glx:ast.fn=_simple_parse_env
def _simple_parse_env(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = _clean_value(v)
    except Exception:
        pass
    return env


# glx:ast.fn=_load_env_project_only
# glx:mosaic.S=env_loader
# glx:data.in=PROJECT_DIR/.env,PROJECT_DIR/.env.local
def _load_env_project_only(project_dir: Path) -> None:
    """
    Ładuj WYŁĄCZNIE .env/.env.local z katalogu PROJEKTU. Brak → twardy błąd.
    Nie nadpisujemy istniejących zmiennych środowiskowych.
    """
    # glx:event=enter_scope:_load_env_project_only
    base = project_dir / ".env"
    loc = project_dir / ".env.local"
    found = False
    for p in (base, loc):
        if p.exists():
            for k, v in _simple_parse_env(p).items():
                os.environ.setdefault(k, v)
            found = True
    if not found:
        raise RuntimeError(f"[GLX][gateway] Brak .env/.env.local w katalogu projektu: {project_dir}")
    # glx:event=exit_scope


# glx:ast.fn=_norm_from
def _norm_from(root: Path, p: str) -> Path:
    if not p:
        return root.resolve()
    pp = Path(p)
    return pp.resolve() if pp.is_absolute() else (root / pp).resolve()


# glx:ast.fn=_best_guess_package_root
def _best_guess_package_root(anchor: Path, pkg_name: str) -> Path:
    # Jeśli anchor zawiera katalog pakietu → użyj anchor/pkg
    if (anchor / pkg_name).is_dir():
        return (anchor / pkg_name).resolve()
    # Jeśli anchor to bezpośrednio katalog pakietu (nazwa zgodna) → użyj anchor
    if anchor.name == pkg_name:
        return anchor.resolve()
    # Inaczej — przyjmij anchor/pkg (nawet jeśli chwilowo nie istnieje)
    return (anchor / pkg_name).resolve()


# glx:ast.fn=_find_git_root
def _find_git_root(start: Path) -> Path:
    """Znajdź najbliższy katalog zawierający .git; fallback: rodzic[2] (…/glitchlab)."""
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / ".git").exists():
            return p
    # heurystyka dla ścieżki …/glitchlab/analysis/autonomy/gateway.py
    try:
        return start.parents[2]
    except Exception:
        return start.parent


# --- Wyprowadzenie PROJECT_DIR (poziom NAD repo) i GIT_ROOT (repo) -----------

# glx:event=enter_scope:anchors
PKG_ANCHOR: Path = Path(__file__).resolve()
GIT_ROOT: Path = _find_git_root(PKG_ANCHOR)       # ← POPRAWKA: realne repo z .git
PROJECT_DIR: Path = GIT_ROOT.parent               # katalog projektu, poziom wyżej
PKG_NAME: str = PKG_ANCHOR.parents[2].name        # zwykle 'glitchlab'
# glx:event=exit_scope

# Ładowanie ENV tylko z katalogu projektu
_load_env_project_only(PROJECT_DIR)

# GLX_* z ENV + twarde inwarianty ścieżek
GLX_PKG: str = os.getenv("GLX_PKG", PKG_NAME)

# GLX_ROOT MUSI == PROJECT_DIR
GLX_ROOT: Path = _norm_from(PROJECT_DIR, os.getenv("GLX_ROOT", str(PROJECT_DIR)))
if GLX_ROOT != PROJECT_DIR:
    raise RuntimeError(f"[GLX][gateway] GLX_ROOT={GLX_ROOT} ≠ PROJECT_DIR={PROJECT_DIR} — przerwano.")

# OUTY kotwiczymy do GIT_ROOT (repo)
GLX_AUTONOMY_OUT: Path = _norm_from(
    GIT_ROOT,
    os.getenv("GLX_AUTONOMY_OUT", f"{GLX_PKG}/analysis/last/autonomy")
)

# Max rozmiar pliku do skanowania (bytes)
def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

GLX_SCAN_MAX_BYTES: int = _int_env("GLX_SCAN_MAX_BYTES", 1572864)  # 1.5MB

# Katalog pakietu (skan kodu) — z repo
GL_ROOT: Path = _best_guess_package_root(GIT_ROOT, GLX_PKG)

# Ścieżki stanu/diag
GLX_STATE: Path = (PROJECT_DIR / ".glx" / "state.json").resolve()
LAST_DIR: Path = GLX_AUTONOMY_OUT  # diag i wyjścia w repo

# ------------------------------ DATA TYPES -----------------------------------

@dataclass
class Telemetry:
    ok: bool = True
    error: str = ""
    scan_ms: float = 0.0
    ast_ms: float = 0.0
    files_scanned: int = 0
    mods: int = 0
    defs: int = 0
    classes: int = 0
    git_ok: bool = False
    skipped_too_big: int = 0
    skipped_binary_hint: int = 0
    parse_errors: int = 0
    largest_file_bytes: int = 0
    python_ver: str = sys.version.split(" ")[0]


@dataclass
class SelfDocPack:
    schema: str
    schema_ver: str
    generated_at: str
    project_root: str
    package_root: str
    glx_state: Dict[str, Any]
    git: Dict[str, Any]
    code: Dict[str, Any]
    prompts: Dict[str, Any]
    telemetry: Dict[str, Any]
    notes: List[str]
    env_hint: Dict[str, str]

# --------------------------------- UTILS -------------------------------------

def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def _rel_to_repo(p: Path) -> str:
    try:
        return str(p.relative_to(GIT_ROOT)).replace(os.sep, "/")
    except Exception:
        return str(p).replace(os.sep, "/")

def _is_binary_like(path: Path, probe_bytes: int = 4096) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(probe_bytes)
        return b"\x00" in chunk
    except Exception:
        return False

def _redact_env_snapshot(d: Dict[str, str]) -> Dict[str, str]:
    secrets = ("PASS", "PASSWORD", "SECRET", "TOKEN", "KEY")
    out: Dict[str, str] = {}
    for k, v in d.items():
        if k.startswith("GLX_"):
            out[k] = "******" if any(h in k.upper() for h in secrets) else str(v)
    return out

# -------------------------- SCAN (FS + AST) ----------------------------------

def scan_repository(pkg_root: Path, max_bytes: int) -> Tuple[Dict[str, Any], Telemetry]:
    """
    Zbiera listę modułów .py w obrębie katalogu pakietu (GL_ROOT),
    omija artefakty/venvy. Wypluwa też proste metryki i statystykę AST.
    """
    t0 = time.perf_counter()
    telemetry = Telemetry()
    modules: List[Dict[str, Any]] = []

    exclude_dirs = {
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".idea",
        "analysis/last", "analysis/logs", "art", "backup", "resources",
        "dist", "build"
    }

    def _excluded(path: Path) -> bool:
        parts = set(path.parts)
        for ex in exclude_dirs:
            if set(Path(ex).parts).issubset(parts):
                return True
        return False

    candidates = sorted(pkg_root.rglob("*.py"), key=lambda p: str(p).lower())
    for p in candidates:
        if _excluded(p):
            continue
        try:
            sz = p.stat().st_size
            if sz > telemetry.largest_file_bytes:
                telemetry.largest_file_bytes = int(sz)
            if sz > max_bytes:
                telemetry.skipped_too_big += 1
                continue
        except Exception:
            telemetry.skipped_too_big += 1
            continue

        if _is_binary_like(p):
            telemetry.skipped_binary_hint += 1
            continue

        telemetry.files_scanned += 1
        src = _read_text(p)
        loc = src.count("\n") + (1 if src else 0)
        t_ast0 = time.perf_counter()

        rel_path = _rel_to_repo(p)
        n_defs = n_classes = 0
        defs: List[Dict[str, Any]] = []
        classes: List[Dict[str, Any]] = []
        try:
            node = ast.parse(src, filename=str(p))
            for ch in ast.walk(node):
                if isinstance(ch, ast.FunctionDef):
                    n_defs += 1
                    defs.append({"name": ch.name, "lineno": ch.lineno})
                elif isinstance(ch, ast.AsyncFunctionDef):
                    n_defs += 1
                    defs.append({"name": ch.name, "lineno": ch.lineno, "async": True})
                elif isinstance(ch, ast.ClassDef):
                    n_classes += 1
                    classes.append({"name": ch.name, "lineno": ch.lineno})
        except Exception:
            telemetry.parse_errors += 1
            modules.append({"path": rel_path, "loc": loc, "broken_ast": True, "defs": [], "classes": []})
        else:
            telemetry.mods += 1
            telemetry.defs += n_defs
            telemetry.classes += n_classes
            modules.append({
                "path": rel_path,
                "loc": loc,
                "defs": sorted(defs, key=lambda d: d["lineno"]),
                "classes": sorted(classes, key=lambda d: d["lineno"]),
            })
        finally:
            telemetry.ast_ms += (time.perf_counter() - t_ast0) * 1000.0

    telemetry.scan_ms = (time.perf_counter() - t0) * 1000.0

    total_loc = sum(m.get("loc", 0) for m in modules)
    biggest = sorted(modules, key=lambda m: (m.get("loc", 0), m.get("path", "")), reverse=True)[:16]
    modules_sorted = sorted(modules, key=lambda m: m.get("path", ""))

    code_summary = {
        "totals": {
            "loc": total_loc,
            "modules": telemetry.mods,
            "defs": telemetry.defs,
            "classes": telemetry.classes,
        },
        "top_modules_by_loc": biggest,
        "modules": modules_sorted,
    }
    return code_summary, telemetry

# ------------------------------- GIT -----------------------------------------

def _run_git(args: List[str]) -> Tuple[int, str, str]:
    import subprocess
    try:
        p = subprocess.run(["git"] + args, cwd=str(GIT_ROOT),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, check=False)
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except Exception as e:
        return 127, "", f"{e.__class__.__name__}: {e}"

def git_snapshot(max_commits: int = 10) -> Dict[str, Any]:
    rc, head, _ = _run_git(["rev-parse", "HEAD"])
    if rc != 0:
        return {"ok": False, "reason": "no-git"}

    _, branch, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    _, head_abbrev, _ = _run_git(["rev-parse", "--short", "HEAD"])

    log_rc, log_out, _ = _run_git(["log", f"-{max_commits}", "--pretty=%H|%ct|%s"])
    commits: List[Dict[str, Any]] = []
    if log_rc == 0 and log_out:
        for line in log_out.splitlines():
            try:
                h, ts, s = line.split("|", 2)
                commits.append({"hash": h, "ts": int(ts), "subject": s})
            except Exception:
                continue

    if len(commits) >= 2:
        h1, h2 = commits[1]["hash"], commits[0]["hash"]
        drc, dout, _ = _run_git(["diff", "--name-status", f"{h1}..{h2}"])
        delta: List[Dict[str, Any]] = []
        if drc == 0 and dout:
            for ln in dout.splitlines():
                parts = ln.split("\t")
                if len(parts) >= 2:
                    delta.append({"status": parts[0], "path": parts[1]})
    else:
        delta = []

    return {
        "ok": True,
        "head": head,
        "head_abbrev": head_abbrev,
        "branch": branch,
        "commits": commits,
        "delta_last": delta
    }

# ----------------------------- PROMPT PACK -----------------------------------

def make_prompt_pack(pack: SelfDocPack, purpose: str = "architect_review") -> Dict[str, Any]:
    code = pack.code
    totals = code.get("totals", {})
    top_mods = code.get("top_modules_by_loc", [])[:8]

    system = (
        "Jesteś asystentem inżynierskim GlitchLab. Twoja analiza ma być deterministyczna, "
        "konkretna, bez wprowadzania losowości. Oceniaj wpływ zmian na architekturę i HUD/Mosaic⇄AST."
    )
    user = (
        f"Cel: {purpose}. Przygotuj plan refaktoryzacji i mapę ryzyk.\n"
        f"Repo: {_rel_to_repo(Path(pack.project_root))} | LOC={totals.get('loc', 0)}, "
        f"modules={totals.get('modules', 0)}, defs={totals.get('defs', 0)}, classes={totals.get('classes', 0)}.\n"
        "Zidentyfikuj moduły krytyczne i zaproponuj kroki poprawy spójności."
    )
    ctx = {
        "glx_state": pack.glx_state,
        "git_head": pack.git.get("head"),
        "git_branch": pack.git.get("branch"),
        "recent_commits": pack.git.get("commits", [])[:5],
        "delta_last": pack.git.get("delta_last", [])[:20],
        "top_modules": top_mods,
        "notes": pack.notes,
    }
    attachments = {
        "schema": pack.schema,
        "when": pack.generated_at,
        "telemetry": pack.telemetry,
        "python": pack.telemetry.get("python_ver", "")
    }
    return {"system": system, "user": user, "context": ctx, "attachments": attachments}

# ---------------------- SELF DOC BUILD & OUTPUTS -----------------------------

def build_selfdoc(max_bytes: int = GLX_SCAN_MAX_BYTES) -> SelfDocPack:
    glx = _safe_read_json(GLX_STATE)
    code, tel = scan_repository(GL_ROOT, max_bytes=max_bytes)
    git = git_snapshot(max_commits=12)
    if git.get("ok"):
        tel.git_ok = True

    notes: List[str] = []
    if not glx:
        notes.append("GLX: brak .glx/state.json — używam pustych wartości.")
    if not git.get("ok"):
        notes.append("GIT: niedostępny — historie Δ ograniczone.")

    invariants = {"edge_range": "[0,1]", "len(edge)==rows*cols": "delegowane do warstwy Mosaic"}

    code_summary = {
        **code,
        "invariants_hint": invariants,
        "conventions": {
            "CoreMosaic": "glitchlab.core.mosaic (dict)",
            "AnalysisMosaic": "glitchlab.gui.mosaic.hybrid_ast_mosaic.Mosaic (dataclass)",
            "Aliases": "importuj z aliasami w warstwach łączących",
        },
    }

    env_hint = _redact_env_snapshot(os.environ.copy())

    pack = SelfDocPack(
        schema="glx.selfdoc",
        schema_ver="v1.1",
        generated_at=_utc_iso(),
        project_root=str(GIT_ROOT),
        package_root=str(GL_ROOT),
        glx_state=glx,
        git=git,
        code=code_summary,
        prompts={},  # wstrzykniemy za chwilę
        telemetry=asdict(tel),
        notes=notes,
        env_hint=env_hint,
    )
    pack.prompts = {
        "architect_review": make_prompt_pack(pack, "architect_review"),
        "risk_brief": make_prompt_pack(pack, "risk_brief"),
    }
    return pack


def write_outputs(pack: SelfDocPack, out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p_json = out_dir / "pack.json"
    p_md = out_dir / "pack.md"
    p_prompt = out_dir / "prompt.json"

    # JSON (SelfDoc)
    p_json.write_text(json.dumps(asdict(pack), indent=2, ensure_ascii=False), encoding="utf-8")

    # Prompt Pack
    p_prompt.write_text(json.dumps(pack.prompts, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown skrót
    totals = pack.code.get("totals", {})
    md: List[str] = []
    md.append(f"# GLX SelfDoc Pack ({pack.schema} {pack.schema_ver}) — {pack.generated_at}\n")
    md.append(f"- Root: `{_rel_to_repo(Path(pack.project_root))}`")
    md.append(
        f"- LOC: **{totals.get('loc', 0)}**, modules: **{totals.get('modules', 0)}**, "
        f"defs: **{totals.get('defs', 0)}**, classes: **{totals.get('classes', 0)}**"
    )
    md.append(
        f"- Git: branch=`{pack.git.get('branch', '?')}`, head=`{pack.git.get('head_abbrev', pack.git.get('head', '?'))}`")
    if pack.notes:
        md.append("\n**Notes:**")
        for n in pack.notes:
            md.append(f"- {n}")
    md.append("\n## Top modules by LOC\n")
    for m in pack.code.get("top_modules_by_loc", [])[:12]:
        md.append(
            f"- `{m['path']}` — {m['loc']} LOC, defs={len(m.get('defs', []))}, classes={len(m.get('classes', []))}")
    p_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    return {"pack.json": str(p_json), "pack.md": str(p_md), "prompt.json": str(p_prompt)}

# ---------------------------------- CLI --------------------------------------

def _parse_argv(argv: List[str]) -> Dict[str, Any]:
    # Minimalny parser: gateway.py build [--out DIR] [--root PATH] [--pkg NAME] [--max-bytes N]
    out = {"cmd": "build", "out": str(GLX_AUTONOMY_OUT), "root": None, "pkg": None, "max_bytes": None}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "build":
            out["cmd"] = "build"
        elif a in ("--out", "-o") and i + 1 < len(argv):
            out["out"] = argv[i + 1]; i += 1
        elif a == "--root" and i + 1 < len(argv):
            out["root"] = argv[i + 1]; i += 1
        elif a == "--pkg" and i + 1 < len(argv):
            out["pkg"] = argv[i + 1]; i += 1
        elif a == "--max-bytes" and i + 1 < len(argv):
            try:
                out["max_bytes"] = int(argv[i + 1])
            except Exception:
                pass
            i += 1
        i += 1
    return out


def main(argv: Optional[List[str]] = None) -> int:
    # global na początku, jeśli ktoś poda --root/--pkg
    global GIT_ROOT, GL_ROOT, GLX_PKG

    argv = argv or sys.argv[1:]
    try:
        args = _parse_argv(argv)
        if args.get("cmd") != "build":
            print("usage: python -m glitchlab.analysis.autonomy.gateway build [--out DIR] [--root PATH] [--pkg NAME] [--max-bytes N]")
            return 2

        # opcjonalne nadpisania z CLI (z zachowaniem inwariantów)
        root_override = args.get("root")
        pkg_override = args.get("pkg")
        max_bytes = args.get("max_bytes") or GLX_SCAN_MAX_BYTES

        # Nadpisanie ROOT → przelicz GIT_ROOT i GL_ROOT
        if root_override:
            new_root = Path(root_override).resolve()
            if not new_root.exists():
                raise RuntimeError(f"[GLX][gateway] --root nie istnieje: {new_root}")
            # jeśli ktoś poda katalog projektu, spróbuj zejść do repo z .git:
            GIT_ROOT = _find_git_root(new_root)
            GL_ROOT = _best_guess_package_root(GIT_ROOT, GLX_PKG)

        if pkg_override:
            GLX_PKG = str(pkg_override)
            GL_ROOT = _best_guess_package_root(GIT_ROOT, GLX_PKG)

        pack = build_selfdoc(max_bytes=max_bytes)

        out_dir = Path(args.get("out") or GLX_AUTONOMY_OUT)
        if not out_dir.is_absolute():
            out_dir = (GIT_ROOT / out_dir).resolve()  # kotwica w repo
        paths = write_outputs(pack, out_dir)

        print("[GLX] autonomy pack ready:")
        for k, v in paths.items():
            print(f" - {k}: {v}")
        return 0

    except Exception as e:
        err = f"[GLX] ERROR ({e.__class__.__name__}): {e}\n{traceback.format_exc()}"
        # Bezpieczny plik diagnostyczny (w repo, przy outach)
        try:
            LAST_DIR.mkdir(parents=True, exist_ok=True)
            (LAST_DIR / "gateway_error.log").write_text(err, encoding="utf-8")
        except Exception:
            pass
        print(err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
