# -*- coding: utf-8 -*-
# Python: 3.9  (ten kod uruchamia się tylko na 3.9.x)
# Ścieżka: glitchlab/analysis/autonomy/gateway.py
"""
glitchlab.analysis.autonomy.gateway
Rola: Brama danych dla LLM (SelfDoc Pack + Prompt Pack)

Zasady:
- Zero zależności zewnętrznych (python-dotenv opcjonalnie).
- Zero I/O sieciowego.
- Determinizm: brak losowości; w razie błędów → best-effort telemetry + plik diagnostyczny.
- Główne ścieżki i defaulty czerpane z `.env`.

Wejścia (.env – wszystkie opcjonalne, mają bezpieczne fallbacki):
  GLX_ROOT=.<repo_root>
  GLX_PKG=glitchlab
  GLX_AUTONOMY_OUT=glitchlab/analysis/last/autonomy

Wyjścia:
  pack.json  (SelfDoc Pack)
  pack.md    (skrót dla ludzi)
  prompt.json ({system,user,context,attachments})

CLI:
  python -m glitchlab.analysis.autonomy.gateway build [--out DIR]
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# .ENV — ładowanie (python-dotenv jeśli jest; inaczej prosty parser)
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

def _load_dotenv_into_environ(dotenv_dir: Path) -> None:
    # Szukamy .env w katalogu i jeden poziom wyżej (jak w testach)
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
    # Fallback – prosty parser bez nadpisywania istniejących zmiennych
    parsed = _simple_parse_env(dot)
    for k, v in parsed.items():
        os.environ.setdefault(k, v)

# Załaduj .env wcześnie
PKG_ANCHOR = Path(__file__).resolve()
# Heurystycznie: repo root to 3 poziomy wyżej od pliku (glitchlab/analysis/autonomy/gateway.py)
_default_repo_root = PKG_ANCHOR.parents[3]
_load_dotenv_into_environ(_default_repo_root)

# ──────────────────────────────────────────────────────────────────────────────
# Standard .env → główne ścieżki
# ──────────────────────────────────────────────────────────────────────────────

def _norm_from(root: Path, p: str) -> Path:
    if not p:
        return root
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp).resolve()

GLX_ROOT = _norm_from(
    _default_repo_root,
    os.getenv("GLX_ROOT", str(_default_repo_root))
)

GLX_PKG = os.getenv("GLX_PKG", "glitchlab")

# Katalog wyjściowy (autonomy)
GLX_AUTONOMY_OUT = _norm_from(
    GLX_ROOT,
    os.getenv("GLX_AUTONOMY_OUT", f"{GLX_PKG}/analysis/last/autonomy")
)

# Katalog pakietu (do skanowania kodu)
GL_ROOT = (GLX_ROOT / GLX_PKG).resolve()

# Projekt/Repo root (dla Gita i ścieżek względnych)
PROJECT_ROOT = GLX_ROOT

# Dla kompatybilności ze starymi nazwami
LAST_DIR = GLX_AUTONOMY_OUT
GLX_STATE = PROJECT_ROOT / ".glx" / "state.json"

# ──────────────────────────────────────────────────────────────────────────────
# Minimalne typy danych
# ──────────────────────────────────────────────────────────────────────────────

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

@dataclass
class SelfDocPack:
    schema: str
    generated_at: str
    project_root: str
    package_root: str
    glx_state: Dict[str, Any]
    git: Dict[str, Any]
    code: Dict[str, Any]
    prompts: Dict[str, Any]
    telemetry: Dict[str, Any]
    notes: List[str]

# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────

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

def _rel_to_project(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)

# ──────────────────────────────────────────────────────────────────────────────
# Skan repo + AST
# ──────────────────────────────────────────────────────────────────────────────

def scan_repository(pkg_root: Path) -> Tuple[Dict[str, Any], Telemetry]:
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
    }

    def _excluded(path: Path) -> bool:
        parts = set(path.parts)
        for ex in exclude_dirs:
            if set(Path(ex).parts).issubset(parts):
                return True
        return False

    for p in pkg_root.rglob("*.py"):
        if _excluded(p):
            continue
        telemetry.files_scanned += 1

        src = _read_text(p)
        loc = src.count("\n") + (1 if src else 0)

        t_ast0 = time.perf_counter()
        rel_path = _rel_to_project(p).replace(os.sep, "/")
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
    biggest = sorted(modules, key=lambda m: m.get("loc", 0), reverse=True)[:16]

    code_summary = {
        "totals": {
            "loc": total_loc,
            "modules": telemetry.mods,
            "defs": telemetry.defs,
            "classes": telemetry.classes,
        },
        "top_modules_by_loc": biggest,
        "modules": modules,
    }
    return code_summary, telemetry

# ──────────────────────────────────────────────────────────────────────────────
# Git (best-effort, bez twardych wymagań)
# ──────────────────────────────────────────────────────────────────────────────

def _run_git(args: List[str]) -> Tuple[int, str, str]:
    import subprocess
    try:
        p = subprocess.run(["git"] + args, cwd=str(PROJECT_ROOT),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, check=False)
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except Exception as e:
        return 127, "", f"{e.__class__.__name__}: {e}"

def git_snapshot(max_commits: int = 10) -> Dict[str, Any]:
    rc, head, _ = _run_git(["rev-parse", "HEAD"])
    if rc != 0:
        return {"ok": False, "reason": "no-git"}

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

    return {"ok": True, "head": head, "commits": commits, "delta_last": delta}

# ──────────────────────────────────────────────────────────────────────────────
# Prompt Pack
# ──────────────────────────────────────────────────────────────────────────────

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
        f"Repo: {_rel_to_project(Path(pack.project_root))} | LOC={totals.get('loc', 0)}, "
        f"modules={totals.get('modules', 0)}, defs={totals.get('defs', 0)}, classes={totals.get('classes', 0)}.\n"
        "Zidentyfikuj moduły krytyczne i zaproponuj kroki poprawy spójności."
    )
    ctx = {
        "glx_state": pack.glx_state,
        "git_head": pack.git.get("head"),
        "recent_commits": pack.git.get("commits", [])[:5],
        "delta_last": pack.git.get("delta_last", [])[:20],
        "top_modules": top_mods,
        "notes": pack.notes,
    }
    attachments = {
        "schema": pack.schema,
        "when": pack.generated_at,
        "telemetry": pack.telemetry,
    }
    return {"system": system, "user": user, "context": ctx, "attachments": attachments}

# ──────────────────────────────────────────────────────────────────────────────
# Składanie SelfDoc Pack + zapis
# ──────────────────────────────────────────────────────────────────────────────

def build_selfdoc() -> SelfDocPack:
    glx = _safe_read_json(GLX_STATE)
    code, tel = scan_repository(GL_ROOT)
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

    pack = SelfDocPack(
        schema="glx.selfdoc.v1",
        generated_at=_utc_iso(),
        project_root=str(PROJECT_ROOT),
        package_root=str(GL_ROOT),
        glx_state=glx,
        git=git,
        code=code_summary,
        prompts={},  # wstrzykniemy za chwilę
        telemetry=asdict(tel),
        notes=notes,
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
    md.append(f"# GLX SelfDoc Pack ({pack.schema}) — {pack.generated_at}\n")
    md.append(f"- Root: `{_rel_to_project(Path(pack.project_root))}`")
    md.append(
        f"- LOC: **{totals.get('loc', 0)}**, modules: **{totals.get('modules', 0)}**, "
        f"defs: **{totals.get('defs', 0)}**, classes: **{totals.get('classes', 0)}**"
    )
    md.append(f"- Git head: `{pack.git.get('head', '?')}`")
    if pack.notes:
        md.append("\n**Notes:**")
        for n in pack.notes:
            md.append(f"- {n}")
    md.append("\n## Top modules by LOC\n")
    for m in pack.code.get("top_modules_by_loc", [])[:12]:
        md.append(f"- `{m['path']}` — {m['loc']} LOC, defs={len(m.get('defs', []))}, classes={len(m.get('classes', []))}")
    p_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    return {"pack_json": str(p_json), "pack_md": str(p_md), "prompt_json": str(p_prompt)}

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_argv(argv: List[str]) -> Dict[str, Any]:
    # Minimalny parser: gateway.py build [--out DIR]
    out = {"cmd": "build", "out": str(GLX_AUTONOMY_OUT)}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "build":
            out["cmd"] = "build"
        elif a in ("--out", "-o") and i + 1 < len(argv):
            out["out"] = argv[i + 1]
            i += 1
        i += 1
    return out

def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or sys.argv[1:]
    try:
        args = _parse_argv(argv)
        if args.get("cmd") != "build":
            print("usage: python -m glitchlab.analysis.autonomy.gateway build [--out DIR]")
            return 2

        pack = build_selfdoc()
        out_dir = Path(args.get("out") or GLX_AUTONOMY_OUT)
        if not out_dir.is_absolute():
            out_dir = (PROJECT_ROOT / out_dir).resolve()
        paths = write_outputs(pack, out_dir)

        print("[GLX] autonomy pack ready:")
        for k, v in paths.items():
            print(f" - {k}: {v}")
        return 0
    except Exception as e:
        err = f"[GLX] ERROR ({e.__class__.__name__}): {e}\n{traceback.format_exc()}"
        # Bezpieczny plik diagnostyczny w katalogu z .env/out
        try:
            LAST_DIR.mkdir(parents=True, exist_ok=True)
            (LAST_DIR / "gateway_error.log").write_text(err, encoding="utf-8")
        except Exception:
            pass
        print(err)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
