# -*- coding: utf-8 -*-
# Python: 3.9  (ten kod uruchamia się tylko na 3.9.x)
"""
---
version: 1
kind: module
id: "analysis-autonomy-gateway"
created_at: "2025-09-30"
name: "glitchlab.analysis.autonomy.gateway"
author: "GlitchLab v2"
role: "Brama danych dla LLM (SelfDoc Pack + Prompt Pack)"
description: >
  Deterministyczny moduł samoopisu i samoanalizy. Składa spójny pakiet danych
  (SelfDoc Pack) na podstawie drzewa repozytorium, AST plików .py, metadanych GLX
  oraz krótkiej historii Gita. Pakiet ten jest przystosowany do karmienia modeli
  językowych (Prompt Pack), bez ujawniania surowych patchy ani sekretów.
  Zero zależności zewnętrznych, zero I/O sieciowego. Best-effort, brak twardych
  wyjątków – zamiast tego status i diagnoza w polu 'telemetry'.
inputs:
  root: "Path projektu (domyślnie: katalog zawierający folder 'glitchlab')"
outputs:
  analysis/last/autonomy/pack.json: "SelfDoc Pack (schema v1)"
  analysis/last/autonomy/pack.md:   "skrót dla ludzi"
  analysis/last/autonomy/prompt.json: "{system,user,context,attachments}"
policy:
  deterministic: true
  side_effects: false (poza świadomym zapisem /analysis/last/autonomy/*)
constraints:
  - "Zero losowości"
  - "Nie czyta binariów; parsuje wyłącznie .py i lekkie JSON-y"
  - "Git opcjonalny; fallbacki jeżeli brak"
telemetry:
  counters: ["files_scanned","mods","defs","classes"]
  timers:   ["scan_ms","ast_ms"]
license: "Proprietary"
---
"""
# glitchlab/analysis/autonomy/gateway.py
from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Ścieżki bazowe i aliasy
# ──────────────────────────────────────────────────────────────────────────────
PKG_ANCHOR = Path(__file__).resolve()


def _find_project_root(start: Path) -> Path:
    cur = start
    for _ in range(6):  # „korytarz bezpieczeństwa”
        if (cur / "glitchlab").is_dir() and (cur / "glitchlab" / "__init__.py").exists():
            return cur
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # fallback: trzy poziomy wyżej niż .../glitchlab/analysis/autonomy/gateway.py = root repo
    return PKG_ANCHOR.parents[3]


PROJECT_ROOT = _find_project_root(PKG_ANCHOR)
GL_ROOT = PROJECT_ROOT / "glitchlab"
LAST_DIR = GL_ROOT / "analysis" / "last" / "autonomy"
GLX_STATE = PROJECT_ROOT / ".glx" / "state.json"


# ──────────────────────────────────────────────────────────────────────────────
# Minimalne typy danych (proste, bez TypedDict – brak zależności)
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
# Użyteczniki
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


def _is_py(path: Path) -> bool:
    name = path.name
    if not name.endswith(".py"):
        return False
    if name.endswith((".pyc",)) or name.startswith("."):
        return False
    return True


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


# ──────────────────────────────────────────────────────────────────────────────
# Skan repo + AST (deterministycznie)
# ──────────────────────────────────────────────────────────────────────────────

def scan_repository(root: Path) -> Tuple[Dict[str, Any], Telemetry]:
    """
    Zbiera listę modułów .py (bez venv/.git/artefaktów), proste metryki rozmiaru
    oraz wyciąga z AST: liczby klas/defów i ich nazwy+linie.
    """
    t0 = time.perf_counter()
    telemetry = Telemetry()
    modules: List[Dict[str, Any]] = []

    exclude_dirs = {
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".idea",
        "art", "backup", "resources", "analysis/last", "analysis/logs",
    }

    # Heurystyczny filtr katalogów (po częściach ścieżki)
    def _excluded(path: Path) -> bool:
        parts = set(path.parts)
        for ex in exclude_dirs:
            ex_parts = set(Path(ex).parts)
            if ex_parts.issubset(parts):
                return True
        return False

    for p in root.rglob("*.py"):
        if _excluded(p):
            continue
        telemetry.files_scanned += 1

        src = _read_text(p)
        loc = src.count("\n") + (1 if src else 0)

        # AST
        t_ast0 = time.perf_counter()
        mod_name = _rel(p).replace(os.sep, "/")
        n_defs, n_classes = 0, 0
        defs, classes = [], []
        try:
            node = ast.parse(src, filename=str(p))
            for child in ast.walk(node):
                if isinstance(child, ast.FunctionDef):
                    n_defs += 1
                    defs.append({"name": child.name, "lineno": child.lineno})
                elif isinstance(child, ast.AsyncFunctionDef):
                    n_defs += 1
                    defs.append({"name": child.name, "lineno": child.lineno, "async": True})
                elif isinstance(child, ast.ClassDef):
                    n_classes += 1
                    classes.append({"name": child.name, "lineno": child.lineno})
        except Exception:
            # nie wysadzamy – zapisujemy „broken_ast: True”
            modules.append({
                "path": mod_name, "loc": loc, "broken_ast": True,
                "defs": [], "classes": []
            })
            continue
        finally:
            telemetry.ast_ms += (time.perf_counter() - t_ast0) * 1000.0

        telemetry.mods += 1
        telemetry.defs += n_defs
        telemetry.classes += n_classes
        modules.append({
            "path": mod_name,
            "loc": loc,
            "defs": sorted(defs, key=lambda d: d["lineno"]),
            "classes": sorted(classes, key=lambda d: d["lineno"]),
        })

    telemetry.scan_ms = (time.perf_counter() - t0) * 1000.0

    # agregaty
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
        "modules": modules,  # pełna lista – konsument może pociąć
    }
    return code_summary, telemetry


# ──────────────────────────────────────────────────────────────────────────────
# Git (best-effort)
# ──────────────────────────────────────────────────────────────────────────────

def _run_git(args: List[str]) -> Tuple[int, str, str]:
    import subprocess
    try:
        p = subprocess.run(["git"] + args, cwd=PROJECT_ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, check=False)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 127, "", f"{e.__class__.__name__}: {e}"


def git_snapshot(max_commits: int = 10) -> Dict[str, Any]:
    rc, head, _ = _run_git(["rev-parse", "HEAD"])
    if rc != 0:
        return {"ok": False, "reason": "no-git"}
    log_rc, log_out, _ = _run_git(["log", f"-{max_commits}", "--pretty=%H|%ct|%s"])
    commits = []
    if log_rc == 0:
        for line in log_out.splitlines():
            try:
                h, ts, s = line.split("|", 2)
                commits.append({"hash": h, "ts": int(ts), "subject": s})
            except Exception:
                continue
    # prosta lista zmienionych plików między dwoma ostatnimi commitami
    if len(commits) >= 2:
        h1, h2 = commits[1]["hash"], commits[0]["hash"]
        drc, dout, _ = _run_git(["diff", "--name-status", f"{h1}..{h2}"])
        delta = []
        if drc == 0:
            for ln in dout.splitlines():
                parts = ln.split("\t")
                if len(parts) >= 2:
                    delta.append({"status": parts[0], "path": parts[1]})
    else:
        delta = []

    return {
        "ok": True,
        "head": head,
        "commits": commits,
        "delta_last": delta
    }


# ──────────────────────────────────────────────────────────────────────────────
# Prompt Pack – gotowe do karmienia LLM
# ──────────────────────────────────────────────────────────────────────────────

def make_prompt_pack(pack: SelfDocPack, purpose: str = "architect_review") -> Dict[str, Any]:
    """
    Zwraca strukturę {system,user,context,attachments}.
    Zero sekretów. Krótko i treściwie, na bazie SelfDoc Pack.
    """
    code = pack.code
    totals = code.get("totals", {})
    top_mods = code.get("top_modules_by_loc", [])[:8]

    system = (
        "Jesteś asystentem inżynierskim GlitchLab. Twoja analiza ma być deterministyczna, "
        "konkretna, bez wprowadzania losowości. Oceniaj wpływ zmian na architekturę i HUD/Mosaic⇄AST."
    )
    user = (
        f"Cel: {purpose}. Przygotuj plan refaktoryzacji i mapę ryzyk.\n"
        f"Repo: {_rel(Path(pack.project_root))} | LOC={totals.get('loc', 0)}, "
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

def build_selfdoc(root: Path = PROJECT_ROOT) -> SelfDocPack:
    glx = _safe_read_json(GLX_STATE)
    code, tel = scan_repository(GL_ROOT)
    git = git_snapshot(max_commits=12)
    if git.get("ok"):
        tel.git_ok = True

    notes: List[str] = []
    if not glx:
        notes.append("GLX: brak .glx/state.json – używam pustych wartości.")
    if not git.get("ok"):
        notes.append("GIT: niedostępny – historie Δ ograniczone.")

    # minimalna walidacja inwariantów (nasza teoria)
    # α+β≈1 nie jest tutaj liczona, ale przykład „policy”:
    invariants = {"edge_range": "[0,1]", "len(edge)==rows*cols": "delegowane do warstwy Mosaic"}

    code_summary = {
        **code,
        "invariants_hint": invariants,
        "conventions": {
            "CoreMosaic": "glitchlab.core.mosaic (dict)",
            "AnalysisMosaic": "glitchlab.mosaic.hybrid_ast_mosaic.Mosaic (dataclass)",
            "Aliases": "importuj z aliasami w warstwach łączących",
        },
    }

    pack = SelfDocPack(
        schema="glx.selfdoc.v1",
        generated_at=_utc_iso(),
        project_root=str(root),
        package_root=str(GL_ROOT),
        glx_state=glx,
        git=git,
        code=code_summary,
        prompts={},  # uzupełnimy po wygenerowaniu
        telemetry=asdict(tel),
        notes=notes,
    )
    # wstrzykuj Prompt Pack
    pack.prompts = {
        "architect_review": make_prompt_pack(pack, "architect_review"),
        "risk_brief": make_prompt_pack(pack, "risk_brief"),
    }
    return pack


def write_outputs(pack: SelfDocPack, out_dir: Path = LAST_DIR) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p_json = out_dir / "pack.json"
    p_md = out_dir / "pack.md"
    p_prompt = out_dir / "prompt.json"

    # JSON (SelfDoc)
    p_json.write_text(json.dumps(asdict(pack), indent=2, ensure_ascii=False), encoding="utf-8")

    # Prompt Pack (dla narzędzi, które chcą tylko promptu)
    p_prompt.write_text(json.dumps(pack.prompts, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown skrót
    totals = pack.code.get("totals", {})
    md = []
    md.append(f"# GLX SelfDoc Pack ({pack.schema}) — {pack.generated_at}\n")
    md.append(f"- Root: `{_rel(Path(pack.project_root))}`")
    md.append(f"- LOC: **{totals.get('loc', 0)}**, modules: **{totals.get('modules', 0)}**, "
              f"defs: **{totals.get('defs', 0)}**, classes: **{totals.get('classes', 0)}**")
    md.append(f"- Git head: `{pack.git.get('head', '?')}`")
    if pack.notes:
        md.append("\n**Notes:**")
        for n in pack.notes:
            md.append(f"- {n}")
    md.append("\n## Top modules by LOC\n")
    for m in pack.code.get("top_modules_by_loc", [])[:12]:
        md.append(
            f"- `{m['path']}` — {m['loc']} LOC, defs={len(m.get('defs', []))}, classes={len(m.get('classes', []))}")
    p_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    return {"pack_json": str(p_json), "pack_md": str(p_md), "prompt_json": str(p_prompt)}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_argv(argv: List[str]) -> Dict[str, Any]:
    # Minimalny parser:  gateway.py build --out DIR
    out = {"cmd": "build", "out": str(LAST_DIR)}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("build",):
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

        pack = build_selfdoc(PROJECT_ROOT)
        paths = write_outputs(pack, Path(args.get("out") or LAST_DIR))
        print("[GLX] autonomy pack ready:")
        for k, v in paths.items():
            print(f" - {k}: {v}")
        return 0
    except Exception as e:
        # Best-effort: nie wybuchamy na STDOUT – raportujemy czytelnie
        err = f"[GLX] ERROR ({e.__class__.__name__}): {e}\n{traceback.format_exc()}"
        # Zabezpieczenie: minimalny pliczek diagnostyczny
        LAST_DIR.mkdir(parents=True, exist_ok=True)
        (LAST_DIR / "gateway_error.log").write_text(err, encoding="utf-8")
        print(err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
