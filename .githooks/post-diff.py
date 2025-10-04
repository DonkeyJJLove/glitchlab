#!/usr/bin/env python
# glitchlab/.githooks/post-diff.py
# -*- coding: utf-8 -*-
# Python 3.9+
"""
POST-DIFF (uruchamiane PO commit):
- Dołącza HEAD i finalny zakres do .glx/commit_analysis.json,
- Archiwizuje raport do analysis/logs/commit_<ts>.json,
- Opcjonalnie dopisuje wiersz do docs/DIFF_SUMMARY.md.

Kompatybilność:
- Szuka .env/.env.local tak jak inne hooki (łańcuch przodków, .env.local > .env),
- Ścieżki względne liczone względem GLX_ROOT,
- Domyślnie włącza dopisywanie do DIFF_SUMMARY.md; można wyłączyć:
  GLX_SUMMARY_MD=0 (lub false/no/off).
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path
from typing import Optional

# wymuś UTF-8, zwłaszcza na Windows
os.environ.setdefault("PYTHONUTF8", "1")

# ── import wspólnych utili ────────────────────────────────────────────────────
HERE = Path(__file__).resolve()
HOOKS = HERE.parent
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

from _common import (  # type: ignore
    log,
    git,
    rev_parse,
    repo_top_from_here,
    load_env,
    norm_paths_in_env,
)

# ── helpers ──────────────────────────────────────────────────────────────────
def _rev_ok(ref: str, repo: Path) -> bool:
    r = git(["rev-parse", "--verify", ref], repo)
    return r.returncode == 0 and r.stdout.strip() != ""


def _truthy(x: object, default: bool = True) -> bool:
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    # 0) repo + env
    repo_top = repo_top_from_here(HERE)
    env = load_env(repo_top)
    env = norm_paths_in_env(env, repo_top)

    glx_root = Path(env["GLX_ROOT"]).resolve()
    glx = glx_root / ".glx"
    rep = glx / "commit_analysis.json"

    if not rep.exists():
        log("[post-diff] no pre-diff report; skipping")
        return 0

    # 1) wczytaj raport
    try:
        data = json.loads(rep.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"[post-diff] malformed commit_analysis.json: {e}; skipping")
        return 0

    # 2) HEAD i zakres
    head = rev_parse("HEAD", repo_top)
    prev: Optional[str] = rev_parse("HEAD~1", repo_top) if _rev_ok("HEAD~1", repo_top) else None
    rng = f"{prev[:7]}..{head[:7]}" if prev else f"{head[:7]}"

    data.update({"commit": head, "range": rng})
    rep.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) archiwum do analysis/logs
    logs = glx_root / "analysis" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    outp = logs / f"commit_{ts}.json"
    outp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[post-diff] saved: {outp}")

    # 4) opcjonalny append do docs/DIFF_SUMMARY.md
    if _truthy(env.get("GLX_SUMMARY_MD", "1"), default=True):
        docs = glx_root / "docs"
        if docs.exists():
            md = docs / "DIFF_SUMMARY.md"
            avg_align = 0.0
            try:
                avg_align = float(data.get("aggregate", {}).get("avg_align", 0.0))
            except Exception:
                avg_align = 0.0
            files_cnt = 0
            try:
                files_cnt = int(len(data.get("files", [])))
            except Exception:
                files_cnt = 0
            line = f"- {ts}  {rng}  files={files_cnt}  Align≈{avg_align:.2f}\n"
            with md.open("a", encoding="utf-8") as f:
                f.write(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
