#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""
.githooks/post-commit.py â€” lokalny driver post-commit (Python 3.9)

Funkcje poÅ‚Ä…czone (kompatybilnoÅ›Ä‡ ze â€žstarÄ…â€ wersjÄ… + nowa telemetria GLX):
- ObsÅ‚uga GLX_RUN (A/M/E/Z) z odczytem ENV wyÅ‚Ä…cznie z katalogu projektu (.env/.env.local).
- Wyznaczanie base_sha: .glx/state.json â†’ merge-base (origin/{main,master}) â†’ HEAD~1 â†’ empty-tree.
- A: uruchomienie moduÅ‚u autonomy.gateway (budowa artefaktÃ³w do GLX_AUTONOMY_OUT).
- M: generowanie mozaiki (hybrid_ast_mosaic) z parametrami z ENV (Î¦ policy opcjonalnie).
- E: (best-effort) eksport pakietu analitycznego, jeÅ¼eli istnieje CLI; w przeciwnym razie ostrzeÅ¼enie.
- Z: utworzenie archiwum AUDIT_*.zip z GLX_OUT oraz GLX_AUTONOMY_OUT.
- Dodatkowo (nieblokujÄ…co): Î”-fingerprint + invariants_check â†’ .glx/delta_report.json / commit_analysis.json.
- Zapis zdarzenia do .glx/events.log.jsonl oraz wygenerowanie .glx/commit_snippet.txt.

Wymagania:
- Python 3.9+
- .githooks/_common.py (helpery)
- Struktura katalogÃ³w GLX w repo
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
from typing import Dict, List, Optional, Tuple

# Helpery wspÃ³lne
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
import _common as H  # noqa: E402


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Parsowanie .env z katalogu projektu (wyÅ‚Ä…cznie stamtÄ…d)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _strip_inline_comment(s: str) -> str:
    s = s or ""
    in_s = False
    in_d = False
    out = []
    for ch in s:
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            break
        out.append(ch)
    return "".join(out).strip()


def _parse_env_file(p: Path) -> Dict[str, str]:
    env = {}
    if not p.exists():
        return env
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _strip_inline_comment(raw)
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            env[k] = v
    return env


def read_project_env(project_dir: Path) -> Dict[str, str]:
    env = {}
    env.update(_parse_env_file(project_dir / ".env"))
    env.update(_parse_env_file(project_dir / ".env.local"))
    return env


def ensure_required(env: Dict[str, str], keys: List[str]) -> None:
    missing = [k for k in keys if not env.get(k)]
    if missing:
        H.fail("Brak wymaganych zmiennych ENV: " + ", ".join(missing))


def ensure_inside_repo(path: Path, repo_root: Path, name: str) -> Path:
    p = path.resolve()
    r = repo_root.resolve()
    try:
        p.relative_to(r)
    except Exception:
        H.fail("%s musi wskazywaÄ‡ Å›cieÅ¼kÄ™ wewnÄ…trz repo: %s" % (name, p))
    p.mkdir(parents=True, exist_ok=True)
    return p


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# GLX_RUN (A/M/E/Z)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FLAGS = {"A": 0x1, "M": 0x2, "E": 0x4, "Z": 0x8}
ORDER = ["A", "M", "E", "Z"]


def parse_glx_run(s: str) -> Tuple[int, List[str]]:
    s0 = _strip_inline_comment((s or "").strip())
    if not s0:
        H.fail("GLX_RUN puste. PrzykÅ‚ady: 'A', 'A+M', 'MEAZ', '0xD' lub '13'.")
    up = s0.upper().replace(" ", "")
    # formy liczbowe
    try:
        if up.startswith("0X"):
            mask = int(up, 16)
            order = [k for k in ORDER if FLAGS[k] & mask]
        elif up.isdigit():
            mask = int(up, 10)
            order = [k for k in ORDER if FLAGS[k] & mask]
        else:
            raise ValueError
        if not mask:
            H.fail("GLX_RUN=0/0x0 nie zawiera trybÃ³w.")
        return mask, order
    except ValueError:
        pass
    # forma literowa
    if re.findall(r"[^AMEZ\s,+;/|.\-]", up):
        H.fail("GLX_RUN zawiera niedozwolone znaki (dozwolone: A,M,E,Z).")
    letters = re.findall(r"[AMEZ]", up)
    if not letters:
        H.fail("GLX_RUN nie zawiera liter z [A M E Z].")
    order, seen = [], set()
    for ch in letters:
        if ch not in seen:
            seen.add(ch)
            order.append(ch)
    mask = 0
    for ch in order:
        mask |= FLAGS[ch]
    return mask, order


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Wykrycie base_sha (dla â€žMâ€)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def detect_base(git_root: Path, state_path: Path) -> str:
    def rev_ok(ref: str) -> bool:
        r = subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            cwd=str(git_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return r.returncode == 0

    # 1) .glx/state.json
    if state_path.exists():
        try:
            bs = (json.loads(state_path.read_text(encoding="utf-8")).get("base_sha") or "").strip()
            if bs and rev_ok(bs):
                return bs
        except Exception as e:
            H.warn("state.json: %s" % e)

    # 2) merge-base z upstream
    for br in ("origin/master", "origin/main"):
        r = subprocess.run(
            ["git", "merge-base", "HEAD", br],
            cwd=str(git_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        cand = r.stdout.strip()
        if r.returncode == 0 and cand and rev_ok(cand):
            return cand

    # 3) HEAD~1
    r = subprocess.run(
        ["git", "rev-parse", "HEAD~1"],
        cwd=str(git_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if r.returncode == 0 and r.stdout.strip() and rev_ok(r.stdout.strip()):
        return r.stdout.strip()

    # 4) empty-tree
    r = subprocess.run(
        ["git", "hash-object", "-t", "tree", "/dev/null"],
        cwd=str(git_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return r.stdout.strip() or "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Archiwum AUDIT_*.zip (Z)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _zip_deterministic_writestr(zf, arcname: str, data: bytes) -> None:
    zi = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
    zi.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(zi, data)


def make_audit_zip(env: Dict[str, str], git_root: Path) -> None:
    out_dir = Path(env["GLX_OUT"])
    auto_dir = Path(env["GLX_AUTONOMY_OUT"])
    backups = git_root / "backup"
    backups.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    zip_path = backups / ("AUDIT_%s.zip" % ts)

    def should_zip(p: Path) -> bool:
        rel = str(p).replace("\\", "/")
        return p.is_file() and "/.git/" not in rel and "/__pycache__/" not in rel

    added = set()
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for root in (out_dir, auto_dir):
            if not root.exists():
                continue
            files = [p for p in root.rglob("*") if should_zip(p)]
            files.sort(key=lambda p: str(p.relative_to(git_root)).replace("\\", "/"))
            for p in files:
                arc = str(p.relative_to(git_root)).replace("\\", "/")
                if arc in added:
                    continue
                added.add(arc)
                zf.write(p, arc)
        meta = {
            "created_utc": ts + "Z",
            "glx_run": env.get("GLX_RUN", "?"),
            "project": env.get("GLX_PROJECT", "?"),
            "branch": env.get("GLX_BRANCH", "?"),
            "ver": env.get("GLX_VER", "?"),
            "anchors": {"env": "project_dir", "paths": "git"},
        }
        _zip_deterministic_writestr(zf, "GLX_AUDIT_META.json", json.dumps(meta, indent=2).encode("utf-8"))
    H.log("audit zip: %s" % zip_path.name)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# GÅ‚Ã³wna procedura
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _head_range(repo: Path) -> str:
    parent = H.rev_parse(repo, "HEAD^")
    head = H.rev_parse(repo, "HEAD")
    if head and parent:
        return "%s..%s" % (parent, head)
    return head or "HEAD"


def main() -> int:
    repo = H.git_root(THIS_DIR)
    project = repo  # projekt = katalog gÅ‚Ã³wny repo (wymÃ³g: ENV tylko stÄ…d)
    env = read_project_env(project)
    if not env:
        H.fail(".env/.env.local nie znaleziono w katalogu projektu: %s" % project)

    # Twarde kontrakty ENV
    env.setdefault("GLX_ROOT", str(project))
    if Path(env["GLX_ROOT"]).resolve() != project:
        H.fail("GLX_ROOT musi wskazywaÄ‡ katalog projektu (%s)" % project)

    ensure_required(env, ["GLX_RUN", "GLX_OUT", "GLX_AUTONOMY_OUT"])
    out_dir = ensure_inside_repo(Path(env["GLX_OUT"]), repo, "GLX_OUT")
    auto_dir = ensure_inside_repo(Path(env["GLX_AUTONOMY_OUT"]), repo, "GLX_AUTONOMY_OUT")

    mask, order = parse_glx_run(env.get("GLX_RUN", ""))
    if "Z" in order and not (("A" in order) or ("M" in order)):
        H.fail("GLX_RUN zawiera 'Z' bez 'A' lub 'M' â€” brak artefaktÃ³w do spakowania.")

    base_sha = detect_base(repo, repo / ".glx" / "state.json")
    H.log("post-commit: base_sha = %s" % base_sha)

    # A â€” autonomy.gateway (jeÅ›li dostÄ™pny)
    if "A" in order:
        pkg = env.get("GLX_PKG", "glitchlab")
        rc = H.run_glx_module(
            "gateway",
            repo,
            "%s.analysis.autonomy.gateway" % pkg,
            "analysis.autonomy.gateway",  # fallback dla starszych ukÅ‚adÃ³w
            args=["build", "--out", str(auto_dir)],
        )
        if rc != 0:
            H.fail("gateway: rc=%d" % rc)
        # zapis Å›rodowiska z redakcjÄ… sekretÃ³w
        redacted = {}
        for k, v in env.items():
            up = k.upper()
            if "PASS" in up or "SECRET" in up or "TOKEN" in up or "KEY" in up:
                redacted[k] = "******"
            else:
                redacted[k] = v
        (auto_dir / "env.json").write_text(json.dumps(redacted, indent=2), encoding="utf-8")

    # M â€” mozaika (hybrid_ast_mosaic)
    if "M" in order:
        # wymagane parametry Å›rodowiskowe (jak w starej wersji)
        req = ["GLX_ROWS", "GLX_COLS", "GLX_EDGE_THR", "GLX_DELTA", "GLX_KAPPA", "GLX_PHI", "GLX_MOSAIC"]
        ensure_required(env, req)
        phi = (env["GLX_PHI"] or "").lower()
        args = [
            "--mosaic",
            env["GLX_MOSAIC"],
            "--rows",
            env["GLX_ROWS"],
            "--cols",
            env["GLX_COLS"],
            "--edge-thr",
            env["GLX_EDGE_THR"],
            "--kappa-ab",
            env["GLX_KAPPA"],
            "--phi",
            phi,
            "--repo-root",
            str(repo),
            "from-git-dump",
            "--base",
            base_sha,
            "--head",
            "HEAD",
            "--delta",
            env["GLX_DELTA"],
            "--out",
            str(out_dir),
            "--strict-artifacts",
        ]
        if phi == "policy":
            pol = Path(env.get("GLX_POLICY", "analysis/policy.json"))
            pol = pol if pol.is_absolute() else (Path(env["GLX_ROOT"]) / pol)
            if not pol.exists():
                H.fail("GLX_PHI=policy, brak pliku polityki: %s" % pol)
            # wstaw przed --out
            insert_at = len(args) - 3
            args.insert(insert_at, "--policy-file")
            args.insert(insert_at + 1, str(pol))
        rc = H.run_glx_module(
            "mosaic",
            repo,
            "glitchlab.mosaic.hybrid_ast_mosaic",
            "mosaic.hybrid_ast_mosaic",  # fallback
            args=args,
        )
        if rc != 0:
            H.fail("mosaic: rc=%d" % rc)

    # E â€” exporters (best effort; nie blokuje, jeÅ›li CLI nie istnieje)
    if "E" in order:
        rc = H.run_glx_module(
            "exporters",
            repo,
            "glitchlab.analysis.exporters",
            "analysis.exporters",
            args=["--out", str(out_dir)],
        )
        if rc != 0:
            H.warn("exporters: moduÅ‚ CLI nieosiÄ…galny lub zakoÅ„czony z bÅ‚Ä™dem (pomijam)")

    # Z â€” archiwum AUDIT_*.zip
    if "Z" in order:
        make_audit_zip(env, repo)

    # â”€â”€ Nowa telemetria (nieblokujÄ…ca) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    rng = _head_range(repo)
    rc_df = H.run_glx_module(
        "delta_fingerprint",
        repo,
        "glx.tools.delta_fingerprint",
        "glitchlab.glx.tools.delta_fingerprint",
        args=["--range", rng],
    )
    if rc_df != 0:
        H.warn("delta_fingerprint: bÅ‚Ä…d (pomijam)")

    rc_inv = H.run_glx_module(
        "invariants_check",
        repo,
        "glx.tools.invariants_check",
        "glitchlab.glx.tools.invariants_check",
        args=["--range", rng],
    )
    if rc_inv != 0:
        H.warn("invariants_check: bÅ‚Ä…d/naruszenia (post-commit nie blokuje)")

    # Snippet do commitÃ³w
    try:
        out = H.write_commit_snippet(repo)
        if out:
            H.log("Zapisano snippet: %s" % out)
    except Exception:
        H.warn("Nie udaÅ‚o siÄ™ zapisaÄ‡ commit_snippet.txt")

    # Zdarzenie do dziennika
    evt = {"ts": datetime.utcnow().isoformat() + "Z", "event": "post-commit", "range": rng, "commit": H.rev_parse(repo, "HEAD")}
    try:
        H.append_jsonline(H.glx_dir(repo) / "events.log.jsonl", evt)
    except Exception:
        H.warn("Nie udaÅ‚o siÄ™ dopisaÄ‡ do .glx/events.log.jsonl")

    H.log("OK (order=%s)" % "".join(order))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        H.fail("Przerwano przez uÅ¼ytkownika")
