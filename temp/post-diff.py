#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""
.githooks/post-diff.py — raport DIFF + Δ-tokens + (opcjonalnie) mozaika (Python 3.9)

Nie jest to standardowy hook Gita — traktuj jako narzędzie uruchamiane lokalnie
(ręcznie lub z aliasu), które tworzy zbiorczy raport z różnic między dwoma rewizjami.

Funkcje:
- Ustala zakres diff: --range A..B (domyślnie HEAD~1..HEAD) lub --base/--head.
- Zapisuje:
  * .glx/post_diff.json  — statystyki diff, lista plików, krótkie podsumowanie
  * .glx/diff_summary.txt — czytelny skrót zmian
- Uruchamia Δ-fingerprint (glx.tools.delta_fingerprint) i dołącza wynik (histogram + hash).
- (Opcjonalnie) generuje wizualizację mozaiki zmian (--with-mosaic) jako .glx/mosaic_diff.png (best-effort).
- Emisja zdarzenia do .glx/events.log.jsonl: event="post-diff".

Użycie:
    .githooks/post-diff.py [--range A..B] [--base <sha>] [--head <sha>] [--with-mosaic]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Helpery wspólne
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
import _common as H  # noqa: E402


def _resolve_range(repo: Path, args: argparse.Namespace) -> str:
    if args.range:
        return args.range
    if args.base and args.head:
        return f"{args.base}..{args.head}"
    # domyślnie HEAD~1..HEAD (a jeśli brak rodzica, zostanie przetworzone jako pojedynczy HEAD)
    parent = H.rev_parse(repo, "HEAD^")
    head = H.rev_parse(repo, "HEAD")
    if head and parent:
        return f"{parent}..{head}"
    return head or "HEAD"


def _git(repo: Path, *cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), cwd=str(repo), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _diff_stats(repo: Path, rng: str) -> Dict:
    # name-status (A/M/D/R/T)
    ns = _git(repo, "git", "diff", "--name-status", rng)
    added = modified = deleted = renamed = typechg = 0
    files_ns: List[Tuple[str, str]] = []
    for line in ns.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]
        path = parts[-1]
        files_ns.append((status, path))
        if status.startswith("A"):
            added += 1
        elif status.startswith("M"):
            modified += 1
        elif status.startswith("D"):
            deleted += 1
        elif status.startswith("R"):
            renamed += 1
        elif status.startswith("T"):
            typechg += 1

    # numstat (wstawienia/usunięcia per plik)
    num = _git(repo, "git", "diff", "--numstat", rng)
    insertions = deletions = 0
    details: List[Dict[str, int]] = []
    for line in num.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            ins, delt, path = parts[0], parts[1], parts[2]
            try:
                ins_v = 0 if ins == "-" else int(ins)
                del_v = 0 if delt == "-" else int(delt)
            except ValueError:
                ins_v = del_v = 0
            insertions += ins_v
            deletions += del_v
            details.append({"path": path, "insertions": ins_v, "deletions": del_v})

    return {
        "range": rng,
        "files": {
            "added": added,
            "modified": modified,
            "deleted": deleted,
            "renamed": renamed,
            "type_changed": typechg,
            "total": len(files_ns),
            "by_status": [{"status": s, "path": p} for s, p in files_ns],
        },
        "churn": {"insertions": insertions, "deletions": deletions, "total": insertions + deletions, "by_file": details},
    }


def _write_text_summary(glx: Path, stats: Dict, delta: Optional[Dict]) -> Path:
    out = glx / "diff_summary.txt"
    lines: List[str] = []
    lines.append(f"DIFF: {stats['range']}")
    f = stats["files"]
    c = stats["churn"]
    lines.append(f"Files: +{f['added']} ~{f['modified']} -{f['deleted']} R{f['renamed']} T{f['type_changed']} (total {f['total']})")
    lines.append(f"Churn: +{c['insertions']} -{c['deletions']} (sum {c['total']})")
    if delta:
        hist = delta.get("hist") or {}
        if hist:
            # top-k tokenów
            top = sorted(hist.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:8]
            lines.append("Δ-tokens: " + ", ".join(f"{k}×{v}" for k, v in top))
        if delta.get("hash"):
            lines.append(f"Fingerprint: {delta['hash']}")
    # krótka lista plików (pierwsze 30)
    by_status = f["by_status"][:30]
    if by_status:
        lines.append("Files (sample):")
        for s, p in by_status:
            lines.append(f"  {s}\t{p}")
        if f["total"] > 30:
            lines.append(f"  … ({f['total']-30} more)")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _load_delta(glx: Path) -> Optional[Dict]:
    p = glx / "delta_report.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _maybe_mosaic(repo: Path, out_dir: Path, rng: str) -> Optional[Path]:
    """
    Best-effort: próbujemy uruchomić mozaikę różnic (nieblokująco).
    Poszukujemy CLI w:
      - glitchlab.mosaic.hybrid_ast_mosaic  (nowa ścieżka)
      - mosaic.hybrid_ast_mosaic            (fallback)
    Subkomendy różnią się między wersjami, więc używamy bezpiecznej „preview-diff”.
    """
    # Preferowana nazwa pliku wyjściowego (niektóre CLI same wybierają nazwę)
    target = out_dir / "mosaic_diff.png"
    args = ["preview-diff", "--range", rng, "--out", str(out_dir)]
    rc = H.run_glx_module(
        "mosaic-preview",
        repo,
        "glitchlab.mosaic.hybrid_ast_mosaic",
        "mosaic.hybrid_ast_mosaic",
        args=args,
    )
    if rc == 0 and target.exists():
        return target
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--range", default=None, help="Zakres diff w formacie A..B lub pojedynczy commit")
    parser.add_argument("--base", default=None, help="Alternatywnie: baza (A)")
    parser.add_argument("--head", default=None, help="Alternatywnie: czubek (B)")
    parser.add_argument("--with-mosaic", action="store_true", help="Próba wygenerowania wizualizacji mozaiki zmian")
    args = parser.parse_args(argv)

    repo = H.git_root(THIS_DIR)
    glx = H.ensure_glx_dir(repo)

    rng = _resolve_range(repo, args)
    H.log(f"post-diff: zakres = {rng}")

    # Statystyki diff
    stats = _diff_stats(repo, rng)

    # Δ-fingerprint (best-effort)
    rc_df = H.run_glx_module(
        "delta_fingerprint",
        repo,
        "glx.tools.delta_fingerprint",
        "glitchlab.glx.tools.delta_fingerprint",
        args=["--range", rng],
    )
    if rc_df != 0:
        H.warn("delta_fingerprint: błąd (pomijam)")

    delta = _load_delta(glx)

    # Zapis JSON + czytelne podsumowanie
    (glx / "post_diff.json").write_text(json.dumps({"range": rng, "stats": stats, "delta": delta}, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = _write_text_summary(glx, stats, delta)
    H.log(f"post-diff: zapisano {summary_path}")

    # Opcjonalnie: mozaika
    if args.with_mosaic:
        mosaic_path = _maybe_mosaic(repo, glx, rng)
        if mosaic_path:
            H.log(f"post-diff: zapisano mozaikę {mosaic_path}")
        else:
            H.warn("post-diff: mozaika niedostępna w tej wersji CLI (pomijam)")

    # Zdarzenie do dziennika
    evt = {"event": "post-diff", "range": rng}
    try:
        H.append_jsonline(glx / "events.log.jsonl", evt)
    except Exception:
        H.warn("Nie udało się dopisać do .glx/events.log.jsonl")

    H.log("post-diff: OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        H.fail("Przerwano przez użytkownika")
