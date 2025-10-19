#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glitchlab.io.artifacts — jednolite API do artefaktów GLX (.glx/*) • Python 3.9

Cel
----
Zapewnić spójny, bezpieczny i przewidywalny mechanizm zapisu/odczytu artefaktów
w katalogu `glitchlab/.glx/` oraz prostą obsługę kopii audytowych (ZIP),
retencji i stanu (np. `base_sha`).

Założenia
---------
- Tylko stdlib; brak zewnętrznych zależności.
- Zapis atomowy (tmp → rename).
- Walidacja, że wszystkie ścieżki docelowe znajdują się *wewnątrz repo*.
- Udostępnia funkcje wysokiego poziomu oraz klasę `GlxArtifacts`.

Układ artefaktów (kanoniczny)
-----------------------------
.glx/
  ├─ delta_report.json         (wynik glx.tools.delta_fingerprint)
  ├─ commit_analysis.json      (wynik glx.tools.invariants_check)
  ├─ spec_state.json           (progi α/β/Z i metadane kalibracji)
  ├─ state.json                (np. { "base_sha": "<sha>" })
  ├─ events.log.jsonl          (strumień zdarzeń lokalnych)
  ├─ commit_snippet.txt        (prefiks do wiadomości commit)
  ├─ post_diff.json            (raport z post-diff)
  ├─ diff_summary.txt          (ludzki skrót zmian)
  ├─ mosaic_diff.png           (opcjonalnie wizualizacja różnic)
  └─ *.png / *.json            (inne artefakty niekrytyczne)
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# ──────────────────────────────────────────────────────────────────────────────
# Stałe: nazwy artefaktów i domyślne progi
# ──────────────────────────────────────────────────────────────────────────────

ART_DELTA_REPORT = "delta_report.json"
ART_COMMIT_ANALYSIS = "commit_analysis.json"
ART_SPEC_STATE = "spec_state.json"
ART_STATE = "state.json"
ART_EVENTS_LOG = "events.log.jsonl"
ART_COMMIT_SNIPPET = "commit_snippet.txt"
ART_POST_DIFF_JSON = "post_diff.json"
ART_DIFF_SUMMARY_TXT = "diff_summary.txt"
ART_MOSAIC_DIFF_PNG = "mosaic_diff.png"

DEFAULT_THRESHOLDS = {"alpha": 0.85, "beta": 0.92, "z": 0.99}

# ──────────────────────────────────────────────────────────────────────────────
# Pomocnicze: repo root i ścieżki .glx/*
# ──────────────────────────────────────────────────────────────────────────────


def _git_root(start: Path) -> Path:
    """Wykryj katalog główny repozytorium Git; fallback: dwa poziomy w górę."""
    start = start.resolve()
    try:
        import subprocess

        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(start),
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip()).resolve()
    except Exception:
        pass
    # fallback: standardowa lokalizacja (np. glitchlab/io/* → root to 3 poziomy wyżej)
    parents = list(start.parents)
    return (parents[2] if len(parents) >= 3 else start).resolve()


def _ensure_inside_repo(path: Path, repo_root: Path, *, name: str = "path") -> Path:
    """Waliduje, że `path` leży wewnątrz `repo_root`; tworzy katalogi nadrzędne."""
    p = path.resolve()
    r = repo_root.resolve()
    try:
        p.relative_to(r)
    except Exception:
        raise ValueError(f"{name} musi wskazywać ścieżkę wewnątrz repo: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Zapis atomowy pliku bajtowego."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    _atomic_write_bytes(path, text.encode(encoding))


def _atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent))


# ──────────────────────────────────────────────────────────────────────────────
# Główny interfejs
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GlxPaths:
    """Wyliczenie kanonicznych ścieżek w obrębie .glx/*."""

    dir: Path
    delta_report: Path
    commit_analysis: Path
    spec_state: Path
    state: Path
    events_log: Path
    commit_snippet: Path
    post_diff_json: Path
    diff_summary_txt: Path
    mosaic_diff_png: Path

    @staticmethod
    def from_root(repo_root: Path) -> "GlxPaths":
        base = (repo_root / "glitchlab" / ".glx").resolve()
        return GlxPaths(
            dir=base,
            delta_report=base / ART_DELTA_REPORT,
            commit_analysis=base / ART_COMMIT_ANALYSIS,
            spec_state=base / ART_SPEC_STATE,
            state=base / ART_STATE,
            events_log=base / ART_EVENTS_LOG,
            commit_snippet=base / ART_COMMIT_SNIPPET,
            post_diff_json=base / ART_POST_DIFF_JSON,
            diff_summary_txt=base / ART_DIFF_SUMMARY_TXT,
            mosaic_diff_png=base / ART_MOSAIC_DIFF_PNG,
        )


class GlxArtifacts:
    """
    Jednolity dostęp do artefaktów GLX.

    Przykład:
        af = GlxArtifacts()                        # autodetekcja repo root
        af.write_delta_report({"hist": {...}})
        af.append_event({"event": "post-commit", "ts": time.time()})
        af.make_audit_zip(glx_out=..., autonomy_out=..., backups_dir=repo_root/"backup")
    """

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self.repo_root: Path = (repo_root or _git_root(Path(__file__))).resolve()
        self.paths: GlxPaths = GlxPaths.from_root(self.repo_root)
        self.paths.dir.mkdir(parents=True, exist_ok=True)

    # ── zapisy/odczyty podstawowe ────────────────────────────────────────────

    def write_json(self, rel_name: str, obj: Any, *, indent: int = 2) -> Path:
        p = _ensure_inside_repo(self.paths.dir / rel_name, self.repo_root, name=rel_name)
        _atomic_write_json(p, obj, indent=indent)
        return p

    def read_json(self, rel_name: str, default: Any = None) -> Any:
        p = self.paths.dir / rel_name
        if not p.exists():
            return default
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default

    def write_text(self, rel_name: str, text: str) -> Path:
        p = _ensure_inside_repo(self.paths.dir / rel_name, self.repo_root, name=rel_name)
        _atomic_write_text(p, text)
        return p

    def write_bytes(self, rel_name: str, data: bytes) -> Path:
        p = _ensure_inside_repo(self.paths.dir / rel_name, self.repo_root, name=rel_name)
        _atomic_write_bytes(p, data)
        return p

    def append_jsonline(self, rel_name: str, line: Dict[str, Any]) -> Path:
        p = _ensure_inside_repo(self.paths.dir / rel_name, self.repo_root, name=rel_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return p

    # ── artefakty kanoniczne ────────────────────────────────────────────────

    # delta_report.json
    def write_delta_report(self, report: Dict[str, Any]) -> Path:
        return self.write_json(ART_DELTA_REPORT, report)

    def read_delta_report(self) -> Dict[str, Any]:
        return self.read_json(ART_DELTA_REPORT, default={}) or {}

    # commit_analysis.json
    def write_commit_analysis(self, analysis: Dict[str, Any]) -> Path:
        return self.write_json(ART_COMMIT_ANALYSIS, analysis)

    def read_commit_analysis(self) -> Dict[str, Any]:
        return self.read_json(ART_COMMIT_ANALYSIS, default={}) or {}

    # spec_state.json (progi, kalibracja)
    def read_thresholds(self) -> Dict[str, float]:
        obj = self.read_json(ART_SPEC_STATE, default=None)
        if obj and isinstance(obj, dict):
            repo_t = (obj.get("thresholds") or {}).get("repo") or {}
            try:
                return {
                    "alpha": float(repo_t.get("alpha", DEFAULT_THRESHOLDS["alpha"])),
                    "beta": float(repo_t.get("beta", DEFAULT_THRESHOLDS["beta"])),
                    "z": float(repo_t.get("z", DEFAULT_THRESHOLDS["z"])),
                }
            except Exception:
                pass
        # baseline
        base = {
            "ts": time.time(),
            "note": "baseline (auto)",
            "thresholds": {"repo": dict(DEFAULT_THRESHOLDS)},
        }
        self.write_json(ART_SPEC_STATE, base)
        return dict(DEFAULT_THRESHOLDS)

    def update_thresholds(
        self,
        *,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        z: Optional[float] = None,
        note: str = "manual update",
    ) -> Path:
        cur = self.read_json(ART_SPEC_STATE, default={}) or {}
        repo = ((cur.get("thresholds") or {}).get("repo") or {}).copy()
        if alpha is not None:
            repo["alpha"] = float(alpha)
        if beta is not None:
            repo["beta"] = float(beta)
        if z is not None:
            repo["z"] = float(z)
        obj = {
            "ts": time.time(),
            "note": note,
            "thresholds": {"repo": repo or dict(DEFAULT_THRESHOLDS)},
        }
        return self.write_json(ART_SPEC_STATE, obj)

    # state.json (np. base_sha)
    def read_state(self) -> Dict[str, Any]:
        return self.read_json(ART_STATE, default={}) or {}

    def write_state(self, **kv: Any) -> Path:
        state = self.read_state()
        state.update(kv)
        return self.write_json(ART_STATE, state)

    # events.log.jsonl
    def append_event(self, event: Dict[str, Any]) -> Path:
        if "ts" not in event:
            event = dict(event)
            event["ts"] = time.time()
        return self.append_jsonline(ART_EVENTS_LOG, event)

    # commit_snippet.txt (budowany zwykle z delta_report)
    def write_commit_snippet(self, text: str) -> Path:
        return self.write_text(ART_COMMIT_SNIPPET, text.rstrip() + "\n")

    def build_commit_snippet_from_delta(self, top_k: int = 8) -> str:
        rep = self.read_delta_report()
        hist = rep.get("hist") or {}
        fp = rep.get("hash") or ""
        if not hist and not fp:
            return ""
        top = sorted(((k, int(v)) for k, v in hist.items()), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        lines = []
        lines.append("[GLX] Δ-tokens: " + (", ".join(f"{k}×{v}" for k, v in top) if top else "—"))
        if fp:
            lines.append(f"[GLX] Fingerprint: {fp}")
        return "\n".join(lines)

    # post_diff.json / diff_summary.txt / mosaic_diff.png
    def write_post_diff(self, payload: Dict[str, Any], summary: str) -> Tuple[Path, Path]:
        p_json = self.write_json(ART_POST_DIFF_JSON, payload)
        p_summary = self.write_text(ART_DIFF_SUMMARY_TXT, summary.rstrip() + "\n")
        return p_json, p_summary

    # ── retencja / listing ───────────────────────────────────────────────────

    def list_artifacts(self) -> List[Path]:
        if not self.paths.dir.exists():
            return []
        return sorted(p for p in self.paths.dir.iterdir() if p.is_file())

    def rotate_by_keep(self, pattern: str, keep: int = 5) -> None:
        """
        Zachowaj ostatnie `keep` plików dla wzorca (glob w .glx/*), resztę usuń.
        Uwaga: sortujemy po mtime malejąco.
        """
        files = sorted((self.paths.dir.glob(pattern)), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[keep:]:
            try:
                p.unlink()
            except Exception:
                pass

    # ── audyt: deterministyczny ZIP ──────────────────────────────────────────

    @staticmethod
    def _zip_writestr_deterministic(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
        zi = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, data)

    def make_audit_zip(
        self,
        *,
        glx_out: Path,
        autonomy_out: Optional[Path] = None,
        backups_dir: Optional[Path] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Tworzy deterministyczne archiwum ZIP z zawartością `glx_out` oraz (opcjonalnie) `autonomy_out`.
        Archiwum trafia do `backups_dir` (domyślnie <repo>/backup) z nazwą AUDIT_YYYYmmdd-HHMMSS.zip.
        Plik zawiera też GLX_AUDIT_META.json.
        """
        repo = self.repo_root
        glx_out = _ensure_inside_repo(Path(glx_out), repo, name="glx_out")
        autonomy_out = Path(autonomy_out).resolve() if autonomy_out else None
        if autonomy_out:
            _ensure_inside_repo(autonomy_out, repo, name="autonomy_out")

        backups = backups_dir or (repo / "backup")
        backups.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        zip_path = backups / f"AUDIT_{ts}.zip"

        def should_zip(p: Path) -> bool:
            rel = str(p).replace("\\", "/")
            return p.is_file() and "/.git/" not in rel and "/__pycache__/" not in rel

        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            added: set = set()

            def add_dir(root: Path) -> None:
                files = [p for p in root.rglob("*") if should_zip(p)]
                files.sort(key=lambda p: str(p.relative_to(repo)).replace("\\", "/"))
                for p in files:
                    arc = str(p.relative_to(repo)).replace("\\", "/")
                    if arc in added:
                        continue
                    added.add(arc)
                    zf.write(p, arc)

            add_dir(glx_out)
            if autonomy_out and autonomy_out.exists():
                add_dir(autonomy_out)

            meta_obj = {
                "created_utc": ts + "Z",
                "project_root": str(repo),
                "glx_out": str(glx_out),
                "autonomy_out": str(autonomy_out) if autonomy_out else None,
            }
            if meta:
                meta_obj.update(meta)
            self._zip_writestr_deterministic(
                zf, "GLX_AUDIT_META.json", json.dumps(meta_obj, indent=2).encode("utf-8")
            )

        return zip_path


# ──────────────────────────────────────────────────────────────────────────────
# Interfejs proceduralny (opcjonalny)
# ──────────────────────────────────────────────────────────────────────────────

# Wygodne funkcje skrótowe (z domyślną instancją):

_DEF = GlxArtifacts()


def write_delta_report(obj: Dict[str, Any]) -> Path:
    return _DEF.write_delta_report(obj)


def write_commit_analysis(obj: Dict[str, Any]) -> Path:
    return _DEF.write_commit_analysis(obj)


def read_thresholds() -> Dict[str, float]:
    return _DEF.read_thresholds()


def update_thresholds(*, alpha: float = None, beta: float = None, z: float = None, note: str = "manual update") -> Path:
    return _DEF.update_thresholds(alpha=alpha, beta=beta, z=z, note=note)


def append_event(evt: Dict[str, Any]) -> Path:
    return _DEF.append_event(evt)


def build_commit_snippet_from_delta(top_k: int = 8) -> str:
    return _DEF.build_commit_snippet_from_delta(top_k=top_k)


def write_commit_snippet(text: str) -> Path:
    return _DEF.write_commit_snippet(text)


def make_audit_zip(*, glx_out: Path, autonomy_out: Path = None, backups_dir: Path = None, meta: Dict[str, Any] = None) -> Path:
    return _DEF.make_audit_zip(glx_out=glx_out, autonomy_out=autonomy_out, backups_dir=backups_dir, meta=meta)


# ──────────────────────────────────────────────────────────────────────────────
# CLI minimalistyczne
# ──────────────────────────────────────────────────────────────────────────────

def _cli(argv: List[str]) -> int:
    """
    Prosty CLI (debug/ops):

      python -m glitchlab.io.artifacts thresholds             → wypisz progi
      python -m glitchlab.io.artifacts thresholds set a b z   → ustaw progi
      python -m glitchlab.io.artifacts snippet                → wygeneruj i zapisz commit_snippet.txt
      python -m glitchlab.io.artifacts audit <glx_out> [auto] → zrób ZIP audytowy
    """
    if not argv or argv[0] in ("-h", "--help", "help"):
        sys.stdout.write(_cli.__doc__ or "")
        return 0

    af = GlxArtifacts()
    cmd, *rest = argv

    if cmd == "thresholds":
        if rest and rest[0] == "set" and len(rest) >= 4:
            a, b, z = float(rest[1]), float(rest[2]), float(rest[3])
            af.update_thresholds(alpha=a, beta=b, z=z, note="cli update")
            print(json.dumps(af.read_thresholds(), ensure_ascii=False))
            return 0
        print(json.dumps(af.read_thresholds(), ensure_ascii=False))
        return 0

    if cmd == "snippet":
        s = af.build_commit_snippet_from_delta()
        if not s:
            print("no delta_report.json / empty histogram", file=sys.stderr)
            return 1
        af.write_commit_snippet(s)
        print(s)
        return 0

    if cmd == "audit":
        if not rest:
            print("usage: audit <glx_out> [autonomy_out]", file=sys.stderr)
            return 2
        glx_out = Path(rest[0])
        auto_out = Path(rest[1]) if len(rest) >= 2 else None
        z = af.make_audit_zip(glx_out=glx_out, autonomy_out=auto_out)
        print(str(z))
        return 0

    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
