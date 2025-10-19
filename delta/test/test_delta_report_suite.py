# tests/delta/test_delta_report_suite.py
# -*- coding: utf-8 -*-
"""
Testy warstwy Δ (delta):
  1) stabilność fingerprintu (hash) dla tego samego zakresu,
  2) filtracja „śmieci” (np. *.png, .glx/*, __pycache__/),
  3) kształt i poprawność typów cech (brak NaN/Inf, kluczowe pola obecne).

Wymaga: git w PATH.
Python: 3.9
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict

import pytest

from glitchlab.delta import build_delta_report
from glitchlab.delta import build_features
from glitchlab.delta import tokenize_diff


# ──────────────────────────────────────────────────────────────────────────────
# Pomocnicze: mini API do pracy z lokalnym repozytorium git w tmp_path
# ──────────────────────────────────────────────────────────────────────────────

def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _git_ok(cwd: Path, *args: str) -> str:
    p = _git(cwd, *args)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def _repo_init(tmp: Path) -> Path:
    repo = tmp / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _git_ok(repo, "init", "-b", "main")
    _git_ok(repo, "config", "user.email", "test@example.com")
    _git_ok(repo, "config", "user.name", "Test User")
    return repo


def _write(repo: Path, rel: str, content: str) -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _commit_all(repo: Path, msg: str) -> str:
    _git_ok(repo, "add", "-A")
    _git_ok(repo, "commit", "-m", msg)
    return _git_ok(repo, "rev-parse", "HEAD")


def _range_last(repo: Path) -> str:
    # HEAD~1..HEAD, ale jeśli to pierwszy commit, używamy pustego drzewa (edge)
    out = _git(repo, "rev-parse", "HEAD~1")
    if out.returncode == 0 and out.stdout.strip():
        return "HEAD~1..HEAD"
    # fallback: puste drzewo → HEAD
    empty_tree = _git_ok(repo, "hash-object", "-t", "tree", "/dev/null")
    head = _git_ok(repo, "rev-parse", "HEAD")
    return f"{empty_tree}..{head}"


# ──────────────────────────────────────────────────────────────────────────────
# 1) Stabilność hash dla tego samego range
# ──────────────────────────────────────────────────────────────────────────────

def test_fingerprint_stability_same_range(tmp_path: Path) -> None:
    repo = _repo_init(tmp_path)

    # BASE
    _write(repo, "core/mod_a.py", "def foo(x):\n    return x + 1\n")
    _commit_all(repo, "base: add mod_a.py")

    # HEAD — modyfikacja ciała + „śmieć” PNG (powinien być zignorowany)
    _write(repo, "core/mod_a.py", "def foo(x):\n    return x + 2\n")  # body change
    _write(repo, "resources/img/test.png", "PNG\x00\x00garbage")  # fake binary
    _commit_all(repo, "head: modify body; add png")

    rng = _range_last(repo)

    rep1 = build_delta_report(repo, rng)
    rep2 = build_delta_report(repo, rng)

    assert isinstance(rep1["hash"], str) and len(rep1["hash"]) == 64
    assert rep1["hash"] == rep2["hash"], "Fingerprint musi być deterministyczny dla tego samego range"

    # Sanity: histogram zawiera typ modyfikacji ciała funkcji
    assert rep1["hist"].get("T:MODIFY_BODY_FN", 0) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# 2) Filtracja „śmieci” (np. PNG) — brak tokenów gdy zmiany dotyczą tylko ignorowanych plików
# ──────────────────────────────────────────────────────────────────────────────

def test_ignore_trash_only_png_changes(tmp_path: Path) -> None:
    repo = _repo_init(tmp_path)

    # BASE
    _write(repo, "docs/readme.md", "# doc\n")
    _commit_all(repo, "base: docs")

    # HEAD — tylko dodanie PNG (wg polityki defaultowej powinno być ignorowane)
    _write(repo, "resources/img/only.png", "PNG\x89garbage")
    _commit_all(repo, "head: add png")

    rng = _range_last(repo)
    hist = tokenize_diff(repo, rng)  # używa domyślnej polityki ignore

    # ŻADNE „T:” kubełki nie powinny się pojawić, bo nie ma sensownych tokenów
    t_buckets = {k: v for k, v in hist.items() if k.startswith("T:")}
    assert sum(t_buckets.values()) == 0, f"Oczekiwano braku tokenów, otrzymano: {t_buckets}"


# ──────────────────────────────────────────────────────────────────────────────
# 3) Kształt i typy cech (brak NaN/Inf, kluczowe pola)
# ──────────────────────────────────────────────────────────────────────────────

def test_features_shape_and_finiteness(tmp_path: Path) -> None:
    repo = _repo_init(tmp_path)

    # BASE
    _write(repo, "analysis/alfa.py", "def a():\n    return 1\n")
    _write(repo, "analysis/beta.py", "class B:\n    def x(self):\n        return 1\n")
    _commit_all(repo, "base: alfa+beta")

    # HEAD — modyfikacje (dodanie funkcji, zmiana sygnatury, usunięcie klasy)
    _write(repo, "analysis/alfa.py", "def a(z, y=1):\n    return z + y\n\ndef b():\n    return 0\n")
    # delete beta.py
    (repo / "analysis/beta.py").unlink(missing_ok=True)
    _commit_all(repo, "head: modify alfa (+b,+sig), delete beta")

    rng = _range_last(repo)
    rep = build_delta_report(repo, rng)

    feats = build_features(rep["hist"])

    required_keys = [
        "tokens_total", "distinct_types",
        "share_add", "share_del", "share_rename", "share_modify", "share_imports",
        "file_ops", "density", "churn", "churn_per_file",
        "type_entropy", "type_gini", "top_type_share",
        "lang_share.py", "lang_share.other",
        "kind_share.fn", "kind_share.class",
        "path_share.analysis", "path_share.other"
    ]
    for k in required_keys:
        assert k in feats, f"Brakuje cechy: {k}"
        v = float(feats[k])
        assert not math.isnan(v) and not math.isinf(v), f"Cecha {k} ma niepoprawną wartość: {v}"

    # Zmiany dotyczyły głównie Pythona i gałęzi 'analysis'
    assert feats["lang_share.py"] >= 0.5
    assert feats["path_share.analysis"] >= 0.5
