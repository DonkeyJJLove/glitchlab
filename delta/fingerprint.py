# glitchlab/delta/fingerprint.py
# -*- coding: utf-8 -*-
"""
Raport Δ oraz fingerprint (hash) — warstwa „delta”.

Funkcje:
  - build_delta_report(repo_root: Path, diff_range: str, *, policy: dict|None) -> DeltaReport

Właściwości:
  - Zgodność ze schematem `spec/schemas/delta_report.json` (klucze: range, hist, features, hash, meta).
  - Hash liczony z KANONICZNEJ, PO_SORTOWANEJ reprezentacji JSON:
      json.dumps(obj, sort_keys=True, separators=(',', ':')) → sha256.hexdigest()
  - Determinizm: do obiektu haszowanego NIE trafiają pola zmienne w czasie (np. created_utc).
  - Python 3.9 (bez wymagań na zewn. biblioteki).
"""
from __future__ import annotations

import json
import hashlib
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from typing import TypedDict

from .tokens import tokenize_diff, VOCAB_VERSION
from .features import build_features, FEATURES_VERSION


REPORT_VERSION = "v1"


class DeltaReport(TypedDict, total=False):
    range: str
    hist: Dict[str, int]
    features: Dict[str, float]
    hash: str
    meta: Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _now_utc_iso() -> string:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _parse_range(diff_range: str) -> (str, str):
    """
    Akceptuje:
      - 'A..B'  → (A,B)
      - 'HEAD'  → (HEAD~1, HEAD)
      - '<sha>' → (<sha>~1, <sha>)
    """
    dr = (diff_range or "").strip()
    if ".." in dr:
        a, b = dr.split("..", 1)
        return a.strip(), b.strip()
    return f"{dr}~1", dr

def _git_rev_parse(repo_root: Path, ref: str) -> Optional[str]:
    try:
        p = subprocess.run(
            ["git", "rev-parse", ref],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if p.returncode == 0:
            sha = p.stdout.strip()
            return sha if sha else None
    except Exception:
        pass
    return None

def _canonicalize_hist(hist: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in (hist or {}).items():
        try:
            kk = str(k)
            vv = int(v)
            if vv > 0:
                out[kk] = vv
        except Exception:
            continue
    # posortowane klucze zapewni json.dumps(sort_keys=True)
    return out

def _canonicalize_features(feats: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (feats or {}).items():
        try:
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                x = 0.0
            out[str(k)] = x
        except Exception:
            out[str(k)] = 0.0
    return out

def _fingerprint_payload(obj: Dict[str, Any]) -> bytes:
    # Kanoniczna serializacja: sort_keys + zwarte separatory → stabilny hash
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Główny interfejs
# ──────────────────────────────────────────────────────────────────────────────

def build_delta_report(repo_root: Path, diff_range: str, *, policy: Optional[dict] = None) -> DeltaReport:
    """
    Generuje raport Δ (VOCAB v1 → features v1 → hash v1).

    Zgodny schematycznie z `spec/schemas/delta_report.json`:
      {
        "range": "<A..B|HEAD>",
        "hist": { "T:ADD_FN": 2, "PATH:core": 3, ... },
        "features": { "tokens_total": 10.0, ... },
        "hash": "<sha256 hex>",
        "meta": {
          "report_version": "v1",
          "vocab_version":  "v1",
          "features_version":"v1",
          "resolved": {"base": "<sha>", "head": "<sha>"},
          "created_utc": "YYYY-MM-DDTHH:MM:SSZ"
        }
      }
    """
    repo = Path(repo_root)
    policy = policy or {}

    base_ref, head_ref = _parse_range(diff_range)
    base_sha = _git_rev_parse(repo, base_ref) or base_ref
    head_sha = _git_rev_parse(repo, head_ref) or head_ref

    # 1) Histogram Δ (VOCAB v1)
    hist_raw = tokenize_diff(repo, diff_range, policy=policy)
    hist = _canonicalize_hist(hist_raw)

    # 2) Cechy (FEATURES v1)
    features_raw = build_features(hist)
    features = _canonicalize_features(features_raw)

    # 3) Fingerprint — na bazie KANONICZNEGO obiektu bez pól lotnych (np. created_utc)
    canonical_obj = {
        "range": diff_range,
        "resolved": {"base": base_sha, "head": head_sha},
        "versions": {
            "report": REPORT_VERSION,
            "vocab": VOCAB_VERSION,
            "features": FEATURES_VERSION,
        },
        "hist": hist,
        "features": features,
    }
    digest = _sha256_hex(_fingerprint_payload(canonical_obj))

    # 4) Raport końcowy (meta ma created_utc, ale nie wpływa na hash)
    report: DeltaReport = {
        "range": diff_range,
        "hist": hist,
        "features": features,
        "hash": digest,
        "meta": {
            "report_version": REPORT_VERSION,
            "vocab_version": VOCAB_VERSION,
            "features_version": FEATURES_VERSION,
            "resolved": {"base": base_sha, "head": head_sha},
            "created_utc": _now_utc_iso(),
        },
    }
    return report
