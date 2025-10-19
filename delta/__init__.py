#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
delta.__init__ — SSOT dla Δ (tokeny, cechy, fingerprint)

Nowe, stabilne API (preferowane):
  - tokenize_diff(repo_root: Path, diff_range: str, *, policy: dict|None) -> dict[str,int]
  - build_features(hist: dict[str,int]) -> dict[str,float]
  - build_delta_report(repo_root: Path, diff_range: str, *, policy: dict|None) -> DeltaReport

Kompatybilność wsteczna (DEPRECATED, pozostawiona dla istniejących wywołań):
  - extract_from_sources(...)
  - extract_from_files(...)
  - extract_from_git(...)
  - fingerprint_from_tokens(hist) -> Fingerprint
  - features_from_tokens(hist) -> dict[str,float]
"""
from __future__ import annotations

import json
import hashlib
import warnings
from pathlib import Path
from typing import Dict, Optional, Any

# ──────────────────────────────────────────────────────────────────────────────
# Nowe API (re-exporty) + wersje
# ──────────────────────────────────────────────────────────────────────────────
try:
    from .tokens import tokenize_diff, Vocabulary, VOCAB_VERSION  # type: ignore
except Exception as _e:
    # Tymczasowy placeholder gdy plik jeszcze nie został wygenerowany w trakcie refaktoryzacji.
    def tokenize_diff(repo_root: Path, diff_range: str, *, policy: Optional[dict] = None) -> Dict[str, int]:
        raise ImportError("delta.tokens.tokenize_diff nie jest jeszcze dostępne") from _e


    class Vocabulary:  # type: ignore
        VERSION = "v0"


    VOCAB_VERSION = "v0"  # type: ignore

try:
    from .features import build_features, FEATURES_VERSION  # type: ignore
except Exception as _e:
    def build_features(hist: Dict[str, int]) -> Dict[str, float]:
        raise ImportError("delta.features.build_features nie jest jeszcze dostępne") from _e


    FEATURES_VERSION = "v0"  # type: ignore

# Typ raportu i budowa raportu (nowe API)
try:
    from .fingerprint import DeltaReport, build_delta_report, REPORT_VERSION  # type: ignore
except Exception as _e:
    # Minimalny, kompatybilny typ i funkcja — do czasu wygenerowania fingerprint.py
    from typing import TypedDict


    class DeltaReport(TypedDict, total=False):  # type: ignore
        range: str
        hist: Dict[str, int]
        features: Dict[str, float]
        hash: str
        meta: Dict[str, Any]


    def build_delta_report(repo_root: Path, diff_range: str, *, policy: Optional[dict] = None) -> "DeltaReport":
        raise ImportError("delta.fingerprint.build_delta_report nie jest jeszcze dostępne") from _e


    REPORT_VERSION = "v0"  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Warstwa kompatybilności wstecznej (DEPRECATED) — zachowana na czas migracji
# ──────────────────────────────────────────────────────────────────────────────

# extract_from_git → tokenize_diff
def extract_from_git(repo_root: Path, diff_range: str, policy: Optional[dict] = None) -> Dict[str, int]:
    """
    DEPRECATED: użyj tokenize_diff(repo_root, diff_range, policy=...)
    """
    warnings.warn("delta.extract_from_git jest przestarzałe; użyj delta.tokenize_diff",
                  DeprecationWarning, stacklevel=2)
    return tokenize_diff(Path(repo_root), str(diff_range), policy=policy)


# extract_from_files / extract_from_sources — jeśli istnieją w starym tokens.py, owiń z ostrzeżeniem.
# W przeciwnym razie podnieś kontrolowany wyjątek, aby nie mylić zachowania.
try:
    from .tokens import extract_from_files as _legacy_extract_from_files  # type: ignore
except Exception:
    def extract_from_files(*args, **kwargs):  # type: ignore
        warnings.warn("delta.extract_from_files jest przestarzałe i może nie być dostępne po refaktoryzacji; "
                      "preferuj Δ z zakresu gita przez tokenize_diff()", DeprecationWarning, stacklevel=2)
        raise NotImplementedError("extract_from_files nieobsługiwane w nowym API")
else:
    def extract_from_files(*args, **kwargs):  # type: ignore
        warnings.warn("delta.extract_from_files jest przestarzałe; preferuj tokenize_diff()",
                      DeprecationWarning, stacklevel=2)
        return _legacy_extract_from_files(*args, **kwargs)

try:
    from .tokens import extract_from_sources as _legacy_extract_from_sources  # type: ignore
except Exception:
    def extract_from_sources(*args, **kwargs):  # type: ignore
        warnings.warn("delta.extract_from_sources jest przestarzałe i może nie być dostępne po refaktoryzacji; "
                      "preferuj tokenize_diff()", DeprecationWarning, stacklevel=2)
        raise NotImplementedError("extract_from_sources nieobsługiwane w nowym API")
else:
    def extract_from_sources(*args, **kwargs):  # type: ignore
        warnings.warn("delta.extract_from_sources jest przestarzałe; preferuj tokenize_diff()",
                      DeprecationWarning, stacklevel=2)
        return _legacy_extract_from_sources(*args, **kwargs)


# fingerprint_from_tokens — zapewnijmy zachowanie (stabilny hash z kanonicznej serializacji)
class Fingerprint(str):
    """Wstecznie kompatybilny typ aliasujący fingerprint (SHA-256 w hex)."""
    pass


def fingerprint_from_tokens(hist: Dict[str, int]) -> Fingerprint:
    """
    DEPRECATED: preferuj build_delta_report(...).hash
    Zachowuje deterministyczne liczenie: SHA-256 z posortowanego JSON histogramu.
    """
    warnings.warn("delta.fingerprint_from_tokens jest przestarzałe; użyj build_delta_report(...).hash",
                  DeprecationWarning, stacklevel=2)
    payload = json.dumps(hist or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return Fingerprint(hashlib.sha256(payload).hexdigest())


# features_from_tokens — alias do nowego build_features
def features_from_tokens(hist: Dict[str, int]) -> Dict[str, float]:
    """
    DEPRECATED: preferuj build_features(hist)
    """
    warnings.warn("delta.features_from_tokens jest przestarzałe; użyj delta.build_features",
                  DeprecationWarning, stacklevel=2)
    return build_features(hist)


# ──────────────────────────────────────────────────────────────────────────────
# Publiczny interfejs
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Nowe API
    "tokenize_diff", "Vocabulary", "build_features", "build_delta_report",
    "DeltaReport", "VOCAB_VERSION", "FEATURES_VERSION", "REPORT_VERSION",
    # Kompatybilność wsteczna
    "extract_from_sources", "extract_from_files", "extract_from_git",
    "fingerprint_from_tokens", "Fingerprint", "features_from_tokens",
]
