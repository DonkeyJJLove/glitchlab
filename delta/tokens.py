# glitchlab/delta/tokens.py
# -*- coding: utf-8 -*-
"""
Δ-tokenizacja zmian kodu oraz słownik (SSOT) dla warstwy „delta”.

Zakres:
- Ekstrakcja bogatych tokenów zmian na bazie porównania AST (Python) + zdarzenia plikowe (inne języki).
- Integracja z Gitem (base..head) bez zależności od warstw „analysis/*”.
- Budowa histogramu (VOCAB v1) z listy tokenów.
- Stabilność i determinizm (posortowane klucze, kanoniczne formy).

Python: 3.9
Zależności opcjonalne (jeśli dostępne): libcst, radon

Publiczne API:
- class Token(dataclass)
- class Vocabulary (VOCAB_VERSION="v1")
- extract_from_sources(prev_src, curr_src, *, filename) -> List[Token]
- extract_from_files(prev_file, curr_file) -> List[Token]
- extract_from_git(repo, base, head, policy=None) -> List[Token]
- tokens_to_hist(tokens, policy=None) -> Dict[str, int]
- tokenize_diff(repo_root, diff_range, *, policy=None) -> Dict[str, int]
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Zależności opcjonalne
# ──────────────────────────────────────────────────────────────────────────────
try:
    import libcst as cst  # type: ignore
except Exception:  # pragma: no cover
    cst = None  # type: ignore

try:
    from radon.complexity import cc_visit  # type: ignore
except Exception:  # pragma: no cover
    cc_visit = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Token:
    """Δ-token opisujący elementarną zmianę."""
    type: str                # np. ADD_FN, DEL_FN, MODIFY_SIG_FN, ΔIMPORT+, ADD_FILE, MOVE_FILE…
    file: str                # ścieżka relatywna względem repo_root
    target: str              # np. qualname funkcji/klasy, ścieżka importu, lub nazwa pliku
    meta: Dict[str, object]  # dane dodatkowe (before/after, rename.to, itp.)


# ──────────────────────────────────────────────────────────────────────────────
# Pomocnicze
# ──────────────────────────────────────────────────────────────────────────────

def _read_text(p: Optional[Path]) -> str:
    return p.read_text(encoding="utf-8", errors="replace") if p and p.exists() else ""

def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _normalize_src(src: str) -> str:
    # Normalizacja końców linii i trimming końcowych spacji (stabilność AST)
    return "\n".join(line.rstrip() for line in src.replace("\r\n", "\n").replace("\r", "\n").split("\n"))

def _ast_index(src: str) -> Dict[str, Dict[str, object]]:
    """
    Lekki indeks definicji/importów wraz z sygnaturami oraz skrótem ciała.
    Klucze: 'def:<qualname>' | 'class:<qualname>' | 'imp:<module.or.name>'
    """
    idx: Dict[str, Dict[str, object]] = {}
    try:
        tree = ast.parse(src)
    except Exception:
        return idx

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: List[str] = []

        def _qual(self, name: str) -> str:
            return ".".join([*self.scope, name]) if self.scope else name

        @staticmethod
        def _body_hash(node: ast.AST) -> str:
            # Stabilizacja: usuwamy atrybuty pozycyjne zanim wykonamy dump
            try:
                for n in ast.walk(node):
                    for a in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
                        if hasattr(n, a):
                            setattr(n, a, None)
                dumped = ast.dump(node, include_attributes=False)
            except Exception:
                dumped = repr(node)
            return _hash_bytes(dumped.encode("utf-8"))

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            sig = {
                "args": [a.arg for a in node.args.args],
                "defaults": len(node.args.defaults),
                "vararg": bool(node.args.vararg),
                "kwonly": [a.arg for a in node.args.kwonlyargs],
                "kw_defaults": len([d for d in node.args.kw_defaults if d is not None]),
                "kwarg": bool(node.args.kwarg),
                "returns": (ast.unparse(node.returns) if hasattr(ast, "unparse") and node.returns else None),
                "decorators": [ast.unparse(d) if hasattr(ast, "unparse") else "" for d in node.decorator_list],
            }
            q = self._qual(node.name)
            idx[f"def:{q}"] = {
                "kind": "fn",
                "name": node.name,
                "sig": sig,
                "public": not node.name.startswith("_"),
                "body": self._body_hash(node),
            }
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.visit_FunctionDef(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            q = self._qual(node.name)
            idx[f"class:{q}"] = {
                "kind": "class",
                "name": node.name,
                "bases": [ast.unparse(b) if hasattr(ast, "unparse") else "" for b in node.bases],
                "decorators": [ast.unparse(d) if hasattr(ast, "unparse") else "" for d in node.decorator_list],
                "public": not node.name.startswith("_"),
                "body": self._body_hash(node),
            }
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                idx[f"imp:{alias.name}"] = {"kind": "imp", "name": alias.name, "asname": alias.asname}

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            mod = node.module or ""
            for alias in node.names:
                idx[f"imp:{mod}.{alias.name}"] = {"kind": "imp", "name": f"{mod}.{alias.name}", "asname": alias.asname}

    Visitor().visit(tree)
    return idx

def _complexity_map(src: str) -> Dict[str, int]:
    """Zwraca mapę złożoności cyklomatycznej (radon, jeśli dostępny)."""
    out: Dict[str, int] = {}
    if cc_visit is None:
        return out
    try:
        for b in cc_visit(src):
            q = b.fullname.split(":", 1)[-1]
            out[q] = int(b.complexity)
    except Exception:
        pass
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Porównanie indeksów AST → tokeny
# ──────────────────────────────────────────────────────────────────────────────

def _match_by_body(prev_defs: Dict[str, Dict[str, object]], curr_defs: Dict[str, Dict[str, object]]) -> Dict[str, str]:
    """Wykrywa potencjalne rename po identycznym skrócie ciała (prev_key → curr_key)."""
    prev_by_body: Dict[str, str] = {}
    for k, v in prev_defs.items():
        if v.get("kind") in {"fn", "class"}:
            prev_by_body[v.get("body", "")] = k
    mapping: Dict[str, str] = {}
    for k, v in curr_defs.items():
        if v.get("kind") in {"fn", "class"}:
            body = v.get("body", "")
            if body in prev_by_body and prev_by_body[body] != k:
                mapping[prev_by_body[body]] = k
    return mapping

def compare_indices(prev_idx: Dict[str, Dict[str, object]], curr_idx: Dict[str, Dict[str, object]], *, file: str) -> List[Token]:
    tokens: List[Token] = []

    # Importy
    prev_imps = {k: v for k, v in prev_idx.items() if k.startswith("imp:")}
    curr_imps = {k: v for k, v in curr_idx.items() if k.startswith("imp:")}
    for k in sorted(set(prev_imps) - set(curr_imps)):
        tokens.append(Token("ΔIMPORT-", file, k[4:], {"as": prev_imps[k].get("asname")}))
    for k in sorted(set(curr_imps) - set(prev_imps)):
        tokens.append(Token("ΔIMPORT+", file, k[4:], {"as": curr_imps[k].get("asname")}))

    # Definicje/klasy
    prev_defs = {k: v for k, v in prev_idx.items() if k.startswith(("def:", "class:"))}
    curr_defs = {k: v for k, v in curr_idx.items() if k.startswith(("def:", "class:"))}

    # Rename po hashach ciał
    rename_map = _match_by_body(prev_defs, curr_defs)  # prev_key -> curr_key

    # Usunięcia (bez rename)
    for k in sorted(set(prev_defs) - set(curr_defs) - set(rename_map.keys())):
        kind = prev_defs[k]["kind"]
        name = k.split(":", 1)[1]
        tokens.append(Token(("DEL_CLASS" if kind == "class" else "DEL_FN"), file, name, {}))

    # Dodania (bez rename)
    for k in sorted(set(curr_defs) - set(prev_defs) - set(rename_map.values())):
        kind = curr_defs[k]["kind"]
        name = k.split(":", 1)[1]
        tokens.append(Token(("ADD_CLASS" if kind == "class" else "ADD_FN"), file, name, {}))

    # Rename
    for prev_k, curr_k in sorted(rename_map.items()):
        kind = curr_defs[curr_k]["kind"]
        prev_name = prev_k.split(":", 1)[1]
        curr_name = curr_k.split(":", 1)[1]
        tokens.append(Token(("RENAME_CLASS" if kind == "class" else "RENAME_FN"), file, prev_name, {"to": curr_name}))

    # Modyfikacje
    intersect = (set(prev_defs) & set(curr_defs)) | set(rename_map.values())
    for k in sorted(intersect):
        prev = prev_defs.get(k) or prev_defs.get(next((p for p, c in rename_map.items() if c == k), ""))
        curr = curr_defs[k]
        name = k.split(":", 1)[1]
        if prev and curr:
            if prev["kind"] == "fn" and prev.get("sig") != curr.get("sig"):
                tokens.append(Token("MODIFY_SIG_FN", file, name, {"before": prev.get("sig"), "after": curr.get("sig")}))
            if prev.get("body") != curr.get("body"):
                tokens.append(Token(("MODIFY_BODY_CLASS" if curr["kind"] == "class" else "MODIFY_BODY_FN"), file, name, {}))

    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Ekstrakcja z tekstów / plików
# ──────────────────────────────────────────────────────────────────────────────

def extract_from_sources(prev_src: Optional[str], curr_src: Optional[str], *, filename: str = "<memory>") -> List[Token]:
    prev_src_n = _normalize_src(prev_src or "")
    curr_src_n = _normalize_src(curr_src or "")
    prev_idx = _ast_index(prev_src_n) if prev_src_n.strip() else {}
    curr_idx = _ast_index(curr_src_n) if curr_src_n.strip() else {}
    return compare_indices(prev_idx, curr_idx, file=filename)

def extract_from_files(prev_file: Optional[Path], curr_file: Optional[Path]) -> List[Token]:
    prev_src = _read_text(prev_file) if prev_file else ""
    curr_src = _read_text(curr_file) if curr_file else ""
    filename = str((curr_file or prev_file or Path("<memory>")).as_posix())
    return extract_from_sources(prev_src, curr_src, filename=filename)


# ──────────────────────────────────────────────────────────────────────────────
# Git helpers
# ──────────────────────────────────────────────────────────────────────────────

def _git(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def _git_text(args: List[str], cwd: Path) -> str:
    r = _git(args, cwd)
    return r.stdout if r.returncode == 0 else ""

def _changed_files(repo: Path, base: str, head: str) -> List[Tuple[str, str, str]]:
    """
    Zwraca listę (status, old_path, new_path) dla wszystkich plików (A,M,D,R…).
    Używa `git diff --name-status -M -C base..head`.
    """
    out: List[Tuple[str, str, str]] = []
    raw = _git_text(["diff", "--name-status", "-M", "-C", f"{base}..{head}"], repo)
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") and len(parts) >= 3:
            out.append(("R", parts[1], parts[2]))
        elif len(parts) >= 2:
            p = parts[1]
            out.append((status[0], p, p))
    return out

def _git_show(repo: Path, rev: str, path: str) -> str:
    return _git_text(["show", f"{rev}:{path}"], repo)


# ──────────────────────────────────────────────────────────────────────────────
# Ekstrakcja z Gita
# ──────────────────────────────────────────────────────────────────────────────

def extract_from_git(repo: Path, base: str, head: str, policy: Optional[dict] = None) -> List[Token]:
    """
    Buduje tokeny Δ dla wszystkich zmienionych plików między base..head.
    - Dla *.py wykonuje porównanie AST (def/class/import + rename/sig/body).
    - Dla innych typów dodaje tokeny plikowe (ADD_FILE/DEL_FILE/MOVE_FILE/MOD_FILE).
    """
    policy = policy or {}
    ignore_re = policy.get(
        "ignore_re",
        r"(^|/)(\.git|\.glx|__pycache__|\.pytest_cache|\.mypy_cache|\.venv|env|build|dist)/|\.png$|\.jpg$|\.jpeg$|\.ico$|\.zip$|\.bin$",
    )
    ignore = re.compile(ignore_re)

    tokens: List[Token] = []
    files = _changed_files(repo, base, head)

    for status, old_p, new_p in files:
        # pomijamy śmieci / binaria wg polityki
        if ignore.search(old_p) or ignore.search(new_p):
            continue

        is_py_old = old_p.endswith(".py")
        is_py_new = new_p.endswith(".py")
        prev_src = _git_show(repo, base, old_p) if status in {"M", "D", "R"} else ""
        curr_src = _git_show(repo, head, new_p) if status in {"M", "A", "R"} else ""

        # Preferujemy nową ścieżkę jako kontekst
        fname = new_p if status != "D" else old_p

        # Tokeny plikowe (zawsze)
        if status == "R":
            tokens.append(Token("MOVE_FILE", new_p, old_p, {"from": old_p, "to": new_p}))
        elif status == "A":
            tokens.append(Token("ADD_FILE", new_p, new_p, {}))
        elif status == "D":
            tokens.append(Token("DEL_FILE", old_p, old_p, {}))
        elif status == "M":
            tokens.append(Token("MOD_FILE", fname, fname, {}))

        # AST tylko dla Pythona
        if is_py_old or is_py_new:
            tokens.extend(extract_from_sources(prev_src, curr_src, filename=fname))

    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# VOCABULARY v1 — budowa histogramu z tokenów
# ──────────────────────────────────────────────────────────────────────────────

VOCAB_VERSION = "v1"

class Vocabulary:
    """
    Słownik mapujący Token → klucze histogramu (VOCAB v1).
    Zasady:
      • Zawsze inkrementujemy klucz typu tokena: f"T:{token.type}"
      • Dodajemy kubełek ścieżki top-level: f"PATH:{top}" (app|core|analysis|mosaic|delta|docs|tests|bench|io|spec|…)
      • Dodajemy język: LANG:py | LANG:other
      • Dla funkcji/klas dopinamy KIND:fn|class, gdy możliwe.
    """
    TOPS = ("app", "core", "analysis", "mosaic", "delta", "docs", "tests", "bench", "io", "spec")
    LANG_PY = "LANG:py"
    LANG_OTHER = "LANG:other"

    @staticmethod
    def top_bucket(path: str) -> str:
        p = path.lstrip("/").split("/", 1)[0] if path else ""
        return f"PATH:{p}" if p in Vocabulary.TOPS else "PATH:other"

    @staticmethod
    def lang_bucket(path: str) -> str:
        return Vocabulary.LANG_PY if path.endswith(".py") else Vocabulary.LANG_OTHER

    @staticmethod
    def kind_bucket(token: Token) -> Optional[str]:
        if token.type.endswith("_FN") or token.type in ("RENAME_FN", "MODIFY_SIG_FN", "MODIFY_BODY_FN"):
            return "KIND:fn"
        if token.type.endswith("_CLASS") or token.type in ("RENAME_CLASS", "MODIFY_BODY_CLASS"):
            return "KIND:class"
        return None

    def expand(self, token: Token) -> List[str]:
        keys = [f"T:{token.type}", self.top_bucket(token.file), self.lang_bucket(token.file)]
        k = self.kind_bucket(token)
        if k:
            keys.append(k)
        return keys


def tokens_to_hist(tokens: Iterable[Token], policy: Optional[dict] = None) -> Dict[str, int]:
    """
    Buduje histogram kluczy wg VOCAB v1.
    Polityka może zawierać:
      - include_types / exclude_types: zestawy typów tokenów
      - include_paths_re / exclude_paths_re: regexy ścieżek
    """
    policy = policy or {}
    include_types = set(policy.get("include_types") or [])
    exclude_types = set(policy.get("exclude_types") or [])
    inc_re = re.compile(policy.get("include_paths_re", r".*"))
    exc_re = re.compile(policy.get("exclude_paths_re", r"$^"))

    vocab = Vocabulary()
    hist: Dict[str, int] = {}

    def bump(k: str) -> None:
        hist[k] = hist.get(k, 0) + 1

    for t in tokens:
        if include_types and t.type not in include_types:
            continue
        if t.type in exclude_types:
            continue
        path = t.file or ""
        if not inc_re.search(path) or exc_re.search(path):
            continue
        for key in vocab.expand(t):
            bump(key)

    return hist


# ──────────────────────────────────────────────────────────────────────────────
# Główna funkcja „Δ → histogram” (na potrzeby CI/HOOK)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_range(diff_range: str) -> Tuple[str, str]:
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
    # fallback: HEAD-like
    return f"{dr}~1", dr

def tokenize_diff(repo_root: Path, diff_range: str, *, policy: Optional[dict] = None) -> Dict[str, int]:
    """
    Główne wejście warstwy delta: liczy histogram Δ dla zakresu gita.
    """
    base, head = _parse_range(diff_range)
    tokens = extract_from_git(Path(repo_root), base, head, policy=policy)
    return tokens_to_hist(tokens, policy=policy)
