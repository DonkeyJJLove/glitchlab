# glitchlab/analysis/ast_index.py
# -*- coding: utf-8 -*-
"""
Deterministyczny indeks AST + parser #glx-tagów (publish/subscribe/request_reply).

Python: 3.9 (stdlib only)

Zakres:
- Indeksowanie definicji (def/async def/class), użyć (Name.Load), wywołań (Call)
  i importów (Import/ImportFrom) z kwalifikowanymi nazwami.
- Zliczanie S/H/Z (jak w warstwie mozaiki) + alpha/beta.
- Parsowanie tagów w komentarzach:  # glx:key=value
  Obsługiwane klucze (min.): topic.publish, topic.subscribe, topic.request_reply,
  tile, port.exposed, port.led, fallback.plan, schema.
- Wygodny model danych w dataclass: AstIndex, DefItem, UseItem, CallItem,
  ImportItem, GlxTag, AstNodeLite, AstSummary.
- API:
    ast_index_of_source(src, file_path="<memory>") -> AstIndex
    ast_index_of_file(path) -> Optional[AstIndex]
    ast_index_of_rev(path, rev="HEAD") -> Optional[AstIndex]
    ast_summary_of_source/src/file/rev  (kompatybilne z wcześniejszym kodem)
    ast_summary_simple_of_file(path) -> Dict[str, object]   # LEKKI słownik metryk
    export_ast_index_json(repo_root=None) -> Path           # zapis .glx/graphs/ast_index.json
    parse_glx_tags_from_source(src, file_path="<memory>") -> list[GlxTag]
    glx_tags_to_grammar_events(tags, component=None) -> list[dict]

Uwaga:
- Brak zależności zewnętrznych. Nie używamy RNG – wszystko deterministyczne.
- Kwalifikowanie wywołań (Call) działa heurystycznie: Name i łańcuch Attribute.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Lokalny (opcjonalny) import do odczytu z gita
try:
    from .git_io import show_file_at_rev  # type: ignore
except Exception:  # pragma: no cover
    def show_file_at_rev(path: str, rev: str = "HEAD") -> Optional[str]:
        return None

__all__ = [
    # modele indeksu
    "DefItem", "UseItem", "CallItem", "ImportItem", "GlxTag",
    "AstIndex",
    # główne API indeksu
    "ast_index_of_source", "ast_index_of_file", "ast_index_of_rev",
    # kompatybilna warstwa „summary”
    "AstNodeLite", "AstSummary",
    "ast_summary_of_source", "ast_summary_of_file", "ast_summary_of_rev",
    "ast_summary_simple_of_file",
    "export_ast_index_json",
    "summarize_labels",
    # #glx
    "parse_glx_tags_from_source", "glx_tags_to_grammar_events",
]

# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses (indeks)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DefItem:
    qualname: str                 # module-local qualname: "Class.method" | "function" | "Class"
    kind: str                     # "function" | "async_function" | "class" | "method" | "async_method"
    lineno: int
    col: int
    args: List[str] = field(default_factory=list)     # funkcje/metody
    returns: Optional[str] = None                     # str(ast.unparse) jeżeli dostępne
    decorators: List[str] = field(default_factory=list)
    public: bool = True


@dataclass(frozen=True)
class UseItem:
    name: str                     # identyfikator (Name) w kontekście Load
    lineno: int
    col: int
    scope: str                    # aktywna ścieżka (np. "Class.method" albo "<module>")


@dataclass(frozen=True)
class CallItem:
    func: str                     # "pkg.mod.fn" | "obj.method" | "fn"
    lineno: int
    col: int
    scope: str                    # aktywna ścieżka (np. "Class.method" albo "<module>")


@dataclass(frozen=True)
class ImportItem:
    what: str                     # "module" lub "module.name"
    asname: Optional[str]
    lineno: int
    col: int


@dataclass(frozen=True)
class GlxTag:
    key: str                      # np. "topic.publish"
    value: object                 # str | list[str] | dict (np. {"request": "...","reply":"..."}), zawsze JSON-serializable
    line: int
    col: int
    raw: str
    file_path: str


@dataclass
class AstIndex:
    file_path: str
    defs: Dict[str, DefItem]                  # map qualname -> DefItem
    calls: List[CallItem]
    uses: List[UseItem]
    imports: List[ImportItem]
    tags: List[GlxTag]
    # zwięzłe metryki S/H/Z
    S: int
    H: int
    Z: int
    alpha: float
    beta: float
    # etykiety (do szybkich statystyk)
    labels: List[str]
    per_label: Dict[str, int]


# ──────────────────────────────────────────────────────────────────────────────
# Kompatybilna warstwa „summary” (zachowana)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AstNodeLite:
    id: int
    label: str
    depth: int
    parent: Optional[int]
    lineno: Optional[int]
    col: Optional[int]
    children: List[int] = field(default_factory=list)
    meta: Tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)


@dataclass
class AstSummary:
    file_path: str
    S: int
    H: int
    Z: int
    maxZ: int
    alpha: float
    beta: float
    nodes: Dict[int, AstNodeLite]
    labels: List[str]
    per_label: Dict[str, int]


# ──────────────────────────────────────────────────────────────────────────────
# Heurystyki/metryki deterministyczne S/H/Z
# ──────────────────────────────────────────────────────────────────────────────

_CONTROL_NODES = (
    ast.If, ast.For, ast.While, ast.With, ast.Try, ast.Match
)
_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_CALL_NODES = (ast.Call,)
_DATA_NODES = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Yield, ast.YieldFrom)
_IO_NODES = (ast.Raise, ast.Assert, ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom, ast.Expr)
_LEAF_NODES = (ast.Name, ast.Attribute, ast.Constant)

_W_S = {
    ast.Module: 1,
    ast.FunctionDef: 3, ast.AsyncFunctionDef: 3, ast.ClassDef: 4,
    ast.If: 2, ast.For: 3, ast.While: 3, ast.With: 2, ast.Try: 3, ast.Match: 3,
    ast.Assign: 2, ast.AnnAssign: 2, ast.AugAssign: 2, ast.Return: 1,
    ast.Call: 1, ast.Import: 1, ast.ImportFrom: 1,
}
_W_H = {
    ast.Name: 1, ast.Attribute: 1, ast.Constant: 1,
    ast.Call: 3, ast.Assign: 2, ast.AnnAssign: 2, ast.AugAssign: 2,
    ast.Import: 2, ast.ImportFrom: 2,
}

def _clamp01(x: float) -> float:
    return 0.0 if x <= 0 else (1.0 if x >= 1.0 else float(x))

def _entropy(proportions: List[float]) -> float:
    ps = [p for p in proportions if p > 0]
    if not ps:
        return 0.0
    H = -sum(p * math.log2(p) for p in ps)
    Hmax = math.log2(len(ps)) if len(ps) > 1 else 1.0
    return _clamp01(H / Hmax)

def _label_props(labels: List[str]) -> Dict[str, float]:
    n = max(1, len(labels))
    counts: Dict[str, int] = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return {k: v / n for k, v in counts.items()}

def is_import(a: ast.AST) -> bool:
    return isinstance(a, (ast.Import, ast.ImportFrom))

def _node_meta(a: ast.AST, depth: int, siblings: int, label_props_global: Dict[str, float]) -> Tuple[float, float, float, float, float, float]:
    is_ctrl = isinstance(a, _CONTROL_NODES)
    is_def = isinstance(a, _DEF_NODES)
    is_call = isinstance(a, _CALL_NODES)
    is_leaf = isinstance(a, _LEAF_NODES)
    # L
    L = 0.75 if is_leaf else (0.35 if is_ctrl else 0.55)
    # S
    s_w = 0
    for t, w in _W_S.items():
        if isinstance(a, t):
            s_w = max(s_w, w)
    Smeta = _clamp01(s_w / 4.0)
    # Sel
    if is_ctrl:
        Sel = 0.8
    elif is_call:
        Sel = 0.7
    elif is_def:
        Sel = 0.55
    else:
        Sel = 0.5
    # Stab
    if isinstance(a, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Constant)):
        Stab = 0.8
    elif is_def:
        Stab = 0.75
    elif is_call or is_import(a):
        Stab = 0.45
    else:
        Stab = 0.6
    # Cau
    if is_call or isinstance(a, (ast.Return, ast.Raise, ast.Yield, ast.YieldFrom)):
        Cau = 0.75
    elif is_ctrl:
        Cau = 0.65
    elif is_def:
        Cau = 0.6
    else:
        Cau = 0.5
    # H (entropia globalna + głębokość + rozgałęzienie)
    lbl = a.__class__.__name__
    p_lbl = label_props_global.get(lbl, 0.0)
    Hglob = _entropy(list(label_props_global.values()))
    dep_term = _clamp01(depth / 12.0)
    sib_term = _clamp01(siblings / 6.0)
    Hmeta = _clamp01(0.6 * Hglob + 0.25 * dep_term + 0.15 * sib_term + 0.05 * (1.0 - p_lbl))
    return (_clamp01(L), _clamp01(Smeta), _clamp01(Sel), _clamp01(Stab), _clamp01(Cau), _clamp01(Hmeta))


# ──────────────────────────────────────────────────────────────────────────────
# Parser #glx-tagów
# ──────────────────────────────────────────────────────────────────────────────

# Wzorzec: „… # glx:key=value [# komentarz]” — działa w dowolnym miejscu linii
_GLX_RE = re.compile(r"#\s*glx:([a-zA-Z0-9_.-]+)\s*=\s*(.+?)(?=\s*(?:#|$))")

def _parse_tag_value(raw: str) -> object:
    """Próba JSON → CSV → raw string; specjalna obsługa A->B dla request_reply."""
    s = raw.strip()
    # JSON?
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            pass
    # A->B (request_reply)
    if "->" in s and "," not in s and ":" not in s:
        a, b = s.split("->", 1)
        return {"request": a.strip(), "reply": b.strip()}
    # CSV (bez spacji)
    if "," in s:
        return [t.strip() for t in s.split(",") if t.strip()]
    return s  # proste value (np. nazwa portu)

def parse_glx_tags_from_source(src: str, file_path: str = "<memory>") -> List[GlxTag]:
    tags: List[GlxTag] = []
    for i, line in enumerate(src.splitlines(), start=1):
        for m in _GLX_RE.finditer(line):
            key = m.group(1).strip()
            val_raw = m.group(2).strip()
            val = _parse_tag_value(val_raw)
            tags.append(GlxTag(key=key, value=val, line=i, col=m.start(), raw=line.rstrip("\n"), file_path=file_path))
    return tags

def glx_tags_to_grammar_events(tags: Iterable[GlxTag], component: Optional[str] = None) -> List[Dict[str, object]]:
    """
    Minimalna projekcja tagów do „zdarzeń gramatyki” (lekki adapter pod EGDB).
    - topic.publish:   → {'kind':'topic_publish','topic':t, 'component':component}
    - topic.subscribe: → {'kind':'topic_subscribe','topic':t, 'component':component}
    - topic.request_reply: „A->B” lub {'request':A,'reply':B}
    Inne tagi przepuszczamy jako 'meta' do ewentualnej dalszej projekcji.
    """
    ev: List[Dict[str, object]] = []
    for t in tags:
        k = t.key.lower()
        v = t.value
        if k == "topic.publish":
            items = v if isinstance(v, list) else [v]
            for it in items:
                ev.append({"kind": "topic_publish", "topic": str(it), "component": component, "line": t.line})
        elif k == "topic.subscribe":
            items = v if isinstance(v, list) else [v]
            for it in items:
                ev.append({"kind": "topic_subscribe", "topic": str(it), "component": component, "line": t.line})
        elif k == "topic.request_reply":
            if isinstance(v, dict) and "request" in v and "reply" in v:
                ev.append({"kind": "topic_request_reply", "request": str(v["request"]), "reply": str(v["reply"]), "component": component, "line": t.line})
            elif isinstance(v, str) and "->" in v:
                a, b = v.split("->", 1)
                ev.append({"kind": "topic_request_reply", "request": a.strip(), "reply": b.strip(), "component": component, "line": t.line})
        else:
            # meta (np. tile, port.exposed, port.led, fallback.plan, schema)
            ev.append({"kind": "meta", "key": t.key, "value": t.value, "component": component, "line": t.line})
    return ev


# ──────────────────────────────────────────────────────────────────────────────
# Indeksowanie AST (defs/uses/calls/imports) + S/H/Z
# ──────────────────────────────────────────────────────────────────────────────

def _unparse(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        # Python 3.9+: ast.unparse dostępne
        return ast.unparse(node)  # type: ignore[attr-defined]
    except Exception:
        return None

def _attr_to_dotted(n: ast.AST) -> Optional[str]:
    """
    Buduje „dotted” z Name/Attribute/Call.func.
    Przykłady: Name('foo') -> "foo"; Attribute(Name('a'), 'b') -> "a.b"
    """
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        base = _attr_to_dotted(n.value)
        if base:
            return f"{base}.{n.attr}"
        return n.attr
    # np. Subscript / Call (zagnieżdżone) – spróbuj z dzieckiem .func
    if isinstance(n, ast.Call):
        return _attr_to_dotted(n.func)
    return None

class _IdxVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        # stos scope do budowy qualname
        self.scope: List[str] = []
        self.defs: Dict[str, DefItem] = {}
        self.uses: List[UseItem] = []
        self.calls: List[CallItem] = []
        self.imports: List[ImportItem] = []
        self.labels: List[str] = []
        # S/H/Z
        self.S = 0
        self.H = 0
        self.maxZ = 0

    # utils
    def _cur_scope(self) -> str:
        return ".".join(self.scope) if self.scope else "<module>"

    def _push(self, name: str) -> None:
        self.scope.append(name)

    def _pop(self) -> None:
        if self.scope:
            self.scope.pop()

    # S/H akumulacja
    def _acc_SH(self, node: ast.AST) -> None:
        for t, w in _W_S.items():
            if isinstance(node, t):
                self.S += w
                break
        for t, w in _W_H.items():
            if isinstance(node, t):
                self.H += w
                break

    # główne wizyty
    def generic_visit(self, node: ast.AST) -> None:
        self.labels.append(node.__class__.__name__)
        self._acc_SH(node)
        if isinstance(node, _CONTROL_NODES + _DEF_NODES):
            self.maxZ = max(self.maxZ, len(self.scope))
        super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = "method" if self.scope and self.scope[-1] and self.scope[-1][0].isupper() else "function"
        q = ".".join([*self.scope, node.name]) if self.scope else node.name
        d = DefItem(
            qualname=q,
            kind="method" if kind == "method" else "function",
            lineno=node.lineno, col=node.col_offset,
            args=[a.arg for a in node.args.args],
            returns=_unparse(node.returns),
            decorators=[_unparse(d) or "" for d in node.decorator_list],
            public=not node.name.startswith("_"),
        )
        self.defs[q] = d
        self._push(node.name)
        self.generic_visit(node)
        self._pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        kind = "async_method" if self.scope and self.scope[-1] and self.scope[-1][0].isupper() else "async_function"
        q = ".".join([*self.scope, node.name]) if self.scope else node.name
        d = DefItem(
            qualname=q,
            kind=kind,
            lineno=node.lineno, col=node.col_offset,
            args=[a.arg for a in node.args.args],
            returns=_unparse(node.returns),
            decorators=[_unparse(d) or "" for d in node.decorator_list],
            public=not node.name.startswith("_"),
        )
        self.defs[q] = d
        self._push(node.name)
        self.generic_visit(node)
        self._pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        q = ".".join([*self.scope, node.name]) if self.scope else node.name
        d = DefItem(
            qualname=q,
            kind="class",
            lineno=node.lineno, col=node.col_offset,
            args=[],
            returns=None,
            decorators=[_unparse(d) or "" for d in node.decorator_list],
            public=not node.name.startswith("_"),
        )
        self.defs[q] = d
        self._push(node.name)
        self.generic_visit(node)
        self._pop()

    def visit_Name(self, node: ast.Name) -> None:
        # Interesuje nas Load jako „use”
        if isinstance(node.ctx, ast.Load):
            self.uses.append(UseItem(name=node.id, lineno=node.lineno, col=node.col_offset, scope=self._cur_scope()))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = _attr_to_dotted(node.func) or (_unparse(node.func) or "")
        self.calls.append(CallItem(func=f, lineno=node.lineno, col=node.col_offset, scope=self._cur_scope()))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(ImportItem(what=alias.name, asname=alias.asname, lineno=node.lineno, col=node.col_offset))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            what = f"{mod}.{alias.name}" if mod else alias.name
            self.imports.append(ImportItem(what=what, asname=alias.asname, lineno=node.lineno, col=node.col_offset))
        self.generic_visit(node)


# ──────────────────────────────────────────────────────────────────────────────
# Główne API indeksu
# ──────────────────────────────────────────────────────────────────────────────

def ast_index_of_source(src: str, file_path: str = "<memory>") -> AstIndex:
    # 1) parsowanie AST
    tree = ast.parse(src, filename=file_path)

    # 2) przejście indeksujące
    v = _IdxVisitor()
    v.visit(tree)

    # 3) S/H/Z → alpha/beta
    tot = max(1, v.S + v.H)
    alpha = v.S / tot
    beta = v.H / tot
    Z = max(1, v.maxZ)

    # 4) #glx tagi
    tags = parse_glx_tags_from_source(src, file_path=file_path)

    return AstIndex(
        file_path=file_path,
        defs=v.defs,
        calls=v.calls,
        uses=v.uses,
        imports=v.imports,
        tags=tags,
        S=int(v.S),
        H=int(v.H),
        Z=int(Z),
        alpha=float(alpha),
        beta=float(beta),
        labels=v.labels,
        per_label=summarize_labels(v.labels),
    )

def ast_index_of_file(path: str | Path) -> Optional[AstIndex]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        src = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        src = p.read_text(encoding="latin-1", errors="replace")
    return ast_index_of_source(src, str(p))

def ast_index_of_rev(path: str, rev: str = "HEAD") -> Optional[AstIndex]:
    src = show_file_at_rev(path, rev)
    if src is None:
        return None
    return ast_index_of_source(src, f"{rev}:{path}")


# ──────────────────────────────────────────────────────────────────────────────
# Warstwa „summary” (kompatybilność + lekki słownik metryk)
# ──────────────────────────────────────────────────────────────────────────────

def ast_summary_of_source(src: str, file_path: str = "<memory>") -> AstSummary:
    idx = ast_index_of_source(src, file_path=file_path)
    nodes: Dict[int, AstNodeLite] = {}
    nodes[0] = AstNodeLite(id=0, label="Module", depth=0, parent=None, lineno=1, col=0, children=[], meta=(0.6, 0.5, 0.5, 0.6, 0.5, 0.5))
    return AstSummary(
        file_path=file_path,
        S=idx.S,
        H=idx.H,
        Z=idx.Z,
        maxZ=idx.Z,
        alpha=idx.alpha,
        beta=idx.beta,
        nodes=nodes,
        labels=idx.labels,
        per_label=idx.per_label,
    )

def ast_summary_of_file(path: str | Path) -> Optional[AstSummary]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        src = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        src = p.read_text(encoding="latin-1", errors="replace")
    return ast_summary_of_source(src, str(p))

def ast_summary_of_rev(path: str, rev: str = "HEAD") -> Optional[AstSummary]:
    src = show_file_at_rev(path, rev)
    if src is None:
        return None
    return ast_summary_of_source(src, f"{rev}:{path}")

def summarize_labels(labels: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for l in labels:
        out[l] = out.get(l, 0) + 1
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Dodatkowe metryki jakościowe na plik (funkcje/klasy, średnie długości, tagi)
# ──────────────────────────────────────────────────────────────────────────────

_DEF_KIND_FUN = {"function", "async_function", "method", "async_method"}
_DEF_KIND_CLASS = {"class"}

def _estimate_block_lengths(lines: List[str], def_lines: List[int]) -> List[int]:
    """
    Heurystycznie szacuje długości bloków (funkcje/klasy) w liniach:
    - sortuje wszystkie linie definicji (def/class),
    - długość = (linia następnej definicji) - (linia bieżącej), ostatnia do końca pliku.
    """
    if not def_lines:
        return []
    total = len(lines)
    xs = sorted(int(x) for x in def_lines if x > 0 and x <= total)
    lens: List[int] = []
    for i, ln in enumerate(xs):
        next_ln = xs[i + 1] if i + 1 < len(xs) else total + 1
        L = max(1, int(next_ln) - int(ln))
        lens.append(L)
    return lens

def _read_lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return path.read_text(encoding="latin-1", errors="replace").splitlines()

def _per_file_quality(idx: AstIndex) -> Dict[str, object]:
    """
    Zwraca:
      - num_funcs, num_classes
      - avg_func_len_lines, avg_class_len_lines (heurystycznie)
      - tag counters: publish_count, subscribe_count, rr_count
    """
    p = Path(idx.file_path)
    lines = _read_lines(p)

    # zbiory linii dla funkcji i klas
    func_lines = [d.lineno for d in idx.defs.values() if d.kind in _DEF_KIND_FUN]
    class_lines = [d.lineno for d in idx.defs.values() if d.kind in _DEF_KIND_CLASS]
    all_def_lines = sorted(func_lines + class_lines)

    # oszacowanie długości na bazie wszystkich definicji
    all_lens = _estimate_block_lengths(lines, all_def_lines)
    # rozdzielnie: wyciągnij długości odpowiadające funkcjom/klasom zachowując kolejność
    # (przyjmujemy mapowanie po indeksie w posortowanej liście linii)
    idx_map = {ln: k for k, ln in enumerate(sorted(all_def_lines))}
    fun_lens = []
    cls_lens = []
    for ln in func_lines:
        k = idx_map.get(ln)
        if k is not None and k < len(all_lens):
            fun_lens.append(all_lens[k])
    for ln in class_lines:
        k = idx_map.get(ln)
        if k is not None and k < len(all_lens):
            cls_lens.append(all_lens[k])

    def _avg(xs: List[int]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    # liczniki tagów
    pub = sub = rr = 0
    for t in idx.tags:
        k = t.key.lower()
        if k == "topic.publish":
            pub += len(t.value) if isinstance(t.value, list) else 1
        elif k == "topic.subscribe":
            sub += len(t.value) if isinstance(t.value, list) else 1
        elif k == "topic.request_reply":
            rr += 1

    return {
        "num_funcs": int(len(func_lines)),
        "num_classes": int(len(class_lines)),
        "avg_func_len_lines": round(_avg(fun_lens), 2),
        "avg_class_len_lines": round(_avg(cls_lens), 2),
        "publish_count": int(pub),
        "subscribe_count": int(sub),
        "request_reply_count": int(rr),
    }

def ast_summary_simple_of_file(path: str | Path) -> Optional[Dict[str, object]]:
    """
    Lekka projekcja metryk na słownik {S,H,Z,alpha,beta,num_funcs,num_classes,avg_*_len_lines,publish/subscribe/rr}.
    Nie modyfikuje istniejącego API (ast_summary_of_file nadal zwraca dataclass).
    """
    idx = ast_index_of_file(path)
    if idx is None:
        return None
    q = _per_file_quality(idx)
    return {
        "S": int(idx.S),
        "H": int(idx.H),
        "Z": int(idx.Z),
        "alpha": float(idx.alpha),
        "beta": float(idx.beta),
        **q,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Eksport zbiorczy: .glx/graphs/ast_index.json
# ──────────────────────────────────────────────────────────────────────────────

def _iter_py_files(root: Path) -> List[Path]:
    """
    Zwraca listę plików *.py z pominięciem .git/, .glx/, __pycache__/, venv*/, .venv*/
    """
    out: List[Path] = []
    skip_parts = {".git", ".glx", "__pycache__", "venv", ".venv"}
    for p in root.rglob("*.py"):
        parts = set(p.parts)
        if parts & skip_parts:
            continue
        # pomiń ścieżki typu site-packages w repo (jeśli istnieją)
        if "site-packages" in parts:
            continue
        out.append(p)
    return out

def _get_glx_dir(base: Optional[Path] = None) -> Path:
    """
    Ustal katalog artefaktów .glx (preferuj glitchlab.io.artifacts, fallback na <base>/.glx).
    """
    base = Path.cwd() if base is None else Path(base)
    # spróbuj załadować io.artifacts
    try:
        import glitchlab.io.artifacts as art  # type: ignore
        for name in ("ensure_glx_dir", "get_glx_dir", "ensure_artifacts_dir", "artifacts_dir"):
            if hasattr(art, name):
                try:
                    p = getattr(art, name)(base)  # type: ignore
                except TypeError:
                    p = getattr(art, name)()      # type: ignore
                p = Path(p)
                p.mkdir(parents=True, exist_ok=True)
                return p.resolve()
    except Exception:
        pass
    p = (base / ".glx").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_p = Path(tmp.name)
    tmp_p.replace(path)

def export_ast_index_json(repo_root: Optional[Path] = None) -> Path:
    """
    Buduje zbiorczy indeks AST (plik→metryki) i zapisuje do .glx/graphs/ast_index.json.
    Struktura:
    {
      "version": "ast_index.v1",
      "created_at": "...Z",
      "files": {
        "<abs-posix-path>": {S,H,Z,alpha,beta,num_funcs,num_classes,avg_func_len_lines,avg_class_len_lines,
                             publish_count,subscribe_count,request_reply_count},
        ...
      }
    }
    """
    root = Path(repo_root) if repo_root else Path.cwd()
    files = _iter_py_files(root)

    result: Dict[str, Dict[str, object]] = {}
    for p in files:
        rec = ast_summary_simple_of_file(p)
        if rec is not None:
            result[p.resolve().as_posix()] = rec

    payload = {
        "version": "ast_index.v1",
        "created_at": (Path and __import__("datetime").datetime.utcnow().isoformat() + "Z"),
        "files": result,
    }
    glx = _get_glx_dir(root)
    out = glx / "graphs" / "ast_index.json"
    _atomic_write_json(out, payload)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI – szybka inspekcja / eksport
# ──────────────────────────────────────────────────────────────────────────────

def _cli(argv: Optional[List[str]] = None) -> None:
    import argparse
    from datetime import datetime, timezone

    p = argparse.ArgumentParser(prog="analysis.ast_index", description="Indeks AST (defs/uses/calls/imports) + #glx-tags + S/H/Z + eksport zbiorczy")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("file", help="indeks z lokalnego pliku")
    q.add_argument("path")

    r = sub.add_parser("rev", help="indeks z gita (rev:path)")
    r.add_argument("rev")
    r.add_argument("path")

    j = sub.add_parser("tags", help="wypisz #glx tagi z pliku")
    j.add_argument("path")

    e = sub.add_parser("export", help="zapisz zbiorczy ast_index.json do .glx/graphs/")
    e.add_argument("--repo-root", default=None, help="katalog repo (domyślnie CWD)")

    args = p.parse_args(argv)

    if args.cmd == "file":
        idx = ast_index_of_file(args.path)
        if idx is None:
            print(json.dumps({"ok": False, "error": "file not found"}, indent=2))
            return
        # szybkie podsumowanie + jakościowe
        qrec = _per_file_quality(idx)
        payload = dict(
            file=idx.file_path,
            S=idx.S, H=idx.H, Z=idx.Z, alpha=round(idx.alpha, 4), beta=round(idx.beta, 4),
            defs=len(idx.defs), calls=len(idx.calls), uses=len(idx.uses), imports=len(idx.imports),
            tags=len(idx.tags),
            **qrec,
            top_labels=sorted(idx.per_label.items(), key=lambda kv: (-kv[1], kv[0]))[:8],
            sample_defs=sorted(list(idx.defs.keys()))[:6],
            sample_calls=[c.func for c in idx.calls[:6]],
            sample_tags=[{"key": t.key, "value": t.value} for t in idx.tags[:6]],
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.cmd == "rev":
        idx = ast_index_of_rev(args.path, args.rev)
        if idx is None:
            print(json.dumps({"ok": False, "error": "file not found at rev"}, indent=2))
            return
        payload = dict(file=idx.file_path, S=idx.S, H=idx.H, Z=idx.Z, alpha=idx.alpha, beta=idx.beta)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.cmd == "tags":
        pth = Path(args.path)
        src = pth.read_text(encoding="utf-8", errors="ignore")
        tags = parse_glx_tags_from_source(src, str(pth))
        print(json.dumps([asdict(t) for t in tags], ensure_ascii=False, indent=2))
        return

    if args.cmd == "export":
        root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
        out = export_ast_index_json(root)
        print(json.dumps({"ok": True, "path": out.as_posix()}, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    _cli()
