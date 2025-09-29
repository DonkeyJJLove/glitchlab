# glitchlab/analysis/impact.py
# Impact-Zone (IZ): lekki indeks wywołań (callgraph), różnice funkcji
# między base i head, oraz wyznaczanie strefy wpływu (callers/callees).
# Python 3.9+

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Iterable
import ast
import hashlib
import re

# Lokalne zależności
try:
    from .ast_delta import AstDelta
except Exception:  # pragma: no cover
    class AstDelta:  # type: ignore
        pass

__all__ = [
    "FunctionDefInfo",
    "CallSite",
    "CallGraph",
    "callgraph_index",
    "impact_zone",
    "index_to_json",
    "impact_to_json",
    "_cli",
]

# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FunctionDefInfo:
    qualname: str              # np. "ClassA.method" albo "module.func"
    name: str                  # surowa nazwa (bez klasy)
    class_name: Optional[str]  # jeśli metoda: nazwa klasy
    lineno: int
    end_lineno: int
    is_method: bool
    args: List[str]

    @property
    def span(self) -> Tuple[int, int]:
        return (self.lineno, self.end_lineno)


@dataclass(frozen=True)
class CallSite:
    caller: str        # qualname funkcji, w której znaleziono call (lub "<module>")
    callee_sym: str    # symbol wywołany (z atrybutem, np. "np.array" lub "foo")
    lineno: int
    col: int


@dataclass
class CallGraph:
    file_path: str
    defs: Dict[str, FunctionDefInfo]          # qualname -> info
    edges: Dict[str, Set[str]]                # caller qualname -> {callee symbol (string)}
    callsites: List[CallSite]                 # szczegóły
    imports: Dict[str, str]                   # alias -> module (best-effort)
    attr_targets: Dict[str, str]              # lokalne aliasy atrybutów (x = mod.fn → "x"->"mod.fn")

# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────

def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _get_end_lineno(node: ast.AST) -> int:
    # Py3.8+ ma end_lineno, ale fallback mile widziany
    end_ln = getattr(node, "end_lineno", None)
    if isinstance(end_ln, int):
        return end_ln
    # fallback: szacujemy po ostatnim dziecku
    last = None
    for last in ast.walk(node):
        pass
    return getattr(last, "lineno", getattr(node, "lineno", 0))


def _extract_name_from_call(func: ast.AST) -> Optional[str]:
    """
    Zamienia node.func (ast.Name/ast.Attribute/Call-lambda) na kropkowany string.
    Przykłady:
      Name("print") -> "print"
      Attribute(Name("np"), "array") -> "np.array"
      Attribute(Attribute(Name("pkg"), "mod"), "fn") -> "pkg.mod.fn"
    Inne przypadki (np. lambdy, subscripty) dają None (best-effort).
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: List[str] = []
        cur = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return None
    return None


def _enclosing_qualname(stack: List[str]) -> str:
    return stack[-1] if stack else "<module>"


def _span_snippet(src: str, start: int, end: int) -> str:
    lines = src.splitlines()
    start = max(1, start)
    end = min(len(lines), max(start, end))
    return "\n".join(lines[start-1:end])


def _args_of(node: ast.AST) -> List[str]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    return [a.arg for a in node.args.args] if getattr(node, "args", None) else []

# ──────────────────────────────────────────────────────────────────────────────
# Indeksowanie jednego pliku
# ──────────────────────────────────────────────────────────────────────────────

class _IndexVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, src: str):
        self.file_path = file_path
        self.src = src
        self.stack: List[str] = []   # stos qualnames (funkcja/klasa)
        self.class_stack: List[str] = []
        self.defs: Dict[str, FunctionDefInfo] = {}
        self.edges: Dict[str, Set[str]] = {}
        self.callsites: List[CallSite] = []
        self.imports: Dict[str, str] = {}      # alias -> module
        self.attr_targets: Dict[str, str] = {} # simple aliasing: x = mod.fn

    # --- Importy (best-effort) ---
    def visit_Import(self, node: ast.Import):
        for n in node.names:
            alias = n.asname or n.name
            self.imports[alias] = n.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        for n in node.names:
            alias = n.asname or n.name
            self.imports[alias] = f"{mod}.{n.name}" if mod else n.name
        self.generic_visit(node)

    # --- Proste aliasowanie atrybutów (x = mod.fn) ---
    def visit_Assign(self, node: ast.Assign):
        try:
            if isinstance(node.value, ast.Attribute):
                rhs = _extract_name_from_call(node.value)
                if rhs:
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            self.attr_targets[t.id] = rhs
        finally:
            self.generic_visit(node)

    # --- Klasy i funkcje ---
    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name)
        self.stack.append(node.name)   # włączamy do ścieżki qualname
        self.generic_visit(node)
        self.stack.pop()
        self.class_stack.pop()

    def _register_def(self, node: ast.AST, name: str, is_method: bool):
        qual = ".".join(self.stack + [name]) if self.stack else name
        info = FunctionDefInfo(
            qualname=qual,
            name=name,
            class_name=self.class_stack[-1] if (is_method and self.class_stack) else None,
            lineno=getattr(node, "lineno", 1),
            end_lineno=_get_end_lineno(node),
            is_method=is_method,
            args=_args_of(node),
        )
        self.defs[qual] = info
        self.edges.setdefault(qual, set())

    def visit_FunctionDef(self, node: ast.FunctionDef):
        is_method = len(self.class_stack) > 0
        self._register_def(node, node.name, is_method)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        is_method = len(self.class_stack) > 0
        self._register_def(node, node.name, is_method)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    # --- Wywołania ---
    def visit_Call(self, node: ast.Call):
        caller = _enclosing_qualname(self.stack)
        callee = _extract_name_from_call(node.func)
        if callee:
            # rozwiąż proste aliasy (x → mod.fn)
            callee = self.attr_targets.get(callee, callee)
            self.edges.setdefault(caller, set()).add(callee)
            self.callsites.append(CallSite(
                caller=caller, callee_sym=callee,
                lineno=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0)
            ))
        self.generic_visit(node)


def callgraph_index(src: str, file_path: str = "<memory>") -> CallGraph:
    """
    Buduje indeks funkcji/wywołań w pojedynczym pliku.
    - definicje (FunctionDef/AsyncFunctionDef, także metody klas),
    - krawędzie caller->callee (string callee, kropkowany jeśli Attribute),
    - callsites z pozycjami,
    - importy i proste aliasy atrybutów (x = mod.fn).
    """
    tree = ast.parse(src)
    v = _IndexVisitor(file_path=file_path, src=src)
    v.visit(tree)
    return CallGraph(
        file_path=file_path,
        defs=v.defs,
        edges=v.edges,
        callsites=v.callsites,
        imports=v.imports,
        attr_targets=v.attr_targets,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Diff funkcji base vs head (added/removed/modified)
# ──────────────────────────────────────────────────────────────────────────────

def _function_spans(index: CallGraph) -> Dict[str, Tuple[int, int]]:
    return {q: d.span for q, d in index.defs.items()}


def _span_hashes(src: str, spans: Dict[str, Tuple[int, int]]) -> Dict[str, str]:
    return {q: _hash_text(_span_snippet(src, a, b)) for q, (a, b) in spans.items()}


def _classless(qname: str) -> str:
    # Ułatwia dopasowanie metod po refaktorze klasy → metoda (ClassA.m → ClassB.m)
    return qname.split(".")[-1]


def _match_functions(base_idx: CallGraph, head_idx: CallGraph,
                     base_src: str, head_src: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Zwraca: (added, removed, modified) w sensie qualname (best-effort).
    - porównuje z użyciem skrótów fragmentów źródła (span hash),
    - jeśli qualname nie pasuje, próbuje dopasować po samej nazwie (classless).
    """
    b_spans = _function_spans(base_idx)
    h_spans = _function_spans(head_idx)
    b_hash = _span_hashes(base_src, b_spans)
    h_hash = _span_hashes(head_src, h_spans)

    b_names = set(b_spans.keys())
    h_names = set(h_spans.keys())

    removed: Set[str] = set()
    added: Set[str] = set()
    modified: Set[str] = set()

    # Bezpośrednie dopasowania po qualname
    common = b_names & h_names
    for q in common:
        if b_hash.get(q) != h_hash.get(q):
            modified.add(q)

    # Pozostałe – dodane/removed po qualname
    removed |= (b_names - h_names)
    added   |= (h_names - b_names)

    # Spróbuj zredukować false-positives: dopasuj po nazwie bez klasy
    if removed and added:
        removed_by_name = {}
        for q in list(removed):
            removed_by_name.setdefault(_classless(q), set()).add(q)
        for q in list(added):
            cn = _classless(q)
            match = removed_by_name.get(cn)
            if match:
                # przeniesiona metoda (inna klasa) – traktuj jako "modified"
                modified.add(q)
                removed.discard(next(iter(match)))
                added.discard(q)

    return added, removed, modified

# ──────────────────────────────────────────────────────────────────────────────
# Impact-Zone (IZ)
# ──────────────────────────────────────────────────────────────────────────────

def _reverse_edges(edges: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    rev: Dict[str, Set[str]] = {}
    for caller, callees in edges.items():
        for cal in callees:
            rev.setdefault(cal, set()).add(caller)
    return rev


def _resolve_local_callees(head_idx: CallGraph) -> Set[str]:
    """
    Zbiera symbole callee, które odpowiadają lokalnym definicjom (best-effort).
    Dopasowanie: symbol == qualname lub symbol == sama nazwa funkcji definiowanej
    (dla metod również rozważ nazwę bez klasy).
    """
    locals_full = set(head_idx.defs.keys())
    locals_short = {_classless(q) for q in locals_full}
    out: Set[str] = set()
    for _, callees in head_idx.edges.items():
        for sym in callees:
            if sym in locals_full or sym in locals_short:
                out.add(sym)
    return out


def impact_zone(
    delta: AstDelta,
    base_src: str,
    head_src: str,
    file_path: str = "<memory>",
    include_transitive_callers: bool = True,
    include_transitive_callees: bool = False,
) -> Dict:
    """
    Wyznacza strefę wpływu (IZ) w pojedynczym pliku:
      - changed.functions: added/removed/modified (qualname),
      - callgraph: edges i callsites w HEAD,
      - impacted.callers_of_changed: kto woła zmienione,
      - impacted.callees_from_changed: kogo wołają zmienione,
      - spans: (base/head) mapy qualname -> [lineno, end_lineno],
      - spans_hash: sha1 snippetów (łatwe porównanie w raportach).
    """
    base_idx = callgraph_index(base_src, file_path=file_path)
    head_idx = callgraph_index(head_src, file_path=file_path)

    added, removed, modified = _match_functions(base_idx, head_idx, base_src, head_src)

    # Reverse edges po HEAD – callers zmienionych
    rev = _reverse_edges(head_idx.edges)

    # Zmienione funkcje – klucze, po których łączymy (qualname i nazwa)
    changed_keys: Set[str] = set(added | removed | modified)
    changed_keys |= {_classless(q) for q in (added | removed | modified)}

    # Kto woła zmienione? (na podstawie symboli)
    callers: Set[str] = set()
    for sym, who in rev.items():
        if sym in changed_keys:
            callers |= who

    # Kogo wołają zmienione?
    callees_from_changed: Set[str] = set()
    for caller, callees in head_idx.edges.items():
        if caller in changed_keys or _classless(caller) in changed_keys:
            callees_from_changed |= set(callees)

    # Transitive zamknięcia (opcjonalnie)
    if include_transitive_callers and callers:
        frontier = set(callers)
        while frontier:
            nxt: Set[str] = set()
            for sym in list(frontier):
                more = rev.get(sym, set())
                for m in more:
                    if m not in callers:
                        callers.add(m)
                        nxt.add(m)
            frontier = nxt

    if include_transitive_callees and callees_from_changed:
        g = head_idx.edges
        frontier = set(callees_from_changed)
        while frontier:
            nxt: Set[str] = set()
            for cur in list(frontier):
                more = g.get(cur, set())
                for m in more:
                    if m not in callees_from_changed:
                        callees_from_changed.add(m)
                        nxt.add(m)
            frontier = nxt

    # Spany i skróty (do raportów)
    base_spans = _function_spans(base_idx)
    head_spans = _function_spans(head_idx)
    base_hash = _span_hashes(base_src, base_spans)
    head_hash = _span_hashes(head_src, head_spans)

    # Krawędzie w postaci list dla JSON
    edges_list = [
        [caller, callee] for caller, cals in head_idx.edges.items() for callee in cals
    ]
    callsites_list = [
        asdict(cs) for cs in head_idx.callsites
    ]

    return {
        "file": file_path,
        "delta": {
            "dS": getattr(delta, "dS", None),
            "dH": getattr(delta, "dH", None),
            "dZ": getattr(delta, "dZ", None),
        },
        "changed": {
            "added": sorted(added),
            "removed": sorted(removed),
            "modified": sorted(modified),
        },
        "callgraph": {
            "edges": edges_list,
            "callsites": callsites_list,
            "imports": head_idx.imports,
        },
        "impacted": {
            "callers_of_changed": sorted(callers),
            "callees_from_changed": sorted(callees_from_changed),
        },
        "spans": {
            "base": {k: list(v) for k, v in base_spans.items()},
            "head": {k: list(v) for k, v in head_spans.items()},
        },
        "spans_hash": {
            "base": base_hash,
            "head": head_hash,
        },
    }

# ──────────────────────────────────────────────────────────────────────────────
# Serializacja pomocnicza
# ──────────────────────────────────────────────────────────────────────────────

def index_to_json(idx: CallGraph) -> Dict:
    return {
        "file": idx.file_path,
        "defs": {q: asdict(info) for q, info in idx.defs.items()},
        "edges": {caller: sorted(list(cals)) for caller, cals in idx.edges.items()},
        "callsites": [asdict(cs) for cs in idx.callsites],
        "imports": idx.imports,
        "attr_targets": idx.attr_targets,
    }


def impact_to_json(impact: Dict) -> Dict:
    # już JSON-owalne – zostawiamy by zachować spójność interfejsu
    return impact

# ──────────────────────────────────────────────────────────────────────────────
# Minimalny CLI (smoke)
# ──────────────────────────────────────────────────────────────────────────────

def _cli(argv: Optional[List[str]] = None) -> None:
    """
    Użycie:
      python -m glitchlab.analysis.impact index FILE.py
      python -m glitchlab.analysis.impact impact BASE.py HEAD.py
    """
    import argparse, json, pathlib
    p = argparse.ArgumentParser(prog="impact", description="Callgraph i Impact-Zone jednego pliku")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("index", help="pokaż callgraph jednego pliku")
    q.add_argument("path")

    r = sub.add_parser("impact", help="IZ z dwóch plików lokalnych (base vs head)")
    r.add_argument("base_path")
    r.add_argument("head_path")

    args = p.parse_args(argv)

    if args.cmd == "index":
        path = pathlib.Path(args.path)
        src = path.read_text(encoding="utf-8", errors="ignore")
        idx = callgraph_index(src, file_path=str(path))
        print(json.dumps(index_to_json(idx), ensure_ascii=False, indent=2))
        return

    if args.cmd == "impact":
        bp = pathlib.Path(args.base_path)
        hp = pathlib.Path(args.head_path)
        bsrc = bp.read_text(encoding="utf-8", errors="ignore")
        hsrc = hp.read_text(encoding="utf-8", errors="ignore")
        fake_delta = AstDelta  # tylko placeholder przy wywołaniu przez CLI lokalne
        iz = impact_zone(delta=fake_delta, base_src=bsrc, head_src=hsrc, file_path=str(hp))
        print(json.dumps(impact_to_json(iz), ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    _cli()
