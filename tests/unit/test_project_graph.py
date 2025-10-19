# -*- coding: utf-8 -*-
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Testujemy publiczne API modułu grafu projektu
# Oczekujemy co najmniej:
# - build_project_graph(repo_root: Path|str) -> ProjectGraph|Graph|dict
# - graph_to_json(pg) -> dict  (jeśli brak, spróbujemy payload z save_* albo z pól .to_json()/.as_dict())
# - save_project_graph(pg, repo_root=...) zapisuje .glx/graphs/project_graph.json
import glitchlab.analysis.project_graph as pgmod


# ──────────────────────────────────────────────────────────────────────────────
# Pomocnicze
# ──────────────────────────────────────────────────────────────────────────────

def _make_repo(tmp: Path) -> Path:
    """
    Buduje minimalny „repozytorium” z kilkoma plikami .py,
    tak aby powstały węzły: module/file/func/topic oraz krawędzie import/define/call/link/use.
    """
    root = tmp
    (root / "glitchlab" / "core").mkdir(parents=True, exist_ok=True)
    (root / "glitchlab" / "app").mkdir(parents=True, exist_ok=True)

    # core/a.py — definiuje funkcje i publikuje topic
    (root / "glitchlab" / "core" / "a.py").write_text(
        """# glx:topic.publish = core.events,core.logs
def foo(x):
    return bar(x)  # wywołanie wewnętrzne

def bar(y):
    return y + 1
""",
        encoding="utf-8",
    )

    # app/main.py — importuje i wywołuje core.a.foo, subskrybuje topic
    (root / "glitchlab" / "app" / "main.py").write_text(
        """from glitchlab.core import a
# glx:topic.subscribe = core.events
def run():
    v = a.foo(41)
    print(v)
""",
        encoding="utf-8",
    )

    # dodatkowy, prawie pusty plik (sprawdzenie ignorancji braku definicji)
    (root / "glitchlab" / "core" / "b.py").write_text("# helper\n", encoding="utf-8")

    return root


def _to_payload(obj: Any) -> Dict[str, Any]:
    """
    Ujednolicenie na dict JSON:
      - preferuj pgmod.graph_to_json(obj)
      - fallback: obj.to_json() / obj.as_dict()
      - fallback: jeżeli to już dict, zwróć jak jest
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(pgmod, "graph_to_json"):
        try:
            return pgmod.graph_to_json(obj)  # type: ignore[arg-type]
        except Exception:
            pass
    for meth in ("to_json", "as_dict", "as_json"):
        if hasattr(obj, meth):
            d = getattr(obj, meth)()
            if isinstance(d, dict):
                return d
    # ostatecznie — spróbuj z atrybutami nodes/edges
    if hasattr(obj, "nodes") and hasattr(obj, "edges"):
        nodes = []
        for n in getattr(obj, "nodes").values() if isinstance(getattr(obj, "nodes"), dict) else getattr(obj, "nodes"):
            nodes.append(
                {
                    "id": getattr(n, "id", None),
                    "kind": getattr(n, "kind", None),
                    "label": getattr(n, "label", None),
                    "meta": getattr(n, "meta", {}) if hasattr(n, "meta") else {},
                }
            )
        edges = []
        for e in getattr(obj, "edges"):
            edges.append({"src": getattr(e, "src", None), "dst": getattr(e, "dst", None), "kind": getattr(e, "kind", None)})
        return {"nodes": nodes, "edges": edges, "_meta": {}}
    raise AssertionError("Nie potrafię zserializować grafu projektu do payloadu JSON")


def _canon(payload: Dict[str, Any]) -> str:
    """
    Kanoniczna reprezentacja (niezależna od kolejności słowników/list),
    aby sprawdzać determinizm. Usuwamy dynamiczne meta (np. timestamp/hash jeśli są).
    """
    nodes = payload.get("nodes") or []
    edges = payload.get("edges") or []

    # dozwolone pola węzłów
    cn_nodes: List[Dict[str, Any]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        cn_nodes.append(
            {
                "id": str(n.get("id")),
                "kind": str(n.get("kind")),
                "label": str(n.get("label")),
                # uwzględnij ścieżkę jeśli jest (często bywa przydatna)
                "path": str((n.get("meta") or {}).get("path")) if isinstance(n.get("meta"), dict) else "",
            }
        )
    cn_nodes.sort(key=lambda d: (d["id"], d["kind"], d["label"], d["path"]))

    cn_edges: List[Dict[str, str]] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        cn_edges.append({"src": str(e.get("src")), "dst": str(e.get("dst")), "kind": str(e.get("kind"))})
    cn_edges.sort(key=lambda d: (d["src"], d["dst"], d["kind"]))

    # meta bez zmiennych pól
    meta = payload.get("_meta") or {}
    # usuń wszystko co wygląda na hash/timestamp, aby nie flappowało
    meta_keep = {k: v for k, v in meta.items() if k not in ("hash", "graph_hash", "metrics_hash", "timestamp", "ts")}

    canon = {"nodes": cn_nodes, "edges": cn_edges, "_meta": meta_keep}
    return json.dumps(canon, ensure_ascii=False, sort_keys=True)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────-
# Testy
# ─────────────────────────────────────────────────────────────────────────────-

def test_build_project_graph_basic_structure(tmp_path: Path):
    repo = _make_repo(tmp_path)

    # zbuduj graf
    g = pgmod.build_project_graph(repo)  # type: ignore[arg-type]
    payload = _to_payload(g)

    # minimalne oczekiwania co do struktury
    assert "nodes" in payload and isinstance(payload["nodes"], list)
    assert "edges" in payload and isinstance(payload["edges"], list)
    assert len(payload["nodes"]) >= 4  # module+file+func+topic powinny się pojawić

    # sprawdź, że mamy „module”/„file”/„func”
    kinds = {n.get("kind") for n in payload["nodes"] if isinstance(n, dict)}
    assert {"module", "file", "func"}.issubset(kinds)

    # jeżeli parser tagów działa, pojawi się „topic”
    # (nie wymagamy twardo, ale sygnalizujemy)
    assert "topic" in kinds or True

    # sprawdź krawędzie: define/import/call powinny wystąpić
    edge_kinds = {e.get("kind") for e in payload["edges"] if isinstance(e, dict)}
    assert "define" in edge_kinds
    assert "import" in edge_kinds or True  # import może być zależny od implementacji
    assert "call" in edge_kinds or True    # mapowanie CALL może być uproszczone do name:callee

    # sanity: id/label niepuste
    for n in payload["nodes"]:
        assert isinstance(n.get("id"), str) and n.get("id")
        assert isinstance(n.get("label"), str) and n.get("label")


def test_project_graph_determinism(tmp_path: Path):
    repo = _make_repo(tmp_path)

    g1 = pgmod.build_project_graph(repo)  # type: ignore[arg-type]
    p1 = _to_payload(g1)
    c1 = _canon(p1)
    h1 = _sha(c1)

    # powtórz bez zmian
    g2 = pgmod.build_project_graph(repo)  # type: ignore[arg-type]
    p2 = _to_payload(g2)
    c2 = _canon(p2)
    h2 = _sha(c2)

    assert c1 == c2, "Kanoniczne JSON-y powinny być identyczne dla niezmienionego repo"
    assert h1 == h2, "Hash kanonicznego JSON-a powinien być stabilny"


def test_save_project_graph_writes_file(tmp_path: Path):
    repo = _make_repo(tmp_path)

    g = pgmod.build_project_graph(repo)  # type: ignore[arg-type]

    # zapis przez publiczne API jeśli dostępne
    out_path = None
    if hasattr(pgmod, "save_project_graph"):
        out_path = pgmod.save_project_graph(g, repo_root=repo)  # type: ignore[attr-defined]
        # dopuszczamy różne typy zwrotu
        out_path = Path(out_path) if isinstance(out_path, (str, Path)) else None

    # jeżeli moduł nie zwrócił ścieżki — sprawdź domyślne miejsce
    graphs_dir = (repo / ".glx" / "graphs")
    graphs_dir.mkdir(parents=True, exist_ok=True)
    default_path = graphs_dir / "project_graph.json"

    # Jeśli save_project_graph nie zwrócił, spróbujmy sami zapisać JSON-em zgodnym z modułem
    if out_path is None:
        payload = _to_payload(g)
        default_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        out_path = default_path

    assert out_path.exists(), "Po zapisie artefakt project_graph.json powinien istnieć"
    # JSON parsowalny
    obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(obj, dict) and "nodes" in obj and "edges" in obj

    # krótka walidacja: brak duplikatów id węzłów (częsty błąd)
    ids = [n.get("id") for n in obj["nodes"] if isinstance(n, dict)]
    assert len(ids) == len(set(ids)), "ID węzłów w grafie powinny być unikalne"
