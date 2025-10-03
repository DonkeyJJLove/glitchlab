# glitchlab/analysis/ast_delta.py
# ΔAST: porównanie AstSummary base vs head (ΔS/ΔH/ΔZ, per-label add/del/eq)
# Python 3.9+


from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

# Lokalne zależności
try:
    from .ast_index import AstSummary, ast_summary_of_source, ast_summary_of_rev, ast_summary_of_file
except Exception as _e:
    AstSummary = object  # type: ignore


    def ast_summary_of_source(*a, **k):  # type: ignore
        raise RuntimeError("ast_index not available")  # pragma: no cover


    def ast_summary_of_rev(*a, **k):  # type: ignore
        return None  # pragma: no cover


    def ast_summary_of_file(*a, **k):  # type: ignore
        return None  # pragma: no cover

__all__ = [
    "LabelDelta",
    "AstDelta",
    "diff_label_counts",
    "ast_delta",
    "delta_from_sources",
    "delta_of_file_between_revs",
    "delta_of_local_file",
    "merge_deltas",
    "to_jsonable",
]


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LabelDelta:
    add: int = 0  # ile etykiet doszło w head vs base (nadwyżka head)
    delete: int = 0  # ile etykiet ubyło (nadwyżka base)
    same: int = 0  # ile „wspólnych” (min(base, head))

    @property
    def total(self) -> int:
        return self.add + self.delete + self.same


@dataclass
class AstDelta:
    # id pliku/ident (opcjonalnie)
    file_path: str
    base_ref: str
    head_ref: str

    # skalarne różnice
    dS: int
    dH: int
    dZ: int
    dalpha: float
    dbeta: float

    # surowe wartości
    S_base: int
    H_base: int
    Z_base: int
    alpha_base: float
    beta_base: float

    S_head: int
    H_head: int
    Z_head: int
    alpha_head: float
    beta_head: float

    # per-label
    labels: Dict[str, LabelDelta]

    # meta: liczniki/skrót
    n_labels_base: int
    n_labels_head: int
    changed_labels: int  # liczba etykiet z add>0 lub delete>0


# ──────────────────────────────────────────────────────────────────────────────
# Główna logika Δ
# ──────────────────────────────────────────────────────────────────────────────

def diff_label_counts(base_counts: Dict[str, int], head_counts: Dict[str, int]) -> Dict[str, LabelDelta]:
    """
    Zwraca słownik etykieta -> LabelDelta(add, delete, same)
    add    = max(0, head - base)
    delete = max(0, base - head)
    same   = min(base, head)
    """
    out: Dict[str, LabelDelta] = {}
    keys = set(base_counts.keys()) | set(head_counts.keys())
    for k in sorted(keys):
        b = int(base_counts.get(k, 0))
        h = int(head_counts.get(k, 0))
        ld = LabelDelta(
            add=max(0, h - b),
            delete=max(0, b - h),
            same=min(b, h),
        )
        out[k] = ld
    return out


def ast_delta(base: AstSummary, head: AstSummary,
              base_ref: str = "base", head_ref: str = "head") -> AstDelta:
    """
    Porównuje dwie struktury AstSummary. Nie robi heurystyk mapowania węzłów —
    działa na zliczeniach i skalarnych metrykach S/H/Z, α/β.
    """
    # per-label
    labels = diff_label_counts(base.per_label, head.per_label)
    changed = sum(1 for v in labels.values() if (v.add > 0 or v.delete > 0))

    return AstDelta(
        file_path=head.file_path if head.file_path else base.file_path,
        base_ref=str(base_ref),
        head_ref=str(head_ref),

        dS=int(head.S - base.S),
        dH=int(head.H - base.H),
        dZ=int(head.Z - base.Z),

        dalpha=float(head.alpha - base.alpha),
        dbeta=float(head.beta - base.beta),

        S_base=int(base.S), H_base=int(base.H), Z_base=int(base.Z),
        alpha_base=float(base.alpha), beta_base=float(base.beta),

        S_head=int(head.S), H_head=int(head.H), Z_head=int(head.Z),
        alpha_head=float(head.alpha), beta_head=float(head.beta),

        labels=labels,

        n_labels_base=len(base.per_label),
        n_labels_head=len(head.per_label),
        changed_labels=int(changed),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Wygodne wywołania pomocnicze
# ──────────────────────────────────────────────────────────────────────────────

def delta_from_sources(base_src: str, head_src: str, file_path: str = "<memory>",
                       base_ref: str = "base", head_ref: str = "head") -> AstDelta:
    """
    Liczy Δ bezpośrednio z dwóch źródeł tekstowych (stringów).
    """
    bsum = ast_summary_of_source(base_src, file_path=f"{base_ref}:{file_path}")
    hsum = ast_summary_of_source(head_src, file_path=f"{head_ref}:{file_path}")
    return ast_delta(bsum, hsum, base_ref=base_ref, head_ref=head_ref)


def delta_of_file_between_revs(path: str, base_rev: str, head_rev: str = "HEAD") -> Optional[AstDelta]:
    """
    Liczy Δ dla pliku `path` między dwiema rewizjami gita (base_rev..head_rev).
    Jeśli plik nie istnieje w którymś z rev – zwraca None (zachowanie konserwatywne).
    """
    bsum = ast_summary_of_rev(path, base_rev)
    hsum = ast_summary_of_rev(path, head_rev)
    if bsum is None or hsum is None:
        return None
    return ast_delta(bsum, hsum, base_ref=base_rev, head_ref=head_rev)


def delta_of_local_file(path_base: str, path_head: str,
                        base_ref: str = "base_file",
                        head_ref: str = "head_file") -> Optional[AstDelta]:
    """
    Δ dla dwóch lokalnych plików (po ścieżce). Jeśli któryś nie istnieje – None.
    """
    bsum = ast_summary_of_file(path_base)
    hsum = ast_summary_of_file(path_head)
    if bsum is None or hsum is None:
        return None
    return ast_delta(bsum, hsum, base_ref=base_ref, head_ref=head_ref)


def merge_deltas(deltas: List[AstDelta], file_path: str = "<multi>") -> AstDelta:
    """
    Agreguje kilka AstDelta (np. wiele plików) do jednego skrótu.
    Per-label sumujemy add/delete/same.
    S/H/Z i α/β – sumy i średnie ważone przez (S+H); referencje sklejone.
    """
    if not deltas:
        raise ValueError("merge_deltas: empty list")

    def w_pair(d: AstDelta) -> Tuple[float, float, float]:
        w = max(1.0, float(d.S_head + d.H_head))
        return w, d.alpha_head * w, d.beta_head * w

    # skalarne sumy (Δ)
    dS = sum(d.dS for d in deltas)
    dH = sum(d.dH for d in deltas)
    dZ = sum(d.dZ for d in deltas)

    # α/β – ważone headem
    W = sum(w_pair(d)[0] for d in deltas)
    alpha_head_w = sum(w_pair(d)[1] for d in deltas) / max(1.0, W)
    beta_head_w = sum(w_pair(d)[2] for d in deltas) / max(1.0, W)

    # Base i head (sumy S/H/Z; α/β jako ważone)
    S_base = sum(d.S_base for d in deltas)
    H_base = sum(d.H_base for d in deltas)
    Z_base = sum(d.Z_base for d in deltas)
    S_head = sum(d.S_head for d in deltas)
    H_head = sum(d.H_head for d in deltas)
    Z_head = sum(d.Z_head for d in deltas)

    # α/β base – policzmy podobnie ważone
    Wb = sum(max(1.0, float(d.S_base + d.H_base)) for d in deltas)
    alpha_base_w = sum(max(1.0, float(d.S_base + d.H_base)) * d.alpha_base for d in deltas) / max(1.0, Wb)
    beta_base_w = sum(max(1.0, float(d.S_base + d.H_base)) * d.beta_base for d in deltas) / max(1.0, Wb)

    # per-label agregacja
    labels: Dict[str, LabelDelta] = {}
    for d in deltas:
        for k, v in d.labels.items():
            cur = labels.get(k) or LabelDelta()
            labels[k] = LabelDelta(
                add=cur.add + v.add,
                delete=cur.delete + v.delete,
                same=cur.same + v.same
            )

    changed = sum(1 for v in labels.values() if (v.add > 0 or v.delete > 0))

    return AstDelta(
        file_path=file_path,
        base_ref=" + ".join(d.base_ref for d in deltas),
        head_ref=" + ".join(d.head_ref for d in deltas),

        dS=int(dS), dH=int(dH), dZ=int(dZ),
        dalpha=float(alpha_head_w - alpha_base_w),
        dbeta=float(beta_head_w - beta_base_w),

        S_base=int(S_base), H_base=int(H_base), Z_base=int(Z_base),
        alpha_base=float(alpha_base_w), beta_base=float(beta_base_w),

        S_head=int(S_head), H_head=int(H_head), Z_head=int(Z_head),
        alpha_head=float(alpha_head_w), beta_head=float(beta_head_w),

        labels=labels,

        n_labels_base=0,  # bez sensu łączyć – można pominąć/obliczyć inaczej
        n_labels_head=0,
        changed_labels=int(changed),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Serializacja (lekka)
# ──────────────────────────────────────────────────────────────────────────────

def to_jsonable(delta: AstDelta) -> Dict:
    """
    Zwraca „płaski” JSON-owalny słownik (label delty rozwinięte).
    """
    labels = {k: asdict(v) for k, v in delta.labels.items()}
    return dict(
        file_path=delta.file_path,
        base_ref=delta.base_ref,
        head_ref=delta.head_ref,
        dS=delta.dS, dH=delta.dH, dZ=delta.dZ,
        dalpha=round(delta.dalpha, 6), dbeta=round(delta.dbeta, 6),
        S_base=delta.S_base, H_base=delta.H_base, Z_base=delta.Z_base,
        alpha_base=round(delta.alpha_base, 6), beta_base=round(delta.beta_base, 6),
        S_head=delta.S_head, H_head=delta.H_head, Z_head=delta.Z_head,
        alpha_head=round(delta.alpha_head, 6), beta_head=round(delta.beta_head, 6),
        labels=labels,
        n_labels_base=delta.n_labels_base,
        n_labels_head=delta.n_labels_head,
        changed_labels=delta.changed_labels,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Minimalne self-testy CLI (opcjonalnie)
# ──────────────────────────────────────────────────────────────────────────────

def _cli(argv: Optional[List[str]] = None) -> None:
    """
    Szybki podgląd:
      python -m glitchlab.analysis.ast_delta rev PATH baseSHA headSHA
      python -m glitchlab.analysis.ast_delta file BASE.py HEAD.py
    """
    import argparse, json as _json
    p = argparse.ArgumentParser(prog="ast_delta", description="ΔAST base vs head")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rev", help="Δ dla pliku między dwiema rewizjami gita")
    r.add_argument("path")
    r.add_argument("base_rev")
    r.add_argument("head_rev")

    f = sub.add_parser("file", help="Δ dla dwóch lokalnych plików")
    f.add_argument("base_path")
    f.add_argument("head_path")

    s = sub.add_parser("src", help="Δ dla dwóch źródeł (literałów)")
    s.add_argument("base_src")
    s.add_argument("head_src")
    s.add_argument("--path", default="<memory>")

    args = p.parse_args(argv)
    if args.cmd == "rev":
        d = delta_of_file_between_revs(args.path, args.base_rev, args.head_rev)
    elif args.cmd == "file":
        d = delta_of_local_file(args.base_path, args.head_path)
    else:
        d = delta_from_sources(args.base_src, args.head_src, file_path=args.path)

    if d is None:
        print(_json.dumps({"ok": False, "error": "file not present in one of revisions/paths"}, indent=2))
        return
    print(_json.dumps(to_jsonable(d), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
