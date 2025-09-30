# Progi & parametry analizy (GLX)

Dokument opisuje parametry sterujące generacją artefaktów **mozaika ⇄ AST** i wysyłką maili GLX.
Wartości domyślne są spójne z hookiem `post-commit` i modułem `hybrid_ast_mosaic`.

## 1) Minimalne progi decyzji (delta-only docs)

Domyślna polityka (próg „opłaca się wygenerować rozbudowane raporty”):

- `Δfiles_min = 2` — co najmniej 2 zmienione pliki `.py`
- `Δloc_min   = 20` — łączna zmiana linii (dodania + usunięcia) ≥ 20
- `Align_min  = 0.25` — średni Align po Ψ ≥ 0.25

**Wyjątki (zawsze generuj pełne raporty):**

- zmiany w `analysis/**`, `core/mosaic/**`, `core/ast/**`.

Te progi zastosujemy w `scripts/post_diff_report.py` (Etap 0). Jeśli zmiany są „cosmetic/test-only”,
generator zredukuje raport do minimum.

## 2) Parametry mozaiki / selektor Φ

- `GLX_MOSAIC` — rodzaj mozaiki: `grid` (domyślnie) lub `hex`.
- `GLX_ROWS`, `GLX_COLS` — rozmiar dyskretyzacji mozaiki (typowo 6×6 dla lekkiego profilu).
- `GLX_EDGE_THR` — próg krawędzi w [0..1]; determinuje regiony `edges` i liczbę komponentów `Z`.
- `GLX_PHI` — wariant selektora Φ:
  - `basic` — regułowy, prosty,
  - `balanced` — kwartyle Q25/Q75 (stabilniejszy; **domyślny**),
  - `entropy` — „rozmyty” przy granicy progu,
  - `policy` — **policy-aware**, patrz niżej.
- `GLX_POLICY` — ścieżka do `analysis/policy.json`; używana tylko, gdy `GLX_PHI=policy`.

## 3) Sprzężenie Ψ i wagi

- `GLX_DELTA` (`δ∈[0..1]`) — siła Ψ-feedback (aktualizacja meta wektorów na bazie regionów).
- `GLX_KAPPA` (`κ`) — intensywność sprzężenia (α,β) z profilem mozaiki; utrzymuje I1: α+β=1.

**Reguła zdrowego rozsądku:** jeśli repo jest bardzo małe lub zmiana dotyka niewielkiej liczby
kafli (`H` niskie), trzymaj `δ` i `κ` bliżej 0.2–0.35, by nie przesterować.

## 4) Tuning praktyczny

- **Za niski Align** (po Ψ):
  - obniż `GLX_EDGE_THR` o 0.05 (więcej kafli wejdzie do `edges`),
  - sprawdź `GLX_PHI=balanced` (jeśli był `basic`) lub włącz `policy`.
- **Zbyt „poszatkowane” Z** (dużo komponentów spójnych):
  - podnieś `GLX_EDGE_THR` o 0.05,
  - rozważ `GLX_ROWS/GLX_COLS` mniejsze (np. 5×5), by wygładzić maskę.
- **Policy-aware**:
  - w `analysis/policy.json` zdefiniuj preferencje (np. `prefer_edges`, `roi_labels`, `edge_thr_bias`).
  - Przykład:
    ```json
    {
      "avoid_roi_side_effects": true,
      "prefer_edges": true,
      "roi_labels": ["FunctionDef", "ClassDef"],
      "edge_thr_bias": -0.03
    }
    ```

## 5) Jak to uruchomić ręcznie

### a) Generacja artefaktów bez hooka

```bash
python -m glitchlab.mosaic.hybrid_ast_mosaic \
  --mosaic grid --rows 6 --cols 6 --edge-thr 0.55 --kappa-ab 0.35 \
  --phi balanced \
  from-git-dump --base <BASE_SHA> --head HEAD --delta 0.25
