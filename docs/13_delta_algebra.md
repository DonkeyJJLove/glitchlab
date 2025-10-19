# GlitchLab — Spec: **Δ-Algebra tokenów** i Fingerprint PR · vNext

> Ten dokument **normatywnie** definiuje algebraiczny opis zmian kodu w GlitchLab: **tokeny Δ**, reguły ich ekstrakcji i łączenia (monoid `⊕`), odwzorowanie na **wektor cech** `z(Δ)`, **energię** `ΔZ`, oraz stabilny **fingerprint** commita/PR. Spec współgra z inwariantami **I1–I4** (docs/12_invariants.md) i mapowaniem **Φ/Ψ** (AST↔Mozaika).

---

## 0) Zakres, wersjonowanie i konwencje

* Słowa **MUST / SHOULD / MAY** są normatywne (RFC 2119).
* Niniejszy plik opisuje **schemat `delta.tokens.v1`** i **`delta.fingerprint.v1`**.
* Wszystkie przykłady JSON są **informacyjne**, zgodne z definicjami poniżej.
* Artefakty i stany zapisujemy pod `.glx/` (kotwica: git root, UTC ISO-8601).

---

## 1) Intuicja (TL;DR)

Zmiana kodu opisujemy **nie diffem linii**, lecz zbiorem **tokenów semantycznych** (np. `MODIFY_SIG`, `ΔIMPORT`, `EXTRACT_FN`). Tokeny:

1. mają **ładunek** (wagi, źródło, lokalizacja),
2. **składają się** monoidalnie `⊕` (kolejność zachowana, uproszczenia regułami),
3. tworzą wektor cech `z(Δ)` → **energia `ΔZ`** → bramki I1,
4. są rzutowane przez **Φ** na maskę `Φ(Δ_AST)` → **I3** (komutacja z Mozaiką),
5. budują stabilny **fingerprint** PR.

---

## 2) Słownik pojęć (zależne od docs/11_spec_glossary.md)

* **Token Δ** — atomowa, semantyczna jednostka zmiany AST/kodu.
* **Payload tokenu** — szczegóły semantyczne i lokalizacyjne (plik, linia, uid AST, before/after).
* **Monoid Δ** — algebra łączenia tokenów (`⊕`, element neutralny `ε`, reguły uproszczeń).
* **Histogram / n-gramy** — statystyka tokenów (częstości, kolejność lokalna).
* **Fingerprint** — deterministyczny skrót `fp(Δ)` odporny na szum formatowania.
* **Φ/Ψ** — rzutowanie AST→Mozaika i odwrotnie (por. docs/10_architecture.md).

---

## 3) Taksonomia tokenów (kanon `delta.tokens.v1`)

### 3.1 Klasy główne

1. **Struktura (AST)**
   `ADD_FN`, `DEL_FN`, `RENAME_FN`, `EXTRACT_FN`, `INLINE_FN`,
   `ADD_CLASS`, `DEL_CLASS`, `RENAME_CLASS`,
   `MOVE_SYMBOL`, `MOVE_FILE`, `SPLIT_FILE`, `MERGE_FILE`,
   `MODIFY_SIG(arity|types|defaults)`.

2. **Przepływ i złożoność**
   `ΔCC(+k|-k)` (Cyclomatic), `ΔLOOP_NEST(+/-)`, `ΔEXCEPT_BLOCKS(+/-)`,
   `ΔBRANCHING(+/-)`, `ΔASYNC_AWAIT(+/-)`.

3. **Zależności**
   `ΔIMPORT(+mod|-mod)`, `RELOCATE_IMPORT`, `ΔFAN_IN(+/-)`, `ΔFAN_OUT(+/-)`.

4. **Typowanie i kontrakty**
   `ΔTYPE_HINTS(+/-/tighten/loosen)`, `ADD_PROTOCOL`, `ADD_INTERFACE`.

5. **Bezpieczeństwo (SAST-Bridge)**
   `SAST_FINDING(new|worse|fixed)`, `SECRETS_FOUND`, `CRYPTO_WEAK`, `SHELL_TRUE`.

6. **Testy i dokumentacja**
   `ΔTESTS(+/-)`, `ADD_STUB_TESTS`, `DOC_PUBLIC_API(+/-)`, `ΔREADME`, `ΔSPEC`.

7. **GUI/Artefakty wizualne**
   `ΔHUD_PANEL`, `ΔMOSAIC_LAYER`, `ΔEXPORTER_SCHEMA`.

> **MUST:** Każdy token należy do jednej klasy głównej `category ∈ {ast, flow, deps, types, sec, tests, docs, gui}`.

### 3.2 Pola tokenu (schema)

```json
{
  "schema": "delta.tokens.v1",
  "kind": "MODIFY_SIG",
  "category": "ast",
  "file": "glitchlab/core/pipeline.py",
  "range": {"line": 210, "col": 5, "endLine": 238},
  "uid": "AST::<stable-node-id>",
  "before": {"arity": 2, "types": ["np.ndarray","dict"]},
  "after":  {"arity": 3, "types": ["np.ndarray","dict","RunCtx"]},
  "weight_hint": 0.22,
  "evidence": {"snippet": "def run(self, img, ctx): ..."},
  "meta": {"public_api": true, "module": "core"}
}
```

**Pola obowiązkowe:** `schema`, `kind`, `category`, `file`, `uid`.
**Pola zalecane:** `range`, `before/after`, `meta.public_api`, `weight_hint`.

---

## 4) Ekstrakcja tokenów (normatyw)

### 4.1 Źródła

* **libcst/ast** — strukturalny diff AST (rename/move/modify_sig/extract/inline).
* **radon** — złożoność (`ΔCC`), gałęzie, zagnieżdżenia.
* **Import graph** (analiza własna) — `ΔIMPORT`, `ΔFAN_IN/OUT`.
* **SAST-Bridge** — zdarzenia bezpieczeństwa: `SAST_FINDING`, `SECRETS_FOUND`, …
* **Git diff** — fallback do heurystyk (docs/tests/GUI).

### 4.2 Reguły wykrycia (skróty)

* **RENAME_FN**: dopasowanie podobieństwa ciała i sygnatury > θ_rename (domyślnie 0.85 Jaccard/AST).
* **MOVE_SYMBOL**: identyczny `uid`/hash ciała, inny `module_path`.
* **EXTRACT_FN**: blok `before` z dużym podobieństwem rozkłada się na `after` + nowy `ADD_FN` (współdzielony podgraf wywołań).
* **ΔCC**: `CC_after - CC_before`.
* **ΔTYPE_HINTS**: różnica w adnotacjach; `tighten/loosen` na bazie relacji podtypów.
* **Public API**: moduły oznaczone `public` w `/spec/` lub przez dekorator/front-matter.

**MUST:** Tokeny pochodzące z różnych extraktorów mają **ujednolicone UIDs** (mapa `path+node_span → uid`) i **deduplikację** (sekcja 6).

---

## 5) Algebra tokenów: monoid `⟨Δ, ⊕, ε⟩`

### 5.1 Własności

* `ε` — element neutralny (brak zmian).

* `⊕` — **asocjacyjne**, niekomutatywne (zachowujemy kolejność czasoprzestrzenną).

* **Uproszczenia (rewrite rules):**

  1. `ADD_FN ⊕ DEL_FN` tego samego `uid` w jednym PR → `ε` (lokalne odwrócenie).
  2. `RENAME_FN ⊕ RENAME_FN` (łańcuch) → pojedyncze `RENAME_FN(src→dst_final)`.
  3. `MOVE_SYMBOL ⊕ RENAME_FN` → `MOVE_SYMBOL+RENAME_FN` jeśli zmienia się i plik, i nazwa.
  4. `MODIFY_SIG ⊕ MODIFY_SIG` → scal w jedną z agregacją różnic (`arity`, `types`, `defaults`).
  5. `ΔIMPORT(+X) ⊕ ΔIMPORT(-X)` bez pośrednich użyć → `ε`.

* **Porządek**: sort stable — `by(file, start_line, seq_id)`; SAST i bezpieczeństwo **po** zmianach AST (żeby mogły się do nich zbindować).

### 5.2 Normalizacja

`normalize(tokens)` MUST wykonywać deduplikację (UID/zakres), rewritery i sort stabilny, gwarantując deterministyczny wynik.

---

## 6) Deduplikacja i fuzja

* Klucz deduplikacji: `(uid|hash_code, kind, file, line_bucket)` gdzie `line_bucket = floor(line/5)*5`.
* Fuzja atrybutów:

  * **severity** = max, **confidence** = średnia ważona kontekstem,
  * **trace/evidence** = unia z ograniczeniem do N wpisów,
  * **weight_hint** = max.

**MUST:** Zduplikowane tokeny z różnych ekstraktorów łączymy, zachowując **najbardziej restrykcyjny** ładunek (np. `public_api:true`).

---

## 7) Od tokenów do cech `z(Δ)` i energii `ΔZ`

### 7.1 Wektor cech (kanon)

* **Histogram** tokenów per kategoria i `kind` (częstości znormalizowane).
* **N-gramy (bi/tri)** z okna `k` (domyślnie 3) — zachowanie sekwencji (np. `EXTRACT_FN→ADD_STUB_TESTS`).
* **Cechy kontekstowe**: ΔLOC, ΔFiles, ΔFAN_IN/OUT, udział `public_api`, hot-spot (często modyfikowany plik).
* **SAST**: liczba/ciężar CRITICAL/HIGH, secrets.
* **GUI/vis**: ΔHUD/ΔMOSAIC jeśli dotyczy.

### 7.2 Energia `ΔZ`

```
ΔZ_raw = wᵀ · z(Δ)        # w – wagi per moduł (żywe, w /spec/invariants.yaml)
ΔZ     = norm01(ΔZ_raw)   # kalibracja przez spec_state (EWMA+MAD+kwantyle)
```

**SHOULD:** penalizować sekwencje n-gramów znanych jako ryzykowne (np. `RENAME_CLASS` bez `ADD_DEPRECATION_ALIAS`).

---

## 8) Fingerprint `fp(Δ)` (stabilny skrót PR)

### 8.1 Cel

Zapewnić **deterministyczny identyfikator** zmiany, odporny na: formatowanie, reorder importów, białe znaki. Zależny od **semantyki** (tokenów) i ich **lokalnego porządku**.

### 8.2 Algorytm (normatywny)

1. `T = normalize(tokens)`
2. `S = []` — wektor symboli:
   Dla każdego `t ∈ T`: `S.append(f"{t.kind}@{t.meta.get('module','?')}#{bucket(t.line)}")`
   Dodatkowo dołóż **bigrams** `S2` z sąsiadów w tym samym pliku.
3. `payload = "|".join(S + S2)`
4. `fp = "fp:sha256:" + sha256(payload.encode('utf-8')).hexdigest()`

**Opcjonalnie (MAY):** dołączyć skrót mapy `public_api` i hashe ciał funkcji `RENAME/MOVE` (zwiększa kolizyjność odporności).

### 8.3 Przykład

```
S = [
  "MODIFY_SIG@core#210", "EXTRACT_FN@core#312", "ΔIMPORT@core#7",
  "ADD_STUB_TESTS@tests#14"
]
fp = fp:sha256:9e1f…b3
```

---

## 9) Rzutowanie Φ(Δ_AST) i Δ-Mozaika (I3)

### 9.1 Maska z tokenów

`phi_delta_mask(tokens, grid=G)` MUST:

* zamienić tokeny `ast/flow/deps/types` na **obszary zainteresowania** (ROI) w przestrzeni `Mozaiki` (komórki `G` związane z plikiem, modułem, symbolami),
* agregować intensywność per ROI:

  * `MODIFY_SIG`, `RENAME/MOVE` → **wysoka** intensywność,
  * `ΔIMPORT`/`ΔTYPE_HINTS` → **średnia**,
  * `tests/docs` → **niska** (chyba że `public_api:true`).

### 9.2 Zgodność z Δ_MOZ

Wartość **E** (por. docs/12_invariants.md) liczona z maski `Φ(Δ_AST)` i `Δ_MOZ` definiuje I3.
**MUST:** raport zawierać mapę kontrybucji tokenów → piksele Mozaiki (wyjaśnialność).

---

## 10) Artefakty i schematy

### 10.1 `.glx/delta.tokens.v1.json`

```json
{
  "schema": "delta.tokens.v1",
  "sha": "8f1c…",
  "created_utc": "2025-10-11T10:20:12Z",
  "modules": ["core","analysis"],
  "tokens": [ { "...": "..." } ],              // jak w §3.2
  "stats": {
    "histogram": {"MODIFY_SIG": 1, "ΔIMPORT": 2},
    "bigrams": {"MODIFY_SIG→EXTRACT_FN": 1}
  }
}
```

### 10.2 `.glx/delta.fingerprint.v1.json`

```json
{
  "schema": "delta.fingerprint.v1",
  "sha": "8f1c…",
  "fingerprint": "fp:sha256:9e1f…b3",
  "payload_size": 512,
  "explain": ["from 4 tokens + 3 bigrams; modules: core, tests"]
}
```

---

## 11) Integracja z BUS i HUD

**Zdarzenia BUS (normatywne):**

* `delta.tokens` — `{ sha, tokens:[…], histogram, bigrams }`
* `delta.fingerprint` — `{ sha, fp, payload_size }`
* `delta.mosaic` — maska `Φ(Δ_AST)` dla HUD
* `invariants.violation` — powiązanie z I1/I3 (wskazanie tokenów i wkładów)

**HUD MUST**:

* wizualizować histogram i n-gramy,
* odsyłać do plików/zakresów,
* pokazywać wpływ tokenów na `ΔZ` i I3 (`why`).

---

## 12) Wydajność i deterministykę

* **Cache AST** per plik + **inkrementalne** tokenizowanie tylko zmienionych plików.
* **Stabilne UIDy** (hash drzewa + ścieżka) — MUST.
* **Czas**: cel < 300 ms/plik dla typowego modułu (Python 3.10–3.12).
* **Niezmienność**: ten sam PR → ten sam `fp` przy wielokrotnej analizie.

---

## 13) Edge-cases i polityki

* **Generowane/vendor**: ścieżki pasujące do `overrides.ignore_paths` są **wyłączone** lub mają wagi bliskie 0.
* **Duże refaktory** (`SPLIT_FILE/MERGE_FILE`) — tokeny **zgrubne** + sygnatura „refactor set” (HUD ostrzega: niski confidence).
* **Multimoduł**: porządek decyzyjny wg **max severity** modułowej (por. docs/12_invariants.md §7.2).
* **Docs-only**: tokeny `ΔSPEC/ΔREADME` → `ΔZ` ~ 0; I3=N/A.
* **Bezpieczeństwo**: `SHELL_TRUE`, `SECRETS_FOUND` automatycznie zwiększają `ΔZ` (penalty).

---

## 14) Testy kontraktowe (normatywne)

* **Jednostkowe**:

  * `test_tokenize_modify_sig` — poprawne pola i `weight_hint`.
  * `test_rewrite_rules` — `ADD_FN ⊕ DEL_FN → ε` itd.
  * `test_fingerprint_stability` — brak zmian → stały `fp`; reorder importów nie zmienia `fp`.

* **Property-based**:

  * `extract_inline_roundtrip` — `EXTRACT ⊕ INLINE` ~ `ε` (w granicach tolerancji).
  * `rename_chain_collapse` — łańcuch rename’ów → 1 rename.

* **Integracyjne (CI)**:

  * raport `.glx/delta.tokens.v1.json` i `.glx/delta.fingerprint.v1.json` MUST istnieć i przechodzić schemat.

---

## 15) Zgodność i migracje

* Zmiana pól tokenu lub reguł rewritingu → **inkrementuj** `delta.tokens.vN`; zapewnij migrator.
* HUD i narzędzia MUST wspierać `vN` i `vN-1`.
* Snapshot schematu dołączany do artefaktów raportu dla reprodukcji.

---

## 16) Przykład end-to-end (informacyjny)

**Zmiana:** dodanie `ctx` do `pipeline.run`, wyodrębnienie funkcji, nowy import i stub test.

Tokens (po normalizacji):

```json
[
  {"kind":"MODIFY_SIG","file":"core/pipeline.py","uid":"AST::f.run", "meta":{"public_api":true}},
  {"kind":"EXTRACT_FN","file":"core/pipeline.py","uid":"AST::f.build_ctx"},
  {"kind":"ΔIMPORT","file":"core/__init__.py","before":null,"after":"from .pipeline import build_ctx"},
  {"kind":"ADD_STUB_TESTS","file":"tests/test_pipeline_ctx.py"}
]
```

Fingerprint: `fp:sha256:9e1f…b3`
`z(Δ)` → `ΔZ = 0.41` (I1: **REVIEW**),
Φ(Δ_AST) vs Δ_MOZ → `E=0.08 ≤ τ` (I3: **PASS**),
I2: `MODIFY_SIG` **z adapterem** → **PASS**,
I4: testy przechodzą → **PASS**.
**Decyzja łączna:** **REVIEW** (por. docs/12_invariants.md §7.1).

---

## 17) Checklist wdrożeniowy

* [ ] Implementacja ekstraktorów (libcst, radon, import graph, SAST-Bridge).
* [ ] `normalize(tokens)` z rewriterami i deduplikacją.
* [ ] Generacja `.glx/delta.tokens.v1.json` i `.glx/delta.fingerprint.v1.json`.
* [ ] Mapowanie tokenów → `z(Δ)` + integracja wag z `/spec/invariants.yaml`.
* [ ] Φ-maski z tokenów + integracja I3 (raport wkładów).
* [ ] Panele HUD: histogram/n-gramy, śledzenie wkładu tokenów w `ΔZ` i I3.
* [ ] Testy kontraktowe + property-based + walidacja schematów.

---

**Status pliku:** ✅ **Final (Spec – Δ-Algebra & Fingerprint)**
Zmiany w tokenach, rewriterach, wagach lub algorytmie fingerprintu **MUST** zostać odzwierciedlone w tym pliku i w odpowiadających schematach `/spec/schemas/*.json`.
