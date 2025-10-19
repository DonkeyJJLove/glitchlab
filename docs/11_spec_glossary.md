# GlitchLab — Spec: słownik pojęć (glossary) · vNext

> Ten dokument jest **kanonicznym słownikiem** pojęć używanych w GlitchLab. Definiuje terminy matematyczne, nazwy bytów, identyfikatory, artefakty oraz zasady nazewnictwa. Ma charakter **SSOT** dla całej platformy (kod, GUI/HUD, CI, dokumentacja).

---

## 0. Konwencje i normatywność

* **MUST / MUST NOT / SHOULD / MAY** – słowa normatywne (RFC 2119).
* **Wersjonowanie**: semantyczne (SemVer) dla pakietu; schematy mają własne sufiksy `.vN`.
* **Skale**:

  * Pomiary intensywności/energii: `0..1` (float64, chyba że zaznaczono inaczej).
  * Odległości / normy: `ℝ≥0`.
* **Czas**: znaczniki w ISO-8601 UTC (`Z`).
* **Ścieżki**: zawsze względem **root repozytorium** (kotwica: `git`).

---

## 1. Rdzeń matematyki i operatorów

### 1.1 Wektory stanu i zmiany

* **S** – *struktura syntaktyczno-semantyczna kodu*
  Syntaktyka i topologia kodu: AST, graf importów, sygnatury API, fan-in/fan-out, złożoność (CC), typy, zależności modułowe.

* **H** – *sygnały heurystyczne w przestrzeni Mozaiki*
  Projekcje/cechy geometryczne: gęstości/gradienty, mapy „heat”, bloki, widmo (FFT), histogramy, wskaźniki podobieństwa (SSIM/PSNR) dla artefaktów wizualnych i „map kodu”.

* **Z** – *energia agregowana*
  Skalar opisujący „napięcie” zmiany (ryzyko/ciężar): kombinacja ważona cech `S` i `H`, znormalizowana do `0..1`.

* **Δ = (ΔS, ΔH, ΔZ)** – *krok zmiany*
  Elementarna różnica między dwoma stanami: tokeny AST-diff, różnice sygnatur, zmiany w mozaice/heatmapach i wynikowa energia ΔZ.

> **Intuicja**: S „co i jak jest zbudowane”; H „jak to wygląda w mapie/obrazku”; Z „ile zmiany niesiemy”.

---

### 1.2 Operator sprzęgający

* **Φ : AST → Mozaika** (*forward map*)
  Odwzorowanie struktury `S` na **siatkę** komórek (Mozaika). Każdej jednostce kodu (funkcja/klasa/plik) przypisuje region(y) siatki; niesie warstwy tematyczne (np. fan-in, CC, importy).

* **Ψ : Mozaika → AST** (*inverse map – przybliżony*)
  Rekonstrukcja (przybliżona) obszarów AST dotkniętych zmianą na podstawie geometrii i intensywności w Mozaice.

* **Komutacja z Δ**
  Wymaganie, aby „zmiana w AST” po przejściu przez Φ była równoważna „zmianie w Mozaice”:
  `Φ(Δ_AST) ≈ Δ_MOZ` (norma i próg w inwariantach I3).

---

## 2. Inwarianty jakości (skrót pojęć)

> **Pełne reguły, progi i polityki** żyją w `/spec/invariants.yaml` oraz w dokumentacji inwariantów. Poniżej definicje pojęciowe.

* **I1 – Energia ΔZ w granicach**
  ΔZ *MUST* mieścić się poniżej progu `α(module)` (lub z karą „soft”). Nadmiar = sygnał ryzyka.

* **I2 – Kontrakty API / portów**
  Publiczne sygnatury i stabilność interfejsów *MUST NOT* ulec niedozwolonej zmianie. Dopuszczalne są adaptery kompatybilne (np. deprecations).

* **I3 – Komutacja Φ z Δ**
  `|| Φ(Δ_AST) – Δ_MOZ || ≤ τ`. Naruszenie = niespójność między tym, co zmieniono w kodzie, a tym, co widzi Mozaika.

* **I4 – Stabilność obserwowalna**
  Test-coverage, metryki jakości (SSIM/PSNR dla wizualiów, regresja czasów kluczowych kroków, itp.) *MUST NOT* spaść poniżej progów.

**Decyzje bramek (w CI/GUI):**
`score ≤ α → ACCEPT` • `α < score ≤ β → REVIEW` • `score > β → BLOCK` • `≫ Z → HARD BLOCK`.

---

## 3. Δ-Tokens i podpis zmian

### 3.1 Tokeny (kategorie)

Nazwy **kanoniczne** (prefiks „Δ” oznacza modyfikację wartości):

* Struktura i sygnatury:

  * `ADD_FN`, `DEL_FN`, `RENAME_FN`
  * `MODIFY_SIG` (np. `Δarity`, `Δdefaults`, `Δtypes`)
  * `EXTRACT_FN`, `INLINE_FN`
  * `MOVE_BLOCK`, `REORDER`

* Importy i zależności:

  * `ΔIMPORT` (`added[]`, `removed[]`)
  * `ΔDEPENDENCY` (pyproject/deps)

* Typowanie i kontrakty:

  * `ΔTYPE_HINTS`, `ΔDOCSIG` (kontrakt dokumentacyjny)

* Testy i jakość:

  * `ΔTESTS` (± liczba testów/pokrycie)
  * `ΔCC` (złożoność cyklomatyczna)
  * `ΔFAN_IN`, `ΔFAN_OUT`

* Pliki/struktura repo:

  * `ADD_FILE`, `DEL_FILE`, `RENAME_FILE`, `MOVE_FILE`

**Atrybuty wspólne:**

```json
{ "kind":"MODIFY_SIG", "file":"path.py", "line":42, "detail":{ "...": "..." } }
```

### 3.2 Fingerprint (podpis PR/commitu)

Zdeterministyczne streszczenie zmian:

* **Histogram** tokenów (counts).
* **Bigramy** kolejnościowe (np. `ADD_FN→MODIFY_SIG`).
* **Hash**: `fp:sha256:<32B>` od kanonicznej reprezentacji.

---

## 4. Artefakty i ich semantyka

> Artefakty są **JSON/Parquet/PNG** z wersjami schematów (`*.vN`). Wszystkie odkładamy w `.glx/` (repo-root).

* **`delta.tokens.v1`** — lista tokenów z atrybutami (sekcja 3).
* **`delta.fp.v1`** — fingerprint (histogram, bigramy, hash).
* **`delta.report.v1`** — raport decyzji: `{tokens, fingerprint, violations[], score{α,β,ζ,value,decision}}`.
* **`spec.state.v1`** — stan kalibracji progów (per moduł), statystyki (EWMA/MAD/kwantyle), status driftu.
* **`metrics.parquet`** — strumień wektorów cech Δ (do kalibracji).
* **`mosaic_*.png`** — wizualizacje Mozaiki (warstwy/rozmiary w nazwie).
* **`security/merged.sarif`** — zunifikowane SAST (patrz §7).

---

## 5. Identyfikatory i formaty nazw

* **AST Node ID**: `AST::<file>@<line>:<col>#<stable-hash>`
  *Przykład*: `AST::core/pipeline.py@210:8#f3c8…`

* **Komórka Mozaiki**: `MOZ::<grid>@r<row>c<col>`
  *Przykład*: `MOZ::64x64@r12c07`

* **Finding (SAST)**: `NF-<base32(sha256[raw key])>`
  *Przykład*: `NF-8R2J…`

* **Fingerprint hash**: `fp:sha256:<hex>`
  *Przykład*: `fp:sha256:7a9f…`

* **Wersje schematów**: sufiks `.v1`, `.v2`, …
  *Przykład*: `delta.report.v1`

* **Tematy EventBus**: `namespace.subject[.verb]`
  *Przykład*: `core.step.finished`, `invariants.violation`, `SAST.FindingsReady`

---

## 6. Soczewka GLX (Lens) – metaopis generacji

**Cel**: jeden opis (YAML) sterujący generacją/inspekcją pliku/artefaktu.

Pola **kanoniczne**:

```yaml
lens: v1
id: glx.core.metrics.basic
kind: python_module | bash_script | doc_page
inputs:
  - src: core/metrics/basic.py
  - ctx: .glx/spec_state.json
outputs:
  - path: core/metrics/basic.py
  - path: docs/core_metrics_basic.md
invariants: [I1_monotonicity_deltaZ, I3_commutes_with_delta]
security.checks: [no_exec_eval, disallow_subprocess_shell]
generation.strategy:
  - template_first
  - diff_guided_prompting
  - retrieve_repo_context: true
```

**Wymagania**:

* **MUST** wskazywać realne ścieżki w repo (kotwica: `git`).
* **MUST** deklarować inwarianty i reguły bezpieczeństwa, jeśli generuje kod.
* **SHOULD** mieć deterministyczne szablony (template-first), LLM tylko uzupełnia.

---

## 7. SAST-Bridge – pojęcia i kontrakty

* **NF (Normalized Finding)** – zunifikowany rekord naruszenia:

  * `tool`, `rule_id`, `cwe`, `severity`, `confidence`, `category`
  * `location{file,line,col}`, `evidence{snippet,trace[]}`
  * `ast_ref`, `mosaic_cell`, `delta_bind` (`added|modified|unchanged`)
  * `suppression{state,reason}`, `fingerprint`

* **PQ (Prioritized Queue)** – NF z `risk_score` (funkcja wag severities, confidence, reachability, delta/public_api weights).

* **FC (FixCandidate)** – kontrakt na propozycję poprawki:

  * `pattern`, `fix_hint`, `patch_spec{file,hunk}`, `tests_required[]`
  * `constraints{I1..I4, Δ-limit}`, `review_gate{human|auto}`

* **Zdarzenia** (tematy):
  `SAST.ScanRequested`, `SAST.FindingsReady`, `SAST.Prioritized`, `SAST.FixRequested`, `SAST.FixProposed`, `SAST.FixValidated`.

---

## 8. EventBus – koperty i reguły

**Envelope (uogólniony):**

```json
{
  "topic":"core.step.finished",
  "ts":"2025-10-09T12:34:56Z",
  "sha":"<git sha>",
  "payload":{ "...": "..." },
  "schema":"spec/schemas/events/core.step.finished.v1.json"
}
```

**Reguły:**

* `topic` **MUST** match schemat (JSON Schema).
* `payload` **MUST** być minimalny (bez duplikowania dużych blobów – do artefaktów).
* Zdarzenia ważne dla GUI *SHOULD* mieć link do artefaktów `.glx/*`.

---

## 9. Spec żywy i kalibracja (pojęcia)

* **EWMA** – średnia ważona wykładniczo dla trendu.
* **MAD** – odporna dyspersja (skala ∝ 1.4826·MAD).
* **Kwantyle** – `q90/q95/q99` estymowane strumieniowo (t-digest/P²).
* **Drift** – wykrywanie (Page-Hinkley/ADWIN); stan `frozen` w `spec.state.v1`.

---

## 10. Bezpieczeństwo i polityki

* **Fail-closed** – domyślnie blokujemy przy braku danych/niepewności.
* **.env reżim** – odczyt **wyłącznie** z katalogu projektu; ścieżki wyjściowe **wewnątrz repo**.
* **Sandbox pluginów** – procesy odseparowane, białe listy importów.
* **Sekrety** – **MUST NOT** trafić do artefaktów; naruszenia → Gitleaks → NF/FC.

---

## 11. Konwencje dokumentacji

* Pliki `.md` w `docs/` opisują warstwy:
  `00_overview.md` (przegląd), `10_architecture.md` (architektura), `11_spec_glossary.md` (ten dokument), `12_spec_invariants.md` (reguły), `2x_*` (moduły), `3x_*` (GUI/HUD), `4x_*` (SAST/bezpieczeństwo).

* W dokumentach **MUST** używać nazw kanonicznych z tego słownika.

---

## 12. Słownik skrótów i symboli (A–Z)

* **AST** – Abstract Syntax Tree.
* **CC** – Cyclomatic Complexity.
* **CI/CD** – Continuous Integration / Continuous Delivery.
* **CWE** – Common Weakness Enumeration.
* **EGDB** – Event/Graph Database (baza zdarzeń i metryk).
* **EWMA / MAD** – patrz §9.
* **FP** – Fingerprint (podpis zmian), też „False Positive” w kontekście SAST (kontekst rozróżnia).
* **HUD** – Heads-Up Display (panele GUI).
* **NF / PQ / FC** – SAST: Normalized Finding / Prioritized Queue / FixCandidate.
* **Φ/Ψ** – operatory sprzęgające AST↔Mozaika.
* **Δ** – krok różnicy; **ΔZ** – energia zmiany.
* **α/β/ζ** – progi decyzyjne (quantile-based).
* **Mozaika** – siatka (grid) nośnika wizualnego zmian i warstw semantycznych.
* **Lens** – metaopis generacji/inspekcji dla pliku/artefaktu.
* **I1–I4** – inwarianty jakości (patrz §2).
* **S/H/Z** – stan (struktura/heurystyka/energia), §1.1.

---

## 13. Zgodność i migracje

* Zmiana formatu artefaktu wymaga **inkrementacji wersji schematu** (`*.vN`) oraz dostarczenia migratora (jeśli możliwe).
* **GUI/HUD** *SHOULD* wspierać przynajmniej `vN` i `vN-1`.

---

## 14. Minimalny przykład (koniec-do-końca)

1. Dev zmienia `core/mosaic.py` (dodaje funkcję, zmienia sygnaturę).
2. Tokenizer tworzy `delta.tokens.v1`:

   * `ADD_FN`, `MODIFY_SIG`, `ΔIMPORT`.
3. Fingerprint → `delta.fp.v1` (`fp:sha256:…`).
4. Φ odwzorowuje zmiany na Mozaikę → `mosaic_delta.png`.
5. Inwarianty oceniają Δ:

   * I3: zgodność Φ(Δ_AST) z Δ_MOZ – *OK*.
   * I2: brak złamania publicznego API – *OK*.
   * I1: ΔZ ≤ α(core) – *REVIEW*.
6. `delta.report.v1` → decyzja `REVIEW`; GUI „Delta Inspector” wyjaśnia **dlaczego**.
7. Po merge CI aktualizuje `spec.state.v1` (EWMA/MAD/kwantyle).

---

## 15. Odniesienia (wewnętrzne)

* `docs/00_overview.md` – przegląd produktu.
* `docs/10_architecture.md` – architektura techniczna.
* `docs/12_spec_invariants.md` – definicje formalne i progi.
* `spec/invariants.yaml` – żywe reguły (wersjonowane w repo).
* `spec/schemas/` – JSON Schemas artefaktów i zdarzeń.

---

**Stan: Final (Glossary).**
Ten plik **MUST** być aktualizowany przy każdej zmianie terminologii lub dodaniu nowej kategorii tokenów/artefaktów.
