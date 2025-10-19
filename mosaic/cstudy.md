# Case Study: Generacja i transformacja kodu z AST ⇄ Mozaika (Φ/Ψ)

>[Moziaka](mosaic/README.md) | [Struktura](mosaic/structure.md) | [Case Study](mosaic/cstudy.md)

> **Schemat wizualny:**
> ![Model – schemat](resourcesg/model.png)

To studium przypadku pokazuje **praktyczny** przepływ pracy: od analizy kodu (AST + meta), przez mapowanie Φ i feedback Ψ, po **reguły sterujące** (meta-tagi) i **wymuszenia architektoniczne** (np. *Dependency Injection*, *Factory*, wstrzymanie generacji). Całość opiera się na referencyjnych plikach:

* `glitchlab/mosaic/hybrid_ast_mosaic.py` – algorytm Φ/Ψ (AST⇄Mozaika)
* `vis_ast_kites_all.py` – produkcyjna wizualizacja „latawców” (polityk) per węzeł

---

## 1) Cel i skrót działania

**Chcemy**: sterować generacją/transformacją kodu tak, aby zbiegał do ustalonej ontologii (stylu/architektury) i **mierzalnych** celów.

**Jak to działa (w skrócie):**

1. **AST + meta**: każdy węzeł dostaje wektor `[L,S,Sel,Stab,Cau,H]`.
2. **Φ (phi)**: na bazie etykiety węzła i profilu mozaiki wybieramy region (`edges`, `~edges`, `roi`, `all`).
3. **Ψ (psi)**: miękko aktualizujemy meta wektor węzła zgodnie z charakterem regionu.
4. **Sprzężenie α/β**: balansujemy globalnie udział `α=S/(S+H)` i `β=H/(S+H)` z profilem mozaiki.
5. **Decyzja**: mierzymy `Align` (zgodność), koszty Φ (`J_phi`) i zgodność tagów (`J_meta`) — wybieramy wariant z najlepszą wartością wg kryteriów.

> Algorytm jest **łańcuchowy**: nie potrzebujesz „miliona warstw” — liczy się kolejność i proporcje nacisków (Φ/Ψ/αβ/λ).

---

## 2) Parametry, które naprawdę „robią różnicę”

* **`λ` (lambda)** – kompresja AST (spłaszczanie liści).
  *Wysokie* λ → mniej detali (nazwy/konstanty), mocniejszy szkielet.

* **`Δ` (delta)** – intensywność Ψ (jak mocno region koryguje meta).
  *Wyższe* Δ → szybsze dopasowanie do sceny/mozaiki.

* **`κ_ab` (kappa_ab)** – sprzężenie globalne α/β z mozaiką.
  *Wyższe* κ → generator chętniej „idzie za kontekstem”.

* **`edge_thr`** – próg tnący mozaikę na `edges` i `~edges`.
  Wyższy próg → `edges` są rzadsze (system preferuje stabilność).

* **Meta-tagi (`@meta …`)** – lokalne/globalne priorytety/ograniczenia (patrz §5).

---

## 3) Pipeline (reprodukowany)

```bash
# 1) Uruchom wizualizację (zapisz obraz)
python vis_ast_kites_all.py --rows 6 --cols 6 --edge-thr 0.55 --seed 7 --out out.png

# 2) Pojedynczy przebieg metryk (λ, Δ, itp.)
python -m glitchlab.mosaic.hybrid_ast_mosaic run --lmbd 0.60 --delta 0.25 --rows 6 --cols 6 --edge-thr 0.55

# 3) Sweep λ×Δ + tabela i JSON
python -m glitchlab.mosaic.hybrid_ast_mosaic sweep --rows 6 --cols 6 --edge-thr 0.55 --json
```

Wyniki: `Align` (↑ lepiej), `J_phi*` (↓ lepiej), `CR_AST`, `CR_TO`, `α`, `β` — do podejmowania decyzji.

---

## 4) Meta-tagi: pozytywne/negatywne priorytety

> **Po co?** Żeby deklaratywnie wpływać na Φ/Ψ (lokalnie lub globalnie).

**Składnia (przykłady):**

```python
# @meta +Sel>=0.75 in:edges       # promuj selektywność w edges
# @meta -Stab<0.40 in:edges       # demotywuj niską stabilność w edges
# @meta prefer:roi weight=0.6     # Φ preferuje ROI dla danego węzła
# @meta forbid:edges              # Φ nie może przypisać do edges
# @meta goal Align>=0.85          # globalny cel zgodności
```

Efekty:

* Φ: *prefer/forbid* → przełącza/priorytetyzuje region.
* Ψ: `dψ(tag)` → dosypuje lub zdejmuje komponent meta (np. Sel/Stab/Cau).
* global: koryguje `(α,β)` i/lub wagi `W_DEFAULT` w mierze `Align`.

**Ocena zgodności tagów:** `J_meta = Σ penalty(tag)·(1 − score_tag)` — można dodać do podpisu obrazu i/lub optymalizacji.

---

## 5) Case Study A — **Dependency Injection** (DI)

**Problem**: funkcja sama tworzy zależność (np. `Database()`), co miesza stabilizację i „akcję”.

**Kod wejściowy**

```python
def get_data():
    db = Database()
    return db.query("SELECT * FROM users")
```

**AST + Φ**

* `Assign(db=Database())` → `~edges` (utrwalenie), wysoka **Stab**.
* `Call(Database)` → `edges` (tworzenie), **Sel/Cau** wyżej.
* `Return` → `roi`.

**Ψ i konflikt**
Edges „ciągną” **Sel** w górę, ~edges „pompuje” **Stab** — to *rozjazd* meta.

**Transformacja (DI)**
Przenosimy zależność do parametru funkcji (do ROI):

```python
def get_data(db: Database):
    return db.query("SELECT * FROM users")
```

**Efekt metryczny**

* `Align` ↑ (mniej konfliktu, czystsza przyczynowość w ROI).
* `α` ↑ (więcej strukturalnej stabilności), `β` ↓ (mniej „nerwowych” Call w środku).
* `J_phi` ↓ (Call przestaje „krzyczeć” w złym regionie).

**Meta-tagi (opcjonalnie)**

```python
# @meta prefer:roi for:FunctionDef,Return
# @meta +Stab>=0.8 in:~edges for:Assign
# @meta forbid:edges for:Assign
```

---

## 6) Case Study B — **Wymuszenie wzorca Factory** (Maksymalny Realizm)

**Wejście** – jak w A, ale chcemy **architektonicznie** scentralizować tworzenie obiektów.

**Reguła**

* wykrycie powtarzających się `new()` / `Class()` w edges/~edges,
* Φ przełącza tworzenie do `roi`,
* generator dokleja **Factory**.

**Wynik**

```python
class DatabaseFactory:
    @staticmethod
    def create() -> Database:
        return Database()

def get_data():
    db = DatabaseFactory.create()
    return db.query("SELECT * FROM users")
```

**Efekt metryczny**

* `Cau` w ROI ↑ (przyczynowość w centrum konstrukcji),
* `Sel` w edges spada (Call prostszy, mniej „nerwowy”),
* `Align` ↑, `J_phi` ↓, często `CR_AST` bez zmian (struktura bardziej semantyczna, nie „cięższa”).

**Meta-tagi (opcjonalnie)**

```python
# @meta enforce:factory for:Database,FileReader,Client
# @meta +Cau>=0.7 in:roi for:FunctionDef
```

---

## 7) Case Study C — **Wstrzymanie generacji** (kontrolowane „stuby”)

**Motywacja**: sięgasz po zewnętrzne API/efekty uboczne w miejscu, gdzie nie masz danych lub chcesz zachować bezpieczeństwo.

**Mechanizm**

* jeśli `Sel`/`Cau` node’a w `edges` przekroczy próg (po Ψ), Φ **zamraża** poddrzewo (oznacza jako stub).

**Przykład**

```python
def calc(x):
    if x > 0:
        return external_api(x)   # wysokie Sel w edges
    return 0
```

**Wynik (stub)**

```python
# @meta stop:generate for:Call in:edges reason="external dependency"
result = external_api(x)   # TODO: fill
```

**Efekt**

* Zachowujesz spójność struktury (α), nie przesterowujesz `β`.
* `Align` nie leci w dół „na siłę” — *świadome pominięcie* generacji.

---

## 8) Decyzje i kryteria „go/no-go”

**Dobre sygnały:**

* `Align` ↑ (≥0.72 dla konfiguracji referencyjnych).
* `J_phi` ↓ (Φ trafia w regiony zgodnie z semantyką).
* `J_meta` ↓ (tagi zaspokojone).
* `CR_TO` sensowne (np. ≤20) — próg nie „psuje” dystrybucji.

**Złe sygnały:**

* `Align` spada przy wzroście Δ → Ψ zbyt silnie wypacza zamiast dopasowywać (zmniejsz `Δ`/`κ_ab`).
* `J_phi` rośnie przy λ↑ → kompresja zgubiła informację, obniż λ.

---

## 9) Minimalny przepis wdrożeniowy

1. **Ustawienia startowe**: `edge_thr=0.55`, `λ=0.0..0.25`, `Δ=0.25..0.5`, `κ_ab=0.35`.

2. **Selektor**: `Φ2 (balanced)` + soft-labels (`τ≈0.08`).

3. **Polityka tagów**:

   * oddziel I/O od ROI:

     ```python
     # @meta forbid:roi for:Call,Expr
     # @meta prefer:edges for:Call,Expr weight=0.7
     ```
   * DI/Fabryka:

     ```python
     # @meta prefer:roi for:FunctionDef,Return
     # @meta enforce:factory for:Database,Client
     ```
   * wstrzymanie generacji punktowe:

     ```python
     # @meta stop:generate for:Call in:edges when:Sel>0.85
     ```

4. **Reprodukcja**: odpal `vis_ast_kites_all.py` i `hybrid_ast_mosaic.py run/sweep`, porównaj `Align`, `J_phi`, `J_meta`.

---

## 10) Co realnie „dowozi” system

* **Relatywność** (kontekstowa) — to **mozaika** (dane/scena) dyktuje, które węzły „naciągać”.
* **Miarodajność** — każde przesunięcie ma liczby (meta, centroidy, `D_M`, `Align`, `J_meta`).
* **Sterowalność** — meta-tagi i parametry (`λ,Δ,κ_ab,edge_thr`) pozwalają przewidywalnie kształtować wynik.
* **Architektura** — regułami wymusisz wzorce (*DI*, *Factory*) i bezpieczeństwo (*wstrzymanie*).
* **Łańcuchowość** — najważniejsze są **kolejne kroki** (Φ→Ψ→α/β), nie liczba warstw.

---

### Załącznik: mini-checklista PR/CR

* [ ] **Viz**: czy latawce kluczowych węzłów stoją w oczekiwanych regionach (kolor Φ)?
* [ ] **Align** ≥ próg polityki (np. 0.72)
* [ ] **J_phi** (balanced) ↓ vs poprzedni commit
* [ ] **J_meta**: tagi spełnione / akceptowalne odstępstwa
* [ ] **CR_TO** w normie (higiena progu)
* [ ] Wymuszenia (DI/Factory/stop) zastosowane tam, gdzie reguły tego wymagają

---
