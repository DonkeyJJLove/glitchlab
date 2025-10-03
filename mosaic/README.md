# GlitchLab · AST ⇄ Mozaika (Φ/Ψ)

**Protokół kontekstu i generacji kodu + wizualizacja „latawców”**

> Ten README zastępuje poprzednie „PRZEDAWNIONE README.MD”.
> Spina algorytm `hybrid_ast_mosaic.py`, wizualizacje `vis_ast_kites_all.py` i praktyki sterowania generacją kodu.
> Nacisk: **działanie w łańcuchu** (lekka telemetria per-node), **modalność** płaszczyzn/latawców i **miary**. 

>[Struktura](structure.md) | [Case Study](cstudy.md)
---

## Spis treści

* [1. O co chodzi (w 3 zdaniach)](#1-o-co-chodzi-w-3-zdaniach)
* [2. Model relacji (najważniejsza różnica vs. stary opis)](#2-model-relacji-najważniejsza-różnica-vs-stary-opis)
* [3. Artefakty i dane](#3-artefakty-i-dane)
* [4. Parametry i „cięgna” sterujące](#4-parametry-i-cięgna-sterujące)
* [5. Przepływ Φ/Ψ (łańcuch)](#5-przepływ-φψ-łańcuch)
* [6. Latawce (modalne płaszczyzny polityk)](#6-latawce-modalne-płaszczyzny-polityk)
* [7. Miary i ocena](#7-miary-i-ocena)
* [8. Pozytywne/negatywne meta-tagi (kontrola generacji)](#8-pozytywnenegatywne-meta-tagi-kontrola-generacji)
* [9. Przepisy (recipes): wstrzymanie i wymuszenie wzorca](#9-przepisy-recipes-wstrzymanie-i-wymuszenie-wzorca)
* [10. Uruchomienia i CLI](#10-uruchomienia-i-cli)
* [11. Najczęstsze pytania](#12-najczęstsze-pytania)

---

## 1. O co chodzi (w 3 zdaniach)

1. Z kodu powstaje **AST**; każdym węzłem steruje meta-wektor `[L,S,Sel,Stab,Cau,H]`.
2. **Mozaika** (grid/hex) daje tło kontekstowe: pola **edge** (żywotność/krawędzie) i **roi** (rdzeń).
3. **Φ (phi)** dobiera region mozaiki dla węzła, **Ψ (psi)** koryguje meta zgodnie z regionem; globalne **α/β** balansu-struktura/materia dopasowujemy do profilu mozaiki.
   Wynik: można **mierzyć** i **sterować** generacją/transformacją kodu.

---

## 2. Model relacji (najważniejsza różnica vs. stary opis)

Poprzednio relacja AST↔Mozaika bywała opisywana „warstwowo”. W nowym modelu **relacja jest modalna i per-węzeł**:

* Każdy węzeł ma **dominantę meta** (np. `Sel` dla `Call`, `Stab` dla `Assign` itd.).
* Na tej podstawie tworzymy **latawiec**: trójkąt **O–A–R**

  * **O** — origin `(0,0,0)`
  * **A(node)** — pozycja węzła: `(depth, id, meta[dominanta])`
  * **R(region)** — centroid regionu mozaiki wskazanego przez Φ + średni `edge`
* Latawiec to **płaszczyzna polityki**: lokalny ster mówiący „jak ma działać ten node w danym kontekście mozaiki”.
* **Nie potrzebujemy wielu grubych warstw** – starczy lekka telemetria i łańcuchowe korekty Φ/Ψ.

---

## 3. Artefakty i dane

### AST

* `AstNode.meta = [L, S, Sel, Stab, Cau, H] ∈ [0,1]^6`
* **Dominanta** jest inferowana z etykiety (reguły w kodzie).

### Mozaika

* `edge ∈ [0,1]` (gęstość krawędzi/żywotność), `roi ∈ {0,1}` (centralny obszar).
* Hex lub grid. Próg `edge_thr` wyznacza regiony: `edges`, `~edges`, `roi`, `all`.

### Metaprzestrzeń globalna

* `α = S/(S+H)`, `β = 1−α` (udziały struktura/materia), `Z` (warstwowość).
* W porównaniu do mozaiki: `aM, bM` z jej profilu.

---

## 4. Parametry i „cięgna” sterujące

* **λ** – kompresja AST (redukcja liści i spłaszczenie Z) – zwykle **nisko** (by nie tracić kontekstu).
* **Δ** – siła Ψ-feedback (jak mocno meta ma się dostroić do regionu) – **główne pokrętło**.
* **κ_ab** – sprzężenie `(α,β)` do profilu mozaiki (zależne od niepewności `std(edge)`).
* **TAU** – temperatura miękkich etykiet (stabilność progów).
* **W** – wagi dla Align (istotność składowych w dopasowaniu globalnym).

---

## 5. Przepływ Φ/Ψ (łańcuch)

1. **AST + meta** → `ast_deltas(src)`
2. **Kompresja** (opcjonalnie) → `compress_ast(ast, λ)`
3. **Region Φ** per węzeł → `phi_region_for_balanced` (używa kwartyli edge dla stabilizacji)
4. **Koszt Φ** → `J_φ` z `D_M` (miękkie etykiety + kara długości)
5. **Ψ-feedback** → miękka aktualizacja meta wektorów (`Δ`)
6. **Sprzężenie α/β** → blend z profilem mozaiki (`κ_ab · Δ`)
7. **Decyzja** → kryteria (Pareto: Align↑, J_φ↓, `CR_TO` higiena progu)

To działa **iteracyjnie**; najczęściej wystarcza 1–2 kroki, bo zależy nam na **łańcuchowym, celowym** dociąganiu, nie na „przemieleniu” wszystkiego.

---

## 6. Latawce (modalne płaszczyzny polityk)

Wizualizacja: `vis_ast_kites_all.py` (produkcja).
Kolory regionów: `edges` (czerwony), `~edges` (niebieski), `roi` (fiolet), `all` (pomarańcz).
**Każdy node** ma swoją płaszczyznę **O–A–R**. To:

* **czytelny ster** dla węzła (gdzie i w jakim aspekcie meta ma pracować),
* **miarodajny** — z liczb (meta, centroidy, mean edge),
* **relatywny** — zależny od aktualnej mozaiki (zmienisz rozkład `edge` → zmienią się latawce).

---

## 7. Miary i ocena

* **J_φ** *(↓)* — średni koszt dopasowania regionów (symetryczny Earth-Mover-lite + kara długości + miękkie etykiety).
* **Align(AST↔M)** *(↑)* — zbieżność `(α,β,Z)` z profilem mozaiki `(aM,bM, …)` z wagami `W`.
* **CR_AST** *(↑ w granicach rozsądku)* — kompresja struktury po `λ`.
* **CR_TO** *(higiena progu)* — kara za skrajny rozkład `edge` względem `edge_thr`.
* **Inwarianty**: I1 (`α+β=1`), I2 (własności `D_M`), I3 (monotonia kompresji).

> Praktyka: **Φ=balanced** jako domyślny selektor, **Δ** kręci Align, **λ** trzymać nisko, **CR_TO** pilnować.

---

## 8. Pozytywne/negatywne meta-tagi (kontrola generacji)

Meta-tag = lekki **sygnał polityki** przypięty do węzła/fragmentu, który Ψ interpretuje przy aktualizacji meta i wyborze Φ.
**Pozytywny** — „dodaj, zintensyfikuj”; **Negatywny** — „wytnij/odrzuć/nie rozbudowuj”.

Przykłady (skróty; pod spodem działa zwykłe Φ/Ψ):

* `+STAB` – podbij **Stab**; preferuj `~edges`, generuj/utrwalaj inicjalizacje, wyprowadzaj side-effecty z ROI.
* `-SEL` – zbij **Sel**; ogranicz `Call/Expr` w ROI; przenieś I/O do edges.
* `+CAU` – wzmocnij **Cau**; preferuj `FunctionDef/Return` w ROI (czysty przepływ wartości).
* `-H` – redukuj **H**; minimalizuj heterogeniczność gałęzi (prostsze poddrzewa).

**Jak to działa praktycznie?**
Tag → modyfikuje lokalne `psi` i/lub wybór Φ → zmienia wektor meta po Ψ → **generator** widzi preferencje (np. wstrzymanie, przestawienie regionu, przekształcenie do wzorca).

---

## 9. Przepisy (recipes): wstrzymanie i wymuszenie wzorca

### 9.1. Wstrzymanie generacji

**Warunek:** meta + region dają ryzyko (np. `Sel>0.85` w `edges` przy braku wiedzy).
**Działanie:** Φ oznacza węzeł jako **stub** i blokuje rozwijanie poddrzewa (zachowujesz hak, nie generujesz fikcji).

```python
# zamiast budować zewnętrzny efekt:
result = external_api(x)  # TODO: stub – wymaga realnego systemu
```

### 9.2. Wymuszenie wzorca **Fabryka (Factory)**

**Detekcja:** powtarzalne `Assign(X = Class())` + `Call` instancjonujące w ROI/edges.
**Akcja:** Φ przełącza region do `roi`, Ψ podbija `Cau/Stab` → generator tworzy **Factory**.

```python
class DatabaseFactory:
    @staticmethod
    def create() -> Database:
        return Database()

def get_user():
    db = DatabaseFactory.create()
    return db.query("SELECT * FROM users")
```

Efekt: minimalizacja `Sel` w rdzeniu, centralizacja przyczynowości (`Cau`) i utrwalenia (`Stab`), czystsze ROI.

---

## 10. Uruchomienia i CLI

### Wizualizacja latawców

```bash
python vis_ast_kites_all.py --rows 6 --cols 6 --edge-thr 0.55 --seed 7 --out out.png
```

### Pojedynczy przebieg metryk

```bash
python hybrid_ast_mosaic.py run --rows 6 --cols 6 --edge-thr 0.55 --lmbd 0.25 --delta 0.5
```

### Sweep i Pareto

```bash
python hybrid_ast_mosaic.py sweep --rows 6 --cols 6 --edge-thr 0.55 --json
```

### Testy spójności/inwariantów

```bash
python hybrid_ast_mosaic.py test --rows 6 --cols 6 --edge-thr 0.55 --lmbd 0.6 --runs 100
```

**Domyślne, które działają dobrze:**
Φ=`balanced`, `Δ∈{0.25,0.5}`, `λ≈0…0.25`, `κ_ab=0.35`, `TAU≈0.08`, `W={wS=1,wH=1,wZ=0.4}`.

---

## 11. Najczęstsze pytania

**Czy muszę używać tylu warstw na wizualizacji?**
Nie. To **przegląd**. W praktyce wystarcza lekki podgląd (kilka adnotacji) + miary (Align, J_φ).

**Co realnie steruje generacją?**
Tagi meta (+/−), **Δ** (Ψ-feedback), wybór Φ (region), oraz reguły wzorców (np. Fabryka). To 4 „cięgna”.

**Czy to deterministyczne?**
Mapa meta w trybie `det` jest deterministyczna; mozaika ma seed; miękkie etykiety stabilizują decyzje progu.

**Czemu Align czasem nie rośnie mimo dużego Δ?**
Sprawdź `CR_TO` (higiena progu). Jeśli rozkład `edge` jest skrajny, Φ/Ψ będą „przeciągać linę” bez realnej poprawy.

---

### TL;DR

* **Relacja** jest **per-węzeł, modalna** (płaszczyzny O–A–R).
* **Łańcuch Φ/Ψ** to lekkie, iteracyjne dociąganie meta do kontekstu mozaiki.
* **Miary** (Align, J_φ, CR_TO) pozwalają **mierzyć i sterować** generacją.
* **Meta-tagi** i **przepisy** (wstrzymanie, Fabryka) dają praktyczny uchwyt na ontologię kodu.

---


