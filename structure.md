# AST ⇄ Mozaika — „latawce”/płaszczyzny polityk (opis obrazu)

![Model – schemat](resources/img/model.png)

## Co widać

* **Zielone punkty** – węzły AST (`AstNode`). Każdy punkt ma etykietę (np. `If`, `Assign`, `Call`, `Return`).
* **Półprzezroczyste płaszczyzny** („**latawce**”) – dla **każdego** węzła rysowana jest płaszczyzna polityki O–A–R:

  * **O** (origin) = `(0,0,0)` – wspólna kotwica.
  * **A(node)** = `(depth, id, dominanta_meta)` – pozycja węzła w przestrzeni AST×meta.
  * **R(region)** = `(cx, cy, mean_edge_region)` – centroid i średni „edge” wybranego regionu mozaiki.
* **Kolory płaszczyzn** = **region Φ** wskazany przez selektor „balanced”:

  * **czerwony** – `edges` (kafelki o wysokim edge),
  * **niebieski** – `~edges` (stabilne obszary),
  * **fioletowy** – `roi` (obszar istotny),
  * **pomarańczowy** – `all` (wpływ globalny).
* **Oś X** – głębokość węzła w AST (`depth`).
* **Oś Y** – identyfikator / indeks węzła (porządek przejścia).
* **Oś Z** – **dominanta meta** w węźle lub „edge” regionu (skalowane do `[0,1]`).

> Intuicja: każda płaszczyzna to „ster” polityki dla jednego węzła – **jak ma działać** (jaki region mozaiki go „ciągnie”) oraz **w jakim aspekcie meta** (Sel/Stab/Cau/H) powinien przeważać wpływ.

---

## Skąd biorą się punkty i płaszczyzny

1. **AST + meta**
   Z kodu przykładowego powstaje AST z meta-wektorami `[L,S,Sel,Stab,Cau,H]` liczonymi deterministycznie ze struktury (głębokość, rozgałęzienie, wielkość poddrzewa, różnorodność). Dla każdego węzła wybieramy **dominantę** (najistotniejszy komponent meta), którą rysujemy na osi Z.

2. **Mozaika heksagonalna**
   Budowana jako siatka hex (centra, promień `R=1.0`), z polem **edge** (gęstość krawędzi) i **ROI** (środek kadru). Użyty jest **próg** `edge_thr` (domyślnie `0.55`).

3. **Selektor Φ (balanced)**
   Dobór regionu **per węzeł** na podstawie etykiety i kwartyli rozkładu edge (`Q25/Q75`):

   * `Call/Expr` → **edges**,
   * `Assign` → **~edges**,
   * `FunctionDef/Return` → **roi**,
   * `If/While/For` → **all**.
     Wariant „balanced” stabilizuje wybór przy niejednoznacznym rozkładzie.

4. **„Latawiec” O–A–R**
   Dla wybranego regionu liczymy **centroid** `(cx,cy)` i średnie `edge`. Trójkąt **O–A–R** jest następnie wypełniony jako półprzezroczysta płaszczyzna – to **modalna polityka** węzła: które kafelki (Φ), w jakim kontekście (meta) i z jaką „materią” (edge) mają dominować.

---

## Meta-tagi sterujące (pozytywne/negatywne)

> Meta-tag to *deklaracja preferencji* dla komponentów meta `[L,S,Sel,Stab,Cau,H]`, opcjonalnie z parametrami regionu Φ i siłą wpływu. Tag działa jako **priorytet** (miękki) lub **ograniczenie** (twarde) w pętli Φ/Ψ.

### Składnia (propozycja praktyczna)

W komentarzach nad węzłem/funkcją lub w profilu projektu:

```python
# @meta +Sel>=0.75 in:edges    # pozytywny – promuj selektywność w regionie edges
# @meta -Stab<0.40 in:edges    # negatywny – unikaj niskiej stabilności w edges
# @meta prefer:roi weight=0.6  # priorytet wyboru ROI dla tego węzła
# @meta forbid:edges           # twardy zakaz przypisania do edges
# @meta goal Align>=0.85       # cel globalny dla zbieżności AST↔Mozaika
```

**Polaryzacja:**

* **Pozytywny (`+`)** – „ciągnij w górę” wskazaną metrykę/meta; Ψ podbija komponent meta, Φ zwiększa szansę regionu.
* **Negatywny (`-`)** – „tłum w dół” (demotywuj); Ψ obniża komponent meta; Φ redukuje prawdopodobieństwo regionu.

**Modalność i zakres:**

* **scope=node** (domyślnie) – działa na pojedynczy węzeł (etykieta/linia kodu).
* **scope=region** – działa na wszystkie węzły trafiące w dany Φ-region.
* **scope=global** – zmienia nastawy sprzężenia `(α,β)` i wag w `Align`.

### Jak tag zamienia się w działanie algorytmu

* **Na etapie Φ (projekcja):**
  tag może:

  * **przeważać selektor** (np. `prefer:roi`, `forbid:edges`),
  * **zmieniać próg** skuteczności regionu (niższy koszt Φ, jeśli zgodność z tagiem).

* **Na etapie Ψ (feedback):**
  tag mapujemy na **wektor dψ** dodany do aktualizacji meta:

  ```
  meta_new = (1−δ)·meta_old + δ·ψ(M,region) + λ_tag·dψ(tag)
  ```

  gdzie `λ_tag` to siła tagu (np. `weight`), a `dψ(+Sel>=0.75)` zwiększa komponent `Sel` i „ciągnie” A(node) w górę osi Z.

* **Na poziomie globalnym:**
  tagi typu `goal Align>=0.85` lub preferencje regionów wpływają na **kappa_ab** (mieszanie `(α,β)` z profilem mozaiki) oraz na wagi `W_DEFAULT` w mierze `Align`.

### Przykładowe zastosowania

* **„Oddziel I/O od ROI”**

  ```
  # @meta forbid:roi for:Call,Expr
  # @meta prefer:edges for:Call,Expr weight=0.7
  ```

  Efekt: wywołania (I/O) będą wypychane na krawędź (edges), nawet jeśli semantyka kodu skłaniałaby do ROI.

* **„Utrwal stan w tle”**

  ```
  # @meta +Stab>=0.8 in:~edges for:Assign
  ```

  Efekt: przypisania dostają Ψ-boost stabilności i Φ woli ~edges.

* **„Eksponuj przyczynowość w ROI”**

  ```
  # @meta +Cau>=0.7 in:roi for:Return,FunctionDef
  ```

### Ocena spełnienia tagów (zgodność)

Dla każdego tagu liczony jest **score spełnienia** i **kara naruszenia**:

```
score_tag = clamp( target − observed, … )     # zależnie od kierunku +/−
J_meta = Σ (penalty(tag) · (1 − score_tag))
```

`J_meta` można:

* dopisać do podpisu obrazu,
* dodać do **średniego kosztu Φ**,
* użyć jako **constraint** w pętli generacyjnej (wybierz wariant z minimalnym `J_meta`).

---

## Jak czytać jeden „latawiec”

Dla węzła `Call(print)` (wysoka **Sel**):

* **A** wysoko na osi Z (duża selektywność), X≈głębszy poziom, Y≈id węzła.
* **R** z regionu `edges` (czerwony).
* Płaszczyzna O–A–R „ciągnie” w stronę kafelków o wysokim edge: **side-effecty/I-O** powinny żyć na „krawędziach” systemu, a nie w rdzeniu ROI.
* **Tagowe sterowanie:**
  `@meta prefer:edges for:Call` – zwiększa szansę wyboru edges;
  `@meta -Sel<0.6 in:roi` – zniechęca do I/O w ROI (demotywuje Sel w ROI).

Dla `Assign(y=...)` (wysoka **Stab**):

* **A** wysoko w **Stab**, region `~edges` (niebieski).
* Płaszczyzna preferuje **utrwalenie stanu** w stabilnym tle (poza krawędziami).
* **Tagowe sterowanie:**
  `@meta +Stab>=0.8 in:~edges for:Assign` – dψ podnosi Stab i utrwala wybór ~edges.

Dla `FunctionDef/Return` (wyższa **Cau**):

* **R = roi** (fiolet). Szkielet przyczynowy kotwiczymy w ROI.
* **Tagowe sterowanie:**
  `@meta +Cau>=0.7 in:roi for:Return,FunctionDef`.

Dla `If` (większa **H**):

* **R = all** (pomarańczowy). Decyzja ma globalne skutki.
* **Tagowe sterowanie:**
  `@meta +H>=0.7 for:If` – wzmacnia dyfuzję wpływu;
  `@meta forbid:edges for:If` – jeśli chcesz unikać zbytniej „nerwowości” na krawędziach.

---

## Jak **ten obraz** powstał (skrót algorytmu)

1. `ast = ast_deltas(EXAMPLE_SRC)` → AST + meta.
2. `M = build_mosaic_hex(rows, cols, seed, R)` → hex-mozaika + edge/roi.
3. Dla każdego węzła:

   * `kind = Φ(label, M, thr)` (selector **balanced**),
   * `R = centroid(region_ids(M, kind, thr))`,
   * `A = (depth, id, meta_dominant)`,
   * rysujemy płaszczyznę **O–A–R** w kolorze regionu.
4. **Miękkie etykiety** (sigmoida `τ=0.08`) służą w metryce kafelków; do wizualizacji używamy regionów binarnych z legendy.
5. **Meta-tagi** (jeśli obecne) modulują:

   * selektor Φ (prefer/forbid, zmiana priorytetu),
   * Ψ-update (dodanie `dψ(tag)`),
   * wagi globalne `(α,β)` i `W_DEFAULT`.

---

## Jak oceniać (miary, które można dopisać w podpisie)

* **Koszt Φ** (per węzeł i średni):
  `J_φ = D_M(ids_region, ids_kontr, M, thr)` – im **niżej**, tym lepiej (węzeł trafił w „swój” region).

* **Zgodność tagów**:
  `J_meta = Σ penalty(tag)·(1 − score_tag)` – naruszenia tagów podnoszą koszt; spełnienia go obniżają.

* **Align(AST↔M)** (globalnie):
  `Align = 1 − min(1, wS·|α−aM| + wH·|β−bM| + wZ·|Z/maxZ − 0|)` – zbieżność udziałów struktury (α) i „materii” (β) do profilu mozaiki `(aM,bM)`.

* **Sprzężenie Ψ→(α,β)**:
  po projekcji Φ aktualizujemy meta i lekko **blendujemy** `(α,β)` z profilem mozaiki, ważone `kappa_ab` i niepewnością `std(edge)`.

* **Profil mozaiki** (dla podpisu):
  `edge_thr = 0.55`, `p(edge) = mean(edge > thr)`, `mean(edge|region)` oraz centroidy `C_edges / C_~edges / C_roi / C_all`.

---

## Dlaczego to jest **modalne** i „miarodajne”

* **Modalne** – każdy latawiec odpowiada **trybowi działania** węzła (dominancie meta: Sel/Stab/Cau/H) **w danym kontekście** mozaiki (edges/~edges/roi/all). Meta-tagi zmieniają *tryb*: przesuwają A(node), zmieniają preferencje Φ i wzmacniają/hamują Ψ.
* **Miarodajne** – wszystkie decyzje mają **liczbowe** uzasadnienie (meta, centroidy, `D_M`, `Align`, `J_meta`, udział edge powyżej progu). Tymi miarami sterujesz **generacją** i możesz porównywać warianty.

---

## „0–ro” i sterowanie generacją

* **O** to stała kotwica systemowa (punkt odniesienia).
* **R/O (ro)** to **relacja** między regionem mozaiki a originem – geometryczna „dźwignia”: im dalej i wyżej `R` (większe `edge`/przeciążenie regionu), tym silniejszy wpływ regionu na aktualizację meta (Ψ) i na dobór kolejnych kroków AST.
* **Meta-tagi a o–ro:** pozytywne tagi „wydłużają dźwignię” (większe `λ_tag`), negatywne – „skracają” lub blokują połączenie O→R dla wybranych etykiet/regionów.
* **Przesunięcia „o–ro”** (zmiana rozkładu edge/ROI) powodują, że **te same** węzły AST generują **inne** konfiguracje (inne latawce), bo Φ wybiera inne regiony, Ψ inaczej koryguje meta, a tagi modulują oba efekty.

---

> **Alt-text / TL;DR:** 3D-wizualizacja pokazuje węzły AST (zielone) oraz ich **polityki** jako płaszczyzny O–A–R (kolory regionów mozaiki). Meta-tagi dodatnie/ujemne modulują Φ (wybór kontekstu) i Ψ (aktualizację meta), a miary `D_M`, `Align` i `J_meta` pozwalają mierzyć, czy generacja zbiega do założonej ontologii.
