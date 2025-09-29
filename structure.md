# AST ⇄ Mozaika — „latawce”/płaszczyzny polityk (opis obrazu)
![Model – schemat](glitchlab/resources/img/model.png)
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
     (Wariant „balanced” stabilizuje wybór przy niejednoznacznym rozkładzie.) 

4. **„Latawiec” O–A–R**
   Dla wybranego regionu liczymy **centroid** `(cx,cy)` i średnie `edge`. Trójkąt **O–A–R** jest następnie wypełniony jako półprzezroczysta płaszczyzna – to **modalna polityka** węzła: które kafelki (Φ), w jakim kontekście (meta) i z jaką „materią” (edge) mają dominować. 

---

## Jak czytać jeden „latawiec”

Dla węzła `Call(print)` (wysoka **Sel**):

* **A** wysoko na osi Z (duża selektywność), X≈głębszy poziom, Y≈id węzła.
* **R** z regionu `edges` (czerwony).
* Płaszczyzna O–A–R „ciągnie” w stronę kafelków o wysokim edge: **side-effecty/I-O** powinny żyć na „krawędziach” systemu, a nie w rdzeniu ROI.

Dla `Assign(y=...)` (wysoka **Stab**):

* **A** wysoko w **Stab**, region `~edges` (niebieski).
* Płaszczyzna preferuje **utrwalenie stanu** w stabilnym tle (poza krawędziami).

Dla `FunctionDef/Return` (wyższa **Cau**):

* **R = roi** (fiolet).
* Szkielet przyczynowy powinien być **kotwiczony** w ROI, z ograniczeniem efektów ubocznych.

Dla `If` (większa **H**):

* **R = all** (pomarańczowy).
* Decyzja ma **globalne rozproszenie skutków** – plane obejmuje cały kontekst.


---

## Jak **ten obraz** powstał (skrót algorytmu)

1. `ast = ast_deltas(EXAMPLE_SRC)` → AST + meta.
2. `M = build_mosaic_hex(rows, cols, seed, R)` → hex-mozaika + edge/roi.
3. Dla każdego węzła:

   * `kind = Φ(label, M, thr)` (selector **balanced**),
   * `R = centroid(region_ids(M, kind, thr))`,
   * `A = (depth, id, meta_dominant)`,
   * rysujemy płaszczyznę **O–A–R** w kolorze regionu.
4. **Miękkie etykiety** (sigmoida z `tau=0.08`) są używane w metryce kafelków, ale do **wizualizacji** używamy regionów binarnych zgodnych z legendą.


---

## Jak oceniać (miary, które można dopisać w podpisie)

* **Koszt Φ** (per węzeł i średni):
  `J_φ = D_M(ids_region, ids_kontr, M, thr)`
  – im **niżej**, tym lepiej (węzeł trafił w „swój” region). `D_M` to symetryczny koszt dopasowania zbiorów kafelków (Earth-Mover-lite) z karą długości i miękkimi etykietami. 

* **Align(AST↔M)** (globalnie):
  `Align = 1 − min(1, wS·|α−aM| + wH·|β−bM| + wZ·|Z/maxZ − 0|)`
  – zbieżność udziałów struktury (α=S/(S+H)) i „materii” (β=H/(S+H)) do profilu mozaiki `(aM,bM)`. **Wyższy = lepszy**. 

* **Sprzężenie Ψ→(α,β)**:
  po projekcji Φ aktualizujemy meta i lekko **blendujemy** (α,β) z profilem mozaiki, ważone `kappa_ab` i niepewnością edge (`std(edge)`). To zmienia „nastawienie” generatora (więcej stabilizacji vs więcej akcji). 

* **Profil mozaiki** (dla opisu obok obrazka):
  `edge_thr = 0.55`, `p(edge) = mean(edge > thr)`, `mean(edge|region)` i centroidy `C_edges / C_~edges / C_roi / C_all`. (Te liczby pochodzą bezpośrednio z mozaiki użytej do renderu.) 

---

## Dlaczego to jest **modalne** i „miarodajne”

* **Modalne** – każdy latawiec odpowiada **trybowi działania** węzła (dominancie meta: Sel/Stab/Cau/H) **w danym kontekście** mozaiki (edges/~edges/roi/all). Zmiana trybu/meta lub regionu **natychmiast** zmienia geometrię płaszczyzny (O–A–R), co widać na wykresie.
* **Miarodajne** – wszystkie decyzje mają **liczbowe** uzasadnienie (dominanty meta, centroidy regionów, `D_M`, `Align`, udział edge powyżej progu). Tymi miarami sterujesz **generacją** (Ψ podbija/obniża komponenty meta, selektor Φ może się przełączyć, a w efekcie generowany AST zmienia kształt). 

---

## „o–ro” i sterowanie generacją

* **O** to stała kotwica systemowa (punkt odniesienia).
* **R/O (ro)** to **relacja** między regionem mozaiki a originem – geometryczna „dźwignia”: im dalej i wyżej `R` (większe `edge`/przeciążenie regionu), tym silniejszy wpływ regionu na aktualizację meta (Ψ) i na dobór kolejnych kroków AST.
* **Przesunięcia „o–ro”** (zmiana rozkładu edge/ROI) powodują, że **te same** węzły AST generują **inne** konfiguracje (inne latawce), bo Φ wybiera inne regiony, a Ψ inaczej koryguje meta.

---


> 3D-wizualizacja pokazuje węzły AST (zielone) oraz ich **polityki** jako płaszczyzny O–A–R (kolory regionów mozaiki), co pozwala **modalnie** sterować generacją kodu: Φ wybiera kontekst kafelków, Ψ koryguje metę, a miary `D_M` i `Align` mówią, czy plan zbiega do danych. 

