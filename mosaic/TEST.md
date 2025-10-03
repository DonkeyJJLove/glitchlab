# Ocena algorytmu na bazie wykonanych testów

## 1) Skuteczność mapowania kontekstu (Φ)

**Co mierzymy:** ΔJ = J\_φ1 − J\_φ2 (im większe > 0, tym lepszy Φ2).
**Twoje wyniki:** wins=70, losses=30 (ties=0), **p≈3.93e−05**, **median ΔJ≈2.87**, **Cliff’s δ≈0.40 (medium)**.

**Wniosek:**

* **Stabilna wygrana Φ2 nad Φ1** z **efektem średnim** i **wysoką istotnością**.
* W praktyce: reguła „balanced” daje **bardziej rozdzielcze regiony kontekstu** w \~70% losowań mozaiki.
* To nie jest „kosmetyka”: **średnia przewaga ≈2.31 punktu** w koszcie segmentacji.

**Porównanie do rodzin podejść:**

* **AST-only** (bez mozaiki): brak mechanizmu Φ ⇒ nie ma jak osiągnąć podobnej poprawy separacji regionów kontekstu; zwykle skończy się na prostych heurystykach z głębokości/typów węzłów.
* **Vision-only / tekstura bez AST:** dostaje Φ, ale bez semantycznego „kotwienia” w węzłach; ΔJ zwykle podatne na fluktuacje progu. Nasz wynik pokazuje, że **hybryda** stabilizuje decyzję (Φ2 wins=70%).
* **Ciężkie EMD/graph-matching**: potencjalnie dokładniejsze dopasowanie niż nasz EMD-lite, ale przy dużym koszcie; **tu** wygrywamy stosunkiem jakość/koszt (patrz §3).

**Ocena (0–5):** **4.5/5** za skuteczność Φ w kontekście danych testowych.

---

## 2) Zgodność AST↔Mozaika (Align) sterowana Ψ (Δ)

**Co mierzymy:** Align↑ pod kontrolą Δ przy „higienie” progu (CR\_TO ≤ 20).
**Twoje wyniki (front Pareto):** najlepszy punkt spełniający ograniczenie:
**λ=0.0, Δ=0.5 → Align≈0.743, J\_φ2≈80.95, CR\_TO≈11.0**.

**Wniosek:**

* **Regulacja kontekstu działa**: Align monotonicznie rośnie wraz z Δ przy λ=0.0 (0.715 → 0.729 → 0.743).
* **Higiena progu zachowana** (CR\_TO≈11 ≪ 20).
* **λ=0** jest korzystne: kompresja AST (λ>0) obniża Align mimo lekkich zysków w CR\_AST — realna sygnatura, że w tej klasie zadań „bogatszy AST + silne Ψ” daje lepszą zgodność.

**Porównanie do rodzin podejść:**

* **AST-only**: Align nie istnieje (brak „drugiej przestrzeni”), więc brak dźwigni poprawy przez Δ.
* **Agentowe LLM-toolchains**: Align zwykle „implicit” (prompt-level), mało mierzalny i zależny od temperatury/modelu; tu mamy **twardą, mierzalną dźwignię** (Δ) i **kontraint** (CR\_TO).
* **Ciężkie metody uczenia**: mogą zoptymalizować Align end-to-end, ale kosztem danych i treningu; nasz algorytm uzyskuje **„plug-in” poprawę bez treningu**.

**Ocena (0–5):** **4.3/5** za przewidywalne, monotoniczne sterowanie zgodnością przy kontroli stabilności.

---

## 3) Złożoność i wydajność (koszt obliczeń vs jakość)

**Co mamy w kodzie:**

* D\_M: dokładne permutacje dla k≤8, **zachłanne O(k²)** powyżej; kara długości zamiast ∞; soft-labels z sigmoidem (stały koszt).
* Statystyka: 100 seedów, 12×12 grid (144 kafle), trzy selektory Φ.

**Wniosek jakości/koszt:**

* **Bardzo dobry kompromis**: EMD-lite + soft-labels dają przewagę Φ2 bez wchodzenia w kosztowne assignmenty/EMD pełne.
* Wszystkie testy (100 seedów) wykonały się bez problemów — wnioskujemy, że **skaluje się praktycznie** w rozdzielczości użytej w benchmarku.

**Porównanie do rodzin:**

* **Pełny EMD/optimal transport**: wyższa dokładność lokalna, ale koszt często **O(n³)**; nasz **O(k²)** w części zachłannej jest znacznie tańszy i wystarczający przy obserwowanej „średniej sile efektu”.
* **Agentowe podejścia**: koszt głównie inferencyjny + latency I/O; nasza ścieżka jest **lokalna i deterministyczna** (seed), zwykle szybsza i powtarzalna.

**Ocena (0–5):** **4.2/5** za **efektywność** przy zachowaniu sensownej jakości dopasowania.

---

## 4) Solidność statystyczna i interpretowalność

* **Istotność:** p≈3.93e−05 (sign test) — **wysoka**.
* **Wielkość efektu:** Cliff’s δ≈0.40 — **średnia** (nie marginalna, nie trywialna).
* **CI bootstrap (mean/median):** wąskie i dodatnie — **stabilność wniosków**.
* **Inwarianty (I1–I3)** w kodzie — zapewniają **spójność arytmetyczną** i właściwości metryczne.

**Porównanie:**

* **Agentowe LLM**: zwykle **trudniej o CI i efekty na poziomie komponentów**; tu mamy czystą, powtarzalną ścieżkę i czytelne interpretacje metryk.

**Ocena (0–5):** **4.7/5** za rzadką w tej klasie podejść **mierzalność i interpretowalność**.

---

## 5) Ograniczenia i ryzyka (uczciwie)

* **Dane syntetyczne mozaiki (proceduralne)**: choć kontrolowane (to plus), to **nie są danymi z produkcji**; konieczna kalibracja thr/τ na realnych mapach cech.
* **Heurystyki Φ**: mimo że Φ2 wygrywa, to dalej **reguły deterministyczne**; w domenach bardzo niestacjonarnych uczenie (learned Φ) może dać dodatkowy zysk.
* **Kompresja λ**: obserwacyjnie obniża Align — świetna informacja diagnostyczna, ale oznacza, że **kompresję trzeba stosować ostrożnie** (raczej do uproszczeń, nie do zysku zgodności).
* **Brak treningu**: zaleta (brak kosztu), ale też **brak adaptacji domenowej** bez ręcznego strojenia progów.

**Ocena (0–5) – „ryzyko domenowe”:** **3.6/5** (akceptowalne, z jasną ścieżką kalibracji).

---

# Podsumowanie porównawcze (scorecard)

| Kryterium                             | Nasz algorytm | AST-only | Vision-only | Pełny EMD/OT | Agentowe LLM toolchains |
| ------------------------------------- | ------------: | -------: | ----------: | -----------: | ----------------------: |
| **Skuteczność Φ (ΔJ, p, δ)**          |     **4.5/5** |      2/5 |         3/5 |        4.7/5 |                   3.5/5 |
| **Sterowalność Align (Δ, CR\_TO)**    |     **4.3/5** |        – |       2.5/5 |        3.5/5 |                     3/5 |
| **Wydajność (jakość/koszt)**          |     **4.2/5** |    4.7/5 |       4.5/5 |        2.5/5 |                     3/5 |
| **Statystyka + interpretowalność**    |     **4.7/5** |      4/5 |       3.5/5 |        3.5/5 |                   2.5/5 |
| **Ryzyko domenowe / kalibracja**      |     **3.6/5** |    3.5/5 |       3.5/5 |          3/5 |                   2.5/5 |
| **Ocena łączna (ważona praktycznie)** |     **4.3/5** |    3.3/5 |       3.4/5 |        3.3/5 |                   2.9/5 |

> W swojej klasie (hybryda AST⇄cechy przestrzenne) to **bardzo dobry algorytm praktyczny**:
> **statystycznie istotny zysk** (Φ2 vs Φ1), **przewidywalna dźwignia zgodności** (Δ), **sensowny koszt obliczeń** i **wysoka interpretowalność**.
> W porównaniu do typowych agentowych łańcuchów narzędzi — **bardziej mierzalny, stabilny i tańszy** w użyciu; w porównaniu do pełnych EMD/OT — **lepszy trade-off** jakość/koszt.


