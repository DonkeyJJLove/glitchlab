# GLX :: **Matematyka i Formalizm Mozaiki** — v5

*(Φ/Ψ, Δ-rachunek S/H/Z, faktoryzacja relacji, inwarianty I1–I4 i semantyka BUS)*

> Ten dokument jest **formalizacją** modelu Mozaiki GLX. Materiał łączy: grafy/relacje,
> proste kategorie, algebrę półpierścieniową delt, rachunek wkładek (*insertion calculus*)
> oraz sprzężenie **AST ⇄ Mozaika** za pomocą funktorów **Φ/Ψ**. Dokument jest spójny
> z założeniami EGDB, walidatorów BUS (I1–I4) i planera refaktoryzacji.

---

## 0) Słownik i konwencje

* **AST** — drzewo składniowe kodu (deterministyczne).
* **Mozaika** — meta-graf nad AST (centroidy, mosty domen, bucket’y abstrakcji).
* **S/H/Z** — trzy składowe zliczające: **S** (strukturalne), **H** (semantyczne), **Z** (poziom).
* **Δ** — wektor kroku, $\Delta=(\Delta S,\ \Delta H,\ \Delta Z)$.
* **α, β** — udziały: $\alpha=S/(S+H)$, $\beta=H/(S+H)$ (jeśli $S+H>0$).
* **Φ** — projekcja AST→Mozaika; **Ψ** — podnoszenie Mozaika→AST.
* **EGDB** — Event Grammar DB: ujednolicone ślady statyczne i runtime.
* **BUS** — szyna zdarzeń, kontrakty topiców i bramki I1–I4 (*fail-closed*).
* **EGQL** — DSL zapytań po gramatyce zdarzeń (ścieżki, wzorce, warunki Δ).

---

## 1) Przestrzeń Mozaiki jako graf typowany

Niech $\mathcal{G}=(V,E,t)$ będzie **grafem typowanym**, gdzie:

* $V=V_{\text{AST}} \cup V_{\text{MZ}}$ (węzły AST i węzły Mozaiki/centroidy),
* $E = E_S \cup E_H \cup E_Z$ (krawędzie strukturalne, semantyczne oraz „skoki poziomu”),
* $t: E \to {S,H,Z}$ typuje krawędzie.

**Interpretacje:**

* $E_S$: relacje zawierania/kolejności (rodzic→dziecko, moduł→definicje itp.).
* $E_H$: *def→use*, *import*, kontrakty schema/API, mosty między centroidami.
* $E_Z$: dyskretne przejścia poziomów (*bucket(n) → bucket(n±1)*).

> AST to szczególny przypadek: $V=V_{\text{AST}},\ E=E_S \cup E_H$, a **Z** wynika z głębokości bloku.

---

## 2) Δ-rachunek (półpierścień kroków) i kumulacja

Definiujemy **półpierścień delt** $\mathbb{D}=(D,\oplus,\otimes,\mathbf{0},\mathbf{1})$, gdzie:

* $D=\mathbb{Z}*{\ge 0}\times \mathbb{Z}*{\ge 0}\times \mathbb{Z}$, elementy to $\Delta=(\Delta S,\Delta H,\Delta Z)$,
* $\oplus$: suma punktowa, $(a,b,c)\oplus(a',b',c')=(a+a',,b+b',,c+c')$,
* $\otimes$: kompozycja kroków na ścieżce (asocjatywna, z neutralnym $\mathbf{1}=(0,0,0)$),
* $\mathbf{0}=(0,0,0)$.

**Kumulacja:** dla zdarzeń $e_1,\dots,e_k$ na ścieżce $\pi$ mamy
$$
\Delta(\pi) = \bigoplus_{i=1}^k \Delta(e_i).
$$

**Inwariant głębokości (I1):** dla każdej ścieżki $\pi$ w AST
$$
\sum_{e\in \pi \cap E_Z} \Delta Z(e) = Z_{\text{koniec}} - Z_{\text{start}}.
$$
W Mozaice $\Delta Z$ jest legalne **wyłącznie** przez `bucket_jump` lub zmianę współczynnika abstrakcji $\lambda$.

---

## 3) Rachunek wkładek (*insertion calculus*) — faktoryzacja relacji

Własność „**obiekt włożony między 1,a,2 zajmuje miejsce 2-ki**” formalizujemy operatorem **wkładki**
$\blacktriangleright$ na łańcuchach częściowego porządku $(X,\preceq)$.

Niech $x_1 \prec a \prec x_2$ oraz $o$ — nowy obiekt. Definiujemy:
$$
\text{insert}(o,|,a!\prec!x_2):=
\begin{cases}
a \prec o \prec x_2,\
x_2 \mapsto x_3\ \text{(następnik)},
\end{cases}
\quad\Rightarrow\quad
\Delta S \gets \Delta S + 1,\quad
\Delta H\ \text{wg typów połączeń z } a/x_2.
$$

W skrócie: wkładka **nie zrywa** relacji, lecz **refaktoryzuje** lokalny porządek, przesuwając „odcinek”.
Semantyka wkładek działa identycznie w AST (np. wstawienie bloku) i Mozaice (np. nowy centroid).

**Faktoryzacja relacji:** dowolną relację L2 (np. wywołanie funkcji) rozkładamy na relacje prostsze L1
(np. definicja symbolu, import, użycie, kontrakt), po czym wkładamy nowy obiekt („adapter”, „guard”),
tak, by otrzymać minimalne $\Delta$ i **niezerwanie** spójności mostów H.

---

## 4) Φ/Ψ jako funktory (AST ⇄ Mozaika) i transformacje naturalne

Traktujemy AST i Mozaikę jako kategorie $\mathcal{A}$ i $\mathcal{M}$, gdzie obiektami są węzły,
a morfizmami — krawędzie S/H/Z. Definiujemy funktory:

* **Φ:** $\mathcal{A}\to\mathcal{M}$ — projekcja węzłów AST do centroidów i mostów,
* **Ψ:** $\mathcal{M}\to\mathcal{A}$ — podnoszenie planu/mostów do kodu (generacja, adaptery).

**Naturalność (zarys):** dla morfizmu $f:a\to b$ w $\mathcal{A}$ istnieje $\eta$ taka, że
$$
\Phi(f) \circ \eta_a = \eta_b \circ \Psi(\Phi(f)).
$$
W praktyce: transformacje „shadow/plan” są spójne — brak „teleportacji” semantyki między warstwami.

---

## 5) Metryki i pseudo-odległość

* $\alpha=\dfrac{S}{S+H}$, $\ \beta=\dfrac{H}{S+H}$; przy $S+H=0$ odkładamy $(\alpha,\beta)=(0,0)$.
* **Głębokość:** $Z$ oraz poziom *bucket(n)* (skwantowany).
* **Dystans** (pseudo-metryka) dla jednostek $X,Y$:
  $$
  d(X,Y)= w_S,\lvert\alpha_X-\alpha_Y\rvert
  + w_H,\lvert\beta_X-\beta_Y\rvert
  + w_Z,\lvert Z_X - Z_Y\rvert,
  $$
  gdzie $w_S,w_H,w_Z\ge 0$ (GUI/HUD). Dystans służy do klastrów (rdzeń/UI/integracje/eksperymenty).

---

## 6) Reguły Δ dla gramatyki zdarzeń

Zdarzenia gramatyczne $g$ odwzorowujemy w $\Delta(g)$ wg tabeli:

| Zdarzenie                |           AST Δ |       Mozaika Δ | Uwaga                                           |
| ------------------------ | --------------: | --------------: | ----------------------------------------------- |
| `enter_scope(name)`      |    $(+1,+1,+1)$ |      $(+1,0,0)$ | nowy węzeł, definicja symbolu, wejście w poziom |
| `exit_scope()`           |      $(0,0,-1)$ |       $(0,0,0)$ | wyjście z bloku (AST)                           |
| `define(symbol)`         |      $(0,+1,0)$ |      $(0,+1,0)$ | definicja/kontrakt                              |
| `use(symbol)`            |      $(0,+1,0)$ |      $(0,+1,0)$ | użycie/ref                                      |
| `link(A,B)`              |      $(0,+1,0)$ |      $(0,+1,0)$ | most między grupami                             |
| `bucket_jump(b\to b')`   |    $(0,0,b'-b)$ |    $(0,0,b'-b)$ | zmiana poziomu                                  |
| `reassign(node,K\to K')` |     $(+1,+h,0)$ |     $(+1,+h,0)$ | migracja do innego centroidu                    |
| `contract(K,k)`          | $( -(k-1),0,0)$ | $( -(k-1),0,0)$ | scalenie $k$ elementów                          |
| `expand(K,k)`            | $( +(k-1),0,0)$ | $( +(k-1),0,0)$ | rozbicie                                        |

**Monotoniczność celu (I4):** plan refaktoryzacji jest akceptowalny, gdy globalna funkcja celu **nie rośnie**:
$$
\mathcal{J} = \lambda_H \sum \beta ;+; \lambda_Z \sum \lvert\Delta Z\rvert ;+; \lambda_S \sum \text{rozproszenie}(S).
$$

---

## 7) Inwarianty I1–I4 (logika odrzuceń w BUS)

* **I1 (spójność typów/nośników):** krawędzie H muszą łączyć kompatybilne symbole/kontrakty.
* **I2 (spójność warstw/kontraktów):** każde `run.start` musi domknąć się `run.done|error` w oknie $T$.
* **I3 (lokalność zmian):** Δ wprowadzane przez `reassign/contract/expand` nie mogą przeciekać poza zadeklarowany **scope**.
* **I4 (monotoniczność celu):** $\mathcal{J}*{\text{po}} \le \mathcal{J}*{\text{przed}}$.

**Walidatory BUS (*fail-closed*):** wiadomość $m$ jest akceptowana *iff* $\bigwedge_{i=1}^4 I_i(m)=\mathtt{true}$.
W przeciwnym wypadku generowany jest `fail-closed` z dowodem $\pi$ (ścieżka/kontrakt/Δ).

---

## 8) EGDB i semantyka EGQL (czas + struktura + Δ)

**Model:** graf zdarzeń $\mathcal{E}$ z węzłami (topic, plik, węzeł AST, centroid) i łączami:
czasowymi, semantycznymi oraz Δ-projekcją. EGQL to wzorce ścieżek z warunkami na $(\alpha,\beta,Z)$.

**Przykłady:**

* *Sekwencja* `run.start >> run.done` w czasie $\le T$: znajduje ścieżki i sprawdza okno czasowe.
* *Producenci `image.result` o $\beta>\beta^*$:* wybiera pliki/centroidy z dominującym H.
* *Naruszenia I2:* brak domknięcia ścieżki wyzwala *fallback-plan*.

---

## 9) BUS jako algebra procesów (monada błędu)

BUS implementuje kompozycję efektów z monadą błędu $\mathsf{E}$.
Kompozycja zdarzeń `m >>= f` propaguje `fail-closed`. **Fallback** to morfizm
$\kappa: \mathsf{E}X \to \mathsf{E}X$ wybierający plan naprawczy na podstawie $\Delta$
i ścieżki $\pi$ naruszenia — preferując minimalne $\Delta H$ i lokalne $\Delta S$.

---

## 10) Procedura optymalizacji planu (Δ-programowanie)

Celem jest minimalizacja **niedopasowania** $d_\Phi$ między AST a Mozaiką przy ograniczeniach I1–I4:
$$
\min_{\text{akcje}} \ d_\Phi(\text{AST},\text{MZ})
\quad \text{s.t.}\quad I_1\wedge I_2\wedge I_3\wedge I_4.
$$
Heurystyka: *hotspots-first*, *temporal-coupling*, *β-dominant bridges*, *λ-schedule* (kontrakcja→ekspansja).

---

## 11) Minimalny dowód spójności ΔZ

Dla dowolnej ścieżki w AST wejście/wyjście z bloku jest parzyste: każdemu `enter` odpowiada `exit`.
Stąd suma $\Delta Z$ po ścieżce kończy się różnicą głębokości.
W Mozaice $\Delta Z$ pojawia się **tylko** w `bucket_jump` lub przez zmianę $\lambda$ (reguły sterujące).
Walidator I1 odrzuca inne źródła $\Delta Z$.

---

## 12) Mapa implementacyjna

* **Parser #glx-tagów** → `grammar_events` (+ projekcja Δ).
* **Runtime indexer BUS** → `runtime_events` (+ projekcja Δ).
* **Widoki EGDB** → $\alpha/\beta/\delta/\varepsilon$ per plik/kafelek/temat; naruszenia I1–I4.
* **Planer** → minimalny $d_\Phi$, priorytet = *β-dominance × coupling*.
* **GUI/HUD** → kontrola $(w_S,w_H,w_Z)$, $\lambda$, *bucket(n)*, tryb *Delta-only*.

---

## 13) Wnioski i zastosowania

Model Δ-rachunku i wkładek pozwala **refaktoryzować przez faktoryzację relacji** — bez zrywania mostów.
Funktory Φ/Ψ zapewniają spójność między kodem a meta-grafem. Wspólne metryki umożliwiają obiektywną
ocenę postępu, a walidatory BUS gwarantują bezpieczeństwo i *fail-closed* z dowodem.

> **Status:** implementacja eksperymentalna; formalizm **stabilny** (rachunek Δ, I1–I4).
> Parametryzacja GUI $(w_S,w_H,w_Z,\ \lambda)$ i polityki EGQL będą strojenie-zależne.
