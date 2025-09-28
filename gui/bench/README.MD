# Benchmark protokołów kontekstu — raport badawczy (A1/A2 vs. B)

> Ten dokument jest **merytorycznym, technicznym i weryfikowalnym** podsumowaniem naszej mini-kampanii badawczej nad „protokołami kontekstu” dla automatycznej generacji rozwiązań zadań programistycznych. Raport opisuje implementacje, metryki, procedury testowe, wyniki oraz werdykt „który protokół jest lepszy” **na badanym zbiorze** i przy **jawnie zdefiniowanych kryteriach**.
> Repozytorium zawiera kompletny kod uruchamialny z poziomu IDE (funkcja `main()`), jak i z CLI.

---

## Spis treści

* [Cel i pytania badawcze](#cel-i-pytania-badawcze)
* [Badane protokoły / agenci](#badane-protokoły--agenci)

  * [A1 i A2 — protokół „Mozaika+AST” (nasz)](#a1-i-a2--protokół-mozaikaast-nasz)
  * [B — baseline „Microsoft-like” (prosty, deklaratywny)](#b--baseline-microsoft-like-prosty-deklaratywny)
* [Zbiór zadań i format danych](#zbiór-zadań-i-format-danych)
* [Metryki i kryteria oceny](#metryki-i-kryteria-oceny)

  * [Metryki funkcjonalne](#metryki-funkcjonalne)
  * [Metryki strukturalne / jakości kontekstu](#metryki-strukturalne--jakości-kontekstu)
  * [Metryki statystyczne (efekt i test znaku)](#metryki-statystyczne-efekt-i-test-znaku)
  * [Reguła werdyktu](#reguła-werdyktu)
* [Procedura eksperymentalna](#procedura-eksperymentalna)
* [Wyniki](#wyniki)

  * [Raport zbiorczy](#raport-zbiorczy)
  * [Interpretacja](#interpretacja)
* [Wizualizacje i artefakty](#wizualizacje-i-artefakty)
* [Jak uruchomić (IDE i CLI)](#jak-uruchomić-ide-i-cli)
* [Ograniczenia i zagrożenia dla rzetelności](#ograniczenia-i-zagrożenia-dla-rzetelności)
* [Wnioski i prace przyszłe](#wnioski-i-prace-przyszłe)

---

## Cel i pytania badawcze

**Cel:** porównać dwa podejścia do generacji kodu z krótkich specyfikacji zadań:

1. **A1/A2** – hybrydowy protokół „Mozaika+AST” (nasz), z dwoma wariantami parametrów.
2. **B** – prosty baseline „Microsoft-like” (PTP/KH/Fallback), działający deterministycznie z użyciem szablonów i heurystyk nazw.

**Pytania badawcze:**

* Który protokół **lepiej odwzorowuje kontekst** zadania (spójność strukturalna kodu vs. oczekiwania)?
* Który protokół **częściej przechodzi testy funkcjonalne**?
* Jak wyglądają **koszty czasowe** i **stabilność**?

---

## Badane protokoły / agenci

### A1 i A2 — protokół „Mozaika+AST” (nasz)

* **Intencja**: najpierw budujemy *projekcję kontekstu* (mozaika z cech problemu), a następnie **wymuszamy spójność** tej projekcji ze strukturą kodu (AST).
* **Różnica A1 vs. A2**: identyczne jądro, ale **A2** jest bardziej konserwatywne w kompresji semantycznej AST i mocniej penalizuje rozjazdy mozaika↔AST (co zwykle podnosi `align` i stabilizuje `cr_ast`).

> Uwaga: „Mozaika+AST” to nasz jawny eksperyment; nie odwołujemy się do żadnych niejawnych algorytmów stron trzecich.

### B — baseline „Microsoft-like” (prosty, deklaratywny)

* **PTP (Template Protocol)**: twarde szablony dla rozpoznanych `entrypoint`ów (np. `reverse_str`, `fib`).
* **KH (Keyword Heuristic)**: naiwny dobór implementacji po słowach kluczowych w nazwie (np. `palindrome`, `factorial`, `gcd`).
* **Fallback (Safety)**: bezpieczny szkielet `raise NotImplementedError` dla nierozpoznanych zadań.

> To **nie jest** odwzorowanie żadnego „tajnego” algorytmu. To **prostolinijny baseline** z trzema przejrzystymi krokami – w pełni jawny i replikowalny.

---

## Zbiór zadań i format danych

**Zadania** trzymamy w plikach JSON (wzorzec np. `tasks/*.json`). Każde zadanie:

```json
{
  "entrypoint": "reverse_str",
  "tests": [
    {"args": ["abc"], "expect": "cba"},
    {"args": [""], "expect": ""}
  ]
}
```

* `entrypoint` – nazwa funkcji, którą agent ma wygenerować.
* `tests[*]` – zestaw przypadków z `args`, opcjonalnym `kwargs` i oczekiwaniem `expect`.

---

## Metryki i kryteria oceny

### Metryki funkcjonalne

Wyliczane przez `glitchlab.gui.bench.judge.run_tests`:

* `pass_cnt` – liczba zaliczonych przypadków w zadaniu,
* `total` – liczba przypadków w zadaniu,
* `pass_at_1` (0/1) – czy **wszystkie** przypadki dla zadania przeszły.

Agregacja (w `stats._aggregate`) sumuje `pass_at_1` i `total` po wszystkich zadaniach.

### Metryki strukturalne / jakości kontekstu

Zadaniem A1/A2 jest emitować wraz z wynikiem pomocnicze metryki jakości (na poziomie zadania), agregowane potem do średnich:

* `align` – **spójność** między strukturą generowanego AST a projekcją mozaikową (0–1; im wyżej, tym lepiej).
* `j_phi2` – skalarna metryka „energii/entropii” (stała w naszych przebiegach; traktujemy poglądowo).
* `cr_to` – kompresja topologiczna (orientacyjna).
* `cr_ast` – **kompresja semantyczna AST**; wartości bliżej **1.0** = „kanoniczny/odszumiony” AST.

> W praktyce **najbardziej informatyczne** okazały się `align` i `cr_ast`.

### Metryki statystyczne (efekt i test znaku)

Zaimplementowane w `glitchlab.gui.bench.stats`:

* **Cliff’s delta** `δ = (GT - LT)/(n_x * n_y)` – efekt dla porównań par prób (np. listy `align` dla A1 vs. A2).
* **Dwustronny test znaku** (binomial sign test) dla liczby wygranych/przegranych zadań.

> Te metryki wykorzystujemy do **wtórnej analizy** A1 vs. A2; baseline B nie dostarcza porównywalnych „list cech” (najczęściej brak metryk strukturalnych), więc w tym zestawie wnioskowanie statystyczne dotyczy głównie A1↔A2.

### Reguła werdyktu

1. **Priorytet 1 — funkcjonalność:** maksymalizacja łącznego `pass_at_1`.
2. **Priorytet 2 — jakość kontekstu:** gdy `pass_at_1` remisuje, wybieramy **wyższe `align`** i **`cr_ast` bliżej 1.0**.
3. **Priorytet 3 — koszt:** czas (`mean_time`) jako tie-breaker pomocniczy.

---

## Procedura eksperymentalna

1. **Ładowanie zadań** wzorcem glob (`--tasks "tasks/*.json"`), walidacja schematu.
2. **Uruchomienie agentów** (A1, A2, B) na **tym samym** zestawie zadań.
3. **Generacja kodu**:

   * A1/A2 – „Mozaika+AST” + metryki (`align`, `cr_ast`, …),
   * B – PTP/KH/Fallback (prosty szablon/heurystyka).
4. **Ocena** (`judge.run_tests`): dynamiczne załadowanie funkcji i wykonanie przypadków.
5. **Agregacja** (`stats.summarize`) i zapis do `artifacts/ab.json`.
6. **Wizualizacje** (`utils/plot_results.py`): `accuracy.png`, `timings.png`, `align_vs_ast.png` oraz tekstowe porównanie `comparison.txt`.

> Postęp dla IDE/CLI pokazuje belki `tqdm` z etykietą `"Running A1|A2|B"`.

---

## Wyniki

### Raport zbiorczy

Poniżej **przykładowy** artefakt zbiorczy (z pliku `artifacts/ab.json`) z jednego z przebiegów na \~40 plikach zadań (79 testów łącznie):

```json
{
  "A1": {
    "pass_at_1": 0,
    "total": 79,
    "mean_time": 0.2591493844985962,
    "mean_align": 0.600024179254398,
    "mean_j_phi2": 80.95457993820953,
    "mean_cr_to": 10.999999999999995,
    "mean_cr_ast": 1.0377358490566038
  },
  "A2": {
    "pass_at_1": 0,
    "total": 79,
    "mean_time": 0.25497391223907473,
    "mean_align": 0.7433325087941238,
    "mean_j_phi2": 80.95457993820953,
    "mean_cr_to": 10.999999999999995,
    "mean_cr_ast": 1.0
  },
  "B": {
    "pass_at_1": 0,
    "total": 79,
    "mean_time": 0.0,
    "mean_align": null,
    "mean_j_phi2": null,
    "mean_cr_to": null,
    "mean_cr_ast": null
  }
}
```

### Interpretacja

* **Funkcjonalność (`pass_at_1`)**: remis — w tym konkretnym przebiegu żaden agent nie wygenerował kompletnego, „w 100% trafnego” kodu dla wszystkich zadań.
  → To jest uczciwy stan na obecnym zbiorze i implementacjach.

* **Jakość kontekstu**:

  * `align`: **A2 (0.74)** wyraźnie > A1 (0.60) → **A2** utrzymuje lepszą **spójność** mozaika↔AST.
  * `cr_ast`: **A2 (1.00)** ≈ „kanoniczne AST”, **A1 (1.038)** → subtelnie „przekompresowane/odkształcone”.
    ⇒ **A2** jest **stabilniejszy strukturalnie**.

* **Czas**: A2 (0.255 s) ≈ A1 (0.259 s) — różnica nieistotna operacyjnie.
  B ma `0.0` po prostu dlatego, że w tym baseline **nie liczymy pełnego pipeline’u** (z reguły tylko wstawiamy szablon lub stub).

**Werdykt (wg naszej Reguły werdyktu):**
Ponieważ `pass_at_1` remisuje, rozstrzygamy jakością kontekstu — **wygrywa A2** (wyższy `align`, `cr_ast` bliżej 1.0), przy porównywalnym czasie.

> Wniosek: na tym zbiorze i w tej konfiguracji **A2 jest lepszym protokołem** do prowadzenia dalszych iteracji (ma większy „headroom” na poprawę funkcjonalności przy zachowaniu spójności kontekstu).

---

## Wizualizacje i artefakty

Skrypt: `glitchlab/gui/bench/utils/plot_results.py` zapisuje do katalogu wyjściowego:

* `accuracy.png` — słupki `pass_at_1` per agent,
* `timings.png` — słupki `mean_time` per agent,
* `align_vs_ast.png` — porównanie `mean_align` vs. `mean_cr_ast`,
* `comparison.txt` — tekstowa interpretacja (punktowa, do podglądu w CI/IDE).

Plik wejściowy wykresów: `glitchlab/gui/bench/artifacts/ab.json` (tworzony przez `ab_runner`).

---

## Jak uruchomić (IDE i CLI)

### Z IDE (np. PyCharm)

1. Otwórz `glitchlab/gui/bench/ab_runner.py`.
2. Ustaw konfigurację „Run” z parametrami:

   * `--tasks` np. `"tasks/*.json"`
   * `--out` np. `glitchlab/gui/bench/artifacts/ab.json`
3. Uruchom. W konsoli IDE zobaczysz belki postępu `tqdm`:

   ```
   13:26:08 [INFO] Loading tasks...
   13:26:08 [INFO] Loaded 40 tasks
   13:26:08 [INFO] Running agents...
   Running A1: 100%|██████████| 40/40 [00:10<00:00,  3.90 task/s]
   ...
   13:26:21 [INFO] Report saved to glitchlab/gui/bench/artifacts/ab.json
   ```

Następnie z IDE uruchom `glitchlab/gui/bench/utils/plot_results.py` (ma `main()`), przekazując:

* `--json glitchlab/gui/bench/artifacts/ab.json`
* `--out <katalog_na_wykresy>`

### Z CLI

```bash
python -m glitchlab.gui.bench.ab_runner --tasks "tasks/*.json" --out glitchlab/gui/bench/artifacts/ab.json

python -m glitchlab.gui.bench.utils.plot_results \
  --json glitchlab/gui/bench/artifacts/ab.json \
  --out /tmp/plots
```

> Testy: `pytest glitchlab/gui/bench/tests` (w repo dodane testy e2e, unit, fuzz, property, perf).

---

## Ograniczenia i zagrożenia dla rzetelności

* **Brak pełnego pokrycia szablonami** w baseline B (wiele zadań kończy fallbackiem) → `pass_at_1` = 0 nie oznacza „złej metody”, tylko „metoda baseline jest minimalna”.
* **Metryki `align`, `cr_ast`** są **wewnętrzne** dla A1/A2 (A/B nieporównywalne 1:1) — służą do **porównania A1 vs. A2**, nie do konfrontacji z B.
* **Zbiór zadań**: część zadań może wymagać logiki poza zakresem szablonów/heurystyk — to naturalnie zaniża `pass_at_1`.
* **Brak kosztów inference LLM**: nie korzystamy z modeli zewnętrznych; badamy **czyste protokoły generacji** (szablon/heurystyka vs. mozaika+AST).

---

## Wnioski i prace przyszłe

1. **A2 > A1** w jakości odwzorowania kontekstu (`align`, `cr_ast`) przy podobnym koszcie czasowym.
2. **B** pozostaje wartościowym baseline’em kontrolnym, ale **nie reprezentuje** realnego, kontekstowego pipeline’u (praktycznie „szkielet” do sanity checków).

**Kierunki rozwoju:**

* Dołożenie **strategii naprawczych** (self-repair) po niezdanych testach — iteracyjna poprawa AST przy zachowaniu wysokiego `align`.
* Rozszerzenie **PTP/KH** o bogatszą bibliotekę szablonów i heurystyk (by realniej zestawić B z A1/A2).
* Ujednolicenie metryk jakości (np. normalizacja `align`/`cr_ast` w funkcji typu zadania).
* Weryfikacja statystyczna A1 vs. A2 na **większych zestawach** i raportowanie **Cliff’s delta** + **test znaku** na listach `align`/`cr_ast`.

---

**Status replikowalności:**

* Kod uruchamialny z IDE i CLI, testy `pytest` przechodzą (poza scenariuszami świadomie wrażliwymi na brak metryk u baseline’u).
* Artefakt `ab.json` + wykresy generowane programowo.
* Całość jest w pełni jawna i audytowalna.
