# GlitchLab GUI v4 — Architektura (rozszerzona)

> Ten dokument opisuje **wyłącznie warstwę GUI**: strukturę pakietu, kontrakty między widokami a rdzeniem (Core v2), przepływ zdarzeń, mechanikę ładowania paneli, diagnostykę, stan aplikacji i decyzje projektowe. Język: polski, ze spójną terminologią.


## 1) Cel i filozofia GUI

- **Stabilny layout**: przewidywalny podział na podgląd (lewa strona), parametry (prawa strona) i HUD/telemetrię (spód).
- **Niska kruchość**: każdy błąd w panelu / filtrze nie powinien „wysadzić” aplikacji — fallbacki i logi powinny poprowadzić użytkownika do rozwiązania.
- **Diagnostyka pierwszej klasy**: GUI ma wspierać iterację „zmiana → run → wgląd w telemetrię” z minimalnym tarciem.
- **Rozszerzalność**: dodanie nowego panelu nie wymaga dotykania loadera; panele działają w dwóch konwencjach nazewniczych.


## 2) Anatomia pakietu GUI

```
glitchlab/gui/
  app.py                  # ramka aplikacji (bez mainloop)
  docking.py              # DockManager (dock/undock dla slotów)
  panel_loader.py         # fabryka klas paneli (heurystyki + fallback)
  panel_base.py           # PanelBase / PanelContext (API dla paneli)
  views/
    tab_filters.py        # TabFilter: wybór filtra, loader panelu, Apply, diagnostyka
    tab_general.py        # zakładka „General” (maski, amplitude, edge, itp.)
    tab_preset.py         # menedżer presetów (otwieranie/zapis, historia kroków)
  panels/
    __init__.py           # AUTO-IMPORT paneli (panel_<n>.py i <n>_panel.py)
    base.py               # aliasy kompatybilnościowe (BasicPanel, register_panel)
    panel_<name>.py       # dedykowane panele (np. panel_anisotropic_contour_warp.py)
    <name>_panel.py       # alternatywna konwencja nazwy
  widgets/
    image_canvas.py       # płótno z powiększaniem/przesuwaniem
    hud.py                # HUD (sloty, miniatury, metryki, diff)
    param_form.py         # ParamForm – autogenerowany formularz (fallback)
    diag_console.py       # opcjonalna konsola diagnostyczna (odbiornik EventBus)
```

**Decyzje:**  
- `panels/__init__.py` skanuje podpakiet i importuje wszystkie moduły paneli, **obsługując oba schematy nazw**. Dzięki temu stare i nowe pliki współistnieją.  
- `panel_loader.py` nadal potrafi **wprost załadować** panel na podstawie nazwy filtra, ale TabFilter preferuje mechanizm z **dwoma kandydatami** (patrz §5).


## 3) Stan aplikacji (AppState) i jego użycie

```python
class AppState:
    image: PIL.Image | None
    preset: dict | None
    cache: dict                 # ostatni cache z pipeline (dla HUD)
    masks: dict[str, np.ndarray]
```

- **Źródło prawdy** dla bieżącej sesji GUI.  
- Udostępniany jako `ctx_ref` do zakładek/paneli; panele **nie modyfikują** stanu bezpośrednio — emituje się zmiany parametrów, a `app.py` decyduje co z nimi zrobić (np. uruchomić filtr/pipeline).  
- `cache` jest przekazywany do HUD; komponenty HUD są **czytnikami** wybranych kluczy.


## 4) EventBus i tematy (kanały)

GUI publikuje i (opcjonalnie) subskrybuje zdarzenia. Minimalny zestaw:

- `ui.filter.select` — zmiana filtra w TabFilter `{name}`  
- `ui.filter.params_changed` — zmiana parametrów panelu `{name, params}`  
- `ui.run.apply_filter` — żądanie uruchomienia pojedynczego kroku `{step}`  
- `ui.presets.save_request` — zapis presetu (np. z menu)  
- `diag.log` — linie logu diagnostycznego `{level, msg}` (odbierane przez `DiagConsole`, a w razie braku — lecą na stdout)

**Zasada:** interakcje użytkownika → *publish* → `app.py` (lub serwisy) reagują. Dzięki temu wymiana widżetów nie wymaga zmian w logice.


## 5) System paneli — rozpoznawanie i ładowanie

### 5.1 Konwencje nazewnicze modułów
- `glitchlab.gui.panels.panel_<nazwa>` **oraz** `glitchlab.gui.panels.<nazwa>_panel`  
- `panels/__init__.py` auto-importuje wszystkie takie pliki podczas importu pakietu.  

### 5.2 Wybór klasy panelu
- Priorytet ma symbol eksportowany jako **`Panel`**.  
- Jeśli brak, wybierana jest **pierwsza klasa kończąca się na `Panel`**.

### 5.3 Konstrukcja i kontekst
- Preferowany konstruktor: `Cls(parent, ctx=PanelContext(...))`.  
- Dla starszych paneli dopuszczalne: `Cls(parent)` — TabFilter wykrywa `TypeError` i próbuje bez `ctx`.

**PanelContext — minimalny interface dla panelu:**
```python
PanelContext(
  filter_name: str,             # kanoniczna nazwa filtra
  defaults: dict,               # defaults z registry.meta(name)
  params: dict,                 # wstępne parametry (często puste)
  on_change: Callable[[dict], None],   # callback — panel emituje tu aktualne parametry
  cache_ref: dict | None,       # referencja do cache (np. do odczytu listy kluczy)
  get_mask_keys: Callable[[], list[str]] | None  # źródło listy masek (np. do dropdownu)
)
```

### 5.4 Fallback: ParamForm
Jeżeli nie udało się odnaleźć/utworzyć panelu, TabFilter tworzy **ParamForm**, który:
- próbuje odczytać **schemat parametrów** z rejestru (`registry.meta(name)["defaults"]` / `schema()`), a gdy to niemożliwe — **z sygnatury** funkcji filtra,
- generuje kontrolki `int/float/bool/str/enum`,
- emituje `on_change` po modyfikacji pól.


## 6) Zakładka Filters (TabFilter) — mechanika i diagnostyka

### 6.1 Przepływ
1. `Load Filters` → wymusza `import glitchlab.filters` (rejestr nie będzie pusty).  
2. `Rescan` → skanuje `glitchlab.gui.panels` i **uzupełnia listę filtrów** nawet bez rejestru.  
3. Zmiana pozycji w combobox → `ui.filter.select` i próba montażu panelu.  
4. Brak panelu? → ParamForm (fallback).  
5. Zmiana parametru → `ui.filter.params_changed`.  
6. `Apply` → `ui.run.apply_filter { step }`.

### 6.2 Diagnostyka
- **Probe**: `find_spec`, próba `import`, lista klas `*Panel` — logi na EventBus.  
- **Reload**: przeładowanie modułu panelu (oraz ostatnio użytego) → ponowny mount.  
- W logach: statusy `OK/WARN/ERROR/DEBUG` z przyczyną (traceback przy błędach).


## 7) HUD — odczyt z cache i mapowanie slotów

- HUD nie liczy niczego — **tylko czyta** wpisy z `ctx.cache`.  
- Klucze standardowe (per etap):  
  `stage/{i}/in|out|diff|t_ms|metrics_in|metrics_out|diff_stats`  
- Klucze globalne: `ast/json`, `format/jpg_grid`, `format/notes`, `cfg/*`, `run/id`, `diag/*`.

**Mapowanie (domyślne):**
```yaml
hud:
  slot1: ["stage/0/in", "stage/0/metrics_in", "format/jpg_grid"]
  slot2: ["stage/0/out", "stage/0/metrics_out", "stage/0/fft_mag"]
  slot3: ["stage/0/diff", "stage/0/diff_stats", "ast/json"]
```
HUD wybiera **pierwszy istniejący** klucz z listy przy renderowaniu slotu.


## 8) Dock/Undock (DockManager)

- Każdy slot (np. `right`, `hud`) może zostać „odczepiony” do `Toplevel`.  
- `DockManager` trzyma mapę `{slot_id: Frame}` i potrafi przenosić dzieci między ramami.  
- Warto przechowywać **layout** (pozycje splitterów, stan slotów) w `~/.glitchlab/ui.json`.


## 9) Integracja z Core v2 — kontrakty i typy

- GUI przekazuje **zawsze** obraz jako `np.ndarray uint8 RGB` do pipeline.  
- `build_ctx(img_u8, seed, cfg)` zwraca `Ctx` z `amplitude/masks/cache/meta`.  
- `apply_pipeline(img_u8, ctx, steps)` realizuje kroki, zapisuje telemetrię.  
- Po zakończeniu: `AppState.cache = ctx.cache`, HUD robi `render_from_cache(cache)`.

**Uwaga:** jeśli masz obraz w `Pillow.Image`, konwersja w GUI: `np.asarray(img.convert("RGB"), np.uint8)`.


## 10) Obsługa błędów i komunikaty

- Panele ładowane w `try/except`, błędy idą do logu diagnostycznego.  
- Braki w rejestrze → sugestia „Load Filters” lub „Rescan” (komunikat w UI).  
- Operacje I/O (`Open/Save`) — standardowe okna i krótkie komunikaty z wyjątków.  
- Nie maskujemy wyjątków pipeline, jeśli `fail_fast=True` — wyświetlamy je w dialogu i logu.


## 11) Wydajność i ergonomia

- `ImageCanvas` powinien cache’ować `PhotoImage` dla bieżącego zoomu; zmniejszać obraz do ekranu.  
- HUD operuje na miniaturach z cache (`_thumb_rgb` po stronie Core).  
- Parametry w panelach: debouncing zmian (np. `trace_add` + `after_idle`) — mniej rerunów.  
- Ruler/crosshair rysowane na lekkich `Canvas`-overlay, aktualizowane tylko przy potrzebie.


## 12) Rozszerzanie: nowy panel w 5 minut

1. Dodaj plik `glitchlab/gui/panels/panel_myfilter.py` z klasą `Panel`.  
2. W konstruktorze przyjmij `ctx: PanelContext`; zbuduj UI i **emituj on_change** po każdej zmianie.  
3. Niczego nie rejestrujesz ręcznie — `panels/__init__.py` i loader zrobią resztę.  
4. Jeśli nie chcesz dedykowanego panelu — zaktualizuj `registry.meta("myfilter")["defaults"]`; ParamForm utworzy formularz sam.


## 13) Testy (praktyczne)

- **Smoke GUI (headless)**: Inicjalizacja `App` + sztuczny obraz `np.zeros((96,128,3), u8)+40` + „pierwszy filtr z `available()`”. Sprawdź, że TabFilter buduje panel lub ParamForm i że `Apply` nie rzuca wyjątkiem.  
- **Probe**: uruchom `Probe` dla kilku filtrów; w logu powinny pojawić się linie `find_spec OK`, `import OK`, `panel classes: [...]`.  
- **Reload**: zmień kod panelu, kliknij `Reload`, sprawdź że nowy UI się montuje bez restartu aplikacji.


## 14) Dziennik zmian (co zostało poprawione w tej linii rozwojowej)

- **Auto-import paneli** (`panels/__init__.py`) — obsługa *dwóch* konwencji nazw plików paneli; brak ręcznych importów.  
- **TabFilter** — nowy loader z listą kandydatów + komplet narzędzi diagnostycznych (`Load Filters`, `Rescan`, `Probe`, `Reload`).  
- **Lepsze fallbacki** — `PanelContext` i `PanelBase` mają łagodne definicje awaryjne, ParamForm jest odporny na braki metadanych.  
- **Spójne logowanie po polsku** — wszystkie komunikaty i dokumentacja w jednym języku.  
- **Uszczelnienie kontraktów Core↔GUI** — konsekwentne użycie `np.ndarray uint8 RGB` i `build_ctx/apply_pipeline`.


## 15) Checklist wdrożeniowy (GUI)

- [ ] `panels/__init__.py` obecny i działa (wyświetla nowe panele po dodaniu pliku).  
- [ ] `tab_filters.py` pokazuje listę filtrów z rejestru; gdy pusty — potrafi znaleźć po skanie paneli.  
- [ ] `Probe/Reload` działają i logują do EventBus.  
- [ ] ParamForm umie zbudować formularz choćby z sygnatury funkcji filtra.  
- [ ] HUD renderuje miniatury/metryki/diff po `Apply`.  
- [ ] `app.py` poprawnie konwertuje obraz do `np.uint8 RGB` i obsługuje błędy pipeline.

---

### Aneks: Przykładowy panel (skrót)

```python
# glitchlab/gui/panels/panel_myfilter.py
import tkinter as tk
from tkinter import ttk

try:
    from glitchlab.gui.panel_base import PanelContext
except Exception:
    class PanelContext: 
        def __init__(self, **kw): self.__dict__.update(kw)

class Panel(ttk.Frame):
    def __init__(self, master, ctx: PanelContext = None):
        super().__init__(master)
        self.ctx = ctx or PanelContext(filter_name="myfilter", defaults={}, params={}, on_change=None)
        self.var_strength = tk.DoubleVar(value=float(self.ctx.defaults.get("strength", 1.0)))
        ttk.Label(self, text="Siła").grid(row=0, column=0, sticky="w")
        ttk.Scale(self, from_=0.0, to=5.0, variable=self.var_strength).grid(row=0, column=1, sticky="ew")
        ttk.Entry(self, textvariable=self.var_strength, width=6).grid(row=0, column=2, sticky="w")
        self.columnconfigure(1, weight=1)
        self.var_strength.trace_add("write", lambda *_: self._emit())  # emituj po zmianie
        self._emit()

    def _emit(self):
        cb = getattr(self.ctx, "on_change", None)
        if callable(cb):
            cb({"strength": float(self.var_strength.get())})
```
