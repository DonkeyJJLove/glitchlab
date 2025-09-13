# GlitchLab GUI v2 — ARCHITEKTURA (stan bieżący)

> Ten plik opisuje wyłącznie **aktualny stan** warstwy GUI (to, co jest w paczce `gui/`). Refaktoryzacje i propozycje zmian opiszemy osobno.

---

## 1) Zakres i cel

Warstwa **GUI** zapewnia:

* interaktywne uruchamianie pipeline’u (`core`),
* edycję presetów v2 (YAML) i parametrów filtrów,
* przegląd i wizualizację kanałów HUD z `ctx.cache`,
* podgląd grafu procesu (AST/DAG) i „mozaikowej” soczewki,
* zarządzanie maskami/ROI i amplitudą,
* ergonomię pracy (dokowanie, skróty, status/logi).

Projekt jest oparty o **Tkinter/ttk** + **Pillow** + **NumPy**, z **miękkimi zależnościami** na `glitchlab.core.*` (fallbacki pozwalają odpalić UI bez zainstalowanego core).

---

## 2) Struktura pakietu GUI (faktyczna)

```
gui/
  ARCHITECTURE.md                # ten dokument (wersja z paczki)
  __init__.py

  # rdzeń GUI
  app.py                         # główna ramka aplikacji (layout, menu, akcje, bieg asynch.)
  controls.py                    # proste fabryki widżetów (spin/enum/checkbox/slider)
  docking.py                     # DockManager: undock/dock + serializacja stanu
  exporters.py                   # export_hud_bundle(ctx), save_layout/load_layout
  generic_form_panel.py          # GenericFormPanel – fallback formularza parametrów
  image_canvas.py                # starszy/alternatywny ImageCanvas (scroll/zoom) – wersja prostsza
  log_window.py                  # Toplevel logów „Show log”
  panel_base.py                  # PanelBase/PanelContext – kontrakty danych dla paneli
  panel_loader.py                # get_panel_class() – próba importu panelu dedykowanego
  paths.py                       # wykrywanie katalogu projektu/presetów (prosty wariant)
  preset_dir_fallback.py         # fallback logiki presetów (jeśli manager nie działa)
  preset_manager.py              # PresetManager (edycja YAML, historia)
  preset_paths.py                # trwałe ścieżki presetów (odczyt/zapis config)
  state.py                       # UiState – spójny zapis bieżącego stanu GUI
  welcome_panel.py               # prosty panel powitalny (nieużywany w app.py)

  # panele filtrów (dedykowane)
  panels/
    __init__.py
    base.py                      # rejestr paneli (register_panel/get_panel/list_panels)
    panel_anisotropic_contour_warp.py
    panel_block_mosh.py
    panel_block_mosh_grid.py
    panel_default_identity.py
    panel_depth_displace.py
    panel_gamma_gain.py
    panel_phase_glitch.py
    panel_pixel_sort_adaptive.py
    panel_rgb_glow.py
    panel_rgb_offset.py
    panel_spectral_shaper.py
    panel_tile_tess_probe.py

  # widżety „nowej generacji”
  widgets/
    __init__.py
    graph_view.py                # placeholder widoku grafu (AST/DAG)
    hud.py                       # Hud: 3 sloty, autoskan kluczy cache
    image_canvas.py              # ImageCanvas: zoom/pan, viewport, callbacki
    mask_browser.py              # eksploracja masek
    mask_chooser.py              # wybór/dodawanie masek z plików
    mask_picker.py               # picker punktowy
    menu_model.py                # model menu
    mosaic_view.py               # MosaicMini: licznik nodes/edges z ast/json
    panel_toolbox.py             # toolbox trybów (hand/zoom/pick)
    pipeline_preview.py          # siatka miniaturek (np. wejście/wyjście/kroki)
    preset_folder.py             # UI wyboru folderu presetów
    preset_manager.py            # PresetManager (wariant modułowy)
    welcome_panel.py             # panel powitalny (wariant modułowy)
```

**Uwaga o duplikatach:** w paczce występują równoległe implementacje (np. `image_canvas.py` w korzeniu oraz `widgets/image_canvas.py`, podobnie PresetManager/WelcomePanel). `app.py` **preferuje** moduły z `widgets/` (import w try/except) i ma fallbacki, aby UI działało także bez części komponentów.

---

## 3) Zależności i środowisko

* **Tkinter/ttk** (systemowy), **Pillow**, **NumPy** (twarde).
* **PyYAML** (opcjonalnie) — edycja presetów w `preset_manager.py`.
* **glitchlab.core** (miękkie):
  `app.py` próbuje importować:

  * `glitchlab.core.registry.available/get` (lista filtrów, pobranie funkcji),
  * `glitchlab.core.pipeline.apply_pipeline` (uruchomienie kroków).
    Brak core → wbudowane **fallbacki** (UI wstaje, akcje operują na stubach).

---

## 4) Model danych w GUI

### 4.1 UiState (`gui/state.py`)

```python
@dataclass
class UiState:
    image: Any | None
    preset_cfg: Dict[str, Any] | None
    single_filter: str | None
    filter_params: Dict[str, Any]
    seed: int
    hud_mapping: Dict[str, list[str]]  # domyślnie 3 sloty: in/out/diff + metryki
```

GUI utrzymuje też „ostatni kontekst” (`self.last_ctx`: `rng`, `amplitude`, `masks`, `cache`, `meta`) jako lekki, dynamiczny obiekt – przechowuje telemetry z ostatniego runu oraz wgrywane maski.

### 4.2 Kanały HUD (konsumowane)

HUD czyta z `ctx.cache` m.in.:

* `stage/{i}/in|out|diff|t_ms`
* `stage/{i}/metrics_in|metrics_out|diff_stats`
* `stage/{i}/fft_mag|hist`
* `stage/{i}/mosaic|mosaic_meta`
* `diag/<filter>/...`
* `ast/json`, `format/jpg_grid`, `format/notes`
* `cfg/*`, `run/id`, `run/snapshot`

Wyświetlacz HUD (`widgets/hud.Hud`) **sam „zgaduje”** najlepsze źródła do Slotów 1–3 (np. amplitude/edge/FFT) – priorytetowo szuka znanych kluczy.

---

## 5) Główne komponenty

### 5.1 `App` (główna ramka; `gui/app.py`)

**Odpowiada za:**

* budowę layoutu i paneli,
* zarządzanie obrazem wejściowym i jego podglądem,
* integrację z presetami i panelami filtrów,
* uruchamianie pipeline’u (async worker + progres),
* synchronizację HUD/miniatur/overlays,
* zapis logów i status.

**Layout (faktyczny):**

* **Statusbar** (dół): progressbar, status, koordynaty `(x,y)`, przycisk logów, skróty F8/F9/F10 do chowania paneli.

* **Hor. split**: `viewer` | `tools_left` | `tools_right`

  * **viewer**: `widgets.image_canvas.ImageCanvas` + *axes overlay*, crosshair oraz „rulery” (*Ruler*) i pomocnicze markery współrzędnych; opcjonalna projekcja transformacji widoku (roll/pitch/yaw — prosta transformacja z Pillow).
  * **tools\_left**: **toolbox** (hand/zoom/pick), współrzędne markera (X/Y), przyciski osi, itp.
  * **tools\_right (zakładki)**:

    * **Global**: folder presetów, seed, amplitude & edge globalne, maski (via `MaskChooser`).
    * **Filter**: wybór filtra, osadzenie panelu dedykowanego lub GenericForm, przycisk **Apply**.
    * **Presets**: `PresetManager` (lista, edycja YAML, historia).

* **HUD (na dole)**: 3 sloty `_HudSlot` (`widgets/hud.py`) – obraz/tekst/miniatury wybierane automatycznie z cache.

**Kluczowe metody:**

* `on_open()` / `on_save()` — I/O obrazów (Pillow).
* `on_apply_filter()` — uruchomienie pojedynczego kroku (`apply_pipeline` na liście `steps=[{name,params}]`).
* `on_apply_preset()` — uruchomienie sekwencji kroków z YAML.
* `_build_ctx_for_run()` — składanie lekkiego `Ctx` (rng/maski/amplituda/cache/meta) z aktualnego stanu UI.
* `_run_async(label, func, on_done)` — prosty worker wątku z `Progressbar` i logami w `LogWindow`.
* `_refresh_display()` — rysowanie obrazu + ewentualna transformacja widoku (roll/pitch/yaw).
* `_update_hud()` — render slotów na podstawie `last_ctx.cache`.

**Skróty klawiszowe (zdefiniowane):**

* `Ctrl+O` (open), `Ctrl+S` (save),
* `F8` (toggle lewy toolbox), `F9` (toggle prawa kolumna), `F10` (toggle HUD bottom).

### 5.2 System paneli filtrów

* **PanelLoader** (`gui/panel_loader.py`):
  próbuje `glitchlab.gui.panels.panel_<filter_name>`, zwraca klasę `ttk.Frame`; w razie błędu → **GenericFormPanel**.

* **PanelBase / PanelContext** (`gui/panel_base.py`):
  kontrakty na dane i kontekst panelu (dostęp do `defaults/params`, listy masek, callback `on_change`).

* **GenericFormPanel** (`gui/generic_form_panel.py`):
  dynamiczny formularz; wszystkie pola zawierające „mask” wyświetla jako **Combobox** z kluczami masek; łączy `defaults` i bieżące `params` filtra.

* **Panele dedykowane** (`gui/panels/*.py`) – implementują własne UI:
  `anisotropic_contour_warp`, `block_mosh`, `block_mosh_grid`, `default_identity`,
  `depth_displace`, `gamma_gain`, `phase_glitch`, `pixel_sort_adaptive`,
  `rgb_glow`, `rgb_offset`, `spectral_shaper`, `tile_tess_probe`.

> App dynamicznie montuje panel po wyborze filtra, a zmiany emitowane przez panel zasilają **`_collect_current_filter_params()`** przed runem.

### 5.3 HUD (3-slotowy)

`widgets/hud.py`:

* Slot przyjmuje obraz (`np.ndarray` lub `PIL.Image`) **lub** tekst (wówczas etykieta/tekst — np. metryki).
* Konwertuje typy: `(H,W)`/`(H,W,1)`/`(H,W,3)` oraz float/uint8.
* Domyślny „routowanie” kluczy:
  Slot 1 → amplitude/FFT, Slot 2 → krawędzie/diag, Slot 3 → maski/spectral.

### 5.4 Podgląd obrazu

`widgets/image_canvas.py` (nowszy, używany) oraz `gui/image_canvas.py` (starszy, prostszy wariant):

* funkcje: `set_image`, `zoom_in/out/to`, `fit`, `center`, odczyt viewportu w pikselach obrazu, callback `on_view_changed`.
* obsługa kółka myszy/drag/pan (zależnie od trybu narzędzia).
* współpraca z nakładkami (overlay osi, crosshair, markery).

### 5.5 Presety

* `gui/preset_manager.py` oraz `widgets/preset_manager.py` (wariant modułowy) — edycja/załadunek presetów, historia kroków, walidacja minimalna (YAML).
* `preset_paths.py` — wykrywanie katalogów presetów, zapamiętywanie ostatnio użytego (`~/.config/...`).
* `paths.py` — proste heurystyki ścieżek (np. `presets/` obok projektu).

### 5.6 Dokowanie

`DockManager` (`gui/docking.py`):

* `undock(slot_id, title)` → przenosi zawartość slotu do `Toplevel`.
* `dock(slot_id)` → zwraca widżety ze „świecącego” okna do slotu.
* `save_layout()/load_layout()` → zapisuje listę slotów pływających.

### 5.7 Eksport/diagnostyka

`exporters.py`:

* `export_hud_bundle(ctx)` — zwraca lekki DTO (run/ast/stages/format) **bez obrazów** (GUI referuje obrazki po kluczach w cache).
* `save_layout(path, d)` / `load_layout(path)` — serializacja ustawień layoutu.

### 5.8 Widoki analityczne

* `widgets/graph_view.py` — **placeholder**: pokazuje tekstową informację o grafie z `ast/json`.
* `widgets/mosaic_view.py` — **MosaicMini**: parsuje `ast/json` i wyświetla licznik `nodes/edges` (spójne z ideą „mozaiki” jako metastruktury projekcyjnej).

---

## 6) Integracja z `core`

GUI zakłada następujące interfejsy:

* **Rejestr filtrów** (`glitchlab.core.registry`):

  * `available()` — lista nazw filtrów (do Comboboxa),
  * `get(name)` — callable filtra (dla metadanych i fallbacków).

* **Pipeline** (`glitchlab.core.pipeline.apply_pipeline(img_u8, ctx, steps, ...) -> np.ndarray`):

  * `img_u8`: `uint8 RGB (H,W,3)`,
  * `ctx`: obiekt z polami: `rng`, `amplitude: (H,W) f32 [0..1]`, `masks: {str -> (H,W) f32}`, `cache` (dict), `meta` (dict),
  * `steps`: `[{"name": <filter>, "params": {...}}]`.

**Przepływ GUI → core:**

1. Wczytaj obraz (`on_open`) → `img_u8`.
2. Zbierz preset/parametry → `steps`.
3. Złóż `ctx` (`_build_ctx_for_run`) – przenieś maski z UI do `ctx.masks`; przygotuj `amplitude`; wstaw `meta` (seed, źródło).
4. Uruchom `apply_pipeline` w workerze (`_run_async`) → dostaniesz `out` i wypełniony `ctx.cache`.
5. Odśwież **viewer** + **HUD**; zaktualizuj historię kroków w `PresetManager`.

**Fallbacki:** jeśli moduły `core` nie są dostępne, `app.py` wstawia stuby (UI nadal działa, ale bez realnej transformacji).

---

## 7) Kontrakty UI (ważne fragmenty API)

### 7.1 Panel (dedykowany i generowany)

```python
# gui/panel_base.py
class PanelBase(ttk.Frame):
    def mount(self, ctx: PanelContext) -> None: ...
    def values(self) -> dict: ...   # parametry filtra do pipeline
    # ctx.defaults / ctx.params / ctx.mask_names() / ctx.emit(...)
```

```python
# gui/generic_form_panel.py
class GenericFormPanel(BasicPanel):
    # generuje pola z ctx.defaults/params; klucze zawierające 'mask' → Combobox
```

### 7.2 Loader paneli

```python
# gui/panel_loader.py
def get_panel_class(filter_name: str) -> type[ttk.Frame]:
    # próbuje importu glitchlab.gui.panels.panel_<filter_name>
    # w razie niepowodzenia → GenericFormPanel
```

### 7.3 HUD

```python
# widgets/hud.py
class Hud(ttk.Frame):
    def render_from_cache(self, ctx_like) -> None:
        # automatycznie wybiera źródła do 3 slotów (priorytety znanych kluczy)
```

### 7.4 Eksport bundle

```python
# gui/exporters.py
def export_hud_bundle(ctx_like) -> dict:
    # {"run": {...}, "ast": {...}, "stages": [...], "format": {...}}
```

---

## 8) Przepływy użytkowe (E2E)

### 8.1 Pojedynczy filtr

```
Open image → Filter tab: wybór filtra → panel (dedykowany / GenericForm)
→ Apply → _build_ctx_for_run → apply_pipeline([ {name, params} ])
→ GUI aktualizuje viewer + HUD
```

### 8.2 Preset YAML (sekwencja kroków)

```
Open image → Presets tab: wybór/edycja YAML v2
→ Apply preset → _build_ctx_for_run → apply_pipeline(steps z YAML)
→ Historia kroków w PresetManager; viewer + HUD odświeżone
```

### 8.3 Maski i amplitude

```
Global tab → MaskChooser: Add mask (z pliku) → ctx.masks["key"] = f32 [0..1]
→ Parametry filtrów mogą wskazać "mask_key"
→ HUD slot 1 zwykle pokazuje amplitude/fft; slot 2 – edge/diag; slot 3 – spectral/mask
```

---

## 9) Spójność z metastrukturą „mozaiki”

GUI przewiduje **wspólny język telemetrii** z `core/analysis`: obrazy i mapy są przekazywane przez stabilne **klucze** w `ctx.cache`. Dzięki temu:

* `widgets/mosaic_view.MosaicMini` może pobrać `ast/json` i wyświetlać parametry „mozaiki” (liczność `nodes/edges`) niezależnie od tego, jak powstał graf.
* Sloty HUD są **agnostyczne** – pokażą dowolną mapę (`stage/{i}/mosaic`, `stage/{i}/fft_mag`, `diag/<filter>/*`) bez dodatkowej logiki w GUI.
* „Mozaika” pełni rolę **ontologicznego rdzenia projekcji**: GUI nie interpretuje macierzy – jedynie je **renderuje** i etykietuje, ufając spójności kontraktów.

---

## 10) Błędy i obsługa

* Operacje (apply) są **asynchroniczne** — worker thread + progressbar.
* Wyjątki są przechwytywane, komunikowane `messagebox.showerror`, logowane w `LogWindow`.
* `fail_fast=True` jest przekazywane do pipeline (po stronie core).
* „Puste” zależności (brak PIL/core/yaml) mają **fallback**: UI działa, ale funkcje zależne wyświetlą komunikat lub no-op.

---

## 11) Wydajność i UX (stan)

* Miniatury/HUD generowane na podstawie danych z `ctx.cache` — bez kosztownych obliczeń w GUI.
* Viewer korzysta z `ImageTk.PhotoImage`, aktualizowany przy zmianach zoom/pan/overlay.
* Proste **debounce**/ograniczenie odświeżeń wynika z konstrukcji callbacków (brak twardego throttlingu — zarządza tym App).

---

## 12) Testowalność (co jest w kodzie)

* Brak osobnych testów jednostkowych – GUI można wstawić w środowisku headless (np. z `tkinter.Tcl().eval('…')`), ale bieżący kod nie zawiera gotowych testów.
* Łatwość smoke: uruchomić `App` z obrazem dummy i pustym `ctx.cache`, sprawdzić działanie slotów HUD (placeholders).

---

## 13) Interfejsy zewnętrzne (publiczne, użyteczne dla innych modułów)

* `App`:

  * `on_open()/on_save()` – I/O,
  * `on_apply_filter()/on_apply_preset()` – uruchamianie pipeline’u,
  * `set_image(pil_image: Image.Image)` / `_show_on_canvas(u8: np.ndarray)` – odświeżanie podglądu,
  * `preset_mgr.*` – interakcja z managerem presetów (API przyjazne, ale nieustandaryzowane).

* `DockManager`:

  * `undock(slot_id, title)` / `dock(slot_id)`,
  * `save_layout()` / `load_layout(d)`.

* `exporters`:

  * `export_hud_bundle(ctx_like)` – thin DTO dla HUD (bez obrazów),
  * `save_layout(path, d)` / `load_layout(path)`.

* `panel_loader`:

  * `get_panel_class(filter_name)` – strategia odnajdywania panelu dedykowanego.

---

## 14) Zgodność z kontraktami systemowymi

GUI pozostaje zgodne z ogólną specyfikacją projektu:

* **Filter API v2** (pośrednio – przez `core`),
* **Preset schema v2** (`version/name/amplitude/edge_mask/steps`),
* **HUD channels** (`ctx.cache` kluczowane po stałych nazwach),
* **AST/DAG** – `ast/json` jako „lekki” opis procesu.

---

## 15) Checklista utrzymaniowa GUI (stan/oczekiwania)

* [ ] `panel_loader` – komplet paneli dedykowanych dla filtrów z rejestru.
* [ ] `widgets/hud` – priorytety kluczy HUD zgodne z tym, co emituje `core`.
* [ ] `export_hud_bundle` – zgodność ze strukturą `core.graph`.
* [ ] `DockManager` – zapis/odczyt layoutu (integracja z plikami ustawień).
* [ ] `PresetManager` – dostęp do aktualnego katalogu presetów (środowisko/`preset_paths.py`).
* [ ] Viewer – stabilne odświeżanie przy zoom/pan (bez migotania).

---

## 16) Krótka ściąga (dla deweloperów paneli)

1. **Dodaj panel**: `gui/panels/panel_<filter>.py`, klasa `ttk.Frame` (lub `PanelBase`) z metodami `mount(ctx)` i `values()`.
2. **Import przez nazwę filtra**: `panel_loader` szuka `glitchlab.gui.panels.panel_<filter>`.
3. **Maski**: użyj `ctx.mask_names()` oraz pola typu `"mask_key"` w formularzu.
4. **Emituj zmiany**: `ctx.emit({...})` → `App` pobierze parametry przed runem.
5. **HUD**: wszystko, co filtr odkłada do `ctx.cache["diag/<filter>/..."]`, będzie widoczne w slotach (priorytet zależny od kluczy).

---

## 17) Licencja i autorzy

Open Source — D2J3 aka Cha0s (for test and fun).

---


