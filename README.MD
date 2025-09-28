# GlitchLab GUI v4.5

**GlitchLab GUI v4.5** to modularna aplikacja graficzna do testowania, uruchamiania i diagnostyki filtrów obrazu. Projekt kładzie nacisk na **stabilność interfejsu**, **niską kruchość (wysoką odporność na błędy)**, **diagnostykę pierwszej klasy** i **łatwą rozszerzalność**.

---

## Cele i założenia

* **Stabilny układ interfejsu** – stały layout:

  * lewy panel – podgląd obrazu,
  * prawa kolumna – parametry filtra,
  * dolny HUD – telemetria i diagnostyka.

* **Niska kruchość** – błędy w panelach czy filtrach nie wyłączają aplikacji. Mechanizmy *fallback* (np. automatyczny formularz parametrów) zapewniają dalszą pracę mimo problemów.

* **Diagnostyka pierwszej klasy** – wszystkie dane z rdzenia (obrazy pośrednie, metryki, grafy AST/DAG) są natychmiast dostępne w GUI (HUD, konsola diagnostyczna).

* **Rozszerzalność** – nowe filtry i panele można dodawać bez modyfikacji istniejącego kodu (system auto-importu, fallback formularza).

---

## Struktura projektu (skrót)

```
glitchlab/gui/
├── app.py            # główne okno i pętla zdarzeń
├── docking.py        # mechanizm dock/undock paneli
├── panel_loader.py   # fabryka i loader paneli
├── panel_base.py     # klasy bazowe i API paneli
├── views/            # główne zakładki UI
│   ├── tab_filters.py
│   ├── tab_general.py
│   └── tab_preset.py
├── panels/           # dedykowane panele filtrów
│   └── __init__.py   # auto-import paneli (panel_*.py, *_panel.py)
├── widgets/          # niestandardowe widżety
│   ├── image_canvas.py
│   ├── hud.py
│   ├── param_form.py
│   └── diag_console.py
```

---

## Uruchamianie

1. **Zainstaluj wymagania** (Python 3.10+):

   ```bash
   pip install -r requirements.txt
   ```

2. **Uruchom aplikację**:

   ```bash
   python -m glitchlab.gui.app
   ```

3. **Załaduj filtr** w zakładce *Filtry* → kliknij `Load Filters`.

---

## Kluczowe funkcje

* **Zakładka Filtry (TabFilter):** wybór i uruchamianie filtrów, diagnostyka (`Probe`, `Reload`).
* **HUD (Heads-Up Display):** 3 sloty na obrazy, metryki, różnice, grafy AST/DAG.
* **Docking:** możliwość odpinania paneli i HUD jako osobnych okien (np. na drugi monitor).
* **Fallback ParamForm:** automatyczny formularz, gdy brak dedykowanego panelu.
* **EventBus:** architektura zdarzeniowa (luźno powiązane komponenty).
* **Skróty klawiszowe:**

  * `Ctrl+O / Ctrl+S` – otwórz / zapisz,
  * `Ctrl+R` – uruchom filtr,
  * `Ctrl+E` – eksport konfiguracji,
  * `F` – pełnoekranowy podgląd,
  * `1 / 2 / 3` – przełączanie slotów HUD.

---

## Dodawanie nowych paneli

1. Utwórz plik w `glitchlab/gui/panels/` nazwany:

   * `panel_nazwafiltra.py` **lub** `nazwafiltra_panel.py`.

2. Zdefiniuj klasę `Panel` (lub inną kończącą się na `Panel`).

3. Obsłuż kontekst `PanelContext` (`ctx.defaults`, `ctx.params`, `ctx.on_change`).

4. Uruchom aplikację i kliknij `Reload`, aby wczytać nowy panel bez restartu.

> Jeśli brak panelu → GUI sam wygeneruje formularz (`ParamForm`).

---

## Diagnostyka i testowanie

* **DiagConsole:** kanał logów `diag.log` (DEBUG/INFO/WARN/ERROR).
* **Probe:** sprawdza dostępność panelu, loguje etapy importu.
* **Reload:** ponownie ładuje panel (hot-reload).
* **Smoke test GUI:** uruchom `App` z testowym obrazem numpy.
* **Plansza testowa:** `glitchlab_testchart_v1.png` do wizualnego sprawdzania jakości filtrów.

---

## Changelog v4.5

* Automatyczny import paneli (`panel_*.py`, `*_panel.py`).
* Nowe narzędzia diagnostyczne: `Probe`, `Reload`.
* Ulepszony fallback `ParamForm` (obsługa sygnatur funkcji).
* Spójne logowanie i komunikaty po polsku.
* Uszczelniony kontrakt GUI ↔ Core (obrazy `np.ndarray uint8 RGB`).

---

## Checklist wdrożeniowy

* [ ] Auto-import paneli działa (`Rescan`).
* [ ] TabFilter poprawnie ładuje listę filtrów.
* [ ] `Probe` i `Reload` działają i logują wyniki.
* [ ] ParamForm generuje formularze przy braku panelu.
* [ ] HUD poprawnie czyta i pokazuje dane z cache.
* [ ] Obsługa błędów pipeline działa (brak crashów).

---

## Licencja

Projekt GlitchLab GUI v4.5 jest rozwijany jako **open-source**.
Licencja: [MIT](LICENSE).


