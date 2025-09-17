# GlitchLab GUI v4.5 – ARCHITECTURE.md

**Autorzy i licencja:** Open Source — D2J3 aka Cha0s (for test and fun)

---

## Spis treści

1. [Overview](#1-overview)  
2. [EventBus i AppState](#2-eventbus-i-appstate)  
3. [Układ GUI](#3-układ-gui)  
4. [Moduły GUI](#4-moduły-gui)  
5. [Fallback i odporność na błędy](#5-fallback-i-odporność-na-błędy)  
6. [Integracja z Core](#6-integracja-z-core)  
7. [Status wdrożenia](#7-status-wdrożenia)  

---

## 1. Overview

- Stabilny układ: **canvas (lewa strona)**, **zakładki paneli filtrów (prawa strona)**, **HUD/telemetria (dół)**.  
- Architektura zgodna z modelem *Core ↔ GUI ↔ HUD*.  
- GUI pełni rolę cienkiego klienta – wszystkie obliczenia realizuje Core.  
- Komunikacja przez kanały HUD w `ctx.cache`.  

---

## 2. EventBus i AppState

- **AppState** – centralny obiekt stanu (image, preset, cache, masks).  
- **EventBus** – lekki system zdarzeń (publish/subscribe), zapewniający luźne powiązania między komponentami.  

### Kluczowe zdarzenia

- `ui.filter.select`  
- `ui.filter.params_changed`  
- `ui.run.apply_filter`  
- `ui.run.finished`  
- `ui.presets.save_request`  
- `diag.log`  

---

## 3. Układ GUI

- **Viewport** – obszar canvas (`ImageCanvas` + `CanvasContainer`).  
- **Tabs** – trzy zakładki:  
  - `Filters` (wybór i parametry filtra)  
  - `General` (ustawienia globalne, ROI, amplitude)  
  - `Presets` (zapisywanie/ładowanie, historia kroków)  
- **HUD** – trzy sloty na dane diagnostyczne (obrazy, metryki, graf AST, mozaiki).  
- **Menu + Statusbar** – standardowe elementy aplikacji.  

---

## 4. Moduły GUI

- `app.py` – klasa App, inicjalizacja i główna pętla Tk.  
- `docking.py` – DockManager (dock/undock paneli).  
- `panel_loader.py` – dynamiczne ładowanie paneli filtrów.  
- `panel_base.py` – klasy bazowe i kontekst paneli.  
- `views/` – widoki globalne (zakładki, viewport, hud, menu, statusbar).  
- `widgets/` – niskopoziomowe komponenty (ImageCanvas, CanvasContainer, HUD slot, ParamForm, DiagConsole).  
- `panels/` – dedykowane panele filtrów (`panel_<filter>.py`).  

---

## 5. Fallback i odporność na błędy

- Brak panelu filtra → użycie **ParamForm** (formularz awaryjny).  
- Błąd w kodzie panelu → aplikacja działa dalej, panel zastępowany fallbackiem.  
- Błąd w pipeline Core → komunikat błędu, aplikacja nie przerywa działania.  

---

## 6. Integracja z Core

- GUI wywołuje `core.pipeline.apply_pipeline` poprzez **PipelineRunner**.  
- Wyniki (obrazy, metryki, AST, mozaiki) → trafiają do `AppState.cache`.  
- HUD + viewport → renderują tylko dane z cache (GUI nie liczy nic samodzielnie).  

---

## 7. Status wdrożenia

Wszystkie opisane komponenty są zaimplementowane zgodnie z architekturą.  
Elementy w trakcie implementacji oznaczone są jako **on-run**:  

- DockManager z pełnym zapisem/odtwarzaniem pozycji paneli między sesjami — **on-run**.  
- Automatyczny hot-reload paneli filtrów (bez restartu aplikacji) — **on-run**.  
- Rozszerzone widoki HUD (`MosaicView`, `GraphView` jako interaktywne) — **on-run**.  
- Integracja GUI z usługami history/undo — **on-run**.  

---
