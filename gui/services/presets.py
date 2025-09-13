"""
---
version: 3
kind: module
id: "gui-services-presets"
created_at: "2025-09-13"
name: "glitchlab.gui.services.presets"
author: "GlitchLab v3"
role: "Preset I/O + parsing (YAML/JSON) i lekka walidacja; integracja z EventBus"
description: >
  Serwis do obsługi presetów: otwieranie/zapisywanie plików, parsowanie YAML/JSON,
  normalizacja schematu (przez core.normalize_preset, jeśli dostępne) i publikacja
  zdarzeń dla UI. Nie wykonuje pipeline’u – tylko przygotowuje cfg i statusy.

inputs:
  ui.presets.open_request: {payload: {}, effect: "file dialog → preset.loaded/preset.status"}
  ui.presets.save_request: {payload: {text:str}, effect: "file dialog → zapis tekstu"}
  ui.presets.apply:        {payload: {text:str}, effect: "parse+normalize → preset.parsed/preset.status"}

outputs:
  preset.loaded: {payload: {text:str}}
  preset.parsed: {payload: {cfg:dict}}
  preset.status: {payload: {message:str}}

interfaces:
  exports:
    - "parse_preset_text(text:str) -> dict"
    - "load_preset_file(path:PathLike) -> tuple[str, dict]"
    - "save_preset_text(text:str, path:PathLike) -> None"
    - "PresetService(root_like, bus).(auto-subscribe ui.presets.*)"

depends_on: ["json", "yaml? (opcjonalnie)", "pathlib", "glitchlab.core.pipeline.normalize_preset?"]
used_by: ["glitchlab.gui.app", "glitchlab.gui.views.notebook"]
policy:
  deterministic: true
  ui_thread_only: true
constraints:
  - "brak I/O sieciowego; wyłącznie file dialogs/dysk"
  - "brak side-effectów poza dyskiem i EventBus"
license: "Proprietary"
---
"""
# glitchlab/gui/services/presets.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable

import json

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # YAML opcjonalny


# --- interfejs core'owy, miękkie zależności ---
try:
    from glitchlab.core.pipeline import normalize_preset  # type: ignore
except Exception:
    def normalize_preset(cfg: Dict[str, Any]) -> Dict[str, Any]:
        # fallback: akceptuj słownik i dopilnuj podstaw
        if not isinstance(cfg, dict):
            raise ValueError("preset must be a dict")
        cfg = dict(cfg)
        cfg.setdefault("version", 2)
        cfg.setdefault("steps", [])
        return cfg


def parse_preset_text(text: str) -> Dict[str, Any]:
    """
    Próbuje YAML, potem JSON. Zwraca już znormalizowany preset (core.normalize_preset jeśli dostępny).
    """
    last_err: Optional[Exception] = None
    if yaml is not None:
        try:
            cfg = yaml.safe_load(text)
            return normalize_preset(cfg or {})
        except Exception as e:
            last_err = e
    try:
        cfg = json.loads(text)
        return normalize_preset(cfg or {})
    except Exception as e2:
        last_err = e2
    raise ValueError(f"Nie udało się sparsować preset-u (YAML/JSON). Ostatni błąd: {last_err}")


def load_preset_file(path: str | Path) -> Tuple[str, Dict[str, Any]]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    cfg = parse_preset_text(txt)
    return txt, cfg


def save_preset_text(text: str, path: str | Path) -> None:
    Path(path).write_text(text, encoding="utf-8")


# Opcjonalna warstwa serwisowa na EventBus (UI inicjuje „ui.presets.*”)
@dataclass
class PresetService:
    root_like: Any
    bus: Any  # EventBus-like

    def __post_init__(self) -> None:
        # Nasłuchujemy na zdarzenia z notebooka
        try:
            self.bus.subscribe("ui.presets.open_request", self._on_open_request, on_ui=True)
            self.bus.subscribe("ui.presets.save_request", self._on_save_request, on_ui=True)
            self.bus.subscribe("ui.presets.apply", self._on_apply, on_ui=True)
        except Exception:
            pass

    # Handlers

    def _on_open_request(self, _t: str, _d: Dict[str, Any]) -> None:
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(
            title="Open preset",
            filetypes=[("YAML/JSON", "*.yaml;*.yml;*.json;*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            txt, _cfg = load_preset_file(path)
            self.bus.publish("preset.loaded", {"text": txt})
            self.bus.publish("preset.status", {"message": f"Loaded: {path}"})
        except Exception as ex:
            messagebox.showerror("Preset", str(ex))

    def _on_save_request(self, _t: str, data: Dict[str, Any]) -> None:
        from tkinter import filedialog, messagebox
        text = (data or {}).get("text") or ""
        path = filedialog.asksaveasfilename(
            title="Save preset as",
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml;*.yml"), ("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            save_preset_text(str(text), path)
            self.bus.publish("preset.status", {"message": f"Saved: {path}"})
        except Exception as ex:
            messagebox.showerror("Preset", str(ex))

    def _on_apply(self, _t: str, data: Dict[str, Any]) -> None:
        """
        Tu tylko walidujemy i publikujemy znormalizowany preset.
        Samo wykonanie kroków robi AppShell (albo osobny RunService).
        """
        from tkinter import messagebox
        text = (data or {}).get("text") or ""
        try:
            cfg = parse_preset_text(text)
            # downstream: app shell/subskrybent decyduje jak wykonać kroki
            self.bus.publish("preset.parsed", {"cfg": cfg})
            self.bus.publish("preset.status", {"message": "Preset OK"})
        except Exception as ex:
            messagebox.showerror("Preset", f"Preset nieprawidłowy:\n{ex}")
