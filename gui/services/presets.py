# glitchlab/gui/services/presets.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import tkinter as tk
from tkinter import filedialog, messagebox

# --- opcjonalne zależności ---
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# --- core normalize (opcjonalne) ---
try:
    from glitchlab.core.pipeline import normalize_preset  # type: ignore
except Exception:
    def normalize_preset(cfg: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        return cfg


@dataclass
class PresetServiceConfig:
    """
    Konfiguracja serwisu presetów.
    - initial_dir: jeśli None, wykryjemy automatycznie katalog 'glitchlab/presets'
    - default_name: sugerowana nazwa pliku przy zapisie
    """
    initial_dir: Optional[str] = None
    default_name: str = "preset.yaml"


def _detect_presets_dir(user_hint: Optional[str] = None) -> str:
    """
    Znajdź katalog 'glitchlab/presets':
      1) jawny 'user_hint' jeśli istnieje,
      2) lokalizacja modułu glitchlab → '../presets',
      3) przeszukanie w górę względem bieżącego pliku,
      4) cwd/presets,
      5) HOME/presets (ostatni fallback).
    """
    # 1) jawny hint użytkownika
    if user_hint and os.path.isdir(user_hint):
        return os.path.abspath(user_hint)

    # 2) moduł glitchlab
    try:
        import glitchlab  # type: ignore
        base = os.path.dirname(os.path.abspath(glitchlab.__file__))  # type: ignore[attr-defined]
        p = os.path.join(base, "presets")
        if os.path.isdir(p):
            return p
    except Exception:
        pass

    # 3) przeszukaj do góry względem tego pliku
    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    for _ in range(6):
        cand = os.path.abspath(os.path.join(cur, "..", "..", "presets"))
        if os.path.isdir(cand):
            return cand
        nxt = os.path.abspath(os.path.join(cur, ".."))
        if nxt == cur:
            break
        cur = nxt

    # 4) cwd/presets
    cand = os.path.join(os.getcwd(), "presets")
    if os.path.isdir(cand):
        return cand

    # 5) HOME/presets
    home = os.path.expanduser("~")
    cand = os.path.join(home, "presets")
    return cand if os.path.isdir(cand) else home


class PresetService:
    """
    Serwis do ładowania/zapisu presetów (YAML / JSON).

    Integracja z EventBus (best-effort; zdarzenia opcjonalne):
      • subscribe:
          - 'ui.preset.open'             -> pokaż open dialog
          - 'ui.preset.save'             -> pokaż save dialog (payload: {'cfg': dict})
          - 'image.loaded'               -> podpowiedz katalog (payload: {'path': str})
          - 'ui.image.loaded'            -> jw.
          - 'ui.image.path'              -> jw.
      • publish:
          - 'preset.loaded'              -> {'path','text'}
          - 'preset.parsed'              -> {'path','cfg'}
          - 'diag.log'                   -> {'level','msg'}
    """

    def __init__(self, master: tk.Misc, bus: Any, cfg: Optional[PresetServiceConfig] = None) -> None:
        self.master = master
        self.bus = bus
        self.cfg = cfg or PresetServiceConfig()

        # wykryj initial_dir (uwzględnij ew. hint użytkownika z cfg)
        self.initial_dir = _detect_presets_dir(self.cfg.initial_dir)

        # ostatnie katalogi dla dialogów
        self._last_dir_open: str = self.initial_dir
        self._last_dir_save: str = self.initial_dir

        # katalog obrazu (jeśli wiemy — do powiązania presetów kontekstowo)
        self._image_dir: Optional[str] = None

        self._wire_bus()

    # ------------------------------------------------------------------ services

    def _wire_bus(self) -> None:
        if not hasattr(self.bus, "subscribe"):
            return
        try:
            self.bus.subscribe("ui.preset.open", lambda _t, _d: self.open_dialog())
            self.bus.subscribe("ui.preset.save", lambda _t, d: self.save_dialog((d or {}).get("cfg") or {}))

            # hint katalogu z wydarzeń obrazu (dowolna z nazw)
            img_dir_cb = lambda _t, d: self.set_hint_dir(os.path.dirname(str(d.get("path", ""))) if d else None)
            self.bus.subscribe("image.loaded", img_dir_cb)
            self.bus.subscribe("ui.image.loaded", img_dir_cb)
            self.bus.subscribe("ui.image.path", img_dir_cb)
        except Exception:
            pass

    # ------------------------------------------------------------------ public helpers

    def set_hint_dir(self, path: Optional[str]) -> None:
        """Ustaw domyślny katalog (np. katalog bieżącego obrazu)."""
        if not path:
            return
        try:
            if os.path.isdir(path):
                self._image_dir = os.path.abspath(path)
        except Exception:
            pass

    # ------------------------------------------------------------------ UI actions

    def _default_open_dir(self) -> str:
        # preferuj katalog obrazu, potem ostatnio użyty, potem initial_dir
        return self._image_dir or self._last_dir_open or self.initial_dir

    def _default_save_dir(self) -> str:
        return self._image_dir or self._last_dir_save or self.initial_dir

    def open_dialog(self) -> None:
        """Pokaż dialog otwarcia, wczytaj preset, opublikuj 'preset.loaded' i 'preset.parsed'."""
        types = [
            ("YAML", "*.yaml;*.yml"),
            ("JSON", "*.json"),
            ("All files", "*.*"),
        ]
        try:
            path = filedialog.askopenfilename(
                title="Open preset",
                initialdir=self._default_open_dir(),
                filetypes=types,
                parent=self.master if isinstance(self.master, (tk.Tk, tk.Toplevel)) else None,
            )
        except Exception as ex:
            messagebox.showerror("Open preset", str(ex))
            return

        if not path:
            return

        self._last_dir_open = os.path.dirname(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as ex:
            messagebox.showerror("Open preset", str(ex))
            return

        # emit preset.loaded
        self._publish("preset.loaded", {"path": path, "text": text})

        # parse + normalize
        cfg = self._parse_preset_text(text)
        if not isinstance(cfg, dict):
            messagebox.showwarning("Preset", "Unrecognized preset format.")
            return

        try:
            cfg = normalize_preset(cfg)
        except Exception:
            # jeśli normalizacja zawiedzie – mimo wszystko wyślij surowy cfg
            pass

        self._publish("preset.parsed", {"path": path, "cfg": cfg})

    def save_dialog(self, cfg: Dict[str, Any]) -> None:
        """Zapisz preset (YAML preferowane, fallback JSON)."""
        if not isinstance(cfg, dict) or not cfg:
            messagebox.showwarning("Save preset", "No configuration to save.")
            return

        # domyślna nazwa
        default = self.cfg.default_name or "preset.yaml"
        if not default.lower().endswith((".yaml", ".yml", ".json")):
            default += ".yaml"

        try:
            path = filedialog.asksaveasfilename(
                title="Save preset as…",
                defaultextension=(".yaml", ".yml"),
                filetypes=[("YAML", "*.yaml;*.yml"), ("JSON", "*.json"), ("All files", "*.*")],
                initialdir=self._default_save_dir(),
                initialfile=default,
                parent=self.master if isinstance(self.master, (tk.Tk, tk.Toplevel)) else None,
            )
        except Exception as ex:
            messagebox.showerror("Save preset", str(ex))
            return

        if not path:
            return

        self._last_dir_save = os.path.dirname(path)

        # serializacja
        text: str
        try:
            if path.lower().endswith((".yaml", ".yml")) and yaml is not None:
                text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)  # type: ignore[attr-defined]
            else:
                text = json.dumps(cfg, ensure_ascii=False, indent=2)
        except Exception as ex:
            messagebox.showerror("Save preset", f"Serialize error: {ex}")
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as ex:
            messagebox.showerror("Save preset", str(ex))
            return

        self._publish("diag.log", {"level": "OK", "msg": f"Preset saved: {path}"})

    # ------------------------------------------------------------------ parsing

    def _parse_preset_text(self, text: str) -> Dict[str, Any] | Any:
        text = text.strip()
        if not text:
            return {}

        # Prefer YAML
        if yaml is not None:
            try:
                data = yaml.safe_load(text)  # type: ignore[attr-defined]
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        # JSON fallback
        try:
            data = json.loads(text)
            return data
        except Exception:
            pass

        return {}

    # ------------------------------------------------------------------ helpers

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if hasattr(self.bus, "publish"):
            try:
                self.bus.publish(topic, dict(payload))
            except Exception:
                pass
