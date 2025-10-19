"""
---
version: 3
kind: module
id: "app-services-layout"
created_at: "2025-09-13"
name: "glitchlab.app.services.layout"
author: "GlitchLab v3"
role: "Persistencja layoutu GUI (DockManager bridge)"
description: >
Zapis/odczyt ustawień interfejsu: geometria okna, pozycje splitterów, sloty
dokowane/pływające wraz z ich rozmiarami i przypisaniami HUD.
inputs:
save.path: {type: "str"}
save.state: {type: "dict", keys: ["geometry","splitters","floating","hud_mapping"]}
load.path: {type: "str"}
outputs:
state: {type: "dict", keys: ["geometry","splitters","floating","hud_mapping"]}
interfaces:
exports: ["LayoutService"]
depends_on: ["json","os","pathlib","typing"]
used_by: ["glitchlab.app.app","glitchlab.app.docking","glitchlab.app.exporters"]
policy:
deterministic: true
side_effects: ["filesystem"]
constraints:

"Format JSON kompatybilny między wersjami UI (best-effort)"

"Brak zależności od Tkinter w warstwie zapisu/odczytu"
telemetry:
counters: ["saves","loads","errors"]
last_error: {type: "str|null"}
license: "Proprietary"
---
"""
# glitchlab/app/services/layout.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
import json
import os
import platform
import time
from pathlib import Path

# Types for call-ins
DockSaveFn = Callable[[], Dict[str, Any]]
DockLoadFn = Callable[[Dict[str, Any]], None]
GetMappingFn = Callable[[], Dict[str, Any]]
SetMappingFn = Callable[[Dict[str, Any]], None]


def _default_config_dir(app_name: str = "GlitchLab") -> Path:
    sys = platform.system()
    if sys == "Windows":
        base = os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
        return Path(base) / app_name
    if sys == "Darwin":
        return Path.home() / "Library" / "Application Support" / app_name
    # Linux/other *nix
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / app_name.lower()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
        return True
    except Exception:
        return False


def _parse_geometry(geom: str) -> Dict[str, int]:
    # Expects "WxH+X+Y"
    try:
        size, pos = geom.split("+", 1)
        w, h = size.split("x", 1)
        x_str, y_str = pos.split("+", 1)
        return {"w": int(w), "h": int(h), "x": int(x_str), "y": int(y_str)}
    except Exception:
        return {}


def _mk_geometry(g: Dict[str, int]) -> str:
    w = int(g.get("w", 1200))
    h = int(g.get("h", 800))
    x = int(g.get("x", 50))
    y = int(g.get("y", 50))
    return f"{w}x{h}+{x}+{y}"


def _clamp_geometry_to_screen(g: Dict[str, int], scr_w: int, scr_h: int) -> Dict[str, int]:
    out = dict(g)
    w = max(320, min(int(out.get("w", 1200)), scr_w))
    h = max(240, min(int(out.get("h", 800)), scr_h))
    x = int(out.get("x", 50))
    y = int(out.get("y", 50))
    # Ensure top-left corner is visible (allow partial offscreen for large windows)
    x = max(-w + 100, min(x, scr_w - 100))
    y = max(-h + 100, min(y, scr_h - 100))
    out.update({"w": w, "h": h, "x": x, "y": y})
    return out


@dataclass
class LayoutService:
    """
    Persist and restore GUI layout:
      - main window geometry/state,
      - DockManager layout blob (dock/float slots),
      - HUD mapping (slot -> keys),
      - optional 'recent' section (files, presets_dir).
    """
    root_like: Any
    dock_save: Optional[DockSaveFn] = None
    dock_load: Optional[DockLoadFn] = None
    get_hud_mapping: Optional[GetMappingFn] = None
    set_hud_mapping: Optional[SetMappingFn] = None
    app_name: str = "GlitchLab"
    filename: str = "ui.json"
    version: int = 3
    path: Optional[Path] = None
    _last_saved_blob: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------- paths ---------------------------------

    def config_path(self) -> Path:
        return self.path or (_default_config_dir(self.app_name) / self.filename)

    # ---------------------------- capture/apply --------------------------

    def capture(self, *, include_recent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        root = self.root_like
        # window geometry & state
        try:
            geom = _parse_geometry(root.winfo_geometry())
        except Exception:
            geom = {}
        state = "normal"
        try:
            # Tk reports 'zoomed' on Windows; 'normal'/'iconic'
            st = str(getattr(root, "state", lambda: "normal")())
            state = "maximized" if st == "zoomed" else st
        except Exception:
            pass

        # dock layout
        dock_blob: Dict[str, Any] = {}
        if self.dock_save is not None:
            try:
                dock_blob = dict(self.dock_save())
            except Exception:
                dock_blob = {}

        # hud mapping
        hud: Dict[str, Any] = {}
        if self.get_hud_mapping is not None:
            try:
                hud["mapping"] = dict(self.get_hud_mapping())
            except Exception:
                hud["mapping"] = {}

        # screen info
        try:
            scr_w = int(root.winfo_screenwidth())
            scr_h = int(root.winfo_screenheight())
        except Exception:
            scr_w, scr_h = 1920, 1080

        blob: Dict[str, Any] = {
            "version": self.version,
            "saved_at": time.time(),
            "screen": {"w": scr_w, "h": scr_h},
            "window": {"geometry": geom, "state": state},
            "dock": dock_blob,
            "hud": hud,
        }
        if include_recent:
            blob["recent"] = dict(include_recent)
        return blob

    def apply(self, data: Dict[str, Any]) -> None:
        root = self.root_like
        # geometry
        try:
            scr_w = int(root.winfo_screenwidth())
            scr_h = int(root.winfo_screenheight())
        except Exception:
            scr_w, scr_h = 1920, 1080

        geom = data.get("window", {}).get("geometry") or {}
        if geom:
            g = _clamp_geometry_to_screen(geom, scr_w, scr_h)
            try:
                root.geometry(_mk_geometry(g))
            except Exception:
                pass
        # state
        state = (data.get("window", {}).get("state") or "normal").lower()
        try:
            if state in ("zoomed", "maximized"):
                # Windows: 'zoomed'; others: try 'zoomed' as well
                root.state("zoomed")
            elif state == "iconic":
                root.iconify()
            else:
                root.state("normal")
        except Exception:
            pass

        # dock
        dock_blob = data.get("dock") or {}
        if self.dock_load is not None and dock_blob:
            try:
                self.dock_load(dock_blob)
            except Exception:
                pass

        # hud mapping
        hud = data.get("hud") or {}
        mapping = hud.get("mapping")
        if mapping and self.set_hud_mapping is not None:
            try:
                self.set_hud_mapping(mapping)
            except Exception:
                pass

    # ---------------------------- load/save --------------------------------

    def load(self) -> Dict[str, Any]:
        cfg_path = self.config_path()
        data = _read_json(cfg_path)
        if not data:
            return {}
        # Minimal version check; allow forward-compat by ignoring unknown fields
        if int(data.get("version", self.version)) <= 0:
            return {}
        self.apply(data)
        return data

    def save(self, *, include_recent: Optional[Dict[str, Any]] = None, force: bool = False) -> bool:
        blob = self.capture(include_recent=include_recent)
        # Avoid frequent identical writes
        if not force and blob == self._last_saved_blob:
            return True
        ok = _write_json_atomic(self.config_path(), blob)
        if ok:
            self._last_saved_blob = blob
        return ok

    # ---------------------------- helpers for App -------------------------

    def wire_dock_manager(self, dock_manager: Any) -> None:
        """
        Convenience: infer save/load call-ins from a DockManager-like object.
        Expects methods: save_layout() -> dict, load_layout(dict) -> None
        """
        if hasattr(dock_manager, "save_layout") and callable(dock_manager.save_layout):
            self.dock_save = dock_manager.save_layout  # type: ignore[assignment]
        if hasattr(dock_manager, "load_layout") and callable(dock_manager.load_layout):
            self.dock_load = dock_manager.load_layout  # type: ignore[assignment]

    def wire_hud_mapping(
        self,
        *,
        getter: Optional[GetMappingFn] = None,
        setter: Optional[SetMappingFn] = None,
    ) -> None:
        if getter is not None:
            self.get_hud_mapping = getter
        if setter is not None:
            self.set_hud_mapping = setter
