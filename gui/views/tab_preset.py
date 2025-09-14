# glitchlab/gui/views/tab_presets.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Dict, List, Optional, Callable

# --- rejestr filtrów (do listy w PresetManager) -----------------------------
try:
    from glitchlab.core.registry import available as registry_available  # type: ignore
except Exception:  # pragma: no cover
    def registry_available() -> List[str]:
        return []

# --- widget menedżera presetów ---------------------------------------------
try:
    from glitchlab.gui.widgets.preset_manager import PresetManager  # type: ignore
except Exception:  # pragma: no cover
    PresetManager = None  # type: ignore


class PresetsTabConfig:
    """Opcje konfiguracyjne dla zakładki Presets."""

    def __init__(self, allow_change_dir: bool = True, allow_save: bool = True) -> None:
        self.allow_change_dir = allow_change_dir
        self.allow_save = allow_save


class PresetsTab(ttk.Frame):
    """
    Zakładka „Presets”:
      - Pełny PresetManager jeśli dostępny
      - Fallback (Load/Save/Apply, tekstowy podgląd kroków)
      - Integracja z EventBus
    """

    def __init__(
            self,
            master: tk.Misc,
            *,
            bus: Optional[Any] = None,
            cfg: Optional[PresetsTabConfig] = None,
            on_apply: Optional[Callable[[Dict[str, Any]], None]] = None,
            get_current_step: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(master)
        self.bus = bus
        self.cfg = cfg or PresetsTabConfig()
        self._on_apply = on_apply
        self._get_current_step = get_current_step

        # aktualna konfiguracja presetów
        self._cfg: Dict[str, Any] = {"version": 2, "steps": [], "__preset_dir": ""}
        self._preset_dir = Path(os.getcwd())
        self._mgr_widget: Optional[tk.Widget] = None

        # --- UI
        self._build_ui()
        self._build_manager_or_fallback()

    # ---------------------------- Public API -----------------------------

    def set_cfg(self, cfg: Dict[str, Any]) -> None:
        self._cfg = dict(cfg or {})
        self._preset_dir = Path(self._cfg.get("__preset_dir") or os.getcwd())
        self._sync_dir_label()
        self.refresh()

    def get_cfg(self) -> Dict[str, Any]:
        self._cfg["__preset_dir"] = str(self._preset_dir)
        return dict(self._cfg)

    def set_preset_dir(self, path: str | Path) -> None:
        self._preset_dir = Path(path)
        self._cfg["__preset_dir"] = str(self._preset_dir)
        self._sync_dir_label()
        self.refresh()

    def refresh(self) -> None:
        if self._mgr_widget and hasattr(self._mgr_widget, "refresh"):
            try:
                self._mgr_widget.refresh()
            except Exception:
                pass
        elif not self._mgr_widget:
            self._rebuild_fallback()

    def preview(self, path: str) -> None:
        """Podgląd YAML w okienku (tylko fallback)."""
        if not path or not os.path.exists(path):
            return
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            messagebox.showinfo("Preset preview", str(cfg)[:2000])
        except Exception as e:
            messagebox.showerror("Preview error", str(e))

    # ------------------------------ UI ----------------------------------

    def _build_ui(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(6, 2))
        if self.cfg.allow_change_dir:
            ttk.Button(bar, text="Change…", command=self._choose_dir).pack(side="left", padx=(6, 0))
        self._lbl_dir = ttk.Label(bar, text="Preset folder: …")
        self._lbl_dir.pack(side="left", padx=(6, 0))
        self._host = ttk.Frame(self)
        self._host.pack(fill="both", expand=True)

    def _choose_dir(self) -> None:
        d = filedialog.askdirectory(
            title="Choose preset folder",
            initialdir=str(self._preset_dir or os.getcwd()),
        )
        if d:
            self.set_preset_dir(d)

    def _sync_dir_label(self) -> None:
        p = str(self._preset_dir)
        self._lbl_dir.config(text=f"Preset folder: {p if len(p) <= 80 else '…' + p[-78:]}")

    # --------------------- Manager or fallback ---------------------------

    def _build_manager_or_fallback(self) -> None:
        for w in self._host.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        self._mgr_widget = None

        if PresetManager is None:
            self._rebuild_fallback()
            return

        try:
            mgr = PresetManager(
                self._host,
                lambda: self.get_cfg(),
                lambda c: self.set_cfg(c),
                lambda: self._apply_clicked(),
                lambda: sorted(registry_available()),
                self._get_current_step or (lambda: {}),
            )
            mgr.pack(fill="both", expand=True)
            self._mgr_widget = mgr
            self.refresh()
        except Exception as e:
            print(f"[presets-tab] PresetManager init failed: {e}")
            self._rebuild_fallback()

    def _apply_clicked(self) -> None:
        cfg = self.get_cfg()
        if self._on_apply:
            try:
                self._on_apply(cfg)
                return
            except Exception:
                pass
        self._publish("ui.run.apply_preset", {"cfg": cfg})

    # ------------------------- Fallback mode -----------------------------

    def _rebuild_fallback(self) -> None:
        frame = ttk.Frame(self._host)
        frame.pack(fill="both", expand=True)

        info = ttk.Label(
            frame,
            text="(PresetManager unavailable — fallback)\n• Load preset (YAML)\n• Apply\n• Save As…",
            justify="left",
        )
        info.pack(anchor="w", padx=8, pady=(8, 4))

        btns = ttk.Frame(frame)
        btns.pack(anchor="w", padx=8, pady=(0, 8))
        ttk.Button(btns, text="Load…", command=self._fallback_load).pack(side="left")
        ttk.Button(btns, text="Apply", command=self._fallback_apply).pack(side="left", padx=(6, 0))
        if self.cfg.allow_save:
            ttk.Button(btns, text="Save As…", command=self._fallback_save).pack(side="left", padx=(6, 0))

        self._fallback_steps = tk.Text(frame, height=10, width=64)
        self._fallback_steps.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._fallback_refresh_text()

        self._mgr_widget = None

    def _fallback_refresh_text(self) -> None:
        try:
            self._fallback_steps.config(state="normal")
            self._fallback_steps.delete("1.0", "end")
            steps = self._cfg.get("steps") or []
            self._fallback_steps.insert("1.0", f"Preset dir: {self._preset_dir}\nSteps:\n")
            for i, step in enumerate(steps):
                self._fallback_steps.insert("end", f"  {i + 1}. {step.get('name')} {step.get('params', {})}\n")
            self._fallback_steps.config(state="disabled")
        except Exception:
            pass

    def _fallback_load(self) -> None:
        fn = filedialog.askopenfilename(
            title="Open Preset YAML",
            filetypes=[("YAML", "*.yml;*.yaml"), ("All files", "*.*")],
            initialdir=str(self._preset_dir or os.getcwd()),
        )
        if not fn:
            return
        try:
            import yaml
            with open(fn, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if not isinstance(cfg, dict):
                raise ValueError("Malformed preset (not a dict).")
            self.set_cfg(cfg)
            self._fallback_refresh_text()
        except Exception as e:
            messagebox.showerror("Load preset", str(e))

    def _fallback_save(self) -> None:
        fn = filedialog.asksaveasfilename(
            title="Save Preset As",
            defaultextension=".yml",
            filetypes=[("YAML", "*.yml;*.yaml")],
            initialdir=str(self._preset_dir or os.getcwd()),
        )
        if not fn:
            return
        try:
            import yaml
            with open(fn, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.get_cfg(), f, sort_keys=False, allow_unicode=True)
        except Exception as e:
            messagebox.showerror("Save preset", str(e))

    def _fallback_apply(self) -> None:
        cfg = self.get_cfg()
        if self._on_apply:
            try:
                self._on_apply(cfg)
                return
            except Exception:
                pass
        self._publish("ui.run.apply_preset", {"cfg": cfg})

    # ------------------------------ Utils --------------------------------

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self.bus is not None and hasattr(self.bus, "publish"):
            try:
                self.bus.publish(topic, dict(payload))
            except Exception:
                pass
