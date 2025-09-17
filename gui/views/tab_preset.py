# glitchlab/gui/views/tab_preset.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Dict, List, Optional, Callable, Iterable

# ──────────────────────────────────────────────────────────────────────────────
# Opcjonalne zależności / helpery
# ──────────────────────────────────────────────────────────────────────────────
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# rejestr filtrów (do listy w menedżerze – best effort)
try:
    from glitchlab.core.registry import available as registry_available  # type: ignore
except Exception:  # pragma: no cover
    def registry_available() -> List[str]:
        return []

# Preferuj wspólną detekcję katalogu presetów z serwisu
try:
    from glitchlab.gui.services.presets import _detect_presets_dir  # type: ignore
except Exception:
    def _detect_presets_dir(_hint: Optional[str] = None) -> str:
        here = Path(__file__).resolve()
        for p in (
            here.parents[2] / "presets",
            Path.cwd() / "presets",
            Path.home() / "presets",
            Path.cwd(),
        ):
            if p.exists():
                return str(p)
        return str(Path.cwd())

# Opcjonalny widget – jeśli istnieje, użyjemy pełnego menedżera
try:
    from glitchlab.gui.widgets.preset_manager import PresetManager  # type: ignore
except Exception:  # pragma: no cover
    PresetManager = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Konfiguracja
# ──────────────────────────────────────────────────────────────────────────────
class PresetsTabConfig:
    """Opcje konfiguracyjne dla zakładki Presets."""

    def __init__(
        self,
        allow_change_dir: bool = True,
        allow_save: bool = True,
        remember_last: bool = True,
    ) -> None:
        self.allow_change_dir = allow_change_dir
        self.allow_save = allow_save
        self.remember_last = remember_last


# ──────────────────────────────────────────────────────────────────────────────
# Zakładka Presets
# ──────────────────────────────────────────────────────────────────────────────
class PresetsTab(ttk.Frame):
    """
    Zakładka „Presets”:
      • Pełny PresetManager (jeśli dostępny) lub fallback (Load/Apply/Save)
      • Prawidłowe ścieżki (domyślnie: glitchlab/presets)
      • Zapamiętywanie ostatniej lokalizacji
      • Integracja z EventBus:
         - publikuje: ui.preset.open / ui.preset.save / ui.run.apply_preset
         - odbiera:   run.start / run.done / run.error
                      + (opcjonalnie) preset.loaded / preset.parsed / ui.image.loaded
      • Auto-disable kontrolek na czas wykonywania.
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

        # Bieżący preset oraz katalog
        self._cfg: Dict[str, Any] = {"version": 2, "steps": [], "__preset_dir": ""}
        self._preset_dir = Path(_detect_presets_dir())
        self._last_dir: Path = self._preset_dir

        # UI host
        self._mgr_widget: Optional[tk.Widget] = None
        self._fallback_steps: Optional[tk.Text] = None

        # zebrane kontrolki do auto-disable
        self._disable_nodes: list[tk.Widget] = []

        # UI
        self._build_ui()
        self._build_manager_or_fallback()

        # Bus (opcjonalnie)
        self._wire_bus()

    # ─────────────────────────── Public API ─────────────────────────────

    def set_cfg(self, cfg: Dict[str, Any]) -> None:
        self._cfg = dict(cfg or {})
        preset_dir = self._cfg.get("__preset_dir")
        if preset_dir:
            self._preset_dir = Path(preset_dir)
            self._last_dir = self._preset_dir
        self._sync_dir_label()
        self.refresh()

    def get_cfg(self) -> Dict[str, Any]:
        c = dict(self._cfg or {})
        c["__preset_dir"] = str(self._preset_dir)
        return c

    def set_preset_dir(self, path: str | Path) -> None:
        p = Path(path)
        if p.exists():
            self._preset_dir = p
            self._last_dir = p
            self._cfg["__preset_dir"] = str(p)
            self._sync_dir_label()
            self.refresh()

    def set_image_path(self, image_path: str | Path) -> None:
        try:
            img = Path(image_path).resolve()
            if not img.exists():
                return
            repo_presets = Path(__file__).resolve().parents[2] / "presets"
            if repo_presets.exists():
                self.set_preset_dir(repo_presets)
                return
            near = img.parent / "presets"
            if near.exists():
                self.set_preset_dir(near)
        except Exception:
            pass

    def refresh(self) -> None:
        if self._mgr_widget and hasattr(self._mgr_widget, "refresh"):
            try:
                self._mgr_widget.refresh()
                return
            except Exception:
                pass
        if not self._mgr_widget:
            self._rebuild_fallback()

    # ───────────────────────────── UI ────────────────────────────────

    def _build_ui(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(6, 2))

        if self.cfg.allow_change_dir:
            btn_change = ttk.Button(bar, text="Change…", command=self._choose_dir)
            btn_change.pack(side="left", padx=(6, 0))
            self._disable_nodes.append(btn_change)

        self._lbl_dir = ttk.Label(bar, text="Preset folder: …", anchor="w")
        self._lbl_dir.pack(side="left", padx=(6, 0), fill="x", expand=True)

        quick = ttk.Frame(bar)
        quick.pack(side="right", padx=6)

        btn_open = ttk.Button(quick, text="Open…", command=lambda: self._publish("ui.preset.open", {}))
        btn_open.pack(side="left")
        self._disable_nodes.append(btn_open)

        if self.cfg.allow_save:
            btn_save = ttk.Button(
                quick, text="Save As…",
                command=lambda: self._publish("ui.preset.save", {"cfg": self.get_cfg()}),
            )
            btn_save.pack(side="left", padx=(6, 0))
            self._disable_nodes.append(btn_save)

        self._host = ttk.Frame(self)
        self._host.pack(fill="both", expand=True)

        self._sync_dir_label()

    def _sync_dir_label(self) -> None:
        p = str(self._preset_dir)
        self._lbl_dir.config(text=f"Preset folder: {p if len(p) <= 90 else '…' + p[-88:]}")

    def _choose_dir(self) -> None:
        d = filedialog.askdirectory(
            title="Choose preset folder",
            initialdir=str(self._last_dir if self.cfg.remember_last else self._preset_dir),
        )
        if d:
            self.set_preset_dir(d)

    # ───────────────────── Manager or fallback ────────────────────────

    def _build_manager_or_fallback(self) -> None:
        for w in self._host.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        self._mgr_widget = None
        self._fallback_steps = None

        if PresetManager is None:
            self._rebuild_fallback()
            return

        try:
            mgr = PresetManager(
                self._host,
                get_config=lambda: self.get_cfg(),
                set_config=lambda c: self.set_cfg(c),
                on_apply=lambda: self._apply_clicked(),
                get_available_filters=lambda: sorted(registry_available()),
                get_current_step=self._get_current_step or (lambda: {}),
                base_dir=str(self._preset_dir),
            )
            mgr.pack(fill="both", expand=True)
            self._mgr_widget = mgr
            # zbieramy disable_nodes dopiero po wyrenderowaniu
            self.after_idle(lambda: self._collect_disable_nodes(mgr))
            self.refresh()
        except Exception as e:
            print(f"[presets-tab] PresetManager init failed: {e}")
            self._rebuild_fallback()

    # ─────────────────────────── Fallback ─────────────────────────────

    def _rebuild_fallback(self) -> None:
        frame = ttk.Frame(self._host)
        frame.pack(fill="both", expand=True)

        info = ttk.Label(
            frame,
            text=(
                "(PresetManager unavailable — fallback)\n"
                "• Load preset (YAML/JSON)\n"
                "• Apply (publishes ui.run.apply_preset)\n"
                "• Save As…"
            ),
            justify="left",
        )
        info.pack(anchor="w", padx=8, pady=(8, 4))

        btns = ttk.Frame(frame)
        btns.pack(anchor="w", padx=8, pady=(0, 8))
        b_load = ttk.Button(btns, text="Load…", command=self._fallback_load)
        b_apply = ttk.Button(btns, text="Apply", command=self._fallback_apply)
        b_load.pack(side="left")
        b_apply.pack(side="left", padx=(6, 0))
        self._disable_nodes.extend([b_load, b_apply])

        if self.cfg.allow_save:
            b_save = ttk.Button(btns, text="Save As…", command=self._fallback_save)
            b_save.pack(side="left", padx=(6, 0))
            self._disable_nodes.append(b_save)

        self._fallback_steps = tk.Text(frame, height=12, width=70)
        self._fallback_steps.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        # Text nie używa state="disabled" w sensie ttk, ustawiamy przez config
        self._disable_nodes.append(self._fallback_steps)

        self._fallback_refresh_text()

    def _fallback_refresh_text(self) -> None:
        if not self._fallback_steps:
            return
        try:
            self._fallback_steps.config(state="normal")
            self._fallback_steps.delete("1.0", "end")
            steps = self._cfg.get("steps") or []
            self._fallback_steps.insert("1.0", f"Preset dir: {self._preset_dir}\nSteps:\n")
            for i, step in enumerate(steps):
                self._fallback_steps.insert(
                    "end", f"  {i + 1}. {step.get('name')} {step.get('params', {})}\n"
                )
            self._fallback_steps.config(state="disabled")
        except Exception:
            pass

    def _fallback_load(self) -> None:
        types = [("YAML", "*.yml;*.yaml"), ("JSON", "*.json"), ("All files", "*.*")]
        fn = filedialog.askopenfilename(
            title="Open Preset",
            filetypes=types,
            initialdir=str(self._last_dir if self.cfg.remember_last else self._preset_dir),
        )
        if not fn:
            return
        self._last_dir = Path(fn).parent
        try:
            text = Path(fn).read_text(encoding="utf-8")
            cfg = self._parse_text(text)
            if not isinstance(cfg, dict):
                raise ValueError("Malformed preset (expected mapping).")
            self._cfg = dict(cfg)
            self._cfg["__preset_dir"] = str(self._last_dir)
            self._preset_dir = self._last_dir
            self._sync_dir_label()
            self._fallback_refresh_text()
            # zachowaj zgodność z serwisem – wyemituj zdarzenia
            self._publish("preset.loaded", {"path": fn, "text": text})
            self._publish("preset.parsed", {"path": fn, "cfg": self.get_cfg()})
        except Exception as e:
            messagebox.showerror("Load preset", str(e))

    def _fallback_save(self) -> None:
        types = [("YAML", "*.yml;*.yaml"), ("JSON", "*.json"), ("All files", "*.*")]
        fn = filedialog.asksaveasfilename(
            title="Save Preset As",
            defaultextension=".yml",
            filetypes=types,
            initialdir=str(self._last_dir if self.cfg.remember_last else self._preset_dir),
        )
        if not fn:
            return
        self._last_dir = Path(fn).parent
        cfg = self.get_cfg()
        try:
            if fn.lower().endswith((".yaml", ".yml")) and yaml is not None:
                text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)  # type: ignore[attr-defined]
            else:
                import json
                text = json.dumps(cfg, ensure_ascii=False, indent=2)
            Path(fn).write_text(text, encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Save preset", str(e))
            return
        self._publish("diag.log", {"level": "OK", "msg": f"Preset saved: {fn}"})

    def _fallback_apply(self) -> None:
        cfg = self.get_cfg()
        if self._on_apply:
            try:
                self._on_apply(cfg)
                return
            except Exception:
                pass
        self._publish("ui.run.apply_preset", {"cfg": cfg})

    # ─────────────────────────── Helpers ──────────────────────────────

    @staticmethod
    def _parse_text(text: str) -> Dict[str, Any] | Any:
        t = (text or "").strip()
        if not t:
            return {}
        if yaml is not None:
            try:
                data = yaml.safe_load(t)  # type: ignore[attr-defined]
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        # JSON fallback
        try:
            import json
            return json.loads(t)
        except Exception:
            return {}

    def _apply_clicked(self) -> None:
        cfg = self.get_cfg()
        if self._on_apply:
            try:
                self._on_apply(cfg)
                return
            except Exception:
                pass
        self._publish("ui.run.apply_preset", {"cfg": cfg})

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self.bus is not None and hasattr(self.bus, "publish"):
            try:
                self.bus.publish(topic, dict(payload))
            except Exception:
                pass

    # ─────────────────────────── Auto-disable ─────────────────────────
    def _collect_disable_nodes(self, root: tk.Misc) -> None:
        """
        Zbierz dzieci (rekurencyjnie) do listy _disable_nodes, aby móc
        włączyć/wyłączyć interakcję na czas wykonywania.
        """
        try:
            for w in self._iter_widgets(root):
                # nie zbieramy etykiet (Label) – one nie są interaktywne
                if isinstance(w, (ttk.Button, ttk.Entry, ttk.Combobox, ttk.Checkbutton, ttk.Radiobutton)):
                    self._disable_nodes.append(w)
                elif isinstance(w, (tk.Listbox, tk.Text, tk.Spinbox, tk.Scale)):
                    self._disable_nodes.append(w)
        except Exception:
            pass

    def _iter_widgets(self, root: tk.Misc) -> Iterable[tk.Widget]:
        try:
            for w in root.winfo_children():
                yield w  # type: ignore[misc]
                yield from self._iter_widgets(w)  # type: ignore[misc]
        except Exception:
            return
            yield  # pragma: no cover

    def _set_enabled(self, enabled: bool) -> None:
        """
        Ustaw stan wszystkich interaktywnych elementów.
        • ttk.Button/Entry/… → state !disabled / disabled
        • tk.Listbox/Text    → state="normal"/"disabled"
        """
        for w in list(self._disable_nodes):
            try:
                if isinstance(w, (tk.Listbox, tk.Text)):
                    w.config(state=("normal" if enabled else "disabled"))
                elif isinstance(w, (tk.Spinbox, tk.Scale)):
                    w.config(state=("normal" if enabled else "disabled"))
                else:
                    # ttk
                    if enabled:
                        w.state(["!disabled"])
                    else:
                        w.state(["disabled"])
            except Exception:
                pass

    # ─────────────────────────── Bus wiring ───────────────────────────

    def _wire_bus(self) -> None:
        if self.bus is None or not hasattr(self.bus, "subscribe"):
            return

        def on_loaded(_t: str, d: Dict[str, Any]) -> None:
            # zapamiętaj folder pliku
            p = (d or {}).get("path")
            if p:
                self._last_dir = Path(p).parent
                if self.cfg.remember_last:
                    self._preset_dir = self._last_dir
                    self._cfg["__preset_dir"] = str(self._preset_dir)
                    self._sync_dir_label()

        def on_parsed(_t: str, d: Dict[str, Any]) -> None:
            cfg = (d or {}).get("cfg")
            if isinstance(cfg, dict):
                # nie nadpisuj __preset_dir jeśli przyszło spoza GUI
                keep_dir = self._cfg.get("__preset_dir", None)
                self._cfg = dict(cfg)
                if keep_dir:
                    self._cfg["__preset_dir"] = keep_dir
                self._fallback_refresh_text()

        def on_image_loaded(_t: str, d: Dict[str, Any]) -> None:
            path = (d or {}).get("path")
            if path:
                self.set_image_path(str(path))

        # BLK/UNBLK podczas przetwarzania
        def on_run_start(_t: str, _d: Dict[str, Any]) -> None:
            self._set_enabled(False)

        def on_run_finish(_t: str, _d: Dict[str, Any]) -> None:
            self._set_enabled(True)

        try:
            self.bus.subscribe("preset.loaded", on_loaded)
            self.bus.subscribe("preset.parsed", on_parsed)
            self.bus.subscribe("ui.image.loaded", on_image_loaded)
            # run lifecycle
            self.bus.subscribe("run.start", on_run_start)
            self.bus.subscribe("run.done", on_run_finish)
            self.bus.subscribe("run.error", on_run_finish)
        except Exception:
            pass
