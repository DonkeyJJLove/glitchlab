# glitchlab/gui/views/tab_filter.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import importlib, importlib.util, pkgutil, sys, traceback
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional, List, Tuple, Type, Iterable

# --- Registry filtrów (miękko) ----------------------------------------------
try:
    from glitchlab.core.registry import (
        available as registry_available,
        get as registry_get,
        canonical,
        meta,
    )
except Exception:
    def registry_available() -> List[str]: return []
    def registry_get(name: str): raise KeyError(name)
    def canonical(name: str) -> str: return str(name or "")
    def meta(name: str) -> Dict[str, Any]: return {"defaults": {}, "doc": ""}

# --- Panel loader (dedykowane panele filtrów) -------------------------------
try:
    from glitchlab.gui.panel_loader import instantiate_panel  # type: ignore
except Exception:
    instantiate_panel = None  # type: ignore

# --- PanelContext ------------------------------------------------------------
try:
    from glitchlab.gui.panel_base import PanelContext  # type: ignore
except Exception:
    class PanelContext:  # fallback
        def __init__(self, **kw): self.__dict__.update(kw)

# --- ParamForm (fallback) ----------------------------------------------------
try:
    from glitchlab.gui.widgets.param_form import ParamForm  # type: ignore
except Exception:
    ParamForm = None  # type: ignore


class FilterTabConfig:
    def __init__(self, allow_apply: bool = True) -> None:
        self.allow_apply = allow_apply


class TabFilter(ttk.Frame):
    """
    Zakładka „Filters”:
      • wybór filtra (Combobox),
      • próba załadowania dedykowanego panelu (panel_loader),
      • fallback do ParamForm (jeśli brak panelu),
      • Apply,
      • narzędzia: Load Filters / Rescan / Probe / Reload,
      • auto-disable kontrolek na czas wykonywania,
      • logowanie przez EventBus (diag.log).
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        bus: Optional[Any] = None,
        cfg: Optional[FilterTabConfig] = None,
        ctx_ref: Optional[Any] = None,
    ) -> None:
        super().__init__(master)
        self.bus = bus
        self.cfg = cfg or FilterTabConfig()
        self.ctx_ref = ctx_ref

        self._panel: Optional[tk.Widget] = None
        self._fallback_form: Optional[Any] = None
        self._filter_params_cache: Dict[str, Any] = {}
        self._last_loaded_module: Optional[str] = None

        # do auto-disable
        self._disable_nodes: List[tk.Widget] = []

        self._build_ui()

        # >>> WAŻNE: metoda musi istnieć – wcześniej jej brak powodował AttributeError
        self._ensure_filters_imported()
        self._refresh_filter_list()
        self._mount_panel(self.cmb_filter.get() or "")

        # services
        self._wire_bus()

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        top = ttk.Frame(self); top.pack(fill="x", pady=(6, 4))
        ttk.Label(top, text="Filter:").pack(side="left", padx=(6, 4))

        self.cmb_filter = ttk.Combobox(top, values=[], state="readonly", width=36)
        self.cmb_filter.pack(side="left", fill="x", expand=True)
        self._disable_nodes.append(self.cmb_filter)

        # Akcje: Load/Rescan/Probe/Reload
        act = ttk.Frame(top); act.pack(side="left", padx=(6, 0))
        for txt, cmd in (
            ("Load Filters", self._on_load_filters),
            ("Rescan", self._on_rescan),
            ("Probe", self._probe_current),
            ("Reload", self._reload_current),
        ):
            b = ttk.Button(act, text=txt, command=cmd)
            b.pack(side="left", padx=(4 if txt != "Load Filters" else 0, 0))
            self._disable_nodes.append(b)

        if self.cfg.allow_apply:
            b_apply = ttk.Button(top, text="Apply", command=self._apply_clicked)
            b_apply.pack(side="left", padx=8)
            self._disable_nodes.append(b_apply)

        self.panel_host = ttk.Frame(self)
        self.panel_host.pack(fill="x", padx=6, pady=(0, 6))

        self.cmb_filter.bind("<<ComboboxSelected>>",
                             lambda _e=None: self._mount_panel(self.cmb_filter.get()))

    # --------------------------------------------------------------- actions ---
    def _on_load_filters(self) -> None:
        self._ensure_filters_imported()
        self._refresh_filter_list()
        self._mount_panel(self.cmb_filter.get() or "")

    def _on_rescan(self) -> None:
        self._refresh_filter_list(prefer_scan=True)
        self._mount_panel(self.cmb_filter.get() or "")

    # ------------------------------------------------------------------ API ---
    def set_ctx(self, ctx: Any) -> None:
        self.ctx_ref = ctx

    def set_filter(self, name: str) -> None:
        vals = list(self.cmb_filter["values"])
        if name in vals:
            self.cmb_filter.set(name)
            self._mount_panel(name)

    def get_current_step(self) -> Dict[str, Any]:
        name = (self.cmb_filter.get() or "").strip()
        if self._filter_params_cache:
            params = dict(self._filter_params_cache)
        elif self._fallback_form is not None:
            try:
                params = dict(self._fallback_form.values())
            except Exception:
                params = {}
        else:
            params = {}
        return {"name": name, "params": params}

    # ------------------------------------------------------------- internals ---
    def _apply_clicked(self) -> None:
        step = self.get_current_step()
        self._publish("ui.run.apply_filter", {"step": step})

    def _on_filter_params_changed(self, params: Dict[str, Any]) -> None:
        self._filter_params_cache = dict(params or {})
        self._publish("ui.filter.params_changed", {
            "name": self.cmb_filter.get(),
            "params": dict(params or {})
        })

    def _clear_host(self) -> None:
        for w in self.panel_host.winfo_children():
            try: w.destroy()
            except Exception: pass
        self._panel = None
        self._fallback_form = None
        self._filter_params_cache = {}

    def _refresh_filter_list(self, *, prefer_scan: bool = False) -> None:
        names: List[str] = []
        reg = list(registry_available() or [])
        if reg and not prefer_scan:
            names = sorted({self._canon(n) for n in reg})
            self._log(f"registry_available → {len(names)}", "DEBUG")
        else:
            scanned = self._scan_panels_dir()
            if scanned:
                names = sorted(scanned)
                self._log(f"panels scan → {len(names)}", "DEBUG")

        self.cmb_filter["values"] = names
        if names:
            if (self.cmb_filter.get() or "") not in names:
                self.cmb_filter.set(names[0])
        else:
            self.cmb_filter.set("")

    def _mount_panel(self, filter_name: str) -> None:
        self._clear_host()
        if not filter_name:
            self._log("No filter selected (empty registry?).", "WARN")
            f = ttk.Frame(self.panel_host)
            ttk.Label(f, text="(No filters detected)").pack(padx=6, pady=6, anchor="w")
            f.pack(fill="x")
            self._panel = f
            return

        self._log(f"Mounting panel for filter='{filter_name}'", "DEBUG")

        # 1) próbujemy dedykowany panel (panel_loader)
        panel_widget: Optional[tk.Widget] = None
        if instantiate_panel is not None:
            try:
                # Defaults z registry (jak dostępne)
                try:
                    dflt = dict(meta(self._canon(filter_name)).get("defaults", {}))
                except Exception:
                    dflt = {}
                ctx = PanelContext(
                    filter_name=self._canon(filter_name),
                    defaults=dflt,
                    params={},
                    on_change=self._on_filter_params_changed,
                    cache_ref=getattr(self.ctx_ref, "cache", None) if self.ctx_ref else None,
                    get_mask_keys=(lambda: list(getattr(self.ctx_ref, "masks", {}).keys()))
                        if self.ctx_ref and hasattr(self.ctx_ref, "masks") else None,
                )
                panel_widget = instantiate_panel(self.panel_host, self._canon(filter_name), ctx=ctx)
            except Exception as ex:
                self._log(f"instantiate_panel failed: {ex}", "ERROR")
                self._log(traceback.format_exc(), "DEBUG")

        # 2) fallback do ParamForm (jeśli brak panelu)
        if panel_widget is None:
            self._log(f"Falling back to ParamForm for '{filter_name}'", "WARN")
            panel_widget = self._fallback_panel(filter_name)

        panel_widget.pack(fill="x")
        self._panel = panel_widget

        # zbierz kontrolki do disable_nodes
        self.after_idle(lambda: self._collect_disable_nodes(panel_widget))

        self._publish("ui.filter.select", {"name": filter_name})

    # ---------------------------------------------------------- auto-disable ---
    def _collect_disable_nodes(self, root: tk.Misc) -> None:
        try:
            for w in self._iter_widgets(root):
                if isinstance(w, (ttk.Button, ttk.Entry, ttk.Combobox,
                                  ttk.Checkbutton, ttk.Radiobutton)):
                    self._disable_nodes.append(w)
                elif isinstance(w, (tk.Listbox, tk.Text, tk.Spinbox, tk.Scale)):
                    self._disable_nodes.append(w)
        except Exception:
            pass

    def _iter_widgets(self, root: tk.Misc) -> Iterable[tk.Widget]:
        try:
            for w in root.winfo_children():
                yield w
                yield from self._iter_widgets(w)
        except Exception:
            return
            yield

    def _set_enabled(self, enabled: bool) -> None:
        for w in list(self._disable_nodes):
            try:
                if isinstance(w, (tk.Listbox, tk.Text, tk.Spinbox, tk.Scale)):
                    w.config(state=("normal" if enabled else "disabled"))
                else:
                    if enabled:
                        w.state(["!disabled"])
                    else:
                        w.state(["disabled"])
            except Exception:
                pass

    def _wire_bus(self) -> None:
        if not (self.bus and hasattr(self.bus, "subscribe")):
            return

        def on_run_start(_t, _d): self._set_enabled(False)
        def on_run_finish(_t, _d): self._set_enabled(True)

        try:
            self.bus.subscribe("run.start", on_run_start)
            self.bus.subscribe("run.done", on_run_finish)
            self.bus.subscribe("run.error", on_run_finish)
        except Exception:
            pass

    # ---------------------------------------------------------- diagnostics ---
    def _probe_current(self) -> None:
        name = (self.cmb_filter.get() or "").strip()
        if not name:
            self._log("Probe: no filter selected.", "WARN")
            return
        self._diag_probe(name)

    def _reload_current(self) -> None:
        cand_mods = []
        if self._last_loaded_module:
            cand_mods.append(self._last_loaded_module)
        name = (self.cmb_filter.get() or "").strip()
        if name:
            cand_mods.extend(self._module_candidates(name))

        seen = set()
        for mod_name in cand_mods:
            if not mod_name or mod_name in seen:
                continue
            seen.add(mod_name)
            if mod_name in sys.modules:
                self._log(f"Reloading module: {mod_name}", "DEBUG")
                try:
                    importlib.reload(sys.modules[mod_name])
                    self._log("Reload OK", "OK")
                except Exception as ex:
                    self._log(f"Reload failed: {ex}", "ERROR")
                    self._log(traceback.format_exc(), "DEBUG")
        if name:
            self._mount_panel(name)

    def _diag_probe(self, filter_name: str) -> None:
        canon = self._canon(filter_name)
        mod_names = self._module_candidates(canon)
        self._log(f"[PROBE] filter='{filter_name}' → canonical='{canon}'", "INFO")
        for mod_name in mod_names:
            spec = importlib.util.find_spec(mod_name)
            if spec is None:
                self._log(f"find_spec: NOT FOUND ({mod_name})", "ERROR")
            else:
                self._log(f"find_spec: OK (origin={getattr(spec, 'origin', None)})", "OK")
                try:
                    mod = importlib.import_module(mod_name)
                    self._log(f"import: OK (__file__={getattr(mod, '__file__', None)})", "OK")
                    picks = self._discover_panel_classes(mod)
                    if not picks:
                        self._log(f"{mod_name}: no *Panel class found", "WARN")
                    else:
                        self._log(f"{mod_name}: panel classes: {picks}", "OK")
                except Exception as ex:
                    self._log(f"import: FAILED → {ex}", "ERROR")
                    self._log(traceback.format_exc(), "DEBUG")

    # --------------------------------------------------------- panel loader ---
    @staticmethod
    def _discover_panel_classes(mod) -> List[str]:
        picks: List[str] = []
        for attr in dir(mod):
            obj = getattr(mod, attr)
            try:
                if isinstance(obj, type) and attr.lower().endswith("panel"):
                    picks.append(attr)
            except Exception:
                pass
        return picks

    def _canon(self, name: str) -> str:
        try:
            return canonical(name).lower()
        except Exception:
            return str(name or "").lower()

    def _module_candidates(self, name: str) -> List[str]:
        canon = self._canon(name)
        return [
            f"glitchlab.gui.panels.panel_{canon}",
            f"glitchlab.gui.panels.{canon}_panel",
        ]

    def _import_first_available(self, candidates: List[str]) -> Tuple[Optional[str], Optional[object]]:
        for mod_name in candidates:
            try:
                mod = importlib.import_module(mod_name)
                return mod_name, mod
            except Exception:
                continue
        return None, None

    def _pick_panel_class(self, mod) -> Optional[Type]:
        cls = getattr(mod, "Panel", None)
        if isinstance(cls, type) and cls.__name__.lower().endswith("panel"):
            return cls
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and attr.lower().endswith("panel"):
                return obj
        return None

    def _try_load_panel(self, filter_name: str) -> Optional[tk.Widget]:
        mod_name, mod = self._import_first_available(self._module_candidates(filter_name))
        if mod is None:
            self._log(f"import failed for both module patterns of '{filter_name}'", "ERROR")
            return None

        cls = self._pick_panel_class(mod)
        if cls is None:
            self._log(f"{mod.__name__}: no class '*Panel' found", "WARN")
            return None

        # Defaults z registry (jak dostępne)
        try:
            dflt = dict(meta(self._canon(filter_name)).get("defaults", {}))
        except Exception:
            dflt = {}

        ctx = PanelContext(
            filter_name=self._canon(filter_name),
            defaults=dflt,
            params={},
            on_change=self._on_filter_params_changed,
            cache_ref=getattr(self.ctx_ref, "cache", None) if self.ctx_ref else None,
            get_mask_keys=(lambda: list(getattr(self.ctx_ref, "masks", {}).keys()))
                if self.ctx_ref and hasattr(self.ctx_ref, "masks") else None,
        )

        try:
            inst = cls(self.panel_host, ctx=ctx)
            self._last_loaded_module = mod_name
            self._log(f"panel constructed: {cls.__module__}.{cls.__name__}", "OK")
            return inst
        except TypeError:
            try:
                inst = cls(self.panel_host)  # starsze panele bez ctx
                self._last_loaded_module = mod_name
                self._log(f"panel constructed (no ctx): {cls.__module__}.{cls.__name__}", "OK")
                return inst
            except Exception as ex2:
                self._log(f"panel ctor failed: {ex2}", "ERROR")
                self._log(traceback.format_exc(), "DEBUG")
                return None
        except Exception as ex:
            self._log(f"panel ctor failed: {ex}", "ERROR")
            self._log(traceback.format_exc(), "DEBUG")
            return None

    def _fallback_panel(self, filter_name: str) -> tk.Widget:
        if ParamForm is None:
            f = ttk.Frame(self.panel_host)
            ttk.Label(f, text=f"(No panel & ParamForm missing for '{filter_name}')").pack(
                padx=6, pady=6, anchor="w"
            )
            return f
        form = ParamForm(self.panel_host, get_filter_callable=registry_get)
        try:
            form.build_for(filter_name)
            self._log("ParamForm built OK", "OK")
        except Exception as e:
            self._log(f"ParamForm build failed: {e}", "ERROR")
            self._log(traceback.format_exc(), "DEBUG")
            ttk.Label(form, text=f"(ParamForm failed: {e})").pack(padx=6, pady=6, anchor="w")
        self._fallback_form = form
        return form

    # ------------------------------------------------------- filters import ---
    def _ensure_filters_imported(self) -> None:
        """
        Wymuś import pakietu `glitchlab.filters`, żeby rejestr filtrów nie był pusty.
        Jeśli import się nie powiedzie, logujemy, ale kontynuujemy (będzie skan paneli).
        """
        try:
            if "glitchlab.filters" not in sys.modules:
                importlib.import_module("glitchlab.filters")
                self._log("filters imported (glitchlab.filters)", "OK")
        except Exception as ex:
            self._log(f"filters import failed: {ex}", "WARN")
            self._log(traceback.format_exc(), "DEBUG")

    def _scan_panels_dir(self) -> List[str]:
        """
        Gdy registry puste: spróbuj odczytać listę filtrów po samych plikach paneli.
        Szukamy:
          - glitchlab.gui.panels.panel_<name>.py  → <name>
          - glitchlab.gui.panels.<name>_panel.py  → <name>
        """
        names: set[str] = set()
        try:
            pkg = importlib.import_module("glitchlab.gui.panels")
        except Exception as ex:
            self._log(f"scan panels: import 'glitchlab.gui.panels' failed → {ex}", "WARN")
            return []
        try:
            if hasattr(pkg, "__path__"):
                for _finder, mod_name, ispkg in pkgutil.iter_modules(pkg.__path__):
                    if ispkg:
                        continue
                    n = mod_name
                    if n.startswith("panel_"):
                        names.add(n[len("panel_"):].lower())
                    elif n.endswith("_panel"):
                        names.add(n[:-len("_panel")].lower())
        except Exception as ex:
            self._log(f"scan panels failed: {ex}", "WARN")
            self._log(traceback.format_exc(), "DEBUG")
        return sorted(names)

    # ------------------------------------------------------------- helpers ---
    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self.bus and hasattr(self.bus, "publish"):
            try:
                self.bus.publish(topic, dict(payload))
            except Exception:
                pass

    def _log(self, msg: str, level: str = "INFO") -> None:
        # kieruj do konsoli diagnostycznej (services → DiagConsole), albo stdout
        if self.bus and hasattr(self.bus, "publish"):
            try:
                self.bus.publish("diag.log", {"msg": f"[Filters] {msg}", "level": level})
                return
            except Exception:
                pass
        print(f"[{level}] {msg}")
