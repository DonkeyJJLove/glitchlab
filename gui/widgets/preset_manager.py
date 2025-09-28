# glitchlab/gui/widgets/preset_manager.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from collections import deque
from pathlib import Path
from typing import Callable, Optional, Any, List, Deque

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ──────────────────────────────────────────────────────────────────────────────
# YAML (opcjonalnie)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


# ──────────────────────────────────────────────────────────────────────────────
# Pomocnicze: preset v2
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_v2(cfg: dict) -> dict:
    """Minimalna normalizacja preset v2 (bez modyfikowania wejścia in-place)."""
    c = dict(cfg or {})
    c.setdefault("version", 2)
    c.setdefault("seed", 7)
    c.setdefault("amplitude", {"kind": "none", "strength": 1.0})
    c.setdefault("edge_mask", {"thresh": 60, "dilate": 0, "ksize": 3})
    c.setdefault("steps", [])
    c.setdefault("__preset_dir", str(Path.cwd() / "presets"))  # tylko w pamięci
    return c


def _dumps_yaml(d: dict) -> str:
    if _HAS_YAML:
        return yaml.safe_dump(d, sort_keys=False, allow_unicode=True)  # type: ignore[attr-defined]
    # fallback: „yamlopodobny” przez JSON + komentarz ostrzegawczy
    return "# (YAML unavailable; showing JSON)\n" + json.dumps(d, indent=2, ensure_ascii=False)


def _loads_yaml(txt: str) -> dict:
    if _HAS_YAML:
        return yaml.safe_load(txt) or {}  # type: ignore[attr-defined]
    # w trybie fallback akceptujemy JSON
    return json.loads(txt)


# ──────────────────────────────────────────────────────────────────────────────
# PresetManager (kompatybilny z obiema wersjami PresetsTab)
# ──────────────────────────────────────────────────────────────────────────────
class PresetManager(ttk.Frame):
    """
    Presets tab widget.

    Wymagane wątki funkcjonalne (przekazywane przez PresetsTab / App):
      • get_cfg() -> dict
      • set_cfg(dict) -> None
      • apply_preset_steps() -> None
      • get_available_filters() -> list[str]

    Opcjonalnie:
      • get_current_step() -> {"name": str, "params": dict}
      • base_dir: str|Path  (startowy katalog presetów)

    Zgodność:
      - stary styl (pozycje):   (parent, get_cfg, set_cfg, apply_preset_steps, get_available_filters, get_current_step=None)
      - nowy styl (nazwy):      (parent, get_config=..., set_config=..., on_apply=..., get_available_filters=..., get_current_step=..., base_dir=...)
    """

    # Akceptujemy zarówno pozycje, jak i nazwy – plus **kw na przyszłość
    def __init__(
        self,
        parent: tk.Widget,
        get_cfg: Optional[Callable[[], dict]] = None,
        set_cfg: Optional[Callable[[dict], None]] = None,
        apply_preset_steps: Optional[Callable[[], None]] = None,
        get_available_filters: Optional[Callable[[], List[str]]] = None,
        get_current_step: Optional[Callable[[], dict]] = None,
        **kw: Any,
    ) -> None:
        # Mapowanie aliasów (nowe nazwy ⇒ stare)
        get_cfg = kw.get("get_config", None) or get_cfg
        set_cfg = kw.get("set_config", None) or set_cfg
        apply_preset_steps = kw.get("on_apply", None) or apply_preset_steps
        get_available_filters = kw.get("get_available_filters", None) or get_available_filters
        get_current_step = kw.get("get_current_step", None) or get_current_step
        base_dir = kw.get("base_dir", None)

        super().__init__(parent)

        # Walidacja minimalna
        if not callable(get_cfg) or not callable(set_cfg) or not callable(apply_preset_steps) or not callable(get_available_filters):
            raise TypeError(
                "PresetManager requires get_cfg/set_cfg/apply_preset_steps/get_available_filters callbacks "
                "(or their keyword aliases get_config/set_config/on_apply/get_available_filters)."
            )

        # Callbacks
        self._get_cfg: Callable[[], dict] = get_cfg  # type: ignore[assignment]
        self._set_cfg: Callable[[dict], None] = set_cfg  # type: ignore[assignment]
        self._apply_cb: Callable[[], None] = apply_preset_steps  # type: ignore[assignment]
        self._get_filters: Callable[[], List[str]] = get_available_filters  # type: ignore[assignment]
        self._get_current_step: Optional[Callable[[], dict]] = get_current_step

        # Historia ostatnich kroków (może być uzupełniana z App)
        self.history: Deque[dict] = deque(maxlen=20)

        # ───── UI: top bar ─────
        top = ttk.Frame(self)
        top.pack(fill="x", pady=(6, 4))

        ttk.Label(top, text="Preset folder:").pack(side="left", padx=(6, 4))
        self.dir_var = tk.StringVar(value="")
        self.dir_entry = ttk.Entry(top, textvariable=self.dir_var)
        self.dir_entry.pack(side="left", fill="x", expand=True)

        ttk.Button(top, text="…", width=3, command=self._choose_dir).pack(side="left", padx=4)
        ttk.Button(top, text="Refresh", command=self._refresh_from_cfg).pack(side="left", padx=(4, 0))
        ttk.Button(top, text="Load", command=self._load_from_disk).pack(side="left", padx=(4, 0))
        ttk.Button(top, text="Save As…", command=self._save_as).pack(side="left", padx=(4, 0))

        # ───── middle: editors + steps ─────
        mid = ttk.Panedwindow(self, orient="vertical")
        mid.pack(fill="both", expand=True)

        # Editors
        editors = ttk.Notebook(mid)
        mid.add(editors, weight=3)

        self.txt_yaml = tk.Text(editors, wrap="none", undo=True, height=12)
        self.txt_json = tk.Text(editors, wrap="none", undo=True, height=12)

        editors.add(self.txt_yaml, text="YAML")
        editors.add(self.txt_json, text="JSON")

        # Steps panel
        steps_frame = ttk.Frame(mid)
        mid.add(steps_frame, weight=2)

        # left: list
        left = ttk.Frame(steps_frame)
        left.pack(side="left", fill="both", expand=True, padx=(6, 3), pady=(4, 6))

        ttk.Label(left, text="Steps").pack(anchor="w")
        self.lb_steps = tk.Listbox(left, height=8, exportselection=False)
        yscroll = ttk.Scrollbar(left, orient="vertical", command=self.lb_steps.yview)
        self.lb_steps.config(yscrollcommand=yscroll.set)
        self.lb_steps.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="left", fill="y")

        # right: controls
        right = ttk.Frame(steps_frame)
        right.pack(side="left", fill="y", padx=(3, 6), pady=(4, 6))

        self.btn_add_current = ttk.Button(
            right, text="Add current step", command=self._add_current_step, state="disabled"
        )
        if self._get_current_step is not None:
            self.btn_add_current.config(state="normal")
        self.btn_add_current.pack(fill="x", pady=(0, 6))

        ttk.Button(right, text="Apply all", command=self._apply_all).pack(fill="x")
        ttk.Button(right, text="Apply selected", command=self._apply_selected).pack(fill="x", pady=(6, 0))
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)
        ttk.Button(right, text="Up", command=lambda: self._move_step(-1)).pack(fill="x")
        ttk.Button(right, text="Down", command=lambda: self._move_step(+1)).pack(fill="x", pady=(6, 0))
        ttk.Button(right, text="Delete", command=self._del_step).pack(fill="x", pady=(6, 0))
        ttk.Button(right, text="Clear", command=self._clear_steps).pack(fill="x", pady=(6, 0))
        ttk.Button(right, text="Edit as JSON…", command=self._edit_selected_json).pack(fill="x", pady=(6, 0))

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(right, text="Recent (last 20)").pack(anchor="w")
        self.lb_hist = tk.Listbox(right, height=6)
        self.lb_hist.pack(fill="both", expand=False)
        ttk.Button(right, text="Add from history", command=self._add_from_history).pack(fill="x", pady=(6, 0))

        # ───── inicjalizacja: katalog bazowy i edytory ─────
        # 1) źródło: base_dir (z nowego API),
        # 2) fallback: __preset_dir z get_cfg(),
        # 3) ostatecznie: CWD/presets przez _ensure_v2().
        start_cfg = _ensure_v2(self._get_cfg())
        start_dir = str(base_dir) if base_dir else (start_cfg.get("__preset_dir") or "")
        self.dir_var.set(start_dir)

        self._refresh_from_cfg()

        # utrzymuj synchronizację edytorów po opuszczeniu focusu
        self.txt_yaml.bind("<FocusOut>", lambda e: self._sync_from_yaml())
        self.txt_json.bind("<FocusOut>", lambda e: self._sync_from_json())

    # ───────────────────── public API (dla App) ────────────────────────
    def push_history(self, step: dict) -> None:
        """Wywołuj z App po każdym uruchomieniu pojedynczego filtra (do listy Recent)."""
        try:
            s = {"name": step.get("name"), "params": dict(step.get("params") or {})}
            if s.get("name"):
                self.history.appendleft(s)
                self._rebuild_history_listbox()
        except Exception:
            pass

    # ───────────────────────── akcje UI ────────────────────────────────
    def _choose_dir(self) -> None:
        try:
            start = self.dir_var.get().strip() or str(Path.cwd())
            d = filedialog.askdirectory(title="Choose preset folder", initialdir=start)
            if not d:
                return
            cfg = self._current_cfg_from_editors()
            cfg["__preset_dir"] = d
            self.dir_var.set(d)
            self._set_cfg(cfg)  # propaguj do App
        except Exception as e:
            messagebox.showerror("Preset folder", str(e))

    def _refresh_from_cfg(self) -> None:
        """Wczytaj stan z App i uzupełnij edytory + listy."""
        try:
            cfg = _ensure_v2(self._get_cfg())
            # jeśli pole dir puste, spróbuj wypełnić __preset_dir
            if not (self.dir_var.get().strip()):
                self.dir_var.set(str(cfg.get("__preset_dir", "")))
            # do edytorów (bez __preset_dir)
            clean = dict(cfg)
            clean.pop("__preset_dir", None)
            self._set_editors_from_cfg(clean)
            self._rebuild_steps_listbox(clean.get("steps") or [])
        except Exception as e:
            messagebox.showerror("Refresh", str(e))

    def _load_from_disk(self) -> None:
        try:
            base = Path(self.dir_var.get().strip() or ".")
            f = filedialog.askopenfilename(
                title="Open Preset",
                initialdir=str(base),
                filetypes=[("YAML/JSON", "*.yml;*.yaml;*.json"), ("All files", "*.*")]
            )
            if not f:
                return
            txt = Path(f).read_text(encoding="utf-8")
            if f.lower().endswith((".yml", ".yaml")):
                data = _loads_yaml(txt)
            else:
                data = json.loads(txt)
            data = _ensure_v2(data)
            # nie nadpisuj __preset_dir tym z pliku – trzymaj bieżący folder UI
            if self.dir_var.get().strip():
                data["__preset_dir"] = self.dir_var.get().strip()
            self._set_cfg(data)
            self._refresh_from_cfg()
        except Exception as e:
            messagebox.showerror("Load preset", str(e))

    def _save_as(self) -> None:
        try:
            cfg = self._current_cfg_from_editors()
            base = Path(self.dir_var.get().strip() or ".")
            base.mkdir(parents=True, exist_ok=True)
            f = filedialog.asksaveasfilename(
                title="Save Preset As",
                initialdir=str(base),
                defaultextension=".yml",
                filetypes=[("YAML", "*.yml;*.yaml"), ("JSON", "*.json"), ("All files", "*.*")]
            )
            if not f:
                return
            to_write = dict(cfg)
            to_write.pop("__preset_dir", None)  # nie zapisujemy prywatnego pola
            if f.lower().endswith(".json"):
                Path(f).write_text(json.dumps(to_write, indent=2, ensure_ascii=False), encoding="utf-8")
            else:
                Path(f).write_text(_dumps_yaml(to_write), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Save preset", str(e))

    def _apply_all(self) -> None:
        """Zapisz zmiany do App i odpal cały preset."""
        try:
            cfg = self._current_cfg_from_editors()
            self._set_cfg(cfg)
            self._apply_cb()
        except Exception as e:
            messagebox.showerror("Apply steps", str(e))

    def _apply_selected(self) -> None:
        """Uruchom TYLKO zaznaczony krok (bez psucia bieżącego presetu)."""
        try:
            sel = self.lb_steps.curselection()
            if not sel:
                messagebox.showinfo("Apply selected", "Zaznacz krok na liście.")
                return
            idx = sel[0]
            full_cfg = self._current_cfg_from_editors()
            steps = list(full_cfg.get("steps") or [])
            if idx < 0 or idx >= len(steps):
                return
            # tymczasowa podmiana w App
            original = _ensure_v2(self._get_cfg())
            temp = dict(original)
            temp["steps"] = [steps[idx]]
            self._set_cfg(temp)
            self._apply_cb()
            # przywróć
            self._set_cfg(original)
        except Exception as e:
            messagebox.showerror("Apply selected", str(e))

    def _add_current_step(self) -> None:
        if self._get_current_step is None:
            return
        try:
            step = self._get_current_step() or {}
            if not step.get("name"):
                messagebox.showinfo("Add current step", "Brak aktywnego filtra / krok niekompletny.")
                return
            cfg = self._current_cfg_from_editors()
            steps = list(cfg.get("steps") or [])
            steps.append({"name": step["name"], "params": dict(step.get("params") or {})})
            cfg["steps"] = steps
            self._set_editors_from_cfg(cfg)
            self._rebuild_steps_listbox(steps)
        except Exception as e:
            messagebox.showerror("Add current step", str(e))

    def _add_from_history(self) -> None:
        try:
            sel = self.lb_hist.curselection()
            if not sel:
                return
            idx = sel[0]
            if idx < 0 or idx >= len(self.history):
                return
            step = self.history[idx]
            cfg = self._current_cfg_from_editors()
            steps = list(cfg.get("steps") or [])
            steps.append(dict(step))
            cfg["steps"] = steps
            self._set_editors_from_cfg(cfg)
            self._rebuild_steps_listbox(steps)
        except Exception as e:
            messagebox.showerror("History", str(e))

    def _move_step(self, delta: int) -> None:
        try:
            sel = self.lb_steps.curselection()
            if not sel:
                return
            idx = sel[0]
            cfg = self._current_cfg_from_editors()
            steps = list(cfg.get("steps") or [])
            j = idx + delta
            if j < 0 or j >= len(steps):
                return
            steps[idx], steps[j] = steps[j], steps[idx]
            cfg["steps"] = steps
            self._set_editors_from_cfg(cfg)
            self._rebuild_steps_listbox(steps)
            self.lb_steps.selection_clear(0, "end")
            self.lb_steps.selection_set(j)
            self.lb_steps.see(j)
        except Exception as e:
            messagebox.showerror("Reorder", str(e))

    def _del_step(self) -> None:
        try:
            sel = self.lb_steps.curselection()
            if not sel:
                return
            idx = sel[0]
            cfg = self._current_cfg_from_editors()
            steps = list(cfg.get("steps") or [])
            if 0 <= idx < len(steps):
                steps.pop(idx)
                cfg["steps"] = steps
                self._set_editors_from_cfg(cfg)
                self._rebuild_steps_listbox(steps)
        except Exception as e:
            messagebox.showerror("Delete step", str(e))

    def _clear_steps(self) -> None:
        try:
            if messagebox.askyesno("Clear steps", "Usunąć wszystkie kroki z bieżącego presetu?"):
                cfg = self._current_cfg_from_editors()
                cfg["steps"] = []
                self._set_editors_from_cfg(cfg)
                self._rebuild_steps_listbox([])
        except Exception as e:
            messagebox.showerror("Clear steps", str(e))

    def _edit_selected_json(self) -> None:
        sel = self.lb_steps.curselection()
        if not sel:
            return
        idx = sel[0]
        cfg = self._current_cfg_from_editors()
        steps = list(cfg.get("steps") or [])
        if not (0 <= idx < len(steps)):
            return
        step = dict(steps[idx])

        # proste okno edycji JSON
        win = tk.Toplevel(self)
        win.title(f"Edit step #{idx}")
        win.transient(self.winfo_toplevel())
        win.grab_set()

        txt = tk.Text(win, width=60, height=18)
        txt.pack(fill="both", expand=True, padx=6, pady=6)
        txt.insert("1.0", json.dumps(step, indent=2, ensure_ascii=False))

        def ok():
            try:
                data = json.loads(txt.get("1.0", "end"))
                if not isinstance(data, dict) or "name" not in data:
                    raise ValueError("Step musi być dict z polem 'name'.")
                steps[idx] = data
                cfg["steps"] = steps
                self._set_editors_from_cfg(cfg)
                self._rebuild_steps_listbox(steps)
                win.destroy()
            except Exception as e:
                messagebox.showerror("Edit step", str(e))

        ttk.Button(win, text="OK", command=ok).pack(side="right", padx=6, pady=(0, 6))
        ttk.Button(win, text="Cancel", command=win.destroy).pack(side="right", pady=(0, 6))

    # ───────────────────── helpers ─────────────────────────────────────
    def _set_editors_from_cfg(self, cfg_no_dir: dict) -> None:
        """Ustaw edytory na podstawie cfg *bez* klucza __preset_dir."""
        try:
            # YAML
            self.txt_yaml.delete("1.0", "end")
            self.txt_yaml.insert("1.0", _dumps_yaml(cfg_no_dir))
            # JSON
            self.txt_json.delete("1.0", "end")
            self.txt_json.insert("1.0", json.dumps(cfg_no_dir, indent=2, ensure_ascii=False))
        except Exception as e:
            messagebox.showerror("Editors", str(e))

    def _current_cfg_from_editors(self) -> dict:
        """Zbierz config z aktywnego edytora + dir z pola na górze."""
        try:
            txt_y = self.txt_yaml.get("1.0", "end").strip()
            txt_j = self.txt_json.get("1.0", "end").strip()
            try:
                data = _loads_yaml(txt_y)
            except Exception:
                data = json.loads(txt_j)
            data = _ensure_v2(data)
            # preferuj to, co w UI
            ui_dir = self.dir_var.get().strip()
            if ui_dir:
                data["__preset_dir"] = ui_dir
            return data
        except Exception as e:
            messagebox.showerror("Preset parse", str(e))
            return _ensure_v2(self._get_cfg())

    def _sync_from_yaml(self) -> None:
        """Kiedy użytkownik edytuje YAML – przepisz JSON (bez __preset_dir)."""
        try:
            d = _loads_yaml(self.txt_yaml.get("1.0", "end"))
            d = _ensure_v2(d)
            d.pop("__preset_dir", None)
            self.txt_json.delete("1.0", "end")
            self.txt_json.insert("1.0", json.dumps(d, indent=2, ensure_ascii=False))
            self._rebuild_steps_listbox(d.get("steps") or [])
        except Exception:
            # edycja w toku – nie przeszkadzamy
            pass

    def _sync_from_json(self) -> None:
        """Kiedy użytkownik edytuje JSON – przepisz YAML (bez __preset_dir)."""
        try:
            d = json.loads(self.txt_json.get("1.0", "end"))
            d = _ensure_v2(d)
            d.pop("__preset_dir", None)
            self.txt_yaml.delete("1.0", "end")
            self.txt_yaml.insert("1.0", _dumps_yaml(d))
            self._rebuild_steps_listbox(d.get("steps") or [])
        except Exception:
            pass

    def _rebuild_steps_listbox(self, steps: List[dict]) -> None:
        self.lb_steps.delete(0, "end")
        for i, st in enumerate(steps):
            nm = st.get("name", "<unnamed>")
            self.lb_steps.insert("end", f"{i:02d}: {nm}")

    def _rebuild_history_listbox(self) -> None:
        self.lb_hist.delete(0, "end")
        for st in self.history:
            self.lb_hist.insert("end", st.get("name", "<unnamed>"))
