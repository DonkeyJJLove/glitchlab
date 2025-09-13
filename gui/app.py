# glitchlab/gui/app.py
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --- GUI infra ---
from glitchlab.gui.event_bus import EventBus
from glitchlab.gui.state import UiState, reduce as reduce_state, Selectors
from glitchlab.gui.views.menu import MenuBar
from glitchlab.gui.views.viewport import Viewport
from glitchlab.gui.views.notebook import RightNotebook
from glitchlab.gui.views.hud import HUDView

# --- Services (files / runner) ---
from glitchlab.gui.services.files import load_image, save_image, default_recent_store
from glitchlab.gui.services.pipeline_runner import PipelineRunner

# --- MaskService (miękkie) ---
try:
    from glitchlab.gui.services.masks import MaskService  # type: ignore
except Exception:
    class MaskService:  # łagodny fallback
        def __init__(self, *_args, **_kw):
            self.masks: Dict[str, np.ndarray] = {}
        def add_from_file(self, path: str, key: Optional[str] = None) -> str:
            from PIL import Image
            im = Image.open(path).convert("L")
            arr = np.asarray(im, dtype=np.float32) / 255.0
            k = key or Path(path).stem
            self.masks[k] = arr
            return k
        def clear(self) -> None:
            self.masks.clear()
        def keys(self) -> List[str]:
            return list(self.masks.keys())
        def as_dict(self) -> Dict[str, np.ndarray]:
            return dict(self.masks)

def _init_mask_service(root_like, bus) -> MaskService:
    """
    Tworzy MaskService niezależnie od sygnatury konstruktora.
    Obsługiwane: (root_like, bus), (root_like), (bus), ().
    """
    try:
        return MaskService(root_like, bus)               # positional
    except TypeError:
        pass
    try:
        return MaskService(root_like=root_like, bus=bus)  # kw
    except TypeError:
        pass
    try:
        return MaskService(root_like)                    # only root
    except TypeError:
        pass
    try:
        return MaskService(bus=bus)                      # only bus
    except TypeError:
        pass
    return MaskService()                                 # bare fallback


# --- Core (miękkie zależności) ---
try:
    from glitchlab.core.pipeline import build_ctx, apply_pipeline, normalize_preset  # type: ignore
except Exception:
    def build_ctx(img_u8, *, seed=None, cfg=None):
        # minimalny kontekst zgodny z oczekiwaniami HUD
        return type("Ctx", (), {"rng": None, "amplitude": None, "masks": {}, "cache": {}, "meta": {}})()
    def apply_pipeline(img_u8, ctx, steps, *, fail_fast=True, debug_log=None, metrics=True):
        return img_u8
    def normalize_preset(cfg):
        return {"version": 2, "steps": []}

try:
    from glitchlab.core.registry import available as registry_available  # type: ignore
except Exception:
    def registry_available() -> List[str]:
        return []


# ---------------------------------
# Pomocnicze
# ---------------------------------
def _np_u8(img: Any) -> Optional[np.ndarray]:
    if img is None:
        return None
    if isinstance(img, np.ndarray) and img.dtype == np.uint8:
        return img
    try:
        arr = np.asarray(img)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            # flatten alpha na czarne tło
            a = (arr[..., 3:4].astype(np.float32) / 255.0)
            rgb = (arr[..., :3].astype(np.float32) * a + 0.5).astype(np.uint8)
            return rgb
        return arr
    except Exception:
        return None


# ---------------------------------
# StatusBar (lekki, lokalny)
# ---------------------------------
class StatusBar(ttk.Frame):
    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self._prog = ttk.Progressbar(self, mode="indeterminate", length=140)
        self._lbl = ttk.Label(self, text="Ready")
        self._btn = ttk.Button(self, text="Show log", command=self._show_log)
        self._log_text: Optional[tk.Text] = None
        self._log_win: Optional[tk.Toplevel] = None

        self._prog.pack(side="left", padx=(6, 4), pady=2)
        self._lbl.pack(side="left", padx=(6, 4))
        self._btn.pack(side="right", padx=(4, 6))

    def set_text(self, txt: str) -> None:
        try: self._lbl.config(text=txt)
        except Exception: pass

    def start(self) -> None:
        try: self._prog.start(80)
        except Exception: pass

    def stop(self) -> None:
        try: self._prog.stop()
        except Exception: pass

    def log(self, line: str) -> None:
        if self._log_text is None or not self._log_text.winfo_exists():
            return
        try:
            self._log_text.insert("end", line + "\n")
            self._log_text.see("end")
        except Exception:
            pass

    def _show_log(self) -> None:
        if self._log_win and self._log_win.winfo_exists():
            self._log_win.lift(); return
        win = tk.Toplevel(self)
        win.title("Tech log"); win.geometry("800x280")
        txt = tk.Text(win, bg="#161616", fg="#e8e8e8", insertbackground="#e8e8e8")
        txt.pack(fill="both", expand=True)
        self._log_win = win; self._log_text = txt
        self.log("--- started ---")


# ---------------------------------
# AppShell
# ---------------------------------
class AppShell(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master
        self.master.title("GlitchLab")
        self.pack(fill="both", expand=True)

        # infra
        self.bus = EventBus(self.master)
        self.state = UiState()
        self.sel = Selectors()
        self.runner = PipelineRunner(self.master)
        self.masks = _init_mask_service(self.master, self.bus)

        # runtime
        self._last_ctx: Any = type("Ctx", (), {"rng": None, "amplitude": None, "masks": {}, "cache": {}, "meta": {}})()
        self._recent = default_recent_store()

        # UI
        self.menu: Optional[MenuBar] = None
        self.status: Optional[StatusBar] = None
        self.viewport: Optional[Viewport] = None
        self.notebook: Optional[RightNotebook] = None
        self.hud: Optional[HUDView] = None

        # build + wire + bootstrap
        self._build_ui()
        self._wire_events()
        self._bootstrap_lists()

    # -----------------------------
    # UI
    # -----------------------------
    def _build_ui(self) -> None:
        # menu
        self.menu = MenuBar(self.master, bus=self.bus)

        # split główny: hsplit -> [viewer_left | right_tabs]
        hsplit = ttk.Panedwindow(self, orient="horizontal")
        hsplit.pack(fill="both", expand=True)

        # left (viewer + bottom HUD)
        left = ttk.Frame(hsplit); hsplit.add(left, weight=3)
        vsplit = ttk.Panedwindow(left, orient="vertical"); vsplit.pack(fill="both", expand=True)

        # viewer
        viewer_host = ttk.Frame(vsplit); vsplit.add(viewer_host, weight=10)
        self.viewport = Viewport(viewer_host)
        self.viewport.pack(fill="both", expand=True)

        # bottom HUD
        hud_host = ttk.Frame(vsplit); vsplit.add(hud_host, weight=1)
        self.hud = HUDView(hud_host)
        self.hud.pack(fill="both", expand=True)

        # right (tabs)
        right = ttk.Frame(hsplit); hsplit.add(right, weight=2)
        self.notebook = RightNotebook(right, bus=self.bus)
        self.notebook.pack(fill="both", expand=True)

        # status
        self.status = StatusBar(self)
        self.status.pack(side="bottom", fill="x")

        # skróty
        self.master.bind("<Control-o>", lambda _e=None: self.bus.publish("files.open", {}))
        self.master.bind("<Control-s>", lambda _e=None: self.bus.publish("files.save", {}))

    # -----------------------------
    # Event wiring
    # -----------------------------
    def _wire_events(self) -> None:
        b = self.bus

        # Files
        b.subscribe("files.open", self._ev_open, on_ui=True)
        b.subscribe("files.save", self._ev_save, on_ui=True)

        # Help / Global
        b.subscribe("ui.help.about", self._ev_about, on_ui=True)
        b.subscribe("ui.global.seed_changed", self._ev_seed_changed, on_ui=True)

        # Masks
        b.subscribe("ui.masks.add_request", self._ev_mask_add_request, on_ui=True)
        b.subscribe("ui.masks.clear_request", self._ev_mask_clear_request, on_ui=True)

        # Filter/Run (z notebooka)
        b.subscribe("ui.filter.select", self._ev_filter_select, on_ui=True)
        b.subscribe("ui.filter.apply", self._ev_apply_filter, on_ui=True)

        # Presets
        b.subscribe("ui.presets.open_request", self._ev_preset_open_request, on_ui=True)
        b.subscribe("ui.presets.save_request", self._ev_preset_save_request, on_ui=True)
        b.subscribe("ui.presets.apply", self._ev_apply_preset, on_ui=True)

        # Telemetria runnera (opcjonalnie podpinane przy submit)
        def _on_prog(_jid: str, percent: float, msg: Optional[str]) -> None:
            if self.status:
                self.status.set_text(f"Working… {int(percent)}%{(' — ' + msg) if msg else ''}")
                if percent <= 1.0:
                    self.status.start()
        def _on_done(_jid: str, _result: Any) -> None:
            if self.status:
                self.status.stop(); self.status.set_text("Ready")
        def _on_err(_jid: str, exc: BaseException) -> None:
            if self.status:
                self.status.stop(); self.status.set_text("Failed")
                self.status.log(f"[runner] error: {exc!r}")
        self._runner_on_progress = _on_prog  # noqa: SLF001
        self._runner_on_done = _on_done      # noqa: SLF001
        self._runner_on_err = _on_err        # noqa: SLF001

    def _bootstrap_lists(self) -> None:
        # filtry
        names = sorted(registry_available() or [])
        self.bus.publish("filters.available", {"names": names, "select": (names[0] if names else None)})
        # maski (z serwisu)
        try:
            keys = self.masks.keys() if hasattr(self.masks, "keys") else []
        except Exception:
            keys = []
        self.bus.publish("masks.list", {"names": list(keys)})
        # seed do UI
        if self.notebook:
            self.notebook.set_seed(self.state.seed)

    # -----------------------------
    # Handlery BUS
    # -----------------------------
    def _ev_about(self, _t: str, _d: Dict[str, Any]) -> None:
        messagebox.showinfo("About GlitchLab", "GlitchLab — controlled glitch for analysis\nGUI v3 shell")

    def _ev_seed_changed(self, _t: str, data: Dict[str, Any]) -> None:
        try:
            self.state.seed = int(data.get("seed", self.state.seed))
            if self.status:
                self.status.set_text(f"Seed set to {self.state.seed}")
        except Exception:
            pass

    def _ev_filter_select(self, _t: str, data: Dict[str, Any]) -> None:
        nm = data.get("name")
        if isinstance(nm, str) and nm and self.status:
            self.status.set_text(f"Filter selected: {nm}")

    # --- FILES ---
    def _ev_open(self, _t: str, _d: Dict[str, Any]) -> None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            im = load_image(path)
            arr = _np_u8(im)
            if arr is None:
                raise ValueError("Unsupported image format")
            # update state
            self.state = reduce_state(self.state, ("files.opened", {"path": path}))
            self.state.image_in = arr
            self.state.image_out = arr
            # recent
            try: self._recent.touch(path)
            except Exception: pass
            # show
            if self.viewport:
                self.viewport.set_image(arr)
            # reset ctx/cache
            cfg = normalize_preset(self.state.preset_cfg or {"version": 2, "steps": []})
            self._last_ctx = build_ctx(arr, seed=self.state.seed, cfg=cfg)
            self._last_ctx.cache.clear()
            # HUD refresh
            self._refresh_hud()
            # status
            if self.status:
                self.status.set_text(f"Opened: {path}")
                self.status.log(f"Opened: {path}")
        except Exception as ex:
            if self.status:
                self.status.log("Open failed:\n" + traceback.format_exc())
            messagebox.showerror("Open image", str(ex))

    def _ev_save(self, _t: str, _d: Dict[str, Any]) -> None:
        img = self.sel.current_image(self.state) or self.state.image_out or self.state.image_in
        if img is None:
            messagebox.showinfo("Save", "Brak obrazu do zapisu."); return
        path = filedialog.asksaveasfilename(
            title="Save image as",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("All files", "*.*")],
        )
        if not path: return
        try:
            save_image(img, path)
            if self.status:
                self.status.set_text(f"Saved: {path}")
                self.status.log(f"Saved: {path}")
        except Exception as ex:
            if self.status:
                self.status.log("Save failed:\n" + traceback.format_exc())
            messagebox.showerror("Save image", str(ex))

    # --- MASKS ---
    def _ev_mask_add_request(self, _t: str, _d: Dict[str, Any]) -> None:
        path = filedialog.askopenfilename(
            title="Add mask",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            # dodaj do serwisu (różne API — spróbujmy kilku nazw)
            key = None
            for meth in ("add_from_file", "load_from_file", "add_file", "add"):
                fn = getattr(self.masks, meth, None)
                if callable(fn):
                    res = fn(path)  # może zwrócić key lub None
                    key = res if isinstance(res, str) else (Path(path).stem)
                    break
            if key is None:
                key = Path(path).stem
                # fallback: ręczne doładowanie do .masks (jeśli istnieje)
                from PIL import Image
                arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
                if hasattr(self.masks, "masks") and isinstance(self.masks.masks, dict):
                    self.masks.masks[key] = arr  # type: ignore[attr-defined]
            # przenieś maski do ctx (jeśli już mamy kontekst)
            if hasattr(self.masks, "as_dict"):
                try:
                    all_masks = self.masks.as_dict()
                except Exception:
                    all_masks = {}
            else:
                all_masks = getattr(self.masks, "masks", {}) if hasattr(self.masks, "masks") else {}
            if isinstance(all_masks, dict):
                self._last_ctx.masks = dict(all_masks)
            # powiadom notebook o liście masek
            self.bus.publish("masks.list", {"names": list(all_masks.keys())})
            # odśwież HUD (cache może zawierać np. cfg/masks/keys)
            try:
                self._last_ctx.cache["cfg/masks/keys"] = list(all_masks.keys())
            except Exception:
                pass
            self._refresh_hud()
            if self.status:
                self.status.set_text(f"Mask added: {key}")
        except Exception as ex:
            messagebox.showerror("Mask", str(ex))

    def _ev_mask_clear_request(self, _t: str, _d: Dict[str, Any]) -> None:
        try:
            if hasattr(self.masks, "clear"):
                self.masks.clear()
            self._last_ctx.masks = {}
            try:
                self._last_ctx.cache["cfg/masks/keys"] = []
            except Exception:
                pass
            self.bus.publish("masks.list", {"names": []})
            self._refresh_hud()
            if self.status:
                self.status.set_text("Masks cleared")
        except Exception as ex:
            messagebox.showerror("Mask", str(ex))

    # --- PRESETS ---
    def _ev_preset_open_request(self, _t: str, _d: Dict[str, Any]) -> None:
        try:
            path = filedialog.askopenfilename(
                title="Open preset",
                filetypes=[("YAML/JSON", "*.yaml;*.yml;*.json;*.txt"), ("All files", "*.*")],
            )
            if not path: return
            text = Path(path).read_text(encoding="utf-8")
            self.bus.publish("preset.loaded", {"text": text})
            if self.notebook:
                self.notebook.set_preset_text(text)
                self.notebook.set_preset_status(f"Loaded: {path}")
        except Exception as ex:
            messagebox.showerror("Preset", str(ex))

    def _ev_preset_save_request(self, _t: str, data: Dict[str, Any]) -> None:
        text = data.get("text")
        if not isinstance(text, str) and self.notebook:
            text = self.notebook.get_preset_text()
        path = filedialog.asksaveasfilename(
            title="Save preset as",
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml;*.yml"), ("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path: return
        try:
            Path(path).write_text(text or "", encoding="utf-8")
            if self.notebook:
                self.notebook.set_preset_status(f"Saved: {path}")
        except Exception as ex:
            messagebox.showerror("Preset", str(ex))

    def _ev_apply_preset(self, _t: str, data: Dict[str, Any]) -> None:
        text = data.get("text")
        if not isinstance(text, str) and self.notebook:
            text = self.notebook.get_preset_text()
        # parse YAML/JSON
        cfg = None
        try:
            try:
                import yaml  # type: ignore
                cfg = yaml.safe_load(text) if text else {"version": 2, "steps": []}
            except Exception:
                cfg = json.loads(text) if text else {"version": 2, "steps": []}
        except Exception as ex:
            messagebox.showerror("Preset", f"Nie można zinterpretować preset-u:\n{ex}")
            return
        try:
            cfg = normalize_preset(cfg or {"version": 2, "steps": []})
        except Exception as ex:
            messagebox.showerror("Preset", f"Preset nieprawidłowy:\n{ex}")
            return
        steps = list(cfg.get("steps") or [])
        if not steps:
            messagebox.showinfo("Preset", "Preset nie ma kroków."); return
        self._run_steps(steps, label="Apply preset", cfg_override=cfg)

    # --- FILTER/RUN ---
    def _ev_apply_filter(self, _t: str, data: Dict[str, Any]) -> None:
        name = data.get("name")
        params = data.get("params") or {}
        if not isinstance(name, str) or not name:
            messagebox.showinfo("Filter", "Wybierz filtr."); return
        step = {"name": name, "params": dict(params)}
        self._run_steps([step], label=f"Apply filter: {name}")

    def _ev_run_request(self, _t: str, payload: Dict[str, Any]) -> None:
        steps = payload.get("steps", [])
        if not steps:
            messagebox.showinfo("Run", "Brak kroków do uruchomienia."); return
        cfg = payload.get("cfg", None)
        self._run_steps(steps, label="Run", cfg_override=cfg)

    # -----------------------------
    # Running pipeline
    # -----------------------------
    def _run_steps(self, steps: List[Dict[str, Any]], *, label: str, cfg_override: Optional[Dict[str, Any]] = None) -> None:
        img = self.state.image_out or self.state.image_in
        if img is None:
            messagebox.showinfo("Run", "Najpierw otwórz obraz."); return

        cfg = normalize_preset(cfg_override or self.state.preset_cfg or {"version": 2, "steps": []})
        ctx = build_ctx(img, seed=self.state.seed, cfg=cfg)
        # wstrzyknij maski z serwisu (jeśli są)
        try:
            if hasattr(self.masks, "as_dict"):
                ctx.masks = dict(self.masks.as_dict())
            elif hasattr(self.masks, "masks"):
                ctx.masks = dict(getattr(self.masks, "masks"))
        except Exception:
            pass
        try:
            ctx.cache["cfg/masks/keys"] = list(ctx.masks.keys())
        except Exception:
            pass

        if self.status:
            self.status.set_text(label); self.status.start()

        self._last_job_id = self.runner.submit(  # noqa: SLF001
            label,
            lambda: apply_pipeline(img, ctx, steps, fail_fast=True, debug_log=[], metrics=True),
            on_done=lambda _jid, _out: self._after_run_success(ctx, _out),
            on_error=lambda _jid, exc: self._after_run_error(exc),
            on_progress=self._runner_on_progress,  # type: ignore[attr-defined]
        )

    def _after_run_success(self, ctx: Any, out_img: Any) -> None:
        try:
            arr = _np_u8(out_img)
            if arr is None:
                raise ValueError("Pipeline returned invalid image")
            self.state.image_out = arr
            self._last_ctx = ctx
            if self.viewport:
                self.viewport.set_image(arr)
            self._refresh_hud()
            if self.status:
                self.status.stop(); self.status.set_text("Done"); self.status.log("Run OK")
        except Exception as ex:
            self._after_run_error(ex)

    def _after_run_error(self, exc: BaseException) -> None:
        if self.status:
            self.status.stop(); self.status.set_text("Failed")
            self.status.log("Run failed:\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        messagebox.showerror("Run failed", str(exc))

    def _refresh_hud(self) -> None:
        try:
            if self.hud and hasattr(self.hud, "render_from_cache"):
                self.hud.render_from_cache(self._last_ctx)
        except Exception:
            pass


# ---------------------------------
# Bootstrap
# ---------------------------------
def main() -> None:
    root = tk.Tk()
    app = AppShell(root)
    root.geometry("1280x860")
    root.minsize(900, 600)
    root.mainloop()


if __name__ == "__main__":
    main()
