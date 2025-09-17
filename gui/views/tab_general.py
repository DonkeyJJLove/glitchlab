# glitchlab/gui/views/tab_general.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Iterable, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# opcjonalne
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageTk = None  # type: ignore


@dataclass
class GeneralTabConfig:
    """Ustawienia zakładki „General/Global”."""
    # Minimalny rozmiar miniatury (gdy host jest mniejszy).
    preview_size: int = 220
    # (Opcjonalny) maksymalny rozmiar miniatury. Gdy None – brak twardego limitu
    # poza bezpiecznym sufitem wewnętrznym (4096 px).
    preview_max_size: Optional[int] = None
    allow_add_from_file: bool = True
    allow_remove: bool = True
    allow_normalize: bool = True
    left_minwidth: int = 200
    left_default: int = 240


class GeneralTab(ttk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        ctx_ref: Optional[Any] = None,
        cfg: Optional[GeneralTabConfig] = None,
        bus: Optional[Any] = None,
        masks_service: Optional[Any] = None,
        on_masks_changed: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_mask_selected: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.ctx_ref = ctx_ref
        self.cfg = cfg or GeneralTabConfig()
        self.bus = bus
        self.svc = masks_service
        self.on_masks_changed = on_masks_changed
        self.on_mask_selected = on_mask_selected

        self._tk_preview: Optional[Any] = None
        self._current_preview_name: Optional[str] = None
        self._last_render_box: Tuple[int, int] = (0, 0)
        self._resize_job: Optional[str] = None

        self._build_ui()
        self._wire_bus()
        self.refresh()

    # ——— Public API ———
    def set_ctx(self, ctx_ref: Any) -> None:
        self.ctx_ref = ctx_ref
        self.refresh()

    def refresh(self) -> None:
        self._ensure_ctx()
        names = self._names()
        self.lst.delete(0, "end")
        for n in names:
            self.lst.insert("end", n)

        if names:
            target = self._current_preview_name or names[0]
            idx = self._index_of(target) or 0
            self.lst.selection_clear(0, "end")
            self.lst.selection_set(idx)
            self.lst.see(idx)
            self.lst.event_generate("<<ListboxSelect>>")
        else:
            self._current_preview_name = None
            self._clear_preview()

        self._sync_cache_keys()
        self._publish("ui.masks.refresh", {})

    # ——— UI ———
    def _build_ui(self) -> None:
        pad = dict(padx=6, pady=6)

        bar = ttk.Frame(self); bar.pack(fill="x", **pad)
        ttk.Label(bar, text="Masks in memory", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="right")
        if self.cfg.allow_normalize:
            ttk.Button(bar, text="Normalize 0..1→u8", command=self._on_normalize).pack(side="right", padx=(0, 6))
        if self.cfg.allow_remove:
            ttk.Button(bar, text="Remove", command=self._on_remove).pack(side="right", padx=(0, 6))
        if self.cfg.allow_add_from_file:
            ttk.Button(bar, text="Add mask…", command=self._on_add_from_file).pack(side="right", padx=(0, 6))

        self.split = ttk.Panedwindow(self, orient="horizontal"); self.split.pack(fill="both", expand=True, **pad)
        left = ttk.Frame(self.split)
        right = ttk.Frame(self.split)
        self.split.add(left, weight=0)
        self.split.add(right, weight=1)

        lst_box = ttk.Frame(left); lst_box.pack(fill="both", expand=True)
        self.lst = tk.Listbox(lst_box, exportselection=False, height=12)
        self.lst.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lst_box, command=self.lst.yview)
        sb.pack(side="right", fill="y"); self.lst.configure(yscrollcommand=sb.set)
        self.lst.bind("<<ListboxSelect>>", self._on_select)

        prev_box = ttk.Frame(right); prev_box.pack(fill="both", expand=True)
        self.preview_host = ttk.LabelFrame(prev_box, text="Preview")
        # najważniejsze: wyłącz propagację, żeby obraz nie „rozpychał” hosta
        self.preview_host.pack_propagate(False)
        self.preview_host.pack(fill="both", expand=True)

        self.preview = ttk.Label(self.preview_host, anchor="center")
        self.preview.pack(fill="both", expand=True, padx=6, pady=6)
        self.preview_host.bind("<Configure>", self._on_preview_resize)

        self.info = ttk.Label(prev_box, text="—", justify="left")
        self.info.pack(fill="x", pady=(6, 0))

        def _init_sash():
            try:
                total = self.split.winfo_width()
                if total < self.cfg.left_default + 100:
                    self.after(40, _init_sash); return
                self.split.sashpos(0, self.cfg.left_default)
            except Exception:
                pass
        self.after_idle(_init_sash)

    # ——— Events ———
    def _on_select(self, _evt=None) -> None:
        name = self._current_name()
        self._current_preview_name = name
        if not name:
            self._clear_preview(); return
        self._render_preview(self._get(name))
        self._publish("ui.masks.select", {"name": name})
        if callable(self.on_mask_selected):
            try: self.on_mask_selected(name)
            except Exception: pass

    def _on_preview_resize(self, _evt=None) -> None:
        if self._resize_job:
            try: self.after_cancel(self._resize_job)
            except Exception: pass

        def _do():
            name = self._current_preview_name or self._current_name()
            if not name: return
            arr = self._get(name)
            if arr is None: return
            # host aktualny
            host_w = max(1, int(self.preview_host.winfo_width()))
            host_h = max(1, int(self.preview_host.winfo_height()))
            # minimalny (cfg.preview_size) i opcjonalny maksymalny (cfg.preview_max_size)
            min_w = max(32, self.cfg.preview_size)
            min_h = max(32, self.cfg.preview_size)
            max_w = min(4096, self.cfg.preview_max_size or 4096)
            max_h = min(4096, self.cfg.preview_max_size or 4096)
            # box = host - padding, ale nie mniejszy niż min i nie większy niż max
            box_w = max(min_w, min(host_w - 12, max_w))
            box_h = max(min_h, min(host_h - 28, max_h))
            if (box_w, box_h) != self._last_render_box:
                self._render_preview(arr, (box_w, box_h))

        self._resize_job = self.after(60, _do)

    # ——— Actions ———
    def _on_add_from_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Add mask (image)",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff"), ("All files", "*.*")]
        )
        if not path: return
        self._publish("ui.masks.add_file", {"path": path})

        key: Optional[str] = None
        if self._has(self.svc, "add_from_file"):
            try: key = self.svc.add_from_file(path)  # type: ignore[attr-defined]
            except Exception: key = None
        if key is None:
            key = self._local_add_from_file(path)
        if key:
            self.refresh()
            self._select_by_name(key)

    def _on_remove(self) -> None:
        name = self._current_name()
        if not name: return
        self._publish("ui.masks.remove", {"name": name})
        if self._has(self.svc, "remove"):
            try: self.svc.remove(name)  # type: ignore[attr-defined]
            except Exception: pass
        else:
            self._ensure_ctx(); self.ctx_ref.masks.pop(name, None)
        self.refresh()

    def _on_normalize(self) -> None:
        if np is None: return
        name = self._current_name()
        if not name: return
        self._publish("ui.masks.normalize", {"name": name})
        if self._has(self.svc, "normalize"):
            try:
                arr = self.svc.normalize(name)  # type: ignore[attr-defined]
                if arr is not None:
                    self._ensure_ctx(); self.ctx_ref.masks[name] = arr
            except Exception:
                pass
        else:
            arr = self._get(name)
            if arr is None: return
            a = np.asarray(arr, dtype=np.float32)
            mn, mx = float(a.min()), float(a.max())
            rng = max(1e-9, mx - mn)
            u8 = np.clip((a - mn) * (255.0 / rng), 0, 255).astype(np.uint8)
            self._ensure_ctx(); self.ctx_ref.masks[name] = u8
        self._notify_masks_changed()
        self._render_preview(self._get(name))

    # ——— Rendering ———
    def _render_preview(self, arr: Optional["np.ndarray"], box_override: Optional[Tuple[int, int]] = None) -> None:
        if arr is None:
            self._clear_preview(); return
        h, w = int(arr.shape[0]), int(arr.shape[1])

        if np is not None:
            a = np.asarray(arr)
            mn, mx = float(a.min()), float(a.max())
            mean = float(a.mean())
            txt = f"{w}×{h}  dtype={a.dtype}  min={mn:.3f}  max={mx:.3f}  mean={mean:.3f}"
        else:
            txt = f"{w}×{h}  (NumPy unavailable)"
        self.info.configure(text=txt)

        if Image is None or ImageTk is None or np is None:
            self.preview.configure(text="(Preview requires Pillow & NumPy)", image="")
            self._tk_preview = None
            return

        img = Image.fromarray(self._to_u8_gray(arr)).convert("L")

        host_w = max(1, int(self.preview_host.winfo_width()))
        host_h = max(1, int(self.preview_host.winfo_height()))
        if box_override:
            box_w, box_h = box_override
        else:
            min_w = max(32, self.cfg.preview_size)
            min_h = max(32, self.cfg.preview_size)
            max_w = min(4096, self.cfg.preview_max_size or 4096)
            max_h = min(4096, self.cfg.preview_max_size or 4096)
            box_w = max(min_w, min(host_w - 12, max_w))
            box_h = max(min_h, min(host_h - 28, max_h))

        self._last_render_box = (box_w, box_h)

        scale = min(box_w / max(1, w), box_h / max(1, h))
        sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((sw, sh), Image.NEAREST)

        full = Image.new("L", (box_w, box_h), color=24)
        full.paste(img, ((box_w - sw) // 2, (box_h - sh) // 2))
        self._tk_preview = ImageTk.PhotoImage(full)
        self.preview.configure(image=self._tk_preview, text="")

    def _clear_preview(self) -> None:
        self.preview.configure(image="", text="(no mask selected)")
        self.info.configure(text="—")
        self._tk_preview = None

    # ——— Data helpers ———
    def _ensure_ctx(self) -> None:
        if self.ctx_ref is None:
            self.ctx_ref = type("Ctx", (), {})()
        if not hasattr(self.ctx_ref, "masks") or self.ctx_ref.masks is None:
            self.ctx_ref.masks = {}
        if not hasattr(self.ctx_ref, "cache") or self.ctx_ref.cache is None:
            self.ctx_ref.cache = {}

    def _names(self) -> List[str]:
        if self._has(self.svc, "list"):
            try:
                names = list(self.svc.list())  # type: ignore[attr-defined]
                return sorted([n for n in names if isinstance(n, str)])
            except Exception:
                pass
        try:
            names = list(getattr(self.ctx_ref, "masks", {}).keys())
            return sorted([n for n in names if isinstance(n, str)])
        except Exception:
            return []

    def _get(self, name: str) -> Optional["np.ndarray"]:
        if self._has(self.svc, "get"):
            try: return self.svc.get(name)  # type: ignore[attr-defined]
            except Exception: pass
        try: return getattr(self.ctx_ref, "masks", {}).get(name)
        except Exception: return None

    def _local_add_from_file(self, path: str) -> Optional[str]:
        if np is None:
            messagebox.showerror("Masks", "NumPy is required."); return None
        if Image is None:
            messagebox.showerror("Masks", "Pillow is required to load images."); return None
        try:
            img = Image.open(path)
            if img.mode not in ("L", "I;16"):
                img = img.convert("L")
            arr = np.array(img, dtype=np.uint8)
            key = self._unique_key_from_path(path)
            self._ensure_ctx(); self.ctx_ref.masks[key] = arr
            self._sync_cache_keys(); self._notify_masks_changed()
            return key
        except Exception as e:
            messagebox.showerror("Add mask", str(e)); return None

    def _sync_cache_keys(self) -> None:
        try:
            self.ctx_ref.cache["cfg/masks/keys"] = self._names()
        except Exception:
            pass

    # ——— Utils ———
    def _current_name(self) -> Optional[str]:
        sel = self.lst.curselection()
        return None if not sel else self.lst.get(sel[0])

    def _index_of(self, name: str) -> Optional[int]:
        for i in range(self.lst.size()):
            if self.lst.get(i) == name:
                return i
        return None

    def _select_by_name(self, name: str) -> None:
        idx = self._index_of(name)
        if idx is None: return
        self.lst.selection_clear(0, "end")
        self.lst.selection_set(idx)
        self.lst.see(idx)
        self.lst.event_generate("<<ListboxSelect>>")

    def _unique_key_from_path(self, path: str) -> str:
        base = os.path.splitext(os.path.basename(path))[0] or "mask"
        name, i, existing = base, 1, set(self._names())
        while name in existing:
            i += 1; name = f"{base}_{i}"
        return name

    def _to_u8_gray(self, arr: "np.ndarray") -> "np.ndarray":
        a = np.asarray(arr)
        if a.ndim == 3:
            a = a[..., 0]
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        return a

    def _notify_masks_changed(self) -> None:
        if callable(self.on_masks_changed):
            try: self.on_masks_changed(getattr(self.ctx_ref, "masks", {}) or {})
            except Exception: pass

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self.bus is not None and hasattr(self.bus, "publish"):
            try: self.bus.publish(topic, dict(payload))
            except Exception: pass

    @staticmethod
    def _has(obj: Any, name: str) -> bool:
        return obj is not None and hasattr(obj, name) and callable(getattr(obj, name))

    # ——— Bus wiring ———
    def _wire_bus(self) -> None:
        if self.bus is None or not hasattr(self.bus, "subscribe"):
            return

        def _try_refresh(_t: str, _d: Dict[str, Any]) -> None:
            try: self.refresh()
            except Exception: pass

        try:
            self.bus.subscribe("masks.changed", _try_refresh)
            self.bus.subscribe("masks.added", _try_refresh)
            self.bus.subscribe("masks.removed", _try_refresh)
            self.bus.subscribe("masks.loaded", _try_refresh)
            self.bus.subscribe("preset.parsed", _try_refresh)
            self.bus.subscribe("run.done", _try_refresh)
            self.bus.subscribe("run.error", _try_refresh)
            self.bus.subscribe("ui.image.loaded", _try_refresh)
        except Exception:
            pass
