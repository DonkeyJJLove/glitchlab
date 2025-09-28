# glitchlab/gui/views/hud.py
# -*- coding: utf-8 -*-
"""
HUDView — 3-slotowy HUD (miniatury + podpisy) z overlayem.

Zmiany vs. poprzednie:
• render_from_cache akceptuje zarówno obiekt ctx (z atrybutem .cache), jak i goły dict (cache).
• Bezpieczniejsze generowanie miniatur (obsługa PIL/ndarray/bytes), lepsze skalowanie i trzymanie referencji.
• Deterministyczny wybór źródeł przez mosaic.router (fallback, gdy brak modułu).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List

import tkinter as tk
from tkinter import ttk

# --- NumPy (opcjonalnie) ----------------------------------------------------
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore[assignment]

# --- PIL (opcjonalnie: miniatury/konwersje) ---------------------------------
try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore

# --- Mosaic Router (preferowany) --------------------------------------------
try:
    # właściwa ścieżka w projekcie
    from glitchlab.gui.mosaic.router import resolve_selection  # type: ignore
except Exception:
    # awaryjny fallback: puste wybory
    def resolve_selection(_spec, _keys, **_kwargs):
        return ({"slot1": None, "slot2": None, "slot3": None, "overlay": None}, {}, {"fallback": True})


__all__ = ["HUDView"]


# ============================================================================

def _is_nd(x: Any) -> bool:
    return (np is not None) and isinstance(x, np.ndarray)  # type: ignore[arg-type]


def _is_pil(x: Any) -> bool:
    return (Image is not None) and isinstance(x, Image.Image)  # type: ignore[attr-defined]


def _to_pil_rgb(value: Any) -> Optional["Image.Image"]:
    """
    Bezpieczna konwersja: PIL.Image | np.ndarray | bytes -> PIL.Image(RGB)
    Zwraca None, gdy nie potrafi zinterpretować wartości jako obrazu.
    """
    if Image is None:
        return None
    try:
        # PIL bezpośrednio
        if _is_pil(value):
            im = value  # type: ignore[assignment]
            return im.convert("RGB") if getattr(im, "mode", "") != "RGB" else im

        # NumPy array
        if _is_nd(value):
            arr = value  # type: ignore[assignment]
            # normalizacja typu/zakresu
            if getattr(arr, "dtype", None) != (np.uint8 if np is not None else None):  # type: ignore[comparison-overlap]
                a = arr.astype("float32")
                # heurystyka: jeśli max ≤ 1.001, to skala [0..1] -> [0..255]
                try:
                    if a.size and float(a.max()) <= 1.001:
                        a = a * 255.0
                except Exception:
                    pass
                arr = a.clip(0, 255).astype("uint8")
            # kanały
            if arr.ndim == 2:
                arr = (np.dstack([arr, arr, arr]) if np is not None else arr)  # type: ignore[operator]
            if arr.ndim == 3:
                c = arr.shape[-1]
                if c == 1 and np is not None:
                    arr = np.repeat(arr, 3, axis=-1)  # type: ignore[operator]
                elif c == 4:
                    # RGBA -> RGB na czarnym tle (deterministycznie)
                    try:
                        im = Image.fromarray(arr, "RGBA")  # type: ignore[arg-type]
                        bg = Image.new("RGB", im.size, (0, 0, 0))
                        bg.paste(im, mask=im.split()[-1])
                        return bg
                    except Exception:
                        arr = arr[..., :3]
                if arr.shape[-1] == 3:
                    return Image.fromarray(arr, "RGB")  # type: ignore[arg-type]

        # bytes/bytearray (np. PNG/JPEG)
        if isinstance(value, (bytes, bytearray)):
            from io import BytesIO
            try:
                return Image.open(BytesIO(value)).convert("RGB")
            except Exception:
                return None
    except Exception:
        return None
    return None


def _make_thumb(img: "Image.Image", max_side: int = 160) -> "Image.Image":
    """Zachowaj proporcje, użyj wysokiej jakości resamplingu."""
    if Image is None:
        return img
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    try:
        return img.resize((nw, nh), resample=getattr(Image, "LANCZOS", 1))
    except Exception:
        return img.resize((nw, nh))


def _fit_into(img: "Image.Image", W: int, H: int) -> "Image.Image":
    """Dopasuj obraz do prostokąta (do podglądu w oknie „View…”)."""
    if Image is None:
        return img
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = max(1e-3, min(W / float(w), H / float(h)))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    try:
        return img.resize((nw, nh), resample=getattr(Image, "LANCZOS", 1))
    except Exception:
        return img.resize((nw, nh))


def _short_text(v: Any, max_len: int = 180) -> str:
    """Zwięzły opis wartości nie-obrazowych."""
    try:
        if v is None:
            return "None"
        if _is_nd(v):
            shape = getattr(v, "shape", None)
            dt = getattr(v, "dtype", None)
            return f"ndarray shape={tuple(shape)} dtype={str(dt)}"
        if _is_pil(v):
            return f"PIL.Image size={v.size} mode={getattr(v, 'mode', '?')}"
        if isinstance(v, (bytes, bytearray)):
            return f"bytes[{len(v)}]"
        if isinstance(v, (list, tuple, dict)):
            import json
            try:
                s = json.dumps(v, ensure_ascii=False)
            except Exception:
                s = str(v)
            return s if len(s) <= max_len else s[: max_len - 1] + "…"
        s = str(v)
        return s if len(s) <= max_len else s[: max_len - 1] + "…"
    except Exception:
        return "<unrenderable>"


# ============================================================================

@dataclass
class _History:
    slot1: Optional[str] = None
    slot2: Optional[str] = None
    slot3: Optional[str] = None


class _HudSlot(ttk.LabelFrame):
    """Pojedynczy slot HUD z miniaturą, podpisem i narzędziami (View…, Copy key)."""

    def __init__(self, master, title: str):
        super().__init__(master, text=title)
        self._thumb = ttk.Label(self, anchor="center")
        self._thumb.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 0))

        self._text = ttk.Label(self, text="—", anchor="w", justify="left", wraplength=420)
        self._text.grid(row=1, column=0, sticky="ew", padx=6, pady=(4, 6))

        tools = ttk.Frame(self)
        tools.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))
        self._lbl_key = ttk.Label(tools, text="", foreground="#777")
        self._lbl_key.pack(side="left")

        self._btn_copy = ttk.Button(tools, text="Copy key", width=9, command=self._copy_key)
        self._btn_copy.pack(side="right", padx=(6, 0))
        self._btn_view = ttk.Button(tools, text="View…", width=7, command=self._open_viewer)
        self._btn_view.pack(side="right")

        self._cur_key: Optional[str] = None
        self._cur_val: Any = None
        self._tk_img: Optional["ImageTk.PhotoImage"] = None  # trzymamy referencję

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    # -- API --

    def clear(self) -> None:
        self._cur_key = None
        self._cur_val = None
        self._tk_img = None
        self._thumb.configure(image="", text="(no preview)")
        self._text.configure(text="—")
        self._lbl_key.configure(text="")

    def set_content(self, key: Optional[str], value: Any) -> None:
        self._cur_key = key
        self._cur_val = value
        self._lbl_key.configure(text=(key or ""))

        # opis
        self._text.configure(text=_short_text(value))

        # miniatura
        img = _to_pil_rgb(value)
        if img is not None and ImageTk is not None:
            try:
                t = _make_thumb(img, max_side=160)
                self._tk_img = ImageTk.PhotoImage(t)
                self._thumb.configure(image=self._tk_img, text="")
            except Exception:
                self._tk_img = None
                self._thumb.configure(image="", text="(no preview)")
        else:
            self._tk_img = None
            self._thumb.configure(image="", text="(no preview)")

    # -- actions --

    def _copy_key(self) -> None:
        try:
            if not self._cur_key:
                return
            self.clipboard_clear()
            self.clipboard_append(self._cur_key)
        except Exception:
            pass

    def _open_viewer(self) -> None:
        if self._cur_key is None:
            return
        v = self._cur_val
        win = tk.Toplevel(self)
        win.title(self._cur_key)
        win.geometry("640x520")

        img = _to_pil_rgb(v)
        if img is not None and ImageTk is not None:
            frm = ttk.Frame(win)
            frm.pack(fill="both", expand=True)
            cv = tk.Canvas(frm, bg="#101010", highlightthickness=0)
            cv.pack(fill="both", expand=True)

            def redraw(_evt=None):
                try:
                    W, H = max(1, cv.winfo_width()), max(1, cv.winfo_height())
                    thumb = _fit_into(img, W, H)
                    tkimg = ImageTk.PhotoImage(thumb)
                    cv.delete("all")
                    cv.create_image(W // 2, H // 2, image=tkimg, anchor="center")
                    cv._img = tkimg  # keep ref
                except Exception:
                    pass

            cv.bind("<Configure>", redraw)
            redraw()
        else:
            txt = tk.Text(win, wrap="word", bg="#141414", fg="#e6e6e6", insertbackground="#e6e6e6")
            txt.pack(fill="both", expand=True)
            try:
                import json
                if isinstance(v, (dict, list, tuple)):
                    txt.insert("1.0", json.dumps(v, ensure_ascii=False, indent=2))
                else:
                    txt.insert("1.0", str(v))
            except Exception:
                txt.insert("1.0", "<unrenderable>")

        ttk.Label(win, text=self._cur_key, foreground="#888").pack(side="bottom", anchor="w", padx=8, pady=6)


# ============================================================================

_DEFAULT_SPEC: Dict[str, Any] = {
    "slots": {
        "slot1": ["stage/0/in", "stage/*/in", "stage/*/metrics_in", "format/jpg_grid"],
        "slot2": ["stage/0/out", "stage/*/out", "stage/*/metrics_out", "stage/*/fft_mag"],
        "slot3": ["stage/0/diff", "stage/*/diff", "stage/*/diff_stats", "ast/json"],
    },
    "overlay": ["stage/*/mosaic", "diag/*/*"],
}


class HUDView(ttk.Frame):
    """
    3-slotowy HUD:
      • wybór kluczy z ctx.cache przez Mosaic Router (spec konfigurowalny),
      • miniatury dla obrazów (PIL/ndarray/bytes), tekst dla reszty,
      • podgląd „View…” i „Copy key”.

    API:
      • render_from_cache(ctx_or_cache)
      • set_spec(spec_dict) / get_spec()
      • set_history(mapping)  — opcjonalnie (utrzymanie preferencji)
    """

    def __init__(self, master, *, spec: Optional[Mapping[str, Any]] = None):
        super().__init__(master)
        self._spec: Dict[str, Any] = dict(spec or _DEFAULT_SPEC)
        self._history = _History()

        # layout 2 wiersze: [slots][overlay]
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        self._slot1 = _HudSlot(self, "Slot 1"); self._slot1.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self._slot2 = _HudSlot(self, "Slot 2"); self._slot2.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        self._slot3 = _HudSlot(self, "Slot 3"); self._slot3.grid(row=0, column=2, sticky="nsew", padx=6, pady=6)

        self._overlay = ttk.LabelFrame(self, text="Overlay")
        self._overlay.grid(row=1, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 6))
        self._overlay_lbl = ttk.Label(self._overlay, text="—", foreground="#777", anchor="w")
        self._overlay_lbl.pack(fill="x", padx=6, pady=6)

    # ---- konfiguracja ----

    def set_spec(self, spec: Mapping[str, Any]) -> None:
        self._spec = dict(spec or {})

    def get_spec(self) -> Dict[str, Any]:
        return dict(self._spec)

    def set_history(self, mapping: Mapping[str, str]) -> None:
        self._history = _History(
            slot1=mapping.get("slot1"),  # type: ignore[arg-type]
            slot2=mapping.get("slot2"),  # type: ignore[arg-type]
            slot3=mapping.get("slot3"),  # type: ignore[arg-type]
        )

    # ---- render ----

    def render_from_cache(self, ctx_or_cache: Any) -> None:
        """
        Przyjmuje:
          • ctx (z atrybutem `.cache`)
          • dict — bezpośrednio cache
        """
        try:
            if isinstance(ctx_or_cache, dict):
                cache: Dict[str, Any] = ctx_or_cache
            else:
                cache = getattr(ctx_or_cache, "cache", {}) or {}
                if not isinstance(cache, dict):
                    cache = {}

            keys = list(cache.keys())

            sel, _rank, _exp = resolve_selection(
                self._spec,
                keys,
                history={"slot1": self._history.slot1, "slot2": self._history.slot2, "slot3": self._history.slot3},
            )

            k1, k2, k3 = sel.get("slot1"), sel.get("slot2"), sel.get("slot3")
            if k1: self._slot1.set_content(k1, cache.get(k1))
            else:  self._slot1.clear()
            if k2: self._slot2.set_content(k2, cache.get(k2))
            else:  self._slot2.clear()
            if k3: self._slot3.set_content(k3, cache.get(k3))
            else:  self._slot3.clear()

            kov = sel.get("overlay")
            self._overlay_lbl.configure(text=(kov or "—"))

            # utrzymanie historii (jeśli coś wybrano)
            self._history.slot1 = k1 or self._history.slot1
            self._history.slot2 = k2 or self._history.slot2
            self._history.slot3 = k3 or self._history.slot3

        except Exception as ex:
            # pokaż komunikat błędu w slotach, ale nie wysypuj UI
            for s in (self._slot1, self._slot2, self._slot3):
                s.set_content(None, f"HUD error: {ex}")
            try:
                self._overlay_lbl.configure(text="—")
            except Exception:
                pass
