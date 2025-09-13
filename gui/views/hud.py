"""
---
version: 3
kind: module
id: "gui-views-hud"
created_at: "2025-09-13"
name: "glitchlab.gui.views.hud"
author: "GlitchLab v3"
role: "3-slot HUD (miniatury + podpisy) z routingiem mozaiki"
description: >
  Renderuje trzy sloty HUD (slot1..slot3) oraz informacyjny overlay, wybierając
  źródła danych z ctx.cache wg MosaicSpec poprzez resolver (mosaic.router).
  Dla wartości obrazowych generuje miniatury (PIL/ndarray), dla pozostałych
  pokazuje skrócony opis. Umożliwia szybki podgląd („View…”) i skopiowanie klucza.

inputs:
  ctx.cache:   {type: "Mapping[str, Any]", desc: "źródła danych (obrazy/metryki/teksty…)"}
  spec:        {type: "Mapping", default: "domyślny kanon", desc: "wzorce slotów i overlay"}
  history:     {type: "Mapping[str,str]", optional: true, desc: "ostatnie wybory per slot (bias)"}

outputs:
  ui:          {type: "None", note: "widget nie publikuje zdarzeń – tylko renderuje"}

interfaces:
  exports:
    - "HUDView(master, *, spec: Mapping|None = None)"
    - "HUDView.set_spec(spec: Mapping) -> None"
    - "HUDView.get_spec() -> dict"
    - "HUDView.set_history(mapping: Mapping[str,str]) -> None"
    - "HUDView.render_from_cache(ctx: Any) -> None"

depends_on:
  - "tkinter/ttk"
  - "glitchlab.gui.mosaic.router::resolve_selection"
  - "Pillow (opcjonalnie, miniatury)"
  - "NumPy (opcjonalnie, detekcja ndarray)"

used_by:
  - "glitchlab.gui.app  (AppShell)"
  - "glitchlab.gui.views.viewport (jako warstwa informacyjna, pośrednio)"

policy:
  deterministic: true
  ui_thread_only: true
  side_effects: false

constraints:
  - "brak zależności od EventBus"
  - "brak I/O i logiki obliczeniowej; tylko prezentacja"
  - "miniatury tworzone best-effort (gdy dostępny PIL/ndarray)"

hud:
  channels.read:
    - "stage/*/in|out|diff"
    - "stage/*/metrics_*"
    - "stage/*/fft_*"
    - "ast/json"
    - "format/*"
  overlay.pref: ["stage/*/mosaic","diag/*/*"]

license: "Proprietary"
---
"""
# glitchlab/gui/views/hud.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import tkinter as tk
from tkinter import ttk

try:
    import numpy as np  # type: ignore
except Exception:  # numpy opcjonalny w UI
    np = None  # type: ignore[assignment]

# --- Mosaic Router (preferowana ścieżka) ---
try:
    # Uwaga: poprawna ścieżka w Twoim projekcie to glitchlab.gui.mosaic.router
    from glitchlab.gui.mosaic.router import resolve_selection  # type: ignore
except Exception:
    # fallback – wybór „none”
    def resolve_selection(spec, keys, **_):
        return ({"slot1": None, "slot2": None, "slot3": None, "overlay": None}, {}, {"fallback": True})


# --- PIL (miniaturki jeśli dostępny) ---
try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore


__all__ = ["HUDView"]


# --------------------------
# Pomocnicze formatowanie
# --------------------------

def _is_ndarray(x: Any) -> bool:
    return (np is not None) and isinstance(x, np.ndarray)  # type: ignore[arg-type]


def _is_pil(x: Any) -> bool:
    return (Image is not None) and isinstance(x, Image.Image)  # type: ignore[attr-defined]


def _to_pil_rgb(x: Any) -> Optional["Image.Image"]:
    """Bezpieczna konwersja na PIL.Image RGB (dla miniatur)."""
    if Image is None:
        return None
    try:
        if _is_pil(x):
            im = x  # type: ignore[assignment]
            if im.mode != "RGB":
                return im.convert("RGB")
            return im
        if _is_ndarray(x):
            arr = x  # type: ignore[assignment]
            if arr.ndim == 2:
                arr = np.dstack([arr, arr, arr])  # type: ignore[operator]
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                if arr.shape[-1] == 4:
                    # spłaszcz na czarne tło (deterministycznie)
                    im = Image.fromarray(arr.astype("uint8"), "RGBA")  # type: ignore[attr-defined]
                    bg = Image.new("RGB", im.size, (0, 0, 0))
                    bg.paste(im, mask=im.split()[-1])
                    return bg
                return Image.fromarray(arr.astype("uint8"), "RGB")  # type: ignore[attr-defined]
    except Exception:
        return None
    return None


def _short_value_text(v: Any, max_len: int = 160) -> str:
    """Zwięzły opis wartości (bez obrazków)."""
    try:
        if v is None:
            return "None"
        if _is_ndarray(v):
            shape = getattr(v, "shape", None)
            dt = getattr(v, "dtype", None)
            return f"ndarray shape={tuple(shape)} dtype={str(dt)}"
        if _is_pil(v):
            return f"PIL.Image size={v.size} mode={getattr(v, 'mode', '?')}"
        if isinstance(v, (bytes, bytearray)):
            n = len(v)
            return f"bytes[{n}]"
        if isinstance(v, (list, tuple, dict)):
            try:
                s = json.dumps(v, ensure_ascii=False)  # type: ignore[arg-type]
            except Exception:
                s = str(v)
            return s if len(s) <= max_len else s[: max_len - 1] + "…"
        s = str(v)
        return s if len(s) <= max_len else s[: max_len - 1] + "…"
    except Exception:
        return "<unrenderable>"


# --------------------------
# Slot HUD (pojedynczy)
# --------------------------

class _HudSlot(ttk.LabelFrame):
    """Pojedynczy slot HUD z miniaturą (jeśli możliwa) + podpisem i przyciskami narzędziowymi."""
    def __init__(self, master, title: str):
        super().__init__(master, text=title)
        self._thumb_label = ttk.Label(self, anchor="center")
        self._thumb_label.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 0))

        self._text = ttk.Label(self, text="—", anchor="w", justify="left", wraplength=420)
        self._text.grid(row=1, column=0, sticky="ew", padx=6, pady=(4, 6))

        # wiersz narzędzi
        tools = ttk.Frame(self)
        tools.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))
        self._lbl_key = ttk.Label(tools, text="", foreground="#777")
        self._lbl_key.pack(side="left")

        self._btn_copy = ttk.Button(tools, text="Copy key", width=9, command=self._copy_key)
        self._btn_copy.pack(side="right", padx=(6, 0))
        self._btn_view = ttk.Button(tools, text="View…", width=7, command=self._open_viewer)
        self._btn_view.pack(side="right")

        # dane
        self._current_key: Optional[str] = None
        self._current_value: Any = None
        self._tk_img: Optional["ImageTk.PhotoImage"] = None  # trzymamy referencję

        # grid
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    # ----- API -----

    def clear(self) -> None:
        self._current_key = None
        self._current_value = None
        self._tk_img = None
        self._thumb_label.configure(image="", text="(no preview)")
        self._text.configure(text="—")
        self._lbl_key.configure(text="")

    def set_content(self, key: Optional[str], value: Any) -> None:
        self._current_key = key
        self._current_value = value
        self._lbl_key.configure(text=(key or ""))

        # tekst
        self._text.configure(text=_short_value_text(value))

        # miniatura
        img = _to_pil_rgb(value)
        if img is not None and ImageTk is not None:
            try:
                thumb = _make_thumb(img, max_side=160)
                self._tk_img = ImageTk.PhotoImage(thumb)
                self._thumb_label.configure(image=self._tk_img, text="")
            except Exception:
                self._tk_img = None
                self._thumb_label.configure(image="", text="(no preview)")
        else:
            self._tk_img = None
            self._thumb_label.configure(image="", text="(no preview)")

    # ----- actions -----

    def _copy_key(self) -> None:
        try:
            if not self._current_key:
                return
            self.clipboard_clear()
            self.clipboard_append(self._current_key)
        except Exception:
            pass

    def _open_viewer(self) -> None:
        if self._current_key is None:
            return
        v = self._current_value
        win = tk.Toplevel(self)
        win.title(self._current_key)
        win.geometry("620x480")

        # Spróbuj obrazek → inaczej tekst
        img = _to_pil_rgb(v)
        if img is not None and ImageTk is not None:
            # dopasuj do okna
            frm = ttk.Frame(win); frm.pack(fill="both", expand=True)
            cv = tk.Canvas(frm, bg="#101010", highlightthickness=0)
            cv.pack(fill="both", expand=True)

            def redraw(_evt=None):
                try:
                    w = max(1, cv.winfo_width()); h = max(1, cv.winfo_height())
                    thumb = _fit_into(img, w, h)
                    tkimg = ImageTk.PhotoImage(thumb)
                    cv.delete("all")
                    cv.create_image(w // 2, h // 2, image=tkimg, anchor="center")
                    cv._img = tkimg  # keep ref
                except Exception:
                    pass

            cv.bind("<Configure>", redraw)
            redraw()
        else:
            txt = tk.Text(win, wrap="word", bg="#141414", fg="#e6e6e6", insertbackground="#e6e6e6")
            txt.pack(fill="both", expand=True)
            try:
                if isinstance(v, (dict, list, tuple)):
                    txt.insert("1.0", json.dumps(v, ensure_ascii=False, indent=2))
                else:
                    txt.insert("1.0", str(v))
            except Exception:
                txt.insert("1.0", "<unrenderable>")

        ttk.Label(win, text=self._current_key, foreground="#888").pack(side="bottom", anchor="w", padx=8, pady=6)


# --------------------------
# Mini-grafika
# --------------------------

def _make_thumb(img: "Image.Image", max_side: int = 160) -> "Image.Image":
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    try:
        return img.resize((nw, nh), resample=getattr(Image, "BICUBIC", 3))
    except Exception:
        return img.resize((nw, nh))


def _fit_into(img: "Image.Image", W: int, H: int) -> "Image.Image":
    w, h = img.size
    if w <= 1 or h <= 1:
        return img
    scale = min(W / float(w), H / float(h))
    scale = max(1e-3, scale)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    try:
        return img.resize((nw, nh), resample=getattr(Image, "BICUBIC", 3))
    except Exception:
        return img.resize((nw, nh))


# --------------------------
# HUD główny (3 sloty)
# --------------------------

_DEFAULT_SPEC: Dict[str, Any] = {
    "slots": {
        "slot1": ["stage/0/in", "stage/*/in", "stage/*/metrics_in", "format/jpg_grid"],
        "slot2": ["stage/0/out", "stage/*/out", "stage/*/metrics_out", "stage/*/fft_mag"],
        "slot3": ["stage/0/diff", "stage/*/diff", "stage/*/diff_stats", "ast/json"],
    },
    "overlay": ["stage/*/mosaic", "diag/*/*"],
}


@dataclass
class _History:
    slot1: Optional[str] = None
    slot2: Optional[str] = None
    slot3: Optional[str] = None


class HUDView(ttk.Frame):
    """
    3-slotowy HUD:
      - wybór kluczy z ctx.cache przez Mosaic Router (spec konfigurowalny),
      - miniatury dla obrazów (PIL/ndarray), tekst dla reszty,
      - podgląd „View…” i „Copy key”.
    Publiczne API:
      - render_from_cache(ctx)
      - set_spec(spec_dict) / get_spec()
      - set_history(mapping) opcjonalnie (utrzymanie preferencji)
    """
    def __init__(self, master, *, spec: Optional[Mapping[str, Any]] = None):
        super().__init__(master)
        self._spec: Dict[str, Any] = dict(spec or _DEFAULT_SPEC)
        self._history = _History()

        # layout
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        self._slot1 = _HudSlot(self, "Slot 1"); self._slot1.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self._slot2 = _HudSlot(self, "Slot 2"); self._slot2.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        self._slot3 = _HudSlot(self, "Slot 3"); self._slot3.grid(row=0, column=2, sticky="nsew", padx=6, pady=6)

        # overlay info
        self._overlay = ttk.LabelFrame(self, text="Overlay")
        self._overlay.grid(row=1, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 6))
        self._overlay_lbl = ttk.Label(self._overlay, text="—", foreground="#777", anchor="w")
        self._overlay_lbl.pack(fill="x", padx=6, pady=6)

    # --- konfiguracja ---

    def set_spec(self, spec: Mapping[str, Any]) -> None:
        self._spec = dict(spec or {})
        # bez renderu – wywołujący zwykle zrobi render_from_cache()

    def get_spec(self) -> Dict[str, Any]:
        return dict(self._spec)

    def set_history(self, mapping: Mapping[str, str]) -> None:
        self._history = _History(
            slot1=mapping.get("slot1"),  # type: ignore[arg-type]
            slot2=mapping.get("slot2"),  # type: ignore[arg-type]
            slot3=mapping.get("slot3"),  # type: ignore[arg-type]
        )

    # --- render ---

    def render_from_cache(self, ctx: Any) -> None:
        """
        Odczytuje ctx.cache, przepuszcza przez resolver i renderuje sloty + overlay.
        """
        try:
            cache: Dict[str, Any] = getattr(ctx, "cache", {}) or {}
            keys = list(cache.keys())

            sel, _rank, _exp = resolve_selection(
                self._spec,
                keys,
                history={"slot1": self._history.slot1, "slot2": self._history.slot2, "slot3": self._history.slot3},
            )

            k1, k2, k3 = sel.get("slot1"), sel.get("slot2"), sel.get("slot3")
            self._slot1.set_content(k1, cache.get(k1) if k1 else None) if k1 else self._slot1.clear()
            self._slot2.set_content(k2, cache.get(k2) if k2 else None) if k2 else self._slot2.clear()
            self._slot3.set_content(k3, cache.get(k3) if k3 else None) if k3 else self._slot3.clear()

            kov = sel.get("overlay")
            self._overlay_lbl.configure(text=(kov or "—"))

            # aktualizuj historię (jeśli cokolwiek wybrano)
            self._history.slot1 = k1 or self._history.slot1
            self._history.slot2 = k2 or self._history.slot2
            self._history.slot3 = k3 or self._history.slot3
        except Exception as ex:
            # awaryjnie pokaż błąd w slotach
            for sl in (self._slot1, self._slot2, self._slot3):
                sl.set_content(None, f"HUD error: {ex}")
            try:
                self._overlay_lbl.configure(text="—")
            except Exception:
                pass
