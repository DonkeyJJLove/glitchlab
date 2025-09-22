# -*- coding: utf-8 -*-
from __future__ import annotations

"""
LayerManager — zarządzanie warstwami + kompozycja.

Kompatybilne z:
- app.App (metody: add_layer, remove_layer, update_layer, get_composite_for_viewport)
- LayersPanel (pola: id, name, visible, opacity, blend, offset)
- BLEND_MODES: normal, multiply, screen, overlay, add, subtract, darken, lighten

Konwencja kolejności:
- App/LayersPanel przechowują state.layers w kolejności TOP → BOTTOM.
- Kompozycję liczymy od dołu do góry, więc iterujemy od końca listy.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None  # type: ignore


# ───────────────────────────── utils ─────────────────────────────

def _to_rgb_u8(image: Any) -> "np.ndarray":
    """
    Zamienia wejście (PIL.Image | np.ndarray) na ndarray uint8 w formacie RGB (H,W,3).
    - Skaluje z [0..1] jeśli potrzeba, usuwa alfa przez pre-multiplikację.
    """
    if np is None:
        raise RuntimeError("NumPy is required for LayerManager")

    # PIL -> ndarray
    if PILImage is not None and isinstance(image, PILImage.Image):
        im = image.convert("RGBA")  # zawsze złap alfę
        a = np.array(im, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        a = image
        if a.dtype != np.uint8:
            if np.issubdtype(a.dtype, np.floating):
                a = np.clip(a, 0.0, 1.0) * 255.0
            a = np.clip(a, 0, 255).astype(np.uint8)
        # ujednolić kształty
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)  # gray -> RGB
        elif a.ndim == 3 and a.shape[-1] in (1, 3, 4):
            if a.shape[-1] == 1:
                a = np.repeat(a, 3, axis=-1)
            # 3/4 kanały zostawiamy jak są
        else:
            raise ValueError("Unsupported ndarray shape for image")
    else:
        raise ValueError("Unsupported image type")

    # Pre-multiply alpha, usuń kanał A
    if a.ndim == 3 and a.shape[-1] == 4:
        rgb = a[..., :3].astype(np.float32)
        alpha = (a[..., 3:4].astype(np.float32) / 255.0)
        a = (rgb * alpha).clip(0, 255).astype(np.uint8)
    elif a.ndim == 3 and a.shape[-1] == 3:
        # już RGB u8
        pass
    else:
        # coś nietypowego — spróbuj wymusić RGB
        a = np.atleast_3d(a).astype(np.uint8)
        if a.shape[-1] == 1:
            a = np.repeat(a, 3, axis=-1)
        elif a.shape[-1] != 3:
            raise ValueError("Cannot coerce image to RGB")

    return a


def _ensure_same_size(arr: "np.ndarray", w: int, h: int) -> "np.ndarray":
    """
    Dba, by tablica miała rozmiar (h, w, 3).
    Prosta wersja: jeśli rozmiar nie pasuje, dociśnij/wytnij do minimalnego wspólnego.
    (Brak resamplingu — zakładamy zgodne rozmiary źródeł; do backgroundu i filtrów wystarcza.)
    """
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    # przytnij do wspólnego min
    hh = min(h, arr.shape[0])
    ww = min(w, arr.shape[1])
    return arr[:hh, :ww, :]


# ───────────────────────────── blending ─────────────────────────────

def _blend_normal(dst: "np.ndarray", src: "np.ndarray", opacity: float) -> "np.ndarray":
    return (dst * (1.0 - opacity) + src * opacity).astype(np.float32)

def _blend_multiply(dst: "np.ndarray", src: "np.ndarray", opacity: float) -> "np.ndarray":
    out = (dst * src) / 255.0
    return dst + (out - dst) * opacity

def _blend_screen(dst: "np.ndarray", src: "np.ndarray", opacity: float) -> "np.ndarray":
    out = 255.0 - ((255.0 - dst) * (255.0 - src) / 255.0)
    return dst + (out - dst) * opacity

def _blend_overlay(dst: "np.ndarray", src: "np.ndarray", opacity: float) -> "np.ndarray":
    mask = dst < 128
    out = np.empty_like(dst, dtype=np.float32)
    out[mask] = (2.0 * dst[mask] * src[mask] / 255.0)
    out[~mask] = (255.0 - 2.0 * (255.0 - dst[~mask]) * (255.0 - src[~mask]) / 255.0)
    return dst + (out - dst) * opacity

def _blend_add(dst: "np.ndarray", src: "np.ndarray", opacity: float) -> "np.ndarray":
    out = np.clip(dst + src, 0.0, 255.0)
    return dst + (out - dst) * opacity

def _blend_subtract(dst: "np.ndarray", src: "np.ndarray", opacity: float) -> "np.ndarray":
    out = np.clip(dst - src, 0.0, 255.0)
    return dst + (out - dst) * opacity

def _blend_darken(dst: "np.ndarray", src: "np.ndarray", opacity: float) -> "np.ndarray":
    out = np.minimum(dst, src)
    return dst + (out - dst) * opacity

def _blend_lighten(dst: "np.ndarray", src: "np.ndarray", opacity: float) -> "np.ndarray":
    out = np.maximum(dst, src)
    return dst + (out - dst) * opacity


_BLENDERS: Dict[str, Callable[["np.ndarray", "np.ndarray", float], "np.ndarray"]] = {
    "normal": _blend_normal,
    "multiply": _blend_multiply,
    "screen": _blend_screen,
    "overlay": _blend_overlay,
    "add": _blend_add,
    "subtract": _blend_subtract,
    "darken": _blend_darken,
    "lighten": _blend_lighten,
}


# ───────────────────────────── model ─────────────────────────────

@dataclass
class _Layer:
    id: str
    name: str
    visible: bool
    opacity: float      # 0..1
    blend: str          # normal/multiply/...
    image: "np.ndarray" # RGB uint8 (H,W,3)
    offset: Tuple[int, int] = (0, 0)


# ───────────────────────────── manager ─────────────────────────────

class LayerManager:
    """
    Główny menedżer warstw. Operuje na:
      - state.layers (lista _Layer) — UWAGA: kolejność TOP → BOTTOM,
      - state.active_layer_id (str | None).

    Po każdej zmianie publikuje: "ui.layers.changed", {}.
    """
    def __init__(self, state: Any, publish: Callable[[str, Dict[str, Any]], None]) -> None:
        if np is None:
            raise RuntimeError("NumPy is required for LayerManager")
        self.state = state
        self.publish = publish

        # Upewnij się, że są wymagane atrybuty w state
        if not hasattr(self.state, "layers"):
            self.state.layers = []  # type: ignore[attr-defined]
        if not hasattr(self.state, "active_layer_id"):
            self.state.active_layer_id = None  # type: ignore[attr-defined]

    # ── helpers ──────────────────────────────────────────────────────────────

    def _emit_changed(self) -> None:
        try:
            self.publish("ui.layers.changed", {})
        except Exception:
            pass

    def _find_index(self, lid: str) -> int:
        for i, l in enumerate(self.state.layers):
            if getattr(l, "id", None) == lid:
                return i
        return -1

    def _coerce_blend(self, name: str) -> str:
        return name if name in _BLENDERS else "normal"

    def _coerce_opacity(self, val: float) -> float:
        try:
            v = float(val)
        except Exception:
            v = 1.0
        return max(0.0, min(1.0, v))

    # ── public API ───────────────────────────────────────────────────────────

    def add_layer(
        self,
        image: Any,
        *,
        name: str = "Layer",
        visible: bool = True,
        opacity: float = 1.0,
        blend: str = "normal",
    ) -> str:
        """Dodaje warstwę NA GÓRĘ stosu (czyli na indeks 0 w TOP→BOTTOM)."""
        arr = _to_rgb_u8(image)
        lid = str(uuid.uuid4())
        layer = _Layer(
            id=lid,
            name=str(name or "Layer"),
            visible=bool(visible),
            opacity=self._coerce_opacity(opacity),
            blend=self._coerce_blend(str(blend)),
            image=arr,
            offset=(0, 0),
        )
        # Skoro state.layers to TOP→BOTTOM, nową warstwę kładziemy na początek listy.
        self.state.layers.insert(0, layer)
        self.state.active_layer_id = lid
        self._emit_changed()
        return lid

    def remove_layer(self, lid: str) -> None:
        idx = self._find_index(str(lid))
        if idx < 0:
            return
        del self.state.layers[idx]
        # Ustaw nową aktywną (top), jeśli usunięto aktywną
        if getattr(self.state, "active_layer_id", None) == lid:
            self.state.active_layer_id = (self.state.layers[0].id if self.state.layers else None)
        self._emit_changed()

    def update_layer(self, lid: str, **patch: Any) -> None:
        idx = self._find_index(str(lid))
        if idx < 0:
            return
        layer: _Layer = self.state.layers[idx]

        if "name" in patch:
            layer.name = str(patch["name"])

        if "visible" in patch:
            layer.visible = bool(patch["visible"])

        if "opacity" in patch:
            layer.opacity = self._coerce_opacity(patch["opacity"])

        if "blend" in patch:
            layer.blend = self._coerce_blend(str(patch["blend"]))

        if "image" in patch and patch["image"] is not None:
            layer.image = _to_rgb_u8(patch["image"])

        if "offset" in patch and patch["offset"] is not None:
            off = patch["offset"]
            try:
                dx, dy = int(off[0]), int(off[1])
            except Exception:
                dx, dy = 0, 0
            layer.offset = (dx, dy)

        self._emit_changed()

    # (opcjonalnie, jeśli chcesz API na reorder; App już to robi samodzielnie)
    def reorder(self, top_to_bottom_ids: List[str]) -> None:
        id2layer = {l.id: l for l in self.state.layers}
        new_list = [id2layer[i] for i in top_to_bottom_ids if i in id2layer]
        if new_list:
            self.state.layers = new_list
            # aktywną ustaw, jeśli wypadła z listy
            if self.state.active_layer_id not in top_to_bottom_ids:
                self.state.active_layer_id = self.state.layers[0].id if self.state.layers else None
            self._emit_changed()

    # ── compositing ──────────────────────────────────────────────────────────

    def get_composite_for_viewport(self) -> Optional["np.ndarray"]:
        """
        Zwraca kompozyt RGB uint8 (H,W,3) z wszystkich widocznych warstw,
        skomponowanych z uwzględnieniem 'blend' oraz 'opacity' i przesunięć 'offset'.
        - Oczekuje, że wszystkie obrazy mają ten sam rozmiar (typowo tak jest).
        - Jeśli rozmiary różne, kompozycja wykona się na wspólnym minimum (wycięcie).
        """
        if not self.state.layers:
            return None

        # Ustal rozmiar bazowy: weź pierwszą *od dołu* widoczną warstwę
        # (state.layers = TOP→BOTTOM, więc odwracamy).
        bottom_layers = list(reversed(self.state.layers))
        base_arr = None
        base_w = base_h = None

        for L in bottom_layers:
            if not getattr(L, "visible", True):
                continue
            img = getattr(L, "image", None)
            if img is None:
                continue
            a = _to_rgb_u8(img)
            base_h, base_w = int(a.shape[0]), int(a.shape[1])
            base_arr = a
            break

        if base_arr is None:
            return None

        # Start jako float32 (łatwiejsze blendy)
        comp = base_arr.astype(np.float32)

        # Przechodzimy w górę (czyli po pozostałych warstwach od dołu do góry)
        started = False
        for L in bottom_layers:
            if not getattr(L, "visible", True):
                continue
            src = getattr(L, "image", None)
            if src is None:
                continue
            src_arr = _to_rgb_u8(src).astype(np.float32)

            # docięcie do wspólnego obszaru
            h = min(base_h, src_arr.shape[0])
            w = min(base_w, src_arr.shape[1])
            if h <= 0 or w <= 0:
                continue

            # offset (prosta implementacja: przesuwamy źródło w granicach kompozytu)
            dx, dy = getattr(L, "offset", (0, 0))
            # wylicz wycinki doc/source
            x0_dst = max(0, int(dx))
            y0_dst = max(0, int(dy))
            x1_dst = min(w, w + int(dx)) if dx < 0 else min(w, int(w + 0))
            y1_dst = min(h, h + int(dy)) if dy < 0 else min(h, int(h + 0))

            # Poprawne okna: liczymy tak, by obsłużyć i dodatnie i ujemne przesunięcia
            # Prościej: budujemy puste płótno h×w i wklejamy src w odpowiednie miejsce.
            # (Uproszczenie: na tę chwilę ignorujemy fragment za granicą, co i tak jest ok.)
            pad = np.zeros((h, w, 3), dtype=np.float32)
            x_src0 = max(0, -int(dx))
            y_src0 = max(0, -int(dy))
            x_src1 = min(w - x0_dst, src_arr.shape[1] - x_src0) + x_src0
            y_src1 = min(h - y0_dst, src_arr.shape[0] - y_src0) + y_src0
            if x_src1 <= x_src0 or y_src1 <= y_src0:
                # poza kadrem
                continue
            x_dst0 = max(0, int(dx))
            y_dst0 = max(0, int(dy))
            x_dst1 = x_dst0 + (x_src1 - x_src0)
            y_dst1 = y_dst0 + (y_src1 - y_src0)

            pad[y_dst0:y_dst1, x_dst0:x_dst1, :] = src_arr[y_src0:y_src1, x_src0:x_src1, :]

            # pierwszy widoczny od dołu już mamy w comp (base_arr) — ominąć dubel
            if not started:
                # comp już jest base_arr — to ta sama warstwa: „połknij” i idź dalej
                started = True
                continue

            opacity = float(getattr(L, "opacity", 1.0))
            opacity = max(0.0, min(1.0, opacity))
            blend = str(getattr(L, "blend", "normal"))
            fn = _BLENDERS.get(blend, _blend_normal)

            # blend: pad (src) na comp (dst)
            comp = fn(comp, pad, opacity)
            comp = np.clip(comp, 0.0, 255.0)

        return comp.astype(np.uint8)
