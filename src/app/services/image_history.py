# glitchlab/app/services/image_history.py
# -*- coding: utf-8 -*-
"""
GlitchLab GUI — Image History Service (stack-based)

Cel:
- Utrzymywać historię obrazu (wejście → kolejne wyniki filtrów),
- Zapewnić undo/redo,
- Przechowywać skojarzony cache (telemetria ctx.cache) dla HUD,
- Publikować proste zdarzenia (opcjonalnie) na EventBus.

Konwencje:
- Obraz przechowujemy jako np.ndarray uint8 RGB (H,W,3).
- Wejścia API mogą być PIL.Image / ndarray / bytes -> konwertujemy do u8 RGB.
- Kopiujemy tablice przy wstawianiu, aby unikać aliasowania pamięci.

Zdarzenia (opcjonalne; jeśli services ma .publish):
- "history.changed" {size:int, index:int}
- "history.push"    {index:int, label:str}
- "history.undo"    {index:int}
- "history.redo"    {index:int}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

# PIL jest opcjonalny na wejściu/wyjściu
try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


# ------------------------------ helpers -------------------------------------


def _ensure_np() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for ImageHistory.")


def _to_u8_rgb(img: Any) -> "np.ndarray":
    """
    Akceptuje: PIL.Image | np.ndarray | bytes/bytearray
    Zwraca: np.ndarray uint8 RGB (H,W,3)
    """
    _ensure_np()

    # PIL.Image
    if Image is not None and isinstance(img, Image.Image):  # type: ignore[attr-defined]
        im = img
        if im.mode != "RGB":
            im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr

    # ndarray
    if isinstance(img, np.ndarray):
        a = img
        # float -> u8 (zakładamy [0..1] lub [0..255])
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8, copy=False)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        elif a.ndim == 3:
            if a.shape[-1] == 4:
                # premultiply na czarne tło (stabilnie)
                rgb = a[..., :3].astype(np.float32)
                alpha = (a[..., 3:4].astype(np.float32) / 255.0)
                a = np.clip(rgb * alpha, 0, 255).astype(np.uint8)
            elif a.shape[-1] == 1:
                a = np.repeat(a, 3, axis=-1)
        # ostateczna walidacja
        if a.ndim != 3 or a.shape[-1] != 3:
            raise ValueError(f"Unsupported array shape for RGB: {a.shape}")
        # kopia aby nie aliasować źródła
        return a.astype(np.uint8, copy=True)

    # bytes (np. już skompresowany obraz) — best-effort przez PIL
    if isinstance(img, (bytes, bytearray)) and Image is not None:  # type: ignore[attr-defined]
        from io import BytesIO
        try:
            im = Image.open(BytesIO(img)).convert("RGB")
            return np.asarray(im, dtype=np.uint8)
        except Exception as _:
            raise ValueError("Unsupported bytes payload for image decode.")

    raise TypeError(f"Unsupported image type: {type(img)}")


def _to_pil(img_u8: "np.ndarray") -> "Image.Image":
    _ensure_np()
    if Image is None:
        raise RuntimeError("Pillow is required to export PIL.Image.")
    if img_u8.ndim != 3 or img_u8.shape[-1] != 3 or img_u8.dtype != np.uint8:
        raise ValueError(f"Expected u8 RGB, got shape={img_u8.shape} dtype={img_u8.dtype}")
    return Image.fromarray(img_u8, mode="RGB")  # type: ignore[attr-defined]


# ------------------------------ model ---------------------------------------


@dataclass
class HistoryEntry:
    """Pojedynczy wpis historii."""
    image_u8: "np.ndarray"           # (H,W,3) uint8
    cache: Dict[str, Any]            # ctx.cache (lekki dict; bez dużych blobów)
    label: str = ""                  # np. nazwa filtra / krok
    # Można dodać timestamp/metrics/etc. gdy będzie potrzeba.


# ------------------------------ service -------------------------------------


class ImageHistory:
    """
    Stos (undo/redo) historii obrazu.
    Indeks wskazuje NA BIEŻĄCY stan (0..len-1).

    API (najważniejsze):
      - reset(source_image, cache=None, label="source")  -> wyczyść i ustaw stan 0
      - push(result_image, cache=None, label="step")     -> dorzuć nowy stan i ustaw jako bieżący
      - can_undo()/undo() / can_redo()/redo()
      - get_current_image_u8() / get_current_image_pil()
      - size(), index()
    """

    def __init__(self, *, bus: Optional[Any] = None, max_len: int = 50) -> None:
        _ensure_np()
        self._bus = bus
        self._max_len = max(2, int(max_len))
        self._entries: List[HistoryEntry] = []
        self._index: int = -1  # brak

    # --------------------------- core ops ------------------------------------

    def reset(self, source_image: Any, *, cache: Optional[Dict[str, Any]] = None, label: str = "source") -> None:
        img_u8 = _to_u8_rgb(source_image)
        entry = HistoryEntry(image_u8=img_u8.copy(), cache=dict(cache or {}), label=label)
        self._entries = [entry]
        self._index = 0
        self._trim_if_needed()
        self._publish("history.changed", {"size": self.size(), "index": self.index()})

    def push(self, result_image: Any, *, cache: Optional[Dict[str, Any]] = None, label: str = "step") -> None:
        """
        Dorzuca nowy stan na „wierzch” historii.
        Kasuje ewentualne „redo” (wszystko za bieżącym indeksem).
        """
        if self._index >= 0 and self._index < len(self._entries) - 1:
            # usuwamy gałąź redo
            self._entries = self._entries[: self._index + 1]

        img_u8 = _to_u8_rgb(result_image)
        entry = HistoryEntry(image_u8=img_u8.copy(), cache=dict(cache or {}), label=label)
        self._entries.append(entry)
        self._index = len(self._entries) - 1
        self._trim_if_needed()
        self._publish("history.push", {"index": self._index, "label": label})
        self._publish("history.changed", {"size": self.size(), "index": self.index()})

    # --------------------------- navigation ----------------------------------

    def can_undo(self) -> bool:
        return self._index > 0

    def can_redo(self) -> bool:
        return 0 <= self._index < len(self._entries) - 1

    def undo(self) -> Optional["np.ndarray"]:
        if not self.can_undo():
            return None
        self._index -= 1
        self._publish("history.undo", {"index": self._index})
        self._publish("history.changed", {"size": self.size(), "index": self.index()})
        return self._entries[self._index].image_u8

    def redo(self) -> Optional["np.ndarray"]:
        if not self.can_redo():
            return None
        self._index += 1
        self._publish("history.redo", {"index": self._index})
        self._publish("history.changed", {"size": self.size(), "index": self.index()})
        return self._entries[self._index].image_u8

    # --------------------------- getters -------------------------------------

    def size(self) -> int:
        return len(self._entries)

    def index(self) -> int:
        return self._index

    def get_current(self) -> Optional[HistoryEntry]:
        if 0 <= self._index < len(self._entries):
            return self._entries[self._index]
        return None

    def get_current_image_u8(self) -> Optional["np.ndarray"]:
        ent = self.get_current()
        return ent.image_u8.copy() if ent else None

    def get_current_image_pil(self) -> Optional["Image.Image"]:
        ent = self.get_current()
        if ent is None:
            return None
        return _to_pil(ent.image_u8)

    def get_current_cache(self) -> Dict[str, Any]:
        ent = self.get_current()
        return dict(ent.cache) if ent else {}

    # --------------------------- utils ---------------------------------------

    def _trim_if_needed(self) -> None:
        # Ogranicz długość historii; zachowujemy ostatnie wpisy
        overflow = len(self._entries) - self._max_len
        if overflow > 0:
            # przesuwamy indeks o tyle, ile „odpadło” z przodu
            self._entries = self._entries[overflow:]
            self._index = max(0, self._index - overflow)

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self._bus is not None and hasattr(self._bus, "publish"):
            try:
                self._bus.publish(topic, dict(payload))
            except Exception:
                # nie przerywamy — historia działa również bez busa
                pass

    # --------------------------- convenience ---------------------------------

    def snapshot_from_ctx(self, output_u8: Any, ctx_obj: Any, *, label: str = "step") -> None:
        """
        Wygodny zapis: bierze wynik filtra + ctx (z .cache) i robi push().
        """
        cache = {}
        try:
            cache = dict(getattr(ctx_obj, "cache", {}) or {})
        except Exception:
            cache = {}
        self.push(output_u8, cache=cache, label=label)
