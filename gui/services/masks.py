# glitchlab/gui/services/masks.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

# miękkie zależności na Pillow
try:
    from PIL import Image, ImageOps  # type: ignore
except Exception:  # pragma: no cover
    Image = None
    ImageOps = None


def normalize_mask(arr: np.ndarray) -> np.ndarray:
    """
    Zwraca 2D maskę float32 w [0,1].
    - RGB/RGBA → luminancja
    - uint8 → /255
    - NaN/Inf → 0
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("normalize_mask: expected np.ndarray")
    a = arr
    if a.ndim == 3 and a.shape[-1] in (3, 4):  # RGB/A → L
        # luma (Rec. 601)
        rgb = a[..., :3].astype(np.float32)
        a = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    elif a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    a = a.astype(np.float32, copy=False)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    # jeżeli nadal poza [0,1] – spróbuj przeskalować
    if a.max(initial=0.0) > 1.5 or a.min(initial=0.0) < -0.5:
        a = (a - a.min()) / max(1e-12, (a.max() - a.min()))
    # sanitizacja
    a = np.nan_to_num(a, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    a = np.clip(a, 0.0, 1.0, out=a)
    return a


def load_mask_from_file(path: str | Path, *, exif_transpose: bool = True) -> np.ndarray:
    """
    Wczytuje obraz maski, konwertuje do L, normalizuje do [0,1], float32.
    """
    if Image is None:
        raise RuntimeError("Pillow not available")
    p = Path(path)
    with Image.open(p) as im:
        if exif_transpose and ImageOps is not None:
            im = ImageOps.exif_transpose(im)
        im = im.convert("L")
        arr = np.asarray(im, dtype=np.uint8)
    return normalize_mask(arr)


@dataclass
class MaskService:
    """
    Utrzymuje słownik masek (w pamięci) i publikuje listę kluczy do UI.
    Integracja przez EventBus (opcjonalna):
      - ui.masks.add_request    → dialog → masks.list
      - ui.masks.clear_request  → czyszczenie → masks.list
      - ui.masks.remove_request → usunięcie jednej → masks.list
    Dodatkowo udostępnia get_masks() do wpięcia w kontekst pipeline.
    """
    root_like: Any
    bus: Any  # EventBus-like
    _masks: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            self.bus.subscribe("ui.masks.add_request", self._on_add_request, on_ui=True)
            self.bus.subscribe("ui.masks.clear_request", self._on_clear_request, on_ui=True)
            self.bus.subscribe("ui.masks.remove_request", self._on_remove_request, on_ui=True)
        except Exception:
            pass
        # publikacja stanu początkowego
        self._publish_list()

    # ---------- API dla AppShell/Runner ----------

    def get_masks(self) -> Dict[str, np.ndarray]:
        return dict(self._masks)

    def keys(self) -> List[str]:
        return list(self._masks.keys())

    def set(self, key: str, arr: np.ndarray) -> None:
        self._masks[str(key)] = normalize_mask(arr)
        self._publish_list()

    def remove(self, key: str) -> bool:
        k = str(key)
        if k in self._masks:
            self._masks.pop(k, None)
            self._publish_list()
            return True
        return False

    def clear(self) -> None:
        self._masks.clear()
        self._publish_list()

    # ---------- Handlery BUS ----------

    def _on_add_request(self, _t: str, _d: Dict[str, Any]) -> None:
        from tkinter import filedialog, simpledialog, messagebox
        path = filedialog.askopenfilename(
            title="Add mask",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            arr = load_mask_from_file(path)
        except Exception as ex:
            messagebox.showerror("Mask", str(ex))
            return

        default_key = Path(path).stem
        key = simpledialog.askstring("Mask key", "Enter mask key:", initialvalue=default_key, parent=self.root_like)
        if not key:
            return
        self._masks[str(key)] = arr
        self._publish_list()

    def _on_clear_request(self, _t: str, _d: Dict[str, Any]) -> None:
        self.clear()

    def _on_remove_request(self, _t: str, data: Dict[str, Any]) -> None:
        key = (data or {}).get("key")
        if not isinstance(key, str) or not key:
            return
        self.remove(key)

    # ---------- Publikacja ----------

    def _publish_list(self) -> None:
        try:
            self.bus.publish("masks.list", {"names": list(self._masks.keys())})
        except Exception:
            pass
