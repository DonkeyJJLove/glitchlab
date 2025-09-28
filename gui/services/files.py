# glitchlab/services/files.py
"""
---
version: 3
kind: module
id: "gui-services-files"
created\_at: "2025-09-13"
name: "glitchlab.gui.services.files"
author: "GlitchLab v3"
role: "Image I/O & Recent Files"
description: >
Lekki serwis do wczytywania/zapisu obrazów (Pillow) z korektą orientacji EXIF
i deterministycznym spłaszczaniem alfa (RGBA→RGB), oraz utrzymania listy
ostatnich plików (LRU-like) w \~/.glitchlab/recent\_files.json.
inputs:
load\_image.path: {type: "str|Path", desc: "ścieżka do pliku obrazu"}
load\_image.force\_rgb: {type: "bool", default: true}
load\_image.exif\_transpose: {type: "bool", default: true}
save\_image.img: {type: "PIL.Image|np.ndarray", shape: "(H,W\[,3|4])", dtype: "uint8"}
save\_image.path: {type: "str|Path"}
save\_image.format: {type: "str", optional: true, desc: "nadpisanie formatu (JPEG/PNG/WEBP…)"}
save\_image.quality: {type: "int", optional: true}
save\_image.optimize: {type: "bool", optional: true}
RecentStore.path: {type: "Path", desc: "plik JSON ze stanem recent"}
outputs:
load\_image: {type: "PIL.Image.Image", mode: "RGB", note: "EXIF fixed, alpha zflattenowana"}
save\_image: {type: "None", side\_effect: "zapis na dysk"}
RecentStore.list: {type: "list\[str]", desc: "istniejące ścieżki w kolejności ostatniego użycia"}
interfaces:
exports: \["load\_image","save\_image","RecentStore","default\_recent\_store"]
depends\_on: \["Pillow","numpy","json","pathlib"]
used\_by: \["glitchlab.gui.app","glitchlab.gui.services.pipeline\_runner","glitchlab.gui.services.presets"]
policy:
deterministic: true
side\_effects: \["filesystem I/O"]
constraints:

* "brak SciPy/OpenCV"
* "zwracane obrazy w trybie RGB 8-bit"
  notes:
* "RGBA spłaszczane na czarne tło (stałe i deterministyczne)"
* "lista recent filtruje nieistniejące pliki i utrzymuje limit max\_items"
  license: "Proprietary"
---
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

# Public API
__all__ = [
    "load_image",
    "save_image",
    "RecentStore",
    "default_recent_store",
]


# ----------------------------
# Image I/O helpers (Pillow)
# ----------------------------

def _to_pil(img: Union[Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[-1] == 4:
                # Flatten alpha over opaque black for deterministic saving
                pil = Image.fromarray(arr, mode="RGBA")
                bg = Image.new("RGB", pil.size, (0, 0, 0))
                bg.paste(pil, mask=pil.split()[-1])
                return bg
            return Image.fromarray(arr, mode="RGB")
    raise TypeError("Unsupported image type. Expected PIL.Image or np.ndarray (H,W[,3|4]).")


def load_image(path: Union[str, Path], *, force_rgb: bool = True, exif_transpose: bool = True) -> Image.Image:
    """
    Loads an image using Pillow. By default:
      - applies EXIF orientation fix,
      - converts to 8-bit RGB.
    """
    p = Path(path)
    with Image.open(p) as im:
        if exif_transpose:
            im = ImageOps.exif_transpose(im)
        if force_rgb:
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")
            elif im.mode == "RGBA":
                # Flatten alpha over opaque black (consistent with _to_pil)
                bg = Image.new("RGB", im.size, (0, 0, 0))
                bg.paste(im, mask=im.split()[-1])
                im = bg
        else:
            im = im.copy()
    return im


def save_image(
    img: Union[Image.Image, np.ndarray],
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
    quality: Optional[int] = None,
    optimize: Optional[bool] = None,
) -> None:
    """
    Saves an image to disk. Format inferred from path suffix unless given explicitly.
    Quality/optimize are passed to Pillow when applicable (e.g., JPEG/WEBP).
    """
    pil = _to_pil(img)
    p = Path(path)
    fmt = (format or p.suffix.lstrip(".")).upper()
    params = {}
    if quality is not None:
        params["quality"] = int(quality)
    if optimize is not None:
        params["optimize"] = bool(optimize)

    # Sensible defaults for common formats
    if fmt in ("JPG", "JPEG"):
        params.setdefault("quality", 95)
        params.setdefault("subsampling", 0)
        fmt = "JPEG"
    elif fmt == "PNG":
        params.setdefault("compress_level", 6)
        fmt = "PNG"
    elif fmt == "WEBP":
        params.setdefault("quality", 95)
        fmt = "WEBP"

    pil.save(p, format=fmt, **params)


# ----------------------------
# Recent files store
# ----------------------------

def _default_config_dir() -> Path:
    # Minimal, cross-platform friendly config dir
    home = Path.home()
    cfg = home / ".glitchlab"
    cfg.mkdir(parents=True, exist_ok=True)
    return cfg


@dataclass
class RecentStore:
    """
    Lightweight LRU-like store of recent file paths.
    """
    path: Path
    max_items: int = 12
    _items: List[Tuple[str, float]] = field(default_factory=list)  # (path, ts)

    @classmethod
    def in_default_location(cls, filename: str = "recent_files.json", max_items: int = 12) -> "RecentStore":
        return cls(_default_config_dir() / filename, max_items=max_items).load()

    # ------------- core ops -------------

    def load(self) -> "RecentStore":
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                items = data.get("items", [])
                self._items = [(str(p), float(ts)) for p, ts in items if isinstance(p, str)]
        except Exception:
            # Corrupted file -> start clean
            self._items = []
        return self

    def save(self) -> None:
        try:
            data = {"items": self._items[: self.max_items]}
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # Best-effort; ignore disk errors silently for UI smoothness
            pass

    def add(self, file_path: Union[str, Path]) -> None:
        p = str(Path(file_path).resolve())
        now = time.time()
        # Remove duplicates
        self._items = [(fp, ts) for (fp, ts) in self._items if fp != p]
        # Prepend new
        self._items.insert(0, (p, now))
        # Trim
        if len(self._items) > self.max_items:
            self._items = self._items[: self.max_items]
        self.save()

    def list(self) -> List[str]:
        # Filter only existing files; keep order
        out: List[str] = []
        for fp, _ in self._items:
            try:
                if Path(fp).exists():
                    out.append(fp)
            except Exception:
                continue
        return out

    def clear(self) -> None:
        self._items = []
        self.save()

    def remove_missing(self) -> None:
        self._items = [(fp, ts) for (fp, ts) in self._items if Path(fp).exists()]
        self.save()

    # ------------- convenience -------------

    def touch(self, file_path: Union[str, Path]) -> None:
        """Alias for add(file_path)."""
        self.add(file_path)

    def extend(self, file_paths: Iterable[Union[str, Path]]) -> None:
        for p in file_paths:
            self.add(p)


# Singleton-style default store
_default_store: Optional[RecentStore] = None


def default_recent_store() -> RecentStore:
    global _default_store
    if _default_store is None:
        _default_store = RecentStore.in_default_location()
    return _default_store
