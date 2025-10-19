# glitchlab/app/widgets/tools/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Protocol, Dict


# ──────────────────────────────────────────────────────────────────────────────
# Kontrakt transformacji i dostępu do danych obrazu/maski
# ──────────────────────────────────────────────────────────────────────────────

class _Coords(Protocol):
    """Kontrakt transformacji współrzędnych ekranu -> obrazu."""

    def __call__(self, sx: int, sy: int) -> Tuple[int, int]: ...


class _Invalidate(Protocol):
    """Żądanie odświeżenia overlay (bez pełnego przerysowania obrazu)."""

    def __call__(self, bbox: Optional[Tuple[int, int, int, int]] = None) -> None: ...


class _Publish(Protocol):
    """Publikacja zdarzenia na EventBus."""

    def __call__(self, topic: str, payload: Dict[str, Any]) -> None: ...


class _GetImage(Protocol):
    """Zwraca bieżący obraz kompozytowany dla viewportu jako np.uint8 RGB (H,W,3)."""

    def __call__(self) -> Any: ...


class _GetMask(Protocol):
    """Zwraca aktywną maskę jako ndarray (H,W) lub None."""

    def __call__(self) -> Optional[Any]: ...


class _SetMask(Protocol):
    """Ustawia/aktualizuje aktywną maskę (H,W)."""

    def __call__(self, mask_nd: Any) -> None: ...


class _GetView(Protocol):
    """Zwraca (zoom, (pan_x, pan_y))."""

    def __call__(self) -> Tuple[float, Tuple[int, int]]: ...


# ──────────────────────────────────────────────────────────────────────────────
# Kontekst wywołań narzędzia
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolEventContext:
    """
    Udostępnia narzędziom (ToolBase) spójny interfejs do:
    - publikacji zdarzeń (EventBus),
    - transformacji współrzędnych,
    - odświeżenia warstwy overlay,
    - dostępu do obrazu i aktywnej maski,
    - pobrania parametrów widoku (zoom/pan).
    """
    publish: _Publish
    to_image_xy: _Coords
    invalidate: _Invalidate
    get_mask: _GetMask
    set_mask: _SetMask
    get_image: _GetImage
    get_zoom_pan: _GetView


# ──────────────────────────────────────────────────────────────────────────────
# Klasa bazowa narzędzia
# ──────────────────────────────────────────────────────────────────────────────

class ToolBase:
    """
    Bazowa klasa narzędzia ImageCanvas. Narzędzia powinny dziedziczyć z ToolBase
    i nadpisywać wybrane metody zdarzeń. Minimalne wymagania:
      - obsługa myszy (down/move/up),
      - (opcjonalnie) obsługa wheel/klawiszy,
      - rysowanie overlay (draw_overlay).
    """
    name: str = "base"

    def __init__(self, ctx: ToolEventContext) -> None:
        self.ctx = ctx
        # Wspólny stan dla narzędzi (np. czy w trakcie drag)
        self._active: bool = False

    # ── cykl życia narzędzia ──────────────────────────────────────────────────
    def on_activate(self, opts: Optional[Dict[str, Any]] = None) -> None:
        """
        Wywoływane przy przełączeniu na to narzędzie.
        opts: parametry startowe (np. rozmiar pędzla).
        """
        self._active = False  # domyślnie nic nie trwa

    def on_deactivate(self) -> None:
        """Wywoływane przy opuszczeniu narzędzia (sprzątanie stanu)."""
        self._active = False

    # ── zdarzenia myszy ───────────────────────────────────────────────────────
    def on_mouse_down(self, ev: Any) -> None:
        """ev: obiekt zdarzenia Tk (posiada .x, .y w space ekranu)."""
        self._active = True

    def on_mouse_move(self, ev: Any) -> None:
        """Ruch myszy; jeżeli trwa drag, implementacja narzędzia rysuje overlay."""
        ...

    def on_mouse_up(self, ev: Any) -> None:
        """Zamknięcie gestu myszy (commit operacji, publikacja eventów)."""
        self._active = False

    # ── koło myszy / klawiatura ───────────────────────────────────────────────
    def on_wheel(self, ev: Any) -> None:
        """Domyślnie brak akcji; narzędzia mogą nadpisać (np. zoom)."""
        ...

    def on_key(self, ev: Any) -> None:
        """Obsługa klawiszy specjalnych (np. ESC do anulowania gestu)."""
        ...

    # ── rysowanie overlay ─────────────────────────────────────────────────────
    def draw_overlay(self, tk_canvas: Any) -> None:
        """
        Rysuje elementy overlay na podanym Canvasie Tk.
        Implementacje powinny rysować WYŁĄCZNIE przebitkę (ramki, uchwyty),
        nigdy nie modyfikować obrazu bazowego.
        """
        ...
