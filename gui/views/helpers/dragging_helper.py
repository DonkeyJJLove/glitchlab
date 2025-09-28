# glitchlab/gui/views/helper/dragging.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover
    tk = None
    ttk = None


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


@dataclass
class FloaterState:
    key: str
    x: int
    y: int
    w: int
    h: int
    visible: bool = True
    minimized: bool = False
    z: int = 0  # prosty z-index (rosnący)



class FloatingLayoutManager:
    """
    Lekki manager pływających paneli (floaterów):
      • register/add → place(),
      • show/hide/toggle,
      • move/resize, bring_to_front,
      • clamp w granicach `bounds`,
      • persist: save_layout() / load_layout(),
      • opcjonalny EventBus: ui.float.register, ui.float.show/hide/toggle/move/resize/front,
                            ui.float.save_layout, ui.float.load_layout, ui.float.bounds
    """
    def __init__(self, bounds: "tk.Misc", bus: Optional[Any] = None) -> None:
        if tk is None:
            raise RuntimeError("Tkinter not available")
        self.bounds = bounds
        self.bus = bus
        self._floaters: Dict[str, Tuple["tk.Misc", FloaterState]] = {}
        self._z_counter = 1

        # reaguj na resize kontenera → dociśnij floatery do granic
        try:
            self.bounds.bind("<Configure>", lambda _e: self.clamp_all(), add="+")
        except Exception:
            pass

        self._wire_bus()

    # ─────────────────────────────────────────── BUS ──────────────────────────
    def _wire_bus(self) -> None:
        if not (self.bus and hasattr(self.bus, "subscribe")):
            return

        def _g(d: Dict[str, Any], k: str, default=None):
            return (d or {}).get(k, default)

        try:
            self.bus.subscribe("ui.float.register",
                               lambda _t, d: self.add(
                                   key=_g(d, "key"),
                                   widget=_g(d, "widget"),
                                   x=_g(d, "x"), y=_g(d, "y"),
                                   w=_g(d, "w", 320), h=_g(d, "h", 200),
                                   relx=_g(d, "relx"), rely=_g(d, "rely"),
                                   visible=_g(d, "visible", True),
                                   make_draggable=_g(d, "make_draggable", False),
                                   drag_handle_selector=_g(d, "drag_handle_selector"),
                               ))
            self.bus.subscribe("ui.float.show",
                               lambda _t, d: self.show(_g(d, "key"), True))
            self.bus.subscribe("ui.float.hide",
                               lambda _t, d: self.show(_g(d, "key"), False))
            self.bus.subscribe("ui.float.toggle",
                               lambda _t, d: self.toggle(_g(d, "key")))
            self.bus.subscribe("ui.float.move",
                               lambda _t, d: self.move_to(_g(d, "key"), _g(d, "x"), _g(d, "y")))
            self.bus.subscribe("ui.float.resize",
                               lambda _t, d: self.resize_to(_g(d, "key"), _g(d, "w"), _g(d, "h")))
            self.bus.subscribe("ui.float.front",
                               lambda _t, d: self.bring_to_front(_g(d, "key")))
            self.bus.subscribe("ui.float.save_layout",
                               lambda _t, _d: self._publish("ui.float.layout_saved", {"layout": self.save_layout()}))
            self.bus.subscribe("ui.float.load_layout",
                               lambda _t, d: self.load_layout(_g(d, "layout") or {}))
            self.bus.subscribe("ui.float.bounds",
                               lambda _t, d: self.set_bounds(_g(d, "bounds") or self.bounds))
        except Exception:
            pass

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self.bus and hasattr(self.bus, "publish"):
            try:
                self.bus.publish(topic, dict(payload))
            except Exception:
                pass

    # ────────────────────────────────────── PUBLIC API ────────────────────────
    def add(
        self,
        *,
        key: str,
        widget: "tk.Misc",
        x: Optional[int] = None,
        y: Optional[int] = None,
        w: int = 320,
        h: int = 200,
        relx: Optional[float] = None,
        rely: Optional[float] = None,
        visible: bool = True,
        make_draggable: bool = False,
        drag_handle_selector: Optional[str] = None,
    ) -> None:
        """Zarejestruj floater i umieść go wewnątrz bounds."""
        if not key or widget is None:
            return

        # wstępne wymiary
        w = max(80, int(w))
        h = max(60, int(h))

        bx, by, bw, bh = self._bounds_geom()
        if relx is not None and rely is not None and (x is None or y is None):
            # pozycja relatywna → absolutna w obrębie bounds
            x = int(bx + relx * (bw - w))
            y = int(by + rely * (bh - h))

        if x is None: x = 100
        if y is None: y = 100

        # przygotuj stan i place
        st = FloaterState(key=key, x=int(x), y=int(y), w=w, h=h, visible=bool(visible), z=self._z_counter)
        self._z_counter += 1
        self._floaters[key] = (widget, st)

        # ewentualny drag – jeśli panel sam nie obsługuje
        if make_draggable:
            try:
                handle = self._select_handle(widget, drag_handle_selector)
                self.make_draggable(widget, handle=handle)
            except Exception:
                pass

        self._apply_place(key)
        if not visible:
            try: widget.place_forget()
            except Exception: pass

        # z-index
        try:
            widget.lift()
        except Exception:
            pass

    def remove(self, key: str) -> None:
        tup = self._floaters.pop(key, None)
        if not tup:
            return
        widget, _ = tup
        try:
            widget.place_forget()
        except Exception:
            pass

    def show(self, key: str, visible: bool) -> None:
        tup = self._floaters.get(key)
        if not tup:
            return
        widget, st = tup
        st.visible = bool(visible)
        if visible:
            self._apply_place(key)
            try: widget.lift()
            except Exception: pass
        else:
            try: widget.place_forget()
            except Exception: pass

    def toggle(self, key: str) -> None:
        tup = self._floaters.get(key)
        if not tup:
            return
        _w, st = tup
        self.show(key, not st.visible)

    def move_to(self, key: str, x: Optional[int], y: Optional[int]) -> None:
        tup = self._floaters.get(key)
        if not tup or x is None or y is None:
            return
        _w, st = tup
        st.x, st.y = int(x), int(y)
        self._apply_place(key, clamp=True)

    def resize_to(self, key: str, w: Optional[int], h: Optional[int]) -> None:
        tup = self._floaters.get(key)
        if not tup or w is None or h is None:
            return
        _w, st = tup
        st.w, st.h = max(80, int(w)), max(60, int(h))
        self._apply_place(key, clamp=True)

    def bring_to_front(self, key: str) -> None:
        tup = self._floaters.get(key)
        if not tup:
            return
        widget, st = tup
        st.z = self._z_counter
        self._z_counter += 1
        try:
            widget.lift()
        except Exception:
            pass

    def clamp_all(self) -> None:
        for key in list(self._floaters.keys()):
            self._apply_place(key, clamp=True)

    def set_bounds(self, new_bounds: "tk.Misc") -> None:
        """Zmiana kontenera granicznego (np. przeniesienie w inne okno/ramek)."""
        if not new_bounds:
            return
        try:
            self.bounds.unbind("<Configure>")
        except Exception:
            pass
        self.bounds = new_bounds
        try:
            self.bounds.bind("<Configure>", lambda _e: self.clamp_all(), add="+")
        except Exception:
            pass
        self.clamp_all()

    # ───────────────────────────── persist (save/load) ────────────────────────
    def save_layout(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for k, (_w, st) in self._floaters.items():
            out[k] = asdict(st)
        return out

    def load_layout(self, data: Dict[str, Dict[str, Any]]) -> None:
        if not data:
            return
        for k, (_w, st) in list(self._floaters.items()):
            lay = data.get(k)
            if not lay:
                continue
            try:
                st.x = int(lay.get("x", st.x))
                st.y = int(lay.get("y", st.y))
                st.w = max(80, int(lay.get("w", st.w)))
                st.h = max(60, int(lay.get("h", st.h)))
                st.visible = bool(lay.get("visible", st.visible))
                st.minimized = bool(lay.get("minimized", st.minimized))
            except Exception:
                pass
            # zastosuj
            if st.visible:
                self._apply_place(k, clamp=True)
            else:
                try: self._floaters[k][0].place_forget()
                except Exception: pass

    # ───────────────────────── helper: drag dla dowolnego panelu ─────────────
    def make_draggable(self, widget: "tk.Misc", *, handle: Optional["tk.Misc"] = None) -> None:
        """
        Dodaje obsługę drag (screen→bounds, DPI-safe) do `widget`.
        `handle` – widżet, na którym złapiemy mysz (np. belka tytułowa).
        """
        handle = handle or widget
        state = {"drag": False, "off": (0, 0), "start": (0, 0), "job": None}

        def _grab(_e=None):
            # start – policz offset w układzie bounds
            sx, sy = widget.winfo_x(), widget.winfo_y()
            brx, bry = self.bounds.winfo_rootx(), self.bounds.winfo_rooty()
            mx = widget.winfo_pointerx() - brx
            my = widget.winfo_pointery() - bry
            state["off"] = (mx - sx, my - sy)
            state["start"] = (mx, my)
            state["drag"] = False
            try: handle.grab_set()
            except Exception: pass
            return "break"

        def _move(_e=None):
            if state["job"] is not None:
                return "break"

            def _apply():
                state["job"] = None
                brx, bry = self.bounds.winfo_rootx(), self.bounds.winfo_rooty()
                mx = widget.winfo_pointerx() - brx
                my = widget.winfo_pointery() - bry

                if not state["drag"]:
                    sx, sy = state["start"]
                    if abs(mx - sx) < 2 and abs(my - sy) < 2:
                        return
                    state["drag"] = True

                offx, offy = state["off"]
                nx, ny = int(mx - offx), int(my - offy)
                # clamp
                bx, by, bw, bh = self._bounds_geom()
                ww, wh = max(1, int(widget.winfo_width())), max(1, int(widget.winfo_height()))
                nx = _clamp(nx, 0, bw - ww)
                ny = _clamp(ny, 0, bh - wh)
                try:
                    widget.place_configure(x=nx, y=ny)
                except Exception:
                    pass

            state["job"] = widget.after(16, _apply)
            return "break"

        def _release(_e=None):
            state["drag"] = False
            if state["job"] is not None:
                try: widget.after_cancel(state["job"])
                except Exception: pass
                state["job"] = None
            try: handle.grab_release()
            except Exception: pass
            return "break"

        for ev, fn in (("<Button-1>", _grab), ("<B1-Motion>", _move), ("<ButtonRelease-1>", _release)):
            handle.bind(ev, fn, add="+")
        # zablokuj bąbelkowanie do canvasa
        for ev in ("<Button-1>", "<B1-Motion>", "<ButtonRelease-1>", "<MouseWheel>", "<Button-4>", "<Button-5>"):
            handle.bind(ev, lambda _e: "break", add="+")  # noqa: E731

    # ───────────────────────────────────── internals ──────────────────────────
    def _apply_place(self, key: str, *, clamp: bool = False) -> None:
        tup = self._floaters.get(key)
        if not tup:
            return
        widget, st = tup
        bx, by, bw, bh = self._bounds_geom()
        x, y, w, h = st.x, st.y, st.w, st.h
        if clamp:
            ww = max(1, int(widget.winfo_width() or w))
            wh = max(1, int(widget.winfo_height() or h))
            x = _clamp(x, 0, bw - ww)
            y = _clamp(y, 0, bh - wh)
            st.x, st.y = x, y
        try:
            widget.place(x=x, y=y, width=w, height=h)
        except Exception:
            pass

    def _bounds_geom(self) -> Tuple[int, int, int, int]:
        """Zwraca (x, y, w, h) geometry bounds w układzie własnym."""
        try:
            w = max(1, int(self.bounds.winfo_width()))
            h = max(1, int(self.bounds.winfo_height()))
        except Exception:
            w, h = 1, 1
        return 0, 0, w, h

    def _select_handle(self, widget: "tk.Misc", selector: Optional[str]) -> "tk.Misc":
        """
        Prosty 'selector': 'children:name' (np. 'children:title', gdy handle ma widget.winfo_name() == 'title').
        Gdy brak lub nie znaleziono – zwróć widget.
        """
        if not selector:
            return widget
        try:
            if selector.startswith("children:"):
                name = selector.split(":", 1)[1]
                for ch in widget.winfo_children():
                    if str(ch.winfo_name()) == name:
                        return ch
        except Exception:
            pass
        return widget
