# glitchlab/gui/views/layer_panel.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Dict, List, Optional

# Dostępne tryby mieszania (zgodne z gui/services/compositor.py)
BLEND_MODES = [
    "normal", "multiply", "screen", "overlay",
    "add", "subtract", "darken", "lighten"
]


class LayersPanel(ttk.Frame):
    """
    Panel zarządzania warstwami:
      • Lista warstw (TOP→BOTTOM), wybór aktywnej
      • Widoczność, krycie (opacity), blend
      • Dodaj / Usuń
      • Reorder (Up / Down)
      • Szczegóły aktywnej warstwy

    Panel jest pozycjonowany przez `place()` i działa jako „floating”.
    Przeciąganie liczone w screen coords → konwersja do układu *bounds*
    (DPI-safe). Po zmianie rozmiaru bounds panel jest utrzymywany w ramach.
    """

    MIN_WIDTH = 480
    MIN_HEIGHT = 300

    def __init__(self, master: tk.Misc, *, bus: Optional[Any] = None, bounds: Optional[tk.Misc] = None) -> None:
        super().__init__(master, relief="raised", borderwidth=2)
        self.bus = bus
        self._is_minimized = False

        # obszar ograniczeń (domyślnie rodzic panelu)
        self._bounds = bounds if bounds is not None else master

        # rozmiar (utrzymujemy stały podczas przesuwania)
        self._fixed_w = self.MIN_WIDTH
        self._fixed_h = self.MIN_HEIGHT

        # zmienne UI
        self._name_var = tk.StringVar()
        self._visible_var = tk.BooleanVar(value=True)
        self._opacity_var = tk.DoubleVar(value=1.0)
        self._blend_var = tk.StringVar(value="normal")

        self._layers: List[Dict[str, Any]] = []
        self._active_id: Optional[str] = None
        self._suspend_events = False

        # Belka tytułowa (uchwyt do drag)
        self._titlebar = ttk.Frame(self, style="Title.TFrame")
        self._titlebar.pack(fill="x")
        ttk.Label(self._titlebar, text="Layers", anchor="center").pack(side="left", padx=4)
        ttk.Button(self._titlebar, text="▭", width=3, command=self._toggle_minimize).pack(side="right")

        # ───── Drag (pointer-based, DPI-safe) ─────
        # używamy bounds do konwersji screen→parent
        self._dragging = False
        self._drag_off_x = 0
        self._drag_off_y = 0
        self._drag_start_parent = (0, 0)
        self._motion_job = None  # throttling

        self._titlebar.bind("<Button-1>", self._on_drag_start, add="+")
        self._titlebar.bind("<B1-Motion>", self._on_drag_motion, add="+")
        self._titlebar.bind("<ButtonRelease-1>", self._on_drag_stop, add="+")
        # zablokuj propagację do canvasu (pan/zoom)
        for ev in ("<Button-1>", "<B1-Motion>", "<ButtonRelease-1>", "<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._titlebar.bind(ev, lambda _e: "break", add="+")  # noqa: E731

        # główna zawartość
        self._content = ttk.Frame(self)
        self._content.pack(fill="both", expand=True)

        self._build_ui()
        self._wire_bus()

        # startowa pozycja (floating)
        self.place(x=100, y=100, width=self._fixed_w, height=self._fixed_h)

        # Re-clamp gdy bounds/okno zmienia rozmiar
        try:
            self._bounds.bind("<Configure>", self._on_bounds_configure, add="+")
        except Exception:
            pass

    # ───── Drag & Move (robust) ─────
    def _on_drag_start(self, event: tk.Event):
        # pozycja panelu względem *własnego mastera* (tam żyje place)
        sx, sy = self.winfo_x(), self.winfo_y()

        # pozycja myszy względem *bounds*: x_root/y_root minus root bounds
        b_rx, b_ry = self._bounds.winfo_rootx(), self._bounds.winfo_rooty()
        mx = event.x_root - b_rx
        my = event.y_root - b_ry

        # offset w układzie bounds → podczas ruchu trzymamy stały
        self._drag_off_x = mx - sx
        self._drag_off_y = my - sy
        self._dragging = False
        self._drag_start_parent = (mx, my)

        try:
            self._titlebar.grab_set()
        except Exception:
            pass

        print(f"[DEBUG] drag_start panel=({sx},{sy}) mouse_parent=({mx},{my})")
        return "break"

    def _on_drag_motion(self, _event: tk.Event):
        if self._motion_job is not None:
            # throttling — renderuj co ~16 ms
            return "break"

        def _apply():
            self._motion_job = None
            # bieżąca pozycja myszy względem *bounds*
            b_rx, b_ry = self._bounds.winfo_rootx(), self._bounds.winfo_rooty()
            mx = self.winfo_pointerx() - b_rx
            my = self.winfo_pointery() - b_ry

            if not self._dragging:
                sx, sy = self._drag_start_parent
                if abs(mx - sx) < 2 and abs(my - sy) < 2:
                    return
                self._dragging = True

            # nowe (x,y) panelu w układzie master/place
            new_x = int(mx - self._drag_off_x)
            new_y = int(my - self._drag_off_y)

            # Clamp do bounds
            self.update_idletasks()
            bw = max(1, int(self._bounds.winfo_width()))
            bh = max(1, int(self._bounds.winfo_height()))
            ww = max(1, int(self.winfo_width()))
            wh = max(1, int(self.winfo_height()))
            new_x = max(0, min(new_x, bw - ww))
            new_y = max(0, min(new_y, bh - wh))

            try:
                self.place_configure(x=new_x, y=new_y)
            except Exception:
                pass

            print(f"[DEBUG] drag_motion mouse_parent=({mx},{my}) new=({new_x},{new_y}) size={ww}x{wh}")

        # ~60 Hz
        self._motion_job = self.after(16, _apply)
        return "break"

    def _on_drag_stop(self, _event: tk.Event):
        self._dragging = False
        if self._motion_job is not None:
            try: self.after_cancel(self._motion_job)
            except Exception: pass
            self._motion_job = None
        try:
            self._titlebar.grab_release()
        except Exception:
            pass
        print("[DEBUG] drag_stop")
        return "break"

    def _on_bounds_configure(self, _e: tk.Event) -> None:
        """Utrzymuj panel w granicach po zmianie rozmiaru bounds."""
        try:
            info = self.place_info()
            if not info:
                return
            x = int(float(info.get("x", 0)))
            y = int(float(info.get("y", 0)))
        except Exception:
            x, y = self.winfo_x(), self.winfo_y()

        self.update_idletasks()
        bw = max(1, int(self._bounds.winfo_width()))
        bh = max(1, int(self._bounds.winfo_height()))
        ww = max(1, int(self.winfo_width()))
        wh = max(1, int(self.winfo_height()))
        new_x = max(0, min(x, bw - ww))
        new_y = max(0, min(y, bh - wh))
        if (new_x, new_y) != (x, y):
            self.place_configure(x=int(new_x), y=int(new_y))

    def _toggle_minimize(self) -> None:
        # Minimalizacja: chowamy/odkrywamy zawartość, ale nie zmieniamy place()
        if self._is_minimized:
            self._content.pack(fill="both", expand=True)
            self._is_minimized = False
        else:
            self._content.forget()
            self._is_minimized = True

    # ───── UI BUILD ─────
    def _build_ui(self) -> None:
        frame = self._content
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        # Nagłówek + przyciski
        header = ttk.Frame(frame)
        header.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))
        ttk.Label(header, text="Layers", font=("", 10, "bold")).pack(side="left")
        header_btns = ttk.Frame(header)
        header_btns.pack(side="right")
        ttk.Button(header_btns, text="+", width=3, command=self._on_add).pack(side="left", padx=2)
        ttk.Button(header_btns, text="−", width=3, command=self._on_remove).pack(side="left", padx=2)
        ttk.Button(header_btns, text="Up", width=4, command=self._on_move_up).pack(side="left", padx=4)
        ttk.Button(header_btns, text="Down", width=6, command=self._on_move_down).pack(side="left")

        # Lista warstw
        self.tree = ttk.Treeview(frame, show="headings", selectmode="browse", height=10)
        self.tree.grid(row=1, column=0, sticky="nsew", padx=4, pady=2)

        self.tree["columns"] = ("name", "visible", "opacity", "blend")
        self.tree.heading("name", text="Name")
        self.tree.heading("visible", text="👁")
        self.tree.heading("opacity", text="Opacity")
        self.tree.heading("blend", text="Blend")

        self.tree.column("name", width=160, anchor="w")
        self.tree.column("visible", width=40, anchor="center")
        self.tree.column("opacity", width=80, anchor="center")
        self.tree.column("blend", width=100, anchor="center")

        self.tree.bind("<<TreeviewSelect>>", self._on_select_row, add="+")
        self.tree.bind("<Double-1>", self._on_toggle_visible, add="+")
        # niech Treeview nie „przebija” się do canvasa
        for ev in ("<Button-1>", "<B1-Motion>", "<ButtonRelease-1>"):
            self.tree.bind(ev, lambda _e: None, add="+")  # nie zwracamy "break" by działała selekcja

        # Szczegóły aktywnej warstwy
        details = ttk.LabelFrame(frame, text="Active Layer")
        details.grid(row=2, column=0, sticky="ew", padx=4, pady=(4, 6))
        details.columnconfigure(1, weight=1)

        ttk.Label(details, text="Name").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        e_name = ttk.Entry(details, textvariable=self._name_var)
        e_name.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        e_name.bind("<Return>", lambda _e: self._emit_update({"name": self._name_var.get()}), add="+")
        e_name.bind("<FocusOut>", lambda _e: self._emit_update({"name": self._name_var.get()}), add="+")

        cb_vis = ttk.Checkbutton(
            details, text="Visible", variable=self._visible_var,
            command=lambda: self._emit_update({"visible": bool(self._visible_var.get())})
        )
        cb_vis.grid(row=1, column=0, sticky="w", padx=6, pady=4)

        ttk.Label(details, text="Opacity").grid(row=1, column=1, sticky="w", padx=(6, 0), pady=4)
        sc_op = ttk.Scale(
            details, from_=0.0, to=1.0, variable=self._opacity_var,
            command=lambda _v: self._emit_update({"opacity": float(self._opacity_var.get())})
        )
        sc_op.grid(row=1, column=1, sticky="ew", padx=(70, 6), pady=4)

        ttk.Label(details, text="Blend").grid(row=2, column=0, sticky="w", padx=6, pady=(4, 8))
        cb_blend = ttk.Combobox(details, state="readonly", values=BLEND_MODES, textvariable=self._blend_var)
        cb_blend.grid(row=2, column=1, sticky="ew", padx=6, pady=(4, 8))
        cb_blend.bind("<<ComboboxSelected>>", lambda _e: self._emit_update({"blend": self._blend_var.get()}), add="+")


        # Pasek statusu
        self._status = ttk.Label(frame, text="", anchor="w")
        self._status.grid(row=3, column=0, sticky="ew", padx=4, pady=(0, 4))

    # ───── BUS ─────
    def _wire_bus(self) -> None:
        if not (self.bus and hasattr(self.bus, "subscribe")):
            return
        try:
            self.bus.subscribe("ui.layers.changed", lambda *_: self.refresh_from_state())
        except Exception:
            pass
        try:
            self.after(0, self.refresh_from_state)
        except Exception:
            pass

    def refresh_from_state(self) -> None:
        if not self.bus:
            return
        snap = None
        # 1) spróbuj synchronicznego requestu (jeśli EventBus posiada)
        try:
            if hasattr(self.bus, "request"):
                snap = self.bus.request("ui.layers.dump", {})
        except Exception:
            snap = None
        # 2) fallback na cache publikowany przez App
        if snap is None:
            try:
                snap = getattr(self.bus, "last_layers_snapshot", None)
            except Exception:
                snap = None
        # 3) jeśli nadal brak — poproś App, żeby opublikował snapshot
        if snap is None:
            try:
                self.bus.publish("ui.layers.pull", {})
            except Exception:
                pass
            return
        self._apply_snapshot(snap)

    # ───── SNAPSHOT → UI ─────
    def _apply_snapshot(self, snap: Dict[str, Any]) -> None:
        raw_layers = list(snap.get("layers") or [])
        active = snap.get("active")
        self._suspend_events = True
        try:
            self.tree.delete(*self.tree.get_children())
            self._layers = []
            for l in reversed(raw_layers):
                lid = str(l.get("id"))
                name = str(l.get("name", "Layer"))
                vis = bool(l.get("visible", True))
                op = float(l.get("opacity", 1.0))
                blend = str(l.get("blend", "normal"))
                self._layers.append({"id": lid, "name": name, "visible": vis, "opacity": op, "blend": blend})
                self.tree.insert("", "end", iid=lid, values=(name, "✓" if vis else " ", f"{op:.2f}", blend))

            self._active_id = str(active) if active else (self._layers[0]["id"] if self._layers else None)
            if self._active_id and self.tree.exists(self._active_id):
                self.tree.selection_set(self._active_id)
                self._fill_details_from_active()
            else:
                self._clear_details()

            self._status.config(text=f"Layers: {len(self._layers)}")
        finally:
            self._suspend_events = False

    # ───── DETAILS ─────
    def _fill_details_from_active(self) -> None:
        l = self._get_active_layer_dict()
        if not l:
            self._clear_details()
            return
        self._name_var.set(l["name"])
        self._visible_var.set(bool(l["visible"]))
        self._opacity_var.set(float(l["opacity"]))
        self._blend_var.set(l["blend"] if l["blend"] in BLEND_MODES else "normal")

    def _clear_details(self) -> None:
        self._name_var.set("")
        self._visible_var.set(True)
        self._opacity_var.set(1.0)
        self._blend_var.set("normal")

    # ───── EVENTS ─────
    def _on_select_row(self, _e=None) -> None:
        if self._suspend_events:
            return
        sel = self.tree.selection()
        if not sel:
            return
        lid = sel[0]
        self._active_id = lid
        self._fill_details_from_active()
        self._publish("ui.layer.set_active", {"id": lid})

    def _on_toggle_visible(self, _e=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        lid = sel[0]
        l = self._find_layer(lid)
        if not l:
            return
        new_vis = not bool(l["visible"])
        self._publish("ui.layer.update", {"id": lid, "visible": new_vis})

    def _on_add(self) -> None:
        self._publish("ui.layer.add", {"duplicate_active": True, "name": "Layer"})

    def _on_remove(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        lid = sel[0]
        if len(self._layers) <= 1:
            messagebox.showinfo("Layers", "Cannot remove the last layer.")
            return
        self._publish("ui.layer.remove", {"id": lid})

    def _on_move_up(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        lid = sel[0]
        order = self._current_order()
        try:
            idx = order.index(lid)
        except ValueError:
            return
        if idx <= 0:
            return
        order[idx - 1], order[idx] = order[idx], order[idx - 1]
        self._publish("ui.layers.reorder", {"order": order})

    def _on_move_down(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        lid = sel[0]
        order = self._current_order()
        try:
            idx = order.index(lid)
        except ValueError:
            return
        if idx >= len(order) - 1:
            return
        order[idx + 1], order[idx] = order[idx], order[idx + 1]
        self._publish("ui.layers.reorder", {"order": order})

    # ───── UPDATE EMIT ─────
    def _emit_update(self, patch: Dict[str, Any]) -> None:
        if self._suspend_events or not self._active_id:
            return
        if "blend" in patch and patch["blend"] not in BLEND_MODES:
            patch["blend"] = "normal"
        if "opacity" in patch:
            try:
                patch["opacity"] = float(min(1.0, max(0.0, float(patch["opacity"]))))
            except Exception:
                patch["opacity"] = 1.0
        self._publish("ui.layer.update", {"id": self._active_id, **patch})

    # ───── UTIL ─────
    def _get_active_layer_dict(self) -> Optional[Dict[str, Any]]:
        return self._find_layer(self._active_id) if self._active_id else None

    def _find_layer(self, lid: str) -> Optional[Dict[str, Any]]:
        for l in self._layers:
            if l.get("id") == lid:
                return l
        return None

    def _current_order(self) -> List[str]:
        # kolejność iids w Treeview → TOP→BOTTOM
        return list(self.tree.get_children(""))

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self._suspend_events:
            return
        if self.bus and hasattr(self.bus, "publish"):
            try:
                self.bus.publish(topic, payload)
            except Exception:
                pass
        self._status.config(text=f"{topic}: {payload}")
