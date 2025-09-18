# glitchlab/gui/views/layer_panel.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Dict, List, Optional

# Dostępne tryby mieszania (zgodne z gui/services/compositor.py)
BLEND_MODES = ["normal", "multiply", "screen", "overlay", "add", "subtract", "darken", "lighten"]


class LayersPanel(ttk.Frame):
    """
    Panel zarządzania warstwami:
      • Lista warstw (TOP→BOTTOM), wybór aktywnej
      • Widoczność, krycie (opacity), blend
      • Dodaj / Usuń
      • Reorder (Up / Down)
    Integracja z EventBus (publish/subscribe):
      → publish: ui.layer.set_active, ui.layer.update, ui.layer.add, ui.layer.remove, ui.layers.reorder
      ← subscribe: ui.layers.changed  (odśwież UI z AppState)
    Oczekiwania:
      • App publikuje ui.layers.changed po każdej zmianie (LayerManager / App)
      • AppState dostępny w kontekście, ale panel działa bez bezpośredniego importu stanu
    """

    def __init__(self, master: tk.Misc, *, bus: Optional[Any] = None) -> None:
        super().__init__(master)
        self.bus = bus

        # Dane UI
        self._layers: List[Dict[str, Any]] = []  # cieniowana lista [{id, name, visible, opacity, blend}, ...]
        self._active_id: Optional[str] = None
        self._suspend_events: bool = False  # tłumik emisji przy programowej synchronizacji

        # Kontrolki szczegółów
        self._name_var = tk.StringVar(value="")
        self._visible_var = tk.BooleanVar(value=True)
        self._opacity_var = tk.DoubleVar(value=1.0)
        self._blend_var = tk.StringVar(value=BLEND_MODES[0])

        self._build_ui()
        self._wire_bus()

    # ───────────────────────────── UI BUILD ─────────────────────────────
    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Nagłówek + przyciski
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))
        ttk.Label(header, text="Layers", font=("", 10, "bold")).pack(side="left")
        header_btns = ttk.Frame(header)
        header_btns.pack(side="right")
        ttk.Button(header_btns, text="+", width=3, command=self._on_add).pack(side="left", padx=2)
        ttk.Button(header_btns, text="−", width=3, command=self._on_remove).pack(side="left", padx=2)
        ttk.Button(header_btns, text="Up", width=4, command=self._on_move_up).pack(side="left", padx=4)
        ttk.Button(header_btns, text="Down", width=6, command=self._on_move_down).pack(side="left")

        # Lista warstw (Treeview)
        self.tree = ttk.Treeview(self, show="headings", selectmode="browse", height=10)
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

        self.tree.bind("<<TreeviewSelect>>", self._on_select_row)
        self.tree.bind("<Double-1>", self._on_toggle_visible)  # double-click na wierszu przełącza widoczność

        # Szczegóły aktywnej warstwy
        details = ttk.LabelFrame(self, text="Active Layer")
        details.grid(row=2, column=0, sticky="ew", padx=4, pady=(4, 6))
        details.columnconfigure(1, weight=1)

        ttk.Label(details, text="Name").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        e_name = ttk.Entry(details, textvariable=self._name_var)
        e_name.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        e_name.bind("<Return>", lambda _e: self._emit_update({"name": self._name_var.get()}))
        e_name.bind("<FocusOut>", lambda _e: self._emit_update({"name": self._name_var.get()}))

        cb_vis = ttk.Checkbutton(details, text="Visible", variable=self._visible_var,
                                 command=lambda: self._emit_update({"visible": bool(self._visible_var.get())}))
        cb_vis.grid(row=1, column=0, sticky="w", padx=6, pady=4)

        ttk.Label(details, text="Opacity").grid(row=1, column=1, sticky="w", padx=(6, 0), pady=4)
        sc_op = ttk.Scale(details, from_=0.0, to=1.0, variable=self._opacity_var,
                          command=lambda _v: self._emit_update({"opacity": float(self._opacity_var.get())}))
        sc_op.grid(row=1, column=1, sticky="ew", padx=(70, 6), pady=4)

        ttk.Label(details, text="Blend").grid(row=2, column=0, sticky="w", padx=6, pady=(4, 8))
        cb_blend = ttk.Combobox(details, state="readonly", values=BLEND_MODES, textvariable=self._blend_var)
        cb_blend.grid(row=2, column=1, sticky="ew", padx=6, pady=(4, 8))
        cb_blend.bind("<<ComboboxSelected>>", lambda _e: self._emit_update({"blend": self._blend_var.get()}))

        # Pasek statusu mały (opcjonalnie)
        self._status = ttk.Label(self, text="", anchor="w")
        self._status.grid(row=3, column=0, sticky="ew", padx=4, pady=(0, 4))

    # ───────────────────────────── BUS ─────────────────────────────
    def _wire_bus(self) -> None:
        if not (self.bus and hasattr(self.bus, "subscribe")):
            return
        try:
            self.bus.subscribe("ui.layers.changed", lambda *_: self.refresh_from_state())
        except Exception:
            pass

    # Publiczna metoda do zainicjowania panelu stanem (opcjonalnie)
    def refresh_from_state(self) -> None:
        """
        Odczyt aktualnego stanu warstw z AppState przez bus:
          założenie: AppState jest źródłem prawdy, a panel tylko odbija stan.
        App publikuje 'ui.layers.state' na żądanie i/lub 'ui.layers.changed' po zmianach.
        Aby zachować prostotę i kompatybilność wsteczną, panel wykona best-effort:
          1) Spróbuje pobrać stan przez callable bus.get_state('layers') (jeżeli jest),
          2) W przeciwnym razie — App powinien opublikować 'ui.layers.changed' z minimalnym payloadem:
             {"layers":[{id,name,visible,opacity,blend},...], "active": "<id>"}
        """
        if not self.bus:
            return
        payload = None
        # Preferuj dedykowaną metodę: bus.request(...) jeśli istnieje
        try:
            if hasattr(self.bus, "request"):
                payload = self.bus.request("ui.layers.dump", {})  # type: ignore[attr-defined]
        except Exception:
            payload = None

        # Alternatywnie — App mógł wcześniej ustawić ostatni snapshot:
        if payload is None:
            try:
                # Niektóre implementacje bus mają cache ostatniego eventu — tu tylko best-effort.
                payload = getattr(self.bus, "last_layers_snapshot", None)  # type: ignore[attr-defined]
            except Exception:
                payload = None

        # Fallback: nic nie mamy — spróbuj wywołać event pull (App może odpowiedzieć publish)
        if payload is None:
            try:
                self.bus.publish("ui.layers.pull", {})
            except Exception:
                pass
            # i niech kolejny 'ui.layers.changed' nas zaktualizuje
            return

        self._apply_snapshot(payload)

    # API dla App do bezpośredniego podania snapshotu
    def set_snapshot(self, layers: List[Dict[str, Any]], active: Optional[str]) -> None:
        snap = {"layers": layers, "active": active}
        self._apply_snapshot(snap)

    # ───────────────────────────── SNAPSHOT → UI ─────────────────────────────
    def _apply_snapshot(self, snap: Dict[str, Any]) -> None:
        layers = list(snap.get("layers") or [])
        active = snap.get("active")
        self._suspend_events = True
        try:
            # Odtwórz listę
            self.tree.delete(*self.tree.get_children())
            self._layers = []
            for l in layers:
                lid = str(l.get("id"))
                name = str(l.get("name", "Layer"))
                vis = bool(l.get("visible", True))
                op = float(l.get("opacity", 1.0))
                blend = str(l.get("blend", "normal"))
                self._layers.append({"id": lid, "name": name, "visible": vis, "opacity": op, "blend": blend})
                self.tree.insert("", "end", iid=lid, values=(name, "✓" if vis else " ", f"{op:.2f}", blend))

            # Zaznacz aktywną i wypełnij szczegóły
            self._active_id = str(active) if active else (self._layers[0]["id"] if self._layers else None)
            if self._active_id and self._active_id in self.tree.get_children(""):
                self.tree.selection_set(self._active_id)
                self.tree.see(self._active_id)
                self._fill_details_from_active()
            else:
                self._clear_details()
            self._status.config(text=f"Layers: {len(self._layers)}")
        finally:
            self._suspend_events = False

    # ───────────────────────────── DETAILS BIND ─────────────────────────────
    def _fill_details_from_active(self) -> None:
        l = self._get_active_layer_dict()
        if not l:
            self._clear_details()
            return
        self._name_var.set(l["name"])
        self._visible_var.set(bool(l["visible"]))
        self._opacity_var.set(float(l["opacity"]))
        blend = l["blend"] if l["blend"] in BLEND_MODES else "normal"
        self._blend_var.set(blend)

    def _clear_details(self) -> None:
        self._name_var.set("")
        self._visible_var.set(True)
        self._opacity_var.set(1.0)
        self._blend_var.set("normal")

    # ───────────────────────────── EVENTS ─────────────────────────────
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
        # double-click → toggle visible
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
        # domyślnie: duplikuj aktywną (jeśli jest) — łatwa weryfikacja UX
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

    # ───────────────────────────── EMIT UPDATE ─────────────────────────────
    def _emit_update(self, patch: Dict[str, Any]) -> None:
        if self._suspend_events:
            return
        if not self._active_id:
            return
        # validacja podstawowa
        if "blend" in patch and patch["blend"] not in BLEND_MODES:
            patch["blend"] = "normal"
        if "opacity" in patch:
            try:
                patch["opacity"] = float(min(1.0, max(0.0, float(patch["opacity"]))))
            except Exception:
                patch["opacity"] = 1.0
        self._publish("ui.layer.update", {"id": self._active_id, **patch})

    # ───────────────────────────── UTIL ─────────────────────────────
    def _get_active_layer_dict(self) -> Optional[Dict[str, Any]]:
        if not self._active_id:
            return None
        return self._find_layer(self._active_id)

    def _find_layer(self, lid: str) -> Optional[Dict[str, Any]]:
        for l in self._layers:
            if l.get("id") == lid:
                return l
        return None

    def _current_order(self) -> List[str]:
        # kolejność iids w Treeview od top→bottom
        return list(self.tree.get_children(""))

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self._suspend_events:
            return
        if self.bus and hasattr(self.bus, "publish"):
            try:
                self.bus.publish(topic, payload)
            except Exception:
                pass
        # Local status (debug UX)
        self._status.config(text=f"{topic}: {payload}")
