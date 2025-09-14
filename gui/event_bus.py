"""
event_bus.py
================

This module implements a simple publish/subscribe mechanism for the GUI.

It is a lightweight event bus inspired by the planned refactoring in
``gui/REFACTORING.md``.  Views and services can publish events on
string topics and subscribe callbacks to those topics.  All callbacks
are executed in the Tkinter mainloop via the ``after`` mechanism to
avoid thread‐safety issues.  The bus does not enforce any
thread‐safety on its own; the UI code should schedule messages on the
bus using the master widget's ``after`` method if they originate from
background threads.

Example usage::

    from glitchlab.gui.event_bus import EventBus
    bus = EventBus(master=root)

    def on_run_done(payload: dict) -> None:
        print("pipeline finished", payload)

    bus.subscribe("run.done", on_run_done)
    bus.publish("run.done", {"result": "ok"})

Topics are hierarchical: subscribing to ``"run.*"`` will receive
events published on any topic that starts with ``"run."``.  Callbacks
must accept a single ``dict`` payload.
"""

# glitchlab/gui/event_bus.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List, Tuple

Subscriber = Tuple[Callable[[str, Dict[str, Any]], None], bool]  # (fn, on_ui)

class EventBus:
    """
    Lekki event bus dla GUI.
    - subscribe(topic, fn, on_ui=False): rejestracja subskrypcji
    - publish(topic, payload): publikacja zdarzenia
    Jeśli on_ui=True i podano master w konstruktorze, wywołanie fn trafia przez Tk.after(0).
    """

    def __init__(self, master: Any | None = None) -> None:
        self._master = master  # Tk root lub Frame; może być None (tryb headless)
        self._subs: DefaultDict[str, List[Subscriber]] = defaultdict(list)
        self._lock = threading.RLock()

    # --- API ---

    def subscribe(self, topic: str,
                  fn: Callable[[str, Dict[str, Any]], None],
                  on_ui: bool = False) -> None:
        """Zapisz callback dla topic. Jeśli on_ui=True, dispatch przez Tk.after."""
        if not callable(fn):
            return
        with self._lock:
            self._subs[topic].append((fn, bool(on_ui)))

    def unsubscribe(self, topic: str,
                    fn: Callable[[str, Dict[str, Any]], None]) -> None:
        with self._lock:
            self._subs[topic] = [(f, ui) for (f, ui) in self._subs.get(topic, []) if f is not fn]
            if not self._subs[topic]:
                self._subs.pop(topic, None)

    def publish(self, topic: str, payload: Dict[str, Any] | None = None) -> None:
        """Publikuj zdarzenie do wszystkich subskrybentów danego topicu."""
        data = dict(payload or {})
        with self._lock:
            subs = list(self._subs.get(topic, []))
        for fn, on_ui in subs:
            self._dispatch(fn, topic, data, on_ui)

    # --- wewnętrzne ---

    def _dispatch(self, fn: Callable[[str, Dict[str, Any]], None],
                  topic: str, data: Dict[str, Any], on_ui: bool) -> None:
        if on_ui and self._master is not None:
            try:
                # bezpieczny powrót na wątek Tk
                self._master.after(0, lambda: self._safe_call(fn, topic, data))
                return
            except Exception:
                # w razie braku .after – fallback do synchronicznego
                pass
        self._safe_call(fn, topic, data)

    @staticmethod
    def _safe_call(fn: Callable[[str, Dict[str, Any]], None],
                   topic: str, data: Dict[str, Any]) -> None:
        try:
            fn(topic, data)
        except Exception:
            # celowo „połykamy” wyjątki GUI, żeby nie wysypać całej aplikacji
            pass
