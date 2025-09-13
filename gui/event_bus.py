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

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional

import tkinter as tk

Callback = Callable[[Dict[str, Any]], None]


@dataclass
class EventBus:
    """A simple pub/sub event bus for UI messages.

    Subscribers can register callbacks for specific topics (exact match
    or prefix ``"topic.*"``).  Publishers send a payload dict on a
    topic.  All callbacks are scheduled on the Tkinter event loop via
    ``master.after`` if a master widget was provided on construction.
    """

    master: Optional[tk.Misc] = None
    _subscribers: Dict[str, List[Callback]] = field(default_factory=lambda: defaultdict(list))

    def subscribe(self, topic: str, callback: Callback) -> None:
        """Subscribe a callback to a topic.

        If the topic ends with ``".*"`` the callback will be invoked
        for any event whose topic shares the prefix before the ``".*"``.
        """
        self._subscribers[topic].append(callback)

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Publish an event with a payload.

        All matching subscribers receive a copy of the payload.  If
        ``master`` was provided the calls will be scheduled on the
        Tkinter thread via ``after``; otherwise callbacks fire
        synchronously.
        """
        # Collect callbacks by matching exact topic and prefix subscriptions
        callbacks: List[Callback] = []
        for t, subs in self._subscribers.items():
            if t.endswith(".*"):
                prefix = t[:-2]
                if topic.startswith(prefix):
                    callbacks.extend(subs)
            elif t == topic:
                callbacks.extend(subs)
        # Execute or schedule callbacks
        for cb in callbacks:
            if self.master is not None:
                # schedule on the Tkinter event loop
                self.master.after(0, cb, payload.copy())
            else:
                cb(payload.copy())
