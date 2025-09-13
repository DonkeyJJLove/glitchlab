"""
---
version: 3
kind: module
id: "gui-event-bus"
created_at: "2025-09-13"
name: "glitchlab.gui.event_bus"
author: "GlitchLab v3"
role: "Lekki pub/sub dla akcji UI i zdarzeń runtime (Tk-safe)"
description: >
Zapewnia tematyczny publish/subscribe z doprowadzeniem callbacków do wątku GUI
przez root.after(0, ...). Służy jako wspólna szyna komunikacyjna dla Views,
Services i Paneli, zachowując jednokierunkowy przepływ danych.
event_model:
event: {topic: "str", ts: "float(unix)", payload: "dict"}
qos: "at-most-once"
ordering: "w obrębie tego samego topicu"
dispatch: "UI-thread via root.after(0, ...)"

topics:

• "ui.*" # interakcje użytkownika (open/save/select/apply, zmiany pól)

• "run.*" # życie zadania pipeline (request/progress/done/error/cancel)

• "hud.*" # zmiany mapowania HUD i odświeżenia

• "preset.*" # operacje na presetach (load/validate/save/history)

• "files.*" # open/save/recent

• "masks.*" # load/normalize/list

• "layout.*" # dock/undock, zapis/odczyt layoutu

    interfaces:
    exports:

    • "EventBus(root_like).subscribe(topic: str, handler: Callable[[dict], None]) -> str # sub_id"

    • "EventBus.publish(topic: str, payload: dict) -> None"

    • "EventBus.unsubscribe(sub_id: str) -> bool"

    • "EventBus.clear_topic(topic_pattern: str) -> int # opcjonalnie"
    used_by:

    • "glitchlab.gui.app_shell"

    • "glitchlab.gui.views.*"

    • "glitchlab.gui.services.*"

    • "glitchlab.gui.panels.*"

    depends_on:

• "threading"

• "time"

• "re"

• "weakref"

• "typing"

    policy:
    deterministic: true
    thread_safe_publish: true
    ui_dispatch_guarantee: "tak — wszystkie handlery wywoływane w wątku GUI"
    constraints:

• "brak zewnętrznych zależności"

• "wzorce topiców wspierają '*' (wildcard segment)"

• "handlery powinny być krótkie; ciężkie prace trafiają do Services"
telemetry:
counters: ["events_total","subs_total","subs_active","drops"]
license: "Proprietary"
---
"""
# glitchlab/gui/event_bus.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import threading
import fnmatch
import uuid

EventCallback = Callable[[str, Dict[str, Any]], None]


@dataclass
class _Sub:
    id: str
    pattern: str
    cb: EventCallback
    on_ui: bool


class EventBus:
    """
    Lekki pub/sub z prostymi wzorcami tematów (glob):
      - dokładny temat: "ui.open"
      - wzorzec z '*': "ui.*" (jedna sekcja)
      - wzorzec z '**': "run.**" (dowolna głębokość)
    Jeśli `on_ui=True` i podano `root_like` z metodą `.after(ms, fn)`,
    wywołania trafią do wątku GUI.
    """

    def __init__(self, root_like: Any | None = None) -> None:
        self._root = root_like
        self._subs: List[_Sub] = []
        self._lock = threading.RLock()

    # ---------------------------
    # Subskrypcja / rejestracja
    # ---------------------------

    def subscribe(self, pattern: str, cb: EventCallback, *, on_ui: bool = False) -> str:
        """
        Rejestruje subskrypcję i zwraca subscription_id (string UUID).
        Wzorce zgodne z fnmatch: '*', '?'. Konwencja '**' traktowana jak '*' z kropkami.
        """
        if not isinstance(pattern, str) or not callable(cb):
            raise TypeError("subscribe(pattern:str, cb:callable, on_ui:bool=False)")
        sid = uuid.uuid4().hex
        sub = _Sub(id=sid, pattern=pattern, cb=cb, on_ui=bool(on_ui))
        with self._lock:
            self._subs.append(sub)
        return sid

    def unsubscribe(self, subscription_id: str) -> bool:
        with self._lock:
            before = len(self._subs)
            self._subs = [s for s in self._subs if s.id != subscription_id]
            return len(self._subs) != before

    def clear(self, pattern: Optional[str] = None) -> None:
        with self._lock:
            if pattern is None:
                self._subs.clear()
            else:
                self._subs = [s for s in self._subs if not self._match(pattern, s.pattern)]

    # -------------
    # Publikacja
    # -------------

    def publish(self, topic: str, payload: Dict[str, Any] | None = None) -> None:
        """
        Publikuje zdarzenie. Jeśli subskrypcja ma on_ui=True i mamy root_like,
        callback zostanie wywołany przez `root.after(0, ...)`.
        """
        if not isinstance(topic, str):
            raise TypeError("publish(topic:str, payload:dict|None)")
        data = dict(payload or {})
        data.setdefault("event_id", uuid.uuid4().hex)
        callbacks: List[Tuple[_Sub, EventCallback]] = []

        with self._lock:
            for s in list(self._subs):
                if self._match(s.pattern, topic):
                    callbacks.append((s, s.cb))

        for sub, cb in callbacks:
            if sub.on_ui and hasattr(self._root, "after"):
                try:
                    self._root.after(0, self._safe_call, cb, topic, data)
                except Exception:
                    # Fallback: jeśli root zamknięty, spróbuj synchronicznie
                    self._safe_call(cb, topic, data)
            else:
                self._safe_call(cb, topic, data)

    # ----------------
    # Narzędzia
    # ----------------

    @staticmethod
    def _match(pattern: str, topic: str) -> bool:
        """
        Dopasowanie wzorca do tematu.
        Konwencja '**' → odpowiada dowolnej sekwencji z kropkami.
        """
        if "**" in pattern:
            # Zamień '**' na '*' w duchu fnmatch (glob bezpośredni).
            pattern = pattern.replace("**", "*")
        return fnmatch.fnmatchcase(topic, pattern)

    @staticmethod
    def _safe_call(cb: EventCallback, topic: str, data: Dict[str, Any]) -> None:
        try:
            cb(topic, data)
        except Exception:
            # Brak loggera: celowo nie podnosimy błędów busa
            pass
