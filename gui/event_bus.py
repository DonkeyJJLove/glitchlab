# glitchlab/gui/event_bus.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import traceback
from typing import Any, Callable, DefaultDict, Dict, List, Optional
import tkinter as tk
from collections import defaultdict


class EventBus:
    """
    Prosty event bus z odpornością na reentrancję:
      • subscribe(topic, cb) – rejestruje callback: (topic:str, payload:dict) -> None
      • publish(topic, payload) – ASYNCHRONICZNIE (tk.after_idle) emituje do subskrybentów
      • request(topic, payload) – SYNCHRONICZNE zapytanie: wywołuje pierwszą odpowiedź i zwraca wynik

    Uwagi:
      - Asynchroniczne publish() rozcina łańcuchy „publish w publish”, co eliminuje wieszki
        po otwieraniu pliku (np. ui.layers.changed -> snapshot -> ui.layers.changed ...).
      - request() jest używane oszczędnie (np. panel prosi o snapshot); ma pozostać synchroniczne.
      - Dodatkowe, nieformalne pole `last_layers_snapshot` może być nadawane przez aplikację
        (App._publish_layers_snapshot), panel sobie je odczyta (best-effort).
    """

    def __init__(self, root: tk.Misc) -> None:
        self._root = root
        self._subs: DefaultDict[str, List[Callable[[str, Dict[str, Any]], None]]] = defaultdict(list)
        # prosty bufor kolejkowanych publikacji (opcjonalnie do debugu)
        self._queue_len = 0

    # ───────────────── subscribe ─────────────────
    def subscribe(self, topic: str, cb: Callable[[str, Dict[str, Any]], None]) -> None:
        if not callable(cb):
            return
        self._subs[topic].append(cb)

    # ───────────────── publish (async) ─────────────────
    def publish(self, topic: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Asynchroniczna publikacja – każdy subscriber wywołany przez after_idle."""
        data = dict(payload or {})
        cbs = list(self._subs.get(topic, []))

        # DEBUG: ograniczony ślad tego, co publikujemy (można podejrzeć w konsoli)
        try:
            if topic == "ui.layers.changed":
                src = data.get("source")
                print(f"[BUS] publish ui.layers.changed (source={src}) -> {len(cbs)} subs")
            elif topic.startswith("ui.files.") or topic in ("run.start", "run.done", "run.error"):
                print(f"[BUS] publish {topic} -> {len(cbs)} subs")
        except Exception:
            pass

        def _dispatch_one(cb: Callable[[str, Dict[str, Any]], None]) -> None:
            try:
                cb(topic, data)
            except Exception:
                traceback.print_exc()

        # Kolejkujemy każde wywołanie osobno – minimalizuje czas jednego idle-callbacku
        for cb in cbs:
            try:
                self._queue_len += 1
                self._root.after_idle(self._wrap_dispatch(cb, topic, data))
            except Exception:
                # Jeśli after_idle niedostępne – awaryjnie synchronicznie (lepiej niż zamilknąć)
                _dispatch_one(cb)

    def _wrap_dispatch(self, cb: Callable[[str, Dict[str, Any]], None], topic: str, data: Dict[str, Any]):
        def _runner():
            try:
                cb(topic, data)
            except Exception:
                traceback.print_exc()
            finally:
                try:
                    self._queue_len -= 1
                except Exception:
                    pass
        return _runner

    # ───────────────── request (sync) ─────────────────
    def request(self, topic: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        """
        Syntetyczne „zapytanie” – jeśli ktoś się zasubskrybował pod dany topic jako handler requestów,
        wywołujemy PIERWSZEGO subskrybenta synchronicznie i zwracamy jego wynik.
        UWAGA: używać oszczędnie; publish() i tak robi robotę asynchronicznie.
        """
        cbs = self._subs.get(topic, [])
        if not cbs:
            return None
        cb = cbs[0]
        try:
            return cb(topic, dict(payload or {}))  # type: ignore[return-value]
        except Exception:
            traceback.print_exc()
            return None
