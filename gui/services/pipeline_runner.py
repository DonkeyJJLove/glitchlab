# glitchlab/gui/services/pipeline_runner.py
# -*- coding: utf-8 -*-
"""
Asynchroniczny uruchamiacz potoku (pipeline) z sygnałami postępu dla GUI.

Publikowane zdarzenia na EventBus:
  • run.start     {steps}
  • run.progress  {value: 0..1, text: str}
  • run.done      {ctx, output}
  • run.error     {error: str, steps}

Cechy:
  • pojedynczy aktywny bieg (kolejne żądania są ignorowane)
  • best-effort progress (0% na starcie, 100% na końcu lub przy błędzie)
  • cancel() zatrzymuje publikację zdarzeń końcowych (nie przerywa core)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---- soft import core ----
try:
    from glitchlab.core.pipeline import apply_pipeline  # type: ignore
except Exception:  # pragma: no cover
    def apply_pipeline(img_u8, ctx, steps, **_kw):  # fallback noop
        return img_u8

# ---- EventBus (oczekiwany interfejs: publish(topic, payload)) ----
try:
    from glitchlab.gui.event_bus import EventBus  # type: ignore
except Exception:  # pragma: no cover
    class EventBus:  # minimal stub
        def __init__(self, *_a, **_k): ...
        def publish(self, topic: str, payload: Dict[str, Any]) -> None:
            print(f"[bus] {topic}: {payload}")


@dataclass
class PipelineRunner:
    """
    Uruchamia pipeline w tle (wątku daemon) i publikuje zdarzenia na busie.
    """
    bus: EventBus
    master: Any = None  # niewykorzystywane tutaj, ale zachowana sygnatura
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _cancel_flag: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # ------------------------- PUBLIC API -------------------------

    def is_running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    def run(self, img_u8: Any, ctx: Any, steps: List[Dict[str, Any]]) -> None:
        """
        Startuje asynchroniczne wykonanie `apply_pipeline`.
        Ignoruje wywołanie, jeśli poprzedni bieg jeszcze trwa.
        """
        with self._lock:
            if self.is_running():
                return
            self._cancel_flag = False
            self._thread = threading.Thread(
                target=self._runner, args=(img_u8, ctx, list(steps) or []),
                daemon=True
            )
            self._thread.start()

    def cancel(self) -> None:
        """
        „Miękkie” anulowanie – przestaje publikować wyniki końcowe.
        (Nie zatrzymuje samego `apply_pipeline`, które nie jest przerywalne.)
        """
        with self._lock:
            self._cancel_flag = True

    # ------------------------- INTERNAL --------------------------

    def _emit(self, topic: str, payload: Dict[str, Any]) -> None:
        try:
            self.bus.publish(topic, dict(payload))
        except Exception:
            pass

    def _progress(self, value: float, text: str) -> None:
        self._emit("run.progress", {"value": float(max(0.0, min(1.0, value))), "text": str(text)})

    def _runner(self, img_u8: Any, ctx: Any, steps: List[Dict[str, Any]]) -> None:
        # start
        self._emit("run.start", {"steps": steps})
        self._progress(0.0, "Starting…")

        try:
            # (tu moglibyśmy w przyszłości streamować postęp, jeśli core na to pozwoli)
            out = apply_pipeline(img_u8, ctx, steps, fail_fast=True, metrics=True)  # type: ignore[arg-type]
            if self._cancel_flag:
                return
            self._progress(1.0, "Completed")
            self._emit("run.done", {"ctx": ctx, "output": out})
        except Exception as exc:
            if self._cancel_flag:
                return
            self._progress(1.0, "Error")
            self._emit("run.error", {"error": str(exc), "steps": steps})
        finally:
            # wyczyść referencję wątku
            with self._lock:
                self._thread = None
                self._cancel_flag = False
