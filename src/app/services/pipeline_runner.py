# glitchlab/app/services/pipeline_runner.py
# -*- coding: utf-8 -*-
"""
GLX-CTX:v1
component: app.services.pipeline_runner
role: async pipeline runner with GUI progress signals

# MOSAIC::PROFILE (S/H/Z)
S: threading.Thread,threading.Lock
H: core.pipeline.apply_pipeline,app.event_bus.EventBus.publish
Z: 1

# AST::SURFACE
imports: glitchlab.core.pipeline.apply_pipeline,glitchlab.app.event_bus.EventBus,threading,dataclasses,typing
public_api: PipelineRunner.is_running,PipelineRunner.run,PipelineRunner.cancel

# CONTRACTS::EVENTS
publish:
  run.start: steps:list
  run.progress: value:float[0,1],text:str
  run.done: ctx:any,output:any
  run.error: error:str,steps:list

# INVARIANTS
- single_active_run
- cancel_soft_only
- progress_0_then_1
- publish_is_best_effort

# DATA::SOURCES
img_u8: app.widgets.image_canvas|app.services.compositor
ctx: app.state.State.snapshot|panel caller
steps: app.preset_manager|presets/*.yaml|widgets.param_form

# DATA::SINKS
events: app.event_bus (HUD,log_window,progress)
output: run.done→caller (np. services.compositor)
telemetry: analysis.reporting subskrybuje run.*

# GRAMMAR::HOOKS (komentarze #glx:event=… → Δ)
enter_scope/exit_scope, define/use, link, bucket_jump, reassign, contract/expand

# TAG-SCHEMA
- # glx:ast.fn=<Name>
- # glx:mosaic.S=<csv>
- # glx:mosaic.H=<csv>
- # glx:contracts.publish=<csv>
- # glx:data.in=<csv>   | # glx:data.out=<csv>
- # glx:event=<kind[:args]>   # kind∈{enter_scope,exit_scope,define,use,link,bucket_jump,reassign,contract,expand}

# PARSER
- listy w tagach: CSV (bez spacji); wartości surowe; klucz=wartość
- kompatybilność: dopuszczalny JSON w przyszłości; aktualny parser: CSV
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---- soft import core ----
# glx:ast.section=soft_imports
# glx:event=define:apply_pipeline
# glx:contracts.note=soft-import
try:
    from glitchlab.core.pipeline import apply_pipeline  # type: ignore
except Exception:  # pragma: no cover
    def apply_pipeline(img_u8, ctx, steps, **_kw):  # fallback noop
        return img_u8

# ---- EventBus (oczekiwany interfejs: publish(topic, payload)) ----
# glx:event=define:EventBus
# glx:contracts.note=soft-import
try:
    from glitchlab.app.event_bus import EventBus  # type: ignore
except Exception:  # pragma: no cover
    class EventBus:  # minimal stub
        def __init__(self, *_a, **_k): ...
        def publish(self, topic: str, payload: Dict[str, Any]) -> None:
            print(f"[bus] {topic}: {payload}")


# glx:component=app.services.pipeline_runner
# glx:event=enter_scope:PipelineRunner
# glx:mosaic.S=thread_ctx,lock_guard
# glx:mosaic.H=apply_pipeline,event_bus
# glx:contracts.publish=run.start,run.progress,run.done,run.error
# glx:data.in=img_u8,ctx,steps
# glx:data.out=events,output
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

    # glx:ast.fn=is_running
    # glx:event=enter_scope:is_running
    # glx:mosaic.S=thread_probe
    # glx:invariants=idempotent_probe
    def is_running(self) -> bool:
        t = self._thread
        result = bool(t and t.is_alive())
        # glx:event=exit_scope
        return result

    # glx:ast.fn=run
    # glx:event=enter_scope:run
    # glx:mosaic.S=lock,thread_start
    # glx:mosaic.H=apply_pipeline,event_bus.publish
    # glx:contracts.publish=run.start,run.progress
    # glx:data.in=img_u8,ctx,steps
    # glx:invariants=single_active_run,progress_0_then_1
    def run(self, img_u8: Any, ctx: Any, steps: List[Dict[str, Any]]) -> None:
        """
        Startuje asynchroniczne wykonanie `apply_pipeline`.
        Ignoruje wywołanie, jeśli poprzedni bieg jeszcze trwa.
        """
        with self._lock:
            if self.is_running():
                # glx:event=use:services.publish  # (brak emisji — świadome)
                # glx:event=exit_scope
                return
            self._cancel_flag = False
            self._thread = threading.Thread(
                target=self._runner, args=(img_u8, ctx, list(steps) or []),
                daemon=True
            )
            self._thread.start()
        # glx:event=exit_scope

    # glx:ast.fn=cancel
    # glx:event=enter_scope:cancel
    # glx:mosaic.S=lock
    # glx:invariants=cancel_soft_only
    def cancel(self) -> None:
        """
        „Miękkie” anulowanie – przestaje publikować wyniki końcowe.
        (Nie zatrzymuje samego `apply_pipeline`, które nie jest przerywalne.)
        """
        with self._lock:
            self._cancel_flag = True
        # glx:event=exit_scope

    # ------------------------- INTERNAL --------------------------

    # glx:ast.fn=_emit
    # glx:event=define:services.publish
    # glx:event=use:services.publish
    # glx:contracts.publish=*
    # glx:data.in=topic,payload
    # glx:invariants=publish_is_best_effort
    def _emit(self, topic: str, payload: Dict[str, Any]) -> None:
        try:
            self.bus.publish(topic, dict(payload))
        except Exception:
            pass  # łykamy wyjątki busa

    # glx:ast.fn=_progress
    # glx:event=use:services.publish
    # glx:contracts.publish=run.progress
    # glx:data.in=value∈[0,1],text
    def _progress(self, value: float, text: str) -> None:
        self._emit("run.progress", {"value": float(max(0.0, min(1.0, value))), "text": str(text)})
        # glx:event=exit_scope

    # glx:ast.fn=_runner
    # glx:event=enter_scope:_runner
    # glx:mosaic.S=thread_body,finally_cleanup
    # glx:mosaic.H=apply_pipeline,event_bus.publish
    # glx:contracts.publish=run.start,run.done,run.error
    # glx:data.in=img_u8,ctx,steps
    # glx:data.out=output
    # glx:invariants=progress_0_then_1,cancel_soft_only
    def _runner(self, img_u8: Any, ctx: Any, steps: List[Dict[str, Any]]) -> None:
        # start
        self._emit("run.start", {"steps": steps})
        self._progress(0.0, "Starting…")

        try:
            # (streaming postępu możliwy, gdy core na to pozwoli)
            out = apply_pipeline(img_u8, ctx, steps, fail_fast=True, metrics=True)  # type: ignore[arg-type]
            if self._cancel_flag:
                # glx:event=exit_scope
                return
            self._progress(1.0, "Completed")
            self._emit("run.done", {"ctx": ctx, "output": out})
        except Exception as exc:
            if self._cancel_flag:
                # glx:event=exit_scope
                return
            self._progress(1.0, "Error")
            self._emit("run.error", {"error": str(exc), "steps": steps})
        finally:
            # wyczyść referencję wątku
            with self._lock:
                self._thread = None
                self._cancel_flag = False
        # glx:event=exit_scope
# glx:event=exit_scope
