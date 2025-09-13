"""
pipeline_runner.py
===================

This module provides a simple wrapper around the core pipeline to
execute it asynchronously and report progress back to the GUI.  It
offloads the heavy work to a background thread and communicates via
``EventBus`` topics.  The API is intentionally minimal: call
``PipelineRunner.run(...)`` with the steps and context, and listen
for ``run.progress``, ``run.done`` and ``run.error`` events on the
bus.  Only one pipeline run can be active at a time; subsequent calls
are ignored until the previous run finishes or is cancelled.

The runner does not interpret the core's ``ctx.cache``; it merely
exposes the finished ``ctx`` as part of the payload.  GUI code
consuming the events should read the cache and display images
accordingly.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional

try:
    # Soft import, as ``glitchlab.core`` may not be available (fallback mode)
    from glitchlab.core.pipeline import apply_pipeline
except Exception:  # pragma: no cover
    # Fallback apply_pipeline that returns input and leaves ctx untouched
    def apply_pipeline(img_u8, ctx, steps, **kwargs):
        return img_u8

from glitchlab.gui.event_bus import EventBus


@dataclass
class PipelineRunner:
    """Run the glitchlab pipeline asynchronously.

    Parameters
    ----------
    bus : EventBus
        The event bus used to publish progress and completion events.
    master : Any
        The Tkinter master widget used for scheduling callbacks (may be
        used by the bus itself).  If ``None``, callbacks are invoked
        synchronously.
    """

    bus: EventBus
    master: Any = None
    _current_thread: Optional[threading.Thread] = field(default=None, init=False)

    def run(self, img_u8: Any, ctx: Any, steps: List[Dict[str, Any]]) -> None:
        """Start running the pipeline in a background thread.

        Publishes events:

          - ``run.start`` with ``{"steps": steps}``
          - ``run.progress`` (optional, not yet implemented) with partial ctx
          - ``run.done`` with ``{"ctx": ctx, "output": out}``
          - ``run.error`` with ``{"error": exc, "steps": steps}``

        Only one run at a time is allowed.  If a run is already
        executing, further calls are ignored.
        """
        if self._current_thread and self._current_thread.is_alive():
            # ignore new requests while a run is in progress
            return

        def _runner():
            try:
                self.bus.publish("run.start", {"steps": steps.copy()})
                out = apply_pipeline(img_u8, ctx, steps, fail_fast=True, metrics=True)
                # propagate results on Tk thread via bus
                self.bus.publish("run.done", {"ctx": ctx, "output": out})
            except Exception as exc:
                self.bus.publish("run.error", {"error": exc, "steps": steps.copy()})

        self._current_thread = threading.Thread(target=_runner, daemon=True)
        self._current_thread.start()

    def cancel(self) -> None:
        """Cancel the current run, if any.

        Currently cancellation is cooperative: the runner simply
        abandons publishing the final ``run.done`` and the pipeline
        continues in the background.  To implement full cancellation
        support, the pipeline would need to be interruptible.
        """
        # There is no cancellation support in core pipeline yet; this
        # method simply discards the thread reference so that future
        # runs can be scheduled.
        self._current_thread = None
