"""
---
version: 3
kind: module
id: "gui-services-pipeline_runner"
created_at: "2025-09-13"
name: "glitchlab.gui.services.pipeline_runner"
author: "GlitchLab v3"
role: "Background job queue for pipeline runs (Tk-safe)"
description: >
  Jednowątkowy executor z kolejką zadań do uruchamiania przetwarzania (np. apply_pipeline)
  w tle, z obsługą anulowania przed startem, telemetrią czasu, bezpiecznym przekazywaniem
  callbacków do wątku GUI przez root.after(0, ...) oraz opcjonalnym kanałem postępu.

inputs:
  submit.label:        {type: "str"}
  submit.func:         {type: "Callable", note: "np. core.pipeline.apply_pipeline"}
  submit.args:         {type: "tuple"}
  submit.kw:           {type: "dict"}
  submit.on_done:      {type: "Callable[[job_id, result], None]",  optional: true}
  submit.on_error:     {type: "Callable[[job_id, exc], None]",     optional: true}
  submit.on_progress:  {type: "Callable[[job_id, percent, message|None], None]", optional: true}
  cancel(job_id):      {type: "bool", note: "soft-cancel przez token lub pominięcie w kolejce"}
  cancel_all():        {type: "None"}

outputs:
  job_id:     {type: "str"}
  telemetry:  {type: "dict", keys: ["runs_total","runs_failed","avg_ms","queue_len","current_job"]}

interfaces:
  exports: ["PipelineRunner", "CancellationToken"]
  public_api:
    - "PipelineRunner(root_like).submit(label, func, *args, on_done=None, on_error=None, on_progress=None, **kw) -> str"
    - "PipelineRunner.cancel(job_id) -> bool"
    - "PipelineRunner.cancel_all() -> None"
    - "PipelineRunner.list_jobs() -> list[dict]"
    - "PipelineRunner.telemetry() -> dict"
    - "PipelineRunner.shutdown(wait:bool=False) -> None"

depends_on: ["threading","queue","time","uuid","inspect","typing","dataclasses"]
used_by: ["glitchlab.gui.app","glitchlab.gui.views.statusbar","glitchlab.gui.views.hud","glitchlab.gui.views.notebook"]

policy:
  deterministic: true
  thread_model: "single worker thread + Tk-safe callbacks via root.after"
  side_effects: ["threads"]

constraints:
  - "Tk-callbacks są wołane przez root.after(0, ...)"
  - "Anulowanie jest łagodne: dotyczy zadań w kolejce lub przez token współpracy"
notes:
  - "Jeśli func akceptuje 'cancel_token' i/lub 'progress', runner je poda (po nazwie parametru)"
  - "Brak zależności od EventBus – integracja opcjonalna po stronie AppShell"
license: "Proprietary"
---
"""
# glitchlab/gui/services/pipeline_runner.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, List
import threading
from queue import Queue, Empty
import time
import uuid
import inspect

ProgressCb = Callable[[str, float, Optional[str]], None]
DoneCb = Callable[[str, Any], None]
ErrorCb = Callable[[str, BaseException], None]


class CancellationToken:
    """Współpracujący token anulowania dla funkcji długotrwałych."""
    __slots__ = ("_flag",)

    def __init__(self) -> None:
        self._flag = threading.Event()

    def cancel(self) -> None:
        self._flag.set()

    def is_cancelled(self) -> bool:
        return self._flag.is_set()


@dataclass
class _Job:
    id: str
    label: str
    func: Callable[..., Any]
    args: Tuple[Any, ...]
    kw: Dict[str, Any]
    on_done: Optional[DoneCb] = None
    on_error: Optional[ErrorCb] = None
    on_progress: Optional[ProgressCb] = None
    token: CancellationToken = field(default_factory=CancellationToken)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    status: str = "queued"  # queued|running|done|failed|cancelled
    error: Optional[str] = None
    result: Any = None


class PipelineRunner:
    """
    Jednowątkowy executor z kolejką FIFO. Gwarantuje, że callbacki są wywoływane
    w wątku GUI przez `root.after(0, ...)`.
    """

    def __init__(self, root_like: Any, *, thread_name: str = "PipelineRunner") -> None:
        if not hasattr(root_like, "after"):
            raise TypeError("PipelineRunner: root_like musi mieć metodę .after(delay_ms, func)")
        self._root = root_like
        self._q: Queue[_Job] = Queue()
        self._lock = threading.Lock()
        self._jobs: Dict[str, _Job] = {}
        self._worker = threading.Thread(target=self._loop, name=thread_name, daemon=True)
        self._stop = threading.Event()
        self._telemetry = {
            "runs_total": 0,
            "runs_failed": 0,
            "sum_ms": 0.0,
            "avg_ms": 0.0,
            "queue_len": 0,
            "current_job": None,  # job_id
        }
        self._worker.start()

    # --------------------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------------------

    def submit(
        self,
        label: str,
        func: Callable[..., Any],
        /,
        *args: Any,
        on_done: Optional[DoneCb] = None,
        on_error: Optional[ErrorCb] = None,
        on_progress: Optional[ProgressCb] = None,
        **kw: Any,
    ) -> str:
        """
        Dodaje zadanie do kolejki i zwraca job_id.
        Jeśli `func` akceptuje parametry 'cancel_token' i/lub 'progress', runner je poda.
        """
        job_id = uuid.uuid4().hex
        job = _Job(
            id=job_id,
            label=str(label),
            func=func,
            args=tuple(args),
            kw=dict(kw),
            on_done=on_done,
            on_error=on_error,
            on_progress=on_progress,
        )
        with self._lock:
            self._jobs[job_id] = job
        self._q.put(job)
        self._update_queue_len()
        return job_id

    def cancel(self, job_id: str) -> bool:
        """
        Oznacza zadanie jako anulowane:
          - jeśli jeszcze nie wystartowało, worker pominie jego wykonanie,
          - jeśli funkcja wspiera token ('cancel_token'), może przerwać współpracująco.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            job.token.cancel()
            if job.status == "queued":
                job.status = "cancelled"
                job.finished_at = time.time()
                return True
            return True  # running → soft cancel

    def cancel_all(self) -> None:
        with self._lock:
            for job in self._jobs.values():
                job.token.cancel()

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._job_info(job) if job else None

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._job_info(j) for j in self._jobs.values()]

    def telemetry(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._telemetry)

    def shutdown(self, *, wait: bool = False) -> None:
        """Zatrzymuje pętlę workerską (np. przy zamknięciu aplikacji)."""
        self._stop.set()
        # wpuść „pusty” sygnał do kolejki, by przerwać blokujące get()
        self._q.put_nowait(_Job(id="__stop__", label="__stop__", func=lambda: None, args=(), kw={}))
        if wait:
            self._worker.join(timeout=1.5)

    # --------------------------------------------------------------------------------------
    # Worker
    # --------------------------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.2)
            except Empty:
                continue
            if self._stop.is_set():
                break
            # jeśli już anulowane w kolejce – omiń
            if job.token.is_cancelled() and job.status == "cancelled":
                continue
            # Skip techniczny
            if job.id == "__stop__":
                break

            # start
            now = time.time()
            job.started_at = now
            job.status = "running"
            self._set_current(job.id)

            # jeśli anulowane tuż przed startem – przeskocz
            if job.token.is_cancelled() and job.status != "done":
                job.status = "cancelled"
                job.finished_at = time.time()
                self._finish(job, None, None)
                self._clear_current()
                self._update_queue_len()
                continue

            try:
                result = self._call_job(job)
                job.result = result
                job.status = "done"
                job.finished_at = time.time()
                self._telemetry["runs_total"] += 1
                self._accum_ms(job)
                self._finish(job, result, None)
            except BaseException as ex:  # noqa: BLE001
                job.error = repr(ex)
                job.status = "failed"
                job.finished_at = time.time()
                self._telemetry["runs_total"] += 1
                self._telemetry["runs_failed"] += 1
                self._accum_ms(job)
                self._finish(job, None, ex)
            finally:
                self._clear_current()
                self._update_queue_len()

    # --------------------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------------------

    def _call_job(self, job: _Job) -> Any:
        """
        Woła funkcję zadania. Jeśli funkcja przewiduje 'cancel_token'/'progress'
        (po nazwie parametru), runner je przekaże.
        """
        try:
            sig = inspect.signature(job.func)
        except (TypeError, ValueError):
            sig = None  # builtins / C-callables itp.

        kw = dict(job.kw)
        if sig is not None:
            params = sig.parameters
            if "cancel_token" in params:
                kw["cancel_token"] = job.token
            if "progress" in params:
                kw["progress"] = self._make_progress(job)

        return job.func(*job.args, **kw)

    def _make_progress(self, job: _Job) -> Callable[[float, Optional[str]], None]:
        """
        Zwraca funkcję progress(percent, message|None), która bezpiecznie przeniesie
        zdarzenie do wątku GUI i zadzwoni on_progress(job_id, ...), jeśli podany.
        """
        def _progress(percent: float, message: Optional[str] = None) -> None:
            cb = job.on_progress
            if cb is None:
                return
            self._post_ui(lambda: cb(job.id, float(percent), message))
        return _progress

    def _post_ui(self, fn: Callable[[], None]) -> None:
        try:
            self._root.after(0, fn)
        except Exception:
            # ostateczny fallback – wykonaj synchronicznie (gdy root zamknięty)
            try:
                fn()
            except Exception:
                pass

    def _finish(self, job: _Job, result: Any, error: Optional[BaseException]) -> None:
        cb_done = job.on_done
        cb_err = job.on_error
        if error is None and cb_done is not None:
            self._post_ui(lambda: cb_done(job.id, result))
        if error is not None and cb_err is not None:
            self._post_ui(lambda: cb_err(job.id, error))

    def _accum_ms(self, job: _Job) -> None:
        if job.started_at is None or job.finished_at is None:
            return
        dt_ms = (job.finished_at - job.started_at) * 1000.0
        self._telemetry["sum_ms"] += float(dt_ms)
        runs = max(1, int(self._telemetry["runs_total"]))
        self._telemetry["avg_ms"] = self._telemetry["sum_ms"] / runs

    def _update_queue_len(self) -> None:
        with self._lock:
            # Queue.qsize() bywa przybliżone, ale wystarczy do HUD
            try:
                self._telemetry["queue_len"] = max(0, self._q.qsize())
            except NotImplementedError:
                self._telemetry["queue_len"] = 0

    def _set_current(self, job_id: Optional[str]) -> None:
        with self._lock:
            self._telemetry["current_job"] = job_id

    def _clear_current(self) -> None:
        with self._lock:
            self._telemetry["current_job"] = None

    def _job_info(self, job: Optional[_Job]) -> Dict[str, Any]:
        if job is None:
            return {}
        return {
            "id": job.id,
            "label": job.label,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
            "has_result": job.result is not None,
        }
