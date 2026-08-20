from __future__ import annotations

import inspect
import keyword
import os
import signal
import traceback
import types
from typing import Any, Dict, List


class TimeoutError(Exception):
    """Raised when a single test case execution exceeds time limit."""


class UnsafeExecutionDisabled(RuntimeError):
    """Raised when generated code would execute without an external sandbox opt-in."""


_UNSANDBOXED_OPT_IN = "GLITCHLAB_ALLOW_UNSANDBOXED_EXEC"


def _timeout_handler(signum, frame):
    raise TimeoutError("Task execution timed out")


def _exec_code(src: str) -> types.ModuleType:
    """Execute candidate code only after an explicit unsafe-execution opt-in.

    Python ``exec`` in the current interpreter is *not* a security sandbox.  The
    benchmark historically relied on a comment telling callers to use a trusted
    sandbox, while the function itself enforced nothing.  Default behaviour is
    therefore fail-closed.

    Set ``GLITCHLAB_ALLOW_UNSANDBOXED_EXEC=1`` only inside an external disposable
    sandbox/container whose filesystem, network, credentials and process
    privileges are already constrained.
    """
    if os.environ.get(_UNSANDBOXED_OPT_IN) != "1":
        raise UnsafeExecutionDisabled(
            "Generated-code execution is disabled because this process is not a "
            "sandbox. Run the benchmark in an externally isolated environment and "
            f"set {_UNSANDBOXED_OPT_IN}=1 only inside that sandbox."
        )

    mod = types.ModuleType("candidate")
    exec(src, mod.__dict__)
    return mod


def _is_safe_entrypoint(name: Any) -> bool:
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)


def run_tests(code: str, task: Dict[str, Any], timeout_s: int = 2) -> Dict[str, Any]:
    """Run benchmark cases and return pass/error statistics.

    Candidate code execution is disabled by default.  See ``_exec_code`` for the
    explicit opt-in contract required when the caller has already provided an
    external sandbox.
    """
    total = len(task.get("tests", []))
    errors: List[str] = []
    passed = 0
    name = task.get("entrypoint")

    if not _is_safe_entrypoint(name):
        return dict(
            pass_cnt=0,
            total=total,
            pass_at_1=0,
            errors=[f"invalid_entrypoint: {name!r}"],
        )

    try:
        mod = _exec_code(code)
    except UnsafeExecutionDisabled as exc:
        return dict(
            pass_cnt=0,
            total=total,
            pass_at_1=0,
            errors=[f"unsafe_execution_disabled: {exc}"],
        )
    except Exception:
        tb = traceback.format_exc()
        return dict(pass_cnt=0, total=total, pass_at_1=0, errors=[f"load_error:\n{tb}"])

    if not hasattr(mod, name):
        return dict(
            pass_cnt=0,
            total=total,
            pass_at_1=0,
            errors=[f"entrypoint_missing: {name}"],
        )

    fn = getattr(mod, name)
    if not callable(fn):
        return dict(
            pass_cnt=0,
            total=total,
            pass_at_1=0,
            errors=[f"entrypoint_not_callable: {name}"],
        )

    try:
        src = inspect.getsource(fn)
        if "NotImplementedError" in src:
            return dict(
                pass_cnt=0,
                total=total,
                pass_at_1=0,
                errors=[f"entrypoint_stubbed: {name}"],
            )
    except Exception:
        pass

    for idx, test_case in enumerate(task.get("tests", []), 1):
        args = test_case.get("args", [])
        kwargs = test_case.get("kwargs", {})
        expect = test_case["expect"]

        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_s)
            got = fn(*args, **kwargs)
            signal.alarm(0)
        except TimeoutError as exc:
            errors.append(f"timeout(idx={idx}): {exc}")
            continue
        except Exception:
            tb = traceback.format_exc()
            errors.append(f"runtime(idx={idx}):\n{tb}")
            continue
        finally:
            signal.alarm(0)

        if got == expect:
            passed += 1
        else:
            errors.append(f"mismatch(idx={idx}): got={got!r} expect={expect!r}")

    return dict(
        pass_cnt=passed,
        total=total,
        pass_at_1=int(passed == total),
        errors=errors,
    )
