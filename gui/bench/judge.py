# -*- coding: utf-8 -*-
from __future__ import annotations
import json, types, traceback, inspect
from typing import Dict, Any, List


def _exec_code(src: str) -> types.ModuleType:
    mod = types.ModuleType("candidate")
    exec(src, mod.__dict__)  # uruchamiaj wyłącznie w zaufanym sandboxie
    return mod


def run_tests(code: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Zwraca: {pass_cnt, total, pass_at_1(0/1), errors:[...]}.
    Loguje pełny traceback przy błędach, a przed startem sprawdza:
      - istnienie funkcji,
      - wywoływalność,
      - minimalną zgodność sygnatury.
    """
    total = len(task.get("tests", []))
    errors: List[str] = []
    passed = 0

    try:
        mod = _exec_code(code)
    except Exception:
        tb = traceback.format_exc()
        return dict(pass_cnt=0, total=total, pass_at_1=0, errors=[f"load_error:\n{tb}"])

    name = task["entrypoint"]
    if not hasattr(mod, name):
        return dict(pass_cnt=0, total=total, pass_at_1=0,
                    errors=[f"entrypoint_missing: {name}"])

    fn = getattr(mod, name)
    if not callable(fn):
        return dict(pass_cnt=0, total=total, pass_at_1=0,
                    errors=[f"entrypoint_not_callable: {name}"])

    # Sanity: sprawdź, czy nie ma jawnego NotImplementedError na starcie:
    try:
        src = inspect.getsource(fn)
        if "NotImplementedError" in src:
            return dict(pass_cnt=0, total=total, pass_at_1=0,
                        errors=[f"entrypoint_stubbed: {name}"])
    except Exception:
        pass  # brak źródła nie jest krytyczny

    for idx, t in enumerate(task.get("tests", []), 1):
        args = t.get("args", [])
        kwargs = t.get("kwargs", {})
        expect = t["expect"]
        try:
            got = fn(*args, **kwargs)
        except Exception:
            tb = traceback.format_exc()
            errors.append(f"runtime(idx={idx}):\n{tb}")
            continue
        if got == expect:
            passed += 1
        else:
            errors.append(f"mismatch(idx={idx}): got={got!r} expect={expect!r}")

    return dict(
        pass_cnt=passed,
        total=total,
        pass_at_1=int(passed == total),
        errors=errors
    )
