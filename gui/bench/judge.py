# -*- coding: utf-8 -*-
from __future__ import annotations
import json, types, traceback
from typing import Dict, Any, List


def _exec_code(src: str) -> types.ModuleType:
    mod = types.ModuleType("candidate")
    exec(src, mod.__dict__)  # bezpiecznie tylko w zaufanym środowisku testowym
    return mod


def run_tests(code: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Zwraca: {pass_cnt, total, pass@1(0/1), errors:[]}
    """
    try:
        mod = _exec_code(code)
        fn = getattr(mod, task["entrypoint"])
    except Exception as e:
        return dict(pass_cnt=0, total=len(task["tests"]), pass_at_1=0, errors=[f"load_error: {e}"])

    passed = 0
    errs: List[str] = []
    for t in task["tests"]:
        args = t.get("args", [])
        expect = t["expect"]
        try:
            got = fn(*args)
        except Exception as e:
            errs.append(f"runtime: {e}")
            continue
        if got == expect:
            passed += 1
        else:
            errs.append(f"mismatch: got={got!r} expect={expect!r}")
    return dict(pass_cnt=passed, total=len(task["tests"]), pass_at_1=int(passed == len(task["tests"])), errors=errs)
