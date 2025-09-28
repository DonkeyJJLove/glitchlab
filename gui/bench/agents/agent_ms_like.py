# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from typing import Dict, Any, Optional


# 1) Template Protocol (PTP) – twarde szablony
_TEMPLATES = {
    "reverse_str": (
        "def reverse_str(s: str) -> str:\n"
        "    return s[::-1]\n"
    ),
    "fib": (
        "def fib(n: int) -> int:\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
    ),
}


# 2) Keyword Heuristic Protocol (KH) – heurystyki nazw/oczekiwań
def _heuristic_code(name: str, task: Dict[str, Any]) -> Optional[str]:
    lname = name.lower()
    if "palindrome" in lname:
        return (
            "def is_palindrome(s: str) -> bool:\n"
            "    t = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
            "    return t == t[::-1]\n"
        )
    if "factorial" in lname:
        return (
            "def factorial(n: int) -> int:\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be >= 0')\n"
            "    res = 1\n"
            "    for i in range(2, n+1):\n"
            "        res *= i\n"
            "    return res\n"
        )
    if "gcd" in lname:
        return (
            "def gcd(a: int, b: int) -> int:\n"
            "    a, b = abs(a), abs(b)\n"
            "    while b:\n"
            "        a, b = b, a % b\n"
            "    return a\n"
        )
    return None


# 3) Test-Induction Protocol (TIP) – prosta analiza testów
def _induction_code(name: str, task: Dict[str, Any]) -> Optional[str]:
    tests = task.get("tests", [])
    if not tests:
        return None

    # przykład: reverse detection
    if all(isinstance(t.get("expect"), str) for t in tests):
        if all(
            isinstance(t.get("args", [None])[0], str)
            and t["expect"] == t["args"][0][::-1]
            for t in tests if t.get("args")
        ):
            return (
                f"def {name}(s: str) -> str:\n"
                f"    return s[::-1]\n"
            )

    # przykład: sum of numbers
    if all(isinstance(t.get("expect"), int) for t in tests):
        if all(isinstance(t.get("args", [None])[0], list) for t in tests):
            return (
                f"def {name}(xs: list) -> int:\n"
                f"    return sum(xs)\n"
            )

    return None


# 4) Safety Fallback Protocol – bezpieczny stub
def _fallback_code(name: str) -> str:
    return f"def {name}(*args, **kwargs):\n    raise NotImplementedError\n"


# 5) Główny interfejs
def generate_code(task: Dict[str, Any], mode: str = "B", **kwargs) -> Dict[str, Any]:
    """
    Implementacja baseline Microsoftu:
      - Template Protocol (PTP)
      - Keyword Heuristic Protocol (KH)
      - Test-Induction Protocol (TIP)
      - Safety Fallback
    """
    t0 = time.time()
    name = task["entrypoint"]

    if name in _TEMPLATES:
        code = _TEMPLATES[name]
        proto = "template"
    else:
        hcode = _heuristic_code(name, task)
        if hcode:
            code = hcode
            proto = "heuristic"
        else:
            icode = _induction_code(name, task)
            if icode:
                code = icode
                proto = "induction"
            else:
                code = _fallback_code(name)
                proto = "fallback"

    return dict(
        code=code,
        metrics={"protocol": proto},
        time_s=time.time() - t0
    )
