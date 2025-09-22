# -*- coding: utf-8 -*-
from __future__ import annotations
import time, types, traceback, ast
from typing import Dict, Any, List, Tuple

# ==========
# Narzędzia
# ==========

def _exec_code(src: str) -> Tuple[types.ModuleType, Exception | None]:
    mod = types.ModuleType("candidate_b")
    try:
        exec(src, mod.__dict__)
        return mod, None
    except Exception as e:
        return None, e

def _run_one_test(fn, args, expect) -> Tuple[bool, str | None]:
    try:
        got = fn(*args)
        ok = (got == expect)
        if ok:
            return True, None
        return False, f"mismatch: got={got!r} expect={expect!r}"
    except Exception as e:
        return False, f"runtime: {e}"

def _run_tests_locally(code: str, task: Dict[str, Any]) -> Dict[str, Any]:
    mod, err = _exec_code(code)
    if err is not None:
        return dict(pass_cnt=0, total=len(task["tests"]), errors=[f"load_error: {err}"])
    try:
        fn = getattr(mod, task["entrypoint"])
    except Exception as e:
        return dict(pass_cnt=0, total=len(task["tests"]), errors=[f"entrypoint_missing: {e}"])
    passed = 0
    errs: List[str] = []
    for t in task["tests"]:
        ok, msg = _run_one_test(fn, t.get("args", []), t["expect"])
        if ok:
            passed += 1
        else:
            errs.append(msg or "fail")
    return dict(pass_cnt=passed, total=len(task["tests"]), errors=errs)

# =========================
# Biblioteka „intencji” (NL)
# =========================

def _intent_from_task(task: Dict[str, Any]) -> str:
    """Bardzo prosta heurystyka wydobycia intencji z entrypoint/prompt."""
    ep = (task.get("entrypoint") or "").lower()
    pr = (task.get("prompt") or "").lower()

    if "reverse" in ep or "odwr" in pr or "reverse" in pr:
        return "reverse_string"
    if ep == "fib" or "fib(" in pr or "fibonacci" in pr:
        return "fibonacci"
    if "csv" in ep or "csv" in pr:
        return "sum_csv"
    if "moving_avg" in ep or "średnia krocząca" in pr or "moving average" in pr:
        return "moving_avg"
    if "count_calls" in ep or ("ast" in pr and "call" in pr):
        return "count_calls"
    return "generic"

# ======================
# Szablony rozwiązań (v1)
# ======================

TEMPLATES: Dict[str, List[str]] = {
    "reverse_string": [
        # wariant 1 — slicing
        "def {name}(s: str) -> str:\n    return s[::-1]\n",
        # wariant 2 — odwracanie przez listę
        "def {name}(s: str) -> str:\n    out=list(s)\n    out.reverse()\n    return ''.join(out)\n",
    ],
    "fibonacci": [
        # iteracyjny (szybki, poprawny)
        "def {name}(n: int) -> int:\n"
        "    a,b=0,1\n"
        "    for _ in range(n):\n"
        "        a,b=b,a+b\n"
        "    return a\n",
        # rekurencyjny (dla małych n)
        "def {name}(n: int) -> int:\n"
        "    if n<=1: return n\n"
        "    return {name}(n-1)+{name}(n-2)\n",
    ],
    "sum_csv": [
        # ostrożne strip + int
        "def {name}(csv_text: str) -> int:\n"
        "    parts=[p.strip() for p in csv_text.split(',') if p.strip()]\n"
        "    return sum(int(p) for p in parts)\n",
        # z newlinami
        "def {name}(csv_text: str) -> int:\n"
        "    txt=csv_text.replace('\\n', ',')\n"
        "    parts=[p.strip() for p in txt.split(',') if p.strip()]\n"
        "    return sum(int(p) for p in parts)\n",
    ],
    "moving_avg": [
        # tylko pełne okna
        "def {name}(x: list, k: int) -> list:\n"
        "    if k<=0 or len(x)<k:\n"
        "        return []\n"
        "    out=[]\n"
        "    s=sum(x[:k])\n"
        "    out.append(s/k)\n"
        "    for i in range(k, len(x)):\n"
        "        s += x[i] - x[i-k]\n"
        "        out.append(s/k)\n"
        "    return out\n",
        # prosty wariant przez sumy odcinkowe
        "def {name}(x: list, k: int) -> list:\n"
        "    if k<=0 or len(x)<k:\n"
        "        return []\n"
        "    return [ sum(x[i:i+k])/k for i in range(0, len(x)-k+1) ]\n",
    ],
    "count_calls": [
        # liczenie ast.Call
        "import ast\n"
        "def {name}(src: str) -> int:\n"
        "    t=ast.parse(src)\n"
        "    return sum(1 for n in ast.walk(t) if isinstance(n, ast.Call))\n",
    ],
    "generic": [
        # szkielet: niech wybuchnie — ale staramy się rozpoznać sygnaturę z testów
        "def {name}(*args, **kwargs):\n    raise NotImplementedError\n",
    ],
}

# ===========================
# Wariatory / szybkie poprawki
# ===========================

def _mutate_for_csv_sum(code: str, name: str) -> List[str]:
    """Gdy testy nie przechodzą — sprawdź warianty z innymi separatorami/spacjami."""
    variants = []
    variants.append(
        "def {name}(csv_text: str) -> int:\n"
        "    import re\n"
        "    nums=re.findall(r'-?\\d+', csv_text)\n"
        "    return sum(int(z) for z in nums)\n".format(name=name)
    )
    return variants

def _mutate_for_mavg(code: str, name: str) -> List[str]:
    """Drobne zmiany: typowanie floatów, rzutowania itd."""
    return [
        "def {name}(x: list, k: int) -> list:\n"
        "    if not isinstance(k,int) or k<=0 or len(x)<k:\n"
        "        return []\n"
        "    out=[]\n"
        "    for i in range(len(x)-k+1):\n"
        "        s=0.0\n"
        "        for j in range(k): s+=x[i+j]\n"
        "        out.append(s/float(k))\n"
        "    return out\n".format(name=name)
    ]

MUTATORS = {
    "sum_csv": _mutate_for_csv_sum,
    "moving_avg": _mutate_for_mavg,
}

# ======================
# Główna strategia „B”
# ======================

def generate_code(task: Dict[str, Any], mode: str="B", **kwargs) -> Dict[str, Any]:
    """
    Zwraca: dict(code:str, metrics:dict, time_s:float)

    Strategia:
    1) Rozpoznaj intencję na bazie entrypoint/prompt.
    2) Spróbuj gotowych szablonów; po każdym – uruchom testy lokalnie.
    3) Jeśli nie przejdzie: uruchom lekkie mutatory (naprawy) specyficzne dla intencji.
    4) Zwróć pierwszy wariant przechodzący wszystkie testy, albo najlepszy (max pass_cnt).
    """
    t0 = time.time()
    entry = task["entrypoint"]
    intent = _intent_from_task(task)

    tried: int = 0
    best = {"code": None, "pass_cnt": -1, "errors": []}
    compile_errors = 0
    repair_steps = 0

    # 1) Szablony
    for tpl in TEMPLATES.get(intent, TEMPLATES["generic"]):
        tried += 1
        code = tpl.format(name=entry)
        res = _run_tests_locally(code, task)
        if res["pass_cnt"] > best["pass_cnt"]:
            best.update(code=code, pass_cnt=res["pass_cnt"], errors=res["errors"])
        if res["pass_cnt"] == res["total"]:
            return dict(
                code=code,
                metrics=dict(
                    intent=intent, tried=tried,
                    compile_errors=compile_errors,
                    repair_steps=repair_steps,
                    pass_cnt=res["pass_cnt"], total=res["total"]
                ),
                time_s=time.time() - t0
            )

    # 2) Mutatory (naprawy) — tylko jeśli są dla danej intencji
    if intent in MUTATORS:
        for tpl in TEMPLATES.get(intent, []):
            base = tpl.format(name=entry)
            variants = MUTATORS[intent](base, entry)
            for v in variants:
                tried += 1
                repair_steps += 1
                res = _run_tests_locally(v, task)
                if res["pass_cnt"] > best["pass_cnt"]:
                    best.update(code=v, pass_cnt=res["pass_cnt"], errors=res["errors"])
                if res["pass_cnt"] == res["total"]:
                    return dict(
                        code=v,
                        metrics=dict(
                            intent=intent, tried=tried,
                            compile_errors=compile_errors,
                            repair_steps=repair_steps,
                            pass_cnt=res["pass_cnt"], total=res["total"]
                        ),
                        time_s=time.time() - t0
                    )

    # 3) Ostateczny fallback: „generic”
    if best["code"] is None:
        code = TEMPLATES["generic"][0].format(name=entry)
        res = _run_tests_locally(code, task)
        best.update(code=code, pass_cnt=res["pass_cnt"], errors=res["errors"])

    return dict(
        code=best["code"],
        metrics=dict(
            intent=intent, tried=tried,
            compile_errors=compile_errors,
            repair_steps=repair_steps,
            pass_cnt=best["pass_cnt"], total=len(task["tests"]),
            last_errors=best["errors"][:3]  # skrót
        ),
        time_s=time.time() - t0
    )
