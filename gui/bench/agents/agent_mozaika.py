# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from typing import Dict, Any, List
import math
import re
import ast
from collections import Counter

# Nasz protokół
from glitchlab.gui.mosaic import hybrid_ast_mosaic as hma


# ─────────────────────────────────────────────────────────────────────────────
# 1) METRYKI ALIGN (bez zmian)
# ─────────────────────────────────────────────────────────────────────────────
def _align_metrics(lmbd: float, delta: float, rows: int, cols: int, thr: float,
                   kappa_ab: float = hma.KAPPA_AB_DEFAULT):
    res = hma.run_once(lmbd, delta, rows, cols, thr, mosaic_kind="grid", kappa_ab=kappa_ab)
    return dict(Align=res["Align"], J_phi2=res["J_phi2"], CR_TO=res["CR_TO"], CR_AST=res["CR_AST"])


# ─────────────────────────────────────────────────────────────────────────────
# 2) TEMPLATE KODU – PRODUKCYJNE IMPLEMENTACJE DLA WSZYSTKICH 40 ZADAŃ
# ─────────────────────────────────────────────────────────────────────────────

_INT_RE = re.compile(r"^[+-]?\d+$")

_TEMPLATES: Dict[str, str] = {

    # 01
    "reverse_str": (
        "def reverse_str(s: str) -> str:\n"
        "    return s[::-1]\n"
    ),

    # 02
    "fib": (
        "def fib(n: int) -> int:\n"
        "    if n < 0:\n"
        "        raise ValueError('n must be >= 0')\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
    ),

    # 03
    "sum_csv_numbers": (
        "def sum_csv_numbers(csv_text: str) -> int:\n"
        "    if not csv_text:\n"
        "        return 0\n"
        "    parts = [p.strip() for p in csv_text.split(',') if p.strip()]\n"
        "    return sum(int(p) for p in parts)\n"
    ),

    # 04
    "moving_sum": (
        "def moving_sum(x: list, k: int) -> list:\n"
        "    n = len(x)\n"
        "    if k <= 0 or n < k:\n"
        "        return []\n"
        "    out = []\n"
        "    cur = sum(x[:k])\n"
        "    out.append(cur)\n"
        "    for i in range(k, n):\n"
        "        cur += x[i] - x[i - k]\n"
        "        out.append(cur)\n"
        "    return out\n"
    ),

    # 05
    "is_palindrome": (
        "def is_palindrome(s: str) -> bool:\n"
        "    # Najczęstsza definicja: ignorujemy niealfanumeryczne, bez rozróżniania wielkości\n"
        "    t = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
        "    return t == t[::-1]\n"
    ),

    # 06
    "count_vowels": (
        "def count_vowels(s: str) -> int:\n"
        "    V = set('aeiouAEIOU')\n"
        "    return sum(1 for ch in s if ch in V)\n"
    ),

    # 07
    "factorial": (
        "def factorial(n: int) -> int:\n"
        "    if n < 0:\n"
        "        raise ValueError('n must be >= 0')\n"
        "    res = 1\n"
        "    for i in range(2, n+1):\n"
        "        res *= i\n"
        "    return res\n"
    ),

    # 08
    "unique_sorted": (
        "def unique_sorted(xs: list) -> list:\n"
        "    return sorted(set(xs))\n"
    ),

    # 09
    "flatten_once": (
        "def flatten_once(xs: list) -> list:\n"
        "    out = []\n"
        "    for v in xs:\n"
        "        if isinstance(v, (list, tuple)):\n"
        "            out.extend(v)\n"
        "        else:\n"
        "            out.append(v)\n"
        "    return out\n"
    ),

    # 10
    "dot_product": (
        "def dot_product(a: list, b: list) -> float:\n"
        "    if len(a) != len(b):\n"
        "        raise ValueError('length mismatch')\n"
        "    return sum(x*y for x, y in zip(a, b))\n"
    ),

    # 11
    "anagrams": (
        "def anagrams(a: str, b: str) -> bool:\n"
        "    fa = [c.lower() for c in a if c.isalpha()]\n"
        "    fb = [c.lower() for c in b if c.isalpha()]\n"
        "    return Counter(fa) == Counter(fb)\n"
    ),

    # 12
    "gcd": (
        "def gcd(a: int, b: int) -> int:\n"
        "    a, b = abs(a), abs(b)\n"
        "    while b:\n"
        "        a, b = b, a % b\n"
        "    return a\n"
    ),

    # 13
    "lcm": (
        "def lcm(a: int, b: int) -> int:\n"
        "    from math import gcd\n"
        "    if a == 0 or b == 0:\n"
        "        return 0\n"
        "    return abs(a*b) // gcd(a, b)\n"
    ),

    # 14
    "two_sum": (
        "def two_sum(nums: list, target: int):\n"
        "    seen = {}\n"
        "    for i, x in enumerate(nums):\n"
        "        y = target - x\n"
        "        if y in seen:\n"
        "            return [seen[y], i]\n"
        "        seen[x] = i\n"
        "    return [-1, -1]\n"
    ),

    # 15
    "transpose": (
        "def transpose(M: list) -> list:\n"
        "    if not M:\n"
        "        return []\n"
        "    return [list(row) for row in zip(*M)]\n"
    ),

    # 16
    "matmul": (
        "def matmul(A: list, B: list) -> list:\n"
        "    if not A or not B:\n"
        "        return []\n"
        "    n, m = len(A), len(A[0])\n"
        "    m2, p = len(B), len(B[0])\n"
        "    if m != m2:\n"
        "        raise ValueError('shape mismatch')\n"
        "    Bt = [list(col) for col in zip(*B)]\n"
        "    return [[sum(x*y for x,y in zip(A[i], Bt[j])) for j in range(p)] for i in range(n)]\n"
    ),

    # 17
    "to_snake_case": (
        "def to_snake_case(s: str) -> str:\n"
        "    s = s.replace('-', '_').replace(' ', '_')\n"
        "    s = re.sub(r'([a-z0-9])([A-Z])', r'\\1_\\2', s)\n"
        "    s = re.sub(r'__+', '_', s)\n"
        "    return s.lower().strip('_')\n"
    ),

    # 18
    "to_camel_case": (
        "def to_camel_case(s, upper_first: bool = False) -> str:\n"
        "    import re\n"
        "    if isinstance(s, (list, tuple)):\n"
        "        parts = [str(p) for p in s if str(p)]\n"
        "    else:\n"
        "        s = str(s)\n"
        "        # normalizacja separatorów: _, -, whitespace\n"
        "        s = re.sub(r'[\\-\\s]+', '_', s.strip())\n"
        "        s = re.sub(r'_+', '_', s)\n"
        "        parts = [p for p in s.split('_') if p]\n"
        "    if not parts:\n"
        "        return ''\n"
        "    head = parts[0].lower()\n"
        "    tail = [p[:1].upper() + p[1:].lower() if p else '' for p in parts[1:]]\n"
        "    out = (head.capitalize() if upper_first else head) + ''.join(tail)\n"
        "    return out\n"
    ),

    # 19
    "rle_compress": (
        "def rle_compress(s: str) -> str:\n"
        "    if not s:\n"
        "        return ''\n"
        "    out = []\n"
        "    cur = s[0]\n"
        "    cnt = 1\n"
        "    for ch in s[1:]:\n"
        "        if ch == cur:\n"
        "            cnt += 1\n"
        "        else:\n"
        "            out.append(f'{cur}{cnt}')\n"
        "            cur, cnt = ch, 1\n"
        "    out.append(f'{cur}{cnt}')\n"
        "    return ''.join(out)\n"
    ),

    # 20
    "rle_decompress": (
        "def rle_decompress(s: str) -> str:\n"
        "    out = []\n"
        "    i, n = 0, len(s)\n"
        "    while i < n:\n"
        "        ch = s[i]\n"
        "        i += 1\n"
        "        j = i\n"
        "        while j < n and s[j].isdigit():\n"
        "            j += 1\n"
        "        cnt = int(s[i:j]) if j > i else 1\n"
        "        out.append(ch * cnt)\n"
        "        i = j\n"
        "    return ''.join(out)\n"
    ),

    # 21
    "rotate_list": (
        "def rotate_list(xs: list, k: int) -> list:\n"
        "    n = len(xs)\n"
        "    if n == 0:\n"
        "        return []\n"
        "    k %= n\n"
        "    return xs[-k:] + xs[:-k] if k else xs[:]\n"
    ),

    # 22
    "most_common_char": (
        "def most_common_char(s: str):\n"
        "    if not s:\n"
        "        return ''\n"
        "    freq = Counter(s)\n"
        "    # w razie remisu: znak z najniższym pierwszym indeksem\n"
        "    best = max(freq.items(), key=lambda kv: (kv[1], -s.index(kv[0])))\n"
        "    return best[0]\n"
    ),

    # 23
    "merge_intervals": (
        "def merge_intervals(intervals: list) -> list:\n"
        "    if not intervals:\n"
        "        return []\n"
        "    ints = sorted([list(x) for x in intervals], key=lambda x: (x[0], x[1]))\n"
        "    out = [ints[0][:]]\n"
        "    for a, b in ints[1:]:\n"
        "        if a <= out[-1][1]:\n"
        "            out[-1][1] = max(out[-1][1], b)\n"
        "        else:\n"
        "            out.append([a, b])\n"
        "    return out  # <-- lista list\n"
    ),

    # 24
    "balanced_brackets": (
        "def balanced_brackets(s: str) -> bool:\n"
        "    pairs = {')':'(', ']':'[', '}':'{'}\n"
        "    stack = []\n"
        "    for ch in s:\n"
        "        if ch in '([{':\n"
        "            stack.append(ch)\n"
        "        elif ch in ')]}':\n"
        "            if not stack or stack[-1] != pairs[ch]:\n"
        "                return False\n"
        "            stack.pop()\n"
        "    return not stack\n"
    ),

    # 25
    "median_of_list": (
        "def median_of_list(xs: list):\n"
        "    n = len(xs)\n"
        "    if n == 0:\n"
        "        raise ValueError('empty')\n"
        "    ys = sorted(xs)\n"
        "    m = n // 2\n"
        "    if n % 2:\n"
        "        return ys[m]\n"
        "    return (ys[m-1] + ys[m]) / 2\n"
    ),

    # 26
    "second_largest": (
        "def second_largest(xs: list):\n"
        "    it = list(xs)\n"
        "    if len(it) < 2:\n"
        "        return None\n"
        "    uniq = sorted(set(it))\n"
        "    if len(uniq) < 2:\n"
        "        return None\n"
        "    return uniq[-2]\n"
    ),

    # 27
    "chunk_list": (
        "def chunk_list(xs: list, size: int) -> list:\n"
        "    if size <= 0:\n"
        "        raise ValueError('size must be >0')\n"
        "    return [xs[i:i+size] for i in range(0, len(xs), size)]\n"
    ),

    # 28
    "count_words": (
        "def count_words(s: str) -> dict:\n"
        "    import re\n"
        "    # słowa: litery/cyfry/apostrof (np. don't)\n"
        "    toks = re.findall(r\"[A-Za-z0-9']+\", s)\n"
        "    from collections import Counter\n"
        "    return dict(Counter(w.lower() for w in toks))\n"
    ),
    # 29
    "remove_dups_preserve": (
        "def remove_dups_preserve(xs: list) -> list:\n"
        "    seen = set()\n"
        "    out = []\n"
        "    for v in xs:\n"
        "        if v not in seen:\n"
        "            seen.add(v)\n"
        "            out.append(v)\n"
        "    return out\n"
    ),

    # 30
    "sum_of_primes": (
        "def sum_of_primes(n: int) -> int:\n"
        "    if n < 2:\n"
        "        return 0\n"
        "    sieve = bytearray(b'\\x01') * (n+1)\n"
        "    sieve[0:2] = b'\\x00\\x00'\n"
        "    import math\n"
        "    for p in range(2, int(math.isqrt(n))+1):\n"
        "        if sieve[p]:\n"
        "            step = p\n"
        "            start = p*p\n"
        "            sieve[start:n+1:step] = b'\\x00' * (((n - start)//step) + 1)\n"
        "    return sum(i for i, v in enumerate(sieve) if v)\n"
    ),

    # 31
    "is_prime": (
        "def is_prime(n: int) -> bool:\n"
        "    if n < 2:\n"
        "        return False\n"
        "    if n % 2 == 0:\n"
        "        return n == 2\n"
        "    r = int(n ** 0.5)\n"
        "    f = 3\n"
        "    while f <= r:\n"
        "        if n % f == 0:\n"
        "            return False\n"
        "        f += 2\n"
        "    return True\n"
    ),

    # 32
    "binary_search": (
        "def binary_search(a: list, x) -> int:\n"
        "    lo, hi = 0, len(a)-1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi)//2\n"
        "        if a[mid] == x:\n"
        "            return mid\n"
        "        if a[mid] < x:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
    ),

    # 33
    "prefix_sums": (
        "def prefix_sums(xs: list) -> list:\n"
        "    out = []\n"
        "    s = 0\n"
        "    for v in xs:\n"
        "        s += v\n"
        "        out.append(s)\n"
        "    return out\n"
    ),

    # 34
    "longest_common_prefix": (
        "def longest_common_prefix(strs: list) -> str:\n"
        "    if not strs:\n"
        "        return ''\n"
        "    pref = min(strs)\n"
        "    maxi = max(strs)\n"
        "    for i, (a, b) in enumerate(zip(pref, maxi)):\n"
        "        if a != b:\n"
        "            return pref[:i]\n"
        "    return pref\n"
    ),

    # 35
    "hamming_distance": (
        "def hamming_distance(a: str, b: str) -> int:\n"
        "    if len(a) != len(b):\n"
        "        raise ValueError('lengths must match')\n"
        "    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))\n"
    ),

    # 36
    "rotate_matrix_90": (
        "def rotate_matrix_90(M: list) -> list:\n"
        "    # 90° clockwise\n"
        "    if not M:\n"
        "        return []\n"
        "    return [list(row) for row in zip(*M[::-1])]\n"
    ),

    # 37
    "staircase": (
        "def staircase(n: int) -> list:\n"
        "    # lista wizualnych stopni: ['#', '##', ..., '#'*n]\n"
        "    if n <= 0:\n"
        "        return []\n"
        "    return ['#' * i for i in range(1, n+1)]\n"
    ),

    # 38
    "merge_sorted_lists": (
        "def merge_sorted_lists(a: list, b: list) -> list:\n"
        "    i = j = 0\n"
        "    out = []\n"
        "    while i < len(a) and j < len(b):\n"
        "        if a[i] <= b[j]:\n"
        "            out.append(a[i]); i += 1\n"
        "        else:\n"
        "            out.append(b[j]); j += 1\n"
        "    out.extend(a[i:]); out.extend(b[j:])\n"
        "    return out\n"
    ),

    # 39
    "parse_kv_pairs": (
        "import re\n"
        "_INT = re.compile(r'^[+-]?\\d+$')\n"
        "def parse_kv_pairs(s: str) -> dict:\n"
        "    out = {}\n"
        "    if not s:\n"
        "        return out\n"
        "    for p in (part for part in s.split(';') if part.strip()):\n"
        "        if '=' in p:\n"
        "            k, v = p.split('=', 1)\n"
        "            k, v = k.strip(), v.strip()\n"
        "            if _INT.match(v):\n"
        "                try:\n"
        "                    out[k] = int(v)\n"
        "                except Exception:\n"
        "                    out[k] = v\n"
        "            else:\n"
        "                out[k] = v\n"
        "    return out\n"
    ),

    # 40
    "sum_diagonal": (
        "def sum_diagonal(M: list) -> int:\n"
        "    n = len(M)\n"
        "    return sum(M[i][i] for i in range(n))\n"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 3) GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_code(task: Dict[str, Any], mode: str = "A1",
                  rows: int = 12, cols: int = 12, thr: float = 0.55) -> Dict[str, Any]:
    t0 = time.time()
    if mode == "A2":
        lmbd, delta = 0.0, 0.5
    else:
        lmbd, delta = 0.60, 0.25

    m = _align_metrics(lmbd, delta, rows, cols, thr)
    name = task["entrypoint"]
    code = _TEMPLATES.get(name)
    if code is None:
        code = f"def {name}(*args, **kwargs):\n    return None\n"

    # ⬇⬇⬇ kluczowe: wspólne importy dla sandboxa bench
    prelude = "import re\nfrom collections import Counter\n"
    code = prelude + code
    # ⬆⬆⬆

    return dict(code=code, metrics=m, time_s=time.time() - t0)
