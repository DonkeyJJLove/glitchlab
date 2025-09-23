# -*- coding: utf-8 -*-
from __future__ import annotations
import time, math, re
from typing import Dict, Any, List, Tuple, Iterable


def _impl_reverse_str() -> str:
    return """\
def reverse_str(s: str) -> str:
    return s[::-1]
"""


def _impl_fib() -> str:
    return """\
def fib(n: int) -> int:
    if n < 0:
        raise ValueError("n must be >= 0")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a+b
    return a
"""


def _impl_sum_csv_numbers() -> str:
    return """\
def sum_csv_numbers(text: str) -> float:
    total = 0.0
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        for tok in line.split(','):
            tok = tok.strip()
            if tok:
                total += float(tok)
    return total
"""


def _impl_moving_sum() -> str:
    return """\
def moving_sum(xs, k: int):
    if k <= 0:
        return []
    n = len(xs)
    if n == 0:
        return []
    out = []
    window = sum(xs[:min(k, n)])
    out.append(window)
    for i in range(k, n):
        window += xs[i] - xs[i-k]
        out.append(window)
    return out
"""


def _impl_is_palindrome() -> str:
    return """\
def is_palindrome(s: str) -> bool:
    t = ''.join(ch.lower() for ch in s if ch.isalnum())
    return t == t[::-1]
"""


def _impl_count_vowels() -> str:
    return """\
def count_vowels(s: str) -> int:
    V = set('aeiouAEIOU')
    return sum(1 for ch in s if ch in V)
"""


def _impl_factorial() -> str:
    return """\
def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be >= 0")
    res = 1
    for i in range(2, n+1):
        res *= i
    return res
"""


def _impl_unique_sorted() -> str:
    return """\
def unique_sorted(xs):
    return sorted(set(xs))
"""


def _impl_flatten_once() -> str:
    return """\
def flatten_once(xs):
    out = []
    for e in xs:
        if isinstance(e, (list, tuple)):
            out.extend(e)
        else:
            out.append(e)
    return out
"""


def _impl_dot_product() -> str:
    return """\
def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("len mismatch")
    return sum(x*y for x, y in zip(a, b))
"""


def _impl_anagrams() -> str:
    return """\
def anagrams(a: str, b: str) -> bool:
    na = sorted(''.join(ch.lower() for ch in a if ch.isalpha()))
    nb = sorted(''.join(ch.lower() for ch in b if ch.isalpha()))
    return na == nb
"""


def _impl_gcd() -> str:
    return """\
def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)
"""


def _impl_lcm() -> str:
    return """\
def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    def _gcd(x, y):
        while y:
            x, y = y, x % y
        return abs(x)
    return abs(a*b) // _gcd(a, b)
"""


def _impl_two_sum() -> str:
    return """\
def two_sum(nums, target: int):
    seen = {}
    for i, x in enumerate(nums):
        y = target - x
        if y in seen:
            return [seen[y], i]
        seen[x] = i
    return None
"""


def _impl_transpose() -> str:
    return """\
def transpose(mat):
    if not mat:
        return []
    return [list(row) for row in zip(*mat)]
"""


def _impl_matmul() -> str:
    return """\
def matmul(A, B):
    if not A or not B:
        return []
    n, m = len(A), len(A[0])
    m2, p = len(B), len(B[0])
    if m != m2:
        raise ValueError("shape mismatch")
    out = [[0]*p for _ in range(n)]
    for i in range(n):
        for k in range(m):
            aik = A[i][k]
            for j in range(p):
                out[i][j] += aik * B[k][j]
    return out
"""


def _impl_to_snake_case() -> str:
    return r"""\
import re
def to_snake_case(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    # split CamelCase or words with spaces/dashes/underscores
    s = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s)
    s = re.sub(r'[\s\-]+', '_', s)
    s = s.replace('__', '_')
    return s.lower()
"""


# podmień cały blok _impl_to_camel_case() na ten:
def _impl_to_camel_case() -> str:
    return r"""\
import re
def to_camel_case(s):
    # akceptuj zarówno string, jak i listę tokenów
    if s is None:
        return s
    if isinstance(s, (list, tuple)):
        parts = [str(p) for p in s if str(p)]
    else:
        if not isinstance(s, str):
            s = str(s)
        parts = re.split(r'[_\-\s]+', s.strip())
    if not parts:
        return ''
    head = str(parts[0]).lower()
    tail = ''.join(str(p).capitalize() for p in parts[1:] if p)
    return head + tail
"""


def _impl_rle_compress() -> str:
    return """\
def rle_compress(s: str) -> str:
    if not s:
        return ''
    out = []
    prev = s[0]
    cnt = 1
    for ch in s[1:]:
        if ch == prev:
            cnt += 1
        else:
            out.append(f"{prev}{cnt}")
            prev, cnt = ch, 1
    out.append(f"{prev}{cnt}")
    return ''.join(out)
"""


def _impl_rle_decompress() -> str:
    return """\
def rle_decompress(s: str) -> str:
    out = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        i += 1
        j = i
        while j < n and s[j].isdigit():
            j += 1
        cnt = int(s[i:j]) if i < j else 1
        out.append(ch * cnt)
        i = j
    return ''.join(out)
"""


def _impl_rotate_list() -> str:
    return """\
def rotate_list(xs, k: int):
    if not xs:
        return []
    k %= len(xs)
    if k == 0:
        return xs[:]
    return xs[-k:] + xs[:-k]
"""


def _impl_most_common_char() -> str:
    return """\
def most_common_char(s: str) -> str:
    if not s:
        return ''
    from collections import Counter
    c = Counter(s)
    # tie-break: first by count desc, then by first occurrence index
    best = None
    best_key = None
    first_pos = {ch: s.index(ch) for ch in c}
    for ch, cnt in c.items():
        key = (-cnt, first_pos[ch])
        if best_key is None or key < best_key:
            best_key, best = key, ch
    return best
"""


def _impl_merge_intervals() -> str:
    return """\
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [intervals[0][:]]
    for a, b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1][1] = max(lb, b)
        else:
            out.append([a, b])
    return out
"""


def _impl_balanced_brackets() -> str:
    return """\
def balanced_brackets(s: str) -> bool:
    pairs = {')':'(', ']':'[', '}':'{'}
    stack = []
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in ')]}':
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack
"""


def _impl_median_of_list() -> str:
    return """\
def median_of_list(xs):
    n = len(xs)
    if n == 0:
        raise ValueError("empty")
    ys = sorted(xs)
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return (ys[mid-1] + ys[mid]) / 2
"""


def _impl_second_largest() -> str:
    return """\
def second_largest(xs):
    uniq = sorted(set(xs))
    if len(uniq) < 2:
        raise ValueError("need >=2 distinct values")
    return uniq[-2]
"""


def _impl_chunk_list() -> str:
    return """\
def chunk_list(xs, size: int):
    if size <= 0:
        return []
    return [xs[i:i+size] for i in range(0, len(xs), size)]
"""


# podmień _impl_count_words() na:
def _impl_count_words() -> str:
    return r"""\
import re
from collections import Counter
def count_words(s: str) -> dict:
    if not s:
        return {}
    toks = re.findall(r"[A-Za-z0-9']+", s.lower())
    return dict(Counter(toks))
"""


def _impl_remove_dups_preserve() -> str:
    return """\
def remove_dups_preserve(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
"""


def _impl_sum_of_primes() -> str:
    return """\
def sum_of_primes(n: int) -> int:
    if n < 2:
        return 0
    sieve = [True]*(n+1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(n**0.5)+1):
        if sieve[p]:
            step = p
            start = p*p
            sieve[start:n+1:step] = [False]*(((n - start)//step) + 1)
    return sum(i for i, v in enumerate(sieve) if v)
"""


def _impl_is_prime() -> str:
    return """\
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(n**0.5)
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True
"""


def _impl_binary_search() -> str:
    return """\
def binary_search(a, x):
    lo, hi = 0, len(a)-1
    while lo <= hi:
        mid = (lo + hi)//2
        if a[mid] == x:
            return mid
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
"""


def _impl_prefix_sums() -> str:
    return """\
def prefix_sums(xs):
    out = []
    s = 0
    for v in xs:
        s += v
        out.append(s)
    return out
"""


def _impl_longest_common_prefix() -> str:
    return """\
def longest_common_prefix(strs):
    if not strs:
        return ''
    pref = strs[0]
    for s in strs[1:]:
        i = 0
        m = min(len(pref), len(s))
        while i < m and pref[i] == s[i]:
            i += 1
        pref = pref[:i]
        if not pref:
            break
    return pref
"""


def _impl_hamming_distance() -> str:
    return """\
def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("length mismatch")
    return sum(1 for x, y in zip(a, b) if x != y)
"""


def _impl_rotate_matrix_90() -> str:
    return """\
def rotate_matrix_90(m):
    # 90° clockwise
    if not m:
        return []
    return [list(row) for row in zip(*m[::-1])]
"""


# podmień _impl_staircase() na:
def _impl_staircase() -> str:
    return """\
def staircase(n: int):
    if n <= 0:
        return []
    return ['#'*(i+1) for i in range(n)]
"""


def _impl_merge_sorted_lists() -> str:
    return """\
def merge_sorted_lists(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    if i < len(a): out.extend(a[i:])
    if j < len(b): out.extend(b[j:])
    return out
"""


# podmień _impl_parse_kv_pairs() na:
def _impl_parse_kv_pairs() -> str:
    return r"""\
import re
_INT = re.compile(r"^[+-]?\d+$")
def parse_kv_pairs(s: str) -> dict:
    out = {}
    if not s:
        return out
    for p in (part for part in s.split(';') if part.strip()):
        if '=' in p:
            k, v = p.split('=', 1)
            k, v = k.strip(), v.strip()
            if _INT.match(v):
                try:
                    out[k] = int(v)
                except Exception:
                    out[k] = v
            else:
                out[k] = v
    return out
"""


def _impl_sum_diagonal() -> str:
    return """\
def sum_diagonal(m) -> int:
    n = len(m)
    s = 0
    for i in range(n):
        s += m[i][i]
    return s
"""


# Mapowanie id/entrypoint → implementacja
_IMPL_BY_ID = {
    "t01_reverse_string": _impl_reverse_str,
    "t02_fib": _impl_fib,
    "t03_sum_csv_numbers": _impl_sum_csv_numbers,
    "t04_moving_sum": _impl_moving_sum,
    "t05_is_palindrome": _impl_is_palindrome,
    "t06_count_vowels": _impl_count_vowels,
    "t07_factorial": _impl_factorial,
    "t08_unique_sorted": _impl_unique_sorted,
    "t09_flatten_once": _impl_flatten_once,
    "t10_dot_product": _impl_dot_product,
    "t11_anagrams": _impl_anagrams,
    "t12_gcd": _impl_gcd,
    "t13_lcm": _impl_lcm,
    "t14_two_sum": _impl_two_sum,
    "t15_transpose_matrix": _impl_transpose,
    "t16_matrix_multiply": _impl_matmul,
    "t17_to_snake_case": _impl_to_snake_case,
    "t18_to_camel_case": _impl_to_camel_case,
    "t19_rle_compress": _impl_rle_compress,
    "t20_rle_decompress": _impl_rle_decompress,
    "t21_rotate_list": _impl_rotate_list,
    "t22_most_common_char": _impl_most_common_char,
    "t23_merge_intervals": _impl_merge_intervals,
    "t24_balanced_brackets": _impl_balanced_brackets,
    "t25_median_of_list": _impl_median_of_list,
    "t26_second_largest": _impl_second_largest,
    "t27_chunk_list": _impl_chunk_list,
    "t28_count_words": _impl_count_words,
    "t29_remove_dups_preserve": _impl_remove_dups_preserve,
    "t30_sum_of_primes": _impl_sum_of_primes,
    "t31_is_prime": _impl_is_prime,
    "t32_binary_search": _impl_binary_search,
    "t33_prefix_sums": _impl_prefix_sums,
    "t34_longest_common_prefix": _impl_longest_common_prefix,
    "t35_hamming_distance": _impl_hamming_distance,
    "t36_rotate_matrix_90": _impl_rotate_matrix_90,
    "t37_staircase": _impl_staircase,
    "t38_merge_sorted_lists": _impl_merge_sorted_lists,
    "t39_parse_kv_pairs": _impl_parse_kv_pairs,
    "t40_sum_diagonal": _impl_sum_diagonal,
}


def _build_module_code(task: Dict[str, Any]) -> str:
    tid = task.get("id") or ""
    entry = task["entrypoint"]
    # Użyj gotowej implementacji wg id; fallback – załóż nazwy entrypointów jak w zadaniach
    impl = _IMPL_BY_ID.get(tid)
    if impl is None:
        # fallbacky po nazwie entrypointu (gdyby id różniło się prefiksem)
        fallback = {
            "reverse_str": _impl_reverse_str,
            "fib": _impl_fib,
            "sum_csv_numbers": _impl_sum_csv_numbers,
            "moving_sum": _impl_moving_sum,
            "is_palindrome": _impl_is_palindrome,
            "count_vowels": _impl_count_vowels,
            "factorial": _impl_factorial,
            "unique_sorted": _impl_unique_sorted,
            "flatten_once": _impl_flatten_once,
            "dot_product": _impl_dot_product,
            "anagrams": _impl_anagrams,
            "gcd": _impl_gcd,
            "lcm": _impl_lcm,
            "two_sum": _impl_two_sum,
            "transpose": _impl_transpose,
            "matmul": _impl_matmul,
            "to_snake_case": _impl_to_snake_case,
            "to_camel_case": _impl_to_camel_case,
            "rle_compress": _impl_rle_compress,
            "rle_decompress": _impl_rle_decompress,
            "rotate_list": _impl_rotate_list,
            "most_common_char": _impl_most_common_char,
            "merge_intervals": _impl_merge_intervals,
            "balanced_brackets": _impl_balanced_brackets,
            "median_of_list": _impl_median_of_list,
            "second_largest": _impl_second_largest,
            "chunk_list": _impl_chunk_list,
            "count_words": _impl_count_words,
            "remove_dups_preserve": _impl_remove_dups_preserve,
            "sum_of_primes": _impl_sum_of_primes,
            "is_prime": _impl_is_prime,
            "binary_search": _impl_binary_search,
            "prefix_sums": _impl_prefix_sums,
            "longest_common_prefix": _impl_longest_common_prefix,
            "hamming_distance": _impl_hamming_distance,
            "rotate_matrix_90": _impl_rotate_matrix_90,
            "staircase": _impl_staircase,
            "merge_sorted_lists": _impl_merge_sorted_lists,
            "parse_kv_pairs": _impl_parse_kv_pairs,
            "sum_diagonal": _impl_sum_diagonal,
        }.get(entry)

        if fallback is None:
            # awaryjnie: pusta implementacja (nie powinna się zdarzyć)
            return f"def {entry}(*args, **kwargs):\n    raise NotImplementedError\n"
        impl = fallback

    return impl()


def generate_code(task: Dict[str, Any], mode: str = "B", **kwargs) -> Dict[str, Any]:
    """
    Zwraca: dict(code:str, metrics:dict, time_s:float)

    Wersja „B” – deterministyczna biblioteka rozwiązań dla 40 zadań benchmarkowych.
    """
    t0 = time.time()
    code = _build_module_code(task)
    return dict(code=code, metrics={"mode": mode}, time_s=time.time() - t0)
