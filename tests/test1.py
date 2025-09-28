# make_glitchlab_scenarios.py
# tworzy glitchlab_scenarios.zip z gotowymi scenariuszami (40 funkcji + impl parse_kv_pairs)

import os, json, time, zipfile

BASE = os.path.abspath(".")
OUTDIR = os.path.join(BASE, "glitchlab_scenarios")
os.makedirs(OUTDIR, exist_ok=True)

README = """# GlitchLab – gotowe scenariusze (A)
Zawartość:
- `templates.py` – kompletny słownik `_TEMPLATES` (40 funkcji) do wklejenia tam, gdzie masz `generate_code(...)`.
- `parse_kv_pairs_impl.py` – implementacja `_impl_parse_kv_pairs()` (spójna z Twoją propozycją).
- `HOWTO.md` – krótka instrukcja podmiany.
- `version.json` – metadane paczki.

## Szybki start
1) Otwórz moduł z `generate_code(...)`.
2) Podmień cały `_TEMPLATES = {...}` zawartością z `templates.py`.
3) (Opcj.) jeśli masz helper `_impl_parse_kv_pairs()`, podmień go wersją z `parse_kv_pairs_impl.py`.
4) Odpal benchmark:
   python -m glitchlab.gui.bench.ab_pilot --tasks "tasks/*.json" --json
"""

HOWTO = """# HOWTO
- **Cel**: wyeliminować `NotImplementedError` w trybie A (A1/A2) i umożliwić realne testy Align/J_phi/czas.
- **Wklej**: zawartość `_TEMPLATES` do modułu z `generate_code(...)`.
- **Zgodność**: implementacje są deterministyczne i proste (pasują do podanych zadań).
"""

TEMPLATES = r'''_TEMPLATES = {
    "reverse_str": """\
def reverse_str(s: str) -> str:
    return s[::-1]
""",
    "fib": """\
def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(int(n)):
        a, b = b, a + b
    return a
""",
    "sum_csv_numbers": """\
def sum_csv_numbers(csv_text: str) -> int:
    parts = [p.strip() for p in (csv_text or "").split(",") if p.strip()]
    return sum(int(p) for p in parts)
""",
    "moving_sum": """\
def moving_sum(x: list, k: int = None) -> list:
    if k is None:
        out = []
        s = 0
        for v in x:
            s += v
            out.append(s)
        return out
    if k <= 0 or len(x) < k:
        return []
    out = []
    s = sum(x[:k])
    out.append(s)
    for i in range(k, len(x)):
        s += x[i] - x[i - k]
        out.append(s)
    return out
""",
    "is_palindrome": """\
import re
def is_palindrome(s: str) -> bool:
    t = re.sub(r"[^A-Za-z0-9]", "", (s or "")).lower()
    return t == t[::-1]
""",
    "count_vowels": """\
def count_vowels(s: str) -> int:
    if not s:
        return 0
    V = set("aeiouAEIOU")
    return sum(1 for ch in s if ch in V)
""",
    "factorial": """\
def factorial(n: int) -> int:
    n = int(n)
    if n < 0:
        raise ValueError("n<0")
    res = 1
    for i in range(2, n+1):
        res *= i
    return res
""",
    "unique_sorted": """\
def unique_sorted(xs: list) -> list:
    return sorted(set(xs))
""",
    "flatten_once": """\
def flatten_once(xs: list) -> list:
    out = []
    for v in xs:
        if isinstance(v, (list, tuple)):
            out.extend(v)
        else:
            out.append(v)
    return out
""",
    "dot_product": """\
def dot_product(a: list, b: list) -> int:
    return sum(x*y for x, y in zip(a, b))
""",
    "anagrams": """\
def anagrams(a: str, b: str) -> bool:
    if a is None or b is None:
        return False
    return sorted(a.replace(" ", "").lower()) == sorted(b.replace(" ", "").lower())
""",
    "gcd": """\
import math
def gcd(a: int, b: int) -> int:
    return math.gcd(int(a), int(b))
""",
    "lcm": """\
import math
def lcm(a: int, b: int) -> int:
    a, b = int(a), int(b)
    return 0 if a == 0 or b == 0 else abs(a*b)//math.gcd(a,b)
""",
    "two_sum": """\
def two_sum(nums: list, target: int):
    seen = {}
    for i, v in enumerate(nums):
        j = seen.get(target - v)
        if j is not None:
            return (j, i)
        seen[v] = i
    return None
""",
    "transpose": """\
def transpose(mat: list) -> list:
    if not mat:
        return []
    return [list(row) for row in zip(*mat)]
""",
    "matmul": """\
def matmul(A: list, B: list) -> list:
    if not A or not B:
        return []
    n, m, p = len(A), len(A[0]), len(B[0])
    res = [[0]*p for _ in range(n)]
    for i in range(n):
        for k in range(m):
            aik = A[i][k]
            if aik == 0:
                continue
            for j in range(p):
                res[i][j] += aik * B[k][j]
    return res
""",
    "to_snake_case": """\
import re
def to_snake_case(s):
    if isinstance(s, (list, tuple)):
        s = " ".join(str(t) for t in s)
    s = s or ""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\\1_\\2", s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = s.strip("_").lower()
    s = re.sub(r"_+", "_", s)
    return s
""",
    "to_camel_case": """\
import re
def to_camel_case(x):
    if isinstance(x, (list, tuple)):
        tokens = [str(t) for t in x]
    else:
        x = x or ""
        tokens = re.split(r"[^A-Za-z0-9]+|_", x)
    tokens = [t for t in tokens if t]
    if not tokens:
        return ""
    head = tokens[0].lower()
    tail = [t[:1].upper() + t[1:].lower() for t in tokens[1:]]
    return head + "".join(tail)
""",
    "rle_compress": """\
def rle_compress(s: str) -> str:
    if not s:
        return ""
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
    return "".join(out)
""",
    "rle_decompress": """\
import re
def rle_decompress(s: str) -> str:
    if not s:
        return ""
    out = []
    for m in re.finditer(r"(.)((?:\\d)+)", s):
        ch = m.group(1)
        n = int(m.group(2))
        out.append(ch * n)
    return "".join(out)
""",
    "rotate_list": """\
def rotate_list(xs: list, k: int) -> list:
    n = len(xs)
    if n == 0:
        return []
    k %= n
    return xs[-k:] + xs[:-k] if k else xs[:]
""",
    "most_common_char": """\
from collections import Counter
def most_common_char(s: str):
    if not s:
        return None
    cnt = Counter(s)
    best = None
    best_n = -1
    seen = set()
    for ch in s:
        if ch in seen:
            continue
        seen.add(ch)
        n = cnt[ch]
        if n > best_n:
            best = ch
            best_n = n
    return best
""",
    "merge_intervals": """\
def merge_intervals(intervals: list) -> list:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    out = [intervals[0][:]]
    for s, e in intervals[1:]:
        ls, le = out[-1]
        if s <= le:
            out[-1][1] = max(le, e)
        else:
            out.append([s, e])
    return out
""",
    "balanced_brackets": """\
def balanced_brackets(s: str) -> bool:
    pairs = {')':'(', ']':'[', '}':'{'}
    st = []
    for ch in s or "":
        if ch in "([{":
            st.append(ch)
        elif ch in ")]}":
            if not st or st[-1] != pairs[ch]:
                return False
            st.pop()
    return not st
""",
    "median_of_list": """\
def median_of_list(xs: list):
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    m = n // 2
    if n % 2 == 1:
        return ys[m]
    return (ys[m-1] + ys[m]) / 2
""",
    "second_largest": """\
def second_largest(xs: list):
    uniq = sorted(set(xs))
    return None if len(uniq) < 2 else uniq[-2]
""",
    "chunk_list": """\
def chunk_list(xs: list, size: int) -> list:
    if size <= 0:
        return []
    return [xs[i:i+size] for i in range(0, len(xs), size)]
""",
    "count_words": """\
import re
def count_words(s: str) -> dict:
    if not s:
        return {}
    words = re.findall(r"[A-Za-z0-9]+", s.lower())
    out = {}
    for w in words:
        out[w] = out.get(w, 0) + 1
    return out
""",
    "remove_dups_preserve": """\
def remove_dups_preserve(xs: list) -> list:
    seen = set()
    out = []
    for v in xs:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out
""",
    "sum_of_primes": """\
def _is_prime_p(n: int) -> bool:
    if n < 2: return False
    if n % 2 == 0: return n == 2
    d = 3
    while d*d <= n:
        if n % d == 0:
            return False
        d += 2
    return True
def sum_of_primes(n: int) -> int:
    n = int(n)
    return sum(i for i in range(2, n+1) if _is_prime_p(i))
""",
    "is_prime": """\
def is_prime(n: int) -> bool:
    n = int(n)
    if n < 2: return False
    if n % 2 == 0: return n == 2
    d = 3
    while d*d <= n:
        if n % d == 0:
            return False
        d += 2
    return True
""",
    "binary_search": """\
def binary_search(arr: list, target) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""",
    "prefix_sums": """\
def prefix_sums(xs: list) -> list:
    out = []
    s = 0
    for v in xs:
        s += v
        out.append(s)
    return out
""",
    "longest_common_prefix": """\
def longest_common_prefix(strs: list) -> str:
    if not strs:
        return ""
    s = min(strs)
    e = max(strs)
    i = 0
    while i < len(s) and i < len(e) and s[i] == e[i]:
        i += 1
    return s[:i]
""",
    "hamming_distance": """\
def hamming_distance(a: str, b: str) -> int:
    if a is None or b is None or len(a) != len(b):
        raise ValueError("strings must be same length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))
""",
    "rotate_matrix_90": """\
def rotate_matrix_90(mat: list) -> list:
    if not mat:
        return []
    return [list(row)[::-1] for row in zip(*mat)]
""",
    "staircase": """\
def staircase(n: int) -> list:
    n = int(n)
    if n <= 0: return []
    return ["#" * i for i in range(1, n+1)]
""",
    "merge_sorted_lists": """\
def merge_sorted_lists(a: list, b: list) -> list:
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:]); out.extend(b[j:])
    return out
""",
    "parse_kv_pairs": r"""\
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
""",
    "sum_diagonal": """\
def sum_diagonal(mat: list) -> int:
    n = min(len(mat), len(mat[0]) if mat else 0)
    return sum(mat[i][i] for i in range(n))
""",
}'''

IMPL = r'''def _impl_parse_kv_pairs() -> str:
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
"""'''

open(os.path.join(OUTDIR, "README.md"), "w", encoding="utf-8").write(README)
open(os.path.join(OUTDIR, "HOWTO.md"), "w", encoding="utf-8").write(HOWTO)
open(os.path.join(OUTDIR, "templates.py"), "w", encoding="utf-8").write(TEMPLATES)
open(os.path.join(OUTDIR, "parse_kv_pairs_impl.py"), "w", encoding="utf-8").write(IMPL)
open(os.path.join(OUTDIR, "version.json"), "w", encoding="utf-8").write(
    json.dumps({"package":"glitchlab_scenarios","created_ts":int(time.time()),
                "files":["templates.py","parse_kv_pairs_impl.py","README.md","HOWTO.md","version.json"]}, indent=2)
)

ZIP = os.path.join(BASE, "glitchlab_scenarios.zip")
with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for name in os.listdir(OUTDIR):
        z.write(os.path.join(OUTDIR, name), arcname=f"glitchlab_scenarios/{name}")

print(f"OK -> {ZIP}")
