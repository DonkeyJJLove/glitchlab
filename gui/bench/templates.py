from __future__ import annotations
import re
from collections import Counter
import ast

_TEMPLATES = {
    "reverse_str": "def reverse_str(s: str) -> str:\n    return s[::-1]\n",
    "fib": "def fib(n: int) -> int:\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a\n",
    "sum_csv_numbers": "def sum_csv_numbers(csv_text: str) -> int:\n    parts=[p.strip() for p in csv_text.split(',') if p.strip()]\n    return sum(int(p) for p in parts)\n",
    "moving_sum": "def moving_sum(x: list, k: int) -> list:\n    if k<=0 or len(x)<k: return []\n    out=[]\n    for i in range(len(x)-k+1):\n        out.append(sum(x[i:i+k]))\n    return out\n",
    "to_camel_case": "import re\n\n"
                     "def to_camel_case(s: str) -> str:\n"
                     "    parts=re.split(r'[^a-zA-Z0-9]+', s.strip())\n"
                     "    parts=[p for p in parts if p]\n"
                     "    return parts[0].lower()+''.join(p.capitalize() for p in parts[1:]) if parts else ''\n",
    "to_snake_case": "import re\n\n"
                     "def to_snake_case(s: str) -> str:\n"
                     "    s = re.sub(r'([a-z0-9])([A-Z])', r'\\1_\\2', s).lower()\n"
                     "    s = re.sub(r'[^a-z0-9]+', '_', s)\n"
                     "    return s.strip('_')\n",
    "anagrams": "from collections import Counter\n\n"
                "def anagrams(a: str, b: str) -> bool:\n"
                "    return Counter(a.replace(' ', '').lower())==Counter(b.replace(' ', '').lower())\n",
    "count_words": "import re\nfrom collections import Counter\n\n"
                   "def count_words(s: str) -> dict:\n"
                   "    toks=[t.lower() for t in re.findall(r'[a-zA-Z]+', s)]\n"
                   "    return dict(Counter(toks))\n",
    "merge_intervals": "def merge_intervals(iv):\n"
                       "    iv=sorted([list(x) for x in iv])\n"
                       "    out=[]\n"
                       "    for s,e in iv:\n"
                       "        if not out or s>out[-1][1]: out.append([s,e])\n"
                       "        else: out[-1][1]=max(out[-1][1], e)\n"
                       "    return out\n",
    "two_sum": "def two_sum(nums, target):\n"
               "    seen={}\n"
               "    for i,x in enumerate(nums):\n"
               "        if target-x in seen: return [seen[target-x], i]\n"
               "        seen[x]=i\n"
               "    return None\n",
    "staircase": "def staircase(n: int):\n    return ['#'*(i+1) for i in range(max(0,n))]\n",
    "count_calls": "import ast\n\n"
                   "def count_calls(src: str) -> int:\n"
                   "    t=ast.parse(src)\n"
                   "    return sum(isinstance(n, ast.Call) for n in ast.walk(t))\n",
}


def get_template(name: str) -> str | None:
    return _TEMPLATES.get(name)
