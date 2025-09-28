import random
from glitchlab.gui.bench.templates import get_template


def test_reverse_fuzz():
    code = get_template('reverse_str')
    ns = {};
    exec(code, ns, ns)
    f = ns['reverse_str']
    for _ in range(100):
        s = ''.join(random.choice('abc #$123XYZ') for _ in range(20))
        assert f(f(s)) == s
