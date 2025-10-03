
import time
from bench import get_template

def test_perf_reverse_under_10ms():
    ns={}; exec(get_template('reverse_str'), ns, ns)
    f = ns['reverse_str']
    s='x'*10000
    t0=time.perf_counter(); f(s); dt=time.perf_counter()-t0
    assert dt < 0.01
