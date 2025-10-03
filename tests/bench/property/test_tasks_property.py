
from bench import get_template
def test_generated_code_compiles_smoke():
    for key in ['reverse_str','fib','sum_csv_numbers']:
        code = get_template(key)
        ns = {}
        exec(code, ns, ns)
        assert callable(ns[key])
