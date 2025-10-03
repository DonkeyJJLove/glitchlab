
from bench import get_template
def test_has_basic_templates():
    for name in ['reverse_str','fib','sum_csv_numbers','to_snake_case','to_camel_case','anagrams','count_words','merge_intervals','two_sum']:
        assert get_template(name) is not None
