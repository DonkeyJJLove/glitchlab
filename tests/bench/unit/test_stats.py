
from bench import stats as st

def test_cliffs_delta_bounds():
    assert -1.0 <= st.cliffs_delta([1],[2]) <= 1.0

def test_sign_test_ties():
    assert st.binomial_sign_test_p(0,0) == 1.0
