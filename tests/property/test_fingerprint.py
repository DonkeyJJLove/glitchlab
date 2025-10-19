# tests/property/test_fingerprint.py
from hypothesis import given, strategies as st
import importlib

df = importlib.import_module("glx.tools.delta_fingerprint")


@given(st.text(), st.text())
def test_fingerprint_deterministic(a, b):
    # zakładamy funkcję fingerprint_from_texts
    h1 = df.fingerprint_from_texts(a, b)
    h2 = df.fingerprint_from_texts(a, b)
    assert h1 == h2
