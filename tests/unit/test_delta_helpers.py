# tests/unit/test_delta_helpers.py
import importlib
import pytest

# załaduj moduł (dostosuj ścieżkę jeśli używasz glitchlab.glx)
df = importlib.import_module("glx.tools.delta_fingerprint")


def test_extract_tokens_from_text_simple():
    text_old = "def foo(a,b):\n  return a+b\n"
    text_new = "import os\n\ndef foo(a,b,c):\n  return a+b+c\n"
    # W module powinna być czysta funkcja, np. extract_delta_tokens(old, new)
    # jeśli nazwa różni się, dostosuj
    tokens = df.extract_delta_tokens(text_old, text_new)
    assert isinstance(tokens, dict)
    assert tokens.get("ΔIMPORT", 0) >= 1
    assert tokens.get("MODIFY_SIG", 0) >= 1
