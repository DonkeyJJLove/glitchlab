# tests/conftest.py
import pytest
from pathlib import Path


@pytest.fixture
def simple_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "glitchlab" / ".glx").mkdir(parents=True)
    return repo
