# tests/conftest.py
import pytest


@pytest.fixture
def simple_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "glitchlab" / ".glx").mkdir(parents=True)
    return repo


@pytest.fixture(autouse=True)
def _git_identity_env(monkeypatch):
    """
    Zapewnia tożsamość autora commita dla testów uruchamianych w czystym CI.
    """
    monkeypatch.setenv("GIT_AUTHOR_NAME", "glitchlab-ci")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "glitchlab-ci@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "glitchlab-ci")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "glitchlab-ci@example.com")
