# tests/integration/test_with_real_git.py
import subprocess
from pathlib import Path
import tempfile
import os
import importlib
import sys


def run(cmd, cwd):
    subprocess.check_call(cmd, cwd=cwd, shell=False)


def test_end_to_end_with_git(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run(["git", "init"], cwd=repo)
    (repo / "glitchlab").mkdir()
    (repo / "glitchlab" / "core").mkdir(parents=True)
    (repo / "glitchlab" / "core" / "foo.py").write_text("def foo():\n    return 1\n")
    run(["git", "add", "."], cwd=repo)
    run(["git", "commit", "-m", "initial"], cwd=repo)

    # zmiana
    (repo / "glitchlab" / "core" / "foo.py").write_text("import os\n\ndef foo():\n    return 2\n")
    run(["git", "add", "."], cwd=repo)
    run(["git", "commit", "-m", "changed"], cwd=repo)

    # Uruchom moduł jako CLI (dostosuj ścieżkę do pythona)
    # albo importuj i wywołaj main() (ale pamiętaj o sys.argv)
    sys.path.insert(0, str(repo))
    df = importlib.import_module("glx.tools.delta_fingerprint")
    # zabezpiecz sys.argv jeśli potrzebne
    old_argv = sys.argv[:]
    sys.argv = ["delta_fingerprint", "--range", "HEAD~1..HEAD"]
    try:
        ret = df.main()
    finally:
        sys.argv = old_argv

    assert ret == 0
    assert (repo / "glitchlab" / ".glx" / "delta_report.json").exists()
