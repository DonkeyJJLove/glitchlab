import importlib

pathlib = importlib.import_module("pathlib")
subprocess = importlib.import_module("subprocess")
pytest = importlib.import_module("pytest")
WORKFLOW = pathlib.Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
CONFIG = pathlib.Path("pyproject.toml").read_text(encoding="utf-8")


def _contract_ok(text=WORKFLOW, config=CONFIG):
    required = (
        "ref: ${{ github.event.pull_request.head.sha || github.sha }}",
        'git merge-base "${{ github.event.pull_request.base.sha }}" "$HEAD_SHA"',
        "--select E9,F63,F7,F82",
        '[ "$BASE_RC" -le 1 ] && [ "$HEAD_RC" -le 1 ] || exit 2',
        '[ "$head_bad" -gt "$base_bad" ]',
        "include-hidden-files: true",
    )
    gates = (
        "Ruff changed Python files (no-regression ratchet)",
        "Black (format no-regression ratchet)",
        "Mypy (type check)",
        "Pytest",
        "Doclint (docs consistency)",
        "Delta fingerprint (DIFF-first)",
        "Invariants check (I1–I4 gates)",
    )
    return (
        all(token in text for token in required)
        and text.count("if-no-files-found: error") >= 2
        and '"F821"' not in config
        and all(f"- name: {gate}\n        if: always()" in text for gate in gates)
    )


@pytest.mark.parametrize(
    "needle,replacement",
    [
        ("ref: ${{ github.event.pull_request.head.sha || github.sha }}", "ref: master"),
        ('git merge-base "${{ github.event.pull_request.base.sha }}" "$HEAD_SHA"', "echo base"),
        ("--select E9,F63,F7,F82", "--select E9,F63,F7"),
        ('[ "$BASE_RC" -le 1 ] && [ "$HEAD_RC" -le 1 ] || exit 2', "true"),
        ('[ "$head_bad" -gt "$base_bad" ]', "false"),
        ("include-hidden-files: true", "include-hidden-files: false"),
        ("if-no-files-found: error", "if-no-files-found: ignore"),
        ("- name: Pytest\n        if: always()", "- name: Pytest"),
    ],
)
def test_ci_contract_detects_weakening(needle, replacement):
    assert needle in WORKFLOW
    mutated = WORKFLOW.replace(needle, replacement, 1)
    assert _contract_ok(mutated, CONFIG) is False


def test_ci_contract_detects_f821_suppression():
    mutated = CONFIG + '\nignore = ["F821"]\n'
    assert _contract_ok(WORKFLOW, mutated) is False


def _git(repo, *args):
    command = ["git", "-C", str(repo), *args]
    return subprocess.check_output(command, text=True).strip()


def _commit(repo, message):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _changed_python(repo, base, head):
    out = _git(
        repo,
        "diff",
        "--name-only",
        "--diff-filter=ACMRT",
        base,
        head,
        "--",
        "*.py",
    )
    return {line for line in out.splitlines() if line}


def test_python_path_selection_add_modify_rename_delete(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "rccm@example.invalid")
    _git(repo, "config", "user.name", "RCCM")
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "note.txt").write_text("base\n", encoding="utf-8")
    base = _commit(repo, "base")

    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "b.py").write_text("y = 1\n", encoding="utf-8")
    (repo / "note.txt").write_text("changed\n", encoding="utf-8")
    changed = _commit(repo, "modify-add")
    assert _changed_python(repo, base, changed) == {"a.py", "b.py"}

    _git(repo, "mv", "a.py", "c.py")
    renamed = _commit(repo, "rename")
    rename_paths = _changed_python(repo, changed, renamed)
    assert "c.py" in rename_paths
    assert all(path.endswith(".py") for path in rename_paths)

    (repo / "c.py").unlink()
    deleted = _commit(repo, "delete")
    assert _changed_python(repo, renamed, deleted) == set()

    (repo / "note.txt").write_text("only text\n", encoding="utf-8")
    non_python = _commit(repo, "non-python")
    assert _changed_python(repo, deleted, non_python) == set()
