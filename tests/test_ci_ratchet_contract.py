with open(".github/workflows/ci.yml", encoding="utf-8") as stream:
    TEXT = stream.read()


def test_ci_uses_exact_pr_head_and_merge_base() -> None:
    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in TEXT
    assert 'git merge-base "${{ github.event.pull_request.base.sha }}" "$HEAD_SHA"' in TEXT
    assert 'echo "DIFF_BASE=${BASE_SHA}" >> "$GITHUB_ENV"' in TEXT
    assert 'echo "DIFF_HEAD=${HEAD_SHA}" >> "$GITHUB_ENV"' in TEXT


def test_ratchet_is_fail_closed_and_compares_baseline_to_head() -> None:
    assert "Verify tracked workspace is clean" in TEXT
    assert "git status --porcelain --untracked-files=no" in TEXT
    assert "Ruff changed Python files (no-regression ratchet)" in TEXT
    assert "--output-format json > /tmp/ruff-base.json" in TEXT
    assert "--output-format json > /tmp/ruff-head.json" in TEXT
    assert '[ "$BASE_RC" -le 1 ] && [ "$HEAD_RC" -le 1 ] || exit 2' in TEXT
    assert "n > before[key]" in TEXT


def test_black_is_no_regression_ratchet() -> None:
    assert "Black (format no-regression ratchet)" in TEXT
    assert 'black --check "$BASE_DIR/$file"' in TEXT
    assert 'black --check "$file"' in TEXT
    assert '[ "$head_bad" -gt "$base_bad" ]' in TEXT


def test_mandatory_gates_run_after_earlier_failures() -> None:
    for gate in (
        "Ruff changed Python files (no-regression ratchet)",
        "Black (format no-regression ratchet)",
        "Mypy (type check)",
        "Pytest",
        "Delta fingerprint (DIFF-first)",
        "Invariants check (I1–I4 gates)",
    ):
        assert f"- name: {gate}\n        if: always()" in TEXT
