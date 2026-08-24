#!/usr/bin/env python3
"""Create a read-only, evidence-bound Ruff policy inventory.

The script never invokes Ruff with ``--fix`` and never edits tracked source. Ruff
exit code 1 means findings were observed and is recorded rather than treated as
an inventory failure. Exit codes greater than 1 remain fail-closed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import subprocess
import sys
import tomllib
from typing import Any, Iterable


EXPECTED_RUFF_VERSION = "ruff 0.16.4"
def _scan_definitions(current_ignores: list[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    # Ruff's CLI --select takes precedence over file-level rule selection. Repeat
    # the current ignore universe on the CLI so this inventory has unambiguous,
    # replayable semantics. The empty-ignore universe uses the requested inline
    # TOML override explicitly.
    return (
        (
            "configured-policy",
            (
                "ruff",
                "check",
                ".",
                "--force-exclude",
                "--no-cache",
                "--output-format",
                "json",
            ),
        ),
        (
            "all-current-ignores",
            (
                "ruff",
                "check",
                ".",
                "--force-exclude",
                "--no-cache",
                "--select",
                "ALL",
                "--ignore",
                ",".join(current_ignores),
                "--output-format",
                "json",
            ),
        ),
        (
            "all-ignore-empty",
            (
                "ruff",
                "check",
                ".",
                "--force-exclude",
                "--no-cache",
                "--select",
                "ALL",
                "--config",
                "lint.ignore = []",
                "--output-format",
                "json",
            ),
        ),
    )


def _run(argv: Iterable[str], cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _relative_filename(filename: str, root: pathlib.Path) -> str:
    path = pathlib.Path(filename)
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _fixability(row: dict[str, Any]) -> str:
    fix = row.get("fix")
    if not isinstance(fix, dict):
        return "unavailable"
    applicability = fix.get("applicability")
    if isinstance(applicability, str) and applicability:
        return applicability.lower()
    return "available-unspecified"


def _counter_rows(counter: collections.Counter[Any], labels: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, count in sorted(counter.items()):
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(labels, values))
        row["count"] = count
        rows.append(row)
    return rows


def summarize(findings: list[dict[str, Any]], root: pathlib.Path) -> dict[str, Any]:
    by_file: collections.Counter[str] = collections.Counter()
    by_rule: collections.Counter[str] = collections.Counter()
    by_fixability: collections.Counter[str] = collections.Counter()
    by_file_rule: collections.Counter[tuple[str, str]] = collections.Counter()
    by_file_fixability: collections.Counter[tuple[str, str]] = collections.Counter()
    by_rule_fixability: collections.Counter[tuple[str, str]] = collections.Counter()

    for finding in findings:
        filename = _relative_filename(str(finding.get("filename", "<unknown>")), root)
        rule = str(finding.get("code") or "<unknown>")
        fixability = _fixability(finding)
        by_file[filename] += 1
        by_rule[rule] += 1
        by_fixability[fixability] += 1
        by_file_rule[(filename, rule)] += 1
        by_file_fixability[(filename, fixability)] += 1
        by_rule_fixability[(rule, fixability)] += 1

    return {
        "finding_count": len(findings),
        "by_file": _counter_rows(by_file, ("file",)),
        "by_rule": _counter_rows(by_rule, ("rule",)),
        "by_fixability": _counter_rows(by_fixability, ("fixability",)),
        "by_file_rule": _counter_rows(by_file_rule, ("file", "rule")),
        "by_file_fixability": _counter_rows(
            by_file_fixability, ("file", "fixability")
        ),
        "by_rule_fixability": _counter_rows(
            by_rule_fixability, ("rule", "fixability")
        ),
    }


def _git_value(root: pathlib.Path, *args: str) -> str:
    result = _run(("git", *args), root)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _load_findings(result: subprocess.CompletedProcess[str], scan_id: str) -> list[dict[str, Any]]:
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"{scan_id} failed with exit code {result.returncode}: {result.stderr.strip()}"
        )
    try:
        rows = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{scan_id} did not emit valid JSON") from exc
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{scan_id} emitted an unexpected JSON shape")
    return rows


def inventory(root: pathlib.Path, output: pathlib.Path) -> None:
    root = root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    version_result = _run(("ruff", "--version"), root)
    version = version_result.stdout.strip()
    if version_result.returncode != 0 or version != EXPECTED_RUFF_VERSION:
        raise RuntimeError(
            f"required {EXPECTED_RUFF_VERSION!r}, observed {version!r}: "
            f"{version_result.stderr.strip()}"
        )

    head = _git_value(root, "rev-parse", "HEAD")
    tree = _git_value(root, "show", "-s", "--format=%T", "HEAD")
    config_path = root / "pyproject.toml"
    config_bytes = config_path.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    current_ignores = config.get("tool", {}).get("ruff", {}).get("lint", {}).get("ignore", [])
    if not isinstance(current_ignores, list) or any(
        not isinstance(value, str) for value in current_ignores
    ):
        raise RuntimeError("tool.ruff.lint.ignore must be a list of strings")

    universe_argv = ("ruff", "rule", "--all", "--output-format", "json")
    universe_result = _run(universe_argv, root)
    if universe_result.returncode != 0:
        raise RuntimeError(f"rule universe failed: {universe_result.stderr.strip()}")
    try:
        universe = json.loads(universe_result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("rule universe did not emit valid JSON") from exc
    if not isinstance(universe, list):
        raise RuntimeError("rule universe emitted an unexpected JSON shape")
    _write_json(output / "rule-universe.json", universe)

    settings_argv = ("ruff", "check", ".", "--show-settings")
    settings_result = _run(settings_argv, root)
    if settings_result.returncode != 0:
        raise RuntimeError(f"effective settings failed: {settings_result.stderr.strip()}")
    (output / "effective-settings.txt").write_text(settings_result.stdout, encoding="utf-8")

    scan_manifest: list[dict[str, Any]] = []
    for scan_id, argv in _scan_definitions(current_ignores):
        result = _run(argv, root)
        findings = _load_findings(result, scan_id)
        findings.sort(
            key=lambda row: (
                _relative_filename(str(row.get("filename", "")), root),
                int((row.get("location") or {}).get("row", 0)),
                int((row.get("location") or {}).get("column", 0)),
                str(row.get("code") or ""),
            )
        )
        findings_name = f"{scan_id}.findings.json"
        summary_name = f"{scan_id}.summary.json"
        _write_json(output / findings_name, findings)
        summary = summarize(findings, root)
        summary.update(
            {
                "scan_id": scan_id,
                "argv": list(argv),
                "exit_code": result.returncode,
                "stderr": result.stderr,
                "findings_file": findings_name,
            }
        )
        _write_json(output / summary_name, summary)
        scan_manifest.append(
            {
                "scan_id": scan_id,
                "argv": list(argv),
                "exit_code": result.returncode,
                "finding_count": len(findings),
                "findings_file": findings_name,
                "summary_file": summary_name,
            }
        )

    manifest = {
        "schema_version": "1.0.0",
        "tool": {"name": "ruff", "version": version},
        "repository": {"head": head, "tree": tree},
        "configuration": {
            "path": "pyproject.toml",
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
            "current_lint_ignore": current_ignores,
            "effective_settings_argv": list(settings_argv),
            "effective_settings_file": "effective-settings.txt",
        },
        "rule_universe": {
            "argv": list(universe_argv),
            "count": len(universe),
            "file": "rule-universe.json",
        },
        "scans": scan_manifest,
        "semantics": {
            "ruff_exit_0": "no findings",
            "ruff_exit_1": "findings observed; inventory remains successful",
            "ruff_exit_gt_1": "inventory failure",
            "source_mutation": "prohibited",
        },
    }
    _write_json(output / "manifest.json", manifest)


def self_test() -> None:
    root = pathlib.Path.cwd().resolve()
    rows = [
        {"filename": str(root / "a.py"), "code": "F401", "fix": None},
        {
            "filename": str(root / "a.py"),
            "code": "F401",
            "fix": {"applicability": "safe", "edits": []},
        },
        {
            "filename": str(root / "b.py"),
            "code": "S110",
            "fix": {"applicability": "unsafe", "edits": []},
        },
    ]
    report = summarize(rows, root)
    assert report["finding_count"] == 3
    assert report["by_file"] == [
        {"file": "a.py", "count": 2},
        {"file": "b.py", "count": 1},
    ]
    assert report["by_rule"] == [
        {"rule": "F401", "count": 2},
        {"rule": "S110", "count": 1},
    ]
    assert report["by_fixability"] == [
        {"fixability": "safe", "count": 1},
        {"fixability": "unavailable", "count": 1},
        {"fixability": "unsafe", "count": 1},
    ]
    scans = dict(_scan_definitions(["E402", "F841"]))
    assert scans["all-current-ignores"][-3:] == (
        "E402,F841",
        "--output-format",
        "json",
    )
    assert scans["all-ignore-empty"].count("lint.ignore = []") == 1
    print("ruff_inventory_report self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("ruff-inventory"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    inventory(args.root, args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ruff inventory failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
