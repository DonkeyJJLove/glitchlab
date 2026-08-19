import importlib
import json
import string

import hypothesis
import pytest


df = importlib.import_module("glx.tools.delta_fingerprint")
ic = importlib.import_module("glx.tools.invariants_check")
ev = importlib.import_module("analysis.grammar.events")
PROP = hypothesis.settings(max_examples=100, derandomize=True, deadline=None)


def _bind_glx(monkeypatch, tmp_path):
    glx = tmp_path / ".glx"
    glx.mkdir(exist_ok=True)
    for name, leaf in (
        ("SPEC_STATE", "spec_state.json"),
        ("DELTA_REPORT", "delta_report.json"),
        ("COMMIT_ANALYSIS", "commit_analysis.json"),
    ):
        monkeypatch.setattr(ic, name, glx / leaf)
    monkeypatch.setattr(ic, "GLX_DIR", glx)
    monkeypatch.setattr(ic, "_git_diff_text", lambda _: "")


@PROP
@hypothesis.given(hypothesis.strategies.text(), hypothesis.strategies.text())
def test_fingerprint_properties(old, new):
    fp = df.fingerprint_from_texts(old, new)
    hist = df.extract_delta_tokens(old, new)
    assert fp == df.fingerprint_from_texts(old, new)
    assert len(fp) == 16 and all(ch in string.hexdigits.lower() for ch in fp)
    assert all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in hist.values()
    )


def test_fingerprint_is_order_independent_for_same_histogram():
    left = {"ADD_FN": 2, "ΔIMPORT": 1, "MODIFY_SIG": 3}
    right = dict(reversed(list(left.items())))
    assert df._fingerprint(left) == df._fingerprint(right)


@PROP
@hypothesis.given(
    hypothesis.strategies.one_of(
        hypothesis.strategies.none(),
        hypothesis.strategies.integers(),
        hypothesis.strategies.text(),
        hypothesis.strategies.lists(hypothesis.strategies.integers()),
    ),
)
def test_event_validator_rejects_non_mappings_without_crashing(payload):
    ok, errors = ev.validate_event_payload(ev.TOPIC_ANALYTICS_DELTA_READY, payload)
    assert ok is False and errors


def test_event_validator_shape_contracts():
    rejected = (
        ("unknown.topic", {}),
        (ev.TOPIC_ANALYTICS_DELTA_READY, {}),
        (ev.TOPIC_ANALYTICS_DELTA_READY, {"delta_report": []}),
        (ev.TOPIC_ANALYTICS_INVARIANTS_VIOLATION, {}),
        (ev.TOPIC_SCOPE_META_READY, {"level": 1, "name": "x", "paths": {}}),
        (ev.TOPIC_SCOPE_METRICS_UPDATED, {"paths": {"metrics": 7}}),
    )
    accepted = (
        (ev.TOPIC_ANALYTICS_DELTA_READY, {"delta_report": {}}),
        (ev.TOPIC_ANALYTICS_INVARIANTS_VIOLATION, {"violations": {}}),
        (ev.TOPIC_SCOPE_METRICS_UPDATED, {"kind": "graph_metrics"}),
        (ev.TOPIC_SCOPE_META_READY, {"level": "file", "name": "x", "paths": {"json": "x"}}),
    )
    assert all(not ev.validate_event_payload(topic, payload)[0] for topic, payload in rejected)
    assert all(ev.validate_event_payload(topic, payload)[0] for topic, payload in accepted)


@pytest.mark.parametrize("value", [True, 1.5, "x", None, [], -1])
def test_delta_hist_invalid_values_fail_closed(monkeypatch, tmp_path, value):
    _bind_glx(monkeypatch, tmp_path)
    payload = {"range": "A..B", "hist": {"X": value}}
    ic.DELTA_REPORT.write_text(json.dumps(payload), encoding="utf-8")
    assert ic.main(["--range", "A..B"]) == 1


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf")])
def test_threshold_nonfinite_or_out_of_range_fails_closed(monkeypatch, tmp_path, value):
    _bind_glx(monkeypatch, tmp_path)
    report = {"range": "A..B", "hist": {}}
    ic.DELTA_REPORT.write_text(json.dumps(report), encoding="utf-8")
    spec = {"thresholds": {"repo": {"alpha": value, "beta": 0.92, "z": 0.99}}}
    ic.SPEC_STATE.write_text(json.dumps(spec, allow_nan=True), encoding="utf-8")
    assert ic.main(["--range", "A..B"]) == 1


def test_valid_threshold_policy_and_absent_policy_no_materialize(monkeypatch, tmp_path):
    _bind_glx(monkeypatch, tmp_path)
    assert ic._load_thresholds() == {"alpha": 0.85, "beta": 0.92, "z": 0.99}
    assert not ic.SPEC_STATE.exists()
    policy = {"thresholds": {"repo": {"alpha": 0.1, "beta": 0.5, "z": 0.9}}}
    ic.SPEC_STATE.write_text(json.dumps(policy), encoding="utf-8")
    assert ic._load_thresholds() == policy["thresholds"]["repo"]
