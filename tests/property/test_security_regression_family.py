import importlib

json = importlib.import_module("json")
string = importlib.import_module("string")
hypothesis = importlib.import_module("hypothesis")
pytest = importlib.import_module("pytest")
df = importlib.import_module("glx.tools.delta_fingerprint")
ic = importlib.import_module("glx.tools.invariants_check")
ev = importlib.import_module("analysis.grammar.events")
st = hypothesis.strategies
PROP = hypothesis.settings(max_examples=100, derandomize=True, deadline=None)


def _bind_glx(monkeypatch, tmp_path):
    glx = tmp_path / ".glx"
    glx.mkdir(exist_ok=True)
    paths = (
        ("SPEC_STATE", "spec_state.json"),
        ("DELTA_REPORT", "delta_report.json"),
        ("COMMIT_ANALYSIS", "commit_analysis.json"),
    )
    for name, leaf in paths:
        monkeypatch.setattr(ic, name, glx / leaf)
    monkeypatch.setattr(ic, "GLX_DIR", glx)
    monkeypatch.setattr(ic, "_git_diff_text", lambda _: "")


@PROP
@hypothesis.given(st.text(), st.text())
def test_fingerprint_properties(old, new):
    fp = df.fingerprint_from_texts(old, new)
    hist = df.extract_delta_tokens(old, new)
    assert fp == df.fingerprint_from_texts(old, new)
    assert len(fp) == 16
    assert all(ch in string.hexdigits.lower() for ch in fp)
    assert all(isinstance(value, int) for value in hist.values())
    assert all(not isinstance(value, bool) for value in hist.values())
    assert all(value >= 0 for value in hist.values())


def test_fingerprint_order_independent():
    left = {"ADD_FN": 2, "ΔIMPORT": 1, "MODIFY_SIG": 3}
    right = dict(reversed(list(left.items())))
    assert df._fingerprint(left) == df._fingerprint(right)


@PROP
@hypothesis.given(st.one_of(st.none(), st.integers(), st.text(), st.lists(st.integers())))
def test_event_non_mapping_rejected(payload):
    ok, errors = ev.validate_event_payload(ev.TOPIC_ANALYTICS_DELTA_READY, payload)
    assert ok is False
    assert errors


def test_event_shape_contracts():
    check = ev.validate_event_payload
    assert not check("unknown.topic", {})[0]
    assert not check(ev.TOPIC_ANALYTICS_DELTA_READY, {})[0]
    assert not check(ev.TOPIC_ANALYTICS_DELTA_READY, {"delta_report": []})[0]
    assert not check(ev.TOPIC_ANALYTICS_INVARIANTS_VIOLATION, {})[0]
    assert not check(ev.TOPIC_SCOPE_META_READY, {"level": 1, "name": "x", "paths": {}})[0]
    assert not check(ev.TOPIC_SCOPE_METRICS_UPDATED, {"paths": {"metrics": 7}})[0]
    assert check(ev.TOPIC_ANALYTICS_DELTA_READY, {"delta_report": {}})[0]
    assert check(ev.TOPIC_ANALYTICS_INVARIANTS_VIOLATION, {"violations": {}})[0]
    assert check(ev.TOPIC_SCOPE_METRICS_UPDATED, {"kind": "graph_metrics"})[0]
    payload = {"level": "file", "name": "x", "paths": {"json": "x"}}
    assert check(ev.TOPIC_SCOPE_META_READY, payload)[0]


@pytest.mark.parametrize("value", [True, 1.5, "x", None, [], -1])
def test_delta_hist_invalid_values_fail_closed(monkeypatch, tmp_path, value):
    _bind_glx(monkeypatch, tmp_path)
    payload = {"range": "A..B", "hist": {"X": value}}
    ic.DELTA_REPORT.write_text(json.dumps(payload), encoding="utf-8")
    assert ic.main(["--range", "A..B"]) == 1


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf")])
def test_threshold_invalid_values_fail_closed(monkeypatch, tmp_path, value):
    _bind_glx(monkeypatch, tmp_path)
    report = {"range": "A..B", "hist": {}}
    ic.DELTA_REPORT.write_text(json.dumps(report), encoding="utf-8")
    spec = {"thresholds": {"repo": {"alpha": value, "beta": 0.92, "z": 0.99}}}
    ic.SPEC_STATE.write_text(json.dumps(spec, allow_nan=True), encoding="utf-8")
    assert ic.main(["--range", "A..B"]) == 1


def test_valid_policy_and_absent_policy_no_materialize(monkeypatch, tmp_path):
    _bind_glx(monkeypatch, tmp_path)
    assert ic._load_thresholds() == {"alpha": 0.85, "beta": 0.92, "z": 0.99}
    assert not ic.SPEC_STATE.exists()
    policy = {"thresholds": {"repo": {"alpha": 0.1, "beta": 0.5, "z": 0.9}}}
    ic.SPEC_STATE.write_text(json.dumps(policy), encoding="utf-8")
    assert ic._load_thresholds() == policy["thresholds"]["repo"]
