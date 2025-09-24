import sys, subprocess
from pathlib import Path


def write_task(tdir, tid, entry, tests):
    import json
    path = Path(tdir) / f"{tid}.json"
    with open(path, "w") as f:
        json.dump(dict(entrypoint=entry, tests=tests), f)
    return path


def test_runner_smoke(tmp_path):
    tdir = tmp_path / 'tasks'
    tdir.mkdir()
    write_task(tdir, 't01', 'reverse_str', [dict(args=['abc'], expect='cba')])
    write_task(tdir, 't02', 'fib', [dict(args=[5], expect=5)])
    out = tmp_path / 'rep.json'
    cmd = [sys.executable, '-m', 'glitchlab.gui.bench.ab_runner', '--tasks', str(tdir / '*.json'), '--out', str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert out.exists(), res.stderr
