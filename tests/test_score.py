import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args):
    return subprocess.run([sys.executable, "score.py", *args], cwd=ROOT, capture_output=True, text=True)


def test_one_json_line_both():
    p = run()
    assert p.returncode == 0, p.stderr
    lines = p.stdout.strip().splitlines()
    assert len(lines) == 1
    o = json.loads(lines[0])
    assert set(o) == {"dev", "holdout", "runtime_s"}
    assert all(isinstance(o[k], (int, float)) for k in o)


def test_dev_only_and_errors_file():
    p = run("--split", "dev")
    o = json.loads(p.stdout)
    assert o["holdout"] is None and isinstance(o["dev"], float)
    f = ROOT / "errors" / "dev.jsonl"
    assert f.exists()
    for line in f.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        assert {"id", "score", "input", "pred", "gold"} <= set(r) and r["score"] != 1.0


def test_jobs_matches_serial():
    a = json.loads(run("--jobs", "1").stdout)
    b = json.loads(run("--jobs", "2").stdout)
    assert (a["dev"], a["holdout"]) == (b["dev"], b["holdout"])
