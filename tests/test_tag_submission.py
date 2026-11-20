import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BX = "D:/factory/buildathon/tools/bx.py"


def g(repo, *a):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo, check=True,
                          capture_output=True, text=True).stdout


def test_tag_submission_tags_and_records_board(tmp_path):
    g(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "tools").mkdir()
    shutil.copy(ROOT / "tools" / "tag_submission.py", tmp_path / "tools" / "tag_submission.py")
    (tmp_path / "bx.toml").write_text('score_cmd = "python -c pass"\ntest_cmd = "python -c pass"\n')
    g(tmp_path, "add", "-A"); g(tmp_path, "commit", "-qm", "root")
    env = dict(os.environ, BX=BX)
    for i, score in enumerate(["0.41", "0.52"], start=1):
        p = subprocess.run([sys.executable, str(tmp_path / "tools" / "tag_submission.py"), score, "--note", f"try {i}"],
                           cwd=tmp_path, env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        assert f"tagged sub-{i}" in p.stdout
        g(tmp_path, "tag", "--list", f"sub-{i}")
    assert g(tmp_path, "tag", "--list", "sub-*").split() == ["sub-1", "sub-2"]
    ledger = (tmp_path / "ledger.csv").read_text()
    assert "board" in ledger and "0.41" in ledger and "0.52" in ledger and "try 2" in ledger
    assert g(tmp_path, "status", "--porcelain").strip() == ""
