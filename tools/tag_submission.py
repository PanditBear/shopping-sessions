"""Tag HEAD sub-<n> and record the leaderboard score with bx board.

  python tools/tag_submission.py <board_score> [--note "..."]
Set BX to override the bx.py path.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BX = os.environ.get("BX", "D:/factory/buildathon/tools/bx.py")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("score")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    tags = subprocess.run(["git", "tag", "--list", "sub-*"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    n = max([int(m.group(1)) for t in tags if (m := re.fullmatch(r"sub-(\d+)", t))], default=0) + 1
    subprocess.run(["git", "-c", "user.name=bx", "-c", "user.email=bx@localhost", "tag", f"sub-{n}"], cwd=ROOT, check=True)
    cmd = [sys.executable, BX, "--repo", str(ROOT), "board", a.score]
    if a.note:
        cmd += ["--note", a.note]
    subprocess.run(cmd, check=True)
    print(f"tagged sub-{n}")


if __name__ == "__main__":
    main()
