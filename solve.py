"""solve.py: run the pipeline on one case's input folder and diff its output against the expected tables.

The pipeline command comes from SHOP_IMPL (default: the Python reference, impl/py/pipeline.py). It gets DATA_DIR,
OUTPUT_DIR and SCRATCH_DIR as env vars, like the scorer's container. predict() returns "OK" on an exact match,
else a short diff summary (what workers read in errors/dev.jsonl).
"""
import io
import os
import shlex
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
DEFAULT_IMPL = [sys.executable, str(ROOT / "impl" / "py" / "pipeline.py")]
TIMEOUT_S = int(os.environ.get("SHOP_TIMEOUT", "600"))


def _diff(out: Path, expected: Path) -> str:
    import compare  # scripts/compare.py: exact, order-insensitive, prints what differs
    buf = io.StringIO()
    old = sys.argv
    try:
        sys.argv = ["compare.py", str(out), str(expected)]
        with redirect_stdout(buf):
            rc = compare.main()
    finally:
        sys.argv = old
    return "OK" if rc == 0 else ("DIFF " + buf.getvalue().strip())[:3000]


def predict(case: dict):
    # SHOP_IMPL: a command line with forward slashes, e.g. "C:/bxdata/shop/bin/shop.exe".
    impl = shlex.split(os.environ["SHOP_IMPL"]) if os.environ.get("SHOP_IMPL") else DEFAULT_IMPL
    with tempfile.TemporaryDirectory() as td:
        out, scratch = Path(td) / "out", Path(td) / "scratch"
        out.mkdir(); scratch.mkdir()
        env = {**os.environ, "DATA_DIR": case["input"]["data"], "OUTPUT_DIR": str(out), "SCRATCH_DIR": str(scratch),
               "NCPU": os.environ.get("NCPU", "8"), "MEM_GB": os.environ.get("MEM_GB", "24")}
        p = subprocess.run(impl, env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
        if p.returncode != 0:
            return f"EXIT {p.returncode}: {p.stderr[-1500:]}"
        try:
            return _diff(out, Path(case["input"]["expected"]))
        except Exception as e:  # noqa: BLE001 - unreadable output is a wrong answer, not a crash
            return f"BAD OUTPUT {type(e).__name__}: {e}"[:3000]
