import os
os.environ.setdefault("SHOP_FAST", "1")  # keep the 1M dev set out of the test suite
os.environ.setdefault("SHOP_CASE_LIMIT", "2")  # harness tests need a few cases, not all 36 (the ratchet scores all)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
