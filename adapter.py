"""adapter.py: Shopping Sessions. A case is one input folder run through the pipeline and diffed exactly
against its expected tables (scripts/compare.py logic). Score per case: 1.0 exact match, else 0.0.

dev     = the 36 edge-case fixtures from the kit.
holdout = the 36 fixtures + the 1M-event dev set (seed 1) diffed against dev-answer, when both exist locally.
Set SHOP_FAST=1 to leave the dev set out (tests do this; it is slow in Python).
Speed is the contest metric, but only a 100%-correct pipeline scores at all, so the ratchet here is correctness.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
BXDATA = Path(os.environ.get("SHOP_BXDATA", "C:/bxdata/shop"))
DEVSET = BXDATA / "dev"                       # generate --seed 1 --events 1000000 --clicks 250000 --output dev
DEVANSWER = BXDATA / "kit" / "dev-answer" / "dev-answer"   # user_daily.csv, campaign_daily.csv

HIGHER_IS_BETTER = True
PERFECT_SCORE = 1.0


def _fixture_cases():
    return [{"id": d.name, "input": {"data": str(d / "input"), "expected": str(d / "expected")}, "gold": "OK"}
            for d in sorted(FIXTURES.iterdir()) if (d / "input").is_dir()]


def load_cases(split: str) -> list[dict]:
    cases = _fixture_cases()
    if split == "holdout" and not os.environ.get("SHOP_FAST") and DEVSET.is_dir() and DEVANSWER.is_dir():
        cases.append({"id": "devset", "input": {"data": str(DEVSET), "expected": str(DEVANSWER)}, "gold": "OK"})
    return cases


def metric(preds: dict, cases: list[dict]) -> tuple[float, list[dict]]:
    per_case = []
    for c in cases:
        p = preds.get(c["id"])
        per_case.append({"id": c["id"], "score": 1.0 if p == "OK" else 0.0, "pred": p, "gold": c["gold"]})
    agg = sum(r["score"] for r in per_case) / len(per_case) if per_case else 0.0
    return agg, per_case
