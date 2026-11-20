"""adapter.py: the ONLY file filled at kickoff. Maps the contest's data and published metric onto the template.

Toy default: upper-case a string; metric = exact-match fraction.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# TODO-KICKOFF: True if a bigger aggregate score is better (accuracy, F1); False for errors/losses.
HIGHER_IS_BETTER = True
# TODO-KICKOFF: per-case score that counts as perfect (cases at this score are left out of errors/dev.jsonl).
PERFECT_SCORE = 1.0


def load_cases(split: str) -> list[dict]:
    """Return the cases of `split` ("dev" or "holdout") as a list of dicts, each with keys
    id (unique, str-able), input (whatever solve.predict needs), gold (the label).
    TODO-KICKOFF: read the real files written by tools/split.py (data/dev.*, data/holdout.*) and map the columns."""
    f = DATA / f"{split}.jsonl"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run tools/split.py or fix load_cases")
    cases = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            cases.append({"id": r["id"], "input": r["input"], "gold": r["gold"]})
    return cases


def metric(preds: dict, cases: list[dict]) -> tuple[float, list[dict]]:
    """Mirror the PUBLISHED contest metric exactly.
    preds maps case id -> prediction (None if predict crashed). Returns (aggregate_score, per_case) where per_case is
    a list of {"id","score","pred","gold"}, one per case, in case order.
    TODO-KICKOFF: replace the toy exact-match with the real formula (keep the return shape)."""
    per_case = []
    for c in cases:
        p = preds.get(c["id"])
        per_case.append({"id": c["id"], "score": 1.0 if p == c["gold"] else 0.0, "pred": p, "gold": c["gold"]})
    agg = sum(r["score"] for r in per_case) / len(per_case) if per_case else 0.0
    return agg, per_case
