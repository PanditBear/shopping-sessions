"""solve.py: the solver (predict) and the submission writer (CLI)."""
import argparse
import json
import math
import sys


def predict(case: dict):
    """Return the prediction for one case (dict with id, input, gold; do NOT read gold).
    Must be a top-level function (score.py runs it in a process pool).
    Toy baseline: upper-case ASCII strings only (deliberately imperfect, so there is something to improve)."""
    s = case["input"]
    return s.upper() if s.isascii() else s


def fallback(case: dict):
    """The prediction written when predict() crashes or returns something unwritable.
    TODO-KICKOFF: set to the cheapest valid answer for the real task (e.g. the input unchanged, an empty
    list, the majority class). It must never raise and must be JSON-safe."""
    return case.get("input")


def _finite(x):
    """Recursively replace NaN/Infinity (invalid JSON) with None so the output always parses."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def safe_line(case: dict) -> str:
    """One submission line for a case. Never raises: a crash on one hidden input must not sink the submission."""
    try:
        pred = _finite(predict(case))
        # TODO-KICKOFF: write the contest's real submission format (keep allow_nan=False).
        return json.dumps({"id": case["id"], "pred": pred}, ensure_ascii=False, allow_nan=False)
    except Exception as e:  # noqa: BLE001 - deliberate catch-all
        print(f"predict failed on {case.get('id')!r}: {type(e).__name__}: {e}", file=sys.stderr)
        try:
            return json.dumps({"id": case["id"], "pred": _finite(fallback(case))}, ensure_ascii=False, allow_nan=False)
        except Exception:  # noqa: BLE001
            return json.dumps({"id": case.get("id"), "pred": None}, ensure_ascii=False, default=str)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write the submission file for the given input files")
    ap.add_argument("inputs", nargs="+", help="input file path(s)")
    ap.add_argument("--out", default="submission.jsonl")
    a = ap.parse_args(argv)
    n = bad = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for path in a.inputs:
            # TODO-KICKOFF: parse the contest's real input format into case dicts (id, input).
            for lineno, line in enumerate(open(path, encoding="utf-8"), 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    case = {"id": r["id"], "input": r["input"]}
                except Exception as e:  # noqa: BLE001
                    bad += 1
                    print(f"{path}:{lineno}: unreadable input line skipped ({type(e).__name__})", file=sys.stderr)
                    continue
                out.write(safe_line(case) + "\n")
                n += 1
    print(f"wrote {n} predictions to {a.out}" + (f"; skipped {bad} unreadable lines" if bad else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
