"""score.py: run solve.predict over dev and holdout via adapter; print ONE JSON line to stdout, nothing else.

  python score.py [--split dev|holdout|both] [--jobs N]
Writes errors/dev.jsonl (non-perfect dev cases, worst first) for workers to read. Diagnostics go to stderr.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import adapter  # noqa: E402
import solve  # noqa: E402

CAP = 2000


def _one(case):
    try:
        return case["id"], solve.predict(case), None
    except Exception as e:  # a crash on one case must not kill the run
        return case["id"], None, f"{type(e).__name__}: {e}"


def run_split(split, jobs):
    cases = adapter.load_cases(split)
    if jobs > 1 and len(cases) > 1:
        import multiprocessing as mp
        with mp.Pool(jobs) as pool:
            res = pool.map(_one, cases, chunksize=max(1, len(cases) // (jobs * 4)))
    else:
        res = [_one(c) for c in cases]
    preds = {}
    for cid, p, err in res:
        preds[cid] = p
        if err:
            print(f"predict failed on {cid}: {err}", file=sys.stderr)
    return cases, adapter.metric(preds, cases)


def _cut(v):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s[:CAP]


def write_errors(cases, per_case):
    by_id = {c["id"]: c for c in cases}
    bad = [r for r in per_case if r["score"] != adapter.PERFECT_SCORE]
    bad.sort(key=lambda r: r["score"], reverse=not adapter.HIGHER_IS_BETTER)
    d = ROOT / "errors"
    d.mkdir(exist_ok=True)
    with open(d / "dev.jsonl", "w", encoding="utf-8") as f:
        for r in bad:
            f.write(json.dumps({"id": r["id"], "score": r["score"], "input": _cut(by_id[r["id"]]["input"]),
                                "pred": _cut(r["pred"]), "gold": _cut(r["gold"])}, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "holdout", "both"], default="both")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    real_stdout, sys.stdout = sys.stdout, sys.stderr   # stray prints from solver code go to stderr
    out = {"dev": None, "holdout": None}
    t0 = time.perf_counter()
    for split in ("dev", "holdout"):
        if a.split in (split, "both"):
            cases, (agg, per_case) = run_split(split, a.jobs)
            out[split] = agg
            if split == "dev":
                write_errors(cases, per_case)
    out["runtime_s"] = round(time.perf_counter() - t0, 3)
    sys.stdout = real_stdout
    print(json.dumps(out))


if __name__ == "__main__":
    main()
