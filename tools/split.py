"""Deterministic dev/holdout split of a labelled jsonl/csv file or a directory of them.

  python tools/split.py SRC [--seed 0] [--ratio 0.7] [--stratify KEY] [--out data]
Writes <out>/dev.<ext> and <out>/holdout.<ext> (ext of the source; a directory uses the first file's ext).
"""
import argparse
import csv
import json
import random
from pathlib import Path


def read_rows(p: Path):
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if p.suffix == ".csv":
        with open(p, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise SystemExit(f"unsupported file type: {p}")


def split_rows(rows, seed=0, ratio=0.7, stratify=None):
    rng = random.Random(seed)
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(str(r.get(stratify)) if stratify else "", []).append(i)
    dev, hold = [], []
    for k in sorted(groups):
        idx = groups[k][:]
        rng.shuffle(idx)
        n = round(len(idx) * ratio)
        if len(idx) > 1:
            n = min(max(n, 1), len(idx) - 1)
        dev += idx[:n]
        hold += idx[n:]
    return [rows[i] for i in sorted(dev)], [rows[i] for i in sorted(hold)]


def write_rows(p: Path, rows):
    if p.suffix == ".jsonl":
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    else:
        keys = list(rows[0].keys()) if rows else []
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratio", type=float, default=0.7)
    ap.add_argument("--stratify")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data"))
    a = ap.parse_args(argv)
    src = Path(a.src)
    files = sorted(f for f in src.rglob("*") if f.suffix in (".jsonl", ".csv")) if src.is_dir() else [src]
    if not files:
        raise SystemExit("no .jsonl/.csv files found")
    rows = [r for f in files for r in read_rows(f)]
    dev, hold = split_rows(rows, a.seed, a.ratio, a.stratify)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / f"dev{files[0].suffix}", dev)
    write_rows(out / f"holdout{files[0].suffix}", hold)
    print(f"dev={len(dev)} holdout={len(hold)} -> {out}")


if __name__ == "__main__":
    main()
