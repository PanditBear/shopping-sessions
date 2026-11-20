import csv
import json

from split import split_rows, main


def rows(n=100):
    return [{"id": i, "cls": "a" if i % 4 else "b", "input": str(i), "gold": str(i)} for i in range(n)]


def test_deterministic_and_disjoint():
    d1, h1 = split_rows(rows(), seed=3)
    d2, h2 = split_rows(rows(), seed=3)
    assert (d1, h1) == (d2, h2)
    assert len(d1) == 70 and len(h1) == 30
    assert not {r["id"] for r in d1} & {r["id"] for r in h1}
    assert split_rows(rows(), seed=4)[0] != d1


def test_stratify_keeps_classes():
    d, h = split_rows(rows(), stratify="cls")
    assert sum(r["cls"] == "b" for r in h) in (7, 8) and sum(r["cls"] == "b" for r in d) in (17, 18)


def test_cli_jsonl_and_csv(tmp_path):
    src = tmp_path / "all.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows(20)), encoding="utf-8")
    main([str(src), "--out", str(tmp_path / "o1")])
    assert len((tmp_path / "o1" / "dev.jsonl").read_text().splitlines()) == 14
    c = tmp_path / "all.csv"
    with open(c, "w", newline="") as f:
        w = csv.DictWriter(f, ["id", "cls", "input", "gold"])
        w.writeheader()
        w.writerows(rows(20))
    main([str(c), "--out", str(tmp_path / "o2"), "--stratify", "cls"])
    assert len(list(csv.DictReader(open(tmp_path / "o2" / "holdout.csv")))) == 6
