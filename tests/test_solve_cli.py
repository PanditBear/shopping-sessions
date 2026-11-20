"""The submission writer must never crash and must always write valid, finite JSON (lesson from the dress rehearsal)."""
import json

import solve


def _run(tmp_path, monkeypatch, pred_fn, rows):
    monkeypatch.setattr(solve, "predict", pred_fn)
    inp = tmp_path / "in.jsonl"
    inp.write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = tmp_path / "sub.jsonl"
    solve.main([str(inp), "--out", str(out)])
    lines = out.read_text(encoding="utf-8").splitlines()
    return [json.loads(x, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c))) for x in lines]


def test_crash_on_one_case_uses_fallback(tmp_path, monkeypatch):
    def pred(case):
        if case["id"] == "b":
            raise ZeroDivisionError("boom")
        return "ok"
    rows = [json.dumps({"id": i, "input": "x"}) for i in ("a", "b", "c")]
    got = _run(tmp_path, monkeypatch, pred, rows)
    assert [r["id"] for r in got] == ["a", "b", "c"]
    assert got[0]["pred"] == "ok" and got[1]["pred"] == solve.fallback({"id": "b", "input": "x"})


def test_non_finite_is_made_json_safe(tmp_path, monkeypatch):
    got = _run(tmp_path, monkeypatch, lambda c: {"v": float("inf"), "w": [float("nan"), 1.5]},
               [json.dumps({"id": "a", "input": "x"})])
    assert got[0]["pred"] == {"v": None, "w": [None, 1.5]}


def test_unwritable_prediction_falls_back(tmp_path, monkeypatch):
    got = _run(tmp_path, monkeypatch, lambda c: object(), [json.dumps({"id": "a", "input": "x"})])
    assert got[0]["id"] == "a"


def test_bad_input_line_is_skipped(tmp_path, monkeypatch):
    rows = [json.dumps({"id": "a", "input": "x"}), "{not json", json.dumps({"no_id": 1}), json.dumps({"id": "d", "input": "y"})]
    got = _run(tmp_path, monkeypatch, lambda c: c["input"], rows)
    assert [r["id"] for r in got] == ["a", "d"]
