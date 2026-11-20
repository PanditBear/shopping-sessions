import adapter


def test_cases_contract():
    for split in ("dev", "holdout"):
        cases = adapter.load_cases(split)
        assert cases
        ids = [c["id"] for c in cases]
        assert len(set(ids)) == len(ids)
        for c in cases:
            assert {"id", "input", "gold"} <= set(c)


def test_metric_contract():
    cases = adapter.load_cases("dev")
    agg, per = adapter.metric({c["id"]: c["gold"] for c in cases}, cases)
    assert isinstance(agg, float) and len(per) == len(cases)
    assert all({"id", "score", "pred", "gold"} <= set(r) for r in per)
    assert isinstance(adapter.HIGHER_IS_BETTER, bool)
    best = max if adapter.HIGHER_IS_BETTER else min
    agg2, _ = adapter.metric({}, cases)   # missing predictions must not crash
    assert best(agg, agg2) == agg
