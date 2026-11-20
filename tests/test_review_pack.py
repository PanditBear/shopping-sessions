import subprocess

import review_pack


def g(repo, *a):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo, check=True, capture_output=True)


def test_pack_on_temp_repo(tmp_path):
    g(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "d.jsonl").write_text("secret\n")
    g(tmp_path, "add", "-A"); g(tmp_path, "commit", "-qm", "root")
    g(tmp_path, "tag", "sub-1")
    (tmp_path / "a.py").write_text("x = 2\n")
    (tmp_path / "data" / "d.jsonl").write_text("secret2\n")
    (tmp_path / "tickets").mkdir()
    (tmp_path / "tickets" / "t1.toml").write_text('id = "t1"\nwhy = "w"\n')
    (tmp_path / "ledger.csv").write_text("time,event\n1,land\n")
    g(tmp_path, "add", "-A"); g(tmp_path, "commit", "-qm", "two")
    review_pack.main(["--repo", str(tmp_path)])
    review_pack.main(["--repo", str(tmp_path)])
    t = (tmp_path / "review" / "pack-2.md").read_text()
    assert "+x = 2" in t and "secret" not in t and "tickets/t1.toml" in t and "1,land" in t
    assert "overfitting" in t and "file:line" in t


def test_truncation(tmp_path):
    g(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "a.txt").write_text("a\n")
    g(tmp_path, "add", "-A"); g(tmp_path, "commit", "-qm", "root")
    (tmp_path / "a.txt").write_text("b\n" * 80000)
    g(tmp_path, "add", "-A"); g(tmp_path, "commit", "-qm", "big")
    t = review_pack.build(tmp_path)
    assert len(t) <= 61000 and "TRUNCATED" in t
