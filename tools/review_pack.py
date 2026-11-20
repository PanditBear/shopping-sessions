"""Build review/pack-<n>.md for an external code reviewer.

  python tools/review_pack.py [--ref REF] [--repo .]
REF defaults to the latest tag matching sub-*, else the root commit.
"""
import argparse
import re
import subprocess
from pathlib import Path

CAP = 60000
HEADER = """# Code review request

Review the diff below (a scored coding contest solution). Report findings as a numbered list, each with file:line:
1. correctness bugs
2. overfitting to dev (tuning to dev cases or leaking holdout/gold)
3. crashes on unseen inputs
4. runtime risks (time, memory, scoring-time limits)
5. submission-format errors
Be concrete and brief; skip style nits.

"""


def git(repo, *args):
    p = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.stdout.strip() if p.returncode == 0 else ""


def default_ref(repo):
    tags = git(repo, "tag", "--list", "sub-*", "--sort=-v:refname").splitlines()
    return tags[0] if tags else git(repo, "rev-list", "--max-parents=0", "HEAD").splitlines()[0]


def build(repo, ref=None):
    repo = Path(repo)
    ref = ref or default_ref(repo)
    diff = git(repo, "diff", ref, "HEAD", "--", ".", ":(exclude)data", ":(exclude)errors", ":(exclude)ledger.csv", ":(exclude)ratchet.json")
    tickets = [p for p in git(repo, "diff", "--name-only", "--diff-filter=AM", ref, "HEAD", "--", "tickets/").splitlines()
               if p.endswith(".toml")]
    ledger = repo / "ledger.csv"
    tail = "\n".join(ledger.read_text(encoding="utf-8").splitlines()[-10:]) if ledger.exists() else "(no ledger)"
    tk = ""
    for t in tickets:
        f = repo / t
        body = " ".join(f.read_text(encoding="utf-8").split())[:300] if f.exists() else ""
        tk += f"- {t}: {body}\n"
    head = f"{HEADER}Range: {ref}..HEAD ({git(repo, 'rev-parse', '--short', 'HEAD')})\n\n"
    mid = f"## Tickets landed\n{tk or '(none)'}\n\n## Ledger (last lines)\n```\n{tail}\n```\n\n## Diff\n```diff\n"
    budget = CAP - len(head) - len(mid) - 300
    note = ""
    if len(diff) > budget:
        note = f"\n[TRUNCATED: diff was {len(diff)} chars, showing first {budget}; review the rest in the repo]\n"
        diff = diff[:budget]
    return head + mid + diff + "\n```\n" + note


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    a = ap.parse_args(argv)
    d = Path(a.repo) / "review"
    d.mkdir(exist_ok=True)
    ns = [int(m.group(1)) for f in d.glob("pack-*.md") if (m := re.fullmatch(r"pack-(\d+)\.md", f.name))]
    out = d / f"pack-{max(ns, default=0) + 1}.md"
    out.write_text(build(a.repo, a.ref), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
