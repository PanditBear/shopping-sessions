# Contest template (plugs into tools/bx.py)

Kickoff fill-in, in order (grep `TODO-KICKOFF`):
1. Copy: `python new_project.py <name>` (only after kickoff), then work in `D:/factory/buildathon/<name>`.
2. Delete the toy `data/*`, then `python tools/split.py <file-or-dir> [--stratify KEY]` -> `data/dev.*`, `data/holdout.*`.
3. `adapter.py`: `load_cases` (id/input/gold), `metric` (mirror the published formula), `HIGHER_IS_BETTER`, `PERFECT_SCORE`.
4. `bx.toml`: set `higher_is_better` to match; keep `ratchet_key = "holdout"`.
5. `solve.py`: real baseline in `predict`; a cheap valid `fallback`; the CLI's input parsing and submission format (keep `safe_line`: one bad case must never sink the file).
6. `tests/`: adjust toy-specific expectations (if any); keep the contract tests passing.
7. `python score.py --split dev` prints one JSON line; `errors/dev.jsonl` lists the worst dev cases.
8. Commit, `bx seed` again (new_project seeded on the toy data; the ratchet only moves up from the seed), submit
   baseline, `python tools/tag_submission.py <board score>`.
9. Before each submission: `python tools/review_pack.py`, paste `review/pack-<n>.md` to the reviewer.
