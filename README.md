# Shopping Sessions: Carter Core

Builders Buildathon entry for the Shopping Sessions track. The spec is in `rules.md`.

- `Dockerfile` builds the submission: a Rust pipeline (`impl/rs/`) that reads `DATA_DIR` and writes `user_daily/` and `campaign_daily/` Parquet tables to `OUTPUT_DIR`.
- `impl/py/pipeline.py` is an independent Python reference. It passes all 36 fixtures and the published 1M-event dev answer, and the Rust pipeline is diffed against it on generated data.
- `fixtures/` holds the 36 edge-case fixtures from the starter kit. `scripts/` holds the kit's runner and compare scripts.
- `score.py`, `adapter.py`, `solve.py` and `tools/` form the local harness: `python score.py --jobs 8` runs an implementation on every fixture (set `SHOP_IMPL` to pick one) and prints the fraction that match exactly.
- CI (`.github/workflows/ci.yml`) builds the linux/amd64 image, checks it is 50 MB or less, and runs all fixtures the way the scorer does.

MIT licensed.
