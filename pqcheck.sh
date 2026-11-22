#!/bin/bash
# pqcheck.sh REF... : build each ref the way the scorer image is built (static musl, x86-64-v3), run it on a
# generated dataset, then read both output tables with Spark (vectorized and row readers), pyarrow, duckdb and
# polars. Every reader must agree, every ref must give the rows of the first ref, and `day` must read as a DATE.
set -euo pipefail
EVENTS=${EVENTS:-3000000}; CLICKS=${CLICKS:-800000}; SEED=${SEED:-11}
W=/tmp/pq; mkdir -p $W
curl -sSfL -o $W/gen.zip https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/generator.zip
echo "78f973f75d14b48e792628284f19c75de71fa597d20952e16a9ced93c78bee4a  $W/gen.zip" | sha256sum -c --quiet
(cd $W && unzip -q -o gen.zip && cd generator && cargo build --release -q)
G=$(find $W -path '*release/generate' -type f | head -1)
$G --seed $SEED --events $EVENTS --clicks $CLICKS --output $W/data --threads 4 >/dev/null
OUTS=()
for r in "$@"; do
  git fetch -q origin "$r" 2>/dev/null || true
  sha=$(git rev-parse --short "$r^{commit}" 2>/dev/null || git rev-parse --short "origin/$r")
  git worktree add -q --detach $W/w-$sha "$sha"
  (cd $W/w-$sha/impl/rs && CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_TARGET_DIR=$W/tgt \
     cargo build --release --locked -q --target x86_64-unknown-linux-musl)
  mkdir -p $W/out-$sha $W/scr-$sha
  DATA_DIR=$W/data OUTPUT_DIR=$W/out-$sha SCRATCH_DIR=$W/scr-$sha $W/tgt/x86_64-unknown-linux-musl/release/shop
  echo "== $sha ($(git log -1 --format=%s $sha))"; find $W/out-$sha -maxdepth 2 | head -40
  OUTS+=("$W/out-$sha")
done
python3 -m pip install -q pyspark==3.5.3 duckdb polars pyarrow; java -version 2>&1 | head -1; python3 --version
python3 pqread.py "${OUTS[@]}"
