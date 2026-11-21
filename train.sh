#!/bin/bash
# train.sh SHA: instrumented musl build with the Dockerfile flags, run on generated data, merge profile.
set -euo pipefail
sha=$1; W=/tmp/pgo; mkdir -p $W/raw
git fetch -q origin "$sha" 2>/dev/null || true
git worktree add -q --detach $W/src "$sha"
curl -sSfL -o $W/gen.zip https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/generator.zip
echo "13724cadbac5a99d847e27cea911cd2e338ff972b46c2039da87452011bb101e  $W/gen.zip" | sha256sum -c --quiet
(cd $W && unzip -q -o gen.zip && cd generator && cargo build --release -q)
G=$(find $W -path '*release/generate' -type f | head -1)
(cd $W/src/impl/rs && CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="-C target-cpu=x86-64-v3 -C profile-generate=$W/raw" \
   CARGO_TARGET_DIR=$W/tgt cargo build --release --locked -q --target x86_64-unknown-linux-musl)
B=$W/tgt/x86_64-unknown-linux-musl/release/shop
for seed in 7 11; do
  D=$W/data$seed; $G --seed $seed --events ${EVENTS:-20000000} --clicks ${CLICKS:-5000000} --output $D --threads 4 >/dev/null
  for nc in 4 8; do
    o=$(mktemp -d); s2=$(mktemp -d)
    DATA_DIR=$D OUTPUT_DIR=$o SCRATCH_DIR=$s2 NCPU=$nc SHOP_TIMING=1 $B 2>&1 | tail -3
    find $o $s2 -mindepth 1 -delete
  done
  find $D -mindepth 1 -delete
done
ls -l $W/raw
LP=$(find $(rustc --print sysroot) -name llvm-profdata | head -1)
$LP --version
$LP merge -o $W/merged.profdata $W/raw/*.profraw
ls -l $W/merged.profdata; sha256sum $W/merged.profdata
