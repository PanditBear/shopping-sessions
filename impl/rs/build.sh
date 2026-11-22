#!/bin/sh
# Release build of shop with profile-guided optimization: static musl, x86-64-v3.
# The Dockerfile build stage runs exactly this script; the bench runs it too when a commit has it.
#   sh build.sh OUT
# 1. instrumented build (-Cprofile-generate), same target and flags as the final build
# 2. training: the instrumented binary on each committed dataset pgo/*.tar.gz at NCPU=2 and NCPU=8
#    (a temp dir for data and output, no /data mount needed)
# 3. llvm-profdata merge (rustup component llvm-tools of the same toolchain)
# 4. final build with -Cprofile-use; copies the binary to OUT
# Env: SHOP_PGO=0            plain build without a profile (same toolchain and flags, for A/B)
#      SHOP_PGO_TRAIN=bench  train on the dataset dir SHOP_BENCH_DATA instead of pgo/*.tar.gz (measurement only)
#      CARGO_TARGET_DIR      build dir (default ./target); the instrumented build goes to $CARGO_TARGET_DIR/pgo-gen
set -eu
OUT=${1:?usage: build.sh OUT}
case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac
cd "$(dirname "$0")"
T=x86_64-unknown-linux-musl
FLAGS="-C target-cpu=x86-64-v3"
TD=${CARGO_TARGET_DIR:-$PWD/target}
export CC_x86_64_unknown_linux_musl="${CC_x86_64_unknown_linux_musl:-musl-gcc}"

say() { echo "build.sh: $*"; }
step() { t1=$(date +%s); say "step $1 $((t1 - t0)) s"; t0=$t1; }
cargo_build() { # target-dir rustflags
  CARGO_TARGET_DIR=$1 RUSTFLAGS=$2 cargo build --release --locked --target $T
}
say "$(rustc -V)"
t0=$(date +%s); tall=$t0

if [ "${SHOP_PGO:-1}" = 0 ]; then
  cargo_build "$TD" "$FLAGS"
  cp "$TD/$T/release/shop" "$OUT"
  step plain
  exit 0
fi

host=$(rustc -vV | sed -n 's/^host: //p')
PD=$(rustc --print sysroot)/lib/rustlib/$host/bin/llvm-profdata
[ -x "$PD" ] || rustup component add llvm-tools
[ -x "$PD" ] || { say "llvm-profdata not found at $PD"; exit 1; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
P=$W/prof
mkdir -p "$P"

# 1. instrumented build
cargo_build "$TD/pgo-gen" "$FLAGS -Cprofile-generate=$P"
GEN=$W/shop-gen
cp "$TD/pgo-gen/$T/release/shop" "$GEN"
step instrumented

# 2. training
sets=""
if [ "${SHOP_PGO_TRAIN:-}" = bench ] && [ -f "${SHOP_BENCH_DATA:-/nonexistent}/manifest.json" ]; then
  sets=$SHOP_BENCH_DATA
  say "training on $SHOP_BENCH_DATA ($(du -sh "$SHOP_BENCH_DATA" | cut -f1))"
else
  for f in pgo/*.tar.gz; do
    [ -f "$f" ] || { say "no training data in pgo/"; exit 1; }
    d=$W/data/$(basename "$f" .tar.gz)
    mkdir -p "$d"
    gzip -dc "$f" | tar -xf - -C "$d"
    [ -f "$d/manifest.json" ] && [ -d "$d/events" ] && [ -d "$d/clicks" ] || { say "bad dataset $f"; exit 1; }
    sets="$sets $d"
  done
  say "training on$(for d in $sets; do printf ' %s' "$(basename "$d")"; done)"
fi
for d in $sets; do
  for n in 2 8; do
    o=$W/out-$n; s=$W/scr-$n
    rm -rf "$o" "$s"; mkdir -p "$o" "$s"
    LLVM_PROFILE_FILE="$P/shop-%p.profraw" DATA_DIR=$d OUTPUT_DIR=$o SCRATCH_DIR=$s HOME=$s NCPU=$n "$GEN" >/dev/null
    [ -s "$o/campaign_daily/part-0.parquet" ] && ls "$o"/user_daily/*.parquet >/dev/null 2>&1 \
      || { say "training run wrote no output ($d NCPU=$n)"; exit 1; }
    rm -rf "$o" "$s"
  done
done
nraw=$(ls "$P" | grep -c '\.profraw$' || true)
[ "$nraw" -gt 0 ] || { say "training wrote no profile"; exit 1; }
step training

# 3. merge
M=$W/merged.profdata
"$PD" merge -o "$M" "$P"/*.profraw
nfun=$("$PD" show "$M" | sed -n 's/^Total functions: *//p')
nrun=$("$PD" show --covered "$M" | grep -c '^[^ ]' || true)
say "profile: $nraw raw files, $nfun functions, $nrun executed"
step merge

# 4. final build with the profile
L=$W/use.log
if ! cargo_build "$TD" "$FLAGS -Cprofile-use=$M -Cllvm-args=-pgo-warn-missing-function" >"$L" 2>&1; then
  cat "$L"; exit 1
fi
grep -v 'no profile data available for function' "$L" | grep -v '^ *Compiling ' || true
nmiss=$(grep -c 'no profile data available for function' "$L" || true)
nbad=$(grep -v 'no profile data available for function' "$L" | grep -Eic 'out of date|mismatch|inconsistent|malformed' || true)
grep 'no profile data available for function' "$L" | sed -n '1,10p'
pct() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.1f%%", (b > 0 ? 100 * a / b : 0) }'; }
say "pgo warnings: mismatch $nbad, missing $nmiss; share of functions with profile $(pct "$nfun" $((nfun + nmiss)));"   "executed in training $nrun of $nfun ($(pct "$nrun" "$nfun"))"
cp "$TD/$T/release/shop" "$OUT"
step final
say "total $(($(date +%s) - tall)) s"
