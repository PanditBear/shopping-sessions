#!/bin/bash
# bench.sh REF... : build each ref as the scorer image would be built (static musl, x86-64-v3),
# then run them interleaved with the scorer's docker flags on a generated practice-size dataset.
# Env: RUNS (timed runs per ref), CPUS, MEM, EVENTS, CLICKS, SEED.
set -euo pipefail
RUNS=${RUNS:-5}; CPUS=${CPUS:-4}; MEM=${MEM:-14g}
EVENTS=${EVENTS:-20000000}; CLICKS=${CLICKS:-5000000}; SEED=${SEED:-7}
W=/tmp/bench; mkdir -p $W/img $W/empty
echo "== machine"; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|L2|L3' ; free -g | head -2; uname -r

echo "== generator (organizers' kit)"
curl -sSfL -o $W/gen.zip https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/generator.zip
(cd $W && unzip -q -o gen.zip && cd generator && cargo build --release -q)
G=$(find $W -path '*release/generate' -type f | head -1)
t=$(date +%s.%N); $G --seed $SEED --events $EVENTS --clicks $CLICKS --output $W/data --threads 4 >/dev/null
echo "generated in $(echo "$(date +%s.%N) - $t" | bc) s: $(du -sh $W/data | cut -f1)"
printf '{\n  "window_start": "2028-01-23T00:00:00Z",\n  "window_end": "2028-02-06T00:00:00Z"\n}\n' > $W/empty/manifest.json

i=0; IMGS=()
for r in "$@"; do
  i=$((i+1)); git fetch -q origin "$r" 2>/dev/null || true
  rm -rf $W/w$i; git worktree add -q --detach $W/w$i "$r"
  echo "== build $i: $r ($(git -C $W/w$i log -1 --format='%h %s'))"
  (cd $W/w$i/impl/rs && CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_TARGET_DIR=$W/tgt \
     cargo build --release --locked -q --target x86_64-unknown-linux-musl)
  cp $W/tgt/x86_64-unknown-linux-musl/release/shop $W/img/shop
  printf 'FROM scratch\nCOPY shop /app/shop\nCMD ["/app/shop"]\n' > $W/img/Dockerfile
  docker build -q -t bench$i $W/img >/dev/null
  IMGS+=("bench$i")
done

run() { # image datadir -> "wall internal"
  local out scr t0 t1
  out=$(mktemp -d); scr=$(mktemp -d); chmod 777 $out $scr
  t0=$(date +%s.%N)
  docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 \
    --cpus $CPUS --memory $MEM --memory-swap $MEM --user 1000:1000 \
    -v $2:/data:ro -v $out:/output -v $scr:/scratch \
    -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch \
    -e NCPU=$CPUS -e SHOP_TIMING=1 $1 2>$W/err.txt
  t1=$(date +%s.%N)
  cp $W/err.txt $W/err-$1.txt
  printf '%.3f %s\n' "$(echo "$t1 - $t0" | bc)" "$(grep written $W/err.txt | tr -d '[]s' | awk '{print $1}')"
  if [ "${KEEP:-}" = "$1" ]; then rm -rf $W/keep-$1; cp -r $out $W/keep-$1; fi
  sudo rm -rf $out $scr
}

echo "== warm-up and empty-input overhead"
for im in "${IMGS[@]}"; do run $im $W/data >/dev/null; done
for k in 1 2 3; do echo "empty ${IMGS[0]}: $(run ${IMGS[0]} $W/empty)"; done

echo "== timed (wall internal), interleaved"
for k in $(seq $RUNS); do
  for im in "${IMGS[@]}"; do echo "$im $(run $im $W/data)" | tee -a $W/times.txt; done
done

echo "== medians"
for im in "${IMGS[@]}"; do
  w=$(grep "^$im " $W/times.txt | awk '{print $2}' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  n=$(grep "^$im " $W/times.txt | awk '{print $3}' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  echo "$im wall $w internal $n"
done | tee -a ${GITHUB_STEP_SUMMARY:-/dev/null}

echo "== last-run phase ticks"
for im in "${IMGS[@]}"; do echo "-- $im"; cat $W/err-$im.txt; done
