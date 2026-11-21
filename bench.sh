#!/bin/bash
# bench.sh REF... : build each ref as the scorer image would be built (static musl, x86-64-v3),
# then run them interleaved with the scorer's docker flags on a generated practice-size dataset.
# A ref may carry image ENV lines: REF@VAR=val@VAR2=val (same binary, different environment).
# Env: RUNS (timed runs per ref), CPUS, MEM, EVENTS, CLICKS, SEED, PROBE=1 (one SHOP_PROBE run per image),
#      PERF=1 (profile the first image's binary on the host with perf).
set -euo pipefail
RUNS=${RUNS:-5}; CPUS=${CPUS:-4}; MEM=${MEM:-14g}
EVENTS=${EVENTS:-20000000}; CLICKS=${CLICKS:-5000000}; SEED=${SEED:-7}
W=/tmp/bench; mkdir -p $W/img $W/empty
echo "== machine"; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|L2|L3' ; free -g | head -2; uname -r
for f in enabled defrag; do echo "thp $f: $(cat /sys/kernel/mm/transparent_hugepage/$f)"; done

echo "== generator (organizers' kit)"
curl -sSfL -o $W/gen.zip https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/generator.zip
echo "13724cadbac5a99d847e27cea911cd2e338ff972b46c2039da87452011bb101e  $W/gen.zip" | sha256sum -c --quiet
(cd $W && unzip -q -o gen.zip && cd generator && cargo build --release -q)
G=$(find $W -path '*release/generate' -type f | head -1)
t=$(date +%s.%N); $G --seed $SEED --events $EVENTS --clicks $CLICKS --output $W/data --threads 4 >/dev/null
echo "generated in $(echo "$(date +%s.%N) - $t" | bc) s: $(du -sh $W/data | cut -f1)"
printf '{\n  "window_start": "2028-01-23T00:00:00Z",\n  "window_end": "2028-02-06T00:00:00Z"\n}\n' > $W/empty/manifest.json

i=0; IMGS=()
for spec in "$@"; do
  i=$((i+1)); r=${spec%%@*}; envs=""; [ "$spec" != "$r" ] && envs=${spec#*@}
  git fetch -q origin "$r" 2>/dev/null || true
  sha=$(git rev-parse --short "$r^{commit}" 2>/dev/null || git rev-parse --short "origin/$r")
  if [ ! -f $W/bin-$sha ]; then
    rm -rf $W/w$i; git worktree add -q --detach $W/w$i "$sha"
    (cd $W/w$i/impl/rs && CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_TARGET_DIR=$W/tgt \
       cargo build --release --locked -q --target x86_64-unknown-linux-musl)
    cp $W/tgt/x86_64-unknown-linux-musl/release/shop $W/bin-$sha
  fi
  echo "== image bench$i: $sha ($(git log -1 --format=%s $sha)) env [${envs//@/ }]"
  cp $W/bin-$sha $W/img/shop
  { printf 'FROM scratch\nCOPY shop /app/shop\n'; for e in ${envs//@/ }; do echo "ENV $e"; done; printf 'CMD ["/app/shop"]\n'; } > $W/img/Dockerfile
  docker build -q -t bench$i $W/img >/dev/null
  IMGS+=("bench$i"); [ $i = 1 ] && BIN1=$W/bin-$sha
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
    -e NCPU=$CPUS -e SHOP_TIMING=1 ${EXTRA:-} $1 2>$W/err.txt
  t1=$(date +%s.%N)
  cp $W/err.txt $W/err-$1.txt
  local em ee; em=$(grep -o 'main=[0-9.]*' $W/err.txt | cut -d= -f2 || true); ee=$(grep -o 'end=[0-9.]*' $W/err.txt | cut -d= -f2 || true)
  printf '%.3f %s %.3f %.3f\n' "$(echo "$t1 - $t0" | bc)" "$(grep written $W/err.txt | tr -d '[]s' | awk '{print $1}')" \
    "$(echo "${em:-$t0} - $t0" | bc)" "$(echo "$t1 - ${ee:-$t1}" | bc)"
  if [ "${KEEP:-}" = "$1" ]; then rm -rf $W/keep-$1; cp -r $out $W/keep-$1; fi
  sudo rm -rf $out $scr
}

echo "== warm-up and empty-input overhead"
for im in "${IMGS[@]}"; do run $im $W/data >/dev/null; done
for k in 1 2 3; do echo "empty ${IMGS[0]}: $(run ${IMGS[0]} $W/empty)"; done

echo "== timed (wall internal start exit), interleaved"
for k in $(seq $RUNS); do
  for im in "${IMGS[@]}"; do echo "$im $(run $im $W/data)" | tee -a $W/times.txt; done
done

echo "== medians"
for im in "${IMGS[@]}"; do
  w=$(grep "^$im " $W/times.txt | awk '{print $2}' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  n=$(grep "^$im " $W/times.txt | awk '{print $3}' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  pr=$(grep "^$im " $W/times.txt | awk '{print $4}' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  po=$(grep "^$im " $W/times.txt | awk '{print $5}' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  echo "$im wall $w internal $n start $pr exit $po"
done | tee -a ${GITHUB_STEP_SUMMARY:-/dev/null}

echo "== output parity vs ${IMGS[0]}"
for im in "${IMGS[@]}"; do KEEP=$im run $im $W/data >/dev/null; done
for im in "${IMGS[@]:1}"; do
  if diff -r -q $W/keep-${IMGS[0]} $W/keep-$im >/dev/null; then echo "$im same"; else echo "$im DIFFERENT"; fi
done | tee -a ${GITHUB_STEP_SUMMARY:-/dev/null}

echo "== last-run phase ticks"
for im in "${IMGS[@]}"; do echo "-- $im"; cat $W/err-$im.txt; done

if [ "${PROBE:-}" = 1 ]; then
  echo "== probe (serial component timings, one run per image)"
  for im in "${IMGS[@]}"; do EXTRA="-e SHOP_PROBE=1" run $im $W/data >/dev/null; echo "-- $im"; grep probe $W/err-$im.txt; done
fi

if [ "${PERF:-}" = 1 ]; then
  echo "== perf on the host, binary of ${IMGS[0]}"
  set +e
  sudo apt-get install -y -q --no-install-recommends linux-tools-common "linux-tools-$(uname -r)" >/dev/null 2>&1 \
    || sudo apt-get install -y -q --no-install-recommends linux-tools-common linux-tools-azure >/dev/null 2>&1
  P=$(ls /usr/lib/linux-tools/*/perf 2>/dev/null | head -1)
  if [ -z "$P" ]; then echo "perf unavailable"; else
    o=$(mktemp -d); s2=$(mktemp -d)
    E="DATA_DIR=$W/data OUTPUT_DIR=$o SCRATCH_DIR=$s2 NCPU=$CPUS SHOP_TIMING=1"
    sudo env $E $BIN1 2>&1 | tail -3
    sudo env $E $P stat -e task-clock,page-faults,context-switches,cpu-migrations,cycles,instructions,cache-misses $BIN1 2>&1 | grep -v '^\[' | tail -14
    sudo env $E $P record -q -F 4999 -e cpu-clock -o $W/perf.data $BIN1 >/dev/null 2>&1
    sudo $P report -i $W/perf.data --stdio --no-children --sort sym --percent-limit 0.4 2>/dev/null | grep -v '^#' | grep -v '^$' | head -60
    sudo $P report -i $W/perf.data --stdio --no-children --sort dso --percent-limit 0.5 2>/dev/null | grep -v '^#' | grep -v '^$' | head -10
  fi
  set -e
fi
