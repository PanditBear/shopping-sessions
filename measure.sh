#!/bin/bash
# measure.sh : per-run wall, cpu, memory for our ref and public repositories built from their own Dockerfile.
# Env: SPECS (first = our ref; others ext=<git url>#<sha>), SIZES ("events:clicks:runs ..."), SEED, NEEDAVX (1 = stop if no avx512),
#      CPUS, MEM.  Every container: --network none, data read-only, same cpu and memory limits.
# Nothing is deleted: the runner is thrown away after the job, work area is /mnt (large disk).
set -uo pipefail
CPUS=${CPUS:-4}; MEM=${MEM:-14g}; SEED=${SEED:-11}
HERE=$(pwd)
sudo mkdir -p /mnt/bench && sudo chown "$(id -u)" /mnt/bench
W=/mnt/bench; mkdir -p $W/img; R=$W/res.tsv; : > $R
echo "== machine"; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|L2|L3|Hypervisor'; free -m | head -2; uname -r
for f in enabled defrag; do echo "thp $f: $(cat /sys/kernel/mm/transparent_hugepage/$f)"; done
FL=$(grep -m1 '^flags' /proc/cpuinfo); AVX=1
for f in avx512f avx512bw avx512vl avx512cd avx2 bmi2; do
  if echo "$FL" | grep -qw $f; then echo "cpu flag $f: yes"; else echo "cpu flag $f: NO"; case $f in avx512*) AVX=0;; esac; fi
done
echo "AVX512_ALL=$AVX"
docker info -f 'cgroup v{{.CgroupVersion}} driver {{.CgroupDriver}} storage {{.Driver}}'
df -h /mnt | tail -1
if [ "${NEEDAVX:-1}" = 1 ] && [ $AVX = 0 ]; then echo "NOAVX512: stopping so the run can be dispatched again"; exit 0; fi

# ---- build images
NAMES=(); KINDS=(); i=0
for spec in $SPECS; do
  i=$((i+1))
  if [[ "$spec" == ext=* ]]; then
    u=${spec#ext=}; url=${u%%#*}; xsha=${u#*#}
    X=$(mktemp -d $W/xXXXXXX)
    if ! { git clone -q "$url" $X/src && git -C $X/src checkout -q "$xsha"; }; then echo "CLONE FAILED $url"; continue; fi
    echo "== image m$i: external $url $(git -C $X/src rev-parse HEAD)"
    t=$(date +%s.%N)
    if docker build -q -t m$i $X/src > $W/build-$i.log 2>&1; then
      echo "built in $(echo "$(date +%s.%N) - $t" | bc) s"; NAMES+=(m$i)
      case "$url" in *express-lane*) KINDS+=(xl);; *matthewf725*) KINDS+=(mf);; *) KINDS+=(other);; esac
    else echo "BUILD FAILED for $url"; tail -40 $W/build-$i.log; fi
    continue
  fi
  sha=$(git rev-parse --short "$spec^{commit}")
  git worktree add -q --detach $W/w$i "$sha"
  if [ -f $W/w$i/impl/rs/build.sh ]; then echo "note: ref has build.sh (not used by this script)"; fi
  (cd $W/w$i/impl/rs && CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_TARGET_DIR=$W/tgt \
     cargo build --release --locked -q --target x86_64-unknown-linux-musl) || { echo "OUR BUILD FAILED"; exit 1; }
  cp $W/tgt/x86_64-unknown-linux-musl/release/shop $W/img/shop
  printf 'FROM scratch\nCOPY shop /app/shop\nCMD ["/app/shop"]\n' > $W/img/Dockerfile
  docker build -q -t m$i $W/img >/dev/null; NAMES+=(m$i); KINDS+=(ours); echo "== image m$i: ours $sha"
done
echo "images: ${NAMES[*]} kinds: ${KINDS[*]}"

echo "== generator"
curl -sSfL -o $W/gen.zip https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/generator.zip
sha256sum $W/gen.zip
(cd $W && unzip -q -o gen.zip && cd generator && cargo build --release -q)
G=$(find $W -path '*release/generate*' -type f -perm -u+x ! -name '*.d' | head -1); echo "generator binary: $G"
python3 -m pip install -q duckdb 2>&1 | tail -1
sudo apt-get install -y -q --no-install-recommends linux-tools-common "linux-tools-$(uname -r)" >/dev/null 2>&1 \
  || sudo apt-get install -y -q --no-install-recommends linux-tools-common linux-tools-azure >/dev/null 2>&1
P=$(ls /usr/lib/linux-tools/*/perf 2>/dev/null | head -1); echo "perf=${P:-none}"
sudo sysctl -q kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0 || true

# run LABEL SIZE N IMAGE "docker env args" -- cmd args...
run() {
  local label=$1 size=$2 n=$3 im=$4 envs=$5; shift 6
  local out scr cid t0 t1 ec nm pp ob nf
  out=$W/o-$label-$size-$n; scr=$W/s-$label-$size-$n; mkdir -p $out $scr; chmod 777 $out $scr
  nm=mr-$RANDOM
  cid=$(docker create --name $nm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 \
    --cpus $CPUS --memory $MEM --memory-swap $MEM --user 1000:1000 \
    -v $DATA:/data:ro -v $out:/output -v $scr:/scratch \
    -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch -e NCPU=$CPUS $envs $im "$@")
  python3 $HERE/measure_poll.py $cid > $W/poll.json &
  pp=$!
  t0=$(date +%s.%N); docker start $cid >/dev/null
  timeout 900 docker wait $cid > $W/ec.txt || docker kill $cid >/dev/null 2>&1
  t1=$(date +%s.%N); ec=$(cat $W/ec.txt 2>/dev/null || echo 124)
  docker logs $cid > /dev/null 2> $W/err-$label-$size-$n.txt
  docker rm -f $cid >/dev/null; wait $pp
  ob=$(du -sb $out | cut -f1); nf=$(find $out -type f | wc -l)
  python3 - "$label" "$size" "$n" "$(echo "$t1 - $t0" | bc)" "$ob" "$nf" "$ec" "$W" >> $R <<'PY'
import sys, json
label, size, n, wall, ob, nf, ec, w = sys.argv[1:9]
p = json.load(open(w + "/poll.json"))
print(size, label, n, wall, p["user_us"]/1e6, p["sys_us"]/1e6, p["anon_max"], p["peak"], p["file_max"], ob, nf, ec)
PY
  tail -1 $R
}

for spec in $SIZES; do
  IFS=: read EV CK RUNS <<< "$spec"
  echo; echo "################ events=$EV clicks=$CK runs=$RUNS seed=$SEED"
  DATA=$W/data-$EV; t=$(date +%s.%N)
  $G --seed $SEED --events $EV --clicks $CK --output $DATA --threads 4 >/dev/null
  echo "generated in $(echo "$(date +%s.%N) - $t" | bc) s: $(du -sh $DATA | cut -f1)"; df -h /mnt | tail -1
  SZ=$EV
  echo "== warm-up (outputs kept for content check)"
  for k in "${!NAMES[@]}"; do run ${NAMES[$k]} $SZ warm ${NAMES[$k]} "" --; done
  echo "== content check vs first program (rows as multisets)"
  K=(); for im in "${NAMES[@]}"; do K+=("$W/o-$im-$SZ-warm"); done
  python3 $HERE/content.py "${K[@]}"; echo "content exit $?"
  echo "== timed, interleaved"
  for n in $(seq $RUNS); do
    for im in "${NAMES[@]}"; do run $im $SZ $n $im "" --; done
  done
  echo "== phase and path diagnostics (not in the timed set)"
  for k in "${!NAMES[@]}"; do
    im=${NAMES[$k]}; kind=${KINDS[$k]}
    case $kind in
      ours) run ${im}_timing $SZ 1 $im "-e SHOP_TIMING=1" --;;
      xl)
        run ${im}_check $SZ 1 $im "-e XL_TIMING=1 -e XL_CHECK=1" --
        run ${im}_v3 $SZ 1 $im "-e XL_V3=1" --
        run ${im}_v3_timing $SZ 1 $im "-e XL_V3=1 -e XL_TIMING=1 -e XL_CHECK=1" --
        run ${im}_generic $SZ 1 $im "-e V1ND_LOG=1" /pipeline --generic;;
      mf) run ${im}_profile $SZ 1 $im "-e PROFILE=1" --;;
    esac
  done
  for l in $(ls $W/err-*-$SZ-*.txt | grep -E '_(timing|check|v3|v3_timing|generic|profile)-'); do echo "---- $l"; head -c 6000 $l; done
  echo "== stderr of default runs mentioning FALLBACK / panic / v3b / Illegal"
  grep -l -E 'FALLBACK|panic|v3b|Illegal' $W/err-m*-$SZ-[0-9wm]*.txt 2>/dev/null | head; echo "(end)"
  if [ -n "$P" ]; then
    echo "== perf stat, system wide around one docker run per program"
    for k in "${!NAMES[@]}"; do
      im=${NAMES[$k]}; o=$W/po-$im-$SZ; s2=$W/ps-$im-$SZ; mkdir -p $o $s2; chmod 777 $o $s2
      echo "-- $im"
      sudo $P stat -a -e task-clock,cycles,instructions,cache-misses,page-faults -- docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
        --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 --cpus $CPUS --memory $MEM --memory-swap $MEM --user 1000:1000 \
        -v $DATA:/data:ro -v $o:/output -v $s2:/scratch -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch -e NCPU=$CPUS $im 2>&1 >/dev/null | grep -v '^\[' | tail -14
    done
  fi
  echo "== summary at events=$EV"; grep "^$EV " $R > $W/res-$EV.tsv; python3 $HERE/measure_sum.py $W/res-$EV.tsv
done
echo "== ALL RESULT LINES (size label n wall user sys anon_B peak_B file_B outbytes files exit)"; cat $R
