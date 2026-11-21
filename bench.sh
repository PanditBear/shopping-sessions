#!/bin/bash
# bench.sh REF... : build each ref as the scorer image would be built (static musl, x86-64-v3),
# then run them interleaved with the scorer's docker flags on a generated practice-size dataset.
# A ref may carry image ENV lines: REF@VAR=val@VAR2=val (same binary, different environment).
# Env: RUNS (timed runs per ref), CPUS, MEM, EVENTS, CLICKS, SEED, PROBE=1 (one SHOP_PROBE run per image),
#      PERF=1 (profile the first image's binary on the host with perf),
#      EXITPROBE=4 (what blocks after the container exits: stack samples, tmpfs output, cold cache),
#      EXITPROBE=3 (time after the container exits, 60M run vs empty dataset, with docker events),
#      EXITPROBE=2 (exit time of the first image under different docker cpu and memory limits),
#      EXITPROBE=1 (split the time after the binary's last tick: process teardown vs docker, plus a native run).
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
    RF=$(grep -o 'RUSTFLAGS="[^"]*"' $W/w$i/Dockerfile 2>/dev/null | head -1 | cut -d'"' -f2 | sed "s#/pgo/#$W/w$i/impl/rs/pgo/#")
    RF=${RF:-"-C target-cpu=x86-64-v3"}; echo "rustflags for $sha: $RF"
    (cd $W/w$i/impl/rs && CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="$RF" CARGO_TARGET_DIR=$W/tgt \
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
  for im in "${IMGS[@]}"; do echo "$im $(run $im $W/data)" | tee -a $W/times.txt
    echo "$im $(grep '^prof' $W/err.txt | grep -o '=[0-9.]*' | tr -d = | paste -sd+ | bc)" >> $W/profsum.txt; done
done
echo "== prof cpu-s sum per timed run (image, sum), run order"
cat $W/profsum.txt

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

if [ "${EXITPROBE:-}" = 1 ]; then
  echo "== exit probe: kernel and docker setup"
  set +e
  cat /proc/cmdline
  grep -E 'INIT_ON_(ALLOC|FREE)|^CONFIG_PREEMPT|TRANSPARENT_HUGEPAGE_(ALWAYS|MADVISE)' /boot/config-$(uname -r) 2>/dev/null
  docker info -f 'cgroup v{{.CgroupVersion}} driver {{.CgroupDriver}} storage {{.Driver}} runtime {{.DefaultRuntime}}'
  ep() { date -d "$1" +%s.%N; }
  echo "-- docker runs without --rm (s): create>started started>main end>finished finished>return rm"
  for im in "${IMGS[@]}"; do
    for k in 1 2 3; do
      out=$(mktemp -d); scr=$(mktemp -d); chmod 777 $out $scr; nm=xp-$im-$k
      t0=$(date +%s.%N)
      docker run --name $nm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
        --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 \
        --cpus $CPUS --memory $MEM --memory-swap $MEM --user 1000:1000 \
        -v $W/data:/data:ro -v $out:/output -v $scr:/scratch \
        -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch \
        -e NCPU=$CPUS -e SHOP_TIMING=1 $im 2>$W/xerr.txt
      t1=$(date +%s.%N)
      st=$(ep "$(docker inspect -f '{{.State.StartedAt}}' $nm)"); fa=$(ep "$(docker inspect -f '{{.State.FinishedAt}}' $nm)")
      t2=$(date +%s.%N); docker rm $nm >/dev/null; t3=$(date +%s.%N)
      em=$(grep -o 'main=[0-9.]*' $W/xerr.txt | cut -d= -f2); ee=$(grep -o 'end=[0-9.]*' $W/xerr.txt | cut -d= -f2)
      printf '%s %.3f %.3f %.3f %.3f %.3f\n' $im "$(echo "$st - $t0" | bc)" "$(echo "$em - $st" | bc)" \
        "$(echo "$fa - $ee" | bc)" "$(echo "$t1 - $fa" | bc)" "$(echo "$t3 - $t2" | bc)"
      sudo rm -rf $out $scr
    done
  done
  echo "-- native runs of ${IMGS[0]}'s binary (s): exec>main end>reaped wall"
  for k in 1 2 3; do
    o=$(mktemp -d); s2=$(mktemp -d)
    t0=$(date +%s.%N)
    env DATA_DIR=$W/data OUTPUT_DIR=$o SCRATCH_DIR=$s2 NCPU=$CPUS SHOP_TIMING=1 $BIN1 2>$W/xerr.txt
    t1=$(date +%s.%N)
    em=$(grep -o 'main=[0-9.]*' $W/xerr.txt | cut -d= -f2); ee=$(grep -o 'end=[0-9.]*' $W/xerr.txt | cut -d= -f2)
    printf 'native %.3f %.3f %.3f\n' "$(echo "$em - $t0" | bc)" "$(echo "$t1 - $ee" | bc)" "$(echo "$t1 - $t0" | bc)"
    rm -rf $o $s2
  done
  o=$(mktemp -d); s2=$(mktemp -d)
  env DATA_DIR=$W/data OUTPUT_DIR=$o SCRATCH_DIR=$s2 NCPU=$CPUS /usr/bin/time -v $BIN1 2>&1 | grep -E 'Maximum resident|faults|Elapsed|context switches'
  rm -rf $o $s2
  echo "-- kernel profile of a native run (all cpus, comm shop)"
  P=$(ls /usr/lib/linux-tools/*/perf 2>/dev/null | head -1)
  if [ -z "$P" ]; then
    sudo apt-get install -y -q --no-install-recommends linux-tools-common "linux-tools-$(uname -r)" >/dev/null 2>&1 \
      || sudo apt-get install -y -q --no-install-recommends linux-tools-common linux-tools-azure >/dev/null 2>&1
    P=$(ls /usr/lib/linux-tools/*/perf 2>/dev/null | head -1)
  fi
  if [ -n "$P" ]; then
    o=$(mktemp -d); s2=$(mktemp -d)
    sudo sysctl -q kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0
    sudo env DATA_DIR=$W/data OUTPUT_DIR=$o SCRATCH_DIR=$s2 NCPU=$CPUS $P record -q -a -F 2999 -e cpu-clock -o $W/xperf.data $BIN1 >/dev/null 2>&1
    sudo $P report -i $W/xperf.data --stdio --no-children --comms shop --sort sym --percent-limit 0.5 2>/dev/null | grep -v '^#' | grep -v '^$' | head -45
    sudo rm -rf $o $s2
  else echo "perf unavailable"; fi
  set -e
fi

if [ "${EXITPROBE:-}" = 2 ]; then
  echo "== exit probe 2: end>finished (s) per docker limit variant, first image"
  set +e
  ep() { date -d "$1" +%s.%N; }
  free -g | head -2
  for v in base nomem mem24 cpuset plain; do
    case $v in
      base) lim="--cpus $CPUS --memory $MEM --memory-swap $MEM";;
      nomem) lim="--cpus $CPUS";;
      mem24) lim="--cpus $CPUS --memory 24g --memory-swap 24g";;
      cpuset) lim="--cpuset-cpus 0-$((CPUS-1)) --memory $MEM --memory-swap $MEM";;
      plain) lim="";;
    esac
    for k in 1 2 3; do
      out=$(mktemp -d); scr=$(mktemp -d); chmod 777 $out $scr; nm=xq-$v-$k
      docker run --name $nm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
        --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 $lim --user 1000:1000 \
        -v $W/data:/data:ro -v $out:/output -v $scr:/scratch \
        -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch \
        -e NCPU=$CPUS -e SHOP_TIMING=1 ${IMGS[0]} 2>$W/xerr.txt
      st=$(ep "$(docker inspect -f '{{.State.StartedAt}}' $nm)"); fa=$(ep "$(docker inspect -f '{{.State.FinishedAt}}' $nm)")
      docker rm $nm >/dev/null
      em=$(grep -o 'main=[0-9.]*' $W/xerr.txt | cut -d= -f2); ee=$(grep -o 'end=[0-9.]*' $W/xerr.txt | cut -d= -f2)
      printf 'xq %s internal %.3f exit %.3f\n' $v "$(echo "$ee - $em" | bc)" "$(echo "$fa - $ee" | bc)"
      sudo rm -rf $out $scr
    done
  done
  set -e
fi

if [ "${EXITPROBE:-}" = 3 ]; then
  echo "== exit probe 3 (s): variant end>finished finished>die finished>return rm"
  set +e
  ep() { date -d "$1" +%s.%N; }
  for v in base60 empty plain60 base60 empty; do
    case $v in
      base60) lim="--cpus $CPUS --memory $MEM --memory-swap $MEM"; d=$W/data;;
      plain60) lim=""; d=$W/data;;
      empty) lim="--cpus $CPUS --memory $MEM --memory-swap $MEM"; d=$W/empty;;
    esac
    for k in 1 2 3; do
      out=$(mktemp -d); scr=$(mktemp -d); chmod 777 $out $scr; nm=xr-$v-$k-$RANDOM
      t0=$(date +%s)
      docker run --name $nm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
        --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 $lim --user 1000:1000 \
        -v $d:/data:ro -v $out:/output -v $scr:/scratch \
        -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch \
        -e NCPU=$CPUS -e SHOP_TIMING=1 ${IMGS[0]} 2>$W/xerr.txt
      t1=$(date +%s.%N)
      fa=$(ep "$(docker inspect -f '{{.State.FinishedAt}}' $nm)")
      die=$(docker events --since $t0 --until $(date +%s) --filter container=$nm --filter event=die --format '{{.TimeNano}}' | head -1)
      t2=$(date +%s.%N); docker rm $nm >/dev/null; t3=$(date +%s.%N)
      ee=$(grep -o 'end=[0-9.]*' $W/xerr.txt | cut -d= -f2); [ -z "$ee" ] && ee=$fa
      printf 'xr %s %.3f %.3f %.3f %.3f\n' $v "$(echo "$fa - $ee" | bc)" "$(echo "${die:-0}/1000000000 - $fa" | bc -l)" \
        "$(echo "$t1 - $fa" | bc)" "$(echo "$t3 - $t2" | bc)"
      sudo rm -rf $out $scr
    done
  done
  set -e
fi

if [ "${EXITPROBE:-}" = 4 ]; then
  echo "== exit probe 4 (s): variant end>finished finished>return"
  set +e
  ep() { date -d "$1" +%s.%N; }
  cat > $W/samp.sh <<'SAMP'
#!/bin/bash
# sample kernel stacks of non-shop tasks that are blocked or running, only while no shop process exists;
# stops when the file named by $1 appears
while [ ! -f "$1" ]; do
  if ! pgrep -x shop >/dev/null; then
    for d in /proc/[0-9]*; do
      st=$(awk '{print $3}' $d/stat 2>/dev/null); [ "$st" = D ] || [ "$st" = R ] || continue
      c=$(cat $d/comm 2>/dev/null); case $c in samp.sh|awk|cat|pgrep|bash|sleep|head|cut|tr) continue;; esac
      k=$(head -6 $d/stack 2>/dev/null | awk '{print $2}' | cut -d+ -f1 | tr '\n' '<')
      echo "$st $c $k"
    done
  fi
  sleep 0.01
done
SAMP
  chmod +x $W/samp.sh
  for v in base60 tmpout60 cold60 base60; do
    for k in 1 2 3; do
      out=$(mktemp -d); scr=$(mktemp -d); chmod 777 $out $scr; nm=xs-$v-$k-$RANDOM
      [ $v = tmpout60 ] && sudo mount -t tmpfs -o size=4g,mode=777 tmpfs $out
      [ $v = cold60 ] && { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }
      sf=$W/stop-$v-$k-$RANDOM; sudo $W/samp.sh $sf > $W/samp-$v-$k.txt 2>/dev/null &
      docker run --name $nm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
        --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 \
        --cpus $CPUS --memory $MEM --memory-swap $MEM --user 1000:1000 \
        -v $W/data:/data:ro -v $out:/output -v $scr:/scratch \
        -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch \
        -e NCPU=$CPUS -e SHOP_TIMING=1 ${IMGS[0]} 2>$W/xerr.txt
      t1=$(date +%s.%N); touch $sf; wait
      fa=$(ep "$(docker inspect -f '{{.State.FinishedAt}}' $nm)"); docker rm $nm >/dev/null
      ee=$(grep -o 'end=[0-9.]*' $W/xerr.txt | cut -d= -f2)
      printf 'xs %s %.3f %.3f samples %s\n' $v "$(echo "$fa - $ee" | bc)" "$(echo "$t1 - $fa" | bc)" "$(wc -l < $W/samp-$v-$k.txt)"
      [ $v = tmpout60 ] && sudo umount $out
      sudo rm -rf $out $scr
    done
  done
  echo "-- top blocked/running stacks after exit, base60 runs (count state comm stack)"
  cat $W/samp-base60-*.txt | sort | uniq -c | sort -rn | head -25
  echo "-- same, cold60"
  cat $W/samp-cold60-*.txt | sort | uniq -c | sort -rn | head -12
  echo "-- same, tmpout60"
  cat $W/samp-tmpout60-*.txt | sort | uniq -c | sort -rn | head -12
  set -e
fi
