#!/bin/bash
# job3.sh: --memory sweep at 60M, first-run penalty check
set -uo pipefail
PICK=5ae524bd8bb1440966d8e6a07d11c3b91f057b9e; REF=3067223
W=/tmp/r1; mkdir -p $W
S() { echo "$*" | tee -a $W/summary.txt; }
wipe() { for p in "$@"; do [ -e "$p" ] && sudo find "$p" -delete 2>/dev/null; done; return 0; }
S "cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"; free -g | head -2; grep MemTotal /proc/meminfo | tee -a $W/summary.txt
MT=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
if [ "$MT" -ge 26 ]; then LIMS="12g 14g 24g"; else LIMS="10g 12g 14g"; fi; S "limits: $LIMS (MemTotal ${MT} GiB)"
sudo apt-get update -q >/dev/null && sudo apt-get install -y -q --no-install-recommends musl-tools bc binutils >/dev/null
rustup target add x86_64-unknown-linux-musl >/dev/null 2>&1
git worktree add -q --detach $W/src $PICK; git worktree add -q --detach $W/refsrc $REF
docker build --no-cache --pull -q -t r1img $W/src >/dev/null 2>&1 || { S "BUILD FAILED"; exit 1; }
(cd $W/refsrc/impl/rs && CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_TARGET_DIR=$W/tgt cargo build --release --locked -q --target x86_64-unknown-linux-musl) && cp $W/tgt/x86_64-unknown-linux-musl/release/shop $W/shop-ref || { S "REF BUILD FAILED"; exit 1; }
curl -sSfL -o $W/gen.zip https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/generator.zip
echo "78f973f75d14b48e792628284f19c75de71fa597d20952e16a9ced93c78bee4a  $W/gen.zip" | sha256sum -c --quiet || exit 1
(cd $W && unzip -q -o gen.zip && cd generator && cargo build --release -q)
G=$(find $W -path '*release/generate' -type f | head -1)
python3 -m pip install -q duckdb; git show origin/bench:content.py > $W/content.py
$G --seed 7 --events 60000000 --clicks 15000000 --output $W/data --threads 4 >/dev/null; S "dataset: $(du -sh $W/data | cut -f1)"
refout=$(mktemp -d); chmod 777 $refout; rs=$(mktemp -d); chmod 777 $rs
DATA_DIR=$W/data OUTPUT_DIR=$refout SCRATCH_DIR=$rs NCPU=4 $W/shop-ref >/dev/null 2>&1; S "ref done rc=$?"
run_img() { # LIMIT OUTDIR [STATTAG]
  local lim=$1 out=$2 scr nm=r1c-$RANDOM t0 t1 pid id cg v
  scr=$(mktemp -d); chmod 777 $out $scr; echo 0 > $W/peak.txt; : > $W/stat.txt; t0=$(date +%s.%N)
  docker run --name $nm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 \
    --cpus 4 --memory $lim --memory-swap $lim --user 1000:1000 -v $W/data:/data:ro -v $out:/output -v $scr:/scratch \
    -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch -e NCPU=4 -e SHOP_TIMING=1 r1img 2>/dev/null &
  pid=$!
  ( while kill -0 $pid 2>/dev/null; do
      id=$(docker inspect -f '{{.Id}}' $nm 2>/dev/null)
      if [ -n "$id" ]; then cg=/sys/fs/cgroup/system.slice/docker-$id.scope
        if v=$(cat $cg/memory.peak 2>/dev/null) && [ -n "$v" ]; then echo $v > $W/peak.tmp && mv -f $W/peak.tmp $W/peak.txt; fi
        if cat $cg/memory.stat > $W/stat.tmp 2>/dev/null; then mv -f $W/stat.tmp $W/stat.txt; fi
        cat $cg/memory.events > $W/ev.tmp 2>/dev/null && mv -f $W/ev.tmp $W/ev.txt
      fi; sleep 0.02; done ) &
  wait $pid; RUN_RC=$?; wait; t1=$(date +%s.%N); docker rm $nm >/dev/null 2>&1
  RUN_WALL=$(printf '%.3f' $(echo "$t1 - $t0" | bc)); pk=$(cat $W/peak.txt 2>/dev/null); RUN_PEAK_MB=$(( ${pk:-0}/1048576 )); wipe $scr
}
med() { sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
for l in $LIMS; do : > $W/t-$l.txt; done
for l in $LIMS; do o=$(mktemp -d); run_img $l $o; S "warmup $l rc=$RUN_RC wall=$RUN_WALL peakMB=$RUN_PEAK_MB"
  if [ "$l" = "$(echo $LIMS | cut -d' ' -f1)" ]; then cp $W/stat.txt $W/stat-warm-$l.txt; fi
  if python3 $W/content.py $refout $o > $W/c.txt 2>&1; then S "limit $l content: CONTENT SAME vs 3067223"; else S "limit $l content: CONTENT DIFFERENT"; fi; wipe $o; done
FIRST=$(echo $LIMS | cut -d' ' -f1)
for r in 1 2 3; do for l in $LIMS; do o=$(mktemp -d); run_img $l $o; echo $RUN_WALL >> $W/t-$l.txt; S "round$r $l rc=$RUN_RC wall=$RUN_WALL peakMB=$RUN_PEAK_MB reclaim_events: $(tr '\n' ' ' < $W/ev.txt)"
  if [ "$r" = 1 ] && [ "$l" = "$FIRST" ]; then cp $W/stat.txt $W/stat-r1.txt; fi; wipe $o; done; done
for l in $LIMS; do S "MEDIAN $l wall=$(med < $W/t-$l.txt) runs: $(tr '\n' ' ' < $W/t-$l.txt)"; done
S "---- memory.stat of $FIRST run (last sample before exit, round1) ----"; grep -E "^(anon|file|shmem|pgscan|pgsteal|pgfault|pgmajfault|workingset_refault|workingset_activate|pgrefill|inactive_file|active_file|file_mapped) " $W/stat-r1.txt | tee -a $W/summary.txt
S "all pg/workingset lines:"; grep -E "pgscan|pgsteal|workingset|pgmajfault" $W/stat-r1.txt | tee -a $W/summary.txt
S "== done"
echo "---- SUMMARY ----"; cat $W/summary.txt; { echo '```'; cat $W/summary.txt; echo '```'; } >> $GITHUB_STEP_SUMMARY
