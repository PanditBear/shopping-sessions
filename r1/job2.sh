#!/bin/bash
# job2.sh: two independent --no-cache builds on one runner, binary diff, alternating timing, peak memory
set -uo pipefail
PICK=5ae524bd8bb1440966d8e6a07d11c3b91f057b9e
W=/tmp/r1; mkdir -p $W
S() { echo "$*" | tee -a $W/summary.txt; }
wipe() { for p in "$@"; do [ -e "$p" ] && sudo find "$p" -delete 2>/dev/null; done; return 0; }
S "cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"; free -g | head -2
sudo apt-get update -q >/dev/null && sudo apt-get install -y -q --no-install-recommends musl-tools bc binutils time >/dev/null
git worktree add -q --detach $W/src $PICK
# (1) builds
HS=(); PA=""; PB=""
for i in 1 2 3 4; do
  docker build --no-cache --pull -q -t r1img$i $W/src >/dev/null 2>&1 || { S "build $i FAILED"; continue; }
  cid=$(docker create r1img$i); docker cp $cid:/app/shop $W/bin$i; docker rm $cid >/dev/null
  h=$(sha256sum $W/bin$i | cut -d' ' -f1); S "build $i sha256 $h size $(stat -c %s $W/bin$i)"; HS[$i]=$h
  if [ $i -ge 2 ]; then
    for j in $(seq 1 $((i-1))); do [ "${HS[$j]}" != "$h" ] && [ -z "$PA" ] && { PA=$j; PB=$i; }; done
  fi
  [ -n "$PA" ] && break
done
if [ -z "$PA" ]; then S "ALL BUILDS IDENTICAL; using builds 1 and 2 as A and B"; PA=1; PB=2; else S "differing pair: build $PA = A, build $PB = B"; fi
docker tag r1img$PA r1A; docker tag r1img$PB r1B
A=$W/bin$PA; B=$W/bin$PB
# (2) diff
size -A $A > $W/sizeA.txt; size -A $B > $W/sizeB.txt
if diff $W/sizeA.txt $W/sizeB.txt > $W/sizediff.txt; then S "section sizes identical"; else S "section sizes differ:"; tee -a $W/summary.txt < $W/sizediff.txt; fi
S "text size A: $(grep '^\.text' $W/sizeA.txt)  B: $(grep '^\.text' $W/sizeB.txt)"
S "build-id A: $(readelf -n $A | grep -i 'Build ID')  B: $(readelf -n $B | grep -i 'Build ID')"
S "total differing bytes (cmp -l): $(cmp -l $A $B | wc -l)"
S "first diffs (offset 1-based, octal A, octal B):"; cmp -l $A $B | head -12 | tee -a $W/summary.txt
for s in .text .rodata .data .data.rel.ro .eh_frame .gcc_except_table .symtab .strtab; do
  objcopy -O binary --only-section=$s $A $W/a.sec 2>/dev/null; objcopy -O binary --only-section=$s $B $W/b.sec 2>/dev/null
  if [ -s $W/a.sec ] || [ -s $W/b.sec ]; then
    if cmp -s $W/a.sec $W/b.sec; then S "section $s: same bytes ($(stat -c %s $W/a.sec))"; else S "section $s: DIFFERENT bytes A=$(stat -c %s $W/a.sec) B=$(stat -c %s $W/b.sec) ndiff=$(cmp -l $W/a.sec $W/b.sec 2>/dev/null | wc -l)"; fi
  fi
done
S "first differing byte: $(cmp $A $B)"; S "section map:"; readelf -S -W $A | grep -E '^ +\[' | head -40 | tee -a $W/summary.txt
strings -a -n 6 $A > $W/sa.txt; strings -a -n 6 $B > $W/sb.txt; S "strings diff (A<, B>):"; diff $W/sa.txt $W/sb.txt | head -20 | tee -a $W/summary.txt
S "symbol size diff:"; nm -S $A 2>/dev/null | awk '{print $2,$4}' | sort > $W/na.txt; nm -S $B 2>/dev/null | awk '{print $2,$4}' | sort > $W/nb.txt; diff $W/na.txt $W/nb.txt | head -10 | tee -a $W/summary.txt
# (3) timing
curl -sSfL -o $W/gen.zip https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/generator.zip
echo "78f973f75d14b48e792628284f19c75de71fa597d20952e16a9ced93c78bee4a  $W/gen.zip" | sha256sum -c --quiet || exit 1
(cd $W && unzip -q -o gen.zip && cd generator && cargo build --release -q)
G=$(find $W -path '*release/generate' -type f | head -1)
python3 -m pip install -q duckdb; git show origin/bench:content.py > $W/content.py
gen() { wipe $W/data; $G --seed 7 --events $1 --clicks $2 --output $W/data --threads 4 >/dev/null; S "dataset events=$1: $(du -sh $W/data | cut -f1)"; }
run_img() { # IMAGE OUTDIR -> RUN_WALL RUN_INT RUN_PEAK_MB RUN_RC ; peak = cgroup memory.peak polled every 20 ms (last value kept)
  local img=$1 out=$2 scr nm=r1c-$RANDOM t0 t1 pid id cg em ee
  scr=$(mktemp -d); chmod 777 $out $scr; t0=$(date +%s.%N)
  docker run --name $nm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 \
    --cpus 4 --memory ${MEMLIM:-14g} --memory-swap ${MEMLIM:-14g} --user 1000:1000 -v $W/data:/data:ro -v $out:/output -v $scr:/scratch \
    -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch -e NCPU=4 -e SHOP_TIMING=1 $img 2>$W/err.txt &
  pid=$!; echo 0 > $W/peak.txt
  ( while kill -0 $pid 2>/dev/null; do
      id=$(docker inspect -f '{{.Id}}' $nm 2>/dev/null)
      if [ -n "$id" ]; then cg=/sys/fs/cgroup/system.slice/docker-$id.scope; [ -f $cg/memory.peak ] && cat $cg/memory.peak > $W/peak.txt; fi
      sleep 0.02; done ) &
  wait $pid; RUN_RC=$?; t1=$(date +%s.%N); docker rm $nm >/dev/null 2>&1
  RUN_WALL=$(printf '%.3f' $(echo "$t1 - $t0" | bc)); RUN_PEAK_MB=$(( $(cat $W/peak.txt)/1048576 ))
  em=$(grep -o 'main=[0-9.]*' $W/err.txt | cut -d= -f2); ee=$(grep -o 'end=[0-9.]*' $W/err.txt | cut -d= -f2)
  RUN_INT=$(printf '%.3f' $(echo "${ee:-0} - ${em:-0}" | bc)); wipe $scr
}
med() { sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
timing() { # label
  for im in r1A r1B; do o=$(mktemp -d); run_img $im $o; wipe $o; done
  : > $W/tA.txt; : > $W/tB.txt; : > $W/iA.txt; : > $W/iB.txt
  for k in 1 2 3 4 5 6 7 8 9; do
    for im in A B; do o=$(mktemp -d); run_img r1$im $o; wipe $o; echo $RUN_WALL >> $W/t$im.txt; echo $RUN_INT >> $W/i$im.txt; S "$1 run$k $im rc=$RUN_RC wall=$RUN_WALL internal=$RUN_INT peakMB=$RUN_PEAK_MB"; done
  done
  wa=$(med < $W/tA.txt); wb=$(med < $W/tB.txt); ia=$(med < $W/iA.txt); ib=$(med < $W/iB.txt)
  S "$1 MEDIAN wall A=$wa B=$wb diff(B-A)=$(echo "scale=3; ($wb - $wa)*100/$wa" | bc)%  internal A=$ia B=$ib diff=$(echo "scale=3; ($ib - $ia)*100/$ia" | bc)%"
  paired=0; for k in 1 2 3 4 5 6 7 8 9; do a=$(sed -n ${k}p $W/tA.txt); b=$(sed -n ${k}p $W/tB.txt); [ "$(echo "$a < $b" | bc)" = 1 ] && paired=$((paired+1)); done; S "$1 A faster in $paired/9 pairs (wall)"
}
gen 60000000 15000000
oA=$(mktemp -d); oB=$(mktemp -d); run_img r1A $oA; run_img r1B $oB
if python3 $W/content.py $oA $oB > $W/c.txt 2>&1; then S "A vs B content: SAME"; else S "A vs B content: DIFFERENT"; tail -6 $W/c.txt; fi; wipe $oA $oB
timing 60M
MEMLIM=12g; o=$(mktemp -d); run_img r1A $o; S "60M mem12g cgroup memory.peak polled 20ms: ${RUN_PEAK_MB} MB rc=$RUN_RC"; wipe $o
o=$(mktemp -d); s=$(mktemp -d); chmod 777 $o $s
DATA_DIR=$W/data OUTPUT_DIR=$o SCRATCH_DIR=$s NCPU=4 /usr/bin/time -v $A 2> $W/tv.txt >/dev/null; S "60M native maximum RSS (time -v, counts mmapped input pages): $(grep 'Maximum resident' $W/tv.txt)"; wipe $o $s
unset MEMLIM
gen 20000000 5000000
timing t20m
S "== done"
echo "---- SUMMARY ----"; cat $W/summary.txt; { echo '```'; cat $W/summary.txt; echo '```'; } >> $GITHUB_STEP_SUMMARY
