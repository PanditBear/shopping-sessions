#!/bin/bash
# job5.sh: R1 on e30e4a4 (adaptive user_daily compression)
set -uo pipefail
PICK=e30e4a45b20c208b0ab206d7bd27e288c4758e3d; REF=3067223
W=/tmp/r1; mkdir -p $W
S() { echo "$*" | tee -a $W/summary.txt; }
wipe() { for p in "$@"; do [ -e "$p" ] && sudo find "$p" -delete 2>/dev/null; done; return 0; }
S "cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"; grep MemTotal /proc/meminfo | tee -a $W/summary.txt
sudo apt-get update -q >/dev/null && sudo apt-get install -y -q --no-install-recommends musl-tools bc binutils >/dev/null
rustup target add x86_64-unknown-linux-musl >/dev/null 2>&1
git worktree add -q --detach $W/src $PICK; git worktree add -q --detach $W/refsrc $REF
S "pick: $(git -C $W/src log -1 --format='%H %s')"; S "zauto in source: $(grep -c zauto $W/src/impl/rs/src/main.rs) lines"
# 1. two builds
for i in 1 2; do
  docker build --no-cache --pull -q -t r1img$i $W/src >/dev/null 2>&1 || { S "BUILD $i FAILED"; exit 1; }
  cid=$(docker create r1img$i); docker cp $cid:/app/shop $W/bin$i; docker rm $cid >/dev/null
  S "build $i sha256 $(sha256sum $W/bin$i | cut -d' ' -f1) size $(stat -c %s $W/bin$i)"
done
for s in .text .rodata .data .data.rel.ro .eh_frame; do objcopy -O binary --only-section=$s $W/bin1 $W/a.sec; objcopy -O binary --only-section=$s $W/bin2 $W/b.sec
  if cmp -s $W/a.sec $W/b.sec; then S "section $s: same bytes ($(stat -c %s $W/a.sec))"; else S "section $s: DIFFERENT ndiff=$(cmp -l $W/a.sec $W/b.sec | wc -l)"; fi; done
S "binary cmp -l bytes differing: $(cmp -l $W/bin1 $W/bin2 | wc -l)"
docker tag r1img1 r1img
# reference
(cd $W/refsrc/impl/rs && CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_TARGET_DIR=$W/tgt cargo build --release --locked -q --target x86_64-unknown-linux-musl) && cp $W/tgt/x86_64-unknown-linux-musl/release/shop $W/shop-ref || { S "REF BUILD FAILED"; exit 1; }
curl -sSfL -o $W/gen.zip https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/generator.zip
echo "78f973f75d14b48e792628284f19c75de71fa597d20952e16a9ced93c78bee4a  $W/gen.zip" | sha256sum -c --quiet || exit 1
(cd $W && unzip -q -o gen.zip && cd generator && cargo build --release -q)
G=$(find $W -path '*release/generate' -type f | head -1)
python3 -m pip install -q duckdb; git show origin/bench:content.py > $W/content.py
gen() { wipe $W/data; $G --seed 7 --events $1 --clicks $2 --output $W/data --threads 4 >/dev/null; S "dataset events=$1: $(du -sh $W/data | cut -f1)"; }
mkref() { wipe $W/refout; mkdir -p $W/refout; chmod 777 $W/refout; local s; s=$(mktemp -d); chmod 777 $s; DATA_DIR=$W/data OUTPUT_DIR=$W/refout SCRATCH_DIR=$s NCPU=4 $W/shop-ref >/dev/null 2>&1; S "ref rc=$?"; wipe $s; }
run_img() { # CPUS MEM OUTDIR -> RUN_RC RUN_WALL RUN_PEAK_MB RUN_ZA
  local c=$1 m=$2 out=$3 dc scr nm=r1c-$RANDOM t0 t1 pid id cg v pk; dc=$c; [ $c -gt 4 ] && dc=4
  scr=$(mktemp -d); chmod 777 $out $scr; echo 0 > $W/peak.txt; t0=$(date +%s.%N)
  docker run --name $nm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 \
    --cpus $dc --memory $m --memory-swap $m --user 1000:1000 -v $W/data:/data:ro -v $out:/output -v $scr:/scratch \
    -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch -e NCPU=$c -e SHOP_TIMING=1 r1img 2>$W/err.txt &
  pid=$!
  ( while kill -0 $pid 2>/dev/null; do
      id=$(docker inspect -f '{{.Id}}' $nm 2>/dev/null)
      if [ -n "$id" ]; then cg=/sys/fs/cgroup/system.slice/docker-$id.scope
        if v=$(cat $cg/memory.peak 2>/dev/null) && [ -n "$v" ]; then echo $v > $W/peak.tmp && mv -f $W/peak.tmp $W/peak.txt; fi; fi
      sleep 0.02; done ) &
  wait $pid; RUN_RC=$?; wait; t1=$(date +%s.%N); docker rm $nm >/dev/null 2>&1
  RUN_WALL=$(printf '%.3f' $(echo "$t1 - $t0" | bc)); pk=$(cat $W/peak.txt 2>/dev/null); RUN_PEAK_MB=$(( ${pk:-0}/1048576 ))
  RUN_ZA=$(grep -a 'zauto' $W/err.txt | head -1); [ -z "$RUN_ZA" ] && RUN_ZA="NO ZAUTO LINE"; wipe $scr
}
mkcmp() { if python3 $W/content.py $W/refout $1 > $W/content.txt 2>&1; then CMP="CONTENT SAME"; else CMP="CONTENT FAIL: $(tail -3 $W/content.txt | tr '\n' ' ')"; fi
  CMP="$CMP [out files $(find $1 -type f | wc -l) bytes $(du -sb $1 | cut -f1) vs ref files $(find $W/refout -type f | wc -l) bytes $(du -sb $W/refout | cut -f1)]"; }
med() { sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
gen 20000000 5000000; mkref
for k in 1 2 3; do o=$(mktemp -d); run_img 4 14g $o; mkcmp $o; S "SCORER t20m run$k: rc=$RUN_RC wall=$RUN_WALL peakMB=$RUN_PEAK_MB $CMP"; S "   $RUN_ZA"; wipe $o; done
for n in 1 2 4 8 16; do o=$(mktemp -d); run_img $n 14g $o; mkcmp $o; S "NCPU=$n t20m: rc=$RUN_RC wall=$RUN_WALL peakMB=$RUN_PEAK_MB $CMP"; S "   $RUN_ZA"; wipe $o; done
gen 60000000 15000000; mkref; : > $W/t60.txt
o=$(mktemp -d); run_img 4 14g $o; mkcmp $o; S "60M warmup: rc=$RUN_RC wall=$RUN_WALL peakMB=$RUN_PEAK_MB $CMP"; S "   $RUN_ZA"; wipe $o
for k in 1 2 3; do o=$(mktemp -d); run_img 4 14g $o; mkcmp $o; echo $RUN_WALL >> $W/t60.txt; S "60M run$k: rc=$RUN_RC wall=$RUN_WALL peakMB=$RUN_PEAK_MB $CMP"; S "   $RUN_ZA"; wipe $o; done
S "60M MEDIAN wall $(med < $W/t60.txt) (runs $(tr '\n' ' ' < $W/t60.txt))"
S "== done"
echo "---- SUMMARY ----"; cat $W/summary.txt; { echo '```'; cat $W/summary.txt; echo '```'; } >> $GITHUB_STEP_SUMMARY
