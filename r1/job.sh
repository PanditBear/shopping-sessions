#!/bin/bash
# r1/job.sh ROLE(A|B): rebuild the pick image from scratch, time it, hash the binary, run it with the scorer flags.
set -uo pipefail
ROLE=$1; PICK=5ae524bd8bb1440966d8e6a07d11c3b91f057b9e; REF=3067223
W=/tmp/r1; mkdir -p $W
S() { echo "$*" | tee -a $W/summary.txt; }
wipe() { for p in "$@"; do [ -e "$p" ] && sudo find "$p" -delete 2>/dev/null; done; return 0; }
S "== role $ROLE  $(date -u +%FT%TZ)"
lscpu | grep -E 'Model name|^CPU\(s\)|L2|L3'; free -g | head -2; df -h / /mnt 2>/dev/null
S "cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
sudo apt-get update -q >/dev/null && sudo apt-get install -y -q --no-install-recommends musl-tools bc >/dev/null
rustup target add x86_64-unknown-linux-musl >/dev/null 2>&1
git worktree add -q --detach $W/src $PICK; S "picked: $(git -C $W/src rev-parse --short HEAD)"
# 1. clean image build
docker image prune -af >/dev/null 2>&1; docker builder prune -af >/dev/null 2>&1
t=$(date +%s.%N)
docker build --no-cache --pull --progress=plain -t r1img $W/src > $W/build.log 2>&1; brc=$?
bt=$(echo "$(date +%s.%N) - $t" | bc); S "docker build rc=$brc time=${bt}s"
S "fetch lines: apt Get: $(grep -c ' Get:' $W/build.log), apt fetched: $(grep -E 'Fetched [0-9]' $W/build.log | head -1)"
S "crates downloaded: $(grep -c 'Downloaded ' $W/build.log), rustup: $(grep -Ei 'downloading component|installing component|info: downloading' $W/build.log | head -3 | tr '\n' ';')"
S "base image: $(grep -E 'resolve docker.io/library/rust|FROM docker.io/library/rust' $W/build.log | head -1)"
[ $brc = 0 ] || { S "BUILD FAILED"; tail -40 $W/build.log; exit 1; }
cid=$(docker create r1img); docker cp $cid:/app/shop $W/shop-img; docker rm $cid >/dev/null
S "BINSHA $ROLE $(sha256sum $W/shop-img | cut -d' ' -f1)"
echo "$ROLE $(sha256sum $W/shop-img | cut -d' ' -f1) build=${bt}s" > $W/binsha-$ROLE.txt
docker build --no-cache -q -t r1img2 $W/src > /dev/null 2>&1; cid=$(docker create r1img2); docker cp $cid:/app/shop $W/shop-img2; docker rm $cid >/dev/null
S "SAME-RUNNER REBUILD $ROLE: $(sha256sum $W/shop-img2 | cut -d' ' -f1) vs first $(sha256sum $W/shop-img | cut -d' ' -f1)"
# 2. reference build (native, as bench.sh builds a ref)
git worktree add -q --detach $W/refsrc $REF
(cd $W/refsrc/impl/rs && CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_TARGET_DIR=$W/tgt \
  cargo build --release --locked -q --target x86_64-unknown-linux-musl) && cp $W/tgt/x86_64-unknown-linux-musl/release/shop $W/shop-ref || { S "REF BUILD FAILED"; exit 1; }
# 3. generator
curl -sSfL -o $W/gen.zip https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/generator.zip
echo "78f973f75d14b48e792628284f19c75de71fa597d20952e16a9ced93c78bee4a  $W/gen.zip" | sha256sum -c --quiet || exit 1
(cd $W && unzip -q -o gen.zip && cd generator && cargo build --release -q)
G=$(find $W -path '*release/generate' -type f | head -1)
python3 -m pip install -q duckdb
git show origin/bench:content.py > $W/content.py
gen() { wipe $W/data; $G --seed 7 --events $1 --clicks $2 --output $W/data --threads 4 >/dev/null; S "dataset events=$1 clicks=$2: $(du -sh $W/data | cut -f1)"; }
run_img() { # CPUS MEM OUTDIR -> RUN_RC RUN_WALL RUN_PEAK_MB
  local dc c=$1 m=$2; dc=$c; [ $c -gt 4 ] && dc=4; local m=$2 out=$3 scr nm=r1c-$RANDOM t0 t1 rc pk=0 v pid cg id
  scr=$(mktemp -d); chmod 777 $out $scr
  t0=$(date +%s.%N)
  docker run --name $nm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 \
    --cpus $dc --memory $m --memory-swap $m --user 1000:1000 \
    -v $W/data:/data:ro -v $out:/output -v $scr:/scratch \
    -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch \
    -e NCPU=$c -e SHOP_TIMING=1 r1img 2>$W/err.txt &
  pid=$!
  echo 0 > $W/peak.txt
  ( while kill -0 $pid 2>/dev/null; do
      id=$(docker inspect -f '{{.Id}}' $nm 2>/dev/null)
      if [ -n "$id" ]; then for cg in /sys/fs/cgroup/system.slice/docker-$id.scope /sys/fs/cgroup/docker/$id; do
        if [ -f $cg/memory.peak ]; then cat $cg/memory.peak > $W/peak.txt
        elif [ -f $cg/memory.current ]; then v=$(cat $cg/memory.current); [ "$v" -gt "$(cat $W/peak.txt)" ] && echo $v > $W/peak.txt; fi
      done; fi; sleep 0.25; done ) &
  wait $pid; rc=$?; t1=$(date +%s.%N)
  docker rm $nm >/dev/null 2>&1
  pk=$(cat $W/peak.txt 2>/dev/null); pk=${pk:-0}
  RUN_RC=$rc; RUN_WALL=$(printf '%.3f' $(echo "$t1 - $t0" | bc)); RUN_PEAK_MB=$((pk/1048576))
  wipe $scr
}
mkcmp() {
  if python3 $W/content.py $W/refout $1 > $W/content.txt 2>&1; then CMP="CONTENT SAME"; else CMP="CONTENT DIFFERENT"; fi
  if diff -r -q $W/refout $1 >/dev/null 2>&1; then CMP="BYTE-IDENTICAL / $CMP"; fi
}
mkref() {
  wipe $W/refout; mkdir -p $W/refout; local s t0; s=$(mktemp -d); t0=$(date +%s.%N)
  DATA_DIR=$W/data OUTPUT_DIR=$W/refout SCRATCH_DIR=$s NCPU=4 $W/shop-ref 2>/dev/null; S "ref($REF) native rc=$? wall=$(printf '%.2f' $(echo "$(date +%s.%N) - $t0" | bc))s"
  wipe $s
}
gen 20000000 5000000; mkref
for k in 1 2 3; do
  out=$(mktemp -d); run_img 4 14g $out; mkcmp $out
  S "SCORER-FLAGS t20m run$k (cpus4 mem14g): rc=$RUN_RC wall=${RUN_WALL}s peakMB=$RUN_PEAK_MB -> $CMP"
  wipe $out
done
tail -4 $W/content.txt | tee -a $W/summary.txt
if [ "$ROLE" = A ]; then
  for n in 1 2 4 8 16; do
    out=$(mktemp -d); run_img $n 14g $out; mkcmp $out
    S "NCPU=$n t20m: rc=$RUN_RC wall=${RUN_WALL}s peakMB=$RUN_PEAK_MB -> $CMP"; wipe $out
  done
  wipe $W/refout; gen 60000000 15000000; mkref
  out=$(mktemp -d); run_img 4 12g $out; mkcmp $out
  S "60M mem12g cpus4: rc=$RUN_RC wall=${RUN_WALL}s peakMB(cgroup)=$RUN_PEAK_MB -> $CMP"; tail -4 $W/content.txt | tee -a $W/summary.txt
  tail -5 $W/err.txt | tee -a $W/summary.txt
fi
S "== done $ROLE"
echo "---- SUMMARY ----"; cat $W/summary.txt
{ echo '```'; cat $W/summary.txt; echo '```'; } >> $GITHUB_STEP_SUMMARY
