#!/bin/bash
# cpu.sh REF... : build refs (static musl, x86-64-v3), run them natively interleaved, record whole-process CPU.
set -uo pipefail
RUNS=${RUNS:-5}; CPUS=4; SEED=7
W=/tmp/cpu; mkdir -p $W
echo "== machine"; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|L2|L3'; free -g | head -2; uname -r; df -h /tmp | tail -1
P=$(command -v perf || ls /usr/lib/linux-tools/*/perf 2>/dev/null | head -1 || true); echo "perf=$P"
[ -n "$P" ] && sudo sysctl -w kernel.perf_event_paranoid=-1 >/dev/null
curl -sSfL -o $W/gen.zip https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/generator.zip
echo "13724cadbac5a99d847e27cea911cd2e338ff972b46c2039da87452011bb101e  $W/gen.zip" | sha256sum -c --quiet || { echo 'gen.zip checksum mismatch'; exit 1; }
mkdir -p $W/genx && unzip -q -o $W/gen.zip -d $W/genx && (cd $W/genx/generator && cargo build --release -q)
G=$(find $W/genx -path '*release/generate' -type f | head -1)
for f in enabled defrag; do echo "thp $f: $(cat /sys/kernel/mm/transparent_hugepage/$f)"; done
cat > $W/nothp.c <<'CEOF'
#include <sys/prctl.h>
#include <unistd.h>
int main(int c,char**v){prctl(PR_SET_THP_DISABLE,1,0,0,0);execv(v[1],v+1);return 127;}
CEOF
gcc -O2 -o $W/nothp $W/nothp.c
i=0; BINS=()
for r in "$@"; do
  i=$((i+1)); sha=$(git rev-parse --short "$r^{commit}")
  git worktree add -q --detach $W/w$i "$sha"
  (cd $W/w$i/impl/rs && CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_TARGET_DIR=$W/tgt$i \
     cargo build --release --locked -q --target x86_64-unknown-linux-musl) || echo "BUILD FAIL $sha"
  cp $W/tgt$i/x86_64-unknown-linux-musl/release/shop $W/bin-$sha; BINS+=("$sha")
  echo "built $sha"
done

one() { # sha datadir tag mode(on|off)
  local o s2 tf pf
  o=$(mktemp -d); s2=$(mktemp -d); tf=$W/t.txt; pf=$W/p.txt; : > $tf; : > $pf
  local cmd="/usr/bin/time -v -o $tf $W/bin-$1"
  [ "$4" = off ] && cmd="/usr/bin/time -v -o $tf $W/nothp $W/bin-$1"
  [ -n "$P" ] && cmd="$P stat -x, -o $pf -e task-clock,page-faults,context-switches,cpu-migrations $cmd"
  local t0=$(date +%s.%N)
  env DATA_DIR=$2 OUTPUT_DIR=$o SCRATCH_DIR=$s2 NCPU=$CPUS SHOP_TIMING=1 $cmd 2>$W/err.txt >/dev/null
  local t1=$(date +%s.%N)
  local wall=$(echo "$t1 - $t0" | bc)
  local us=$(awk -F': ' '/User time/{print $2}' $tf) sy=$(awk -F': ' '/System time/{print $2}' $tf)
  local rss=$(awk -F': ' '/Maximum resident/{print $2}' $tf) mf=$(awk -F': ' '/Minor \(reclaiming/{print $2}' $tf)
  local vc=$(awk -F': ' '/Voluntary context/{print $2}' $tf) ic=$(awk -F': ' '/Involuntary context/{print $2}' $tf)
  local tc=NA
  [ -s $pf ] && tc=$(awk -F, '/task-clock/{printf "%.1f",$1}' $pf)
  local ps=$(grep -o 'prof(cpu s):.*' $W/err.txt | tr ' ' '\n' | grep '=' | awk -F= '{s+=$2} END{printf "%.2f",s}')
  echo "CPU $3 $1 thp=$4 wall=$wall user=$us sys=$sy taskclock_ms=$tc minflt=$mf vol=$vc invol=$ic rss_kb=$rss prof=$ps"
  cp $W/err.txt $W/lasterr-$1-$4.txt
  python3 -c "import shutil,sys; [shutil.rmtree(p, ignore_errors=True) for p in sys.argv[1:]]" $o $s2
}

for sz in 60000000:15000000; do
  ev=${sz%%:*}; ck=${sz##*:}; tag=$((ev/1000000))M
  echo "################ $tag events=$ev clicks=$ck"
  D=$W/data$tag; $G --seed $SEED --events $ev --clicks $ck --output $D --threads 4 >/dev/null; du -sh $D
  for b in "${BINS[@]}"; do one $b $D warm-$tag on >/dev/null; done
  for k in $(seq $RUNS); do for b in "${BINS[@]}"; do for m in on off; do one $b $D $tag $m; done; done; done
  echo "-- last-run stderr"; for b in "${BINS[@]}"; do echo "-- $b"; cat $W/lasterr-$b-on.txt; done
  python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" $D
done
