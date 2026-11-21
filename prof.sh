#!/bin/bash
# perf profile of one commit: env SHA SIZES CPUS. Reports land in /tmp/pf/out.
set +e
W=/tmp/pf; O=$W/out; mkdir -p $O
CPUS=${CPUS:-8}; LAST=$((CPUS-1))
{ lscpu; uname -a; free -g; nproc; for f in enabled defrag; do echo "thp $f: $(cat /sys/kernel/mm/transparent_hugepage/$f)"; done; } > $O/machine.txt 2>&1
P=$(command -v perf || ls /usr/lib/linux-tools/*/perf 2>/dev/null | head -1)
echo "perf=$P" | tee -a $O/machine.txt; $P --version | tee -a $O/machine.txt
sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0
curl -sSfL -o $W/gen.zip https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/generator.zip
echo "13724cadbac5a99d847e27cea911cd2e338ff972b46c2039da87452011bb101e  $W/gen.zip" | sha256sum -c --quiet
(cd $W && unzip -q -o gen.zip && cd generator && cargo build --release -q)
G=$(find $W -path '*release/generate' -type f | head -1)
git worktree add -q --detach $W/src "$SHA"; echo "sha: $(git -C $W/src rev-parse HEAD)" | tee -a $O/machine.txt
# A: shipped flags + line tables (no frame pointers)
(cd $W/src/impl/rs && env CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="-C target-cpu=x86-64-v3" \
   CARGO_PROFILE_RELEASE_DEBUG=line-tables-only CARGO_PROFILE_RELEASE_STRIP=none CARGO_TARGET_DIR=$W/tgta \
   cargo build --release --locked -q --target x86_64-unknown-linux-musl) && cp $W/tgta/x86_64-unknown-linux-musl/release/shop $W/shop-a
# B: same plus frame pointers (call chains)
(cd $W/src/impl/rs && env CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="-C target-cpu=x86-64-v3 -C force-frame-pointers=yes" \
   CARGO_PROFILE_RELEASE_DEBUG=line-tables-only CARGO_PROFILE_RELEASE_STRIP=none CARGO_TARGET_DIR=$W/tgtb \
   cargo build --release --locked -q --target x86_64-unknown-linux-musl) && cp $W/tgtb/x86_64-unknown-linux-musl/release/shop $W/shop-b
ls -l $W/shop-a $W/shop-b
n=0
for sz in $SIZES; do
  n=$((n+1)); ev=${sz%%:*}; ck=${sz##*:}; R=$O/e$ev; mkdir -p $R
  D=$W/data$n; $G --seed 7 --events $ev --clicks $ck --output $D --threads 4 >/dev/null
  o=$W/outd; s2=$W/scr; mkdir -p $o $s2
  clean() { sudo find $o $s2 -mindepth 1 -delete; }
  E="DATA_DIR=$D OUTPUT_DIR=$o SCRATCH_DIR=$s2 NCPU=$CPUS SHOP_TIMING=1"
  T="taskset -c 0-$LAST"
  echo "################ events=$ev"
  clean; sudo env $E $T $W/shop-a > /dev/null 2>&1   # warm page cache
  clean; sudo env $E $T $W/shop-a 2>&1 | tee $R/plain.txt | grep -E 'written|prof'
  clean; sudo env $E $T /usr/bin/time -v $W/shop-a > /dev/null 2> $R/time.txt; grep -E 'Maximum resident|faults|context|Elapsed|User time|System time' $R/time.txt
  clean; sudo env $E $T $P stat -e task-clock,page-faults,minor-faults,major-faults,context-switches,cpu-migrations,cycles,instructions,branches,branch-misses,cache-references,cache-misses $W/shop-a 2> $R/stat.txt >/dev/null
  clean; sudo env $E $T $P stat -e L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses $W/shop-a 2> $R/stat2.txt >/dev/null
  clean; sudo env $E $T $P stat -e cycles:u,cycles:k,instructions:u,instructions:k $W/shop-a 2> $R/stat3.txt >/dev/null
  grep -v '^\[' $R/stat.txt | tail -20
  clean; sudo env $E $T strace -f -c -o $R/strace-c.txt $W/shop-a >/dev/null 2>&1
  clean; sudo env $E $T strace -f -T -tt -o $R/strace-full.txt $W/shop-a >/dev/null 2>&1; head -c 300000 $R/strace-full.txt > $R/strace-head.txt; tail -c 100000 $R/strace-full.txt > $R/strace-tail.txt; sudo find $R -name strace-full.txt -delete
  EV=cycles
  clean; sudo env $E $T $P record -q -F 4999 -e cycles -o $R/perfA.data $W/shop-a >/dev/null 2>$R/recA.err
  if [ ! -s $R/perfA.data ] || grep -qi 'not supported\|cannot' $R/recA.err; then head -5 $R/recA.err; EV=cpu-clock; clean
     sudo env $E $T $P record -q -F 4999 -e cpu-clock -o $R/perfA.data $W/shop-a >/dev/null 2>&1; fi
  echo "event used: $EV" | tee $R/event.txt
  rep() { sudo timeout 900 $P report -i $R/perfA.data --stdio "$@" 2>/dev/null | grep -v '^$'; }
  rep --no-children --sort sym | grep -v '^#' | head -60 > $R/top-sym.txt
  rep --no-children --sort comm | head -20 > $R/by-comm.txt
  rep --no-children --sort dso | head -20 > $R/by-dso.txt
  rep --no-children --sort srcline --percent-limit 0.15 | grep -v '^#' | head -150 > $R/top-srcline.txt
  rep --no-children --sort sym,srcline --percent-limit 0.2 | grep -v '^#' | head -200 > $R/top-sym-srcline.txt
  rep --no-children --sort sym --dsos '[kernel.kallsyms]' | grep -v '^#' | head -40 > $R/top-kernel.txt
  rep --no-children --sort sym | grep -v '^#' | grep -E '\[\.\]' | head -3 | sed -E 's/^ *[0-9.]+% +\[.\] //' > $R/syms.txt
  while IFS= read -r s; do
    f=$(echo "$s" | tr -c 'A-Za-z0-9' '_' | cut -c1-60)
    sudo $P annotate -i $R/perfA.data --stdio -l -s "$s" > $R/ann-$f.txt 2>/dev/null
    grep -E '^ +[0-9]+\.[0-9]+ ' $R/ann-$f.txt | sort -rn | head -60 > $R/ann-$f-hot.txt
  done < $R/syms.txt
  clean; sudo env $E $T $P record -q -F 2999 -e $EV -g -o $R/perfB.data $W/shop-b >/dev/null 2>&1
  sudo $P report -i $R/perfB.data --stdio --children -g none --sort sym --percent-limit 1 2>/dev/null | grep -v '^#' | grep -v '^$' | head -60 > $R/children.txt
  sudo $P report -i $R/perfB.data --stdio --no-children -G --sort sym --percent-limit 2.5 2>/dev/null | grep -v '^#' | grep -v '^$' | head -300 > $R/callers.txt
  sudo $P report -i $R/perfB.data --stdio --no-children -g caller --dsos '[kernel.kallsyms]' --sort sym --percent-limit 0.8 2>/dev/null | grep -v '^#' | grep -v '^$' | head -200 > $R/kernel-callers.txt
  clean; sudo env $E $T $P record -q -F 999 -e $EV --call-graph dwarf,16384 -o $R/perfD.data $W/shop-a >/dev/null 2>&1
  sudo $P report -i $R/perfD.data --stdio --children -g none --sort sym --percent-limit 1 2>/dev/null | grep -v '^#' | grep -v '^$' | head -60 > $R/children-dwarf.txt
  sudo find $R -name 'perfD.data' -delete; sudo find $R -name 'perfB.data' -delete
  clean; sudo find $D -delete
  ls -l $R
done
sudo chown -R $USER $O
for d in $O/e*; do [ -f $d/perfA.data ] && gzip -f $d/perfA.data; done
du -sh $O
