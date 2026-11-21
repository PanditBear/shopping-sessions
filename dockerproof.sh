#!/bin/bash
# dockerproof.sh BASE NEW: docker build the repo Dockerfile of each commit, run both on the same data, compare outputs.
set -uo pipefail
W=/tmp/dp; mkdir -p $W
echo "== generator"
curl -sSfL -o $W/gen.zip https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/generator.zip
echo "13724cadbac5a99d847e27cea911cd2e338ff972b46c2039da87452011bb101e  $W/gen.zip" | sha256sum -c --quiet
(cd $W && unzip -q -o gen.zip && cd generator && cargo build --release -q)
G=$(find $W -path '*release/generate' -type f | head -1)
$G --seed 7 --events ${EVENTS:-20000000} --clicks ${CLICKS:-5000000} --output $W/data --threads 4 >/dev/null
i=0
for sha in "$@"; do
  i=$((i+1))
  git fetch -q origin "$sha" 2>/dev/null || true
  git worktree add -q --detach $W/src$i "$sha"
  echo "== docker build $sha"; grep -n "RUSTFLAGS" $W/src$i/Dockerfile
  t=$(date +%s)
  if docker build -t dp$i $W/src$i; then echo "BUILD OK $sha in $(( $(date +%s) - t )) s"; else echo "BUILD FAILED $sha"; exit 1; fi
  echo "IMAGE SIZE $sha: $(docker image inspect -f '{{.Size}}' dp$i) bytes"
  out=$W/out$i; scr=$W/scr$i; mkdir -p $out $scr; chmod 777 $out $scr
  docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL \
    --security-opt no-new-privileges --cpus 4 --memory 14g --user 1000:1000 \
    -v $W/data:/data:ro -v $out:/output -v $scr:/scratch dp$i 2>&1 | tail -2
  echo "run $sha rc=${PIPESTATUS[0]}"
done
pip install -q pyarrow
python3 - <<'PY'
import os, sys, filecmp
import pyarrow.parquet as pq
a, b = "/tmp/dp/out1", "/tmp/dp/out2"
def files(d):
    return sorted(os.path.relpath(os.path.join(r, f), d) for r, _, fs in os.walk(d) for f in fs)
fa, fb = files(a), files(b)
ok = fa == fb
print("file lists equal:", ok, len(fa), len(fb))
nbytes = 0
for f in fa:
    if f not in fb: continue
    pa, pb = os.path.join(a, f), os.path.join(b, f)
    if f.endswith(".parquet"):
        ta, tb = pq.read_table(pa), pq.read_table(pb)
        keys = [(c, "ascending") for c in ta.column_names]
        same = ta.schema.equals(tb.schema) and ta.num_rows == tb.num_rows and ta.sort_by(keys).equals(tb.sort_by(keys))
        byt = filecmp.cmp(pa, pb, shallow=False)
        print(f, "rows", ta.num_rows, "sorted-equal", same, "bytes-equal", byt)
        ok &= same
    else:
        same = filecmp.cmp(pa, pb, shallow=False); print(f, "bytes-equal", same); ok &= same
print("SORTED-EQUAL RESULT:", "PASS" if ok else "FAIL")
PY
