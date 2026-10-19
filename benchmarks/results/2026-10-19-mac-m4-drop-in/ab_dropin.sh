#!/bin/bash
# ab_dropin.sh: phase A's kill criterion, on the Mac (SPIKE.md "Kill criterion").
#
#   old  = public main (619aea2, = v0.9.9's engine) with the PATCHED client, which sends
#          FT.OPTIMIZE after the load: today's path, the gate's.
#   new  = branch vector-drop-in-a with the STOCK client (VectorDBBench 1.0.22 as shipped,
#          no FT.OPTIMIZE): exact answers until the graph builds itself in the background.
#   newp = the new binary with the patched client: the explicit path must not regress.
#
# Each arm: a fresh server (gate flags, --no-wal, its own temp dir, deleted after), one full
# run (load, [optimize], concurrent search 1/5/10 for 5 s each, serial recall), then
# SEARCH_RUNS search-only runs against the same built index. Each pair runs old, new and
# newp once, in an order rotated per pair.
#
# Usage: OUT=<dir> OLD=<binary> NEW=<binary> PAIRS=3 SEARCH_RUNS=2 bash ab_dropin.sh
set -u
OUT=${OUT:?}; OLD=${OLD:?}; NEW=${NEW:?}
PAIRS=${PAIRS:-3}; SEARCH_RUNS=${SEARCH_RUNS:-2}
PATCHED=${PATCHED_VDBB:?path to vectordbbench with the FT.OPTIMIZE patch (pixi run install-vdbbench)}
STOCK=${STOCK_VDBB:?path to an unpatched vectordbbench 1.0.22}
PORT=6395
mkdir -p "$OUT"

vdbb() {  # vdbb <client> <resultsdir> <label> [extra...]
  local client=$1 rdir=$2 label=$3; shift 3
  mkdir -p "$rdir"
  RESULTS_LOCAL_DIR=$rdir "$client" redis --host 127.0.0.1 --port $PORT --no-ssl --cmd \
    --case-type Performance1536D50K --k 100 --m 16 --ef-construction 128 --ef-runtime 150 \
    --num-concurrency 1,5,10 --concurrency-duration 5 --db-label pion --task-label "$label" "$@" \
    > "$rdir/run.txt" 2>&1
  echo "  $label rc=$?"
}

arm() {  # arm <name> <binary> <client>
  local name=$1 bin=$2 client=$3 R=$OUT/$1
  rm -rf "$R"; mkdir -p "$R/data"
  ( cd "$R/data" && exec "$bin" -p $PORT -w 10 --independent-workers --no-auto-detect \
      --no-auto-embed --no-wal > "$R/server.txt" 2>&1 ) &
  local sp=$!
  for _ in $(seq 90); do redis-cli -p $PORT PING 2>/dev/null | grep -q PONG && break; sleep 1; done
  echo "$(date '+%T') $name ($(basename "$bin"), $(basename "$(dirname "$(dirname "$client")")"))"
  vdbb "$client" "$R/full" "${name}_full"
  # the stock arm builds in the background: wait until the graph is served (INFO vector_index)
  for _ in $(seq 120); do
    st=$(redis-cli -p $PORT INFO 2>/dev/null | tr -d '\r' | awk -F: '$1=="vector_index"{print $2}')
    [ -z "$st" ] || [ "$st" = graph ] && break
    sleep 1
  done
  for s in $(seq "$SEARCH_RUNS"); do
    vdbb "$client" "$R/s$s" "${name}_s$s" --skip-drop-old --skip-load
  done
  redis-cli -p $PORT INFO 2>/dev/null | tr -d '\r' | grep '^vector_' > "$R/info.txt"
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  rm -rf "$R/data"
  sleep 3
}

# rotate the order per pair: the first run of a series tends to be the slow one
for p in $(seq "$PAIRS"); do
  case $((p % 3)) in
    1) arm "old_p$p" "$OLD" "$PATCHED"; arm "new_p$p" "$NEW" "$STOCK";   arm "newp_p$p" "$NEW" "$PATCHED" ;;
    2) arm "new_p$p" "$NEW" "$STOCK";   arm "newp_p$p" "$NEW" "$PATCHED"; arm "old_p$p" "$OLD" "$PATCHED" ;;
    0) arm "newp_p$p" "$NEW" "$PATCHED"; arm "old_p$p" "$OLD" "$PATCHED"; arm "new_p$p" "$NEW" "$STOCK" ;;
  esac
done
echo ALL_DONE
