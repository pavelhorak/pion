#!/bin/bash
# k50_series.sh: Performance1536D50K (VectorDBBench 1.0.22), phase A's ab_dropin arms for B:
#   a_stock = A (4c735f8) + STOCK client (no FT.OPTIMIZE): the automatic build after the load
#   b_stock = B + STOCK client: B's trigger also builds DURING the load (16,384 rows, no pause)
#   b_patch = B + PATCHED client (FT.OPTIMIZE after the load): the explicit path
# Each arm: a fresh server (gate flags, --no-wal, own temp dir, deleted after), one full run
# (load, [optimize], concurrent 1/5/10 for 5 s each, serial recall), then SEARCH_RUNS search-only
# runs. Order rotated per pair. Usage: OUT=<dir> A=<bin> B=<bin> PAIRS=2 SEARCH_RUNS=2 bash k50_series.sh
set -u
OUT=${OUT:?}; A=${A:?}; B=${B:?}
PAIRS=${PAIRS:-2}; SEARCH_RUNS=${SEARCH_RUNS:-2}
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
  echo "$(date '+%T') $name"
  vdbb "$client" "$R/full" "${name}_full"
  # wait until the graph serves and no build or save is running
  for _ in $(seq 180); do
    inf=$(redis-cli -p $PORT INFO 2>/dev/null | tr -d '\r')
    st=$(echo "$inf" | awk -F: '$1=="vector_index"{print $2}')
    run=$(echo "$inf" | awk -F: '$1=="vector_build_running"{print $2}')
    { [ "$st" = graph ] && [ "${run:-0}" = 0 ]; } && break
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

for p in $(seq "$PAIRS"); do
  case $((p % 3)) in
    1) arm "a_stock_p$p" "$A" "$STOCK"; arm "b_stock_p$p" "$B" "$STOCK"; arm "b_patch_p$p" "$B" "$PATCHED" ;;
    2) arm "b_stock_p$p" "$B" "$STOCK"; arm "b_patch_p$p" "$B" "$PATCHED"; arm "a_stock_p$p" "$A" "$STOCK" ;;
    0) arm "b_patch_p$p" "$B" "$PATCHED"; arm "a_stock_p$p" "$A" "$STOCK"; arm "b_stock_p$p" "$B" "$STOCK" ;;
  esac
done
echo ALL_DONE
