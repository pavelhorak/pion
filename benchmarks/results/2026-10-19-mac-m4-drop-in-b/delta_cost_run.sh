#!/bin/bash
# delta_cost_run.sh: A's delta_cost.py (spike_2026-10-19) against one binary, -w 1, --no-wal,
# with the settle rebuild off (PION_VECTOR_SETTLE_MS=0) so the measured delta stays put.
# Usage: OUT=<file> BIN=<binary> [DELTA=4900] [PORT=6396] bash delta_cost_run.sh
set -u
OUT=${OUT:?}; BIN=${BIN:?}; DELTA=${DELTA:-4900}; PORT=${PORT:-6396}
PY=${STOCK_PY:?python of an unpatched vectordbbench 1.0.22 install}
DC=$(cd "$(dirname "$0")" && pwd)/delta_cost.py
D=$(mktemp -d /tmp/pion_dc.XXXX)
( cd "$D" && exec env PION_VECTOR_SETTLE_MS=0 "$BIN" -p $PORT -w 1 --no-auto-detect --no-auto-embed --no-wal > "$D/server.txt" 2>&1 ) &
SP=$!
for _ in $(seq 60); do redis-cli -p $PORT PING 2>/dev/null | grep -q PONG && break; sleep 1; done
"$PY" "$DC" $PORT /tmp/vectordb_bench/dataset/openai/openai_small_50k $DELTA > "$OUT" 2>&1
kill $SP; wait $SP 2>/dev/null
rm -rf "$D"
cat "$OUT"
