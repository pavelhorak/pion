#!/bin/bash
# s50_run.sh: one VectorDBBench StreamingPerformanceCase run (OpenAI 50K, 200 rows/s, stages
# 50%/80%, concurrency 5 and 10, then 30 s of reads after the stream and optimize()) through
# the STOCK client, which never sends FT.OPTIMIZE, against a fresh server. Same setup as
# phase A's S50 runs (spike_2026-10-19/raw/s50). After the run, post_check.py reads recall on
# every worker through its affinity port, before the server stops.
#
# Usage: OUT=<dir> BIN=<binary> [LABEL=x] [PORT=6395] [ENVS="K=V ..."] bash s50_run.sh
set -u
OUT=${OUT:?}; BIN=${BIN:?}; LABEL=${LABEL:-s50}; PORT=${PORT:-6395}; ENVS=${ENVS:-}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${STOCK_PY:?python of an unpatched vectordbbench 1.0.22 install}
STREAM=$HERE/../2026-10-19-mac-m4-drop-in/vdbb_stream.py
mkdir -p "$OUT/data"
( cd "$OUT/data" && exec env $ENVS "$BIN" -p $PORT -w 10 --independent-workers --no-auto-detect \
    --no-auto-embed --no-wal > "$OUT/server_log.txt" 2>&1 ) &
SP=$!
for _ in $(seq 90); do redis-cli -p $PORT PING 2>/dev/null | grep -q PONG && break; sleep 1; done
echo "$(date '+%T') $LABEL start ($(basename "$BIN") $ENVS)"
RESULTS_LOCAL_DIR=$OUT "$PY" "$STREAM" redis --host 127.0.0.1 --port $PORT --no-ssl --cmd \
  --case-type StreamingPerformanceCase --k 100 --m 16 --ef-construction 128 --ef-runtime 150 \
  --db-label pion --task-label "$LABEL" > "$OUT/run.txt" 2>&1
echo "  client rc=$?"
"$PY" "$HERE/post_check.py" $PORT > "$OUT/post_check.txt" 2>&1
redis-cli -p $PORT INFO 2>/dev/null | tr -d '\r' | grep '^vector_' > "$OUT/info.txt"
kill $SP 2>/dev/null; wait $SP 2>/dev/null
rm -rf "$OUT/data"
grep -E 'Serial search - .* done|max_qps|Update largest qps|recall=' "$OUT/run.txt" | tail -12
cat "$OUT/post_check.txt"
echo "$(date '+%T') $LABEL done"
