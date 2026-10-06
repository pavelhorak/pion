#!/bin/bash
# Phase 4: the April peak configuration, with raw output, and the same peak
# measured against Redis with an equal number of keyspaces.
#   a) April's Pion run: pion-server -w 32, memtier 16 threads x 50 connections,
#      P=50, 256-byte and 64-byte values (Pion only, as in April).
#   b) Symmetric: 32 keyspaces on each side, 32 client threads and 800
#      connections in total, P=50, 256-byte values:
#      pion-server -w 32 (memtier -t 32 -c 25) against 32 Redis processes
#      (one memtier -t 1 -c 25 each).
# Same Redis 8.10.2 and Pion binary as run_on_box.sh. Writes $OUT/DONE.
set -u
OUT=${OUT:-$HOME/peak}; RAW=$OUT/raw; mkdir -p "$RAW"
PION=${PION:-/opt/Pion/pion-server}
REDIS=$HOME/redis-8.10.2/src/redis-server
RCLI=$HOME/redis-8.10.2/src/redis-cli
REPS=${REPS:-3}
TEST_TIME=${TEST_TIME:-30}
COMMON="--protocol=redis --ratio=1:10 --key-pattern=R:R --key-minimum=1 --key-maximum=1000000 --test-time=$TEST_TIME --hide-histogram --distinct-client-seed --pipeline=50"
fail() { echo "FAILED: $*"; echo "$*" > "$OUT/FAILED"; exit 1; }
stamp() { echo "== $(date +%H:%M:%S) $*"; }
stop_all() {
    pkill -x pion-server 2>/dev/null; pkill -x redis-server 2>/dev/null; sleep 2
    pkill -9 -x pion-server 2>/dev/null; pkill -9 -x redis-server 2>/dev/null; sleep 1
}
start_redis() {   # port
    local p=$1
    rm -rf "/tmp/r$p" && mkdir -p "/tmp/r$p"
    "$REDIS" --port "$p" --bind 127.0.0.1 --protected-mode no --daemonize yes --dir "/tmp/r$p" \
        --io-threads 1 --save "" --appendonly no --loglevel warning || fail "redis $p did not start"
    for _ in $(seq 100); do "$RCLI" -p "$p" ping >/dev/null 2>&1 && return 0; sleep 0.1; done
    fail "redis $p did not answer"
}
start_pion() {    # workers
    rm -rf /tmp/pion && mkdir -p /tmp/pion
    (cd /tmp/pion && "$PION" -p 1974 -w "$1" --independent-workers --profile kv --no-auto-detect --no-auto-embed \
        --no-crash-log --no-wal > "$OUT/pion_server_w$1.log" 2>&1 &)
    for _ in $(seq 300); do "$RCLI" -p 1974 ping >/dev/null 2>&1 && return 0; sleep 0.1; done
    fail "pion -w $1 did not answer"
}
mt() {            # name port threads clients datasize
    memtier_benchmark -s 127.0.0.1 -p "$2" -t "$3" -c "$4" -d "$5" $COMMON \
        --json-out-file="$RAW/$1.json" > "$RAW/$1.txt" 2>&1 || echo "memtier exit $? for $1"
}
stop_all
"$PION" --version > "$OUT/pion_version.txt" 2>&1 || fail "no pion-server at $PION"
"$REDIS" --version > "$OUT/redis_version.txt"
echo "REPS=$REPS TEST_TIME=$TEST_TIME COMMON=$COMMON" > "$OUT/settings.txt"
for rep in $(seq "$REPS"); do
    for d in 256 64; do
        stamp "april pion w32 t16 c50 d$d r$rep"
        start_pion 32; mt "april_pion_w32_t16c50_d${d}_r$rep" 1974 16 50 "$d"; stop_all
    done
    stamp "sym pion w32 t32 c25 r$rep"
    start_pion 32; mt "sym_pion_w32_t32c25_d256_r$rep" 1974 32 25 256; stop_all
    stamp "sym redis x32 r$rep"
    for i in $(seq 0 31); do start_redis $((6400 + i)); done
    for i in $(seq 0 31); do mt "sym_redis_x32_t1c25_d256_r${rep}_i$i" $((6400 + i)) 1 25 256 & done
    wait
    stop_all
done
python3 "$(dirname "$0")/summarize.py" "$OUT" > "$OUT/summary.md" 2>&1 || echo "summarize failed"
tar -C "$(dirname "$OUT")" -czf "$OUT.tgz" "$(basename "$OUT")"
touch "$OUT/DONE"
stamp done
