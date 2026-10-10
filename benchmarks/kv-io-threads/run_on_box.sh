#!/bin/bash
# #465 acceptance: one keyspace, Pion `--io-threads N` against Redis `io-threads N`,
# by the method of benchmarks/results/2026-10-06-linux-epyc-8124p/run_on_box.sh
# (same memtier settings, client threads, connections, run length and repetitions),
# so the rows compare with the README's KV table. Adds:
#   - Pion at --io-threads 1, 2, 4, 8 (1 is the single-thread loop the README quotes);
#   - durable P=10: Pion's WAL at --io-threads 1, 4, 8 vs Redis AOF everysec at io-threads 4, 8;
#   - per-thread CPU (/proc/<pid>/task) over each Pion run, which says whether the
#     executor or the I/O threads ran out first (thread_cpu.py).
#
#   OUT=/opt/session/out/io465 PION=/opt/wt/io/pion-server bash benchmarks/kv-io-threads/run_on_box.sh
set -u
OUT=${OUT:-/opt/session/out/io465}
RAW=$OUT/raw
mkdir -p "$RAW"
HERE=$(cd "$(dirname "$0")" && pwd)
PION=${PION:-$(cd "$HERE/../.." && pwd)/pion-server}
REDIS_TAG=8.10.2
REDIS=${REDIS:-$HOME/redis-$REDIS_TAG/src/redis-server}
RCLI=${RCLI:-$HOME/redis-$REDIS_TAG/src/redis-cli}
REPS=${REPS:-3}
TEST_TIME=${TEST_TIME:-30}
CLIENT_T=${CLIENT_T:-8}
COMMON="--protocol=redis -d 256 --ratio=1:10 --key-pattern=R:R --key-minimum=1 --key-maximum=1000000 --test-time=$TEST_TIME --hide-histogram --distinct-client-seed"

stamp() { echo "== $(date +%H:%M:%S) $*"; }
fail() { echo "FAILED: $*"; echo "$*" > "$OUT/FAILED"; exit 1; }

stop_all() {
    "$RCLI" -p 6379 shutdown nosave >/dev/null 2>&1   # where pkill -x misses a retitled redis (macOS)
    pkill -x pion-server 2>/dev/null; pkill -x redis-server 2>/dev/null; sleep 2
    pkill -9 -x pion-server 2>/dev/null; pkill -9 -x redis-server 2>/dev/null; sleep 1
}

start_redis() {   # io-threads persistence(none|aof)
    local io=$1 pers=$2
    rm -rf /tmp/r6379 && mkdir -p /tmp/r6379
    local args=(--port 6379 --bind 127.0.0.1 --protected-mode no --daemonize yes --dir /tmp/r6379
                --io-threads "$io" --save "" --loglevel warning)
    if [ "$pers" = aof ]; then args+=(--appendonly yes --appendfsync everysec); else args+=(--appendonly no); fi
    "$REDIS" "${args[@]}" || fail "redis did not start"
    for _ in $(seq 100); do "$RCLI" -p 6379 ping >/dev/null 2>&1 && return 0; sleep 0.1; done
    fail "redis did not answer"
}

start_pion() {    # io-threads persistence(none|wal) tag
    local io=$1 pers=$2 tag=$3
    local args=(-p 1974 -w 1 --profile kv --no-auto-detect --no-auto-embed --no-crash-log)
    [ "$io" -gt 1 ] && args+=(--io-threads "$io")
    [ "$pers" = none ] && args+=(--no-wal)
    rm -rf /tmp/pion && mkdir -p /tmp/pion
    (cd /tmp/pion && "$PION" "${args[@]}" > "$RAW/$tag.server.log" 2>&1 &)
    for _ in $(seq 300); do "$RCLI" -p 1974 ping >/dev/null 2>&1 && return 0; sleep 0.1; done
    fail "pion --io-threads $io did not answer"
}

threads_cpu() {   # pid -> "tid utime stime" per thread (clock ticks)
    for t in /proc/"$1"/task/*; do
        awk -v tid="${t##*/}" '{print tid, $14, $15}' "$t/stat" 2>/dev/null
    done
}

mt() {            # name port pipeline [pid]
    local before=""
    [ -n "${4:-}" ] && before=$(threads_cpu "$4")
    memtier_benchmark -s 127.0.0.1 -p "$2" -t "$CLIENT_T" -c 50 --pipeline="$3" $COMMON \
        --json-out-file="$RAW/$1.json" > "$RAW/$1.txt" 2>&1 || echo "memtier exit $? for $1"
    if [ -n "${4:-}" ]; then
        echo "$before" > "$RAW/$1.threads_before"
        threads_cpu "$4" > "$RAW/$1.threads_after"
    fi
}

[ -x "$PION" ] || fail "no pion-server at $PION"
"$PION" --version > "$OUT/pion_version.txt" 2>&1
"$REDIS" --version > "$OUT/redis_version.txt"
memtier_benchmark --version > "$OUT/memtier_version.txt" 2>&1
lscpu > "$OUT/lscpu.txt"; uname -a > "$OUT/uname.txt"
echo "REPS=$REPS TEST_TIME=$TEST_TIME CLIENT_T=$CLIENT_T COMMON=$COMMON" > "$OUT/settings.txt"
stop_all

for rep in $(seq "$REPS"); do
    for P in 1 10 50; do
        for io in 1 4 8; do
            stamp "redis io$io P$P r$rep"
            start_redis "$io" none; mt "single_redis_io${io}_P${P}_r$rep" 6379 "$P"; stop_all
        done
        for io in 1 2 4 8; do
            stamp "pion io$io P$P r$rep"
            start_pion "$io" none "single_pion_io${io}_P${P}_r$rep"
            mt "single_pion_io${io}_P${P}_r$rep" 1974 "$P" "$(pgrep -x pion-server | head -1)"; stop_all
        done
    done
done

for rep in $(seq "$REPS"); do
    for io in 4 8; do
        stamp "durable redis io$io aof r$rep"
        start_redis "$io" aof; mt "durable_redis_io${io}_aof_P10_r$rep" 6379 10; stop_all
    done
    for io in 1 4 8; do
        stamp "durable pion io$io wal r$rep"
        start_pion "$io" wal "durable_pion_io${io}_wal_P10_r$rep"
        mt "durable_pion_io${io}_wal_P10_r$rep" 1974 10 "$(pgrep -x pion-server | head -1)"; stop_all
    done
done

stamp "summary"
python3 "$HERE/../results/2026-10-06-linux-epyc-8124p/summarize.py" "$OUT" > "$OUT/summary.md" 2>&1 || echo "summarize failed"
python3 "$HERE/thread_cpu.py" "$RAW" > "$OUT/thread_cpu.md" 2>&1 || echo "thread_cpu failed"
touch "$OUT/DONE"
stamp "done"
