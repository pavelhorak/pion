#!/bin/bash
# KV head-to-head on ONE machine: Pion (public main) vs Redis 8.10.2, the same
# memtier build and settings for both. Replaces the README's "14.0M ops/s,
# 10.2x Redis", which had no surviving log and compared Pion's 32 independent
# keyspaces with one Redis instance, each at its own best config.
#
#   single keyspace : Redis (io-threads 1, 4, 8)   vs Pion -w 1
#   all cores       : N Redis instances            vs Pion -w N --independent-workers
#                     (N independent keyspaces either way: the same semantics)
#
# Runs every phase itself and ends by writing $OUT/DONE or $OUT/FAILED.
#   nohup bash run_on_box.sh > ~/kvbench.log 2>&1 &
set -u
OUT=${OUT:-$HOME/kvbench}; RAW=$OUT/raw; mkdir -p "$RAW"
PION=${PION:-/opt/Pion/pion-server}
REDIS_TAG=8.10.2
REDIS=$HOME/redis-$REDIS_TAG/src/redis-server
RCLI=$HOME/redis-$REDIS_TAG/src/redis-cli
REPS=${REPS:-3}
TEST_TIME=${TEST_TIME:-30}
CLIENT_T=${CLIENT_T:-8}      # memtier threads, single keyspace
SHARD_T=${SHARD_T:-16}       # total memtier threads, all cores
COMMON="--protocol=redis -d 256 --ratio=1:10 --key-pattern=R:R --key-minimum=1 --key-maximum=1000000 --test-time=$TEST_TIME --hide-histogram --distinct-client-seed"

fail() { echo "FAILED: $*"; echo "$*" > "$OUT/FAILED"; exit 1; }
stamp() { echo "== $(date +%H:%M:%S) $*"; }

stop_all() {
    pkill -x pion-server 2>/dev/null; pkill -x redis-server 2>/dev/null; sleep 2
    pkill -9 -x pion-server 2>/dev/null; pkill -9 -x redis-server 2>/dev/null; sleep 1
}

start_redis() {   # port io-threads persistence(none|aof)
    local p=$1 io=$2 pers=$3
    rm -rf "/tmp/r$p" && mkdir -p "/tmp/r$p"
    local args=(--port "$p" --bind 127.0.0.1 --protected-mode no --daemonize yes --dir "/tmp/r$p"
                --io-threads "$io" --save "" --loglevel warning)
    if [ "$pers" = aof ]; then args+=(--appendonly yes --appendfsync everysec); else args+=(--appendonly no); fi
    "$REDIS" "${args[@]}" || fail "redis $p did not start"
    for _ in $(seq 100); do "$RCLI" -p "$p" ping >/dev/null 2>&1 && return 0; sleep 0.1; done
    fail "redis $p did not answer"
}

start_pion() {    # workers persistence(none|wal)
    local w=$1 pers=$2
    local args=(-p 1974 -w "$w" --profile kv --no-auto-detect --no-auto-embed --no-crash-log)
    [ "$w" -gt 1 ] && args+=(--independent-workers)
    [ "$pers" = none ] && args+=(--no-wal)
    rm -rf /tmp/pion && mkdir -p /tmp/pion
    (cd /tmp/pion && "$PION" "${args[@]}" > "$OUT/pion_server_w$w.log" 2>&1 &)
    for _ in $(seq 300); do "$RCLI" -p 1974 ping >/dev/null 2>&1 && return 0; sleep 0.1; done
    fail "pion -w $w did not answer"
}

mt() {            # name port threads clients pipeline
    memtier_benchmark -s 127.0.0.1 -p "$2" -t "$3" -c "$4" --pipeline="$5" $COMMON \
        --json-out-file="$RAW/$1.json" > "$RAW/$1.txt" 2>&1 || echo "memtier exit $? for $1"
}

# ── build: the release's linux-x86_64 binary is `pixi run build-portable`
# (x86-64-v2, closed vector library, no CUDA); `pixi run build` on linux-64 needs nvcc.
PION_REPO=${PION_REPO:-/opt/Pion}
git config --global --add safe.directory "$PION_REPO"
if [ ! -x "$PION" ]; then
    stamp "build-portable"
    sudo -E bash -c "cd $PION_REPO && export PATH=/opt/pixi/bin:\$PATH && pixi run build-portable && rm -f pion-server && cp pion-server-dev pion-server" \
        > "$OUT/pion_build.log" 2>&1 || fail "pion build-portable (see pion_build.log)"
fi
echo "pixi run build-portable (x86-64-v2 target, closed libpion_vector; the release's linux-x86_64 build)" > "$OUT/pion_build_task.txt"

# ── machine and versions ──────────────────────────────────────────────────────
stamp "facts"
lscpu > "$OUT/lscpu.txt"; uname -a > "$OUT/uname.txt"; free -g > "$OUT/free.txt"
CORES=$(lscpu -p=core | grep -v '^#' | sort -u | wc -l)
echo "$CORES" > "$OUT/physical_cores.txt"
stop_all

stamp "redis $REDIS_TAG"
if [ ! -x "$REDIS" ]; then
    (cd "$HOME" && rm -rf "redis-$REDIS_TAG" && git clone -q --depth 1 --branch "$REDIS_TAG" https://github.com/redis/redis.git "redis-$REDIS_TAG" \
        && make -C "redis-$REDIS_TAG" -j"$(nproc)" BUILD_TLS=no > "$OUT/redis_build.log" 2>&1) || fail "redis build"
fi
"$REDIS" --version > "$OUT/redis_version.txt"
"$PION" --version > "$OUT/pion_version.txt" 2>&1 || fail "no pion-server at $PION"
git -C "$PION_REPO" rev-parse HEAD > "$OUT/pion_commit.txt"
memtier_benchmark --version > "$OUT/memtier_version.txt" 2>&1
echo "REPS=$REPS TEST_TIME=$TEST_TIME CLIENT_T=$CLIENT_T SHARD_T=$SHARD_T COMMON=$COMMON" > "$OUT/settings.txt"

# ── single keyspace, no persistence ───────────────────────────────────────────
for rep in $(seq "$REPS"); do
    for P in 1 10 50; do
        for io in 1 4 8; do
            stamp "single redis io$io P$P r$rep"
            start_redis 6379 "$io" none; mt "single_redis_io${io}_P${P}_r$rep" 6379 "$CLIENT_T" 50 "$P"; stop_all
        done
        stamp "single pion w1 P$P r$rep"
        start_pion 1 none; mt "single_pion_w1_P${P}_r$rep" 1974 "$CLIENT_T" 50 "$P"; stop_all
    done
done

# ── single keyspace, durable: AOF everysec vs Pion's WAL ─────────────────────
for rep in $(seq "$REPS"); do
    stamp "durable P10 r$rep"
    start_redis 6379 4 aof; mt "durable_redis_io4_aof_P10_r$rep" 6379 "$CLIENT_T" 50 10; stop_all
    start_pion 1 wal; mt "durable_pion_w1_wal_P10_r$rep" 1974 "$CLIENT_T" 50 10; stop_all
done

# ── all cores: N independent keyspaces either way ────────────────────────────
N=$CORES
PER=$(( SHARD_T / N )); [ "$PER" -lt 1 ] && PER=1
for rep in $(seq "$REPS"); do
    for P in 10 50; do
        stamp "shard pion w$N P$P r$rep"
        start_pion "$N" none; mt "shard_pion_w${N}_P${P}_r$rep" 1974 "$SHARD_T" 50 "$P"; stop_all
        stamp "shard redis x$N P$P r$rep"
        for i in $(seq 0 $((N - 1))); do start_redis $((6400 + i)) 1 none; done
        for i in $(seq 0 $((N - 1))); do
            mt "shard_redis_x${N}_P${P}_r${rep}_i$i" $((6400 + i)) "$PER" 50 "$P" &
        done
        wait
        stop_all
    done
done

# ── vector: Pion vs Redis 8.10.2 VSET, VectorDBBench Performance1536D50K ─────
# The repo's own harnesses (vectordb-benchmark.py, vset-benchmark.py) on this
# machine, alternating engines, every output kept under $OUT/vec.
VEC=$OUT/vec; mkdir -p "$VEC"
stop_all
stamp "vector setup"
sudo -E bash -c "cd $PION_REPO && export PATH=/opt/pixi/bin:\$PATH && { [ -x venv_zvec/bin/vectordbbench ] || pixi run install-vdbbench; }" \
    > "$VEC/setup.log" 2>&1 || echo "vector setup failed (see $VEC/setup.log)"
W=$CORES; [ "$W" -gt 16 ] && W=16
for rep in $(seq "$REPS"); do
    stamp "vector pion w$W r$rep"
    sudo -E bash -c "cd $PION_REPO && export PATH=/opt/pixi/bin:\$PATH && pkill -9 -x pion-server; rm -f pion.wal.* pion.hnsw.*; python3 benchmarks/VectorDBBench/vectordb-benchmark.py --pion-only --ef-runtime 150 --workers $W --case Performance1536D50K" \
        > "$VEC/pion_w${W}_r$rep.txt" 2>&1
    stamp "vector redis r$rep"
    sudo -E bash -c "cd $PION_REPO && export PATH=/opt/pixi/bin:\$PATH && pkill -9 -x redis-server; python3 benchmarks/VectorDBBench/vset-benchmark.py --redis-server $REDIS" \
        > "$VEC/redis_vset_r$rep.txt" 2>&1
    stop_all
done
sudo cp "$PION_REPO/benchmarks/VectorDBBench/benchmark_results.md" "$VEC/benchmark_results_box.md" 2>/dev/null
sudo chown -R "$(id -u)" "$VEC"

stamp "summary"
python3 "$(dirname "$0")/summarize.py" "$OUT" > "$OUT/summary.md" 2>&1 || echo "summarize failed"
tar -C "$(dirname "$OUT")" -czf "$OUT.tgz" "$(basename "$OUT")"
touch "$OUT/DONE"
stamp "done"
