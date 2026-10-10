#!/bin/bash
# #465, Mac half: one keyspace, Pion --io-threads N vs Redis io-threads N.
# memtier settings as benchmarks/results/2026-10-06-linux-epyc-8124p/run_on_box.sh
# (256 B values, 1:10 SET:GET, 1M keys, 50 connections per client thread), but
# 4 client threads instead of 8 and 10 s runs: the Mac has 10 cores (4P+6E),
# and 8 client threads would leave the server none. Persistence off on both.
# Rounds are interleaved (every config once per round).
set -u
S=${OUT:-./sweep_mac}
mkdir -p "$S/raw"
PION=${PION:-$HOME/Projects/pion-wt/io/pion-server}
REDIS=/opt/homebrew/bin/redis-server
CLIENT_T=${CLIENT_T:-4}
REPS=${REPS:-3}
TT=${TT:-10}
COMMON="--protocol=redis -d 256 --ratio=1:10 --key-pattern=R:R --key-minimum=1 --key-maximum=1000000 --test-time=$TT --hide-histogram --distinct-client-seed"
"$REDIS" --version > "$S/redis_version.txt"
"$PION" --version > "$S/pion_version.txt"
memtier_benchmark --version > "$S/memtier_version.txt" 2>&1
sysctl -n machdep.cpu.brand_string hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu > "$S/cpu.txt"
echo "CLIENT_T=$CLIENT_T REPS=$REPS COMMON=$COMMON" > "$S/settings.txt"

stop_all() { pkill -x pion-server 2>/dev/null; pkill -f 'redis-serve[r].*:7901' 2>/dev/null; sleep 2; }
wait_port() { for _ in $(seq 300); do redis-cli -p "$1" ping 2>/dev/null | grep -q PONG && return 0; sleep 0.1; done; echo "no answer on $1"; return 1; }
mt() { memtier_benchmark -s 127.0.0.1 -p "$2" -t "$CLIENT_T" -c 50 --pipeline="$3" $COMMON \
        --json-out-file="$S/raw/$1.json" > "$S/raw/$1.txt" 2>&1 || echo "memtier exit $? for $1"; }

for rep in $(seq "$REPS"); do
  for P in 1 10 50; do
    for io in 1 2 4; do
      echo "== $(date +%H:%M:%S) pion io$io P$P r$rep"
      rm -rf /tmp/pion465s && mkdir -p /tmp/pion465s
      args=(-p 7900 -w 1 --profile kv --no-wal --no-auto-detect --no-auto-embed --no-crash-log)
      [ "$io" -gt 1 ] && args+=(--io-threads "$io")
      (cd /tmp/pion465s && "$PION" "${args[@]}" > "$S/raw/pion_io${io}_P${P}_r$rep.server.log" 2>&1 &)
      wait_port 7900 && mt "pion_io${io}_P${P}_r$rep" 7900 "$P"
      stop_all
      echo "== $(date +%H:%M:%S) redis io$io P$P r$rep"
      rm -rf /tmp/redis465s && mkdir -p /tmp/redis465s
      "$REDIS" --port 7901 --bind 127.0.0.1 --dir /tmp/redis465s --io-threads "$io" --save "" \
          --appendonly no --daemonize yes --loglevel warning
      wait_port 7901 && mt "redis_io${io}_P${P}_r$rep" 7901 "$P"
      redis-cli -p 7901 shutdown nosave >/dev/null 2>&1; stop_all
    done
  done
done
echo DONE
