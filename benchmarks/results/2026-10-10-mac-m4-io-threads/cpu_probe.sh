#!/bin/bash
# Where does the CPU go at P=1 on the Mac, --io-threads 1 vs 4? memtier's %CPU and
# the server's per-thread %CPU, sampled 5 s into a 10 s run (sweep settings).
S=${OUT:-./cpu_probe}
mkdir -p "$S"
PION=${PION:-$HOME/Projects/pion-wt/io/pion-server}
COMMON="--protocol=redis -d 256 --ratio=1:10 --key-pattern=R:R --key-minimum=1 --key-maximum=1000000 --test-time=10 --hide-histogram --distinct-client-seed"
for P in 1 10; do
for io in 1 4; do
    rm -rf /tmp/pion465c && mkdir -p /tmp/pion465c
    args=(-p 7902 -w 1 --profile kv --no-wal --no-auto-detect --no-auto-embed --no-crash-log)
    [ "$io" -gt 1 ] && args+=(--io-threads "$io")
    (cd /tmp/pion465c && "$PION" "${args[@]}" > "$S/server_io$io.log" 2>&1 &)
    for _ in $(seq 300); do redis-cli -p 7902 ping 2>/dev/null | grep -q PONG && break; sleep 0.1; done
    memtier_benchmark -s 127.0.0.1 -p 7902 -t 4 -c 50 --pipeline=$P $COMMON > "$S/mt_io${io}_P$P.txt" 2>&1 &
    sleep 5
    pid=$(pgrep -x pion-server | head -1)
    {
      echo "== io$io P$P"
      ps -o %cpu=,comm= -p "$(pgrep -x memtier_benchmark | head -1)"
      ps -M -p "$pid" | awk 'NR==2 {print $4} NR>2 {print $2}' | sort -rn | head -6 | tr '\n' ' '; echo
      top -l 2 -s 1 -n 0 | grep "CPU usage" | tail -1
    } >> "$S/cpu.txt"
    wait
    grep Totals "$S/mt_io${io}_P$P.txt" | awk -v t="io$io P$P" '{print t, $2}' >> "$S/cpu.txt"
    pkill -x pion-server; sleep 2
done
done
cat "$S/cpu.txt"
