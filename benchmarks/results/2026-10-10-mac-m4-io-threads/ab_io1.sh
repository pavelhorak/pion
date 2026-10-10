#!/bin/bash
# #465 acceptance item: --io-threads 1 (the default) shows no regression on the
# Mac KV gate. Paired, interleaved A/B of the gate harness: A = main's binary
# (~/Projects/pion, 805636b), B = the branch (~/Projects/pion-wt/io). Order
# ABBA ABBA so neither binary always runs first after a swap.
set -u
S=${OUT:-./ab_io1}
mkdir -p "$S"
run() {   # tag tree idx
    local tag=$1 tree=$2 i=$3
    echo "== $(date +%H:%M:%S) $tag run $i ($tree)"
    python3 "$tree/benchmarks/valkey-benchmark/valkey-benchmark.py" -c 50 -n 100000 -P 10 -w 1 \
        --pion-only --gate --output "$S/${tag}_$i.md" > "$S/${tag}_$i.log" 2>&1
    echo "   exit $?"
    pkill -x pion-server 2>/dev/null; sleep 3
}
A=${A:-$HOME/Projects/pion}
B=${B:-$HOME/Projects/pion-wt/io}
i=0
for round in 1 2; do
    i=$((i+1)); run main "$A" $i
    run branch "$B" $i
    i=$((i+1)); run branch "$B" $i
    run main "$A" $i
done
echo DONE
