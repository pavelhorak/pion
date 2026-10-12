#!/bin/bash
# gh #480 follow-up, pre-registered 2026-10-12 ~12:08 UTC, before any of its data.
# M2 pinned the server to {2,18} and gave the client every other CPU; every row then
# sat in the LOW mode (~590K) and the client ran on another L3 (CCX: 4 cores each,
# {0-3,16-19} {4-7,20-23} {8-11,24-27} {12-15,28-31}). Hypothesis: the ~740K mode of
# the unpinned 10-09 runs is the client landing in the server's L3.
# Arms (server always pinned), 3 rounds each, interleaved:
#   same_l3  : server {2,18}, client {0,1,3,16,17,19}  -> predicted ~740K
#   other_l3 : server {2,18}, client {4-7,20-23}       -> predicted ~590K
#   smt      : server {2},    client {18}              -> no prediction
set -uo pipefail
O=/opt/session/out/f480b; mkdir -p "$O"
until [ -f /opt/session/out/ALL_DONE ] || [ -f /opt/session/out/STOPPED ]; do sleep 5; done
[ -f /opt/session/out/STOPPED ] && exit 1
BIN=/opt/arms/native/pion-server
cd /opt/wt/main
export PATH=/opt/redis-8.10.2/src:/opt/pixi/bin:$PATH
run() {  # tag server-cpus client-cpus
  printf '#!/bin/bash\nexec taskset -c %s %s "$@"\n' "$2" "$BIN" > ./pion-server && chmod +x ./pion-server
  ( while sleep 3; do
      for p in $(pgrep -x pion-server) $(pgrep -x redis-benchmark); do
        echo "$(date -u +%T) $p $(ps -o comm=,psr= -p "$p" 2>/dev/null)"
      done
    done ) > "$O/$1_placement.txt" 2>&1 &
  local s=$!
  taskset -c "$3" python3 benchmarks/valkey-benchmark/valkey-benchmark.py -c 50 -n 100000 -P 10 -w 1 \
    --pion-only --gate --gate-profile linux-epyc-8124p > "$O/$1_kv.log" 2>&1
  echo "$(date -u +%T) $1 rc=$? server=$2 client=$3" >> "$O/progress.txt"
  kill "$s" 2>/dev/null; pkill -x pion-server 2>/dev/null; sleep 2; pkill -9 -x pion-server 2>/dev/null
}
for r in 1 2 3; do
  run "same_l3_r$r"  2,18 0,1,3,16,17,19
  run "other_l3_r$r" 2,18 4-7,20-23
  run "smt_r$r"      2    18
done
rm -f ./pion-server
touch "$O/DONE"
