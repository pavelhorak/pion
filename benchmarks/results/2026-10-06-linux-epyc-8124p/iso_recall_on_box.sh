#!/bin/bash
# Phase 3, after open_vs_closed_on_box.sh: Pion's search ef swept so its recall can
# be read at Redis vector sets' default (0.920 here). Same harness and settings as
# the phase-1 vector runs; only --ef-runtime changes. Writes ~/iso/DONE.
set -u
OUT=$HOME/iso; mkdir -p "$OUT"
until [ -f "$HOME/ovc/DONE" ] || [ -f "$HOME/ovc/FAILED" ]; do sleep 20; done
R=/opt/Pion
W=$(lscpu -p=core | grep -v '^#' | sort -u | wc -l); [ "$W" -gt 16 ] && W=16
stamp() { echo "== $(date +%H:%M:%S) $*"; }
for ef in 32 48 64 100 150; do
    stamp "ef $ef"
    sudo -E bash -c "cd $R && export PATH=/opt/pixi/bin:\$PATH && pkill -9 -x pion-server; rm -f pion.wal.* pion.hnsw.*; python3 benchmarks/VectorDBBench/vectordb-benchmark.py --pion-only --ef-runtime $ef --workers $W --case Performance1536D50K" \
        > "$OUT/pion_ef${ef}.txt" 2>&1
    grep -a "Performance case got result" "$OUT/pion_ef${ef}.txt" | tail -1 | grep -oE " qps=[0-9.]+|recall=np.float64\([0-9.]+|conc_qps_list=\[[^]]*\]" | tr '\n' ' ' | sed "s/^/RESULT ef=$ef /" >> "$OUT/results.txt"; echo >> "$OUT/results.txt"
done
sudo chown -R "$(id -u)" "$OUT"
tar -C "$HOME" -czf "$HOME/iso.tgz" iso
touch "$OUT/DONE"
stamp done
