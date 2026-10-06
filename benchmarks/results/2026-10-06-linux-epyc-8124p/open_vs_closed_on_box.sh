#!/bin/bash
# Phase 2 on the same box: the closed vector library against `pixi run build-open`.
# Both binaries are built here from the same commit with the same CPU target
# (x86-64-v2: build-portable links libpion_vector, build-open does not), so the
# only difference is the vector kernels. Arms alternate closed/open (ABBA x3).
#   nohup bash open_vs_closed_on_box.sh > ~/ovc.log 2>&1 &
set -u
OUT=$HOME/ovc; mkdir -p "$OUT"
R=/opt/Pion
stamp() { echo "== $(date +%H:%M:%S) $*"; }
as_root() { sudo -E bash -c "cd $R && export PATH=/opt/pixi/bin:\$PATH && $1"; }
stamp start
as_root "pkill -9 -x pion-server; rm -f /opt/pion-host-closed && cp pion-server /opt/pion-host-closed"
as_root "pixi run build-portable" > "$OUT/build_portable.log" 2>&1 || { echo FAILED > "$OUT/FAILED"; exit 1; }
as_root "rm -f /opt/pion-v2-closed && cp pion-server-dev /opt/pion-v2-closed"
as_root "pixi run build-open" > "$OUT/build_open.log" 2>&1 || { echo FAILED > "$OUT/FAILED"; exit 1; }
as_root "rm -f /opt/pion-v2-open && cp pion-server /opt/pion-v2-open"
for b in v2-closed v2-open; do /opt/pion-$b --version > "$OUT/version_$b.txt" 2>&1; done
W=$(lscpu -p=core | grep -v '^#' | sort -u | wc -l); [ "$W" -gt 16 ] && W=16
# Three arms: the library at its default (VNNI where the CPU has it), the library
# forced to its x86-64-v2 kernels (PION_VECTOR_VNNI=0), and the open build.
n=0
for arm in closed open closed-v2 open closed-v2 closed closed-v2 closed open; do
    n=$((n+1))
    stamp "run $n $arm"
    bin=${arm%-v2}; envset=""; [ "$arm" = closed-v2 ] && envset="export PION_VECTOR_VNNI=0 &&"
    as_root "$envset pkill -9 -x pion-server; rm -f pion.wal.* pion.hnsw.* pion-server && cp /opt/pion-v2-$bin pion-server && python3 benchmarks/VectorDBBench/vectordb-benchmark.py --pion-only --ef-runtime 150 --workers $W --case Performance1536D50K" \
        > "$OUT/run_${n}_$arm.txt" 2>&1
    grep -a "Performance case got result" "$OUT/run_${n}_$arm.txt" | tail -1 | grep -oE " qps=[0-9.]+|recall=np.float64\([0-9.]+" | tr '\n' ' ' | sed "s/^/RESULT run=$n arm=$arm /" >> "$OUT/results.txt"; echo >> "$OUT/results.txt"
done
as_root "pkill -9 -x pion-server; rm -f pion-server && cp /opt/pion-host-closed pion-server"
sudo chown -R "$(id -u)" "$OUT"
tar -C "$HOME" -czf "$HOME/ovc.tgz" ovc
touch "$OUT/DONE"
stamp done
