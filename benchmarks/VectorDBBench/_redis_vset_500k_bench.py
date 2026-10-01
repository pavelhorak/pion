#!/usr/bin/env python3
"""Redis Vector Sets (VADD/VSIM) benchmark at Performance1536D500K scale.

VADD syntax for Redis 8.6.1:
  VADD key FP32 <binary_blob> element_name  (element name is LAST)
  VSIM key FP32 <binary_blob> COUNT k

Dataset: /tmp/vectordb_bench/dataset/openai/openai_medium_500k/
"""

import subprocess, time, sys, os, threading
import numpy as np
import pyarrow.parquet as pq
import redis as redis_lib

REDIS_PORT = 16390
DATASET_DIR = "/tmp/vectordb_bench/dataset/openai/openai_medium_500k"
DIM = 1536
K = 100
CONCURRENCY_LEVELS = [1, 5, 10]
SEARCH_DURATION = 5  # seconds per concurrency level
INSERT_BATCH = 500


def start_redis():
    subprocess.run(f"lsof -ti:{REDIS_PORT} | xargs kill -9", shell=True,
                   stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    proc = subprocess.Popen(
        f"redis-server --port {REDIS_PORT} --save '' --loglevel warning --daemonize no",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(2)
    if proc.poll() is not None:
        print("Redis failed to start"); sys.exit(1)
    return proc


def stop_redis(proc):
    proc.terminate()
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired: proc.kill()
    subprocess.run(f"lsof -ti:{REDIS_PORT} | xargs kill -9", shell=True,
                   stderr=subprocess.DEVNULL)


def main():
    # Load test queries and ground truth (small — safe to hold in RAM)
    print(f"[Redis VSET 500K] Loading test/neighbors ...", flush=True)
    test_table = pq.read_table(f"{DATASET_DIR}/test.parquet")
    test_vecs = np.array(test_table["emb"].to_pylist(), dtype=np.float32)
    print(f"  Test queries: {len(test_vecs):,} × {DIM}")

    neighbors_table = pq.read_table(f"{DATASET_DIR}/neighbors.parquet")
    neighbors = np.array(neighbors_table["neighbors_id"].to_pylist(), dtype=np.int64)
    print(f"  Ground truth: {neighbors.shape}")

    # Start Redis
    print(f"\n[Redis VSET 500K] Starting redis-server on port {REDIS_PORT} ...", flush=True)
    proc = start_redis()
    r = redis_lib.Redis(port=REDIS_PORT, decode_responses=False)

    # Stream-insert from parquet (avoid loading full 3GB numpy array)
    print(f"[Redis VSET 500K] Streaming insert from parquet ...", flush=True)
    t0 = time.time()
    total_inserted = 0

    pf = pq.ParquetFile(f"{DATASET_DIR}/shuffle_train.parquet")
    pipe = r.pipeline(transaction=False)
    batch_count = 0

    for batch in pf.iter_batches(batch_size=INSERT_BATCH):
        emb_col = batch.column("emb")
        id_col = batch.column("id")
        for row_i in range(len(batch)):
            vec = np.array(emb_col[row_i].as_py(), dtype=np.float32)
            train_id = id_col[row_i].as_py()
            pipe.execute_command('VADD', 'idx', 'FP32', vec.tobytes(), str(train_id))
            batch_count += 1

        if batch_count >= INSERT_BATCH:
            pipe.execute()
            pipe = r.pipeline(transaction=False)
            total_inserted += batch_count
            batch_count = 0
            if total_inserted % 50000 == 0:
                elapsed = time.time() - t0
                print(f"  {total_inserted:,}/500,000 ({elapsed:.1f}s)", flush=True)

    if batch_count > 0:
        pipe.execute()
        total_inserted += batch_count

    load_time = time.time() - t0
    vcard = r.execute_command('VCARD', 'idx')
    print(f"[Redis VSET 500K] Insert complete: {vcard:,} vectors in {load_time:.1f}s")

    # Search benchmark
    print(f"\n[Redis VSET 500K] Running search benchmark ...", flush=True)
    nq = len(test_vecs)
    results_by_conc = {}
    recall_result = [0.0]

    for conc in CONCURRENCY_LEVELS:
        queries_done = [0]
        latencies = []
        lock = threading.Lock()
        stop_flag = [False]
        recall_buf = []

        def worker(tid, _stop=stop_flag, _done=queries_done, _lat=latencies,
                   _lock=lock, _rbuf=recall_buf, _conc=conc):
            local_r = redis_lib.Redis(port=REDIS_PORT, decode_responses=False)
            qidx = tid
            while not _stop[0]:
                q = test_vecs[qidx % nq].tobytes()
                t_s = time.time()
                res = local_r.execute_command('VSIM', 'idx', 'FP32', q, 'COUNT', K)
                t_e = time.time()
                with _lock:
                    _done[0] += 1
                    _lat.append(t_e - t_s)
                    if _conc == 1 and len(_rbuf) < nq:
                        _rbuf.append([int(x) for x in res])
                qidx += _conc

        threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                   for i in range(conc)]
        for t in threads: t.start()
        time.sleep(SEARCH_DURATION)
        stop_flag[0] = True
        for t in threads: t.join(timeout=10)

        qps = queries_done[0] / SEARCH_DURATION
        p99 = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
        results_by_conc[conc] = (qps, p99)
        print(f"  c={conc}: {qps:.0f} QPS, P99={p99*1000:.1f}ms", flush=True)

        if conc == 1 and recall_buf:
            hits = 0
            total = 0
            gt = neighbors[:len(recall_buf), :K]
            for i, res_ids in enumerate(recall_buf):
                if i >= len(gt): break
                hits += len(set(res_ids[:K]) & set(gt[i].tolist()))
                total += K
            recall_result[0] = hits / total if total > 0 else 0.0
            print(f"  Recall@{K}: {recall_result[0]:.4f}", flush=True)

    stop_redis(proc)

    peak_qps = max(v[0] for v in results_by_conc.values())
    p99_c1 = results_by_conc.get(1, (0, 0))[1]
    recall = recall_result[0]

    print("\n" + "=" * 70)
    print("Redis Vector Sets (VADD/VSIM) — Performance1536D500K")
    print("=" * 70)
    print(f"| Metric          | Redis VSET   | Pion (ef=150) |")
    print(f"|:----------------|:------------:|:-------------:|")
    print(f"| Dataset         | 500K × 1536  | 500K × 1536   |")
    print(f"| Load time       | {load_time:.1f}s      | 298.0s        |")
    print(f"| c=1 QPS         | {results_by_conc[1][0]:.0f}       | —             |")
    print(f"| c=5 QPS         | {results_by_conc[5][0]:.0f}       | —             |")
    print(f"| c=10 QPS        | {results_by_conc[10][0]:.0f}       | 7,418         |")
    print(f"| Recall@{K}     | {recall:.4f}      | 0.9167        |")
    print(f"| P99 latency     | {p99_c1*1000:.1f}ms       | 0.9ms         |")
    print("=" * 70)


if __name__ == "__main__":
    main()
