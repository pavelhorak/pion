import sys, time
import numpy as np, pandas as pd, redis
port = int(sys.argv[1]); reps = int(sys.argv[2]); ds = "/tmp/vectordb_bench/dataset/openai/openai_small_50k"
df = pd.read_parquet(f"{ds}/shuffle_train.parquet")
train = np.stack(df["emb"].to_numpy()).astype(np.float32); ids = df["id"].to_numpy()
test = np.stack(pd.read_parquet(f"{ds}/test.parquet")["emb"].to_numpy()).astype(np.float32)[:300]
gt = [set(int(x) for x in row[:100]) for row in pd.read_parquet(f"{ds}/neighbors.parquet")["neighbors_id"].to_numpy()[:300]]
r = redis.Redis(port=port)
def rec(cport):
    c = redis.Redis(port=cport); rs = []
    for qi, q in enumerate(test):
        res = c.execute_command("FT.SEARCH", "index", "*=>[KNN 100 @vector $vec EF_RUNTIME 150 as score]",
                                "PARAMS", 2, "vec", q.tobytes(), "RETURN", 1, "id", "LIMIT", 0, 100, "DIALECT", 2)
        rs.append(len({int(res[j]) for j in range(1, len(res), 2)} & gt[qi]) / 100)
    return np.mean(rs)
for rep in range(reps):
    r.execute_command("FT.CREATE", "index", "ON", "HASH", "SCHEMA", "id", "TAG", "vector", "VECTOR", "HNSW", "10",
                      "TYPE", "FLOAT32", "DIM", "1536", "DISTANCE_METRIC", "COSINE", "M", "16", "EF_CONSTRUCTION", "128")
    p = r.pipeline(transaction=False)
    for i in range(50000):
        p.hset(int(ids[i]), mapping={"id": str(int(ids[i])), "vector": train[i].tobytes()})
        if i % 2000 == 1999: p.execute()
    p.execute()
    while r.info().get("vector_index") != "graph": time.sleep(0.2)
    print(f"build {rep}: w0 {rec(port + 2):.4f} w2 {rec(port + 4):.4f}", flush=True)
    r.execute_command("FT.DROPINDEX", "index")
    r.flushall()
