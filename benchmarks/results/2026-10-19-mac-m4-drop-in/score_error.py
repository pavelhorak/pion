import sys, time
import numpy as np, pandas as pd, redis
port = int(sys.argv[1]); mode = sys.argv[2]; ds = "/tmp/vectordb_bench/dataset/openai/openai_small_50k"
df = pd.read_parquet(f"{ds}/shuffle_train.parquet")
train = np.stack(df["emb"].to_numpy()).astype(np.float32); ids = df["id"].to_numpy()
test = np.stack(pd.read_parquet(f"{ds}/test.parquet")["emb"].to_numpy()).astype(np.float32)[:20]
r = redis.Redis(port=port)
r.execute_command("FT.CREATE", "index", "ON", "HASH", "SCHEMA", "id", "TAG", "vector", "VECTOR", "HNSW", "6",
                  "TYPE", "FLOAT32", "DIM", "1536", "DISTANCE_METRIC", "COSINE")
p = r.pipeline(transaction=False)
for i in range(20000): p.hset(int(ids[i]), mapping={"id": str(int(ids[i])), "vector": train[i].tobytes()})
p.execute()
if mode == "optimize": r.execute_command("FT.OPTIMIZE", "index")
else:
    while r.info().get("vector_index") != "graph": time.sleep(0.2)
tn = train[:20000] / np.linalg.norm(train[:20000], axis=1, keepdims=True)
pos = {int(ids[i]): i for i in range(20000)}
errs = []
for q in test:
    res = r.execute_command("FT.SEARCH", "index", "*=>[KNN 10 @vector $v AS score]", "PARAMS", 2, "v", q.tobytes(),
                            "RETURN", 2, "id", "score", "DIALECT", 2)
    qn = q / np.linalg.norm(q)
    for j in range(1, len(res), 2):
        f = res[j + 1]; d = {f[x]: f[x + 1] for x in range(0, len(f), 2)}
        errs.append(float(d[b"score"]) - (1 - float(tn[pos[int(d[b"id"])]] @ qn)))
print(mode, f"score - exact: mean {np.mean(errs):+.4f} min {np.min(errs):+.4f} max {np.max(errs):+.4f}")
