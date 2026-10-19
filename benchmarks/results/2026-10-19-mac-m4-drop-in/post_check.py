import sys
import numpy as np, pandas as pd, redis
port = int(sys.argv[1]); ds = "/tmp/vectordb_bench/dataset/openai/openai_small_50k"
df = pd.read_parquet(f"{ds}/shuffle_train.parquet")
train = np.stack(df["emb"].to_numpy()).astype(np.float32); ids = df["id"].to_numpy()
test = np.stack(pd.read_parquet(f"{ds}/test.parquet")["emb"].to_numpy()).astype(np.float32)[:200]
gtf = pd.read_parquet(f"{ds}/neighbors.parquet")
gt_vdbb = [set(int(x) for x in row[:100]) for row in gtf["neighbors_id"].to_numpy()[:200]]
tn = train / np.linalg.norm(train, axis=1, keepdims=True); qn = test / np.linalg.norm(test, axis=1, keepdims=True)
gt = np.argsort(-(qn @ tn.T), axis=1)[:, :100]
print("GT agreement (numpy vs dataset neighbors):", np.mean([len(gt_vdbb[i] & set(int(ids[g]) for g in gt[i])) / 100 for i in range(200)]))
for w in range(4):
    c = redis.Redis(port=port + 2 + w)
    rs, dup, n_res = [], 0, 0
    for qi, q in enumerate(test):
        res = c.execute_command("FT.SEARCH", "index", "*=>[KNN 100 @vector $vec EF_RUNTIME 150 as score]",
                                "PARAMS", 2, "vec", q.tobytes(), "SORTBY", "score", "RETURN", 2, "id", "score",
                                "LIMIT", 0, 100, "DIALECT", 2)
        keys = [int(res[j]) for j in range(1, len(res), 2)]
        n_res += len(keys); dup += len(keys) - len(set(keys))
        rs.append(len(set(keys) & gt_vdbb[qi]) / 100)
    i = c.info()
    print(f"w{w}: recall@100 {np.mean(rs):.4f} results/query {n_res/200:.1f} duplicates {dup} "
          f"[{i.get('vector_index')} graph={i.get('vector_graph_rows')} exact={i.get('vector_exact_rows')} keys={i.get('db0')}]")
