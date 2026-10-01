from src.vector.hnsw import HNSWGraph
from std.memory.unsafe_pointer import alloc
from std.time import perf_counter_ns
from std.random import random_float64

def benchmark_bq_hnsw():
    try:
        var dim = 128
        var max_elements = 100000
        # Enable BQ
        var graph = HNSWGraph(max_elements, dim, M=16, ef_construction=100, use_bq=True)
        
        print("Inserting", max_elements, "vectors with BQ...")
        var start_insert = perf_counter_ns()
        for i in range(max_elements):
            if i % 10000 == 0:
                print("Progress:", i, "/", max_elements)
            var v = alloc[Float32](dim)
            for j in range(dim):
                v[j] = random_float64().cast[DType.float32]()
            graph.add_vector(i, v)
            v.free()
        var end_insert = perf_counter_ns()
        print("Insertion took:", (end_insert - start_insert).cast[DType.float64]() / 1e9, "seconds")
        
        # Prepare a query
        var query = alloc[Float32](dim)
        for j in range(dim):
            query[j] = 0.5
        var q_quant = graph.quantize(query)
        
        print("Running 10000 searches with BQ...")
        var start_search = perf_counter_ns()
        for i in range(10000):
            _ = graph.search(q_quant, 10)
        var end_search = perf_counter_ns()
        print("10000 searches took:", (end_search - start_search).cast[DType.float64]() / 1e9, "seconds")
        print("Average search latency:", (end_search - start_search).cast[DType.float64]() / 1e6 / 10000, "ms")
        
        query.free()
        
    except e:
        print("Benchmark failed:", e)

def main():
    benchmark_bq_hnsw()
