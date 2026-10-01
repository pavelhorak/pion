from src.vector.hnsw import HNSWGraph
from std.memory.unsafe_pointer import alloc
from std.time import perf_counter_ns
from std.random import random_float64

def benchmark_hnsw():
    try:
        var dim = 1536
        var max_elements = 50000
        var graph = HNSWGraph(max_elements, dim, M=16, ef_construction=128)
        
        print("Inserting", max_elements, "vectors...")
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
        
        print("Building index...")
        var start_build = perf_counter_ns()
        graph.build_index()
        var end_build = perf_counter_ns()
        print("Build took:", (end_build - start_build).cast[DType.float64]() / 1e9, "seconds")
        
        # Prepare a query
        var query = alloc[Float32](dim)
        print("Running 1000 searches across random query vectors...")
        var start_search = perf_counter_ns()
        var total_results = 0
        for i in range(1000):
            var query_idx = (i * 123) % max_elements
            var offset = query_idx * dim
            for j in range(dim):
                query[j] = graph.fp32_buffer[offset + j]
            var q_quant = graph.quantize(query)
            var res = graph.search(q_quant, 10, ef=100)
            total_results += len(res)
        var end_search = perf_counter_ns()
        print("1000 searches took:", (end_search - start_search).cast[DType.float64]() / 1e9, "seconds")
        print("Average search latency:", (end_search - start_search).cast[DType.float64]() / 1e6 / 1000, "ms")
        print("Average results per search:", Float64(total_results) / 1000.0)
        
        query.free()
        
    except e:
        print("Benchmark failed:", e)

def main():
    benchmark_hnsw()
