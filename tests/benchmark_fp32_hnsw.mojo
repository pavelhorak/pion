from src.vector.hnsw import HNSWGraph
from std.memory.unsafe_pointer import alloc, UnsafePointer
from std.time import perf_counter_ns
from std.random import random_float64

def benchmark_fp32_hnsw():
    try:
        var dim = 128
        var max_elements = 100000
        # Use common dimension 128 to trigger JIT fused kernel
        var graph = HNSWGraph(max_elements, dim, M=16, ef_construction=100)
        
        print("Inserting", max_elements, "vectors of dimension", dim, "...")
        var start_insert = perf_counter_ns()
        for i in range(max_elements):
            if i % 20000 == 0 and i > 0:
                print("Progress:", i, "/", max_elements)
            var v = alloc[Float32](dim)
            for j in range(dim):
                v[j] = random_float64(0, 1).cast[DType.float32]()
            graph.add_vector(i, v)
            v.free()
        var end_insert = perf_counter_ns()
        print("Insertion took:", (end_insert - start_insert).cast[DType.float64]() / 1e9, "seconds")
        
        # Prepare a Float32 query
        var query = alloc[Float32](dim)
        for j in range(dim):
            query[j] = 0.5
        
        print("Running 10000 searches using FP32 Fused JIT kernel...")
        var start_search = perf_counter_ns()
        for i in range(10000):
            _ = graph.search_fp32(query, 10)
        var end_search = perf_counter_ns()
        
        var total_time_ns = (end_search - start_search).cast[DType.float64]()
        print("10000 searches took:", total_time_ns / 1e9, "seconds")
        print("Average search latency:", total_time_ns / 1e6 / 10000, "ms")
        
        query.free()
        
    except e:
        print("Benchmark failed:", e)

def main():
    benchmark_fp32_hnsw()
