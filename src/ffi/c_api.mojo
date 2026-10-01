from src.common.ptr import null_ptr
from src.actor.shard_manager import ShardManager
from src.vector.hnsw import HNSWGraph
from src.common.config import PionConfig
from std.memory.unsafe_pointer import Pointer

@fieldwise_init
struct PionHandle:
    var hnsw: Pointer[HNSWGraph, MutUntrackedOrigin]
    var results_buffer: Pointer[Int32, MutUntrackedOrigin]

@c_call
def pion_init(port: Int32) -> Pointer[PionHandle, MutUntrackedOrigin]:
    """Initialize the Pion engine and return an opaque handle."""
    var config = PionConfig()
    config.server.port = Int(port)
    
    # Initialize HNSW directly for the handle
    var hnsw_ptr = Pointer[HNSWGraph, MutUntrackedOrigin].alloc(1)
    hnsw_ptr.unsafe_write(HNSWGraph(
        config.vector.max_elements,
        config.vector.dimensions,
        M=config.vector.M,
        ef_construction=config.vector.ef_construction,
        use_int4=config.vector.use_int4,
        use_bq=config.vector.use_bq,
        has_gpu=config.vector.has_gpu,
        polarquant=config.vector.polarquant,
    ))
    
    var results_buffer = Pointer[Int32, MutUntrackedOrigin].alloc(1024) # Static result buffer for FFI simplicity
    
    var handle = Pointer[PionHandle, MutUntrackedOrigin].alloc(1)
    handle.unsafe_write(PionHandle(hnsw_ptr, results_buffer))
    return handle

@c_call
def pion_vector_search(
    handle: Pointer[PionHandle, MutUntrackedOrigin], 
    query: Pointer[Float32, MutUntrackedOrigin], 
    dim: Int32, 
    k: Int32
) -> Pointer[Int32, MutUntrackedOrigin]:
    """Perform a vector search directly from C/FFI."""
    try:
        var hnsw = handle[].hnsw
        var k_int = Int(k)
        
        # In a real C-ABI, we'd need to handle quantization of the Float32 query
        # For now, we simulate the path
        var q_vec = hnsw[].quantize(query)
        var results = hnsw[].search(q_vec, k_int, 100)
        
        # Copy results to the FFI buffer
        for i in range(min(k_int, 1024)):
            handle[].results_buffer[i] = Int32(results[i])
            
        hnsw[].vector_allocator.deallocate(q_vec)
        return handle[].results_buffer
    except:
        return null_ptr[Int32, MutUntrackedOrigin]()

@c_call
def pion_shutdown(handle: Pointer[PionHandle, MutUntrackedOrigin]):
    """Clean up and shutdown the Pion engine instance."""
    handle[].hnsw.unsafe_deinit_pointee()
    handle[].hnsw.unsafe_free()
    handle[].results_buffer.unsafe_free()
    handle.unsafe_deinit_pointee()
    handle.unsafe_free()
