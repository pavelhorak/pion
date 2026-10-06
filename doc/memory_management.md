# Memory Management: Deterministic Allocation

**Pion** is a unified, shared-nothing database engine built in Mojo. Its performance is rooted in its explicit control over memory. By bypassing the standard operating system allocator for hot-path operations, Pion eliminates the unpredictable latency (jitter) associated with global memory locks and fragmentation.

## Design Constraint: No Allocation in the Event Loop

The fundamental rule governing Pion's memory architecture: **no `alloc[T]` calls in the event loop**. All memory required for request processing is pre-allocated at startup or acquired from pools/slabs. The only exception is lazy first-EAGAIN per-fd pending buffer allocation in `flush_response`.

Pre-allocated resources per worker:
- `response_buffer` (4 MB) for response formatting
- `SlabAllocator[ListNode]` (10M initial) for list node storage
- `ObjectPool[SlabHashMap]` for SADD/HSET new-key allocation
- `stack_allocation[1, KEvent]()` for kevent changes (never `alloc[KEvent](1)`)

## Slab Allocator

The **Slab Allocator** (`src/memory/slab_allocator.mojo`) is the fundamental building block for memory management in Pion. It provides a way to pool and reuse objects of the same size.

### Key Features
- **Deterministic Latency**: Allocations and deallocations are $O(1)$ operations (bump pointer or free-list pop/push).
- **Huge Pages (2MB Superpages)**: In `cloud` and `desktop` profiles, Pion allocates slabs using 2MB superpages (via `VM_FLAGS_SUPERPAGE_SIZE_2MB` on macOS). This dramatically reduces TLB (Translation Lookaside Buffer) misses during random memory access in massive vector graphs.
- **Minimal Fragmentation**: Memory is allocated in large "slabs" and divided into fixed-size "chunks".
- **Object Re-use**: A `free_list` tracks deallocated chunks, which are immediately available for the next allocation.
- **Shared-Nothing Per-Worker**: Each worker owns its own `SlabAllocator` instances (initialized inside the worker task after `set_thread_affinity(i)` pins the worker to a CPU core). There is no explicit NUMA pinning -- core affinity provides implicit locality on NUMA systems, but Pion does not call `mbind()` or `set_mempolicy()`.
- **Adaptive Slab Doubling**: When a slab is exhausted, the allocator doubles `items_per_slab` (up to 10M), resulting in $O(\log N)$ total `mmap` calls over the lifetime of the allocator.

### Implementation

Callers must call `.deallocate(ptr)` explicitly -- there is no garbage collector.

```mojo
# Example: Creating a Slab Allocator for ListNodes (10M initial capacity)
var list_node_allocator = SlabAllocator[ListNode](10_000_000)
var node_ptr = list_node_allocator.allocate()
# ... use node_ptr ...
list_node_allocator.deallocate(node_ptr)
```

## Object Pool

The **Object Pool** (`src/memory/object_pool.mojo`) is a fixed-capacity stack of pre-allocated pointers, used when the hot path needs a pre-initialized object without any allocation.

### Key Features
- **O(1) acquire/release**: `acquire()` pops from the stack, `release()` pushes back.
- **Caller must call `.reset()`**: Pool objects retain state from their previous use. Always call `.reset()` on acquired objects before use.

### Usage: SADD / HSET New-Key
When SADD or HSET encounters a key that does not yet exist, a new `SlabHashMap` is needed for the set or hash. Rather than allocating on the hot path, `hash_map_pool` provides pre-allocated `SlabHashMap(16)` instances (16 slots each):

```
var map = hash_map_pool.acquire()
map.reset()
# ... populate and store ...
```

## Response Buffer

Each worker pre-allocates a **4 MB response buffer** (`buffer` in `ResponseWriter`) for formatting RESP responses. All `append_*_response()` methods write directly into this buffer, and `flush_response()` sends it once per batch. No allocation occurs during response formatting.

A connection that the socket cannot keep up with gets a 4 MB pending block, allocated on first use, and past that an overflow queue that grows as needed (#49), as a Redis client's reply list does: a normal client's output has no size limit, and a subscriber or MONITOR connection is disconnected once it is owed more than 32 MB. A reply larger than the response buffer is handed to that queue as the buffer fills, so it arrives whole however large it is; the worker never blocks on a slow reader.

## Data Copy Path

Pion's `GenericValue` system uses **copy-on-store** as the default path, not zero-copy:

- **`GenericValue.from_ptr(ptr, len)`** (default): Copies data into either an SSO inline buffer (for values up to 23 bytes -- stored as 3 x UInt64 words, no heap allocation) or a heap-allocated string (for values > 23 bytes). This is used for **all** key/value storage and hash map key lookups.

- **`GenericValue.from_ptr_unsafe(ptr, len)`** (restricted): Creates a raw pointer into the source buffer with no copy. This is **only** safe for temporary values that are consumed immediately in the same call frame, before any mutation to the underlying buffer. It must **never** be used for hash map key lookups (type mismatch between STRING and STRING_SSO causes guaranteed lookup misses).

- **`GenericValue.from_string(s)`** (slow path): Copies from an existing heap `String`. Used only in the slow path where a `String` already exists.

The SSO path (up to 23 bytes) achieves near-zero-copy performance: the entire value fits in three register-width words copied inline, with no heap allocation or pointer indirection.
