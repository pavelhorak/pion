"""pion_glide_demo.py — end-to-end demo of the pion-glide Python client.

Demonstrates:
  - Standalone connection via Valkey GLIDE
  - Standard KV (GET/SET/EXPIRE/TTL)
  - HSET vector ingest + FT.CREATE / FT.OPTIMIZE / FT.SEARCH
  - Cluster mode connection (optional)
  - AI gateway (optional, requires --flare)

Requirements:
    pip install pion-glide
    ./pion-server -w 1                  # standalone
    # OR for cluster:
    ./pion-server -w 1 --cluster --cluster-host 127.0.0.1

Usage:
    python examples/pion_glide_demo.py
    python examples/pion_glide_demo.py --cluster   # cluster mode
    python examples/pion_glide_demo.py --ai        # AI gateway (needs --flare)
"""
from __future__ import annotations

import asyncio
import random
import sys
import time


async def demo_kv(client) -> None:
    print("\n── KV Demo ──────────────────────────────")
    await client.set("pion:greeting", "Hello from Valkey GLIDE!")
    val = await client.get("pion:greeting")
    print(f"  SET / GET  →  {val!r}")

    await client.set("pion:counter", "0")
    for _ in range(5):
        await client.incr("pion:counter")
    count = await client.get("pion:counter")
    print(f"  INCR ×5    →  {count!r}")

    await client.expire("pion:greeting", 60)
    ttl = await client.ttl("pion:greeting")
    print(f"  TTL        →  {ttl}s")

    await client.delete("pion:greeting", "pion:counter")
    print("  DEL        →  OK")


async def demo_hash(client) -> None:
    print("\n── Hash Demo ────────────────────────────")
    await client.hset("pion:doc:1", {
        "title": "Pion vector database",
        "author": "Horak",
        "score": "9.8",
    })
    title = await client.hget("pion:doc:1", "title")
    all_fields = await client.hgetall("pion:doc:1")
    print(f"  HGET title →  {title!r}")
    print(f"  HGETALL    →  {all_fields}")
    await client.delete("pion:doc:1")


async def demo_vector_search(client) -> None:
    print("\n── Vector Search Demo ───────────────────")
    DIM = 128
    INDEX = "glide-demo"

    # Create index
    await client.ft.create(INDEX, field="embedding", dim=DIM, metric="L2")
    print(f"  FT.CREATE  →  {INDEX} (dim={DIM})")

    # Ingest 20 random vectors
    for i in range(20):
        vec = [random.random() for _ in range(DIM)]
        await client.ft.add_vector(INDEX, f"item:{i}", vec)

    # Build HNSW graph
    t0 = time.perf_counter()
    await client.ft.optimize(INDEX)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  FT.OPTIMIZE →  done in {ms:.1f}ms")

    # k-NN search
    query = [random.random() for _ in range(DIM)]
    results = await client.ft.search(INDEX, query, k=5, ef_runtime=50)
    print(f"  FT.SEARCH  →  top-5: {[r.doc_id for r in results]}")

    # Cleanup
    await client.ft.drop(INDEX)
    print(f"  FT.DROP    →  {INDEX} deleted")


async def demo_cluster_info(client) -> None:
    print("\n── Cluster Info ─────────────────────────")
    info = await client.execute("CLUSTER", "INFO")
    if isinstance(info, bytes):
        info = info.decode()
    # Print first 3 lines
    lines = str(info).splitlines()[:3]
    for line in lines:
        print(f"  {line}")
    myid = await client.execute("CLUSTER", "MYID")
    if isinstance(myid, bytes):
        myid = myid.decode()
    print(f"  CLUSTER MYID → {myid}")


async def demo_ai(client) -> None:
    print("\n── AI Gateway Demo (requires --flare) ───")
    try:
        # Semantic cache
        await client.ai.semantic_cache_set("capital of France?", "+Paris\r\n")
        hit = await client.ai.semantic_cache_get(
            "What is the capital of France?", threshold=0.85
        )
        print(f"  SEMANTIC_CACHE GET  →  {hit!r}")

        # Text RAG
        await client.ft.add_text("glide-kb", "doc:1", "Pion achieves 10283 QPS on Linux io_uring")
        await client.ft.add_text("glide-kb", "doc:2", "Valkey GLIDE provides cluster-aware routing")
        results = await client.ft.search_text("glide-kb", "database performance", k=2)
        print(f"  FT.SEARCHTEXT  →  {[r.doc_id for r in results]}")
        await client.ft.drop("glide-kb")
    except Exception as e:
        print(f"  (skipped — server may not be in --flare mode: {e})")


async def main() -> None:
    args = sys.argv[1:]
    cluster_mode = "--cluster" in args
    ai_mode = "--ai" in args

    try:
        from pion_glide import PionClient
    except ImportError:
        print("pion-glide not installed. Run: pip install pion-glide")
        print("Or from source: pip install -e pion_glide/")
        sys.exit(1)

    print("Pion GLIDE Demo")
    print("=" * 42)

    if cluster_mode:
        print("Mode: CLUSTER (--cluster)")
        print("Connecting to 127.0.0.1:1974 (cluster seed)...")
        client = await PionClient.connect_cluster([("127.0.0.1", 1974)])
    else:
        print("Mode: STANDALONE")
        print("Connecting to 127.0.0.1:1974...")
        client = await PionClient.connect("127.0.0.1", 1974)

    pong = await client.ping()
    print(f"PING → {pong}")

    await demo_kv(client)
    await demo_hash(client)
    await demo_vector_search(client)
    await demo_cluster_info(client)

    if ai_mode:
        await demo_ai(client)

    await client.close()
    print("\n✓ Demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
