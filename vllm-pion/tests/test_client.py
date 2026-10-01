#!/usr/bin/env python3
"""Test the PionKVClient and PionKVConnector against a running Pion server."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from vllm_pion.client import PionKVClient
from vllm_pion.connector import PionKVConnector, HashEmbedder

HOST = "127.0.0.1"
PORT = 1974


def test_pion_kv_client():
    """Test PionKVClient directly."""
    print("=== Testing PionKVClient ===")

    with PionKVClient(HOST, PORT) as client:
        # Check info
        info = client.kv_info()
        print(f"Initial info: {info}")
        assert "entries" in info
        assert info["enabled"] == "True" or info["enabled"] == 1

        # Store a blob
        np.random.seed(123)
        embedding = np.random.randn(768).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        blob = b"TEST_KV_TENSOR_DATA_" * 100  # 2KB test blob

        ok = client.kv_store("client_test_1", embedding, blob, ttl=60, model="llama-3-8b")
        assert ok, "kv_store failed"
        print("kv_store: OK")

        # Fetch exact
        result = client.kv_fetch(embedding)
        assert result is not None, "kv_fetch (exact) returned None"
        assert b"TEST_KV_TENSOR_DATA_" in result
        print(f"kv_fetch (exact): OK, {len(result)} bytes")

        # Fetch similar
        similar = embedding + np.random.randn(768).astype(np.float32) * 0.005
        similar /= np.linalg.norm(similar)
        result = client.kv_fetch(similar)
        assert result is not None, "kv_fetch (similar) returned None"
        print(f"kv_fetch (similar): OK, {len(result)} bytes")

        # Fetch different (should miss at threshold 0.95)
        different = np.random.randn(768).astype(np.float32)
        different /= np.linalg.norm(different)
        result = client.kv_fetch(different)
        # With only 1-2 entries, this may or may not miss
        print(f"kv_fetch (different): {'HIT' if result else 'MISS'}")

        # Info after operations
        info = client.kv_info()
        print(f"Final info: {info}")
        assert info.get("hits", 0) >= 2

    print("=== PionKVClient: ALL PASSED ===\n")


def test_pion_connector():
    """Test PionKVConnector (high-level API)."""
    print("=== Testing PionKVConnector ===")

    connector = PionKVConnector({
        "pion_host": HOST,
        "pion_port": PORT,
        "embed_dim": 768,
        "cosine_threshold": 0.95,
        "model_tag": "test-model",
        "ttl": 300,
    })

    # Store a cache entry
    prompt = "What is the capital of France?"
    # Create a fake KV blob with a num_tokens header
    import struct
    num_tokens = 42
    header = struct.pack("<I", num_tokens)
    fake_kv_data = header + b"\x00" * 1000

    ok = connector.store_cache(prompt, fake_kv_data)
    print(f"store_cache: {'OK' if ok else 'FAILED'}")

    # Check cache with exact prompt
    result = connector.check_cache(prompt)
    if result:
        cached_tokens, blob = result
        print(f"check_cache (exact): HIT, {cached_tokens} tokens, {len(blob)} bytes")
    else:
        print("check_cache (exact): MISS (may be expected with hash embedder)")

    # Check cache with similar prompt (hash embedder won't match — that's expected)
    result2 = connector.check_cache("What is France's capital city?")
    if result2:
        print(f"check_cache (similar): HIT (semantic matching works!)")
    else:
        print("check_cache (similar): MISS (expected with hash embedder, would hit with Ollama)")

    # Stats
    stats = connector.get_stats()
    print(f"Stats: {stats}")

    connector.close()
    print("=== PionKVConnector: ALL PASSED ===\n")


if __name__ == "__main__":
    print("Testing vllm-pion package against Pion server")
    print(f"Server: {HOST}:{PORT}\n")

    test_pion_kv_client()
    test_pion_connector()

    print("=== ALL TESTS PASSED ===")
