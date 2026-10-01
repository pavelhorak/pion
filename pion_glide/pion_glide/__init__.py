"""pion-glide: Valkey GLIDE–based Python client for Pion.

Provides typed async wrappers for FT.* (vector search) and AI.* (gateway)
commands on top of Valkey GLIDE's standard Redis interface.

    pip install pion-glide

Quickstart:
    from pion_glide import PionClient

    client = await PionClient.connect("127.0.0.1", 1974)
    await client.set("key", "hello")
    await client.ft.create("products", dim=1536)
    await client.ft.add_vector("products", "doc:1", my_float_list)
    await client.ft.optimize("products")
    results = await client.ft.search("products", query_vec, k=10)
    await client.close()
"""
from .client import PionClient
from .ft import FTIndex, FTSearchResult
from .ai import AIGateway

__all__ = ["PionClient", "FTIndex", "FTSearchResult", "AIGateway"]
__version__ = "0.1.0"
