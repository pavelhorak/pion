"""pion-llamaindex: LlamaIndex VectorStore backed by Pion.

Quick start:
    from pion_llamaindex import PionVectorStore
    from llama_index.core import VectorStoreIndex, Document

    store = PionVectorStore(host="127.0.0.1", port=1974)
    index = VectorStoreIndex.from_documents(documents, vector_store=store)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is Pion?")
"""
from pion_llamaindex.vector_store import PionVectorStore

__all__ = ["PionVectorStore"]
