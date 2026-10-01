# Shared KV cache

Every request that starts with the same tokens — a system prompt, a few-shot
block, a retrieved passage — makes the model compute the same K/V tensors for
them again. Pion keeps those tensors after the first time, keyed by a
*namespace* that names the exact prefix, so the next request fetches them
instead of recomputing: from the same process, from another process, or after
a restart.

This page is the concept: what a namespace must contain, how the cold and warm
paths work, and when Pion should compute the attention itself. The commands
are in the reference — [`KV.PREFIX.*`](/reference/commands/kv-prefix.md),
[`V.*`](/reference/commands/v-store.md),
[`ATTEND.*`](/reference/commands/attend.md) — and the Python API is
[`pion-vllm-mlx`](/reference/python/pion-vllm-mlx.md).

## Quick start

<!-- include-section: doc/shared_kv_cache.md | ## Quick Start -->

## The namespace is the contract

<!-- include-section: doc/shared_kv_cache.md | ## Cache Namespace Contract -->

## How it works: the cold and warm paths

<!-- include-section: doc/shared_kv_cache.md | ## How It Works -->

## Stage 1 or Stage 2: who computes the attention

<!-- include-section: doc/shared_kv_cache.md | ### When to use Stage 1 vs Stage 2 | shift:-1 -->

## Retrieved chunks, not just system prompts

<!-- include-section: doc/shared_kv_cache.md | ### Hybrid Retrieval Cache — RAG K/V hydration | shift:-1 -->

## Where it fits

<!-- include-section: doc/shared_kv_cache.md | ## Use Cases -->

## Where to go next

- [Where Pion does not help](/getting-started/where-it-does-not-help.md) — short prefixes, low hit rates, and the cases where recomputing wins.
- [Persistence and durability](/persistence.md) — why a stored prefix survives a restart.
- [Shared KV cache — full reference](/shared_kv_cache.md) — every section of the design document, including measurements, status and the test inventory.
