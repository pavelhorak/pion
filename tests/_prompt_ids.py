"""Token ids for a prompt assembled from pieces, with exactly one <bos>.

`tok.encode(text)` adds the tokenizer's special tokens. Llama 3 prepends
<|begin_of_text|> by default, and tests/_gemma4_text_filter_load.py turns
Gemma 4's <bos> on (gh #93). A prompt joined from separately encoded pieces
therefore carries a <bos> at every seam, and an encoded filler doubled to a
target length copies its <bos> along. The 64K NIAH prompt held 397 of them,
and neither vanilla mlx-lm nor Pion found the needle.

Encode every piece with `piece()`, put `bos()` once at position 0, and pass
the finished prompt through `one_bos()`.
"""
from __future__ import annotations

from typing import List, Sequence


def piece(tok, text: str) -> List[int]:
    """Ids for one piece of a larger prompt: no special tokens."""
    return list(tok.encode(text, add_special_tokens=False))


def bos(tok) -> List[int]:
    """The prompt's one <bos>, for position 0; empty if the tokenizer has none."""
    b = getattr(tok, "bos_token_id", None)
    return [b] if b is not None else []


def bos_count(tok, ids: Sequence[int]) -> int:
    b = getattr(tok, "bos_token_id", None)
    return 0 if b is None else sum(1 for t in ids if t == b)


def one_bos(tok, ids: Sequence[int]) -> List[int]:
    """Return `ids` if they hold exactly the leading <bos> (or none, for a
    tokenizer without one); raise otherwise, so a harness refuses to measure
    a malformed prompt instead of reporting a plausible number from it."""
    ids = list(ids)
    want = 1 if bos(tok) else 0
    got = bos_count(tok, ids)
    if got != want or (want and ids[0] != bos(tok)[0]):
        raise ValueError(f"prompt has {got} <bos> tokens (want {want}, at position 0)")
    return ids
