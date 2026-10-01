"""Pion attention hook for exo distributed inference.

Offloads V storage to Pion (turbo4 compressed) and optionally runs full
attention on Pion's native Metal SDPA via PionPromptCache(stage2=True).
Enables 128K context on 16GB Macs and Q-only-on-the-wire decode.

Usage in exo:
    from pion_exo import PionAttentionHook
    hook = PionAttentionHook(pion_host="192.168.1.100", mode="gpu_attention")

For gpu_attention mode, run Pion with:
    ./pion-server --kvcache --metal-attention -w 1
"""

from pion_exo.attention_hook import PionAttentionHook
from pion_exo.session_manager import PionSessionManager

__all__ = ["PionAttentionHook", "PionSessionManager"]
