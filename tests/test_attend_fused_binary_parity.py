"""gh #50: parity test for the ATTEND.PREFIX.QUERY_FUSED binary fast lane.

Sanity check that the new 0xCA5E-framed CMD_ATTEND_PREFIX_QUERY_FUSED path
returns bit-equal output to the existing RESP path. Server-side they share
the same Metal kernel (`metal_engine.query_batched_fused`), so the body of
the reply must be byte-for-byte identical.

Server:
  ./pion-server-dev --kvcache --metal-attention -w 1 --no-auto-detect --no-auto-embed
"""
from __future__ import annotations

import os
import sys

import numpy as np


PION_HOST = os.environ.get("PION_HOST", "127.0.0.1")
PION_PORT = int(os.environ.get("PION_PORT", "1974"))


def main() -> int:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "pion-vllm-mlx"))
    from pion_vllm_mlx.prompt_cache import PionPromptCache

    rng = np.random.default_rng(0xBEEFC0DE)
    H_kv, H_q, D = 4, 16, 64
    N_pref = 96
    S_suf = 12
    M = 4
    layer_id = 0
    namespace = "gh50_binary_parity_ns"

    K_pref = rng.standard_normal((H_kv, N_pref, D)).astype(np.float32) * 0.07
    V_pref = rng.standard_normal((H_kv, N_pref, D)).astype(np.float32) * 0.07
    K_suf = rng.standard_normal((H_kv, S_suf, D)).astype(np.float32) * 0.07
    V_suf = rng.standard_normal((H_kv, S_suf, D)).astype(np.float32) * 0.07
    Q = rng.standard_normal((H_q, M, D)).astype(np.float32) * 0.07
    head_map = np.repeat(np.arange(H_kv, dtype=np.uint8), H_q // H_kv)

    # 1) RESP-only baseline. Stores prefix and runs the fused query through
    # the legacy RESP wire path.
    os.environ["PION_PROMPT_CACHE_NO_BINARY"] = "1"
    pc_resp = PionPromptCache(model=None, host=PION_HOST, port=PION_PORT, stage2=True)
    pc_resp.attend_store_layer(namespace, layer_id, K_pref, V_pref)
    out_resp = pc_resp.attend_query_fused(namespace, layer_id, Q, K_suf, V_suf, head_map)
    if pc_resp._binary is not None:
        print("FAIL: PION_PROMPT_CACHE_NO_BINARY=1 still opened a binary socket")
        return 1
    pc_resp.resp.sock.close()

    # 2) Binary fast lane. The prefix is already resident in the Metal
    # session cache from step 1 (same sid), so we only need to re-issue the
    # query via PionPromptCache configured to use the binary path.
    del os.environ["PION_PROMPT_CACHE_NO_BINARY"]
    pc_bin = PionPromptCache(model=None, host=PION_HOST, port=PION_PORT, stage2=True)
    out_bin = pc_bin.attend_query_fused(namespace, layer_id, Q, K_suf, V_suf, head_map)
    if pc_bin._binary is None:
        print("SKIP: binary listener unreachable on port",
              PION_PORT + 1, "— start with --kvcache to exercise gh #50.")
        return 0
    pc_bin._binary.close()
    pc_bin.resp.sock.close()

    max_abs = float(np.abs(out_resp - out_bin).max())
    print(f"max|RESP - BINARY| = {max_abs:.6e}")
    if max_abs != 0.0:
        # Same kernel, same inputs — any divergence means the binary wire
        # layout is feeding different values to the kernel.
        print("FAIL: binary fast lane diverges from RESP path")
        return 1
    print("PASS — gh #50 binary fast lane bit-equal to RESP path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
