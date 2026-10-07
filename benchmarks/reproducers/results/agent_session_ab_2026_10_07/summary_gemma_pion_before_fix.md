| Server | first request, cold | within the session: reused, TTFT | **after a restart**: recomputed; server start + first token | **a second session**: reused, TTFT |
|---|---:|---:|---:|---:|
| Pion serve | 14,391 tokens, 10.92 s | 0.0%, 9.28 s | 16,094 of 16,094; 3.4 + 10.42 s = **13.8 s** | 0 of 14,451, 8.50 s |
