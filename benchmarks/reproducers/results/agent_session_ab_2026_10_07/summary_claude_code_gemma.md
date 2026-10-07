| Server | first request, cold | within the session: reused, TTFT | **after a restart**: recomputed; server start + first token | **a second session**: reused, TTFT |
|---|---:|---:|---:|---:|
| mlx-lm server, stock | 14,426 tokens, 9.89 s | 98.4%, 0.41 s | 16,129 of 16,129; 2.4 + 10.47 s = **12.9 s** | 0 of 14,486, 8.02 s |
| LM Studio | 14,422 tokens, 9.14 s | 15.5%, 10.27 s | 16,121 of 16,121; 7.1 + 11.95 s = **19.1 s** | 0 of 14,482, 8.17 s |
| oMLX | 14,397 tokens, 11.34 s | 98.4%, 0.40 s | 116 of 16,102; 2.7 + 5.46 s = **8.2 s** | 12,288 of 14,457, 1.33 s |
| Pion serve | 14,391 tokens, 9.99 s | 98.4%, 0.40 s | 16,094 of 16,094; 2.4 + 10.22 s = **12.6 s** | 12,087 of 14,451, 1.67 s |
