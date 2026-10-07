| Server | first request, cold | within the session: reused, TTFT | **after a restart**: recomputed; server start + first token | **a second session**: reused, TTFT |
|---|---:|---:|---:|---:|
| mlx-lm server, stock | 16,020 tokens, 15.24 s | 98.9%, 0.49 s | 17,478 of 17,478; 3.0 + 16.21 s = **19.2 s** | 45 of 16,065, 15.36 s |
| Ollama | 13,971 tokens, 18.42 s | 35.3%, 14.66 s | 15,376 of 15,376; 0.6 + 18.34 s = **18.9 s** | 30 of 14,016, 16.98 s |
| LM Studio | 15,921 tokens, 15.01 s | 98.6%, 1.33 s | 18,263 of 18,263; 3.4 + 18.25 s = **21.6 s** | 46 of 15,966, 15.80 s |
| oMLX | 16,009 tokens, 15.04 s | 97.9%, 0.85 s | 469 of 18,389; 1.2 + 1.63 s = **2.8 s** | 13,824 of 16,054, 2.67 s |
| Pion serve | 15,994 tokens, 14.07 s | 98.9%, 0.42 s | 89 of 17,452; 3.5 + 0.66 s = **4.2 s** | 14,056 of 16,039, 2.33 s |
