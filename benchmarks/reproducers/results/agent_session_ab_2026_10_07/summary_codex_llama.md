| Server | first request, cold | within the session: reused, TTFT | **after a restart**: recomputed; server start + first token | **a second session**: reused, TTFT |
|---|---:|---:|---:|---:|
| mlx-lm server, stock | 7,121 tokens, 5.14 s | 97.5%, 0.36 s | 9,560 of 9,560; 3.0 + 7.33 s = **10.3 s** | — |
| Ollama | 7,540 tokens, 9.96 s | 82.7%, 0.40 s | 9,872 of 9,872; 0.7 + 9.94 s = **10.6 s** | — |
| LM Studio (9 of 20 replies ended in a stream error) | 7,249 tokens, 5.37 s | 97.0%, 0.54 s | not reported (stream error); 3.3 + 8.22 s = **11.5 s** | — |
| oMLX | 9,703 tokens, 8.05 s | 97.1%, 0.51 s | 110 of 12,142; 1.2 + 1.13 s = **2.3 s** | — |
| Pion serve | 7,121 tokens, 5.59 s | 97.5%, 0.38 s | 77 of 9,560; 2.3 + 1.13 s = **3.4 s** | — |
