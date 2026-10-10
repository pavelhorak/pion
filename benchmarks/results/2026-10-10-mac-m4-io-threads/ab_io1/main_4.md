### Benchmark Run: 2026-10-10 12:13:57
**Parameters:** Clients: 50, Requests: 100000, Pipeline: 10, Pion workers: 1

| Command | Valkey (RPS) | Redis (RPS) | Dragonfly (RPS) | Pion (RPS) | Winner | Pion vs Best (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| GET | 0 | 0 | 0 | **2,380,952** | **Pion** | 0.0% |
| HSET | 0 | 0 | 0 | **2,380,952** | **Pion** | 0.0% |
| INCR | 0 | 0 | 0 | **2,325,581** | **Pion** | 0.0% |
| LPOP | 0 | 0 | 0 | **2,272,727** | **Pion** | 0.0% |
| LPUSH | 0 | 0 | 0 | **2,325,581** | **Pion** | 0.0% |
| LPUSH (needed to benchmark LRANGE) | 0 | 0 | 0 | **2,325,581** | **Pion** | 0.0% |
| LRANGE_100 (first 100 elements) | 0 | 0 | 0 | **261,780** | **Pion** | 0.0% |
| LRANGE_300 (first 300 elements) | 0 | 0 | 0 | **104,384** | **Pion** | 0.0% |
| LRANGE_500 (first 500 elements) | 0 | 0 | 0 | **62,617** | **Pion** | 0.0% |
| LRANGE_600 (first 600 elements) | 0 | 0 | 0 | **52,798** | **Pion** | 0.0% |
| MSET (10 keys) | 0 | 0 | 0 | **1,851,851** | **Pion** | 0.0% |
| PING_INLINE | 0 | 0 | 0 | **1,492,537** | **Pion** | 0.0% |
| PING_MBULK | 0 | 0 | 0 | **2,439,024** | **Pion** | 0.0% |
| RPOP | 0 | 0 | 0 | **2,380,952** | **Pion** | 0.0% |
| RPUSH | 0 | 0 | 0 | **2,325,581** | **Pion** | 0.0% |
| SADD | 0 | 0 | 0 | **1,724,138** | **Pion** | 0.0% |
| SET | 0 | 0 | 0 | **2,272,727** | **Pion** | 0.0% |
| SPOP | 0 | 0 | 0 | **2,500,000** | **Pion** | 0.0% |
| XADD | 0 | 0 | 0 | **2,173,913** | **Pion** | 0.0% |
| ZADD | 0 | 0 | 0 | **2,325,581** | **Pion** | 0.0% |
| ZPOPMIN | 0 | 0 | 0 | **2,380,952** | **Pion** | 0.0% |


