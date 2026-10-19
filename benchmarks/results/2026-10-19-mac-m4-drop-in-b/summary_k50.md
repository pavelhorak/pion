| arm | run | QPS c1 / c5 / c10 | max QPS | recall | insert s | optimize s |
|---|---|---|---:|---:|---:|---:|
| a_stock_p1 | full | 1,291 / 7,024 / 9,064 | 9,064 | 0.9595 | 3.7 | 0.0 |
| a_stock_p1 | s1 | 1,642 / 6,521 / 8,297 | 8,297 | 0.9595 | 0.0 | 0.0 |
| a_stock_p1 | s2 | 1,653 / 6,469 / 8,231 | 8,231 | 0.9595 | 0.0 | 0.0 |
|  | a_stock_p1 log | [vector] auto-build: 50000 vectors indexed in 3637 ms; FT.SEARCH uses the graph now | | | | |
| a_stock_p2 | full | 1,284 / 6,639 / 8,471 | 8,471 | 0.9600 | 3.3 | 0.0 |
| a_stock_p2 | s1 | 1,553 / 3,947 / 8,308 | 8,308 | 0.9600 | 0.0 | 0.0 |
| a_stock_p2 | s2 | 1,569 / 7,049 / 8,682 | 8,682 | 0.9600 | 0.0 | 0.0 |
|  | a_stock_p2 log | [vector] auto-build: 50000 vectors indexed in 3575 ms; FT.SEARCH uses the graph now | | | | |
| b_patch_p1 | full | 1,608 / 6,837 / 8,936 | 8,936 | 0.9601 | 3.4 | 3.8 |
| b_patch_p1 | s1 | 1,606 / 5,600 / 8,886 | 8,886 | 0.9601 | 0.0 | 0.0 |
| b_patch_p1 | s2 | 1,599 / 5,605 / 8,830 | 8,830 | 0.9601 | 0.0 | 0.0 |
|  | b_patch_p1 log | [vector] auto-build: 50000 vectors indexed in 4019 ms; FT.SEARCH uses the graph now | | | | |
| b_patch_p2 | full | 1,532 / 5,434 / 8,647 | 8,647 | 0.9604 | 3.4 | 2.8 |
| b_patch_p2 | s1 | 1,536 / 6,623 / 8,261 | 8,261 | 0.9604 | 0.0 | 0.0 |
| b_patch_p2 | s2 | 1,535 / 6,346 / 8,770 | 8,770 | 0.9604 | 0.0 | 0.0 |
|  | b_patch_p2 log | [vector] auto-build: 50000 vectors indexed in 3177 ms; FT.SEARCH uses the graph now | | | | |
| b_stock_p1 | full | 1,404 / 5,734 / 8,872 | 8,872 | 0.9598 | 3.3 | 0.0 |
| b_stock_p1 | s1 | 1,730 / 6,789 / 8,557 | 8,557 | 0.9598 | 0.0 | 0.0 |
| b_stock_p1 | s2 | 1,740 / 5,807 / 8,492 | 8,492 | 0.9598 | 0.0 | 0.0 |
|  | b_stock_p1 log | [vector] auto-build: 50000 vectors indexed in 3536 ms; FT.SEARCH uses the graph now | | | | |
| b_stock_p2 | full | 1,222 / 6,593 / 8,707 | 8,707 | 0.9602 | 3.2 | 0.0 |
| b_stock_p2 | s1 | 1,567 / 6,813 / 9,126 | 9,126 | 0.9602 | 0.0 | 0.0 |
| b_stock_p2 | s2 | 1,567 / 5,464 / 8,726 | 8,726 | 0.9602 | 0.0 | 0.0 |
|  | b_stock_p2 log | [vector] auto-build: 50000 vectors indexed in 3534 ms; FT.SEARCH uses the graph now | | | | |

a_stock: search-only max QPS 8,297, 8,231, 8,308, 8,682; median 8,302
b: search-only max QPS 8,886, 8,830, 8,261, 8,770; median 8,800
b_stock: search-only max QPS 8,557, 8,492, 9,126, 8,726; median 8,642
