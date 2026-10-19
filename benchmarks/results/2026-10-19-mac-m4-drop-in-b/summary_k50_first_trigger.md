| arm | run | QPS c1 / c5 / c10 | max QPS | recall | insert s | optimize s |
|---|---|---|---:|---:|---:|---:|
| a_stock_p1 | full | 1,219 / 6,620 / 8,524 | 8,524 | 0.9599 | 3.6 | 0.0 |
| a_stock_p1 | s1 | 1,884 / 6,709 / 8,813 | 8,813 | 0.9599 | 0.0 | 0.0 |
| a_stock_p1 | s2 | 1,587 / 5,664 / 8,901 | 8,901 | 0.9599 | 0.0 | 0.0 |
|  | a_stock_p1 log | [vector] auto-build: 50000 vectors indexed in 3608 ms; FT.SEARCH uses the graph now | | | | |
| a_stock_p2 | full | 1,258 / 5,519 / 8,978 | 8,978 | 0.9597 | 3.3 | 0.0 |
| a_stock_p2 | s1 | 1,645 / 5,520 / 7,918 | 7,918 | 0.9597 | 0.0 | 0.0 |
| a_stock_p2 | s2 | 1,642 / 6,466 / 9,063 | 9,063 | 0.9597 | 0.0 | 0.0 |
|  | a_stock_p2 log | [vector] auto-build: 50000 vectors indexed in 3659 ms; FT.SEARCH uses the graph now | | | | |
| b_patch_p1 | full | 1,471 / 5,386 / 8,810 | 8,810 | 0.9603 | 5.1 | 4.4 |
| b_patch_p1 | s1 | 1,477 / 5,406 / 8,880 | 8,880 | 0.9603 | 0.0 | 0.0 |
| b_patch_p1 | s2 | 1,483 / 6,898 / 8,976 | 8,976 | 0.9603 | 0.0 | 0.0 |
|  | b_patch_p1 log | [vector] auto-build: 16597 vectors indexed in 704 ms; FT.SEARCH uses the graph now | | | | |
|  | b_patch_p1 log | [vector] auto-build: 25340 vectors indexed in 1256 ms; FT.SEARCH uses the graph now | | | | |
|  | b_patch_p1 log | [vector] auto-build: 39110 vectors indexed in 3348 ms; FT.SEARCH uses the graph now | | | | |
| b_patch_p2 | full | 1,607 / 4,090 / 8,882 | 8,882 | 0.9595 | 5.2 | 4.0 |
| b_patch_p2 | s1 | 1,715 / 6,869 / 8,640 | 8,640 | 0.9595 | 0.0 | 0.0 |
| b_patch_p2 | s2 | 1,603 / 6,707 / 8,389 | 8,389 | 0.9595 | 0.0 | 0.0 |
|  | b_patch_p2 log | [vector] auto-build: 16469 vectors indexed in 655 ms; FT.SEARCH uses the graph now | | | | |
|  | b_patch_p2 log | [vector] auto-build: 23478 vectors indexed in 1115 ms; FT.SEARCH uses the graph now | | | | |
|  | b_patch_p2 log | [vector] auto-build: 35202 vectors indexed in 2112 ms; FT.SEARCH uses the graph now | | | | |
|  | b_patch_p2 log | [vector] auto-build: 50000 vectors indexed in 3986 ms; FT.SEARCH uses the graph now | | | | |
| b_stock_p1 | full | 1,259 / 3,958 / 8,839 | 8,839 | 0.9600 | 5.1 | 0.0 |
| b_stock_p1 | s1 | 1,860 / 6,938 / 7,626 | 7,626 | 0.9600 | 0.0 | 0.0 |
| b_stock_p1 | s2 | 1,581 / 6,530 / 7,953 | 7,953 | 0.9600 | 0.0 | 0.0 |
|  | b_stock_p1 log | [vector] auto-build: 16400 vectors indexed in 657 ms; FT.SEARCH uses the graph now | | | | |
|  | b_stock_p1 log | [vector] auto-build: 23354 vectors indexed in 1139 ms; FT.SEARCH uses the graph now | | | | |
|  | b_stock_p1 log | [vector] auto-build: 35536 vectors indexed in 2078 ms; FT.SEARCH uses the graph now | | | | |
|  | b_stock_p1 log | [vector] auto-build: 50000 vectors indexed in 3372 ms; FT.SEARCH uses the graph now | | | | |
| b_stock_p2 | full | 1,302 / 6,850 / 8,698 | 8,698 | 0.9603 | 5.1 | 0.0 |
| b_stock_p2 | s1 | 1,752 / 5,819 / 8,227 | 8,227 | 0.9603 | 0.0 | 0.0 |
| b_stock_p2 | s2 | 1,745 / 4,296 / 8,451 | 8,451 | 0.9603 | 0.0 | 0.0 |
|  | b_stock_p2 log | [vector] auto-build: 16498 vectors indexed in 671 ms; FT.SEARCH uses the graph now | | | | |
|  | b_stock_p2 log | [vector] auto-build: 23703 vectors indexed in 1153 ms; FT.SEARCH uses the graph now | | | | |
|  | b_stock_p2 log | [vector] auto-build: 36112 vectors indexed in 2199 ms; FT.SEARCH uses the graph now | | | | |
|  | b_stock_p2 log | [vector] auto-build: 50000 vectors indexed in 3456 ms; FT.SEARCH uses the graph now | | | | |

a_stock: search-only max QPS 8,813, 8,901, 7,918, 9,063; median 8,857
b: search-only max QPS 8,880, 8,976, 8,640, 8,389; median 8,760
b_stock: search-only max QPS 7,626, 7,953, 8,227, 8,451; median 8,090
