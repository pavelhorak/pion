| arm | run | QPS c1 / c5 / c10 | max QPS | recall | insert s | optimize s |
|---|---|---|---:|---:|---:|---:|
| new_p1 | full | 1,229 / 7,018 / 8,657 | 8,657 | 0.9592 | 3.4 | 0.0 |
| new_p1 | s1 | 1,646 / 5,548 / 8,783 | 8,783 | 0.9592 | 0.0 | 0.0 |
| new_p1 | s2 | 1,649 / 6,720 / 8,779 | 8,779 | 0.9592 | 0.0 | 0.0 |
|  | new_p1 log | [vector] auto-build: 50000 vectors indexed in 3622 ms; FT.SEARCH uses the graph now | | | | |
| new_p2 | full | 1,366 / 5,640 / 9,020 | 9,020 | 0.9600 | 3.4 | 0.0 |
| new_p2 | s1 | 1,645 / 6,995 / 7,780 | 7,780 | 0.9600 | 0.0 | 0.0 |
| new_p2 | s2 | 1,645 / 6,860 / 8,365 | 8,365 | 0.9600 | 0.0 | 0.0 |
|  | new_p2 log | [vector] auto-build: 50000 vectors indexed in 3689 ms; FT.SEARCH uses the graph now | | | | |
| new_p3 | full | 1,139 / 6,866 / 8,902 | 8,902 | 0.9604 | 3.3 | 0.0 |
| new_p3 | s1 | 1,555 / 6,475 / 8,710 | 8,710 | 0.9604 | 0.0 | 0.0 |
| new_p3 | s2 | 1,561 / 5,505 / 8,611 | 8,611 | 0.9604 | 0.0 | 0.0 |
|  | new_p3 log | [vector] auto-build: 50000 vectors indexed in 3766 ms; FT.SEARCH uses the graph now | | | | |
| newp_p1 | full | 1,542 / 6,728 / 8,812 | 8,812 | 0.9597 | 3.4 | 2.7 |
| newp_p1 | s1 | 1,540 / 5,460 / 8,821 | 8,821 | 0.9597 | 0.0 | 0.0 |
| newp_p1 | s2 | 1,540 / 6,542 / 8,745 | 8,745 | 0.9597 | 0.0 | 0.0 |
|  | newp_p1 log | [vector] auto-build: 50000 vectors indexed in 3075 ms; FT.SEARCH uses the graph now | | | | |
| newp_p2 | full | 1,599 / 6,641 / 9,079 | 9,079 | 0.9596 | 3.4 | 2.8 |
| newp_p2 | s1 | 1,601 / 6,634 / 8,796 | 8,796 | 0.9596 | 0.0 | 0.0 |
| newp_p2 | s2 | 1,600 / 6,406 / 8,889 | 8,889 | 0.9596 | 0.0 | 0.0 |
|  | newp_p2 log | [vector] auto-build: 50000 vectors indexed in 3203 ms; FT.SEARCH uses the graph now | | | | |
| newp_p3 | full | 1,685 / 5,602 / 8,469 | 8,469 | 0.9597 | 3.6 | 2.8 |
| newp_p3 | s1 | 1,603 / 6,953 / 8,610 | 8,610 | 0.9597 | 0.0 | 0.0 |
| newp_p3 | s2 | 1,610 / 6,530 / 8,276 | 8,276 | 0.9597 | 0.0 | 0.0 |
|  | newp_p3 log | [vector] auto-build: 50000 vectors indexed in 3106 ms; FT.SEARCH uses the graph now | | | | |
| old_p1 | full | 1,649 / 6,594 / 8,402 | 8,402 | 0.9599 | 3.8 | 4.0 |
| old_p1 | s1 | 1,742 / 5,610 / 8,771 | 8,771 | 0.9599 | 0.0 | 0.0 |
| old_p1 | s2 | 1,624 / 6,892 / 8,689 | 8,689 | 0.9599 | 0.0 | 0.0 |
| old_p2 | full | 1,585 / 6,291 / 8,695 | 8,695 | 0.9602 | 3.5 | 4.1 |
| old_p2 | s1 | 1,601 / 6,781 / 8,514 | 8,514 | 0.9602 | 0.0 | 0.0 |
| old_p2 | s2 | 1,599 / 6,634 / 8,662 | 8,662 | 0.9602 | 0.0 | 0.0 |
| old_p3 | full | 1,690 / 6,413 / 8,962 | 8,962 | 0.9598 | 3.4 | 4.0 |
| old_p3 | s1 | 1,529 / 6,781 / 8,169 | 8,169 | 0.9598 | 0.0 | 0.0 |
| old_p3 | s2 | 1,529 / 6,499 / 9,035 | 9,035 | 0.9598 | 0.0 | 0.0 |

new: search-only max QPS 8,783, 8,779, 7,780, 8,365, 8,710, 8,611; median 8,660
newp: search-only max QPS 8,821, 8,745, 8,796, 8,889, 8,610, 8,276; median 8,771
old: search-only max QPS 8,771, 8,689, 8,514, 8,662, 8,169, 9,035; median 8,675
new/old = 0.998
newp/old = 1.011
