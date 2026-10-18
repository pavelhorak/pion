#!/usr/bin/env python3
# svs_steady_patch.py <redis.py.competitor> ...: the 2026-10-18 box fix for the RediSearch wait.
# SVS-VAMANA keeps vectors below its flush threshold in the tiered index's flat frontend buffer,
# which every search also scans. On the box 562 of 50,000 stayed there with no job pending and
# no worker active, so "frontend buffer == 0" never came true. Done now also when the server is
# idle (indexing 0, percent 1, no pending jobs, no active workers) and the buffer has not changed
# for VDBB_REDIS_WAIT_STEADY seconds (default 10). A buffer of 0 is done at once, as before.
import sys

OLD = '''            done = (_s(info.get("indexing", "1")) == "0" and _s(info.get("percent_indexed", "0")) in ("1", "1.0")
                    and all(v == "0" for v in vals.values()))
            return done, f"indexing={_s(info.get('indexing'))} percent={_s(info.get('percent_indexed'))} {vals}"
'''
NEW = '''            idle = (_s(info.get("indexing", "1")) == "0" and _s(info.get("percent_indexed", "0")) in ("1", "1.0")
                    and all(vals[k] == "0" for k in keys[1:]))
            fb = vals[keys[0]]
            desc = f"indexing={_s(info.get('indexing'))} percent={_s(info.get('percent_indexed'))} {vals}"
            if idle and fb != "0":
                # SVS-VAMANA keeps vectors below its flush threshold in the flat frontend buffer,
                # which searches scan; with no job pending and no worker active nothing moves
                # them (box 2026-10-18: 562 of 50,000). Done once that has held STEADY_S.
                now = time.perf_counter()
                if getattr(self, "_steady_fb", None) != fb:
                    self._steady_fb, self._steady_t = fb, now
                held = now - self._steady_t
                return held >= STEADY_S, desc + f" idle, frontend buffer steady {held:.1f}s"
            self._steady_fb = None
            return idle and fb == "0", desc
'''
OLD_CFG = 'WAIT_TIMEOUT = float(os.environ.get("VDBB_REDIS_WAIT_TIMEOUT", "7200"))\n'
NEW_CFG = OLD_CFG + 'STEADY_S = float(os.environ.get("VDBB_REDIS_WAIT_STEADY", "10"))\n'

for path in sys.argv[1:]:
    s = open(path).read()
    if "STEADY_S" in s:
        print(f"already patched: {path}")
        continue
    assert s.count(OLD) == 1 and s.count(OLD_CFG) == 1, f"anchor not found exactly once in {path}"
    s = s.replace(OLD, NEW).replace(OLD_CFG, NEW_CFG)
    compile(s, path, "exec")
    open(path, "w").write(s)
    print(f"patched: {path}")
