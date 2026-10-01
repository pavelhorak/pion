"""gh #262 — the value receipt.

The server used to measure nothing about the value it delivered: this file
held a `MetricsRegistry` whose `record_op()` / `record_traffic()` /
`get_prometheus_format()` were called from nowhere (instantiated once in
`state.mojo`, never touched). Dead code that looks like a feature is worse
than absence, so it is gone; what replaced it is the one number the wedge
user retains on — "how much prefill did Pion skip for me?" — plus the plane
counters needed to trust it.

`ValueLedger` is per worker (one per `SlowPathHandler`), so it needs no
atomics: every plane it counts is a slow-path command, and a worker is
single-task (V16). Formatting happens on demand in `INFO` / `PION.STATS`,
never on the recording side.

Two kinds of number, kept apart on purpose:
  * `*_measured` — the client reported its own cold prefill time on
    `KV.PREFIX.REGISTER ... PREFILL_MS <ms>`; a later hit on that prefix is
    credited with exactly that time. This is a receipt, not an estimate.
  * the rest of `kvprefix_prefill_us_saved` — hits on prefixes with no
    reported time are credited `prefix_tokens × LEDGER_EST_PREFILL_US_PER_TOKEN`,
    a per-token cost measured on Llama-3.2-1B-Instruct-4bit on an M-series
    Mac (2,022-token prefix: 1,218 ms cold, 95 ms warm → ~555 µs/token
    avoided). Larger models cost more per token, so the estimate is
    conservative for them; it is labelled an estimate wherever it is shown.
"""
from std.ffi import external_call
from std.memory import alloc

# Fallback per-token prefill cost (microseconds) when the client reported none.
comptime LEDGER_EST_PREFILL_US_PER_TOKEN: UInt64 = 555


@always_inline
def _ledger_now_ns() -> Int64:
    var ts = alloc[Int64](2)
    _ = external_call["clock_gettime", Int32](Int32(0), ts)
    var r = ts[unsafe_offset=0] * Int64(1_000_000_000) + ts[unsafe_offset=1]
    ts.unsafe_free()
    return r


def _us_to_seconds_text(us: UInt64) -> String:
    """`S.mmm` — microseconds as seconds with three decimals, integer math only."""
    var s = us // UInt64(1_000_000)
    var ms = (us % UInt64(1_000_000)) // UInt64(1000)
    var t = String(s) + "."
    if ms < UInt64(10):
        t += "00"
    elif ms < UInt64(100):
        t += "0"
    t += String(ms)
    return t


struct ValueLedger(Movable, Copyable):
    var started_at_ns: Int64
    # KV prefix plane: KV.PREFIX.LOOKUP decides hit/miss; V.FETCH delivers bytes.
    var kvprefix_hits: UInt64
    var kvprefix_misses: UInt64
    var kvprefix_hits_measured: UInt64       # hits credited with a client-reported time
    var kvprefix_tokens_served: UInt64       # prefix tokens whose prefill was skipped
    var kvprefix_bytes_served: UInt64        # V.FETCH payload bytes
    var kvprefix_prefill_us_saved: UInt64    # measured + estimated
    var kvprefix_prefill_us_measured: UInt64 # measured only
    # Vector plane: FT.SEARCH + FT.HYBRID queries answered by this worker.
    var vector_queries: UInt64

    def __init__(out self):
        self.started_at_ns = _ledger_now_ns()
        self.kvprefix_hits = UInt64(0)
        self.kvprefix_misses = UInt64(0)
        self.kvprefix_hits_measured = UInt64(0)
        self.kvprefix_tokens_served = UInt64(0)
        self.kvprefix_bytes_served = UInt64(0)
        self.kvprefix_prefill_us_saved = UInt64(0)
        self.kvprefix_prefill_us_measured = UInt64(0)
        self.vector_queries = UInt64(0)

    def reset(mut self):
        """`PION.STATS RESET` — zero the counters; uptime is not a counter."""
        self.kvprefix_hits = UInt64(0)
        self.kvprefix_misses = UInt64(0)
        self.kvprefix_hits_measured = UInt64(0)
        self.kvprefix_tokens_served = UInt64(0)
        self.kvprefix_bytes_served = UInt64(0)
        self.kvprefix_prefill_us_saved = UInt64(0)
        self.kvprefix_prefill_us_measured = UInt64(0)
        self.vector_queries = UInt64(0)

    def uptime_seconds(self) -> Int:
        var d = _ledger_now_ns() - self.started_at_ns
        if d < 0: return 0
        return Int(d // Int64(1_000_000_000))

    @always_inline
    def record_kvprefix_hit(mut self, prefix_tokens: Int, reported_prefill_us: UInt64):
        self.kvprefix_hits += UInt64(1)
        if prefix_tokens > 0:
            self.kvprefix_tokens_served += UInt64(prefix_tokens)
        if reported_prefill_us > UInt64(0):
            self.kvprefix_hits_measured += UInt64(1)
            self.kvprefix_prefill_us_saved += reported_prefill_us
            self.kvprefix_prefill_us_measured += reported_prefill_us
        elif prefix_tokens > 0:
            self.kvprefix_prefill_us_saved += UInt64(prefix_tokens) * LEDGER_EST_PREFILL_US_PER_TOKEN

    @always_inline
    def record_kvprefix_miss(mut self):
        self.kvprefix_misses += UInt64(1)

    @always_inline
    def record_kvprefix_bytes(mut self, n: Int):
        if n > 0:
            self.kvprefix_bytes_served += UInt64(n)

    def prefill_seconds_avoided(self) -> String:
        return _us_to_seconds_text(self.kvprefix_prefill_us_saved)

    def prefill_seconds_avoided_measured(self) -> String:
        return _us_to_seconds_text(self.kvprefix_prefill_us_measured)
