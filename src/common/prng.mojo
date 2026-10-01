"""Xoshiro256++ — fast non-cryptographic PRNG for skip list levels, random pop, etc.

Replaces std.random.random_ui64() which may hit a syscall on Linux.
One instance per worker — no synchronization needed (shared-nothing model).
"""


struct Xoshiro256PlusPlus(Movable, Copyable):
    """Xoshiro256++ PRNG. 256-bit state, 64-bit output. Period: 2^256 - 1."""
    var s0: UInt64
    var s1: UInt64
    var s2: UInt64
    var s3: UInt64

    def __init__(out self, seed: UInt64):
        # SplitMix64 to expand single seed into 4 state words
        var z = seed
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        self.s0 = z
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        self.s1 = z
        z = (z ^ (z >> 31)) * 0xBF58476D1CE4E5B9
        self.s2 = z
        z = (z ^ (z >> 30)) * 0x94D049BB133111EB
        self.s3 = z
        if self.s0 == 0 and self.s1 == 0 and self.s2 == 0 and self.s3 == 0:
            self.s0 = 1  # must not be all-zero

    @always_inline
    def next(mut self) -> UInt64:
        """Generate next 64-bit random number."""
        var result = self._rotl(self.s0 + self.s3, 23) + self.s0
        var t = self.s1 << 17
        self.s2 ^= self.s0
        self.s3 ^= self.s1
        self.s1 ^= self.s2
        self.s0 ^= self.s3
        self.s2 ^= t
        self.s3 = self._rotl(self.s3, 45)
        return result

    @always_inline
    def next_bounded(mut self, upper: UInt64) -> UInt64:
        """Return random value in [0, upper). Rejection sampling for uniformity."""
        if upper <= 1:
            return 0
        var threshold = (~upper + 1) % upper  # 2^64 mod upper
        while True:
            var r = self.next()
            if r >= threshold:
                return r % upper

    @staticmethod
    @always_inline
    def _rotl(x: UInt64, k: Int) -> UInt64:
        return (x << UInt64(k)) | (x >> UInt64(64 - k))
