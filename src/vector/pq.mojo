from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from src.vector.kernels import l2_distance_int8
from std.collections import List

@fieldwise_init
struct ProductQuantizer(Movable):
    var dim: Int
    var m: Int # Number of sub-spaces
    var k: Int # Number of centroids per sub-space (usually 256)
    var sub_dim: Int
    var codebooks: Pointer[Float32, MutUntrackedOrigin] # centroids [m][k][sub_dim]

    def __init__(out self, dim: Int, m: Int, k: Int = 256):
        self.dim = dim
        self.m = m
        self.k = k
        self.sub_dim = dim // m
        self.codebooks = alloc[Float32](m * k * self.sub_dim)
        # Initialize with zeros for prototype (in real system, use K-Means centroids)
        for i in range(m * k * self.sub_dim):
            self.codebooks[i] = 0.0

    def __moveinit__(out self, deinit take: Self):
        self.dim = take.dim
        self.m = take.m
        self.k = take.k
        self.sub_dim = take.sub_dim
        self.codebooks = take.codebooks

    def compute_codes(self, vector: Pointer[Float32, MutUntrackedOrigin], codes: Pointer[UInt8, MutUntrackedOrigin]):
        """Encode a full vector into m bytes."""
        for i in range(self.m):
            var best_idx: Int = 0
            var min_dist: Float32 = 1e30
            
            # Find nearest centroid in sub-space i
            for j in range(self.k):
                var dist: Float32 = 0
                var sub_vec = vector + (i * self.sub_dim)
                var centroid = self.codebooks + (i * self.k * self.sub_dim + j * self.sub_dim)
                
                for d in range(self.sub_dim):
                    var diff = sub_vec[d] - centroid[d]
                    dist += diff * diff
                
                if dist < min_dist:
                    min_dist = dist
                    best_idx = j
            
            codes[i] = UInt8(best_idx)

    def compute_distance(self, codes: Pointer[UInt8, MutUntrackedOrigin], query: Pointer[Float32, MutUntrackedOrigin]) -> Float32:
        """Asymmetric Distance Computation (ADC)."""
        # Precompute look-up table for query
        var lut = alloc[Float32](self.m * self.k)
        
        for i in range(self.m):
            var sub_query = query + (i * self.sub_dim)
            for j in range(self.k):
                var dist: Float32 = 0
                var centroid = self.codebooks + (i * self.k * self.sub_dim + j * self.sub_dim)
                for d in range(self.sub_dim):
                    var diff = sub_query[d] - centroid[d]
                    dist += diff * diff
                lut[i * self.k + j] = dist
        
        var total_dist: Float32 = 0
        for i in range(self.m):
            total_dist += lut[i * self.k + Int(codes[i])]
            
        lut.unsafe_free()
        return total_dist

    def deinit(owned self):
        self.codebooks.unsafe_free()
