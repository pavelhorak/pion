# Level 0 must be strongly connected: a search that enters a region must be
# able to leave it.
#
# _repair_l0_reachability (gh #405) links in every node the entry point cannot
# reach. It does not see the reverse: a set of nodes whose level-0 lists all
# point back into the set. Such a region is reachable and closed. A query whose
# upper-level descent lands in it stays there at any ef and returns the region's
# nodes. A parallel auto-build on x86 left one: 60 of 6,000 nodes, recall 0.908
# against 0.941, 13 of 400 queries under 0.5 even at EF_RUNTIME 1000
# (test_vector_auto_build failed once at 0.856).
#
# Builds a graph, closes a region by hand (every member's list rewritten to
# point only at other members), checks the closure is real, then checks that
# _repair_l0 leaves every node both reachable from the entry point and able to
# reach it, that every member of the region got a link out of it, and that the
# reachability pass alone did not.
#
# Second case: a closed region with an unreachable node beside it. The
# reachability pass links that node from a region member, which opens the
# region through one node; the exit pass must run before it.
#
#   cc -c tests/metal_stubs.c -o build/metal_stubs.o        # macOS only
#   pixi run mojo build -I . tests/test_hnsw_l0_exits.mojo -Xlinker build/metal_stubs.o \
#       -o build/t && build/t

from src.vector.hnsw import HNSWGraph
from std.memory import alloc, unsafe_memset
from std.random import random_float64, seed



def reach_from_entry(g: HNSWGraph) -> Int:
    var n = g.num_nodes
    var seen = alloc[UInt8](n)
    unsafe_memset(seen, 0, n)
    var q = List[Int]()
    q.append(g.entry_point_id)
    seen[g.entry_point_id] = 1
    var head = 0
    while head < len(q):
        var x = q[head]
        head += 1
        for k in range(g.nodes[x].get_neighbor_count(0)):
            var nb = g.nodes[x].get_neighbor(0, k)
            if nb >= 0 and nb < n and seen[nb] == 0:
                seen[nb] = 1
                q.append(nb)
    var c = len(q)
    seen.free()
    return c


def reach_entry(g: HNSWGraph) -> Int:
    """How many nodes have a level-0 path TO the entry point."""
    var n = g.num_nodes
    var back = alloc[UInt8](n)
    unsafe_memset(back, 0, n)
    back[g.entry_point_id] = 1
    var changed = True
    while changed:
        changed = False
        for i in range(n):
            if back[i] != 0:
                continue
            for k in range(g.nodes[i].get_neighbor_count(0)):
                var nb = g.nodes[i].get_neighbor(0, k)
                if nb >= 0 and nb < n and back[nb] != 0:
                    back[i] = 1
                    changed = True
                    break
    var c = 0
    for i in range(n):
        c += Int(back[i])
    back.free()
    return c


def close_region(mut g: HNSWGraph, center: Int, size: Int, cap: Int) -> List[Int]:
    """The `size` nodes nearest `center` (never the entry point), each list
    rewritten to the other members, nearest first, up to `cap` links."""
    var n = g.num_nodes
    var d = List[Float32]()
    for i in range(n):
        d.append(g._dist_int8_int8(g.nodes[center].vector, g.nodes[i].vector))
    var members = List[Int]()
    var taken = alloc[UInt8](n)
    unsafe_memset(taken, 0, n)
    taken[g.entry_point_id] = 1
    while len(members) < size:
        var best = -1
        for i in range(n):
            if taken[i] == 0 and (best < 0 or d[i] < d[best]):
                best = i
        taken[best] = 1
        members.append(best)
    taken.free()
    for x in members:
        var off = g.nodes[x]._get_offset(0)
        var cnt = 0
        var used = List[Int]()
        while cnt < cap and cnt < size - 1:
            var best = -1
            var bd = Float32(0.0)
            for y in members:
                if y == x or y in used:
                    continue
                var dy = g._dist_int8_int8(g.nodes[x].vector, g.nodes[y].vector)
                if best < 0 or dy < bd:
                    best = y
                    bd = dy
            used.append(best)
            g.nodes[x].neighbors[off + cnt] = UInt32(best)
            cnt += 1
        g.nodes[x].neighbor_counts[0] = UInt32(cnt)
    return members^


def links_out(g: HNSWGraph, members: List[Int]) -> Int:
    """How many members have at least one level-0 link out of the set."""
    var n = g.num_nodes
    var inside = alloc[UInt8](n)
    unsafe_memset(inside, 0, n)
    for x in members:
        inside[x] = 1
    var leave = 0
    for x in members:
        for k in range(g.nodes[x].get_neighbor_count(0)):
            if inside[g.nodes[x].get_neighbor(0, k)] == 0:
                leave += 1
                break
    inside.free()
    return leave


def built(n: Int, dim: Int) raises -> HNSWGraph:
    seed(405)
    var g = HNSWGraph(n, dim, M=8, ef_construction=64)
    var v = alloc[Float32](dim)
    for i in range(n):
        for j in range(dim):
            v[j] = Float32(random_float64() * 2.0 - 1.0)
        g.add_vector(i, v)
    v.free()
    g.build_index()
    return g^


def strand_beside(mut g: HNSWGraph, center: Int, members: List[Int]) -> Int:
    """Make the nearest non-member of `center` that links out of the region
    unreachable: every level-0 link to it is removed. Its own list stays, so
    it still reaches the entry point. Returns it."""
    var n = g.num_nodes
    var inside = alloc[UInt8](n)
    unsafe_memset(inside, 0, n)
    for x in members:
        inside[x] = 1
    var u = -1
    var ud = Float32(0.0)
    for i in range(n):
        if inside[i] != 0 or i == g.entry_point_id:
            continue
        var out = False
        for k in range(g.nodes[i].get_neighbor_count(0)):
            if inside[g.nodes[i].get_neighbor(0, k)] == 0:
                out = True
        if not out:
            continue
        var di = g._dist_int8_int8(g.nodes[center].vector, g.nodes[i].vector)
        if u < 0 or di < ud:
            u = i
            ud = di
    inside.free()
    for i in range(n):
        if i == u:
            continue
        var off = g.nodes[i]._get_offset(0)
        var c = g.nodes[i].get_neighbor_count(0)
        var w = 0
        for k in range(c):
            var nb = g.nodes[i].get_neighbor(0, k)
            if nb != u:
                g.nodes[i].neighbors[off + w] = UInt32(nb)
                w += 1
        g.nodes[i].neighbor_counts[0] = UInt32(w)
    return u


def main() raises:
    var n = 3000
    var dim = 32
    var g = built(n, dim)
    var fails = 0

    var r0 = reach_from_entry(g)
    var b0 = reach_entry(g)
    print("built:", n, "nodes; reachable from entry", r0, "; reach the entry", b0)
    if r0 != n or b0 != n:
        print("FAIL: a fresh build must be strongly connected at level 0")
        fails += 1

    var center = (g.entry_point_id + n // 2) % n
    var size = 60
    var members = close_region(g, center, size, 2 * g.M)
    var r1 = reach_from_entry(g)
    var b1 = reach_entry(g)
    print("closed a region of", size, ": reachable from entry", r1, "; reach the entry", b1)
    # The canary: the region must really be closed, or the repair is untested.
    if b1 != n - size:
        print("FAIL: the hand-closed region should be exactly the nodes that cannot reach the entry")
        fails += 1
    if r1 != n:
        print("FAIL: closing the region should leave it reachable (the shape the build left)")
        fails += 1

    # The gh #405 pass alone sees nothing to do: the gap this test is about.
    var links = g._repair_l0_reachability()
    if links != 0 or reach_entry(g) != n - size:
        print("FAIL: expected _repair_l0_reachability to leave the region closed; links", links)
        fails += 1

    g._repair_l0()
    var r2 = reach_from_entry(g)
    var b2 = reach_entry(g)
    print("after _repair_l0: reachable from entry", r2, "; reach the entry", b2)
    if r2 != n or b2 != n:
        print("FAIL: _repair_l0 must leave level 0 strongly connected")
        fails += 1

    # Strongly connected is not enough for a beam. 1.0.7 linked ONE member
    # out, and a search that landed in the region rarely reached it (builds
    # where it fired: recall 0.935 against 0.941). Every member is re-linked
    # now, so nearly all have a link that leaves the region (59 of 60 here; a
    # member at the region's center can rightly keep only members, since the
    # region is the ball around it). One link would leave 1 of 60.
    var leave = links_out(g, members)
    print("members with a link out of the region:", leave, "of", size)
    if leave < size * 3 // 4:
        print("FAIL: every member of a closed region must be re-linked, not just one")
        fails += 1
    _ = members^

    # The reachability pass can open a closed region by accident (seen on x86:
    # one build read 0.936 with only the reachability line logged). A node u
    # beside the region is unreachable; its closest reachable node r is a
    # member, so the pass links r -> u, and u's own links lead out. The region
    # now has one way out, through u, and an exit pass run AFTER it finds
    # nothing closed. Members keep free slots here so that r is a member.
    for variant in range(2):
        var h = built(n, dim)
        var c2 = (h.entry_point_id + n // 3) % n
        var m2 = close_region(h, c2, size, h.M)
        var u = strand_beside(h, c2, m2)
        var rf = reach_from_entry(h)
        var re = reach_entry(h)
        if variant == 0:
            # The canary: the reachability pass alone must open the region
            # through u, or this case is not exercised.
            var l = h._repair_l0_reachability()
            var lo = links_out(h, m2)
            print("region beside a stranded node: reachable", rf, "; reach the entry", re,
                  "; reachability pass alone: links", l, ", reach the entry", reach_entry(h),
                  ", members with a link out", lo)
            if rf != n - 1 or re != n - size or l == 0 or reach_entry(h) != n or lo > 2:
                print("FAIL: expected the reachability pass to open the region through one node")
                fails += 1
        else:
            h._repair_l0()
            var lo = links_out(h, m2)
            print("after _repair_l0: reachable", reach_from_entry(h), "; reach the entry",
                  reach_entry(h), "; members with a link out", lo, "of", size)
            if reach_from_entry(h) != n or reach_entry(h) != n:
                print("FAIL: _repair_l0 must leave level 0 strongly connected")
                fails += 1
            if lo < size * 3 // 4:
                print("FAIL: the exit pass must run before the reachability pass can open a region")
                fails += 1
        _ = m2^
        _ = u
    if fails == 0:
        print("PASS: level 0 strongly connected after the exit repair")
    else:
        print("FAILED:", fails)
        raise Error("test_hnsw_l0_exits failed")
