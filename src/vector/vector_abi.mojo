"""D11 seam: which implementation of the closed vector routines a build uses.

  -D PION_HELD_VECTOR   C-ABI calls into libpion_vector, the closed static
                        library vendored at vendor/pion-vector/<platform>/.
                        `pixi run build` passes this and links the archive.
  (default)             the open reference in src/vector/reference/ — same
                        algorithms with the tuning removed, bit-identical
                        results, slower. `pixi run build-open`, and any build
                        that does not link the library (iOS, standalone
                        `mojo build` of a test), gets this.

The closed routines are entered once per query (beam) or once per matrix op
(PKM); every buffer they touch is engine-owned and described by a view struct
(beam_view.mojo, quant_beam_view.mojo) whose field order is ABI.
"""
from std.ffi import external_call
from std.memory.unsafe_pointer import UnsafePointer
from std.sys import is_defined

from .beam_view import BeamView1536
from .quant_beam_view import QuantBeamView1536
from .reference.beam_1536 import beam_search_1536_ref
from .reference.quant_beams_1536 import quant_beam_search_1536_ref

comptime HELD_VECTOR = is_defined["PION_HELD_VECTOR"]()

# Bumped whenever a view struct or an exported signature changes. The engine
# refuses to start against a library that reports a different value, so a stale
# vendored archive fails loudly instead of reading fields at the wrong offsets.
comptime VECTOR_ABI_VERSION = 1


@always_inline
def beam_search_1536(v: UnsafePointer[BeamView1536, MutUntrackedOrigin]):
    comptime if HELD_VECTOR:
        _ = external_call["pion_v_beam_1536", Int](v)
    else:
        beam_search_1536_ref(v)


@always_inline
def quant_beam_search_1536(v: UnsafePointer[QuantBeamView1536, MutUntrackedOrigin]):
    """KIND in the view: 0 = PolarQuant INT4, 1 = NanoQuant INT2, 2 = TurboQuant INT3 + QJL."""
    comptime if HELD_VECTOR:
        _ = external_call["pion_v_quant_beam_1536", Int](v)
    else:
        quant_beam_search_1536_ref(v)


def vector_backend_line() -> String:
    """What `INFO` and `--version` print. Asks the library itself, so the line
    describes the code that is actually linked, not the build flag."""
    comptime if HELD_VECTOR:
        var abi = external_call["pion_v_abi_version", Int]()
        return String("libpion_vector abi=") + String(abi) + " (closed, dims=1536)"
    else:
        return String("reference (open)")


def vector_lib_abi() -> Int:
    comptime if HELD_VECTOR:
        return external_call["pion_v_abi_version", Int]()
    else:
        return VECTOR_ABI_VERSION


def vector_abi_ok() -> Bool:
    return vector_lib_abi() == VECTOR_ABI_VERSION
