"""Null-pointer sentinels.

Mojo 1.0.0b2 removed the no-arg `Pointer[T, O]()` constructor and marked
`Bool(ptr)` `@unavailable`. Pointers are now non-null by design, and the
stdlib models nullability with `Optional[Pointer[...]]`, which stores
the null address in its `None` niche.

Pion holds raw null sentinels in ~55 structs: lazily-allocated buffers, the
optional cross-worker `SharedHNSWView`, "feature not enabled on this worker"
fields, and so on. Most are read on the event-loop fast path, which
forbids per-request allocation and indirection. Rewriting them as `Optional`
is the right long-term migration, but it is a *semantic* change at 400+ sites
(every read becomes a `.value()` unwrap, and every `is_null` check has to pick
between "absent" and "present but dangling"), so it is deliberately not what
this module does.

Instead these three helpers preserve the exact pre-b2 representation: a
pointer whose address is 0. `null_ptr()` lowers to the same `inttoptr 0` the
old default constructor emitted, so there is no runtime cost, no layout
change, and no behavioural difference from the 0.26.3 build.

Note that `Pointer.unsafe_dangling()` is NOT a substitute: it returns an
aligned but *non-zero* address, so it cannot be tested for absence. Any field
that needs a "not yet initialized" marker must use `null_ptr()` here (or track
initialization in a separate flag).

Future work: migrate the cold-path sites to `Optional[Pointer[...]]` and
shrink this module to the fast-path holdouts.
"""


@always_inline
def null_ptr[T: AnyType, O: Origin]() -> Pointer[T, O]:
    """Returns a pointer whose address is 0, for use as an absence sentinel.

    Replaces the `Pointer[T, O]()` default constructor removed in
    Mojo 1.0.0b2. Reading or writing through the result is undefined
    behaviour; test it with `is_null()` / `is_not_null()` first.

    Parameters:
        T: The pointee type.
        O: The origin of the pointer.

    Returns:
        A null pointer of the requested type.
    """
    # Spelled through a `var` so overload resolution picks the runtime `Int`
    # constructor. The `IntLiteral` overload rejects address 0 at compile time
    # ("Pointer is non-nullable"), which is exactly the case we need here.
    var zero: Int = 0
    return Pointer[T, O](unsafe_from_address=zero)


@always_inline
def is_null[T: AnyType, O: Origin](p: Pointer[T, O]) -> Bool:
    """Returns True if `p` is the null sentinel produced by `null_ptr()`.

    Replaces `not ptr`, since `Bool(ptr)` is `@unavailable` in Mojo 1.0.0b2.

    Parameters:
        T: The pointee type.
        O: The origin of the pointer.

    Args:
        p: The pointer to test.

    Returns:
        True if the pointer's address is 0.
    """
    return Int(p) == 0


@always_inline
def is_not_null[T: AnyType, O: Origin](p: Pointer[T, O]) -> Bool:
    """Returns True if `p` points somewhere other than address 0.

    Replaces `if ptr:`, since `Bool(ptr)` is `@unavailable` in Mojo 1.0.0b2.
    This is an absence check only — a non-null pointer may still be dangling.

    Parameters:
        T: The pointee type.
        O: The origin of the pointer.

    Args:
        p: The pointer to test.

    Returns:
        True if the pointer's address is non-zero.
    """
    return Int(p) != 0
